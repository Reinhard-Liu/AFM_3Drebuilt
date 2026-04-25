"""
SUP-03:
Expanded real-AFM evaluation with identity-level retrieval.

Key design choices:
1. Real queries can come from multiple prepared roots.
2. All real queries participate in retrieval if they have a molecule label.
3. Only GT-compatible queries contribute to GT-referenced geometry metrics.
4. Retrieval is computed at identity level, not raw candidate-file level.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ATOM_TO_IDX, center_coords, parse_xyz
from src.data.real_afm_cases import RealAFMCaseDataset
from src.train_v19_object_joint import build_edge_labels, compute_pred_object_metrics, extract_predicted_objects
from src.utils.mol2d import infer_bonds_from_coords
from src.v19_object_joint_review import _plot_object_2d, _plot_object_3d
from src.v19_visualize_test15 import compute_v19_object_similarity_for_retrieval, load_model_bundle


MAIN_METRICS = [
    ("pred_object_score", "object_score"),
    ("pred_object_type_acc", "type_acc"),
    ("pred_object_macro_f1", "macro_f1"),
    ("pred_object_edge_f1", "edge_f1"),
    ("pred_object_edge_f1_robust", "robust_edge_f1"),
    ("pred_object_z_mae", "z_mae"),
]


@dataclass
class RealAFMRecord:
    case_id: str
    molecule_name: str
    molecule_label: str
    tip: str
    contrast_variant: str
    gt_structure_compatible: bool
    gt_rank: int
    reciprocal_rank: float
    top1_hit: bool
    top3_hit: bool
    top5_hit: bool
    top1_label: str
    top1_candidate_name: str
    top1_sim: float
    top3_labels: list[str]
    top3_candidate_names: list[str]
    top3_sims: list[float]
    gt_atom_count: int | None
    pred_atom_count: int
    pred_object_score: float | None
    pred_object_3d_score: float | None
    pred_object_count_mae: float | None
    pred_object_type_acc: float | None
    pred_object_macro_f1: float | None
    pred_object_hetero_f1: float | None
    pred_object_edge_f1: float | None
    pred_object_edge_f1_robust: float | None
    pred_object_match_coverage_robust: float | None
    pred_object_heavy_rmsd: float | None
    pred_object_z_mae: float | None
    mean_center_score: float
    figure_path: str
    compar_figure_path: str


def parse_geometry_in(path: Path) -> tuple[np.ndarray, list[str]]:
    coords = []
    elements: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] != "atom" or len(parts) < 5:
            continue
        if parts[4] == "Cu":
            continue
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elements.append(parts[4])
    if not coords:
        raise ValueError(f"no atom records found in geometry.in: {path}")
    return np.asarray(coords, dtype=np.float32), elements


def build_edafm_candidates(edafm_root: Path) -> list[dict]:
    pool = []
    for mol_dir in sorted([p for p in edafm_root.iterdir() if p.is_dir()]):
        xyz_path = mol_dir / "mol.xyz"
        if not xyz_path.exists():
            xyz_candidates = sorted(mol_dir.glob("*.xyz"))
            if not xyz_candidates:
                continue
            xyz_path = xyz_candidates[0]
        coords, elements = parse_xyz(str(xyz_path))
        coords = center_coords(coords) / 12.0
        atom_types = np.asarray([ATOM_TO_IDX.get(e, ATOM_TO_IDX["C"]) for e in elements], dtype=np.int64)
        pool.append(
            {
                "name": mol_dir.name,
                "label": mol_dir.name,
                "coords": coords.astype(np.float32),
                "types": atom_types,
                "mask": np.ones(len(atom_types), dtype=np.float32),
            }
        )
    return pool


def build_camphor_candidates(camphor_struct_root: Path) -> list[dict]:
    pool = []
    for geometry_path in sorted(camphor_struct_root.glob("A*_geometry.in")):
        coords, elements = parse_geometry_in(geometry_path)
        coords = center_coords(coords) / 12.0
        atom_types = np.asarray([ATOM_TO_IDX.get(e, ATOM_TO_IDX["C"]) for e in elements], dtype=np.int64)
        stem = geometry_path.stem.replace("_geometry", "")
        pool.append(
            {
                "name": f"camphor_{stem}",
                "label": "camphor",
                "coords": coords.astype(np.float32),
                "types": atom_types,
                "mask": np.ones(len(atom_types), dtype=np.float32),
            }
        )
    return pool


def build_candidate_pool(edafm_root: Path, camphor_struct_root: Path | None) -> list[dict]:
    pool = build_edafm_candidates(edafm_root)
    if camphor_struct_root is not None and camphor_struct_root.exists():
        pool.extend(build_camphor_candidates(camphor_struct_root))
    if not pool:
        raise ValueError("candidate pool is empty")
    return pool


def _collapse_ranked_identity(
    ranked: list[tuple[dict, float]],
) -> list[tuple[str, str, float, np.ndarray, np.ndarray, np.ndarray]]:
    best_by_label: dict[str, tuple[str, float, np.ndarray, np.ndarray, np.ndarray]] = {}
    for item, sim in ranked:
        label = str(item["label"])
        current = best_by_label.get(label)
        if current is None or sim > current[1]:
            best_by_label[label] = (
                str(item["name"]),
                float(sim),
                item["coords"],
                item["types"],
                item["mask"],
            )
    collapsed = [
        (label, candidate_name, sim, coords, types, mask)
        for label, (candidate_name, sim, coords, types, mask) in best_by_label.items()
    ]
    collapsed.sort(key=lambda x: x[2], reverse=True)
    return collapsed


def retrieve_ranked_identity(
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    candidate_pool: list[dict],
    gt_label: str,
) -> tuple[list[tuple[str, str, float, np.ndarray, np.ndarray, np.ndarray]], int]:
    ranked = []
    for item in candidate_pool:
        sim_dict = compute_v19_object_similarity_for_retrieval(
            pred_coords,
            pred_types,
            pred_mask,
            item["coords"],
            item["types"],
            item["mask"],
        )
        ranked.append((item, float(sim_dict["overall"])))
    collapsed = _collapse_ranked_identity(ranked)
    gt_rank = -1
    for idx, (label, _candidate_name, _sim, _coords, _types, _mask) in enumerate(collapsed, start=1):
        if label == gt_label:
            gt_rank = idx
            break
    return collapsed, gt_rank


def _text_metrics(metrics: dict[str, float]) -> str:
    lines = ["SUP-03 Metrics", "=" * 18]
    for field_name, short_name in MAIN_METRICS:
        lines.append(f"{short_name}\n{metrics[field_name]:.4f}")
    return "\n\n".join(lines)


def _text_retrieval(
    molecule_label: str,
    tip: str,
    gt_rank: int,
    top3: list[tuple[str, str, float, np.ndarray, np.ndarray, np.ndarray]],
) -> str:
    lines = ["Identity Top-3 Retrieval", "=" * 24]
    lines.append(f"GT Label: {molecule_label}")
    lines.append(f"Tip: {tip}")
    lines.append(f"GT Rank: {gt_rank}")
    lines.append("")
    for i, (label, candidate_name, sim, _coords, _types, mask) in enumerate(top3, start=1):
        lines.append(f"Top-{i}: {label} | {candidate_name} | sim={sim:.4f} | n={int(mask.sum())}")
    lines.append("")
    lines.append(f"GT in Top3: {'Yes' if any(label == molecule_label for label, *_ in top3) else 'No'}")
    return "\n".join(lines)


def _edge_adj_from_coords(coords: np.ndarray, atom_types: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = int((mask > 0.5).sum())
    adj = np.zeros((max(n, 1), max(n, 1)), dtype=np.int32)
    if n <= 1:
        return adj
    bonds = infer_bonds_from_coords(coords[:n].astype(np.float32) * 12.0, atom_types[:n].astype(np.int64), mask[:n].astype(np.float32))
    for i, j in bonds:
        adj[i, j] = 1
        adj[j, i] = 1
    return adj


def make_case_figure_with_gt(
    afm_stack: np.ndarray,
    gt_coords: np.ndarray,
    gt_types: np.ndarray,
    gt_mask: np.ndarray,
    gt_edge_adj: np.ndarray,
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    pred_edge_adj: np.ndarray,
    metrics: dict[str, float],
    molecule_label: str,
    tip: str,
    top3: list[tuple[str, str, float, np.ndarray, np.ndarray, np.ndarray]],
    gt_rank: int,
    contrast_variant: str,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 4, hspace=0.28, wspace=0.20)

    slice_indices = [0, afm_stack.shape[0] // 2, afm_stack.shape[0] - 1]
    slice_titles = ["AFM Low Slice", "AFM Mid Slice", "AFM High Slice"]
    for i, (si, stitle) in enumerate(zip(slice_indices, slice_titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(afm_stack[si], cmap="afmhot", vmin=0, vmax=1)
        ax.set_title(stitle)
        ax.axis("off")

    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        _text_retrieval(molecule_label, tip, gt_rank, top3),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f4f6f8", alpha=0.95),
    )

    ax = fig.add_subplot(gs[1, 0])
    _plot_object_2d(ax, gt_coords, gt_types, gt_mask, gt_edge_adj, "GT Object 2D")
    ax = fig.add_subplot(gs[1, 1])
    _plot_object_2d(ax, pred_coords, pred_types, pred_mask, pred_edge_adj, "Pred Object 2D")

    ax = fig.add_subplot(gs[1, 2], projection="3d")
    _plot_object_3d(ax, gt_coords, gt_types, gt_mask, gt_edge_adj, "GT 3D Structure")
    ax = fig.add_subplot(gs[1, 3], projection="3d")
    _plot_object_3d(ax, pred_coords, pred_types, pred_mask, pred_edge_adj, "Pred 3D Structure")

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    ax.text(
        0.02,
        0.95,
        _text_metrics(metrics),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#eef3f7", alpha=0.95),
    )
    ax.text(
        0.62,
        0.95,
        f"Atom Count\nGT: {int(gt_mask.sum())}\nPred: {int(pred_mask.sum())}\nContrast: {contrast_variant}",
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#fff4e5", alpha=0.95),
    )
    fig.suptitle(f"SUP-03 Real AFM | {molecule_label} | tip={tip} | contrast={contrast_variant}", fontsize=14, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_case_figure_no_gt(
    afm_stack: np.ndarray,
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    pred_edge_adj: np.ndarray,
    molecule_label: str,
    tip: str,
    top3: list[tuple[str, str, float, np.ndarray, np.ndarray, np.ndarray]],
    gt_rank: int,
    contrast_variant: str,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(18, 8.5))
    gs = fig.add_gridspec(2, 4, hspace=0.28, wspace=0.20)

    slice_indices = [0, afm_stack.shape[0] // 2, afm_stack.shape[0] - 1]
    slice_titles = ["AFM Low Slice", "AFM Mid Slice", "AFM High Slice"]
    for i, (si, stitle) in enumerate(zip(slice_indices, slice_titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(afm_stack[si], cmap="afmhot", vmin=0, vmax=1)
        ax.set_title(stitle)
        ax.axis("off")

    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        _text_retrieval(molecule_label, tip, gt_rank, top3),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f4f6f8", alpha=0.95),
    )

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "No GT-compatible\n3D structure",
        ha="center",
        va="center",
        fontsize=16,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f3f4f6", alpha=0.95),
    )
    ax = fig.add_subplot(gs[1, 1])
    _plot_object_2d(ax, pred_coords, pred_types, pred_mask, pred_edge_adj, "Pred Object 2D")
    ax = fig.add_subplot(gs[1, 2], projection="3d")
    _plot_object_3d(ax, pred_coords, pred_types, pred_mask, pred_edge_adj, "Pred 3D Structure")
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.text(
        0.05,
        0.95,
        f"Pred Atom Count\n{int(pred_mask.sum())}\n\nContrast\n{contrast_variant}",
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#fff4e5", alpha=0.95),
    )
    fig.suptitle(f"SUP-03 Real AFM | {molecule_label} | tip={tip} | no GT | contrast={contrast_variant}", fontsize=14, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_comparison_figure_with_gt(
    gt_coords: np.ndarray,
    gt_types: np.ndarray,
    gt_mask: np.ndarray,
    gt_edge_adj: np.ndarray,
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    pred_edge_adj: np.ndarray,
    top3: list[tuple[str, str, float, np.ndarray, np.ndarray, np.ndarray]],
    metrics: dict[str, float],
    case_title: str,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(22, 5))
    axes = [fig.add_subplot(1, 5, i + 1, projection="3d") for i in range(5)]
    _plot_object_3d(axes[0], gt_coords, gt_types, gt_mask, gt_edge_adj, "GT")
    _plot_object_3d(axes[1], pred_coords, pred_types, pred_mask, pred_edge_adj, "Predicted")
    for i, (_label, candidate_name, sim, coords, atom_types, mask) in enumerate(top3, start=2):
        edge_adj = _edge_adj_from_coords(coords, atom_types, mask)
        _plot_object_3d(axes[i], coords, atom_types, mask, edge_adj, f"Top-{i-1}\n{candidate_name}\nsim={sim:.3f}")
    info = (
        f"{case_title} | pred_object_score={metrics['pred_object_score']:.4f} | "
        f"pred_object_type_acc={metrics['pred_object_type_acc']:.4f} | "
        f"pred_object_macro_f1={metrics['pred_object_macro_f1']:.4f} | "
        f"pred_object_edge_f1={metrics['pred_object_edge_f1']:.4f} | "
        f"pred_object_edge_f1_robust={metrics['pred_object_edge_f1_robust']:.4f} | "
        f"pred_object_z_mae={metrics['pred_object_z_mae']:.4f}"
    )
    fig.text(0.5, 0.02, info, ha="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.9))
    fig.suptitle(f"SUP-03 Real AFM 3D Comparison | {case_title}", fontsize=13, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_comparison_figure_no_gt(
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    pred_edge_adj: np.ndarray,
    top3: list[tuple[str, str, float, np.ndarray, np.ndarray, np.ndarray]],
    case_title: str,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(18, 5))
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]
    _plot_object_3d(axes[0], pred_coords, pred_types, pred_mask, pred_edge_adj, "Predicted")
    for i, (_label, candidate_name, sim, coords, atom_types, mask) in enumerate(top3[:3], start=1):
        edge_adj = _edge_adj_from_coords(coords, atom_types, mask)
        _plot_object_3d(axes[i], coords, atom_types, mask, edge_adj, f"Top-{i}\n{candidate_name}\nsim={sim:.3f}")
    fig.suptitle(f"SUP-03 Real AFM Retrieval Comparison | {case_title}", fontsize=13, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _mean_valid(records: list[RealAFMRecord], field: str) -> float | None:
    vals = [getattr(r, field) for r in records if getattr(r, field) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def summarize_retrieval(records: list[RealAFMRecord]) -> dict:
    if not records:
        return {"num_cases": 0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0, "mean_rank": 0.0}
    ranks = [r.gt_rank for r in records]
    return {
        "num_cases": len(records),
        "top1": float(np.mean([r.top1_hit for r in records])),
        "top3": float(np.mean([r.top3_hit for r in records])),
        "top5": float(np.mean([r.top5_hit for r in records])),
        "mrr": float(np.mean([r.reciprocal_rank for r in records])),
        "mean_rank": float(np.mean(ranks)),
    }


def summarize_gt_metrics(records: list[RealAFMRecord]) -> dict:
    return {
        "num_cases": len(records),
        "mean_pred_object_score": _mean_valid(records, "pred_object_score"),
        "mean_pred_object_3d_score": _mean_valid(records, "pred_object_3d_score"),
        "mean_pred_object_count_mae": _mean_valid(records, "pred_object_count_mae"),
        "mean_pred_object_type_acc": _mean_valid(records, "pred_object_type_acc"),
        "mean_pred_object_macro_f1": _mean_valid(records, "pred_object_macro_f1"),
        "mean_pred_object_hetero_f1": _mean_valid(records, "pred_object_hetero_f1"),
        "mean_pred_object_edge_f1": _mean_valid(records, "pred_object_edge_f1"),
        "mean_pred_object_edge_f1_robust": _mean_valid(records, "pred_object_edge_f1_robust"),
        "mean_pred_object_match_coverage_robust": _mean_valid(records, "pred_object_match_coverage_robust"),
        "mean_pred_object_heavy_rmsd": _mean_valid(records, "pred_object_heavy_rmsd"),
        "mean_pred_object_z_mae": _mean_valid(records, "pred_object_z_mae"),
        "mean_center_score": _mean_valid(records, "mean_center_score"),
    }


def evaluate_variant(
    model,
    type_head,
    edge_head,
    config: dict,
    checkpoint_path: Path,
    device: torch.device,
    real_afm_roots: list[Path],
    candidate_pool: list[dict],
    contrast_variant: str,
    output_dir: Path,
) -> dict:
    datasets = [RealAFMCaseDataset(root, contrast_variant=contrast_variant) for root in real_afm_roots]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img_size = int(config["img_size"])

    fig_dir = output_dir / "figures" / contrast_variant
    compar_dir = output_dir / "comparisons" / contrast_variant
    records: list[RealAFMRecord] = []

    with torch.no_grad():
        for batch in loader:
            afm = batch["afm_stack"].to(device)
            pred, features = model.forward_with_features(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            center_map_01 = torch.sigmoid(features["center_logits"])
            count_logits = features["count_logits"]

            pred_obj = extract_predicted_objects(
                center_map_01,
                pred_01,
                features["enc1"],
                afm,
                type_head,
                edge_head,
                device,
                img_size=img_size,
                count_logits=count_logits,
            )

            molecule_name = str(batch["molecule_name"][0])
            molecule_label = str(batch.get("molecule_label", batch["molecule_name"])[0])
            tip = str(batch["tip"][0])
            case_id = str(batch["case_id"][0])
            gt_compatible = bool(batch.get("gt_structure_compatible", torch.tensor([False]))[0].item())

            ranked, gt_rank = retrieve_ranked_identity(
                pred_obj["coords"],
                pred_obj["types"],
                pred_obj["mask"],
                candidate_pool,
                molecule_label,
            )
            top3 = ranked[:3]
            pred_valid_n = int((pred_obj["mask"] > 0.5).sum())

            metrics = None
            gt_atom_count = None
            fig_path = fig_dir / f"{case_id}.png"
            compar_path = compar_dir / f"{case_id}.png"

            if gt_compatible:
                gt_edge_adj = build_edge_labels(batch, device)[0].detach().cpu().numpy().astype(np.int32)
                gt_coords = batch["coords"][0].detach().cpu().numpy()
                gt_types = batch["atom_types"][0].detach().cpu().numpy()
                gt_mask = batch["atom_mask"][0].detach().cpu().numpy()
                gt_atom_count = int((gt_mask > 0.5).sum())
                metrics = compute_pred_object_metrics(pred_obj, gt_coords, gt_types, gt_mask, gt_edge_adj, img_size=img_size)
                make_case_figure_with_gt(
                    afm_stack=batch["afm_stack"][0].detach().cpu().numpy(),
                    gt_coords=gt_coords,
                    gt_types=gt_types,
                    gt_mask=gt_mask,
                    gt_edge_adj=gt_edge_adj,
                    pred_coords=pred_obj["coords"],
                    pred_types=pred_obj["types"],
                    pred_mask=pred_obj["mask"],
                    pred_edge_adj=pred_obj["edge_adj"],
                    metrics=metrics,
                    molecule_label=molecule_label,
                    tip=tip,
                    top3=top3,
                    gt_rank=gt_rank,
                    contrast_variant=contrast_variant,
                    save_path=fig_path,
                )
                make_comparison_figure_with_gt(
                    gt_coords=gt_coords,
                    gt_types=gt_types,
                    gt_mask=gt_mask,
                    gt_edge_adj=gt_edge_adj,
                    pred_coords=pred_obj["coords"],
                    pred_types=pred_obj["types"],
                    pred_mask=pred_obj["mask"],
                    pred_edge_adj=pred_obj["edge_adj"],
                    top3=top3,
                    metrics=metrics,
                    case_title=f"{molecule_label} | {tip} | {contrast_variant}",
                    save_path=compar_path,
                )
            else:
                make_case_figure_no_gt(
                    afm_stack=batch["afm_stack"][0].detach().cpu().numpy(),
                    pred_coords=pred_obj["coords"],
                    pred_types=pred_obj["types"],
                    pred_mask=pred_obj["mask"],
                    pred_edge_adj=pred_obj["edge_adj"],
                    molecule_label=molecule_label,
                    tip=tip,
                    top3=top3,
                    gt_rank=gt_rank,
                    contrast_variant=contrast_variant,
                    save_path=fig_path,
                )
                make_comparison_figure_no_gt(
                    pred_coords=pred_obj["coords"],
                    pred_types=pred_obj["types"],
                    pred_mask=pred_obj["mask"],
                    pred_edge_adj=pred_obj["edge_adj"],
                    top3=top3,
                    case_title=f"{molecule_label} | {tip} | {contrast_variant}",
                    save_path=compar_path,
                )

            records.append(
                RealAFMRecord(
                    case_id=case_id,
                    molecule_name=molecule_name,
                    molecule_label=molecule_label,
                    tip=tip,
                    contrast_variant=contrast_variant,
                    gt_structure_compatible=gt_compatible,
                    gt_rank=int(gt_rank),
                    reciprocal_rank=(1.0 / gt_rank) if gt_rank > 0 else 0.0,
                    top1_hit=gt_rank == 1,
                    top3_hit=0 < gt_rank <= 3,
                    top5_hit=0 < gt_rank <= 5,
                    top1_label=str(top3[0][0]),
                    top1_candidate_name=str(top3[0][1]),
                    top1_sim=float(top3[0][2]),
                    top3_labels=[str(x[0]) for x in top3],
                    top3_candidate_names=[str(x[1]) for x in top3],
                    top3_sims=[float(x[2]) for x in top3],
                    gt_atom_count=gt_atom_count,
                    pred_atom_count=pred_valid_n,
                    pred_object_score=float(metrics["pred_object_score"]) if metrics is not None else None,
                    pred_object_3d_score=float(metrics["pred_object_3d_score"]) if metrics is not None else None,
                    pred_object_count_mae=float(metrics["pred_object_count_mae"]) if metrics is not None else None,
                    pred_object_type_acc=float(metrics["pred_object_type_acc"]) if metrics is not None else None,
                    pred_object_macro_f1=float(metrics["pred_object_macro_f1"]) if metrics is not None else None,
                    pred_object_hetero_f1=float(metrics["pred_object_hetero_f1"]) if metrics is not None else None,
                    pred_object_edge_f1=float(metrics["pred_object_edge_f1"]) if metrics is not None else None,
                    pred_object_edge_f1_robust=float(metrics["pred_object_edge_f1_robust"]) if metrics is not None else None,
                    pred_object_match_coverage_robust=float(metrics["pred_object_match_coverage_robust"]) if metrics is not None else None,
                    pred_object_heavy_rmsd=float(metrics["pred_object_heavy_rmsd"]) if metrics is not None else None,
                    pred_object_z_mae=float(metrics["pred_object_z_mae"]) if metrics is not None else None,
                    mean_center_score=float(pred_obj.get("mean_center_score", 0.0)),
                    figure_path=str(fig_path),
                    compar_figure_path=str(compar_path),
                )
            )

    gt_records = [r for r in records if r.gt_structure_compatible]
    return {
        "checkpoint": str(checkpoint_path),
        "contrast_variant": contrast_variant,
        "candidate_count": len(candidate_pool),
        "candidate_labels": sorted({str(x["label"]) for x in candidate_pool}),
        "candidate_names": [str(x["name"]) for x in candidate_pool],
        "retrieval_all_cases": summarize_retrieval(records),
        "retrieval_gt_compatible_subset": summarize_retrieval(gt_records),
        "gt_metric_subset": summarize_gt_metrics(gt_records),
        "records": [asdict(r) for r in records],
    }


def write_markdown(summary: dict, output_path: Path) -> None:
    md = []
    md.append("# SUP-03 扩展真实 AFM 实验报告")
    md.append("")
    md.append("## 一、实验设置")
    md.append(f"- checkpoint：`{summary['checkpoint']}`")
    md.append(f"- real AFM roots：`{', '.join(summary['real_afm_roots'])}`")
    md.append(f"- EDAFM source root：`{summary['edafm_root']}`")
    if summary.get("camphor_structure_root"):
        md.append(f"- camphor structure root：`{summary['camphor_structure_root']}`")
    md.append(f"- 检索按 `分子身份` 统计，不按单个候选构型文件统计。")
    md.append("")
    for name in ["normal", "inverted"]:
        block = summary[name]
        ret_all = block["retrieval_all_cases"]
        ret_gt = block["retrieval_gt_compatible_subset"]
        gt = block["gt_metric_subset"]
        md.append(f"## 二、{name} 结果")
        md.append("")
        md.append(f"- 全部真实 case retrieval：`n={ret_all['num_cases']}`，`Top1={100.0 * ret_all['top1']:.2f}%`，`Top3={100.0 * ret_all['top3']:.2f}%`，`Top5={100.0 * ret_all['top5']:.2f}%`，`MRR={ret_all['mrr']:.4f}`，`mean_rank={ret_all['mean_rank']:.2f}`")
        md.append(f"- GT 兼容子集 retrieval：`n={ret_gt['num_cases']}`，`Top1={100.0 * ret_gt['top1']:.2f}%`，`Top3={100.0 * ret_gt['top3']:.2f}%`，`Top5={100.0 * ret_gt['top5']:.2f}%`，`MRR={ret_gt['mrr']:.4f}`，`mean_rank={ret_gt['mean_rank']:.2f}`")
        md.append(f"- GT 兼容子集几何指标：`n={gt['num_cases']}`，`pred_object_score={0.0 if gt['mean_pred_object_score'] is None else gt['mean_pred_object_score']:.4f}`，`type_acc={0.0 if gt['mean_pred_object_type_acc'] is None else gt['mean_pred_object_type_acc']:.4f}`，`macro_f1={0.0 if gt['mean_pred_object_macro_f1'] is None else gt['mean_pred_object_macro_f1']:.4f}`，`edge_f1={0.0 if gt['mean_pred_object_edge_f1'] is None else gt['mean_pred_object_edge_f1']:.4f}`，`robust_edge_f1={0.0 if gt['mean_pred_object_edge_f1_robust'] is None else gt['mean_pred_object_edge_f1_robust']:.4f}`，`z_mae={0.0 if gt['mean_pred_object_z_mae'] is None else gt['mean_pred_object_z_mae']:.4f}`")
        md.append("")
        md.append("| case_id | label | tip | GT-compatible | GT rank | Top1 | Top3 | Top5 | Pred atoms | GT atoms | top3 labels |")
        md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in block["records"]:
            gt_atoms = "" if r["gt_atom_count"] is None else str(r["gt_atom_count"])
            md.append(
                f"| {r['case_id']} | {r['molecule_label']} | {r['tip']} | "
                f"{'Y' if r['gt_structure_compatible'] else 'N'} | {r['gt_rank']} | "
                f"{'Y' if r['top1_hit'] else 'N'} | {'Y' if r['top3_hit'] else 'N'} | {'Y' if r['top5_hit'] else 'N'} | "
                f"{r['pred_atom_count']} | {gt_atoms} | {', '.join(r['top3_labels'])} |"
            )
        md.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/micro/experiments/v20_object_joint_medium10/checkpoints/latest_v19_object_joint.pt")
    parser.add_argument(
        "--real_afm_roots",
        default="/root/autodl-tmp/micro/real_afm/edafm_sup03_cases,/root/autodl-tmp/micro/real_afm/camphor_sup03_cases",
    )
    parser.add_argument("--edafm_root", default="/root/autodl-tmp/real_afm_datasets/edafm_zenodo_10609676/edafm-data/edafm-data")
    parser.add_argument("--camphor_structure_root", default="/root/autodl-tmp/real_afm_datasets/camphor_adsorbate_4710346/structures")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    real_afm_roots = [Path(x.strip()) for x in str(args.real_afm_roots).split(",") if x.strip()]
    edafm_root = Path(args.edafm_root)
    camphor_structure_root = Path(args.camphor_structure_root) if args.camphor_structure_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, type_head, edge_head, config = load_model_bundle(checkpoint_path, device)
    candidate_pool = build_candidate_pool(edafm_root, camphor_structure_root)

    normal = evaluate_variant(
        model=model,
        type_head=type_head,
        edge_head=edge_head,
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        real_afm_roots=real_afm_roots,
        candidate_pool=candidate_pool,
        contrast_variant="normal",
        output_dir=output_dir,
    )
    inverted = evaluate_variant(
        model=model,
        type_head=type_head,
        edge_head=edge_head,
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        real_afm_roots=real_afm_roots,
        candidate_pool=candidate_pool,
        contrast_variant="inverted",
        output_dir=output_dir,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "real_afm_roots": [str(x) for x in real_afm_roots],
        "edafm_root": str(edafm_root),
        "camphor_structure_root": str(camphor_structure_root) if camphor_structure_root else "",
        "candidate_count": len(candidate_pool),
        "candidate_labels": sorted({str(x["label"]) for x in candidate_pool}),
        "candidate_names": [str(x["name"]) for x in candidate_pool],
        "normal": normal,
        "inverted": inverted,
    }

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "sup03_real_afm_summary.json"
    md_path = reports_dir / "sup03_real_afm_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, md_path)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "report_json": str(json_path),
                "report_md": str(md_path),
                "normal_all_top1": normal["retrieval_all_cases"]["top1"],
                "normal_all_top3": normal["retrieval_all_cases"]["top3"],
                "inverted_all_top1": inverted["retrieval_all_cases"]["top1"],
                "inverted_all_top3": inverted["retrieval_all_cases"]["top3"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
