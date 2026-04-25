"""
Generate 15-sample V19 object-level visualization outputs on the test split.

Outputs:
1. Main visualization figures with:
   - AFM low/mid/high slices
   - GT / Pred object 2D
   - GT / Pred object 3D
   - 6 Chinese metrics (field name + meaning)
   - Top-3 CID retrieval results on the test set
2. 5-molecule comparison figures:
   GT | Predicted | Top-1 | Top-2 | Top-3
3. JSON / Markdown summary for the 15 samples
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.data.dataset import (
    ATOM_TO_IDX,
    ATOM_TYPES,
    MAX_ATOMS,
    center_coords,
    create_dataloaders,
    parse_xyz,
)
from src.models.v19_center_edge_head import CenterConditionedEdgeHead
from src.models.v19_center_type_head import CenterConditionedTypeHead
from src.models.v19_joint_model import V19JointUNet
from src.train_v19_object_joint import (
    compute_pred_object_metrics,
    extract_predicted_objects as extract_predicted_objects_v20,
)
from src.utils.mol2d import infer_bonds_from_coords, project_xy_to_pixels
from src.utils.metrics import _hungarian_match_numpy, _macro_type_f1, _safe_f1
from src.utils.visualize import ATOM_COLORS, ATOM_SIZES
from src.v19_object_joint_review import (
    _adjacency_from_logits,
    _per_sample_dense_metrics,
    _plot_object_2d,
    _plot_object_3d,
)


MAIN_METRICS = [
    ("pred_object_score", "纯预测对象2D闭环总分"),
    ("pred_object_type_acc", "纯预测对象原子类型准确率"),
    ("pred_object_macro_f1", "纯预测对象原子类型宏平均F1"),
    ("pred_object_hetero_f1", "纯预测对象杂原子F1"),
    ("pred_object_edge_f1", "纯预测对象对象级边F1"),
    ("pred_object_z_mae", "纯预测对象z平均绝对误差"),
]


def resolve_sample_indices(
    total: int,
    num_samples: int,
    sample_indices: str = "",
    sample_indices_json: str = "",
) -> list[int]:
    if sample_indices.strip():
        indices = [int(x.strip()) for x in sample_indices.split(",") if x.strip()]
    elif sample_indices_json.strip():
        obj = json.loads(Path(sample_indices_json).read_text())
        if isinstance(obj, dict) and "records" in obj:
            indices = [int(r["dataset_index"]) for r in obj["records"]]
        elif isinstance(obj, dict) and "indices" in obj:
            indices = [int(x) for x in obj["indices"]]
        elif isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                indices = [int(r["dataset_index"]) for r in obj]
            else:
                indices = [int(x) for x in obj]
        else:
            raise ValueError(f"unsupported sample_indices_json format: {sample_indices_json}")
    else:
        indices = np.linspace(0, total - 1, min(num_samples, total), dtype=int).tolist()

    filtered = []
    seen = set()
    for idx in indices:
        idx = int(idx)
        if idx < 0 or idx >= total or idx in seen:
            continue
        filtered.append(idx)
        seen.add(idx)
    if not filtered:
        raise ValueError("no valid sample indices resolved")
    return filtered[: min(num_samples, len(filtered))]


@dataclass
class RetrievalHit:
    cid: str
    sim: float
    n_atoms: int


def _count_similarity(n_pred: int, n_db: int) -> float:
    if n_pred == 0 and n_db == 0:
        return 1.0
    if n_pred == 0 or n_db == 0:
        return 0.0
    return float(max(0.0, 1.0 - abs(n_pred - n_db) / max(n_pred, n_db)))


def _infer_edge_set(coords: np.ndarray, atom_types: np.ndarray, mask: np.ndarray) -> set[tuple[int, int]]:
    bonds = infer_bonds_from_coords(coords.astype(np.float32) * 12.0, atom_types.astype(np.int64), mask.astype(np.float32))
    return {tuple(sorted((int(i), int(j)))) for i, j in bonds}


def _matched_edge_f1(
    pred_edges: set[tuple[int, int]],
    db_edges: set[tuple[int, int]],
    pred_to_db: dict[int, int],
) -> tuple[float, float, float]:
    mapped_pred = set()
    for i, j in pred_edges:
        if i not in pred_to_db or j not in pred_to_db:
            continue
        mapped_pred.add(tuple(sorted((pred_to_db[i], pred_to_db[j]))))

    tp = len(mapped_pred & db_edges)
    fp = len(mapped_pred - db_edges)
    fn = len(db_edges - mapped_pred)
    return _safe_f1(tp, fp, fn)


def compute_v19_object_similarity_for_retrieval(
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    db_coords: np.ndarray,
    db_types: np.ndarray,
    db_mask: np.ndarray,
) -> dict:
    n_pred = int((pred_mask > 0).sum())
    n_db = int((db_mask > 0).sum())
    if n_pred == 0 or n_db == 0:
        return {
            "type_acc": 0.0,
            "macro_f1": 0.0,
            "hetero_f1": 0.0,
            "edge_f1": 0.0,
            "coord_score": 0.0,
            "count_score": _count_similarity(n_pred, n_db),
            "heavy_rmsd": 9.999,
            "overall": 0.0,
            "n_pred": n_pred,
            "n_db": n_db,
        }

    pc = pred_coords[:n_pred].astype(np.float32)
    pt = pred_types[:n_pred].astype(np.int64)
    gc = db_coords[:n_db].astype(np.float32)
    gt = db_types[:n_db].astype(np.int64)

    row_ind, col_ind, cost = _hungarian_match_numpy(pc, gc)
    if len(row_ind) == 0:
        return {
            "type_acc": 0.0,
            "macro_f1": 0.0,
            "hetero_f1": 0.0,
            "edge_f1": 0.0,
            "coord_score": 0.0,
            "count_score": _count_similarity(n_pred, n_db),
            "heavy_rmsd": 9.999,
            "overall": 0.0,
            "n_pred": n_pred,
            "n_db": n_db,
        }

    matched_dists = cost[row_ind, col_ind]
    pred_match_types = pt[row_ind]
    db_match_types = gt[col_ind]
    type_acc = float((pred_match_types == db_match_types).mean())
    macro_f1 = _macro_type_f1(pred_match_types, db_match_types, pt, gt)

    pred_het = ~np.isin(pt, [0, 1])
    db_het = ~np.isin(gt, [0, 1])
    tp_het = int((((pred_match_types != 0) & (pred_match_types != 1)) & ((db_match_types != 0) & (db_match_types != 1))).sum())
    fp_het = int(pred_het.sum() - tp_het)
    fn_het = int(db_het.sum() - tp_het)
    _, _, hetero_f1 = _safe_f1(tp_het, fp_het, fn_het)

    heavy_match = (db_match_types != 0)
    if heavy_match.any():
        heavy_rmsd = float(np.sqrt(np.mean(matched_dists[heavy_match] ** 2)))
    else:
        heavy_rmsd = float(np.sqrt(np.mean(matched_dists ** 2)))
    coord_score = float(np.clip(1.0 - heavy_rmsd / 0.35, 0.0, 1.0))

    pred_edges = _infer_edge_set(pc, pt, np.ones(n_pred, dtype=np.float32))
    db_edges = _infer_edge_set(gc, gt, np.ones(n_db, dtype=np.float32))
    pred_to_db = {int(p): int(d) for p, d in zip(row_ind.tolist(), col_ind.tolist())}
    _, _, edge_f1 = _matched_edge_f1(pred_edges, db_edges, pred_to_db)

    count_score = _count_similarity(n_pred, n_db)

    overall = (
        0.30 * coord_score
        + 0.20 * type_acc
        + 0.15 * macro_f1
        + 0.15 * hetero_f1
        + 0.15 * edge_f1
        + 0.05 * count_score
    )
    overall = float(np.clip(overall, 0.0, 1.0))

    return {
        "type_acc": float(type_acc),
        "macro_f1": float(macro_f1),
        "hetero_f1": float(hetero_f1),
        "edge_f1": float(edge_f1),
        "coord_score": float(coord_score),
        "count_score": float(count_score),
        "heavy_rmsd": float(heavy_rmsd),
        "overall": overall,
        "n_pred": n_pred,
        "n_db": n_db,
    }


@dataclass
class SampleRecord:
    dataset_index: int
    gt_cid: str
    gt_rank: int
    top3_cids: list[str]
    top3_sims: list[float]
    gt_in_top3: bool
    pred_atom_count: int
    gt_atom_count: int
    pred_object_score: float
    pred_object_type_acc: float
    pred_object_macro_f1: float
    pred_object_hetero_f1: float
    pred_object_edge_f1: float
    pred_object_z_mae: float
    main_figure: str
    compar_figure: str


def load_model_bundle(checkpoint_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state["config"]
    model = V19JointUNet(in_channels=10, base_ch=config.get("base_ch", 64)).to(device)
    type_head = CenterConditionedTypeHead(shared_feat_dim=config.get("base_ch", 64)).to(device)
    edge_head = CenterConditionedEdgeHead(shared_feat_dim=config.get("base_ch", 64)).to(device)
    model.load_state_dict(state["model"], strict=False)
    type_head.load_state_dict(state["type_head"], strict=False)
    edge_head.load_state_dict(state["edge_head"], strict=False)
    model.eval()
    type_head.eval()
    edge_head.eval()
    return model, type_head, edge_head, config


def _build_retrieval_db(dataset) -> list[dict]:
    db = []
    for sample_meta in dataset.samples:
        cid = sample_meta["cid"]
        coords, elements = parse_xyz(sample_meta["xyz_path"])
        coords = center_coords(coords) / 12.0
        atom_types = np.array([ATOM_TO_IDX.get(e, ATOM_TO_IDX["C"]) for e in elements], dtype=np.int64)
        mask = np.ones(len(atom_types), dtype=np.float32)
        db.append(
            {
                "cid": cid,
                "coords": coords.astype(np.float32),
                "types": atom_types,
                "mask": mask,
            }
        )
    return db


def retrieve_top3(
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    retrieval_db: list[dict],
    gt_cid: str,
) -> tuple[list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray]], int]:
    results = []
    gt_rank = None
    gt_sim = None
    for item in retrieval_db:
        sim_dict = compute_v19_object_similarity_for_retrieval(
            pred_coords,
            pred_types,
            pred_mask,
            item["coords"],
            item["types"],
            item["mask"],
        )
        results.append((item["cid"], sim_dict["overall"], item["coords"], item["types"], item["mask"]))
        if item["cid"] == gt_cid:
            gt_sim = sim_dict["overall"]
    results.sort(key=lambda x: x[1], reverse=True)
    if gt_sim is not None:
        gt_rank = sum(1 for _, sim, _, _, _ in results if sim > gt_sim) + 1
    return results[:3], gt_rank if gt_rank is not None else -1


def _text_metrics(metrics: dict[str, float]) -> str:
    lines = ["Six Core Metrics"]
    lines.append("=" * 18)
    for field_name, zh in MAIN_METRICS:
        value = metrics[field_name]
        lines.append(f"{field_name}\n{value:.4f}")
    return "\n\n".join(lines)


def _text_retrieval(
    gt_cid: str,
    top3: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray]],
    gt_rank: int,
    run_label: str,
) -> str:
    lines = [f"CID Top-3 Retrieval ({run_label} Object Sim)"]
    lines.append("=" * 18)
    lines.append(f"GT CID: {gt_cid}")
    lines.append(f"GT Rank: {gt_rank}")
    lines.append("")
    for i, (cid, sim, coords, atom_types, mask) in enumerate(top3, start=1):
        lines.append(f"Top-{i}: CID={cid} | sim={sim:.4f} | n={int(mask.sum())}")
    lines.append("")
    lines.append(f"GT in Top3: {'Yes' if any(cid == gt_cid for cid, *_ in top3) else 'No'}")
    return "\n".join(lines)


def make_main_figure(
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
    gt_cid: str,
    top3: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray]],
    gt_rank: int,
    run_label: str,
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
        _text_retrieval(gt_cid, top3, gt_rank, run_label),
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
        f"Atom Count\nGT: {int(gt_mask.sum())}\nPred: {int(pred_mask.sum())}",
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#fff4e5", alpha=0.95),
    )

    fig.suptitle(f"{run_label} Test Visualization | GT CID={gt_cid}", fontsize=14, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_comparison_figure(
    gt_coords: np.ndarray,
    gt_types: np.ndarray,
    gt_mask: np.ndarray,
    gt_edge_adj: np.ndarray,
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    pred_edge_adj: np.ndarray,
    top3: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray]],
    metrics: dict[str, float],
    dataset_index: int,
    run_label: str,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(22, 5))
    axes = [fig.add_subplot(1, 5, i + 1, projection="3d") for i in range(5)]
    _plot_object_3d(axes[0], gt_coords, gt_types, gt_mask, gt_edge_adj, "GT")
    _plot_object_3d(axes[1], pred_coords, pred_types, pred_mask, pred_edge_adj, "Predicted")
    for i, (cid, sim, coords, atom_types, mask) in enumerate(top3, start=2):
        edge_adj = np.zeros((len(mask), len(mask)), dtype=np.int32)
        bonds = infer_bonds_from_coords(coords.astype(np.float32) * 12.0, atom_types, mask)
        for a, b in bonds:
            edge_adj[a, b] = 1
            edge_adj[b, a] = 1
        _plot_object_3d(
            axes[i],
            coords,
            atom_types,
            mask,
            edge_adj,
            f"Top-{i-1}\nCID={cid}\nsim={sim:.3f}",
        )

    info = (
        f"Sample #{dataset_index} | "
        f"pred_object_score={metrics['pred_object_score']:.4f} | "
        f"pred_object_type_acc={metrics['pred_object_type_acc']:.4f} | "
        f"pred_object_macro_f1={metrics['pred_object_macro_f1']:.4f} | "
        f"pred_object_hetero_f1={metrics['pred_object_hetero_f1']:.4f} | "
        f"pred_object_edge_f1={metrics['pred_object_edge_f1']:.4f} | "
        f"pred_object_z_mae={metrics['pred_object_z_mae']:.4f}"
    )
    fig.text(
        0.5,
        0.02,
        info,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.9),
    )
    fig.suptitle(f"{run_label} 3D Molecule Comparison | Sample #{dataset_index}", fontsize=13, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--peak_threshold", type=float, default=0.45)
    parser.add_argument("--min_distance_px", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sample_indices", type=str, default="")
    parser.add_argument("--sample_indices_json", type=str, default="")
    parser.add_argument("--run_label", type=str, default="V19 Object Test15")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    output_root = Path(args.output_root)
    main_dir = output_root / "visualizations_object15"
    compar_dir = output_root / "visual_compar_object15"
    report_dir = output_root / "visual_reports_object15"
    report_dir.mkdir(parents=True, exist_ok=True)

    model, type_head, edge_head, config = load_model_bundle(checkpoint_path, device)
    _, _, test_loader, _ = create_dataloaders(
        data_root=config["data_root"],
        param_key=config.get("param_key", "K-1"),
        img_size=config["img_size"],
        min_corrugation=config.get("min_corrugation", 0.0),
        augment_rotation=False,
        require_ring=config.get("require_ring", False),
        batch_size=args.batch_size,
        num_workers=config.get("num_workers", 4),
        max_samples=config.get("max_samples", 0),
        val_size=config.get("val_size", 0),
    )
    test_dataset = test_loader.dataset
    retrieval_db = _build_retrieval_db(test_dataset)

    total = len(test_dataset)
    indices = resolve_sample_indices(
        total=total,
        num_samples=args.num_samples,
        sample_indices=args.sample_indices,
        sample_indices_json=args.sample_indices_json,
    )
    records: list[SampleRecord] = []

    for dataset_index in indices:
        sample = test_dataset[dataset_index]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}
        gt_cid = test_dataset.samples[dataset_index]["cid"]

        with torch.no_grad():
            pred, features = model.forward_with_features(batch["afm_stack"])
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            center_map_01 = torch.sigmoid(features["center_logits"]).clamp(0.0, 1.0)

        pred_obj = extract_predicted_objects_v20(
            center_map_01,
            pred_01,
            features["enc1"],
            batch["afm_stack"],
            type_head,
            edge_head,
            device,
            img_size=int(config["img_size"]),
            count_logits=features.get("count_logits", None),
            peak_threshold=args.peak_threshold,
            min_distance_px=args.min_distance_px,
        )

        gt_coords = batch["coords"][0].detach().cpu().numpy()
        gt_types = batch["atom_types"][0].detach().cpu().numpy()
        gt_mask = batch["atom_mask"][0].detach().cpu().numpy()
        gt_edge_adj = np.zeros((MAX_ATOMS, MAX_ATOMS), dtype=np.int32)
        gt_bonds = infer_bonds_from_coords(gt_coords.astype(np.float32) * 12.0, gt_types, gt_mask)
        for a, b in gt_bonds:
            gt_edge_adj[a, b] = 1
            gt_edge_adj[b, a] = 1

        pred_coords = pred_obj["coords"]
        pred_types = pred_obj["types"]
        pred_mask = pred_obj["mask"]
        pred_edge_adj = pred_obj["edge_adj"]
        metrics = compute_pred_object_metrics(pred_obj, gt_coords, gt_types, gt_mask, gt_edge_adj)

        top3, gt_rank = retrieve_top3(pred_coords, pred_types, pred_mask, retrieval_db, gt_cid)

        main_path = main_dir / f"sample_{dataset_index:05d}.png"
        compar_path = compar_dir / f"sample_{dataset_index:05d}_5mol.png"
        make_main_figure(
            batch["afm_stack"][0].detach().cpu().numpy(),
            gt_coords,
            gt_types,
            gt_mask,
            gt_edge_adj,
            pred_coords,
            pred_types,
            pred_mask,
            pred_edge_adj,
            metrics,
            gt_cid,
            top3,
            gt_rank,
            args.run_label,
            main_path,
        )
        make_comparison_figure(
            gt_coords,
            gt_types,
            gt_mask,
            gt_edge_adj,
            pred_coords,
            pred_types,
            pred_mask,
            pred_edge_adj,
            top3,
            metrics,
            dataset_index,
            args.run_label,
            compar_path,
        )

        records.append(
            SampleRecord(
                dataset_index=dataset_index,
                gt_cid=gt_cid,
                gt_rank=gt_rank,
                top3_cids=[cid for cid, _, _, _, _ in top3],
                top3_sims=[float(sim) for _, sim, _, _, _ in top3],
                gt_in_top3=any(cid == gt_cid for cid, _, _, _, _ in top3),
                pred_atom_count=int(pred_mask.sum()),
                gt_atom_count=int(gt_mask.sum()),
                pred_object_score=float(metrics["pred_object_score"]),
                pred_object_type_acc=float(metrics["pred_object_type_acc"]),
                pred_object_macro_f1=float(metrics["pred_object_macro_f1"]),
                pred_object_hetero_f1=float(metrics["pred_object_hetero_f1"]),
                pred_object_edge_f1=float(metrics["pred_object_edge_f1"]),
                pred_object_z_mae=float(metrics["pred_object_z_mae"]),
                main_figure=str(main_path),
                compar_figure=str(compar_path),
            )
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "num_samples": len(records),
        "metrics_displayed": [{"field_name": f, "zh_meaning": zh} for f, zh in MAIN_METRICS],
        "gt_in_top3_count": int(sum(r.gt_in_top3 for r in records)),
        "records": [asdict(r) for r in records],
    }
    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md = []
    md.append(f"# {args.run_label} 测试集 15 样本对象级可视化报告")
    md.append("")
    md.append("## 六个展示指标")
    for field_name, zh in MAIN_METRICS:
        md.append(f"- `{field_name}`：{zh}")
    md.append("")
    md.append(f"- Top3 命中 GT 数量：`{summary['gt_in_top3_count']}/{len(records)}`")
    md.append("")
    md.append("| 样本编号 | GT CID | GT Rank | GT 是否在 Top3 | Pred 原子数 | GT 原子数 | pred_object_score | pred_object_type_acc | pred_object_macro_f1 | pred_object_hetero_f1 | pred_object_edge_f1 | pred_object_z_mae | Top1 | Top2 | Top3 |")
    md.append("|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for r in records:
        md.append(
            f"| {r.dataset_index} | {r.gt_cid} | {r.gt_rank} | {'是' if r.gt_in_top3 else '否'} | "
            f"{r.pred_atom_count} | {r.gt_atom_count} | "
            f"{r.pred_object_score:.4f} | {r.pred_object_type_acc:.4f} | {r.pred_object_macro_f1:.4f} | "
            f"{r.pred_object_hetero_f1:.4f} | {r.pred_object_edge_f1:.4f} | {r.pred_object_z_mae:.4f} | "
            f"{r.top3_cids[0] if len(r.top3_cids) > 0 else '-'} | "
            f"{r.top3_cids[1] if len(r.top3_cids) > 1 else '-'} | "
            f"{r.top3_cids[2] if len(r.top3_cids) > 2 else '-'} |"
        )
    with open(report_dir / "summary.md", "w") as f:
        f.write("\n".join(md))

    print(json.dumps(
        {
            "main_dir": str(main_dir),
            "compar_dir": str(compar_dir),
            "report_dir": str(report_dir),
            "num_samples": len(records),
            "gt_in_top3_count": summary["gt_in_top3_count"],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
