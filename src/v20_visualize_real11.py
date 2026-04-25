"""
Generate 11-sample real-AFM visualization outputs in a layout aligned with the
existing V20 15-sample validation visualizations.

For EDAFM cases with structure files, GT panels use the actual structure.
For camphor cases without explicit GT coordinates in the prepared case set,
reference panels use the best matching camphor candidate structure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_v19_object_joint import compute_pred_object_metrics, extract_predicted_objects
from src.utils.mol2d import infer_bonds_from_coords
from src.v19_object_joint_review import _plot_object_2d, _plot_object_3d
from src.v19_visualize_test15 import compute_v19_object_similarity_for_retrieval, load_model_bundle
from src.v20_eval_real_afm_cases import build_candidate_pool, retrieve_ranked_identity, _edge_adj_from_coords


GT_METRICS = [
    ("pred_object_score", "pred_object_score"),
    ("pred_object_type_acc", "pred_object_type_acc"),
    ("pred_object_macro_f1", "pred_object_macro_f1"),
    ("pred_object_edge_f1", "pred_object_edge_f1"),
    ("pred_object_edge_f1_robust", "pred_object_edge_f1_robust"),
    ("pred_object_z_mae", "pred_object_z_mae"),
]

REF_METRICS = [
    ("ref_object_sim", "ref_object_sim"),
    ("ref_type_acc", "ref_type_acc"),
    ("ref_macro_f1", "ref_macro_f1"),
    ("ref_edge_f1", "ref_edge_f1"),
    ("ref_coord_score", "ref_coord_score"),
    ("ref_count_score", "ref_count_score"),
]


@dataclass
class RealVisualRecord:
    case_id: str
    molecule_label: str
    tip: str
    chosen_variant: str
    gt_kind: str
    gt_rank: int
    top3_labels: list[str]
    top3_candidate_names: list[str]
    top3_sims: list[float]
    pred_atom_count: int
    ref_atom_count: int
    metric_block: dict[str, float]
    main_figure: str
    compar_figure: str


def load_summary(summary_json: Path) -> dict:
    return json.loads(summary_json.read_text())


def build_case_index(real_afm_roots: list[Path]) -> dict[str, dict]:
    out = {}
    for root in real_afm_roots:
        manifest = json.loads((root / "manifest.json").read_text())
        for case in manifest["cases"]:
            out[case["case_id"]] = {
                **case,
                "root": str(root),
            }
    return out


def choose_best_variant(normal_record: dict, inverted_record: dict) -> dict:
    def key(record: dict):
        gt_rank = int(record["gt_rank"])
        pred_score = record.get("pred_object_score")
        pred_score = -1.0 if pred_score is None else float(pred_score)
        return (
            -int(record["top1_hit"]),
            -int(record["top3_hit"]),
            -int(record["top5_hit"]),
            -float(record["reciprocal_rank"]),
            -float(record["top1_sim"]),
            -pred_score,
        )

    if key(inverted_record) > key(normal_record):
        return inverted_record
    return normal_record


def load_afm_stack(case_meta: dict, variant: str) -> np.ndarray:
    case_dir = Path(case_meta["root"]) / case_meta["case_dir"]
    name = "afm_stack_inverted.npy" if variant == "inverted" else "afm_stack.npy"
    return np.load(case_dir / name).astype(np.float32)


def load_gt_structure_from_case(case_meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    case_dir = Path(case_meta["root"]) / case_meta["case_dir"]
    coords_path = case_dir / "coords_norm.npy"
    types_path = case_dir / "atom_types.npy"
    if not coords_path.exists() or not types_path.exists():
        return None
    coords = np.load(coords_path).astype(np.float32)
    atom_types = np.load(types_path).astype(np.int64)
    mask = np.ones(len(atom_types), dtype=np.float32)
    return coords, atom_types, mask


def best_same_label_candidate(
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    candidate_pool: list[dict],
    label: str,
) -> tuple[dict, dict]:
    best_item = None
    best_sim = None
    for item in candidate_pool:
        if str(item["label"]) != label:
            continue
        sim = compute_v19_object_similarity_for_retrieval(
            pred_coords,
            pred_types,
            pred_mask,
            item["coords"],
            item["types"],
            item["mask"],
        )
        if best_item is None or float(sim["overall"]) > float(best_sim["overall"]):
            best_item = item
            best_sim = sim
    if best_item is None or best_sim is None:
        raise ValueError(f"no same-label candidate found for label={label}")
    return best_item, best_sim


def _text_metrics(metric_items: list[tuple[str, str]], metrics: dict[str, float]) -> str:
    lines = ["Six Core Metrics", "=" * 18]
    for key, label in metric_items:
        lines.append(f"{label}\n{metrics[key]:.4f}")
    return "\n\n".join(lines)


def _text_retrieval(
    gt_label: str,
    tip: str,
    gt_rank: int,
    top3: list[tuple[str, str, float, np.ndarray, np.ndarray, np.ndarray]],
) -> str:
    lines = ["Identity Top-3 Retrieval", "=" * 24]
    lines.append(f"GT Label: {gt_label}")
    lines.append(f"Tip: {tip}")
    lines.append(f"GT Rank: {gt_rank}")
    lines.append("")
    for i, (label, candidate_name, sim, _coords, _types, mask) in enumerate(top3, start=1):
        lines.append(f"Top-{i}: {label} | {candidate_name} | sim={sim:.4f} | n={int(mask.sum())}")
    lines.append("")
    lines.append(f"GT in Top3: {'Yes' if any(label == gt_label for label, *_ in top3) else 'No'}")
    return "\n".join(lines)


def make_main_figure(
    afm_stack: np.ndarray,
    ref_coords: np.ndarray,
    ref_types: np.ndarray,
    ref_mask: np.ndarray,
    ref_edge_adj: np.ndarray,
    ref_title_2d: str,
    ref_title_3d: str,
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    pred_edge_adj: np.ndarray,
    metric_items: list[tuple[str, str]],
    metrics: dict[str, float],
    molecule_label: str,
    tip: str,
    top3: list[tuple[str, str, float, np.ndarray, np.ndarray, np.ndarray]],
    gt_rank: int,
    chosen_variant: str,
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
    _plot_object_2d(ax, ref_coords, ref_types, ref_mask, ref_edge_adj, ref_title_2d)
    ax = fig.add_subplot(gs[1, 1])
    _plot_object_2d(ax, pred_coords, pred_types, pred_mask, pred_edge_adj, "Pred Object 2D")

    ax = fig.add_subplot(gs[1, 2], projection="3d")
    _plot_object_3d(ax, ref_coords, ref_types, ref_mask, ref_edge_adj, ref_title_3d)
    ax = fig.add_subplot(gs[1, 3], projection="3d")
    _plot_object_3d(ax, pred_coords, pred_types, pred_mask, pred_edge_adj, "Pred 3D Structure")

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    ax.text(
        0.02,
        0.95,
        _text_metrics(metric_items, metrics),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#eef3f7", alpha=0.95),
    )
    ax.text(
        0.62,
        0.95,
        f"Atom Count\nRef: {int(ref_mask.sum())}\nPred: {int(pred_mask.sum())}\nVariant: {chosen_variant}",
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#fff4e5", alpha=0.95),
    )
    fig.suptitle(f"V20 Real11 Visualization | Label={molecule_label} | tip={tip}", fontsize=14, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_comparison_figure(
    ref_coords: np.ndarray,
    ref_types: np.ndarray,
    ref_mask: np.ndarray,
    ref_edge_adj: np.ndarray,
    ref_title: str,
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    pred_edge_adj: np.ndarray,
    top3: list[tuple[str, str, float, np.ndarray, np.ndarray, np.ndarray]],
    footer_text: str,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(22, 5))
    axes = [fig.add_subplot(1, 5, i + 1, projection="3d") for i in range(5)]
    _plot_object_3d(axes[0], ref_coords, ref_types, ref_mask, ref_edge_adj, ref_title)
    _plot_object_3d(axes[1], pred_coords, pred_types, pred_mask, pred_edge_adj, "Predicted")
    for i, (_label, candidate_name, sim, coords, atom_types, mask) in enumerate(top3, start=2):
        edge_adj = _edge_adj_from_coords(coords, atom_types, mask)
        _plot_object_3d(axes[i], coords, atom_types, mask, edge_adj, f"Top-{i-1}\n{candidate_name}\nsim={sim:.3f}")
    fig.text(
        0.5,
        0.02,
        footer_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.9),
    )
    fig.suptitle("V20 Real11 3D Molecule Comparison", fontsize=13, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/micro/experiments/v20_object_joint_medium10/checkpoints/latest_v19_object_joint.pt")
    parser.add_argument("--summary_json", default="/root/autodl-tmp/micro/experiments/v20_object_joint_medium10_sup03_real_afm_expanded/reports/sup03_real_afm_summary.json")
    parser.add_argument("--real_afm_roots", default="/root/autodl-tmp/micro/real_afm/edafm_sup03_cases,/root/autodl-tmp/micro/real_afm/camphor_sup03_cases")
    parser.add_argument("--edafm_root", default="/root/autodl-tmp/real_afm_datasets/edafm_zenodo_10609676/edafm-data/edafm-data")
    parser.add_argument("--camphor_structure_root", default="/root/autodl-tmp/real_afm_datasets/camphor_adsorbate_4710346/structures")
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    summary = load_summary(Path(args.summary_json))
    real_afm_roots = [Path(x.strip()) for x in args.real_afm_roots.split(",") if x.strip()]
    case_index = build_case_index(real_afm_roots)
    candidate_pool = build_candidate_pool(Path(args.edafm_root), Path(args.camphor_structure_root))
    model, type_head, edge_head, config = load_model_bundle(checkpoint_path, device)
    img_size = int(config["img_size"])

    normal_records = {r["case_id"]: r for r in summary["normal"]["records"]}
    inverted_records = {r["case_id"]: r for r in summary["inverted"]["records"]}
    case_ids = sorted(normal_records.keys())

    output_root = Path(args.output_root)
    main_dir = output_root / "visualizations_real11"
    compar_dir = output_root / "visual_compar_real11"
    report_dir = output_root / "visual_reports_real11"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_records: list[RealVisualRecord] = []

    with torch.no_grad():
        for case_id in case_ids:
            chosen = choose_best_variant(normal_records[case_id], inverted_records[case_id])
            chosen_variant = str(chosen["contrast_variant"])
            case_meta = case_index[case_id]
            afm_stack = load_afm_stack(case_meta, chosen_variant)
            afm_tensor = torch.from_numpy(afm_stack).unsqueeze(0).to(device)

            pred, features = model.forward_with_features(afm_tensor)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            center_map_01 = torch.sigmoid(features["center_logits"])
            count_logits = features["count_logits"]
            pred_obj = extract_predicted_objects(
                center_map_01,
                pred_01,
                features["enc1"],
                afm_tensor,
                type_head,
                edge_head,
                device,
                img_size=img_size,
                count_logits=count_logits,
            )

            ranked, gt_rank = retrieve_ranked_identity(
                pred_obj["coords"],
                pred_obj["types"],
                pred_obj["mask"],
                candidate_pool,
                str(case_meta.get("molecule_label", case_meta["molecule_name"])),
            )
            top3 = ranked[:3]

            gt_struct = load_gt_structure_from_case(case_meta)
            if gt_struct is not None:
                ref_coords, ref_types, ref_mask = gt_struct
                ref_edge_adj = _edge_adj_from_coords(ref_coords, ref_types, ref_mask)
                gt_metrics = compute_pred_object_metrics(
                    pred_obj,
                    ref_coords,
                    ref_types,
                    ref_mask,
                    ref_edge_adj,
                    img_size=img_size,
                )
                metric_items = GT_METRICS
                metric_block = {key: float(gt_metrics[key]) for key, _ in GT_METRICS}
                ref_title_2d = "GT Object 2D"
                ref_title_3d = "GT 3D Structure"
                ref_title = "GT"
                gt_kind = "gt"
                footer = (
                    f"{case_id} | variant={chosen_variant} | pred_object_score={metric_block['pred_object_score']:.4f} | "
                    f"pred_object_type_acc={metric_block['pred_object_type_acc']:.4f} | "
                    f"pred_object_macro_f1={metric_block['pred_object_macro_f1']:.4f} | "
                    f"pred_object_edge_f1={metric_block['pred_object_edge_f1']:.4f} | "
                    f"pred_object_z_mae={metric_block['pred_object_z_mae']:.4f}"
                )
            else:
                ref_item, ref_sim = best_same_label_candidate(
                    pred_obj["coords"],
                    pred_obj["types"],
                    pred_obj["mask"],
                    candidate_pool,
                    str(case_meta.get("molecule_label", case_meta["molecule_name"])),
                )
                ref_coords = ref_item["coords"]
                ref_types = ref_item["types"]
                ref_mask = ref_item["mask"]
                ref_edge_adj = _edge_adj_from_coords(ref_coords, ref_types, ref_mask)
                metric_items = REF_METRICS
                metric_block = {
                    "ref_object_sim": float(ref_sim["overall"]),
                    "ref_type_acc": float(ref_sim["type_acc"]),
                    "ref_macro_f1": float(ref_sim["macro_f1"]),
                    "ref_edge_f1": float(ref_sim["edge_f1"]),
                    "ref_coord_score": float(ref_sim["coord_score"]),
                    "ref_count_score": float(ref_sim["count_score"]),
                }
                ref_title_2d = "Ref Object 2D"
                ref_title_3d = "Ref 3D Structure"
                ref_title = "Reference"
                gt_kind = "reference"
                footer = (
                    f"{case_id} | variant={chosen_variant} | ref_object_sim={metric_block['ref_object_sim']:.4f} | "
                    f"ref_type_acc={metric_block['ref_type_acc']:.4f} | "
                    f"ref_macro_f1={metric_block['ref_macro_f1']:.4f} | "
                    f"ref_edge_f1={metric_block['ref_edge_f1']:.4f} | "
                    f"ref_coord_score={metric_block['ref_coord_score']:.4f}"
                )

            main_path = main_dir / f"{case_id}.png"
            compar_path = compar_dir / f"{case_id}.png"
            make_main_figure(
                afm_stack=afm_stack,
                ref_coords=ref_coords,
                ref_types=ref_types,
                ref_mask=ref_mask,
                ref_edge_adj=ref_edge_adj,
                ref_title_2d=ref_title_2d,
                ref_title_3d=ref_title_3d,
                pred_coords=pred_obj["coords"],
                pred_types=pred_obj["types"],
                pred_mask=pred_obj["mask"],
                pred_edge_adj=pred_obj["edge_adj"],
                metric_items=metric_items,
                metrics=metric_block,
                molecule_label=str(case_meta.get("molecule_label", case_meta["molecule_name"])),
                tip=str(case_meta.get("tip", "unknown")),
                top3=top3,
                gt_rank=int(gt_rank),
                chosen_variant=chosen_variant,
                save_path=main_path,
            )
            make_comparison_figure(
                ref_coords=ref_coords,
                ref_types=ref_types,
                ref_mask=ref_mask,
                ref_edge_adj=ref_edge_adj,
                ref_title=ref_title,
                pred_coords=pred_obj["coords"],
                pred_types=pred_obj["types"],
                pred_mask=pred_obj["mask"],
                pred_edge_adj=pred_obj["edge_adj"],
                top3=top3,
                footer_text=footer,
                save_path=compar_path,
            )

            report_records.append(
                RealVisualRecord(
                    case_id=case_id,
                    molecule_label=str(case_meta.get("molecule_label", case_meta["molecule_name"])),
                    tip=str(case_meta.get("tip", "unknown")),
                    chosen_variant=chosen_variant,
                    gt_kind=gt_kind,
                    gt_rank=int(gt_rank),
                    top3_labels=[str(x[0]) for x in top3],
                    top3_candidate_names=[str(x[1]) for x in top3],
                    top3_sims=[float(x[2]) for x in top3],
                    pred_atom_count=int(np.sum(pred_obj["mask"] > 0.5)),
                    ref_atom_count=int(np.sum(ref_mask > 0.5)),
                    metric_block=metric_block,
                    main_figure=str(main_path),
                    compar_figure=str(compar_path),
                )
            )

    summary_json = {
        "checkpoint": str(checkpoint_path),
        "summary_source": str(args.summary_json),
        "num_cases": len(report_records),
        "records": [asdict(r) for r in report_records],
    }
    (report_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# V20 Real11 Visualization Summary")
    md.append("")
    md.append(f"- checkpoint: `{checkpoint_path}`")
    md.append(f"- num_cases: `{len(report_records)}`")
    md.append("")
    md.append("| case_id | label | tip | variant | kind | gt_rank | pred_atoms | ref_atoms | top3 labels |")
    md.append("|---|---|---|---|---|---:|---:|---:|---|")
    for r in report_records:
        md.append(
            f"| {r.case_id} | {r.molecule_label} | {r.tip} | {r.chosen_variant} | {r.gt_kind} | "
            f"{r.gt_rank} | {r.pred_atom_count} | {r.ref_atom_count} | {', '.join(r.top3_labels)} |"
        )
    md.append("")
    md.append("- `kind=gt` 表示左侧参考面板使用真实结构。")
    md.append("- `kind=reference` 表示左侧参考面板使用同身份最佳参考构型。")
    (report_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "main_dir": str(main_dir),
                "compar_dir": str(compar_dir),
                "report_dir": str(report_dir),
                "num_cases": len(report_records),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
