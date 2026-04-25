"""
EXP-03:
Strict vs robust object-graph evaluation and gap decomposition on full test.

Outputs:
- overall strict / robust / matched-r3 summary
- per-sample gap records (json/csv)
- diagnostic plots
- top-gap qualitative figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.train_v19_object_joint import (
    _infer_edge_set_from_adj,
    _matched_edge_f1_with_gate,
    build_edge_labels,
    compute_pred_object_metrics,
    extract_predicted_objects,
)
from src.utils.metrics import _hungarian_match_numpy, _macro_type_f1, _safe_f1
from src.v19_object_joint_review import _load_model, _make_sample_figure


EDGE_MATCH_RADIUS_PX = 3.0


@dataclass
class GapRecord:
    dataset_index: int
    cid: str
    gt_atom_count: int
    pred_atom_count: int
    pred_object_score: float
    pred_object_type_acc: float
    pred_object_macro_f1: float
    pred_object_hetero_f1: float
    pred_object_edge_f1: float
    pred_object_edge_f1_robust: float
    pred_object_match_coverage_robust: float
    pred_object_z_mae: float
    matched_xy_count_r3: int
    matched_gt_node_coverage_r3: float
    matched_gt_bond_coverage_r3: float
    edge_f1_xy_r3: float
    edge_gap_robust: float
    edge_gap_xy_r3: float
    matched_type_acc_r3: Optional[float]
    matched_macro_f1_r3: Optional[float]
    matched_hetero_f1_r3: Optional[float]
    mean_xy_match_px_r3: Optional[float]
    mean_z_abs_err_matched_ang: Optional[float]
    matched_pair_dist_mae_ang: Optional[float]
    matched_bond_len_mae_ang: Optional[float]
    figure_path: str = ""


OPTIONAL_MEAN_FIELDS = [
    "matched_type_acc_r3",
    "matched_macro_f1_r3",
    "matched_hetero_f1_r3",
    "mean_xy_match_px_r3",
    "mean_z_abs_err_matched_ang",
    "matched_pair_dist_mae_ang",
    "matched_bond_len_mae_ang",
]

BASE_MEAN_FIELDS = [
    "pred_object_score",
    "pred_object_type_acc",
    "pred_object_macro_f1",
    "pred_object_hetero_f1",
    "pred_object_edge_f1",
    "pred_object_edge_f1_robust",
    "pred_object_match_coverage_robust",
    "pred_object_z_mae",
    "matched_gt_node_coverage_r3",
    "matched_gt_bond_coverage_r3",
    "edge_f1_xy_r3",
    "edge_gap_robust",
    "edge_gap_xy_r3",
]


def _optional_mean(records: list[GapRecord], field: str) -> Optional[float]:
    vals = [getattr(r, field) for r in records if getattr(r, field) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _mean_dict(records: list[GapRecord]) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for field in BASE_MEAN_FIELDS:
        out[field] = float(np.mean([getattr(r, field) for r in records])) if records else 0.0
    for field in OPTIONAL_MEAN_FIELDS:
        out[field] = _optional_mean(records, field)
    return out


def _safe_csv_value(value):
    return "" if value is None else value


def _write_csv(records: list[GapRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fieldnames = list(asdict(records[0]).keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {k: _safe_csv_value(v) for k, v in asdict(record).items()}
            writer.writerow(row)


def _pairwise_dist_mae(pred_coords_ang: np.ndarray, gt_coords_ang: np.ndarray) -> Optional[float]:
    if len(pred_coords_ang) < 2 or len(gt_coords_ang) < 2:
        return None
    pred_diff = pred_coords_ang[:, None, :] - pred_coords_ang[None, :, :]
    gt_diff = gt_coords_ang[:, None, :] - gt_coords_ang[None, :, :]
    pred_d = np.linalg.norm(pred_diff, axis=-1)
    gt_d = np.linalg.norm(gt_diff, axis=-1)
    iu = np.triu_indices(len(pred_coords_ang), k=1)
    return float(np.mean(np.abs(pred_d[iu] - gt_d[iu])))


def _bond_len_mae(
    pred_coords_ang: np.ndarray,
    gt_coords_ang: np.ndarray,
    accepted_gt_nodes: set[int],
    gt_to_pred: dict[int, int],
    gt_edges: set[tuple[int, int]],
) -> tuple[Optional[float], float]:
    if not gt_edges:
        return None, 1.0
    diffs = []
    covered = 0
    for i, j in gt_edges:
        if i not in accepted_gt_nodes or j not in accepted_gt_nodes:
            continue
        if i not in gt_to_pred or j not in gt_to_pred:
            continue
        pi = gt_to_pred[i]
        pj = gt_to_pred[j]
        pred_len = float(np.linalg.norm(pred_coords_ang[pi] - pred_coords_ang[pj]))
        gt_len = float(np.linalg.norm(gt_coords_ang[i] - gt_coords_ang[j]))
        diffs.append(abs(pred_len - gt_len))
        covered += 1
    coverage = float(covered / max(len(gt_edges), 1))
    if not diffs:
        return None, coverage
    return float(np.mean(diffs)), coverage


def _type_metrics_subset(pred_types: np.ndarray, gt_types: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if len(pred_types) == 0 or len(gt_types) == 0:
        return None, None, None
    type_acc = float(np.mean(pred_types == gt_types))
    macro_f1 = float(_macro_type_f1(pred_types, gt_types, pred_types, gt_types))
    pred_het = ~np.isin(pred_types, [0, 1])
    gt_het = ~np.isin(gt_types, [0, 1])
    tp = int((pred_het & gt_het).sum())
    fp = int((pred_het & ~gt_het).sum())
    fn = int((~pred_het & gt_het).sum())
    _, _, hetero_f1 = _safe_f1(tp, fp, fn)
    return type_acc, macro_f1, hetero_f1


def compute_gap_metrics(
    pred_obj: dict,
    gt_coords: np.ndarray,
    gt_types: np.ndarray,
    gt_mask: np.ndarray,
    gt_edge_adj: np.ndarray,
    img_size: int,
    edge_match_radius_px: float = EDGE_MATCH_RADIUS_PX,
) -> dict:
    pred_metrics = compute_pred_object_metrics(
        pred_obj,
        gt_coords,
        gt_types,
        gt_mask,
        gt_edge_adj,
        img_size=img_size,
        edge_match_radius_px=edge_match_radius_px,
    )

    n_pred = int((pred_obj["mask"] > 0.5).sum())
    n_gt = int((gt_mask > 0.5).sum())
    if n_pred == 0 or n_gt == 0:
        return {
            **pred_metrics,
            "matched_xy_count_r3": 0,
            "matched_gt_node_coverage_r3": 0.0,
            "matched_gt_bond_coverage_r3": 0.0,
            "edge_f1_xy_r3": 0.0,
            "edge_gap_robust": float(pred_metrics["pred_object_edge_f1_robust"] - pred_metrics["pred_object_edge_f1"]),
            "edge_gap_xy_r3": float(-pred_metrics["pred_object_edge_f1"]),
            "matched_type_acc_r3": None,
            "matched_macro_f1_r3": None,
            "matched_hetero_f1_r3": None,
            "mean_xy_match_px_r3": None,
            "mean_z_abs_err_matched_ang": None,
            "matched_pair_dist_mae_ang": None,
            "matched_bond_len_mae_ang": None,
        }

    pc = pred_obj["coords"][:n_pred].astype(np.float32)
    pt = pred_obj["types"][:n_pred].astype(np.int64)
    gc = gt_coords[:n_gt].astype(np.float32)
    gt = gt_types[:n_gt].astype(np.int64)

    row_xy, col_xy, cost_xy = _hungarian_match_numpy(pc[:, :2], gc[:, :2])
    xy_radius_norm = 2.0 * float(edge_match_radius_px) / float(max(img_size - 1, 1))
    accepted_pairs = [
        (int(p), int(g))
        for p, g in zip(row_xy.tolist(), col_xy.tolist())
        if float(cost_xy[p, g]) <= xy_radius_norm
    ]

    accepted_pred_nodes = {p for p, _ in accepted_pairs}
    accepted_gt_nodes = {g for _, g in accepted_pairs}
    pred_to_gt_xy = {int(p): int(g) for p, g in zip(row_xy.tolist(), col_xy.tolist())}
    gt_to_pred_xy = {g: p for p, g in accepted_pairs}

    gt_edges = _infer_edge_set_from_adj(gt_edge_adj, gt_mask)
    pred_edges = _infer_edge_set_from_adj(pred_obj["edge_adj"], pred_obj["mask"])
    _, _, edge_f1_xy_r3 = _matched_edge_f1_with_gate(
        pred_edges,
        gt_edges,
        pred_to_gt_xy,
        accepted_pred_nodes,
    )

    matched_gt_node_coverage = float(len(accepted_gt_nodes) / max(n_gt, 1))
    matched_pred_types = np.asarray([pt[p] for p, _ in accepted_pairs], dtype=np.int64)
    matched_gt_types = np.asarray([gt[g] for _, g in accepted_pairs], dtype=np.int64)
    matched_type_acc_r3, matched_macro_f1_r3, matched_hetero_f1_r3 = _type_metrics_subset(
        matched_pred_types,
        matched_gt_types,
    )

    mean_xy_match_px_r3 = None
    if accepted_pairs:
        xy_costs = [float(cost_xy[p, g]) for p, g in accepted_pairs]
        mean_xy_match_px_r3 = float(np.mean(xy_costs) * float(max(img_size - 1, 1)) * 0.5)

    pred_coords_ang = pc * 12.0
    gt_coords_ang = gc * 12.0
    mean_z_abs_err_matched_ang = None
    matched_pair_dist_mae_ang = None
    if accepted_pairs:
        pred_idx = [p for p, _ in accepted_pairs]
        gt_idx = [g for _, g in accepted_pairs]
        pred_sub = pred_coords_ang[pred_idx]
        gt_sub = gt_coords_ang[gt_idx]
        mean_z_abs_err_matched_ang = float(np.mean(np.abs(pred_sub[:, 2] - gt_sub[:, 2])))
        matched_pair_dist_mae_ang = _pairwise_dist_mae(pred_sub, gt_sub)

    matched_bond_len_mae_ang, matched_gt_bond_coverage_r3 = _bond_len_mae(
        pred_coords_ang,
        gt_coords_ang,
        accepted_gt_nodes,
        gt_to_pred_xy,
        gt_edges,
    )

    return {
        **pred_metrics,
        "matched_xy_count_r3": int(len(accepted_pairs)),
        "matched_gt_node_coverage_r3": matched_gt_node_coverage,
        "matched_gt_bond_coverage_r3": matched_gt_bond_coverage_r3,
        "edge_f1_xy_r3": float(edge_f1_xy_r3),
        "edge_gap_robust": float(pred_metrics["pred_object_edge_f1_robust"] - pred_metrics["pred_object_edge_f1"]),
        "edge_gap_xy_r3": float(edge_f1_xy_r3 - pred_metrics["pred_object_edge_f1"]),
        "matched_type_acc_r3": matched_type_acc_r3,
        "matched_macro_f1_r3": matched_macro_f1_r3,
        "matched_hetero_f1_r3": matched_hetero_f1_r3,
        "mean_xy_match_px_r3": mean_xy_match_px_r3,
        "mean_z_abs_err_matched_ang": mean_z_abs_err_matched_ang,
        "matched_pair_dist_mae_ang": matched_pair_dist_mae_ang,
        "matched_bond_len_mae_ang": matched_bond_len_mae_ang,
    }


def _scatter(records: list[GapRecord], x_field: str, y_field: str, title: str, output_path: Path) -> None:
    xs = []
    ys = []
    for r in records:
        x = getattr(r, x_field)
        y = getattr(r, y_field)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(xs, ys, s=18, alpha=0.75, color="#2563eb")
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _hist(values: list[float], title: str, xlabel: str, output_path: Path) -> None:
    if not values:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.hist(values, bins=24, color="#0f766e", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_plots(records: list[GapRecord], output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    _hist(
        [r.edge_gap_robust for r in records],
        "Strict -> Robust Edge Gap",
        "pred_object_edge_f1_robust - pred_object_edge_f1",
        plots_dir / "edge_gap_robust_hist.png",
    )
    _hist(
        [r.edge_gap_xy_r3 for r in records],
        "Strict -> XY-R3 Edge Gap",
        "edge_f1_xy_r3 - pred_object_edge_f1",
        plots_dir / "edge_gap_xy_r3_hist.png",
    )
    _scatter(
        records,
        "matched_gt_node_coverage_r3",
        "edge_gap_robust",
        "Edge Gap vs Matched Node Coverage",
        plots_dir / "gap_vs_node_coverage.png",
    )
    _scatter(
        records,
        "pred_object_type_acc",
        "matched_type_acc_r3",
        "Global Type Acc vs Matched-R3 Type Acc",
        plots_dir / "global_vs_matched_type_acc.png",
    )
    _scatter(
        records,
        "pred_object_z_mae",
        "edge_gap_robust",
        "Edge Gap vs Pred Object Z MAE",
        plots_dir / "gap_vs_z_mae.png",
    )
    _scatter(
        records,
        "edge_gap_robust",
        "matched_bond_len_mae_ang",
        "Edge Gap vs Matched Bond Length MAE",
        plots_dir / "gap_vs_bond_len_mae.png",
    )


def _top_gap_figures(
    model,
    type_head,
    edge_head,
    loader,
    config: dict,
    device: torch.device,
    records: list[GapRecord],
    output_dir: Path,
) -> None:
    selected = sorted(records, key=lambda r: r.edge_gap_robust, reverse=True)[:3]
    selected_idx = {r.dataset_index: r for r in selected}
    if not selected_idx:
        return

    img_size = int(config["img_size"])
    sample_index = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="EXP-03 gap figs", leave=False):
            afm = batch["afm_stack"].to(device)
            coords = batch["coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)
            edge_labels = build_edge_labels(batch, device)

            pred, features = model.forward_with_features(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            center_map_01 = torch.sigmoid(features["center_logits"]).clamp(0.0, 1.0)

            for bi in range(afm.shape[0]):
                if sample_index in selected_idx:
                    record = selected_idx[sample_index]
                    sample_afm = afm[bi : bi + 1]
                    sample_feat = features["enc1"][bi : bi + 1]
                    sample_pred = pred_01[bi : bi + 1]
                    sample_center_map = center_map_01[bi : bi + 1]
                    sample_coords = coords[bi : bi + 1]
                    sample_types = atom_types[bi : bi + 1]
                    sample_mask = mask[bi : bi + 1]
                    sample_edge_labels = edge_labels[bi : bi + 1]
                    pred_obj = extract_predicted_objects(
                        sample_center_map,
                        sample_pred,
                        sample_feat,
                        sample_afm,
                        type_head,
                        edge_head,
                        device,
                        img_size=img_size,
                        count_logits=features.get("count_logits", None)[bi : bi + 1]
                        if features.get("count_logits", None) is not None
                        else None,
                        peak_threshold=float(config.get("proposal_peak_threshold", 0.45)),
                        min_distance_px=int(config.get("proposal_min_distance_px", 2)),
                        max_objects=int(config.get("proposal_max_objects", 64)),
                    )
                    title = (
                        f"TOP GAP | idx={record.dataset_index} | cid={record.cid} | "
                        f"strict_edge={record.pred_object_edge_f1:.3f} | "
                        f"robust_edge={record.pred_object_edge_f1_robust:.3f} | "
                        f"gap={record.edge_gap_robust:.3f} | "
                        f"matched_type={record.matched_type_acc_r3 if record.matched_type_acc_r3 is not None else -1:.3f}"
                    )
                    fig_path = output_dir / "samples" / f"top_gap_sample_{record.dataset_index:04d}.png"
                    _make_sample_figure(
                        afm[bi].detach().cpu().numpy(),
                        np.zeros((13, img_size, img_size), dtype=np.float32),
                        pred_01[bi].detach().cpu().numpy(),
                        center_map_01[bi].detach().cpu().numpy(),
                        sample_coords[0].detach().cpu().numpy(),
                        sample_types[0].detach().cpu().numpy(),
                        sample_mask[0].detach().cpu().numpy(),
                        sample_edge_labels[0].detach().cpu().numpy().astype(np.int32),
                        pred_obj["coords"],
                        pred_obj["types"],
                        pred_obj["mask"],
                        pred_obj["edge_adj"],
                        title,
                        fig_path,
                    )
                    record.figure_path = str(fig_path)
                sample_index += 1


def _build_summary(
    checkpoint_path: Path,
    config: dict,
    records: list[GapRecord],
) -> tuple[dict, str]:
    mean_metrics = _mean_dict(records)
    high_gap = [r for r in records if r.edge_gap_robust >= 0.20]
    robust_good_strict_bad = [
        r for r in records
        if r.pred_object_edge_f1 < 0.70 and r.pred_object_edge_f1_robust >= 0.90
    ]
    type_gain = [
        r for r in records
        if r.matched_type_acc_r3 is not None and (r.matched_type_acc_r3 - r.pred_object_type_acc) >= 0.10
    ]
    top_gap = sorted(records, key=lambda r: r.edge_gap_robust, reverse=True)[:10]
    low_gap = sorted(records, key=lambda r: r.edge_gap_robust)[:10]

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": "test",
        "num_samples": len(records),
        "edge_match_radius_px": EDGE_MATCH_RADIUS_PX,
        "mean_metrics": mean_metrics,
        "counts": {
            "high_gap_ge_0p20": len(high_gap),
            "high_gap_ge_0p20_ratio": float(len(high_gap) / max(len(records), 1)),
            "robust_ge_0p90_and_strict_lt_0p70": len(robust_good_strict_bad),
            "robust_ge_0p90_and_strict_lt_0p70_ratio": float(len(robust_good_strict_bad) / max(len(records), 1)),
            "matched_type_gain_ge_0p10": len(type_gain),
            "matched_type_gain_ge_0p10_ratio": float(len(type_gain) / max(len(records), 1)),
        },
        "high_gap_mean_metrics": _mean_dict(high_gap) if high_gap else {},
        "robust_good_strict_bad_mean_metrics": _mean_dict(robust_good_strict_bad) if robust_good_strict_bad else {},
        "top_gap_samples": [asdict(r) for r in top_gap],
        "low_gap_samples": [asdict(r) for r in low_gap],
    }

    md: list[str] = []
    md.append("# V20 EXP-03 Strict vs Robust Gap Decomposition")
    md.append("")
    md.append("## 一、实验设置")
    md.append(f"- checkpoint：`{checkpoint_path}`")
    md.append("- split：`test`")
    md.append(f"- 样本数：`{len(records)}`")
    md.append(f"- `matched-r3` 半径：`{EDGE_MATCH_RADIUS_PX}` 像素")
    md.append("")
    md.append("## 二、核心均值")
    core_fields = [
        ("pred_object_edge_f1", "严格对象级边F1"),
        ("pred_object_edge_f1_robust", "现有稳健边F1"),
        ("edge_f1_xy_r3", "按XY半径3像素门限后的边F1"),
        ("edge_gap_robust", "稳健边F1与严格边F1的差值"),
        ("edge_gap_xy_r3", "XY-R3边F1与严格边F1的差值"),
        ("pred_object_type_acc", "全局纯预测对象类型准确率"),
        ("matched_type_acc_r3", "matched-r3 类型准确率"),
        ("pred_object_macro_f1", "全局纯预测对象类型宏平均F1"),
        ("matched_macro_f1_r3", "matched-r3 类型宏平均F1"),
        ("matched_gt_node_coverage_r3", "matched-r3 GT节点覆盖率"),
        ("matched_gt_bond_coverage_r3", "matched-r3 GT键覆盖率"),
        ("mean_xy_match_px_r3", "matched-r3 平均XY匹配误差(像素)"),
        ("pred_object_z_mae", "全局纯预测对象z误差"),
        ("mean_z_abs_err_matched_ang", "matched-r3 平均z绝对误差(Å)"),
        ("matched_pair_dist_mae_ang", "matched-r3 成对距离MAE(Å)"),
        ("matched_bond_len_mae_ang", "matched-r3 键长MAE(Å)"),
    ]
    for field, zh in core_fields:
        value = mean_metrics.get(field)
        if value is None:
            md.append(f"- `{field}`：{zh} = `N/A`")
        else:
            md.append(f"- `{field}`：{zh} = `{value:.4f}`")
    md.append("")
    md.append("## 三、计数型判断")
    md.append(
        f"- `edge_gap_robust >= 0.20` 的样本数：`{summary['counts']['high_gap_ge_0p20']}` / `{len(records)}` "
        f"= `{100.0 * summary['counts']['high_gap_ge_0p20_ratio']:.2f}%`"
    )
    md.append(
        f"- `pred_object_edge_f1 < 0.70` 且 `pred_object_edge_f1_robust >= 0.90` 的样本数："
        f"`{summary['counts']['robust_ge_0p90_and_strict_lt_0p70']}` / `{len(records)}` "
        f"= `{100.0 * summary['counts']['robust_ge_0p90_and_strict_lt_0p70_ratio']:.2f}%`"
    )
    md.append(
        f"- `matched_type_acc_r3 - pred_object_type_acc >= 0.10` 的样本数："
        f"`{summary['counts']['matched_type_gain_ge_0p10']}` / `{len(records)}` "
        f"= `{100.0 * summary['counts']['matched_type_gain_ge_0p10_ratio']:.2f}%`"
    )
    md.append("")
    if high_gap:
        high_gap_mean = summary["high_gap_mean_metrics"]
        md.append("## 四、高 gap 子集均值")
        md.append(f"- 高 gap 子集数量：`{len(high_gap)}`")
        for field, zh in [
            ("pred_object_edge_f1", "严格边F1"),
            ("pred_object_edge_f1_robust", "稳健边F1"),
            ("matched_gt_node_coverage_r3", "matched-r3 GT节点覆盖率"),
            ("matched_type_acc_r3", "matched-r3 类型准确率"),
            ("matched_macro_f1_r3", "matched-r3 类型宏平均F1"),
            ("matched_bond_len_mae_ang", "matched-r3 键长MAE(Å)"),
            ("pred_object_z_mae", "纯预测对象z误差"),
        ]:
            value = high_gap_mean.get(field)
            if value is None:
                md.append(f"- `{field}`：{zh} = `N/A`")
            else:
                md.append(f"- `{field}`：{zh} = `{value:.4f}`")
        md.append("")
    md.append("## 五、Top Gap 样本")
    for r in top_gap[:5]:
        md.append(
            f"- `idx={r.dataset_index}` `cid={r.cid}` "
            f"`strict={r.pred_object_edge_f1:.4f}` "
            f"`robust={r.pred_object_edge_f1_robust:.4f}` "
            f"`xy_r3={r.edge_f1_xy_r3:.4f}` "
            f"`gap={r.edge_gap_robust:.4f}` "
            f"`coverage={r.matched_gt_node_coverage_r3:.4f}` "
            f"`matched_type={r.matched_type_acc_r3 if r.matched_type_acc_r3 is not None else -1:.4f}` "
            f"`bond_len_mae={r.matched_bond_len_mae_ang if r.matched_bond_len_mae_ang is not None else -1:.4f}`"
        )
    md.append("")
    md.append("## 六、结论")
    md.append("- 如果 `pred_object_edge_f1_robust` 和 `edge_f1_xy_r3` 明显高于 `pred_object_edge_f1`，说明严格对象级边F1被对象错位显著放大。")
    md.append("- 如果 `matched_type_acc_r3` 明显高于全局 `pred_object_type_acc`，说明一部分类型错误来自错对象匹配，而不是类型头本身完全失效。")
    md.append("- 如果 `matched_bond_len_mae_ang` 和 `matched_pair_dist_mae_ang` 仍较小，则说明在正确对齐的对象子集上，局部几何已经明显优于 strict graph 指标给出的印象。")
    md.append("")
    return summary, "\n".join(md)


def evaluate(
    model,
    type_head,
    edge_head,
    loader,
    config: dict,
    device: torch.device,
) -> list[GapRecord]:
    records: list[GapRecord] = []
    img_size = int(config["img_size"])
    dataset = loader.dataset
    sample_index = 0

    model.eval()
    type_head.eval()
    edge_head.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="EXP-03 gap eval", leave=False):
            afm = batch["afm_stack"].to(device)
            coords = batch["coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)
            edge_labels = build_edge_labels(batch, device)

            pred, features = model.forward_with_features(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            center_map_01 = torch.sigmoid(features["center_logits"]).clamp(0.0, 1.0)

            for bi in range(afm.shape[0]):
                sample_afm = afm[bi : bi + 1]
                sample_feat = features["enc1"][bi : bi + 1]
                sample_pred = pred_01[bi : bi + 1]
                sample_center_map = center_map_01[bi : bi + 1]
                sample_coords = coords[bi : bi + 1]
                sample_types = atom_types[bi : bi + 1]
                sample_mask = mask[bi : bi + 1]
                sample_edge_labels = edge_labels[bi : bi + 1]

                pred_obj = extract_predicted_objects(
                    sample_center_map,
                    sample_pred,
                    sample_feat,
                    sample_afm,
                    type_head,
                    edge_head,
                    device,
                    img_size=img_size,
                    count_logits=features.get("count_logits", None)[bi : bi + 1]
                    if features.get("count_logits", None) is not None
                    else None,
                    peak_threshold=float(config.get("proposal_peak_threshold", 0.45)),
                    min_distance_px=int(config.get("proposal_min_distance_px", 2)),
                    max_objects=int(config.get("proposal_max_objects", 64)),
                )

                gt_coords = sample_coords[0].detach().cpu().numpy()
                gt_types = sample_types[0].detach().cpu().numpy()
                gt_mask = sample_mask[0].detach().cpu().numpy()
                gt_edge_adj = sample_edge_labels[0].detach().cpu().numpy().astype(np.int32)
                gap = compute_gap_metrics(
                    pred_obj,
                    gt_coords,
                    gt_types,
                    gt_mask,
                    gt_edge_adj,
                    img_size=img_size,
                    edge_match_radius_px=EDGE_MATCH_RADIUS_PX,
                )
                cid = str(dataset.samples[sample_index]["cid"])
                records.append(
                    GapRecord(
                        dataset_index=sample_index,
                        cid=cid,
                        gt_atom_count=int((gt_mask > 0.5).sum()),
                        pred_atom_count=int((pred_obj["mask"] > 0.5).sum()),
                        pred_object_score=float(gap["pred_object_score"]),
                        pred_object_type_acc=float(gap["pred_object_type_acc"]),
                        pred_object_macro_f1=float(gap["pred_object_macro_f1"]),
                        pred_object_hetero_f1=float(gap["pred_object_hetero_f1"]),
                        pred_object_edge_f1=float(gap["pred_object_edge_f1"]),
                        pred_object_edge_f1_robust=float(gap["pred_object_edge_f1_robust"]),
                        pred_object_match_coverage_robust=float(gap["pred_object_match_coverage_robust"]),
                        pred_object_z_mae=float(gap["pred_object_z_mae"]),
                        matched_xy_count_r3=int(gap["matched_xy_count_r3"]),
                        matched_gt_node_coverage_r3=float(gap["matched_gt_node_coverage_r3"]),
                        matched_gt_bond_coverage_r3=float(gap["matched_gt_bond_coverage_r3"]),
                        edge_f1_xy_r3=float(gap["edge_f1_xy_r3"]),
                        edge_gap_robust=float(gap["edge_gap_robust"]),
                        edge_gap_xy_r3=float(gap["edge_gap_xy_r3"]),
                        matched_type_acc_r3=gap["matched_type_acc_r3"],
                        matched_macro_f1_r3=gap["matched_macro_f1_r3"],
                        matched_hetero_f1_r3=gap["matched_hetero_f1_r3"],
                        mean_xy_match_px_r3=gap["mean_xy_match_px_r3"],
                        mean_z_abs_err_matched_ang=gap["mean_z_abs_err_matched_ang"],
                        matched_pair_dist_mae_ang=gap["matched_pair_dist_mae_ang"],
                        matched_bond_len_mae_ang=gap["matched_bond_len_mae_ang"],
                    )
                )
                sample_index += 1
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, type_head, edge_head, config, _ = _load_model(checkpoint_path, device)
    train_loader, val_loader, test_loader, _ = create_dataloaders(
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
    del train_loader, val_loader

    records = evaluate(model, type_head, edge_head, test_loader, config, device)
    _save_plots(records, output_dir)
    _top_gap_figures(model, type_head, edge_head, test_loader, config, device, records, output_dir)
    summary, md = _build_summary(checkpoint_path, config, records)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "gap_decomposition_test.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(reports_dir / "gap_decomposition_test.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(reports_dir / "gap_decomposition_test_records.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)
    _write_csv(records, reports_dir / "gap_decomposition_test_records.csv")

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "report_json": str(reports_dir / "gap_decomposition_test.json"),
            "report_md": str(reports_dir / "gap_decomposition_test.md"),
            "num_samples": len(records),
            "pred_object_edge_f1": summary["mean_metrics"]["pred_object_edge_f1"],
            "pred_object_edge_f1_robust": summary["mean_metrics"]["pred_object_edge_f1_robust"],
            "edge_f1_xy_r3": summary["mean_metrics"]["edge_f1_xy_r3"],
            "edge_gap_robust": summary["mean_metrics"]["edge_gap_robust"],
            "matched_type_acc_r3": summary["mean_metrics"]["matched_type_acc_r3"],
            "matched_macro_f1_r3": summary["mean_metrics"]["matched_macro_f1_r3"],
            "matched_gt_node_coverage_r3": summary["mean_metrics"]["matched_gt_node_coverage_r3"],
            "matched_bond_len_mae_ang": summary["mean_metrics"]["matched_bond_len_mae_ang"],
            "high_gap_ge_0p20_ratio": summary["counts"]["high_gap_ge_0p20_ratio"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
