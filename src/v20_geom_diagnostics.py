"""
EXP-04:
Geometry / 3D diagnostics on the V20 object-joint model full test split.

Outputs:
- full-test geometry summary
- per-sample geometry diagnostics (json/csv)
- stratified tables by atom count / ring count / height span / nonplanarity
- diagnostic plots for JMGM-facing 2D+z evidence
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
    _hungarian_match_numpy,
    _infer_edge_set_from_adj,
    build_edge_labels,
    compute_pred_object_metrics,
    extract_predicted_objects,
)
from src.v19_object_joint_review import _load_model


ANGSTROM_SCALE = 12.0
XY_MATCH_RADIUS_PX = 3.0


@dataclass
class GeometryRecord:
    dataset_index: int
    cid: str
    gt_atom_count: int
    gt_ring_count: int
    pred_atom_count: int
    gt_height_span_ang: float
    gt_nonplanarity_ang: float
    pred_object_score: float
    pred_object_type_acc: float
    pred_object_edge_f1: float
    pred_object_edge_f1_robust: float
    pred_object_match_coverage_robust: float
    pred_object_heavy_rmsd_ang: Optional[float]
    pred_object_z_mae: float
    pred_object_pair_dist_mae_r3: Optional[float]
    pred_object_bond_len_mae_r3: Optional[float]
    pred_object_z_corr_r3: Optional[float]
    pred_object_nonplanarity_error_r3: Optional[float]
    matched_gt_node_coverage_r3: float
    matched_gt_bond_coverage_r3: float
    matched_xy_count_r3: int
    mean_xy_match_px_r3: Optional[float]


MEAN_FIELDS = [
    "pred_object_score",
    "pred_object_type_acc",
    "pred_object_edge_f1",
    "pred_object_edge_f1_robust",
    "pred_object_match_coverage_robust",
    "pred_object_z_mae",
    "matched_gt_node_coverage_r3",
    "matched_gt_bond_coverage_r3",
    "gt_height_span_ang",
    "gt_nonplanarity_ang",
]

OPTIONAL_MEAN_FIELDS = [
    "pred_object_heavy_rmsd_ang",
    "pred_object_pair_dist_mae_r3",
    "pred_object_bond_len_mae_r3",
    "pred_object_z_corr_r3",
    "pred_object_nonplanarity_error_r3",
    "mean_xy_match_px_r3",
]


def _safe_csv_value(value):
    return "" if value is None else value


def _write_csv(records: list[GeometryRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fieldnames = list(asdict(records[0]).keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: _safe_csv_value(v) for k, v in asdict(record).items()})


def _optional_mean(records: list[GeometryRecord], field: str) -> Optional[float]:
    vals = [getattr(r, field) for r in records if getattr(r, field) is not None]
    return float(np.mean(vals)) if vals else None


def _optional_rate(records: list[GeometryRecord], field: str, predicate) -> Optional[float]:
    vals = [getattr(r, field) for r in records if getattr(r, field) is not None]
    if not vals:
        return None
    return float(np.mean([predicate(v) for v in vals]))


def _mean_metrics(records: list[GeometryRecord]) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for field in MEAN_FIELDS:
        out[field] = float(np.mean([getattr(r, field) for r in records])) if records else 0.0
    for field in OPTIONAL_MEAN_FIELDS:
        out[field] = _optional_mean(records, field)
    out["pair_dist_mae_r3_le_0p25_rate"] = _optional_rate(records, "pred_object_pair_dist_mae_r3", lambda x: x <= 0.25)
    out["bond_len_mae_r3_le_0p20_rate"] = _optional_rate(records, "pred_object_bond_len_mae_r3", lambda x: x <= 0.20)
    out["z_corr_r3_ge_0p80_rate"] = _optional_rate(records, "pred_object_z_corr_r3", lambda x: x >= 0.80)
    out["nonplanarity_error_r3_le_0p10_rate"] = _optional_rate(records, "pred_object_nonplanarity_error_r3", lambda x: x <= 0.10)
    return out


def atom_count_bin(n: int) -> str:
    if n <= 22:
        return "<=22"
    if n <= 28:
        return "23-28"
    if n <= 34:
        return "29-34"
    return ">=35"


def ring_count_bin(n: int) -> str:
    if n <= 1:
        return "0-1"
    if n == 2:
        return "2"
    return ">=3"


def height_span_bin(v: float) -> str:
    if v < 1.20:
        return "<1.20A"
    if v < 1.75:
        return "1.20-1.75A"
    return ">=1.75A"


def nonplanarity_bin(v: float) -> str:
    if v < 0.20:
        return "<0.20A"
    if v < 0.35:
        return "0.20-0.35A"
    return ">=0.35A"


def _group_by(records: list[GeometryRecord], fn) -> dict[str, dict[str, Optional[float]]]:
    buckets: dict[str, list[GeometryRecord]] = defaultdict(list)
    for record in records:
        buckets[fn(record)].append(record)
    out = {}
    for name, group in buckets.items():
        stats = _mean_metrics(group)
        stats["count"] = len(group)
        out[name] = stats
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    if np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rms_out_of_plane(coords_ang: np.ndarray) -> Optional[float]:
    if len(coords_ang) < 3:
        return None
    centered = coords_ang - coords_ang.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1]
    signed = centered @ normal
    return float(np.sqrt(np.mean(signed ** 2)))


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
    return (float(np.mean(diffs)), coverage) if diffs else (None, coverage)


def _pred_object_heavy_rmsd_ang(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    pred_types: np.ndarray,
    gt_types: np.ndarray,
) -> Optional[float]:
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return None
    row_ind, col_ind, cost = _hungarian_match_numpy(pred_coords, gt_coords)
    if len(row_ind) == 0:
        return None
    pred_match_types = pred_types[row_ind]
    gt_match_types = gt_types[col_ind]
    heavy_mask = gt_match_types != 0
    cost_ang = cost[row_ind, col_ind] * ANGSTROM_SCALE
    if heavy_mask.any():
        return float(np.sqrt(np.mean(cost_ang[heavy_mask] ** 2)))
    return float(np.sqrt(np.mean(cost_ang ** 2)))


def compute_geometry_metrics(
    pred_obj: dict,
    gt_coords: np.ndarray,
    gt_types: np.ndarray,
    gt_mask: np.ndarray,
    gt_edge_adj: np.ndarray,
    img_size: int,
) -> dict[str, Optional[float]]:
    pred_metrics = compute_pred_object_metrics(
        pred_obj,
        gt_coords,
        gt_types,
        gt_mask,
        gt_edge_adj,
        img_size=img_size,
        edge_match_radius_px=XY_MATCH_RADIUS_PX,
    )

    n_pred = int((pred_obj["mask"] > 0.5).sum())
    n_gt = int((gt_mask > 0.5).sum())
    if n_pred == 0 or n_gt == 0:
        return {
            **pred_metrics,
            "pred_object_heavy_rmsd_ang": None,
            "pred_object_pair_dist_mae_r3": None,
            "pred_object_bond_len_mae_r3": None,
            "pred_object_z_corr_r3": None,
            "pred_object_nonplanarity_error_r3": None,
            "matched_gt_node_coverage_r3": 0.0,
            "matched_gt_bond_coverage_r3": 0.0,
            "matched_xy_count_r3": 0,
            "mean_xy_match_px_r3": None,
        }

    pc = pred_obj["coords"][:n_pred].astype(np.float32)
    pt = pred_obj["types"][:n_pred].astype(np.int64)
    gc = gt_coords[:n_gt].astype(np.float32)
    gt = gt_types[:n_gt].astype(np.int64)

    heavy_rmsd_ang = _pred_object_heavy_rmsd_ang(pc, gc, pt, gt)

    row_xy, col_xy, cost_xy = _hungarian_match_numpy(pc[:, :2], gc[:, :2])
    xy_radius_norm = 2.0 * float(XY_MATCH_RADIUS_PX) / float(max(img_size - 1, 1))
    accepted_pairs = [
        (int(p), int(g))
        for p, g in zip(row_xy.tolist(), col_xy.tolist())
        if float(cost_xy[p, g]) <= xy_radius_norm
    ]

    accepted_gt_nodes = {g for _, g in accepted_pairs}
    gt_to_pred = {g: p for p, g in accepted_pairs}
    gt_edges = _infer_edge_set_from_adj(gt_edge_adj, gt_mask)

    matched_gt_node_coverage_r3 = float(len(accepted_gt_nodes) / max(n_gt, 1))
    mean_xy_match_px_r3 = None
    pair_dist_mae = None
    bond_len_mae = None
    matched_gt_bond_coverage_r3 = 0.0
    z_corr = None
    nonplanarity_error = None

    if accepted_pairs:
        pred_idx = [p for p, _ in accepted_pairs]
        gt_idx = [g for _, g in accepted_pairs]
        pred_sub = pc[pred_idx] * ANGSTROM_SCALE
        gt_sub = gc[gt_idx] * ANGSTROM_SCALE

        mean_xy_match_px_r3 = float(np.mean([cost_xy[p, g] for p, g in accepted_pairs]) * float(max(img_size - 1, 1)) * 0.5)
        pair_dist_mae = _pairwise_dist_mae(pred_sub, gt_sub)
        bond_len_mae, matched_gt_bond_coverage_r3 = _bond_len_mae(
            pc * ANGSTROM_SCALE,
            gc * ANGSTROM_SCALE,
            accepted_gt_nodes,
            gt_to_pred,
            gt_edges,
        )
        z_corr = _pearson(pred_sub[:, 2], gt_sub[:, 2])
        pred_nonplanarity = _rms_out_of_plane(pred_sub)
        gt_nonplanarity = _rms_out_of_plane(gt_sub)
        if pred_nonplanarity is not None and gt_nonplanarity is not None:
            nonplanarity_error = float(abs(pred_nonplanarity - gt_nonplanarity))

    return {
        **pred_metrics,
        "pred_object_heavy_rmsd_ang": heavy_rmsd_ang,
        "pred_object_pair_dist_mae_r3": pair_dist_mae,
        "pred_object_bond_len_mae_r3": bond_len_mae,
        "pred_object_z_corr_r3": z_corr,
        "pred_object_nonplanarity_error_r3": nonplanarity_error,
        "matched_gt_node_coverage_r3": matched_gt_node_coverage_r3,
        "matched_gt_bond_coverage_r3": matched_gt_bond_coverage_r3,
        "matched_xy_count_r3": int(len(accepted_pairs)),
        "mean_xy_match_px_r3": mean_xy_match_px_r3,
    }


def _hist(values: list[float], title: str, xlabel: str, output_path: Path) -> None:
    if not values:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.hist(values, bins=24, color="#2563eb", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _scatter(records: list[GeometryRecord], x_field: str, y_field: str, title: str, output_path: Path) -> None:
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
    ax.scatter(xs, ys, s=18, alpha=0.72, color="#dc2626")
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_strat(name: str, strat: dict[str, dict[str, Optional[float]]], output_path: Path) -> None:
    if not strat:
        return
    labels = list(strat.keys())
    z_mae = [float(strat[k]["pred_object_z_mae"] or 0.0) for k in labels]
    pair = [float(strat[k]["pred_object_pair_dist_mae_r3"] or 0.0) for k in labels]
    bond = [float(strat[k]["pred_object_bond_len_mae_r3"] or 0.0) for k in labels]
    nonp = [float(strat[k]["pred_object_nonplanarity_error_r3"] or 0.0) for k in labels]

    x = np.arange(len(labels))
    w = 0.20
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 1.5 * w, z_mae, width=w, label="z_mae")
    ax.bar(x - 0.5 * w, pair, width=w, label="pair_dist_mae_r3")
    ax.bar(x + 0.5 * w, bond, width=w, label="bond_len_mae_r3")
    ax.bar(x + 1.5 * w, nonp, width=w, label="nonplanarity_err_r3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Angstrom")
    ax.set_title(name)
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_plots(records: list[GeometryRecord], output_dir: Path, stratifications: dict[str, dict[str, dict[str, Optional[float]]]]) -> None:
    plots_dir = output_dir / "plots"
    _hist(
        [r.pred_object_z_corr_r3 for r in records if r.pred_object_z_corr_r3 is not None],
        "Matched-R3 Z Correlation",
        "pred_object_z_corr_r3",
        plots_dir / "z_corr_r3_hist.png",
    )
    _hist(
        [r.pred_object_pair_dist_mae_r3 for r in records if r.pred_object_pair_dist_mae_r3 is not None],
        "Matched-R3 Pair Distance MAE",
        "pred_object_pair_dist_mae_r3 (A)",
        plots_dir / "pair_dist_mae_r3_hist.png",
    )
    _hist(
        [r.pred_object_nonplanarity_error_r3 for r in records if r.pred_object_nonplanarity_error_r3 is not None],
        "Matched-R3 Nonplanarity Error",
        "pred_object_nonplanarity_error_r3 (A)",
        plots_dir / "nonplanarity_error_r3_hist.png",
    )
    _scatter(
        records,
        "gt_height_span_ang",
        "pred_object_z_mae",
        "GT Height Span vs Pred Object Z MAE",
        plots_dir / "height_span_vs_z_mae.png",
    )
    _scatter(
        records,
        "gt_nonplanarity_ang",
        "pred_object_nonplanarity_error_r3",
        "GT Nonplanarity vs Nonplanarity Error",
        plots_dir / "gt_nonplanarity_vs_error.png",
    )
    _scatter(
        records,
        "matched_gt_node_coverage_r3",
        "pred_object_pair_dist_mae_r3",
        "Matched Coverage vs Pair Distance MAE",
        plots_dir / "coverage_vs_pair_dist.png",
    )
    for name, strat in stratifications.items():
        _plot_strat(name, strat, plots_dir / f"{name}.png")


def _build_summary(
    checkpoint_path: Path,
    records: list[GeometryRecord],
    stratifications: dict[str, dict[str, dict[str, Optional[float]]]],
) -> tuple[dict, str]:
    mean_metrics = _mean_metrics(records)
    worst_z = sorted(records, key=lambda r: r.pred_object_z_mae, reverse=True)[:10]
    worst_pair = sorted(
        [r for r in records if r.pred_object_pair_dist_mae_r3 is not None],
        key=lambda r: float(r.pred_object_pair_dist_mae_r3),
        reverse=True,
    )[:10]
    worst_nonplanar = sorted(
        [r for r in records if r.pred_object_nonplanarity_error_r3 is not None],
        key=lambda r: float(r.pred_object_nonplanarity_error_r3),
        reverse=True,
    )[:10]

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": "test",
        "num_samples": len(records),
        "xy_match_radius_px": XY_MATCH_RADIUS_PX,
        "mean_metrics": mean_metrics,
        "stratifications": stratifications,
        "worst_z_samples": [asdict(r) for r in worst_z],
        "worst_pair_dist_samples": [asdict(r) for r in worst_pair],
        "worst_nonplanarity_samples": [asdict(r) for r in worst_nonplanar],
    }

    md: list[str] = []
    md.append("# V20 EXP-04 Geometry / 3D Diagnostics")
    md.append("")
    md.append("## 一、实验设置")
    md.append(f"- checkpoint：`{checkpoint_path}`")
    md.append("- split：`test`")
    md.append(f"- 样本数：`{len(records)}`")
    md.append(f"- `matched-r3` 半径：`{XY_MATCH_RADIUS_PX}` 像素")
    md.append("")
    md.append("## 二、几何主指标")
    metric_desc = [
        ("pred_object_heavy_rmsd_ang", "纯预测对象重原子RMSD(Å，3D Hungarian)"),
        ("pred_object_z_mae", "纯预测对象z平均绝对误差(Å)"),
        ("pred_object_pair_dist_mae_r3", "matched-r3 成对距离MAE(Å)"),
        ("pred_object_bond_len_mae_r3", "matched-r3 键长MAE(Å)"),
        ("pred_object_z_corr_r3", "matched-r3 z 相关系数"),
        ("pred_object_nonplanarity_error_r3", "matched-r3 非平面度误差(Å)"),
        ("matched_gt_node_coverage_r3", "matched-r3 GT节点覆盖率"),
        ("matched_gt_bond_coverage_r3", "matched-r3 GT键覆盖率"),
        ("gt_height_span_ang", "GT 高度起伏(Å)"),
        ("gt_nonplanarity_ang", "GT 非平面度(Å)"),
    ]
    for field, desc in metric_desc:
        value = mean_metrics.get(field)
        if value is not None:
            md.append(f"- `{field}`：{desc} = `{value:.4f}`")
    md.append("")
    md.append("## 三、通过率型统计")
    for field in [
        "pair_dist_mae_r3_le_0p25_rate",
        "bond_len_mae_r3_le_0p20_rate",
        "z_corr_r3_ge_0p80_rate",
        "nonplanarity_error_r3_le_0p10_rate",
    ]:
        value = mean_metrics.get(field)
        if value is not None:
            md.append(f"- `{field}` = `{value:.4f}`")
    md.append("")
    md.append("## 四、复杂度分层")
    for name, strat in stratifications.items():
        md.append(f"### {name}")
        md.append("")
        md.append("| 分层 | 样本数 | heavy_rmsd(Å) | z_mae(Å) | pair_dist_mae_r3(Å) | bond_len_mae_r3(Å) | z_corr_r3 | nonplanarity_err_r3(Å) | coverage_r3 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label, stats in strat.items():
            md.append(
                f"| {label} | {int(stats['count'])} | "
                f"{(stats['pred_object_heavy_rmsd_ang'] or 0.0):.4f} | "
                f"{(stats['pred_object_z_mae'] or 0.0):.4f} | "
                f"{(stats['pred_object_pair_dist_mae_r3'] or 0.0):.4f} | "
                f"{(stats['pred_object_bond_len_mae_r3'] or 0.0):.4f} | "
                f"{(stats['pred_object_z_corr_r3'] or 0.0):.4f} | "
                f"{(stats['pred_object_nonplanarity_error_r3'] or 0.0):.4f} | "
                f"{(stats['matched_gt_node_coverage_r3'] or 0.0):.4f} |"
            )
        md.append("")
    md.append("## 五、最差样本")
    md.append("- `worst_z_samples`：按 `pred_object_z_mae` 降序")
    for r in worst_z[:5]:
        md.append(
            f"  - `idx={r.dataset_index}` `cid={r.cid}` `z_mae={r.pred_object_z_mae:.4f}` "
            f"`pair={r.pred_object_pair_dist_mae_r3 if r.pred_object_pair_dist_mae_r3 is not None else -1:.4f}` "
            f"`nonplanarity_err={r.pred_object_nonplanarity_error_r3 if r.pred_object_nonplanarity_error_r3 is not None else -1:.4f}`"
        )
    md.append("- `worst_nonplanarity_samples`：按 `pred_object_nonplanarity_error_r3` 降序")
    for r in worst_nonplanar[:5]:
        md.append(
            f"  - `idx={r.dataset_index}` `cid={r.cid}` "
            f"`gt_nonplanarity={r.gt_nonplanarity_ang:.4f}` "
            f"`err={r.pred_object_nonplanarity_error_r3 if r.pred_object_nonplanarity_error_r3 is not None else -1:.4f}` "
            f"`z_corr={r.pred_object_z_corr_r3 if r.pred_object_z_corr_r3 is not None else -1:.4f}`"
        )
    md.append("")
    md.append("## 六、结论")
    md.append("- 如果 `pair_dist_mae_r3`、`bond_len_mae_r3` 保持较低，同时 `z_corr_r3` 明显为正，说明当前模型的 3D 价值不止体现在单一 `z_mae`。")
    md.append("- 如果复杂度分层中 `height_span` 和 `nonplanarity` 升高时几何误差同步升高，说明当前模型的主要 3D 短板集中在高度起伏更强、非平面度更高的分子。")
    return summary, "\n".join(md) + "\n"


def evaluate(
    model,
    type_head,
    edge_head,
    loader,
    dataset,
    config: dict,
    device: torch.device,
) -> list[GeometryRecord]:
    img_size = int(config["img_size"])
    sample_index = 0
    records: list[GeometryRecord] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="EXP-04 geom eval"):
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

                gt_coords = coords[bi].detach().cpu().numpy()
                gt_types = atom_types[bi].detach().cpu().numpy()
                gt_mask = mask[bi].detach().cpu().numpy()
                gt_edge_adj = edge_labels[bi].detach().cpu().numpy().astype(np.int32)
                n_gt = int((gt_mask > 0.5).sum())
                gt_coords_ang = gt_coords[:n_gt] * ANGSTROM_SCALE
                gt_height_span_ang = float(gt_coords_ang[:, 2].max() - gt_coords_ang[:, 2].min()) if n_gt else 0.0
                gt_nonplanarity_ang = _rms_out_of_plane(gt_coords_ang) or 0.0

                metrics = compute_geometry_metrics(
                    pred_obj,
                    gt_coords,
                    gt_types,
                    gt_mask,
                    gt_edge_adj,
                    img_size=img_size,
                )
                cid = str(dataset.samples[sample_index]["cid"])
                records.append(
                    GeometryRecord(
                        dataset_index=sample_index,
                        cid=cid,
                        gt_atom_count=n_gt,
                        gt_ring_count=int(batch["n_rings"][bi].item()) if "n_rings" in batch else 0,
                        pred_atom_count=int((pred_obj["mask"] > 0.5).sum()),
                        gt_height_span_ang=gt_height_span_ang,
                        gt_nonplanarity_ang=gt_nonplanarity_ang,
                        pred_object_score=float(metrics["pred_object_score"]),
                        pred_object_type_acc=float(metrics["pred_object_type_acc"]),
                        pred_object_edge_f1=float(metrics["pred_object_edge_f1"]),
                        pred_object_edge_f1_robust=float(metrics["pred_object_edge_f1_robust"]),
                        pred_object_match_coverage_robust=float(metrics["pred_object_match_coverage_robust"]),
                        pred_object_heavy_rmsd_ang=metrics["pred_object_heavy_rmsd_ang"],
                        pred_object_z_mae=float(metrics["pred_object_z_mae"]),
                        pred_object_pair_dist_mae_r3=metrics["pred_object_pair_dist_mae_r3"],
                        pred_object_bond_len_mae_r3=metrics["pred_object_bond_len_mae_r3"],
                        pred_object_z_corr_r3=metrics["pred_object_z_corr_r3"],
                        pred_object_nonplanarity_error_r3=metrics["pred_object_nonplanarity_error_r3"],
                        matched_gt_node_coverage_r3=float(metrics["matched_gt_node_coverage_r3"]),
                        matched_gt_bond_coverage_r3=float(metrics["matched_gt_bond_coverage_r3"]),
                        matched_xy_count_r3=int(metrics["matched_xy_count_r3"]),
                        mean_xy_match_px_r3=metrics["mean_xy_match_px_r3"],
                    )
                )
                sample_index += 1
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, type_head, edge_head, config, _state = _load_model(checkpoint_path, device)
    _, _val_loader, test_loader, _num_cids = create_dataloaders(
        data_root=config["data_root"],
        param_key=config["param_key"],
        img_size=config["img_size"],
        min_corrugation=config["min_corrugation"],
        augment_rotation=False,
        require_ring=config.get("require_ring", False),
        batch_size=args.batch_size,
        num_workers=config.get("num_workers", 4),
        max_samples=config.get("max_samples", 0),
        val_size=config.get("val_size", 512),
    )
    test_dataset = test_loader.dataset

    records = evaluate(model, type_head, edge_head, test_loader, test_dataset, config, device)
    stratifications = {
        "atom_count": _group_by(records, lambda r: atom_count_bin(r.gt_atom_count)),
        "ring_count": _group_by(records, lambda r: ring_count_bin(r.gt_ring_count)),
        "height_span": _group_by(records, lambda r: height_span_bin(r.gt_height_span_ang)),
        "nonplanarity": _group_by(records, lambda r: nonplanarity_bin(r.gt_nonplanarity_ang)),
    }

    summary, markdown = _build_summary(checkpoint_path, records, stratifications)
    _save_plots(records, output_dir, stratifications)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "geom_diagnostics_test.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(reports_dir / "geom_diagnostics_test.md", "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(reports_dir / "geom_diagnostics_test_records.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)
    _write_csv(records, reports_dir / "geom_diagnostics_test_records.csv")

    print(json.dumps({
        "output_dir": str(output_dir),
        "report_md": str(reports_dir / "geom_diagnostics_test.md"),
        "report_json": str(reports_dir / "geom_diagnostics_test.json"),
        "num_samples": len(records),
        "pred_object_heavy_rmsd_ang": summary["mean_metrics"]["pred_object_heavy_rmsd_ang"],
        "pred_object_z_mae": summary["mean_metrics"]["pred_object_z_mae"],
        "pred_object_pair_dist_mae_r3": summary["mean_metrics"]["pred_object_pair_dist_mae_r3"],
        "pred_object_bond_len_mae_r3": summary["mean_metrics"]["pred_object_bond_len_mae_r3"],
        "pred_object_z_corr_r3": summary["mean_metrics"]["pred_object_z_corr_r3"],
        "pred_object_nonplanarity_error_r3": summary["mean_metrics"]["pred_object_nonplanarity_error_r3"],
        "matched_gt_node_coverage_r3": summary["mean_metrics"]["matched_gt_node_coverage_r3"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
