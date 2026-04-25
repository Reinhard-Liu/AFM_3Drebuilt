"""
Formal full-test object-level benchmark for the V20 object-joint model.

Outputs:
- full-test aggregate metrics on the test split
- per-sample metrics (json/csv)
- best / median / worst qualitative figures ranked by pred_object_score
- Chinese markdown summary for EXP-01
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.train_v19_object_joint import (
    _edge_metrics_named,
    _object_score,
    _type_metrics_named,
    build_edge_labels,
    build_peak_center_coords,
    build_targets,
    compute_pred_object_metrics,
    extract_predicted_objects,
)
from src.v19_object_joint_review import (
    _load_model,
    _make_sample_figure,
    _per_sample_dense_metrics,
)


@dataclass
class SampleBenchmark:
    dataset_index: int
    cid: str
    pred_object_score: float
    pred_object_3d_score: float
    pred_object_count_mae: float
    pred_object_count_score: float
    pred_object_center_score: float
    pred_object_type_acc: float
    pred_object_macro_f1: float
    pred_object_hetero_f1: float
    pred_object_edge_f1: float
    pred_object_edge_f1_robust: float
    pred_object_match_coverage_robust: float
    pred_object_graph_score: float
    pred_object_heavy_rmsd: float
    pred_object_z_mae: float
    peak_object_score: float
    gt_object_score: float
    atom_center_score_r3: float
    typed_center_score_r3: float
    atom_type_macro_f1_2d: float
    atom_xy_mae: float
    z_map_mae: float
    atom_z_mae_r3: float
    peak_center_type_acc: float
    peak_center_macro_f1: float
    peak_center_hetero_f1: float
    peak_center_edge_f1: float
    peak_center_shift_px: float
    gt_center_type_acc: float
    gt_center_macro_f1: float
    gt_center_hetero_f1: float
    gt_center_edge_f1: float
    figure_path: str = ""


AGGREGATE_FIELDS = [
    "pred_object_score",
    "pred_object_3d_score",
    "pred_object_count_mae",
    "pred_object_count_score",
    "pred_object_center_score",
    "pred_object_type_acc",
    "pred_object_macro_f1",
    "pred_object_hetero_f1",
    "pred_object_edge_f1",
    "pred_object_edge_f1_robust",
    "pred_object_match_coverage_robust",
    "pred_object_graph_score",
    "pred_object_heavy_rmsd",
    "pred_object_z_mae",
    "peak_object_score",
    "gt_object_score",
    "atom_center_score_r3",
    "typed_center_score_r3",
    "atom_type_macro_f1_2d",
    "atom_xy_mae",
    "z_map_mae",
    "atom_z_mae_r3",
    "peak_center_type_acc",
    "peak_center_macro_f1",
    "peak_center_hetero_f1",
    "peak_center_edge_f1",
    "peak_center_shift_px",
    "gt_center_type_acc",
    "gt_center_macro_f1",
    "gt_center_hetero_f1",
    "gt_center_edge_f1",
]


METRIC_DESCRIPTIONS = [
    ("pred_object_score", "纯预测对象闭环对象级总分，越高越好"),
    ("pred_object_3d_score", "纯预测对象3D综合分，越高越好"),
    ("pred_object_count_mae", "纯预测对象原子数平均绝对误差，越低越好"),
    ("pred_object_count_score", "纯预测对象原子数相似度分数，越高越好"),
    ("pred_object_center_score", "纯预测对象proposal中心平均置信度，越高越好"),
    ("pred_object_type_acc", "纯预测对象原子类型准确率，越高越好"),
    ("pred_object_macro_f1", "纯预测对象原子类型宏平均F1，越高越好"),
    ("pred_object_hetero_f1", "纯预测对象杂原子F1，越高越好"),
    ("pred_object_edge_f1", "纯预测对象严格对象级边F1，越高越好"),
    ("pred_object_edge_f1_robust", "纯预测对象距离容忍后的稳健边F1，越高越好"),
    ("pred_object_match_coverage_robust", "稳健匹配覆盖率，越高越好"),
    ("pred_object_graph_score", "纯预测对象图结构综合分，越高越好"),
    ("pred_object_heavy_rmsd", "纯预测对象重原子RMSD，越低越好"),
    ("pred_object_z_mae", "纯预测对象z平均绝对误差，越低越好"),
    ("peak_object_score", "peak-center条件对象级总分，越高越好"),
    ("gt_object_score", "GT-center条件对象级总分，表示上限参考，越高越好"),
    ("atom_center_score_r3", "真实原子中心半径3像素内中心命中分数，越高越好"),
    ("typed_center_score_r3", "真实原子中心半径3像素内位置与类型同时正确的软分数，越高越好"),
    ("atom_type_macro_f1_2d", "稠密2D类型图宏平均F1，越高越好"),
    ("atom_xy_mae", "稠密2D原子图平均绝对误差，越低越好"),
    ("z_map_mae", "稠密z图平均绝对误差，越低越好"),
    ("atom_z_mae_r3", "真实中心附近z平均绝对误差，越低越好"),
    ("peak_center_type_acc", "peak-center条件原子类型准确率，越高越好"),
    ("peak_center_macro_f1", "peak-center条件原子类型宏平均F1，越高越好"),
    ("peak_center_hetero_f1", "peak-center条件杂原子F1，越高越好"),
    ("peak_center_edge_f1", "peak-center条件对象级边F1，越高越好"),
    ("peak_center_shift_px", "peak-center相对真实中心平均偏移像素，越低越好"),
    ("gt_center_type_acc", "GT-center条件原子类型准确率，越高越好"),
    ("gt_center_macro_f1", "GT-center条件原子类型宏平均F1，越高越好"),
    ("gt_center_hetero_f1", "GT-center条件杂原子F1，越高越好"),
    ("gt_center_edge_f1", "GT-center条件对象级边F1，越高越好"),
]


def _mean_records(records: list[SampleBenchmark], fields: list[str]) -> dict[str, float]:
    out = {}
    for field in fields:
        out[field] = float(np.mean([getattr(r, field) for r in records])) if records else 0.0
    return out


def _std_records(records: list[SampleBenchmark], fields: list[str]) -> dict[str, float]:
    out = {}
    for field in fields:
        out[field] = float(np.std([getattr(r, field) for r in records])) if records else 0.0
    return out


def _safe_delta(test_metrics: dict[str, float], val_metrics: dict[str, float], fields: list[str]) -> dict[str, float]:
    out = {}
    for field in fields:
        if field in test_metrics and field in val_metrics:
            out[field] = float(test_metrics[field] - float(val_metrics[field]))
    return out


def _build_summary(
    checkpoint_path: Path,
    state: dict,
    config: dict,
    split: str,
    num_samples: int,
    records: list[SampleBenchmark],
) -> tuple[dict, str]:
    mean_metrics = _mean_records(records, AGGREGATE_FIELDS)
    std_metrics = _std_records(records, AGGREGATE_FIELDS)
    ranked = sorted(records, key=lambda r: r.pred_object_score, reverse=True)
    best = ranked[0]
    median = ranked[len(ranked) // 2]
    worst = ranked[-1]
    val_metrics = state.get("val_metrics", {})
    delta_vs_val = _safe_delta(mean_metrics, val_metrics, AGGREGATE_FIELDS)

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "split": split,
        "sample_count": int(num_samples),
        "config_snapshot": {
            "data_root": config.get("data_root"),
            "param_key": config.get("param_key"),
            "img_size": int(config.get("img_size", 128)),
            "require_ring": bool(config.get("require_ring", False)),
            "max_samples": int(config.get("max_samples", 0)),
            "val_size": int(config.get("val_size", 0)),
        },
        "fulltest_mean_metrics": mean_metrics,
        "fulltest_std_metrics": std_metrics,
        "validation_reference_metrics": val_metrics,
        "fulltest_minus_validation": delta_vs_val,
        "ranking_metric": "pred_object_score",
        "best_sample": asdict(best),
        "median_sample": asdict(median),
        "worst_sample": asdict(worst),
        "top5_samples": [asdict(x) for x in ranked[:5]],
        "bottom5_samples": [asdict(x) for x in ranked[-5:]],
    }

    md: list[str] = []
    md.append("# V20 EXP-01 全测试集对象级正式 Benchmark 报告")
    md.append("")
    md.append("## 一、实验设置")
    md.append(f"- checkpoint：`{checkpoint_path}`")
    md.append(f"- checkpoint 轮次：`{state.get('epoch', -1)}`")
    md.append(f"- split：`{split}`")
    md.append(f"- full-test 样本数：`{num_samples}`")
    md.append(f"- 参数域：`{config.get('param_key', 'K-1')}`")
    md.append(f"- 图像尺寸：`{int(config.get('img_size', 128))}`")
    md.append(f"- 训练时 max_samples：`{int(config.get('max_samples', 0))}`")
    md.append(f"- 训练时 val_size：`{int(config.get('val_size', 0))}`")
    md.append("")
    md.append("## 二、Full-Test 主结果")
    for field, desc in METRIC_DESCRIPTIONS:
        if field in mean_metrics:
            md.append(
                f"- 字段名 `{field}`：{desc}；full-test 均值 = `{mean_metrics[field]:.4f}`；标准差 = `{std_metrics[field]:.4f}`"
            )
    md.append("")
    if val_metrics:
        md.append("## 三、与当前 Validation 口径对照")
        compare_fields = [
            "pred_object_score",
            "pred_object_type_acc",
            "pred_object_macro_f1",
            "pred_object_hetero_f1",
            "pred_object_edge_f1",
            "pred_object_edge_f1_robust",
            "pred_object_count_mae",
            "pred_object_z_mae",
            "peak_object_score",
            "peak_center_type_acc",
            "peak_center_macro_f1",
            "peak_center_hetero_f1",
            "peak_center_edge_f1",
            "peak_center_shift_px",
            "atom_z_mae_r3",
        ]
        for field in compare_fields:
            if field in mean_metrics and field in val_metrics:
                md.append(
                    f"- 字段名 `{field}`：validation=`{float(val_metrics[field]):.4f}`，"
                    f"full-test=`{mean_metrics[field]:.4f}`，"
                    f"差值=`{(mean_metrics[field] - float(val_metrics[field])):+.4f}`"
                )
        md.append("")
    md.append("## 四、代表样本")
    for title, sample in [("最佳样本", best), ("中位样本", median), ("最差样本", worst)]:
        md.append(
            f"- {title}：`dataset_index={sample.dataset_index}`，`cid={sample.cid}`，"
            f"`pred_object_score={sample.pred_object_score:.4f}`，"
            f"`pred_object_type_acc={sample.pred_object_type_acc:.4f}`，"
            f"`pred_object_macro_f1={sample.pred_object_macro_f1:.4f}`，"
            f"`pred_object_edge_f1={sample.pred_object_edge_f1:.4f}`，"
            f"`pred_object_edge_f1_robust={sample.pred_object_edge_f1_robust:.4f}`，"
            f"`pred_object_z_mae={sample.pred_object_z_mae:.4f}`，"
            f"`peak_object_score={sample.peak_object_score:.4f}`，"
            f"`peak_center_type_acc={sample.peak_center_type_acc:.4f}`，"
            f"`peak_center_edge_f1={sample.peak_center_edge_f1:.4f}`"
        )
    md.append("")
    md.append("## 五、核心判断")
    md.append("- 这个 full-test 报告以 `pred_object_score` 作为样本排序主指标，因为它最贴近真实闭环推理条件。")
    md.append("- `peak_*` 指标表示以 peak-center 为锚点的上限式对象级能力，`pred_object_*` 指标表示纯预测对象闭环能力。")
    md.append("- 如果 `pred_object_edge_f1_robust` 明显高于 `pred_object_edge_f1`，说明局部邻接恢复能力强于严格对象对应表现。")
    md.append("- 当前最重要的 gap 仍是 `peak/gt` 到 `pred-object` 的迁移损失，而不是中心、z 或局部邻接完全失效。")
    md.append("")
    md.append("## 六、字段说明")
    for field, desc in METRIC_DESCRIPTIONS:
        md.append(f"- 字段名 `{field}`：{desc}")
    md.append("")
    return summary, "\n".join(md)


def _write_csv(records: list[SampleBenchmark], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()) if records else [])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _evaluate_split(
    model,
    type_head,
    edge_head,
    loader,
    config: dict,
    device: torch.device,
    output_dir: Path,
) -> list[SampleBenchmark]:
    records: list[SampleBenchmark] = []
    img_size = int(config["img_size"])
    sample_index = 0
    dataset = loader.dataset

    model.eval()
    type_head.eval()
    edge_head.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="EXP-01 full-test", leave=False):
            afm = batch["afm_stack"].to(device)
            coords = batch["coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)
            targets = build_targets(batch, img_size, device)
            edge_labels = build_edge_labels(batch, device)

            pred, features = model.forward_with_features(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            center_map_01 = torch.sigmoid(features["center_logits"]).clamp(0.0, 1.0)

            _, gt_type_logits = type_head.compute_loss(
                coords,
                features["enc1"],
                afm,
                atom_types,
                mask,
                class_weight=None,
                center_map=center_map_01,
            )
            _, gt_edge_logits = edge_head.compute_loss(coords, features["enc1"], afm, mask, edge_labels)

            peak_coords, peak_shift = build_peak_center_coords(
                center_map_01,
                pred_01[:, 12:13],
                coords,
                mask,
                img_size,
                alpha=1.0,
                search_radius=int(config.get("center_search_radius", 3)),
            )
            _, peak_type_logits = type_head.compute_loss(
                peak_coords,
                features["enc1"],
                afm,
                atom_types,
                mask,
                class_weight=None,
                center_map=center_map_01,
            )
            _, peak_edge_logits = edge_head.compute_loss(peak_coords, features["enc1"], afm, mask, edge_labels)

            for bi in range(afm.shape[0]):
                dense = _per_sample_dense_metrics(
                    pred_01[bi],
                    center_map_01[bi],
                    targets[bi],
                    coords[bi],
                    atom_types[bi],
                    mask[bi],
                    img_size,
                )

                sample_mask = mask[bi : bi + 1]
                sample_edge_labels = edge_labels[bi : bi + 1]
                sample_coords = coords[bi : bi + 1]
                sample_types = atom_types[bi : bi + 1]
                sample_afm = afm[bi : bi + 1]
                sample_feat = features["enc1"][bi : bi + 1]
                sample_pred = pred_01[bi : bi + 1]
                sample_center_map = center_map_01[bi : bi + 1]

                gt_type = _type_metrics_named(
                    gt_type_logits[bi : bi + 1].argmax(dim=-1),
                    sample_types,
                    sample_mask,
                    prefix="gt_center",
                )
                gt_edge = _edge_metrics_named(
                    gt_edge_logits[bi : bi + 1],
                    sample_edge_labels,
                    sample_mask,
                    prefix="gt_center",
                )
                peak_type = _type_metrics_named(
                    peak_type_logits[bi : bi + 1].argmax(dim=-1),
                    sample_types,
                    sample_mask,
                    prefix="peak_center",
                )
                peak_edge = _edge_metrics_named(
                    peak_edge_logits[bi : bi + 1],
                    sample_edge_labels,
                    sample_mask,
                    prefix="peak_center",
                )

                gt_object_score = _object_score(
                    gt_type["gt_center_type_acc"],
                    gt_type["gt_center_macro_f1"],
                    gt_type["gt_center_hetero_f1"],
                    gt_edge["gt_center_edge_f1"],
                    dense["atom_center_score_r3"],
                    dense["atom_z_mae_r3"],
                    0.0,
                )
                peak_object_score = _object_score(
                    peak_type["peak_center_type_acc"],
                    peak_type["peak_center_macro_f1"],
                    peak_type["peak_center_hetero_f1"],
                    peak_edge["peak_center_edge_f1"],
                    dense["atom_center_score_r3"],
                    dense["atom_z_mae_r3"],
                    float(peak_shift[bi]) if isinstance(peak_shift, np.ndarray) else float(peak_shift),
                )

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
                pred_metrics = compute_pred_object_metrics(
                    pred_obj,
                    sample_coords[0].detach().cpu().numpy(),
                    sample_types[0].detach().cpu().numpy(),
                    sample_mask[0].detach().cpu().numpy(),
                    sample_edge_labels[0].detach().cpu().numpy().astype(np.int32),
                    img_size=img_size,
                    edge_match_radius_px=float(config.get("pred_object_edge_match_radius_px", 3.0)),
                )

                cid = dataset.samples[sample_index]["cid"]
                records.append(
                    SampleBenchmark(
                        dataset_index=sample_index,
                        cid=str(cid),
                        pred_object_score=float(pred_metrics["pred_object_score"]),
                        pred_object_3d_score=float(pred_metrics["pred_object_3d_score"]),
                        pred_object_count_mae=float(pred_metrics["pred_object_count_mae"]),
                        pred_object_count_score=float(pred_metrics["pred_object_count_score"]),
                        pred_object_center_score=float(pred_metrics["pred_object_center_score"]),
                        pred_object_type_acc=float(pred_metrics["pred_object_type_acc"]),
                        pred_object_macro_f1=float(pred_metrics["pred_object_macro_f1"]),
                        pred_object_hetero_f1=float(pred_metrics["pred_object_hetero_f1"]),
                        pred_object_edge_f1=float(pred_metrics["pred_object_edge_f1"]),
                        pred_object_edge_f1_robust=float(pred_metrics["pred_object_edge_f1_robust"]),
                        pred_object_match_coverage_robust=float(pred_metrics["pred_object_match_coverage_robust"]),
                        pred_object_graph_score=float(pred_metrics["pred_object_graph_score"]),
                        pred_object_heavy_rmsd=float(pred_metrics["pred_object_heavy_rmsd"]),
                        pred_object_z_mae=float(pred_metrics["pred_object_z_mae"]),
                        peak_object_score=float(peak_object_score),
                        gt_object_score=float(gt_object_score),
                        atom_center_score_r3=float(dense["atom_center_score_r3"]),
                        typed_center_score_r3=float(dense["typed_center_score_r3"]),
                        atom_type_macro_f1_2d=float(dense["atom_type_macro_f1_2d"]),
                        atom_xy_mae=float(dense["atom_xy_mae"]),
                        z_map_mae=float(dense["z_map_mae"]),
                        atom_z_mae_r3=float(dense["atom_z_mae_r3"]),
                        peak_center_type_acc=float(peak_type["peak_center_type_acc"]),
                        peak_center_macro_f1=float(peak_type["peak_center_macro_f1"]),
                        peak_center_hetero_f1=float(peak_type["peak_center_hetero_f1"]),
                        peak_center_edge_f1=float(peak_edge["peak_center_edge_f1"]),
                        peak_center_shift_px=float(peak_shift[bi]) if isinstance(peak_shift, np.ndarray) else float(peak_shift),
                        gt_center_type_acc=float(gt_type["gt_center_type_acc"]),
                        gt_center_macro_f1=float(gt_type["gt_center_macro_f1"]),
                        gt_center_hetero_f1=float(gt_type["gt_center_hetero_f1"]),
                        gt_center_edge_f1=float(gt_edge["gt_center_edge_f1"]),
                    )
                )
                sample_index += 1

    ranked = sorted(records, key=lambda r: r.pred_object_score, reverse=True)
    selected = {
        "best": ranked[0],
        "median": ranked[len(ranked) // 2],
        "worst": ranked[-1],
    }
    idx_to_name = {sample.dataset_index: name for name, sample in selected.items()}

    sample_index = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="EXP-01 figs", leave=False):
            afm = batch["afm_stack"].to(device)
            coords = batch["coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)
            targets = build_targets(batch, img_size, device)
            edge_labels = build_edge_labels(batch, device)

            pred, features = model.forward_with_features(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            center_map_01 = torch.sigmoid(features["center_logits"]).clamp(0.0, 1.0)

            for bi in range(afm.shape[0]):
                if sample_index in idx_to_name:
                    rank_name = idx_to_name[sample_index]
                    record = next(x for x in records if x.dataset_index == sample_index)
                    sample_mask = mask[bi : bi + 1]
                    sample_edge_labels = edge_labels[bi : bi + 1]
                    sample_coords = coords[bi : bi + 1]
                    sample_types = atom_types[bi : bi + 1]
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
                    gt_edge_adj = sample_edge_labels[0].detach().cpu().numpy().astype(np.int32)
                    title = (
                        f"{rank_name.upper()} | idx={record.dataset_index} | cid={record.cid} | "
                        f"pred_score={record.pred_object_score:.3f} | pred_type={record.pred_object_type_acc:.3f} | "
                        f"pred_macro={record.pred_object_macro_f1:.3f} | pred_edge={record.pred_object_edge_f1:.3f} | "
                        f"robust_edge={record.pred_object_edge_f1_robust:.3f} | z_mae={record.pred_object_z_mae:.3f}"
                    )
                    fig_path = output_dir / "samples" / f"{rank_name}_sample_{record.dataset_index:04d}.png"
                    _make_sample_figure(
                        afm[bi].detach().cpu().numpy(),
                        targets[bi].detach().cpu().numpy(),
                        pred_01[bi].detach().cpu().numpy(),
                        center_map_01[bi].detach().cpu().numpy(),
                        sample_coords[0].detach().cpu().numpy(),
                        sample_types[0].detach().cpu().numpy(),
                        sample_mask[0].detach().cpu().numpy(),
                        gt_edge_adj,
                        pred_obj["coords"],
                        pred_obj["types"],
                        pred_obj["mask"],
                        pred_obj["edge_adj"],
                        title,
                        fig_path,
                    )
                    record.figure_path = str(fig_path)
                sample_index += 1

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, type_head, edge_head, config, state = _load_model(checkpoint_path, device)

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
    del train_loader
    loader = val_loader if args.split == "val" else test_loader

    records = _evaluate_split(model, type_head, edge_head, loader, config, device, output_dir)
    summary, md = _build_summary(checkpoint_path, state, config, args.split, len(loader.dataset), records)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / f"fulltest_object_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(reports_dir / f"fulltest_object_{args.split}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(reports_dir / f"fulltest_object_{args.split}_samples.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)
    _write_csv(records, reports_dir / f"fulltest_object_{args.split}_samples.csv")

    print(json.dumps({
        "output_dir": str(output_dir),
        "report_json": str(reports_dir / f"fulltest_object_{args.split}.json"),
        "report_md": str(reports_dir / f"fulltest_object_{args.split}.md"),
        "sample_count": len(loader.dataset),
        "pred_object_score": summary["fulltest_mean_metrics"]["pred_object_score"],
        "pred_object_type_acc": summary["fulltest_mean_metrics"]["pred_object_type_acc"],
        "pred_object_macro_f1": summary["fulltest_mean_metrics"]["pred_object_macro_f1"],
        "pred_object_edge_f1": summary["fulltest_mean_metrics"]["pred_object_edge_f1"],
        "pred_object_edge_f1_robust": summary["fulltest_mean_metrics"]["pred_object_edge_f1_robust"],
        "pred_object_z_mae": summary["fulltest_mean_metrics"]["pred_object_z_mae"],
        "peak_object_score": summary["fulltest_mean_metrics"]["peak_object_score"],
        "peak_center_type_acc": summary["fulltest_mean_metrics"]["peak_center_type_acc"],
        "peak_center_macro_f1": summary["fulltest_mean_metrics"]["peak_center_macro_f1"],
        "peak_center_edge_f1": summary["fulltest_mean_metrics"]["peak_center_edge_f1"],
        "atom_z_mae_r3": summary["fulltest_mean_metrics"]["atom_z_mae_r3"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
