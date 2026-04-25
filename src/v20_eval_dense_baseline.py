"""
EXP-05:
Evaluate the V19 Stage-1 dense 2D baseline under the V20 object-level protocol.

Current intended use:
- decode stage1 dense maps into object hypotheses
- evaluate on the same full-test split used by V20 EXP-01
- report side-by-side deltas vs the current V20 object-joint model

Important caveat:
- the currently available checkpoint is a debug baseline trained on 128 train / 16 val / 1 epoch
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
from src.models.baselines import UNetGenerator
from src.train_v19_object_joint import (
    _bilinear_sample_np,
    _refine_peak_xy,
    build_edge_labels,
    compute_pred_object_metrics,
    decode_object_peaks,
)
from src.train_v19_stage1 import build_targets, compute_v19_stage1_metrics
from src.utils.mol2d import COVALENT_RADII, V19_2D_TARGET_CHANNELS
from src.v19_object_joint_review import _make_sample_figure


MAX_ATOMS = 85


@dataclass
class DenseBaselineRecord:
    dataset_index: int
    cid: str
    gt_atom_count: int
    pred_atom_count: int
    atom_xy_mae: float
    bond_map_mae: float
    type_map_mae: float
    atom_center_score_r3: float
    typed_center_score_r3: float
    type_top1_local_acc_r3: float
    atom_type_macro_f1_2d: float
    ch_collapse_rate_2d: float
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
    figure_path: str = ""


MEAN_FIELDS = [
    "atom_xy_mae",
    "bond_map_mae",
    "type_map_mae",
    "atom_center_score_r3",
    "typed_center_score_r3",
    "type_top1_local_acc_r3",
    "atom_type_macro_f1_2d",
    "ch_collapse_rate_2d",
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
]


def _write_csv(records: list[DenseBaselineRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fieldnames = list(asdict(records[0]).keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def _mean_records(records: list[DenseBaselineRecord]) -> dict[str, float]:
    return {
        field: float(np.mean([getattr(r, field) for r in records])) if records else 0.0
        for field in MEAN_FIELDS
    }


def _std_records(records: list[DenseBaselineRecord]) -> dict[str, float]:
    return {
        field: float(np.std([getattr(r, field) for r in records])) if records else 0.0
        for field in MEAN_FIELDS
    }


def _resolve_stage1_config(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config.get("data_root") == "auto":
        config["data_root"] = str(ROOT / "dataverse_files" / "SUBMIT_QUAM-AFM" / "QUAM")
    if config.get("save_dir") == "auto":
        config["save_dir"] = str(ROOT / "experiments" / "v19_stage1_2d_debug" / "checkpoints")
    return config


def _load_dense_baseline(checkpoint_path: Path, config_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state.get("config", {}) or _resolve_stage1_config(config_path)
    generator = UNetGenerator(in_channels=10, out_channels=V19_2D_TARGET_CHANNELS).to(device)
    generator.load_state_dict(state["generator"], strict=False)
    generator.eval()
    return generator, config, state


def _line_values(arr: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1))
    xs = np.linspace(x0, x1, steps + 1, dtype=np.float32)
    ys = np.linspace(y0, y1, steps + 1, dtype=np.float32)
    vals = np.asarray([_bilinear_sample_np(arr, float(y), float(x)) for x, y in zip(xs, ys)], dtype=np.float32)
    if len(vals) > 6:
        trim = max(1, int(round(0.12 * len(vals))))
        vals = vals[trim:-trim]
    return vals


def _decode_dense_objects(
    pred_01: torch.Tensor,
    img_size: int,
    peak_threshold: float,
    min_distance_px: int,
    max_objects: int,
    bond_line_mean_threshold: float,
    bond_line_peak_threshold: float,
    bond_length_scale: float,
) -> dict:
    pred_np = pred_01.detach().cpu().numpy()
    atom_map = pred_np[0]
    bond_map = pred_np[1]
    type_maps = pred_np[2:12]

    center_map = torch.from_numpy(atom_map[None, None]).float()
    atom_map_t = torch.from_numpy(atom_map[None, None]).float()
    peaks_batch, _ = decode_object_peaks(
        center_map,
        count_logits=None,
        atom_map_01=atom_map_t,
        peak_threshold=peak_threshold,
        min_distance_px=min_distance_px,
        max_objects=max_objects,
    )
    peaks = peaks_batch[0]

    coords = np.zeros((MAX_ATOMS, 3), dtype=np.float32)
    atom_types = np.zeros((MAX_ATOMS,), dtype=np.int64)
    mask = np.zeros((MAX_ATOMS,), dtype=np.float32)
    edge_adj = np.zeros((MAX_ATOMS, MAX_ATOMS), dtype=np.int32)
    refined_xy = []
    center_scores = []

    n = min(len(peaks), MAX_ATOMS)
    for i, (y, x, score) in enumerate(peaks[:n]):
        y_ref, x_ref = _refine_peak_xy(atom_map, y, x, refine_radius_px=2)
        x_norm = (float(x_ref) / float(max(img_size - 1, 1))) * 2.0 - 1.0
        y_norm = (float(y_ref) / float(max(img_size - 1, 1))) * 2.0 - 1.0
        type_scores = np.asarray([_bilinear_sample_np(type_maps[t], y_ref, x_ref) for t in range(10)], dtype=np.float32)
        pred_type = int(np.argmax(type_scores))

        coords[i] = np.asarray([x_norm, y_norm, 0.0], dtype=np.float32)
        atom_types[i] = pred_type
        mask[i] = 1.0
        refined_xy.append([x_ref, y_ref])
        center_scores.append(float(score))

    if n > 1:
        coords_ang = coords[:n] * 12.0
        for i in range(n):
            ti = int(atom_types[i])
            if not 0 <= ti < len(COVALENT_RADII):
                continue
            xi = float(refined_xy[i][0])
            yi = float(refined_xy[i][1])
            for j in range(i + 1, n):
                tj = int(atom_types[j])
                if not 0 <= tj < len(COVALENT_RADII):
                    continue
                dist_xy_ang = float(np.linalg.norm(coords_ang[i, :2] - coords_ang[j, :2]))
                if dist_xy_ang < 0.45:
                    continue
                cutoff = float(bond_length_scale * (COVALENT_RADII[ti] + COVALENT_RADII[tj]))
                if dist_xy_ang > cutoff:
                    continue
                xj = float(refined_xy[j][0])
                yj = float(refined_xy[j][1])
                vals = _line_values(bond_map, xi, yi, xj, yj)
                if vals.size == 0:
                    continue
                if float(vals.mean()) >= bond_line_mean_threshold or float(vals.max()) >= bond_line_peak_threshold:
                    edge_adj[i, j] = 1
                    edge_adj[j, i] = 1

    return {
        "coords": coords,
        "types": atom_types,
        "mask": mask,
        "edge_adj": edge_adj,
        "peaks": peaks[:n],
        "refined_xy_px": np.asarray(refined_xy, dtype=np.float32) if refined_xy else np.zeros((0, 2), dtype=np.float32),
        "pred_count": int(n),
        "mean_center_score": float(np.mean(center_scores)) if center_scores else 0.0,
    }


def _slice_batch(batch: dict, bi: int) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[bi : bi + 1]
        else:
            out[k] = v
    return out


def _to_joint_map(pred_01_12: np.ndarray) -> np.ndarray:
    out = np.zeros((13, pred_01_12.shape[1], pred_01_12.shape[2]), dtype=np.float32)
    out[:12] = pred_01_12
    return out


def _make_qualitative_figures(
    generator,
    loader,
    dataset,
    config: dict,
    device: torch.device,
    selected_records: list[DenseBaselineRecord],
    output_dir: Path,
    peak_threshold: float,
    min_distance_px: int,
    max_objects: int,
    bond_line_mean_threshold: float,
    bond_line_peak_threshold: float,
    bond_length_scale: float,
) -> None:
    wanted = {r.dataset_index: r for r in selected_records}
    if not wanted:
        return

    sample_index = 0
    img_size = int(config["img_size"])
    with torch.no_grad():
        for batch in tqdm(loader, desc="Dense figs", leave=False):
            afm = batch["afm_stack"].to(device)
            edge_labels = build_edge_labels(batch, device)
            pred = generator(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            target_12 = build_targets(batch, img_size, device).detach().cpu().numpy()

            for bi in range(afm.shape[0]):
                if sample_index not in wanted:
                    sample_index += 1
                    continue
                record = wanted[sample_index]
                pred_obj = _decode_dense_objects(
                    pred_01[bi],
                    img_size=img_size,
                    peak_threshold=peak_threshold,
                    min_distance_px=min_distance_px,
                    max_objects=max_objects,
                    bond_line_mean_threshold=bond_line_mean_threshold,
                    bond_line_peak_threshold=bond_line_peak_threshold,
                    bond_length_scale=bond_length_scale,
                )
                gt_coords = batch["coords"][bi].detach().cpu().numpy()
                gt_types = batch["atom_types"][bi].detach().cpu().numpy()
                gt_mask = batch["atom_mask"][bi].detach().cpu().numpy()
                gt_edge_adj = edge_labels[bi].detach().cpu().numpy().astype(np.int32)

                fig_path = output_dir / "samples" / f"dense_baseline_sample_{record.dataset_index:04d}.png"
                _make_sample_figure(
                    afm[bi].detach().cpu().numpy(),
                    _to_joint_map(target_12[bi]),
                    _to_joint_map(pred_01[bi].detach().cpu().numpy()),
                    pred_01[bi, 0:1].detach().cpu().numpy(),
                    gt_coords,
                    gt_types,
                    gt_mask,
                    gt_edge_adj,
                    pred_obj["coords"],
                    pred_obj["types"],
                    pred_obj["mask"],
                    pred_obj["edge_adj"],
                    (
                        f"Dense Baseline | idx={record.dataset_index} | cid={record.cid} | "
                        f"score={record.pred_object_score:.3f} | type={record.pred_object_type_acc:.3f} | "
                        f"edge={record.pred_object_edge_f1:.3f}"
                    ),
                    fig_path,
                )
                record.figure_path = str(fig_path)
                sample_index += 1


def evaluate(
    generator,
    loader,
    dataset,
    config: dict,
    device: torch.device,
    peak_threshold: float,
    min_distance_px: int,
    max_objects: int,
    bond_line_mean_threshold: float,
    bond_line_peak_threshold: float,
    bond_length_scale: float,
) -> list[DenseBaselineRecord]:
    img_size = int(config["img_size"])
    sample_index = 0
    records: list[DenseBaselineRecord] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Dense baseline eval"):
            afm = batch["afm_stack"].to(device)
            edge_labels = build_edge_labels(batch, device)
            pred = generator(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)

            for bi in range(afm.shape[0]):
                sample_batch = _slice_batch(batch, bi)
                dense_metrics = compute_v19_stage1_metrics(pred_01[bi : bi + 1], sample_batch, img_size)
                pred_obj = _decode_dense_objects(
                    pred_01[bi],
                    img_size=img_size,
                    peak_threshold=peak_threshold,
                    min_distance_px=min_distance_px,
                    max_objects=max_objects,
                    bond_line_mean_threshold=bond_line_mean_threshold,
                    bond_line_peak_threshold=bond_line_peak_threshold,
                    bond_length_scale=bond_length_scale,
                )

                gt_coords = batch["coords"][bi].detach().cpu().numpy()
                gt_types = batch["atom_types"][bi].detach().cpu().numpy()
                gt_mask = batch["atom_mask"][bi].detach().cpu().numpy()
                gt_edge_adj = edge_labels[bi].detach().cpu().numpy().astype(np.int32)
                pred_metrics = compute_pred_object_metrics(pred_obj, gt_coords, gt_types, gt_mask, gt_edge_adj, img_size=img_size)

                cid = str(dataset.samples[sample_index]["cid"])
                records.append(
                    DenseBaselineRecord(
                        dataset_index=sample_index,
                        cid=cid,
                        gt_atom_count=int((gt_mask > 0.5).sum()),
                        pred_atom_count=int((pred_obj["mask"] > 0.5).sum()),
                        atom_xy_mae=float(dense_metrics["atom_xy_mae"]),
                        bond_map_mae=float(dense_metrics["bond_map_mae"]),
                        type_map_mae=float(dense_metrics["type_map_mae"]),
                        atom_center_score_r3=float(dense_metrics["atom_center_score_r3"]),
                        typed_center_score_r3=float(dense_metrics["typed_center_score_r3"]),
                        type_top1_local_acc_r3=float(dense_metrics["type_top1_local_acc_r3"]),
                        atom_type_macro_f1_2d=float(dense_metrics["atom_type_macro_f1_2d"]),
                        ch_collapse_rate_2d=float(dense_metrics["ch_collapse_rate_2d"]),
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
                    )
                )
                sample_index += 1
    return records


def _build_summary(
    checkpoint_path: Path,
    config_path: Path,
    train_config: dict,
    eval_config: dict,
    records: list[DenseBaselineRecord],
    reference_json: Path | None,
) -> tuple[dict, str]:
    mean_metrics = _mean_records(records)
    std_metrics = _std_records(records)
    ranked = sorted(records, key=lambda r: r.pred_object_score, reverse=True)
    best = ranked[0]
    median = ranked[len(ranked) // 2]
    worst = ranked[-1]

    reference_metrics = {}
    delta_vs_v20 = {}
    if reference_json is not None and reference_json.exists():
        ref = json.loads(reference_json.read_text())
        reference_metrics = ref.get("fulltest_mean_metrics", {})
        for field in [
            "pred_object_score",
            "pred_object_type_acc",
            "pred_object_macro_f1",
            "pred_object_hetero_f1",
            "pred_object_edge_f1",
            "pred_object_edge_f1_robust",
            "pred_object_count_mae",
            "pred_object_z_mae",
        ]:
            if field in reference_metrics and field in mean_metrics:
                delta_vs_v20[field] = float(mean_metrics[field] - float(reference_metrics[field]))

    summary = {
        "baseline_checkpoint": str(checkpoint_path),
        "config_path": str(config_path),
        "train_config_snapshot": train_config,
        "eval_config_snapshot": eval_config,
        "num_samples": len(records),
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "v20_reference_metrics": reference_metrics,
        "dense_minus_v20": delta_vs_v20,
        "best_sample": asdict(best),
        "median_sample": asdict(median),
        "worst_sample": asdict(worst),
        "top5_samples": [asdict(r) for r in ranked[:5]],
        "bottom5_samples": [asdict(r) for r in ranked[-5:]],
    }

    md: list[str] = []
    md.append("# V20 EXP-05 Dense Structured-Map Baseline")
    md.append("")
    md.append("## 一、实验设置")
    md.append(f"- baseline checkpoint：`{checkpoint_path}`")
    md.append(f"- baseline config：`{config_path}`")
    md.append(f"- 当前 baseline 训练规模：`train={train_config.get('max_samples', 'NA')}` `val={train_config.get('val_size', 'NA')}` `epochs={train_config.get('epochs', 'NA')}`")
    md.append(f"- 当前评估样本数：`{len(records)}`")
    md.append("- 说明：这是当前仓库中唯一现成的 stage1 dense baseline checkpoint，因此结果属于 `debug baseline`，不是最终公平训练版。")
    md.append("")
    md.append("## 二、Dense Baseline Full-Test 主结果")
    for field in MEAN_FIELDS:
        md.append(f"- `{field}`：均值 = `{mean_metrics[field]:.4f}`；标准差 = `{std_metrics[field]:.4f}`")
    md.append("")
    if reference_metrics:
        md.append("## 三、与 V20 主模型对比")
        md.append("| 字段名 | Dense Baseline | V20 Full-Test | Dense - V20 |")
        md.append("|---|---:|---:|---:|")
        for field in [
            "pred_object_score",
            "pred_object_type_acc",
            "pred_object_macro_f1",
            "pred_object_hetero_f1",
            "pred_object_edge_f1",
            "pred_object_edge_f1_robust",
            "pred_object_count_mae",
            "pred_object_z_mae",
        ]:
            if field in reference_metrics and field in mean_metrics:
                md.append(
                    f"| {field} | {mean_metrics[field]:.4f} | {float(reference_metrics[field]):.4f} | {float(delta_vs_v20[field]):+.4f} |"
                )
        md.append("")
    md.append("## 四、代表样本")
    for tag, record in [("best", best), ("median", median), ("worst", worst)]:
        md.append(
            f"- `{tag}`：`idx={record.dataset_index}` `cid={record.cid}` "
            f"`score={record.pred_object_score:.4f}` `type_acc={record.pred_object_type_acc:.4f}` "
            f"`edge_f1={record.pred_object_edge_f1:.4f}` `count_mae={record.pred_object_count_mae:.4f}`"
        )
    md.append("")
    md.append("## 五、结论")
    md.append("- 该 baseline 的优势主要体现在把 AFM 转成可视化的稠密 2D 图；但它没有对象级 z 分支，也没有 center-conditioned type/edge closure。")
    md.append("- 因此如果它在 `pred_object_score`、`pred_object_type_acc`、`pred_object_edge_f1`、`pred_object_z_mae` 上明显落后于 V20，这更说明当前对象级路线的价值。")
    md.append("- 但因为当前 baseline 只训练了 debug 规模，论文正文里应把它明确写成 `preliminary dense baseline`，后续最好再补一个正式训练版。")
    return summary, "\n".join(md) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/micro/experiments/v19_stage1_2d_debug/checkpoints/best_v19_stage1.pt")
    parser.add_argument("--config_path", default="/root/autodl-tmp/micro/config_v19_stage1_2d_debug.json")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/micro/experiments/v20_dense_baseline_exp05")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--eval_val_size", type=int, default=512)
    parser.add_argument("--peak_threshold", type=float, default=0.45)
    parser.add_argument("--min_distance_px", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=64)
    parser.add_argument("--bond_line_mean_threshold", type=float, default=0.18)
    parser.add_argument("--bond_line_peak_threshold", type=float, default=0.35)
    parser.add_argument("--bond_length_scale", type=float, default=1.35)
    parser.add_argument("--v20_reference_json", default="/root/autodl-tmp/micro/experiments/v20_object_joint_medium10_exp01_fulltest/reports/fulltest_object_test.json")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config_path)
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator, train_config, _state = _load_dense_baseline(checkpoint_path, config_path, device)
    eval_config = dict(train_config)
    eval_config["batch_size"] = int(args.batch_size)
    eval_config["max_samples"] = int(args.eval_max_samples)
    eval_config["val_size"] = int(args.eval_val_size)

    _, _val_loader, test_loader, _num_cids = create_dataloaders(
        data_root=eval_config["data_root"],
        param_key=eval_config.get("param_key", "K-1"),
        img_size=eval_config["img_size"],
        min_corrugation=eval_config.get("min_corrugation", 0.0),
        augment_rotation=False,
        require_ring=eval_config.get("require_ring", False),
        batch_size=eval_config["batch_size"],
        num_workers=eval_config.get("num_workers", 4),
        max_samples=eval_config["max_samples"],
        val_size=eval_config["val_size"],
    )
    test_dataset = test_loader.dataset

    records = evaluate(
        generator,
        test_loader,
        test_dataset,
        eval_config,
        device,
        peak_threshold=args.peak_threshold,
        min_distance_px=args.min_distance_px,
        max_objects=args.max_objects,
        bond_line_mean_threshold=args.bond_line_mean_threshold,
        bond_line_peak_threshold=args.bond_line_peak_threshold,
        bond_length_scale=args.bond_length_scale,
    )

    ranked = sorted(records, key=lambda r: r.pred_object_score, reverse=True)
    selected = [ranked[0], ranked[len(ranked) // 2], ranked[-1]] if ranked else []
    _make_qualitative_figures(
        generator,
        test_loader,
        test_dataset,
        eval_config,
        device,
        selected,
        output_dir,
        peak_threshold=args.peak_threshold,
        min_distance_px=args.min_distance_px,
        max_objects=args.max_objects,
        bond_line_mean_threshold=args.bond_line_mean_threshold,
        bond_line_peak_threshold=args.bond_line_peak_threshold,
        bond_length_scale=args.bond_length_scale,
    )

    summary, markdown = _build_summary(
        checkpoint_path,
        config_path,
        train_config=train_config,
        eval_config=eval_config,
        records=records,
        reference_json=Path(args.v20_reference_json) if args.v20_reference_json else None,
    )

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "dense_baseline_fulltest.md", "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(reports_dir / "dense_baseline_fulltest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(reports_dir / "dense_baseline_fulltest_records.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)
    _write_csv(records, reports_dir / "dense_baseline_fulltest_records.csv")

    print(json.dumps({
        "output_dir": str(output_dir),
        "report_md": str(reports_dir / "dense_baseline_fulltest.md"),
        "report_json": str(reports_dir / "dense_baseline_fulltest.json"),
        "num_samples": len(records),
        "pred_object_score": summary["mean_metrics"]["pred_object_score"],
        "pred_object_type_acc": summary["mean_metrics"]["pred_object_type_acc"],
        "pred_object_edge_f1": summary["mean_metrics"]["pred_object_edge_f1"],
        "pred_object_edge_f1_robust": summary["mean_metrics"]["pred_object_edge_f1_robust"],
        "pred_object_z_mae": summary["mean_metrics"]["pred_object_z_mae"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
