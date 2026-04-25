"""
SUP-02:
Evaluate the graph reconstruction baseline under the V20 object-level protocol.
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
from src.train_v19_object_joint import build_edge_labels, build_targets, compute_pred_object_metrics, extract_predicted_objects
from src.v19_object_joint_review import _make_sample_figure
from src.v20_train_graph_baseline import load_graph_baseline_checkpoint, resolve_config


@dataclass
class GraphBaselineRecord:
    dataset_index: int
    cid: str
    gt_atom_count: int
    pred_atom_count: int
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


def _write_csv(records: list[GraphBaselineRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fieldnames = list(asdict(records[0]).keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _mean_records(records: list[GraphBaselineRecord]) -> dict[str, float]:
    return {
        field: float(np.mean([getattr(r, field) for r in records])) if records else 0.0
        for field in MEAN_FIELDS
    }


def _std_records(records: list[GraphBaselineRecord]) -> dict[str, float]:
    return {
        field: float(np.std([getattr(r, field) for r in records])) if records else 0.0
        for field in MEAN_FIELDS
    }


def _slice_batch(batch: dict, bi: int) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v[bi : bi + 1] if torch.is_tensor(v) else v
    return out


def evaluate(model, type_head, edge_head, loader, dataset, config: dict, device: torch.device) -> list[GraphBaselineRecord]:
    img_size = int(config["img_size"])
    records: list[GraphBaselineRecord] = []
    sample_index = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Graph baseline eval"):
            afm = batch["afm_stack"].to(device)
            edge_labels = build_edge_labels(batch, device)
            pred, features = model.forward_with_features(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            center_map_01 = torch.sigmoid(features["center_logits"])
            count_logits = features["count_logits"]

            for bi in range(afm.shape[0]):
                pred_obj = extract_predicted_objects(
                    center_map_01[bi : bi + 1],
                    pred_01[bi : bi + 1],
                    features["enc1"][bi : bi + 1],
                    afm[bi : bi + 1],
                    type_head,
                    edge_head,
                    device,
                    img_size=img_size,
                    count_logits=count_logits[bi : bi + 1],
                )
                gt_coords = batch["coords"][bi].detach().cpu().numpy()
                gt_types = batch["atom_types"][bi].detach().cpu().numpy()
                gt_mask = batch["atom_mask"][bi].detach().cpu().numpy()
                gt_edge_adj = edge_labels[bi].detach().cpu().numpy().astype(np.int32)
                pred_metrics = compute_pred_object_metrics(pred_obj, gt_coords, gt_types, gt_mask, gt_edge_adj, img_size=img_size)

                records.append(
                    GraphBaselineRecord(
                        dataset_index=sample_index,
                        cid=str(dataset.samples[sample_index]["cid"]),
                        gt_atom_count=int((gt_mask > 0.5).sum()),
                        pred_atom_count=int((pred_obj["mask"] > 0.5).sum()),
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


def _make_qualitative_figures(model, type_head, edge_head, loader, dataset, config: dict, device: torch.device, selected_records: list[GraphBaselineRecord], output_dir: Path) -> None:
    wanted = {r.dataset_index: r for r in selected_records}
    if not wanted:
        return

    img_size = int(config["img_size"])
    sample_index = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Graph figs", leave=False):
            afm = batch["afm_stack"].to(device)
            edge_labels = build_edge_labels(batch, device)
            pred, features = model.forward_with_features(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            center_map_01 = torch.sigmoid(features["center_logits"])
            count_logits = features["count_logits"]
            target = build_targets(batch, img_size, device).detach().cpu().numpy()

            for bi in range(afm.shape[0]):
                if sample_index not in wanted:
                    sample_index += 1
                    continue
                record = wanted[sample_index]
                pred_obj = extract_predicted_objects(
                    center_map_01[bi : bi + 1],
                    pred_01[bi : bi + 1],
                    features["enc1"][bi : bi + 1],
                    afm[bi : bi + 1],
                    type_head,
                    edge_head,
                    device,
                    img_size=img_size,
                    count_logits=count_logits[bi : bi + 1],
                )
                gt_coords = batch["coords"][bi].detach().cpu().numpy()
                gt_types = batch["atom_types"][bi].detach().cpu().numpy()
                gt_mask = batch["atom_mask"][bi].detach().cpu().numpy()
                gt_edge_adj = edge_labels[bi].detach().cpu().numpy().astype(np.int32)
                fig_path = output_dir / "samples" / f"graph_baseline_sample_{record.dataset_index:04d}.png"
                _make_sample_figure(
                    afm[bi].detach().cpu().numpy(),
                    target[bi],
                    pred_01[bi].detach().cpu().numpy(),
                    center_map_01[bi].detach().cpu().numpy(),
                    gt_coords,
                    gt_types,
                    gt_mask,
                    gt_edge_adj,
                    pred_obj["coords"],
                    pred_obj["types"],
                    pred_obj["mask"],
                    pred_obj["edge_adj"],
                    (
                        f"Graph Baseline | idx={record.dataset_index} | cid={record.cid} | "
                        f"score={record.pred_object_score:.3f} | type={record.pred_object_type_acc:.3f} | "
                        f"edge={record.pred_object_edge_f1:.3f}"
                    ),
                    fig_path,
                )
                record.figure_path = str(fig_path)
                sample_index += 1


def _pick_samples(records: list[GraphBaselineRecord]) -> tuple[GraphBaselineRecord | None, GraphBaselineRecord | None, GraphBaselineRecord | None]:
    if not records:
        return None, None, None
    ordered = sorted(records, key=lambda r: r.pred_object_score)
    worst = ordered[0]
    median = ordered[len(ordered) // 2]
    best = ordered[-1]
    return best, median, worst


def _safe_load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _build_summary(checkpoint_path: Path, config_path: Path, records: list[GraphBaselineRecord], v20_reference_json: Path | None, dense_reference_json: Path | None) -> dict:
    mean_metrics = _mean_records(records)
    std_metrics = _std_records(records)
    best, median, worst = _pick_samples(records)
    summary = {
        "checkpoint": str(checkpoint_path),
        "config_path": str(config_path),
        "num_samples": len(records),
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "best_sample": asdict(best) if best else None,
        "median_sample": asdict(median) if median else None,
        "worst_sample": asdict(worst) if worst else None,
        "top5_samples": [asdict(r) for r in sorted(records, key=lambda r: r.pred_object_score, reverse=True)[:5]],
        "bottom5_samples": [asdict(r) for r in sorted(records, key=lambda r: r.pred_object_score)[:5]],
    }

    v20_ref = _safe_load_json(v20_reference_json)
    if v20_ref is not None:
        ref = v20_ref.get("mean_metrics", {})
        summary["v20_reference_metrics"] = ref
        summary["graph_minus_v20"] = {
            key: float(mean_metrics[key] - ref[key])
            for key in mean_metrics.keys()
            if key in ref
        }

    dense_ref = _safe_load_json(dense_reference_json)
    if dense_ref is not None:
        ref = dense_ref.get("mean_metrics", {})
        summary["dense_reference_metrics"] = ref
        summary["graph_minus_dense"] = {
            key: float(mean_metrics[key] - ref[key])
            for key in mean_metrics.keys()
            if key in ref
        }
    return summary


def _write_markdown(summary: dict, output_path: Path) -> None:
    mm = summary["mean_metrics"]
    md = []
    md.append("# SUP-02 Graph Baseline Full-Test Report")
    md.append("")
    md.append("## 一、实验设置")
    md.append(f"- checkpoint：`{summary['checkpoint']}`")
    md.append(f"- config：`{summary['config_path']}`")
    md.append(f"- full-test 样本数：`{summary['num_samples']}`")
    md.append("")
    md.append("## 二、对象级主结果")
    for key in MEAN_FIELDS:
        md.append(f"- `{key}`：`{mm[key]:.4f}`")

    if "graph_minus_v20" in summary:
        md.append("")
        md.append("## 三、与 V20 对比")
        md.append("| 字段名 | Graph | V20 | Graph - V20 |")
        md.append("|---|---:|---:|---:|")
        for key in MEAN_FIELDS:
            if key in summary["v20_reference_metrics"]:
                md.append(
                    f"| {key} | {mm[key]:.4f} | {summary['v20_reference_metrics'][key]:.4f} | {summary['graph_minus_v20'][key]:+.4f} |"
                )

    if "graph_minus_dense" in summary:
        md.append("")
        md.append("## 四、与 SUP-01 Dense 对比")
        md.append("| 字段名 | Graph | Dense | Graph - Dense |")
        md.append("|---|---:|---:|---:|")
        for key in MEAN_FIELDS:
            if key in summary["dense_reference_metrics"]:
                md.append(
                    f"| {key} | {mm[key]:.4f} | {summary['dense_reference_metrics'][key]:.4f} | {summary['graph_minus_dense'][key]:+.4f} |"
                )

    md.append("")
    md.append("## 五、代表样本")
    for name in ("best_sample", "median_sample", "worst_sample"):
        sample = summary.get(name)
        if sample is None:
            continue
        md.append(
            f"- `{name}`：idx=`{sample['dataset_index']}` cid=`{sample['cid']}` "
            f"score=`{sample['pred_object_score']:.4f}` type_acc=`{sample['pred_object_type_acc']:.4f}` "
            f"edge_f1=`{sample['pred_object_edge_f1']:.4f}` figure=`{sample['figure_path']}`"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--eval_val_size", type=int, default=512)
    parser.add_argument("--v20_reference_json", default="/root/autodl-tmp/micro/experiments/v20_object_joint_medium10_exp01_fulltest/reports/fulltest_object_test.json")
    parser.add_argument("--dense_reference_json", default="/root/autodl-tmp/micro/experiments/v20_dense_stage1_medium10_sup01_fulltest/reports/dense_baseline_fulltest.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, type_head, edge_head, train_config, _ = load_graph_baseline_checkpoint(checkpoint_path, config_path, device)
    eval_config = resolve_config(train_config)
    eval_config["batch_size"] = int(args.batch_size)
    eval_config["max_samples"] = int(args.eval_max_samples)
    eval_config["val_size"] = int(args.eval_val_size)

    _, test_loader, _, _ = create_dataloaders(
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
    dataset = test_loader.dataset

    records = evaluate(model, type_head, edge_head, test_loader, dataset, eval_config, device)
    best, median, worst = _pick_samples(records)
    selected = [r for r in (best, median, worst) if r is not None]
    _make_qualitative_figures(model, type_head, edge_head, test_loader, dataset, eval_config, device, selected, output_dir)

    summary = _build_summary(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        records=records,
        v20_reference_json=Path(args.v20_reference_json) if args.v20_reference_json else None,
        dense_reference_json=Path(args.dense_reference_json) if args.dense_reference_json else None,
    )

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(records, reports_dir / "graph_baseline_fulltest_records.csv")
    (reports_dir / "graph_baseline_fulltest_records.json").write_text(
        json.dumps([asdict(r) for r in records], indent=2),
        encoding="utf-8",
    )
    (reports_dir / "graph_baseline_fulltest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, reports_dir / "graph_baseline_fulltest.md")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "report_md": str(reports_dir / "graph_baseline_fulltest.md"),
                "report_json": str(reports_dir / "graph_baseline_fulltest.json"),
                "num_samples": len(records),
                "pred_object_score": summary["mean_metrics"]["pred_object_score"],
                "pred_object_type_acc": summary["mean_metrics"]["pred_object_type_acc"],
                "pred_object_edge_f1": summary["mean_metrics"]["pred_object_edge_f1"],
                "pred_object_edge_f1_robust": summary["mean_metrics"]["pred_object_edge_f1_robust"],
                "pred_object_z_mae": summary["mean_metrics"]["pred_object_z_mae"],
            },
            indent=2,
        )
    )
