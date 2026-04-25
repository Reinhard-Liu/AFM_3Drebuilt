"""
SUP-01:
Full-test retrieval benchmark for the fair dense stage1 baseline.
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.train_v19_object_joint import build_edge_labels, compute_pred_object_metrics
from src.v19_visualize_test15 import _build_retrieval_db, compute_v19_object_similarity_for_retrieval
from src.v20_eval_dense_baseline import _decode_dense_objects, _load_dense_baseline


@dataclass
class DenseRetrievalRecord:
    dataset_index: int
    gt_cid: str
    gt_rank: int
    reciprocal_rank: float
    top1_hit: bool
    top3_hit: bool
    top5_hit: bool
    top1_cid: str
    top1_sim: float
    top3_cids: list[str]
    top3_sims: list[float]
    top5_cids: list[str]
    top5_sims: list[float]
    gt_atom_count: int
    gt_hetero_count: int
    gt_ring_count: int
    pred_atom_count: int
    pred_object_score: float
    pred_object_type_acc: float
    pred_object_macro_f1: float
    pred_object_edge_f1: float
    pred_object_edge_f1_robust: float
    pred_object_match_coverage_robust: float
    pred_object_z_mae: float
    pred_object_count_mae: float


def atom_count_bin(n: int) -> str:
    if n <= 22:
        return "<=22"
    if n <= 28:
        return "23-28"
    if n <= 34:
        return "29-34"
    return ">=35"


def hetero_count_bin(n: int) -> str:
    if n <= 1:
        return "0-1"
    if n <= 3:
        return "2-3"
    return ">=4"


def ring_count_bin(n: int) -> str:
    if n <= 1:
        return "0-1"
    if n == 2:
        return "2"
    return ">=3"


def count_mae_bin(mae: float) -> str:
    if mae < 0.5:
        return "0"
    if mae < 1.5:
        return "1"
    return ">=2"


def pred_score_bin(score: float) -> str:
    if score < 0.65:
        return "<0.65"
    if score < 0.75:
        return "0.65-0.75"
    return ">=0.75"


def z_mae_bin(z: float) -> str:
    if z < 0.05:
        return "<0.05"
    if z < 0.10:
        return "0.05-0.10"
    return ">=0.10"


def _compute_overall(records: list[DenseRetrievalRecord]) -> dict[str, float]:
    if not records:
        return {
            "num_queries": 0,
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
        }
    gt_ranks = [r.gt_rank for r in records]
    return {
        "num_queries": len(records),
        "top1": float(np.mean([r.top1_hit for r in records])),
        "top3": float(np.mean([r.top3_hit for r in records])),
        "top5": float(np.mean([r.top5_hit for r in records])),
        "mrr": float(np.mean([r.reciprocal_rank for r in records])),
        "mean_rank": float(np.mean(gt_ranks)),
        "median_rank": float(np.median(gt_ranks)),
    }


def _compute_group_stats(records: list[DenseRetrievalRecord]) -> dict[str, float]:
    overall = _compute_overall(records)
    overall["mean_pred_object_score"] = float(np.mean([r.pred_object_score for r in records])) if records else 0.0
    overall["mean_pred_object_type_acc"] = float(np.mean([r.pred_object_type_acc for r in records])) if records else 0.0
    overall["mean_pred_object_edge_f1"] = float(np.mean([r.pred_object_edge_f1 for r in records])) if records else 0.0
    overall["mean_pred_object_z_mae"] = float(np.mean([r.pred_object_z_mae for r in records])) if records else 0.0
    return overall


def _group_by(records: list[DenseRetrievalRecord], fn) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[DenseRetrievalRecord]] = defaultdict(list)
    for record in records:
        buckets[fn(record)].append(record)
    return {name: _compute_group_stats(group) for name, group in buckets.items()}


def _rank_histogram(records: list[DenseRetrievalRecord], output_path: Path) -> None:
    if not records:
        return
    ranks = [r.gt_rank for r in records]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = [1, 2, 3, 5, 10, 20, 50, 100, max(max(ranks), 100)]
    ax.hist(ranks, bins=bins, color="#f97316", alpha=0.85, edgecolor="white")
    ax.set_title("Dense Baseline GT Rank Distribution")
    ax.set_xlabel("GT Rank")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_stratification(name: str, strat: dict[str, dict[str, float]], output_path: Path) -> None:
    if not strat:
        return
    labels = list(strat.keys())
    top1 = [100.0 * strat[k]["top1"] for k in labels]
    top3 = [100.0 * strat[k]["top3"] for k in labels]
    top5 = [100.0 * strat[k]["top5"] for k in labels]
    mrr = [100.0 * strat[k]["mrr"] for k in labels]

    x = np.arange(len(labels))
    w = 0.20
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 1.5 * w, top1, width=w, label="Top1")
    ax.bar(x - 0.5 * w, top3, width=w, label="Top3")
    ax.bar(x + 0.5 * w, top5, width=w, label="Top5")
    ax.bar(x + 1.5 * w, mrr, width=w, label="MRR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percent")
    ax.set_title(f"Dense {name}")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _retrieve_ranked(
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    retrieval_db: list[dict],
    gt_cid: str,
) -> tuple[list[tuple[str, float]], int]:
    ranked: list[tuple[str, float]] = []
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
        sim = float(sim_dict["overall"])
        ranked.append((item["cid"], sim))
        if item["cid"] == gt_cid:
            gt_sim = sim
    ranked.sort(key=lambda x: x[1], reverse=True)
    gt_rank = sum(1 for _, sim in ranked if sim > gt_sim) + 1 if gt_sim is not None else -1
    return ranked, gt_rank


def evaluate(
    generator,
    loader,
    config: dict,
    device: torch.device,
    peak_threshold: float,
    min_distance_px: int,
    max_objects: int,
    bond_line_mean_threshold: float,
    bond_line_peak_threshold: float,
    bond_length_scale: float,
) -> list[DenseRetrievalRecord]:
    records: list[DenseRetrievalRecord] = []
    dataset = loader.dataset
    img_size = int(config["img_size"])
    retrieval_db = _build_retrieval_db(dataset)

    sample_index = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Dense retrieval", leave=False):
            afm = batch["afm_stack"].to(device)
            edge_labels = build_edge_labels(batch, device)
            pred = generator(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)

            for bi in range(afm.shape[0]):
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
                pred_metrics = compute_pred_object_metrics(
                    pred_obj,
                    gt_coords,
                    gt_types,
                    gt_mask,
                    gt_edge_adj,
                    img_size=img_size,
                )

                gt_cid = str(dataset.samples[sample_index]["cid"])
                ranked, gt_rank = _retrieve_ranked(
                    pred_obj["coords"],
                    pred_obj["types"],
                    pred_obj["mask"],
                    retrieval_db,
                    gt_cid,
                )
                top5 = ranked[:5]

                valid_gt = gt_mask > 0.5
                gt_atom_count = int(valid_gt.sum())
                gt_types_valid = gt_types[valid_gt]
                gt_hetero_count = int(np.sum(~np.isin(gt_types_valid, [0, 1])))
                gt_ring_count = int(batch["n_rings"][bi].item()) if "n_rings" in batch else 0

                records.append(
                    DenseRetrievalRecord(
                        dataset_index=sample_index,
                        gt_cid=gt_cid,
                        gt_rank=int(gt_rank),
                        reciprocal_rank=float(1.0 / gt_rank) if gt_rank > 0 else 0.0,
                        top1_hit=bool(gt_rank == 1),
                        top3_hit=bool(0 < gt_rank <= 3),
                        top5_hit=bool(0 < gt_rank <= 5),
                        top1_cid=top5[0][0] if top5 else "",
                        top1_sim=float(top5[0][1]) if top5 else 0.0,
                        top3_cids=[cid for cid, _ in top5[:3]],
                        top3_sims=[float(sim) for _, sim in top5[:3]],
                        top5_cids=[cid for cid, _ in top5],
                        top5_sims=[float(sim) for _, sim in top5],
                        gt_atom_count=gt_atom_count,
                        gt_hetero_count=gt_hetero_count,
                        gt_ring_count=gt_ring_count,
                        pred_atom_count=int(np.sum(pred_obj["mask"] > 0.5)),
                        pred_object_score=float(pred_metrics["pred_object_score"]),
                        pred_object_type_acc=float(pred_metrics["pred_object_type_acc"]),
                        pred_object_macro_f1=float(pred_metrics["pred_object_macro_f1"]),
                        pred_object_edge_f1=float(pred_metrics["pred_object_edge_f1"]),
                        pred_object_edge_f1_robust=float(pred_metrics["pred_object_edge_f1_robust"]),
                        pred_object_match_coverage_robust=float(pred_metrics["pred_object_match_coverage_robust"]),
                        pred_object_z_mae=float(pred_metrics["pred_object_z_mae"]),
                        pred_object_count_mae=float(pred_metrics["pred_object_count_mae"]),
                    )
                )
                sample_index += 1
    return records


def _build_summary(
    checkpoint_path: Path,
    config: dict,
    records: list[DenseRetrievalRecord],
    v20_reference_json: Path | None,
) -> tuple[dict, str]:
    overall = _compute_overall(records)
    stratifications = {
        "atom_count": _group_by(records, lambda r: atom_count_bin(r.gt_atom_count)),
        "hetero_count": _group_by(records, lambda r: hetero_count_bin(r.gt_hetero_count)),
        "ring_count": _group_by(records, lambda r: ring_count_bin(r.gt_ring_count)),
        "pred_object_count_mae": _group_by(records, lambda r: count_mae_bin(r.pred_object_count_mae)),
        "pred_object_score": _group_by(records, lambda r: pred_score_bin(r.pred_object_score)),
        "pred_object_z_mae": _group_by(records, lambda r: z_mae_bin(r.pred_object_z_mae)),
    }
    best_rr = sorted(records, key=lambda r: r.reciprocal_rank, reverse=True)[:10]
    worst_rank = sorted(records, key=lambda r: (r.gt_rank, -r.pred_object_score), reverse=True)[:10]

    v20_ref = {}
    delta = {}
    if v20_reference_json is not None and v20_reference_json.exists():
        ref = json.loads(v20_reference_json.read_text())
        v20_ref = ref.get("overall", {})
        for field in ["top1", "top3", "top5", "mrr", "mean_rank"]:
            if field in v20_ref and field in overall:
                delta[field] = float(overall[field] - float(v20_ref[field]))

    summary = {
        "checkpoint": str(checkpoint_path),
        "protocol": "closed_world_test_pool",
        "candidate_pool_size": len(records),
        "overall": overall,
        "v20_reference_overall": v20_ref,
        "dense_minus_v20": delta,
        "stratifications": stratifications,
        "best_rr_samples": [asdict(r) for r in best_rr],
        "worst_rank_samples": [asdict(r) for r in worst_rank],
        "records": [asdict(r) for r in records],
    }

    md: list[str] = []
    md.append("# SUP-01 Dense Baseline Retrieval Report")
    md.append("")
    md.append("## 一、实验设置")
    md.append(f"- checkpoint：`{checkpoint_path}`")
    md.append(f"- 检索协议：`closed_world_test_pool`")
    md.append(f"- 查询样本数：`{len(records)}`")
    md.append("")
    md.append("## 二、总体检索结果")
    md.append(f"- `Top1`：`{100.0 * overall['top1']:.2f}%`")
    md.append(f"- `Top3`：`{100.0 * overall['top3']:.2f}%`")
    md.append(f"- `Top5`：`{100.0 * overall['top5']:.2f}%`")
    md.append(f"- `MRR`：`{overall['mrr']:.4f}`")
    md.append(f"- `mean_rank`：`{overall['mean_rank']:.4f}`")
    md.append(f"- `median_rank`：`{overall['median_rank']:.4f}`")
    md.append("")
    if v20_ref:
        md.append("## 三、与 V20 检索对比")
        md.append("| 字段名 | Dense | V20 | Dense - V20 |")
        md.append("|---|---:|---:|---:|")
        for field in ["top1", "top3", "top5", "mrr", "mean_rank"]:
            md.append(
                f"| {field} | {float(overall[field]):.4f} | {float(v20_ref[field]):.4f} | {float(delta[field]):+.4f} |"
            )
        md.append("")
    md.append("## 四、分层统计")
    for key, strat in stratifications.items():
        md.append(f"### {key}")
        md.append("")
        md.append("| 分层 | 样本数 | Top1 | Top3 | Top5 | MRR | mean_rank | mean_pred_score | mean_type_acc | mean_edge_f1 | mean_z_mae |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for group_name, stats in strat.items():
            md.append(
                f"| {group_name} | {stats['num_queries']} | "
                f"{100.0 * stats['top1']:.2f}% | {100.0 * stats['top3']:.2f}% | {100.0 * stats['top5']:.2f}% | "
                f"{stats['mrr']:.4f} | {stats['mean_rank']:.2f} | {stats['mean_pred_object_score']:.4f} | "
                f"{stats['mean_pred_object_type_acc']:.4f} | {stats['mean_pred_object_edge_f1']:.4f} | "
                f"{stats['mean_pred_object_z_mae']:.4f} |"
            )
        md.append("")
    md.append("## 五、核心判断")
    md.append("- 这组结果回答 dense 2D baseline 是否也能支持 retrieval。")
    md.append("- 如果 `Top1/Top3/MRR` 明显低于 V20，则说明对象级 `2D+z` 中间层不仅改善结构指标，也改善候选缩小能力。")
    return summary, "\n".join(md) + "\n"


def _save_plots(summary: dict, output_dir: Path) -> None:
    records = [DenseRetrievalRecord(**r) for r in summary["records"]]
    plots_dir = output_dir / "plots"
    _rank_histogram(records, plots_dir / "rank_histogram.png")
    for key, strat in summary["stratifications"].items():
        _plot_stratification(key, strat, plots_dir / f"{key}_stratification.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--eval_val_size", type=int, default=512)
    parser.add_argument("--peak_threshold", type=float, default=0.45)
    parser.add_argument("--min_distance_px", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=64)
    parser.add_argument("--bond_line_mean_threshold", type=float, default=0.18)
    parser.add_argument("--bond_line_peak_threshold", type=float, default=0.35)
    parser.add_argument("--bond_length_scale", type=float, default=1.35)
    parser.add_argument("--v20_reference_json", default="/root/autodl-tmp/micro/experiments/v20_object_joint_medium10_exp02_retrieval_fulltest/reports/retrieval_fulltest_test.json")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, train_config, _state = _load_dense_baseline(checkpoint_path, config_path, device)

    eval_config = dict(train_config)
    eval_config["batch_size"] = int(args.batch_size)
    eval_config["max_samples"] = int(args.eval_max_samples)
    eval_config["val_size"] = int(args.eval_val_size)

    _, _, test_loader, _ = create_dataloaders(
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

    records = evaluate(
        generator,
        test_loader,
        eval_config,
        device,
        peak_threshold=args.peak_threshold,
        min_distance_px=args.min_distance_px,
        max_objects=args.max_objects,
        bond_line_mean_threshold=args.bond_line_mean_threshold,
        bond_line_peak_threshold=args.bond_line_peak_threshold,
        bond_length_scale=args.bond_length_scale,
    )
    summary, markdown = _build_summary(
        checkpoint_path=checkpoint_path,
        config=eval_config,
        records=records,
        v20_reference_json=Path(args.v20_reference_json) if args.v20_reference_json else None,
    )
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "dense_retrieval_fulltest_test.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "dense_retrieval_fulltest_test.md").write_text(markdown, encoding="utf-8")
    _save_plots(summary, output_dir)
    print(json.dumps({
        "output_dir": str(output_dir),
        "report_json": str(reports_dir / "dense_retrieval_fulltest_test.json"),
        "report_md": str(reports_dir / "dense_retrieval_fulltest_test.md"),
        "top1": summary["overall"]["top1"],
        "top3": summary["overall"]["top3"],
        "top5": summary["overall"]["top5"],
        "mrr": summary["overall"]["mrr"],
        "mean_rank": summary["overall"]["mean_rank"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
