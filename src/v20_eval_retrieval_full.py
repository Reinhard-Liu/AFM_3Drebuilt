"""
Formal full-test retrieval benchmark and stratified statistics for V20.

Current default protocol:
- closed-world retrieval within the current test split candidate pool
- reports Top1 / Top3 / Top5 / MRR / mean rank
- stratifies by atom count, hetero count, ring complexity,
  pred_object_count_mae, pred_object_score, pred_object_z_mae
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
from src.train_v19_object_joint import (
    build_edge_labels,
    compute_pred_object_metrics,
    extract_predicted_objects,
)
from src.v19_object_joint_review import _load_model
from src.v19_visualize_test15 import (
    _build_retrieval_db,
    compute_v19_object_similarity_for_retrieval,
)


@dataclass
class RetrievalRecord:
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


def _compute_overall(records: list[RetrievalRecord]) -> dict[str, float]:
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


def _compute_group_stats(records: list[RetrievalRecord]) -> dict[str, float]:
    overall = _compute_overall(records)
    overall["mean_pred_object_score"] = float(np.mean([r.pred_object_score for r in records])) if records else 0.0
    overall["mean_pred_object_type_acc"] = float(np.mean([r.pred_object_type_acc for r in records])) if records else 0.0
    overall["mean_pred_object_edge_f1"] = float(np.mean([r.pred_object_edge_f1 for r in records])) if records else 0.0
    overall["mean_pred_object_z_mae"] = float(np.mean([r.pred_object_z_mae for r in records])) if records else 0.0
    return overall


def _group_by(records: list[RetrievalRecord], fn) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[RetrievalRecord]] = defaultdict(list)
    for record in records:
        buckets[fn(record)].append(record)
    return {name: _compute_group_stats(group) for name, group in buckets.items()}


def _rank_histogram(records: list[RetrievalRecord], output_path: Path) -> None:
    if not records:
        return
    ranks = [r.gt_rank for r in records]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = [1, 2, 3, 5, 10, 20, 50, 100, max(max(ranks), 100)]
    ax.hist(ranks, bins=bins, color="#3b82f6", alpha=0.85, edgecolor="white")
    ax.set_title("GT Rank Distribution")
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
    ax.set_title(name)
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


def _build_summary(
    checkpoint_path: Path,
    config: dict,
    split: str,
    candidate_pool_size: int,
    records: list[RetrievalRecord],
    fixed15_top3_hit: int = 13,
    fixed15_total: int = 15,
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
    summary = {
        "checkpoint": str(checkpoint_path),
        "protocol": "closed_world_test_pool",
        "split": split,
        "candidate_pool_size": int(candidate_pool_size),
        "overall": overall,
        "fixed15_reference": {
            "top3_hit_count": int(fixed15_top3_hit),
            "num_samples": int(fixed15_total),
            "top3_rate": float(fixed15_top3_hit / fixed15_total),
        },
        "stratifications": stratifications,
        "best_rr_samples": [asdict(r) for r in best_rr],
        "worst_rank_samples": [asdict(r) for r in worst_rank],
        "records": [asdict(r) for r in records],
    }

    md: list[str] = []
    md.append("# V20 EXP-02 全测试集检索与分层统计报告")
    md.append("")
    md.append("## 一、实验设置")
    md.append(f"- checkpoint：`{checkpoint_path}`")
    md.append(f"- 检索协议：`closed_world_test_pool`")
    md.append(f"- split：`{split}`")
    md.append(f"- 查询样本数：`{len(records)}`")
    md.append(f"- 候选池大小：`{candidate_pool_size}`")
    md.append(f"- 参数域：`{config.get('param_key', 'K-1')}`")
    md.append(f"- 固定15样本参考 Top3：`{fixed15_top3_hit}/{fixed15_total}` = `{100.0 * fixed15_top3_hit / fixed15_total:.2f}%`")
    md.append("")
    md.append("## 二、全测试集总体检索结果")
    md.append(f"- `Top1`：`{100.0 * overall['top1']:.2f}%`")
    md.append(f"- `Top3`：`{100.0 * overall['top3']:.2f}%`")
    md.append(f"- `Top5`：`{100.0 * overall['top5']:.2f}%`")
    md.append(f"- `MRR`：`{overall['mrr']:.4f}`")
    md.append(f"- `mean_rank`：`{overall['mean_rank']:.4f}`")
    md.append(f"- `median_rank`：`{overall['median_rank']:.4f}`")
    md.append("")
    md.append("## 三、分层统计")
    strat_titles = {
        "atom_count": "按原子数分层",
        "hetero_count": "按杂原子数分层",
        "ring_count": "按环复杂度分层",
        "pred_object_count_mae": "按原子数误差分层",
        "pred_object_score": "按纯预测对象总分分层",
        "pred_object_z_mae": "按z误差分层",
    }
    for key, title in strat_titles.items():
        md.append(f"### {title}")
        md.append("")
        md.append("| 分层 | 样本数 | Top1 | Top3 | Top5 | MRR | mean_rank | mean_pred_score | mean_type_acc | mean_edge_f1 | mean_z_mae |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for group_name, stats in stratifications[key].items():
            md.append(
                f"| {group_name} | {stats['num_queries']} | "
                f"{100.0 * stats['top1']:.2f}% | {100.0 * stats['top3']:.2f}% | {100.0 * stats['top5']:.2f}% | "
                f"{stats['mrr']:.4f} | {stats['mean_rank']:.2f} | "
                f"{stats['mean_pred_object_score']:.4f} | {stats['mean_pred_object_type_acc']:.4f} | "
                f"{stats['mean_pred_object_edge_f1']:.4f} | {stats['mean_pred_object_z_mae']:.4f} |"
            )
        md.append("")
    md.append("## 四、代表成功样本")
    for record in best_rr[:5]:
        md.append(
            f"- `idx={record.dataset_index}` `cid={record.gt_cid}` "
            f"`rank={record.gt_rank}` `top1_hit={'是' if record.top1_hit else '否'}` "
            f"`pred_object_score={record.pred_object_score:.4f}` "
            f"`type_acc={record.pred_object_type_acc:.4f}` "
            f"`edge_f1={record.pred_object_edge_f1:.4f}` "
            f"`z_mae={record.pred_object_z_mae:.4f}`"
        )
    md.append("")
    md.append("## 五、代表失败样本")
    for record in worst_rank[:5]:
        md.append(
            f"- `idx={record.dataset_index}` `cid={record.gt_cid}` "
            f"`rank={record.gt_rank}` `top1={record.top1_cid}` "
            f"`pred_object_score={record.pred_object_score:.4f}` "
            f"`type_acc={record.pred_object_type_acc:.4f}` "
            f"`edge_f1={record.pred_object_edge_f1:.4f}` "
            f"`robust_edge_f1={record.pred_object_edge_f1_robust:.4f}` "
            f"`z_mae={record.pred_object_z_mae:.4f}`"
        )
    md.append("")
    md.append("## 六、核心判断")
    md.append("- 这个 EXP-02 使用的是 test split 闭集检索协议，因此它最适合回答“当前对象级恢复能否支持候选缩小”。")
    md.append("- `Top1` 衡量直接命中能力，`Top3/Top5` 更接近当前论文的 candidate set reduction 叙事。")
    md.append("- 分层表应重点观察：复杂样本上检索下降是否主要伴随 `pred_object_type_acc`、`pred_object_edge_f1` 或 `pred_object_z_mae` 恶化。")
    md.append("- 如果高 `pred_object_score` / 低 `pred_object_z_mae` 分层明显更强，说明对象级结构重建和检索能力是同向变化的。")
    md.append("")
    return summary, "\n".join(md)


def _save_plots(summary: dict, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    _rank_histogram(
        [RetrievalRecord(**r) for r in summary["records"]],
        plots_dir / "rank_histogram.png",
    )
    for key, strat in summary["stratifications"].items():
        _plot_stratification(key, strat, plots_dir / f"{key}_stratification.png")


def evaluate(
    model,
    type_head,
    edge_head,
    loader,
    config: dict,
    device: torch.device,
) -> list[RetrievalRecord]:
    records: list[RetrievalRecord] = []
    dataset = loader.dataset
    img_size = int(config["img_size"])
    retrieval_db = _build_retrieval_db(dataset)

    sample_index = 0
    model.eval()
    type_head.eval()
    edge_head.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="EXP-02 retrieval", leave=False):
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
                pred_metrics = compute_pred_object_metrics(
                    pred_obj,
                    gt_coords,
                    gt_types,
                    gt_mask,
                    gt_edge_adj,
                    img_size=img_size,
                    edge_match_radius_px=float(config.get("pred_object_edge_match_radius_px", 3.0)),
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
                    RetrievalRecord(
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test"])
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
    summary, md = _build_summary(
        checkpoint_path=checkpoint_path,
        config=config,
        split=args.split,
        candidate_pool_size=len(test_loader.dataset),
        records=records,
    )

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "retrieval_fulltest_test.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(reports_dir / "retrieval_fulltest_test.md", "w", encoding="utf-8") as f:
        f.write(md)

    _save_plots(summary, output_dir)

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "report_json": str(reports_dir / "retrieval_fulltest_test.json"),
            "report_md": str(reports_dir / "retrieval_fulltest_test.md"),
            "num_queries": summary["overall"]["num_queries"],
            "top1": summary["overall"]["top1"],
            "top3": summary["overall"]["top3"],
            "top5": summary["overall"]["top5"],
            "mrr": summary["overall"]["mrr"],
            "mean_rank": summary["overall"]["mean_rank"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
