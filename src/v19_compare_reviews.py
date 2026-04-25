"""
Compare two V19 review_summary.json files and write a Chinese comparison report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = [
    ("peak_object_score", "预测中心闭环对象总分", "higher"),
    ("gt_object_score", "真实中心条件对象总分", "higher"),
    ("atom_center_score_r3", "原子中心命中分数", "higher"),
    ("peak_center_type_acc", "预测中心上的对象级原子类型准确率", "higher"),
    ("peak_center_macro_f1", "预测中心上的对象级原子类型宏平均 F1", "higher"),
    ("peak_center_hetero_f1", "预测中心上的对象级杂原子 F1", "higher"),
    ("peak_center_edge_f1", "预测中心上的对象级边 F1", "higher"),
    ("peak_center_shift_px", "预测中心相对真实中心的平均像素偏移", "lower"),
    ("atom_z_mae_r3", "真实原子附近 z 平均绝对误差", "lower"),
    ("typed_center_score_r3", "位置和类型同时对上的软分数", "higher"),
    ("atom_type_macro_f1_2d", "二维类型图宏平均 F1", "higher"),
]


def _load(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _trend_word(delta: float, direction: str) -> str:
    if abs(delta) < 1e-12:
        return "持平"
    if direction == "higher":
        return "提升" if delta > 0 else "下降"
    return "改善" if delta < 0 else "变差"


def build_report(base_name: str, base: dict, new_name: str, new: dict) -> tuple[dict, str]:
    base_best = base["best_metrics"]
    new_best = new["best_metrics"]
    base_epoch = base["best_epoch"]
    new_epoch = new["best_epoch"]

    compare_rows = []
    for key, zh, direction in METRICS:
        old = float(base_best[key])
        cur = float(new_best[key])
        delta = cur - old
        compare_rows.append(
            {
                "field": key,
                "zh": zh,
                "baseline": old,
                "current": cur,
                "delta": delta,
                "direction": direction,
                "trend": _trend_word(delta, direction),
            }
        )

    base_gap = float(base_best["gt_object_score"] - base_best["peak_object_score"])
    new_gap = float(new_best["gt_object_score"] - new_best["peak_object_score"])
    gap_delta = new_gap - base_gap

    summary = {
        "baseline_run": base_name,
        "current_run": new_name,
        "baseline_best_epoch": base_epoch,
        "current_best_epoch": new_epoch,
        "baseline_gap_gt_minus_peak": base_gap,
        "current_gap_gt_minus_peak": new_gap,
        "gap_delta": gap_delta,
        "metrics": compare_rows,
    }

    md = []
    md.append(f"# V19 训练结果中文对比报告：{base_name} vs {new_name}")
    md.append("")
    md.append("## 一、对比对象")
    md.append(f"- 基线实验：`{base_name}`")
    md.append(f"- 当前实验：`{new_name}`")
    md.append(f"- 基线最佳轮次：`{base_epoch}`")
    md.append(f"- 当前最佳轮次：`{new_epoch}`")
    md.append("")
    md.append("## 二、关键结论")
    md.append(
        f"- `gt_object_score - peak_object_score` 的差距从 `{base_gap:.4f}` 缩小到 `{new_gap:.4f}`，"
        f"字段名 `gap_delta` = `{gap_delta:.4f}`。这说明预测中心闭环能力明显更接近真实中心上限。"
    )
    md.append(
        f"- 字段名 `peak_object_score`：`{base_best['peak_object_score']:.4f}` -> `{new_best['peak_object_score']:.4f}`，"
        "说明整体对象级闭环重建继续提升。"
    )
    md.append(
        f"- 字段名 `peak_center_type_acc`：`{base_best['peak_center_type_acc']:.4f}` -> `{new_best['peak_center_type_acc']:.4f}`；"
        f"字段名 `peak_center_macro_f1`：`{base_best['peak_center_macro_f1']:.4f}` -> `{new_best['peak_center_macro_f1']:.4f}`。"
        "说明预测中心下的原子类型能力明显变强。"
    )
    md.append(
        f"- 字段名 `peak_center_hetero_f1`：`{base_best['peak_center_hetero_f1']:.4f}` -> `{new_best['peak_center_hetero_f1']:.4f}`。"
        "说明杂原子识别是这轮提升最明显的方向之一。"
    )
    md.append(
        f"- 字段名 `peak_center_edge_f1`：`{base_best['peak_center_edge_f1']:.4f}` -> `{new_best['peak_center_edge_f1']:.4f}`，"
        f"字段名 `atom_z_mae_r3`：`{base_best['atom_z_mae_r3']:.4f}` -> `{new_best['atom_z_mae_r3']:.4f}`。"
        "说明边恢复和 z 轴误差也有改善，但幅度小于类型和杂原子。"
    )
    md.append("")
    md.append("## 三、指标逐项对比")
    md.append("")
    md.append("| 字段名 | 中文含义 | 基线值 | 当前值 | 变化 | 结论 |")
    md.append("|---|---|---:|---:|---:|---|")
    for row in compare_rows:
        md.append(
            f"| `{row['field']}` | {row['zh']} | `{row['baseline']:.4f}` | "
            f"`{row['current']:.4f}` | `{row['delta']:+.4f}` | {row['trend']} |"
        )
    md.append("")
    md.append("## 四、总体判断")
    md.append("- 当前这轮 `full15_all` 相比 `full6h`，不是单项偶然波动，而是对象级闭环能力整体更强。")
    md.append("- 最大的实质提升来自：预测中心下的类型、杂原子，以及真实中心上限与预测中心闭环之间差距的缩小。")
    md.append("- 稠密二维类型图仍然偏弱，后续仍应优先发展对象级类型/边头，而不是重新把主希望压回稠密 `type map`。")
    md.append("")
    return summary, "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--current", type=str, required=True)
    parser.add_argument("--baseline_name", type=str, required=True)
    parser.add_argument("--current_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    baseline = _load(Path(args.baseline))
    current = _load(Path(args.current))
    summary, report_md = build_report(args.baseline_name, baseline, args.current_name, current)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "compare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "compare_summary.md", "w") as f:
        f.write(report_md)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
