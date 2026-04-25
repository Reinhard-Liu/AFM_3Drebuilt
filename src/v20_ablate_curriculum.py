"""
EXP-06:
Curriculum + closure ablation runner for the V20 object-joint training recipe.

Default use:
- debug-scale ablation on top of config_v20_object_joint_debug.json
- reuse existing v20_object_joint_debug full-setting checkpoint when available
- run selected ablation groups and summarize best validation metrics
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_BASE_CONFIG = ROOT / "config_v20_object_joint_debug.json"
DEFAULT_FULL_REF_CKPT = ROOT / "experiments" / "v20_object_joint_debug" / "checkpoints" / "best_v19_object_joint.pt"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "v20_ablate_curriculum_debug"


KEY_METRICS = [
    "peak_object_score",
    "pred_object_score",
    "peak_center_type_acc",
    "pred_object_type_acc",
    "pred_object_macro_f1",
    "pred_object_edge_f1",
    "pred_object_edge_f1_robust",
    "pred_object_count_mae",
    "pred_object_z_mae",
]


GROUP_SPECS: dict[str, dict] = {
    "full": {
        "description": "V20 full setting",
        "overrides": {},
        "reuse_full_reference": True,
    },
    "gt_only": {
        "description": "Only GT object supervision; disable peak/pred closure and curriculum",
        "overrides": {
            "lambda_type_obj_peak_start": 0.0,
            "lambda_type_obj_peak_final": 0.0,
            "lambda_type_obj_pred_start": 0.0,
            "lambda_type_obj_pred_final": 0.0,
            "lambda_edge_obj_peak_start": 0.0,
            "lambda_edge_obj_peak_final": 0.0,
            "lambda_edge_obj_pred_start": 0.0,
            "lambda_edge_obj_pred_final": 0.0,
            "lambda_peak_consistency_start": 0.0,
            "lambda_peak_consistency_final": 0.0,
            "lambda_pred_type_consistency_start": 0.0,
            "lambda_pred_type_consistency_final": 0.0,
            "center_curriculum_alpha_start": 0.0,
            "center_curriculum_alpha_final": 0.0,
            "center_curriculum_warmup_epochs": 1,
        },
    },
    "peak_only": {
        "description": "Only peak-centered object supervision; remove GT and pred closure losses",
        "overrides": {
            "lambda_type_obj_gt": 0.0,
            "lambda_edge_obj_gt": 0.0,
            "lambda_type_obj_pred_start": 0.0,
            "lambda_type_obj_pred_final": 0.0,
            "lambda_edge_obj_pred_start": 0.0,
            "lambda_edge_obj_pred_final": 0.0,
            "lambda_peak_consistency_start": 0.0,
            "lambda_peak_consistency_final": 0.0,
            "lambda_pred_type_consistency_start": 0.0,
            "lambda_pred_type_consistency_final": 0.0,
            "lambda_teacher_type_distill": 0.0,
            "lambda_teacher_type_pred_distill": 0.0,
        },
    },
    "pred_only": {
        "description": "Only predicted-center closure supervision; remove GT and peak branches",
        "overrides": {
            "lambda_type_obj_gt": 0.0,
            "lambda_edge_obj_gt": 0.0,
            "lambda_type_obj_peak_start": 0.0,
            "lambda_type_obj_peak_final": 0.0,
            "lambda_edge_obj_peak_start": 0.0,
            "lambda_edge_obj_peak_final": 0.0,
            "lambda_peak_consistency_start": 0.0,
            "lambda_peak_consistency_final": 0.0,
            "lambda_pred_type_consistency_start": 0.0,
            "lambda_pred_type_consistency_final": 0.0,
            "lambda_teacher_type_distill": 0.0,
            "lambda_teacher_type_pred_distill": 0.0,
        },
    },
    "no_curriculum": {
        "description": "Keep all branches but remove center/loss schedules",
        "overrides": {
            "center_curriculum_alpha_start": 1.0,
            "center_curriculum_alpha_final": 1.0,
            "center_curriculum_warmup_epochs": 1,
            "lambda_type_obj_peak_start": 2.5,
            "lambda_type_obj_peak_final": 2.5,
            "lambda_type_obj_pred_start": 1.5,
            "lambda_type_obj_pred_final": 1.5,
            "lambda_edge_obj_peak_start": 2.5,
            "lambda_edge_obj_peak_final": 2.5,
            "lambda_edge_obj_pred_start": 1.25,
            "lambda_edge_obj_pred_final": 1.25,
            "lambda_peak_consistency_start": 0.5,
            "lambda_peak_consistency_final": 0.5,
            "lambda_pred_type_consistency_start": 0.5,
            "lambda_pred_type_consistency_final": 0.5,
            "loss_warmup_epochs": 1,
            "aux_decay_epochs": 1,
        },
    },
    "no_pred_closure": {
        "description": "Disable predicted-center closure losses and pred consistency",
        "overrides": {
            "lambda_type_obj_pred_start": 0.0,
            "lambda_type_obj_pred_final": 0.0,
            "lambda_edge_obj_pred_start": 0.0,
            "lambda_edge_obj_pred_final": 0.0,
            "lambda_pred_type_consistency_start": 0.0,
            "lambda_pred_type_consistency_final": 0.0,
            "lambda_teacher_type_pred_distill": 0.0,
        },
    },
}


@dataclass
class AblationResult:
    group: str
    description: str
    checkpoint_path: str
    config_path: str
    best_epoch: int
    pred_gap: float
    metrics: dict[str, float]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def _resolve_auto_paths(config: dict) -> dict:
    out = copy.deepcopy(config)
    if out.get("data_root") == "auto":
        out["data_root"] = str(ROOT / "dataverse_files" / "SUBMIT_QUAM-AFM" / "QUAM")
    if out.get("save_dir") == "auto":
        out["save_dir"] = str(ROOT / "experiments" / "v20_object_joint_debug" / "checkpoints")
    return out


def _prepare_config(base_config: dict, output_dir: Path, group: str) -> tuple[dict, Path]:
    spec = GROUP_SPECS[group]
    config = copy.deepcopy(base_config)
    config = _resolve_auto_paths(config)
    config.update(spec.get("overrides", {}))
    config["save_dir"] = str(output_dir / group / "checkpoints")
    config.pop("resume_from_checkpoint", None)
    config_path = output_dir / "configs" / f"{group}.json"
    _save_json(config, config_path)
    return config, config_path


def _load_checkpoint_metrics(checkpoint_path: Path) -> tuple[int, dict[str, float]]:
    state = json.loads("{}")
    import torch

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = loaded.get("val_metrics", {})
    epoch = int(loaded.get("epoch", -1))
    return epoch, metrics


def _train_group(config_path: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "src" / "train_v19_object_joint.py"),
        "--config",
        str(config_path),
    ]
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"training failed for {config_path}, see {log_path}")


def _metric_or_zero(metrics: dict[str, float], key: str) -> float:
    return float(metrics.get(key, 0.0))


def _summarize_results(results: list[AblationResult]) -> tuple[dict, str]:
    summary = {
        "groups": [asdict(r) for r in results],
    }

    ref = next((r for r in results if r.group == "full"), None)
    ref_metrics = ref.metrics if ref is not None else {}

    md: list[str] = []
    md.append("# V20 EXP-06 Curriculum + Closure Ablation")
    md.append("")
    md.append("## 一、实验组")
    for r in results:
        md.append(f"- `{r.group}`：{r.description}")
    md.append("")
    md.append("## 二、主表")
    md.append("| 组别 | best_epoch | peak_object | pred_object | peak->pred gap | peak_type_acc | pred_type_acc | pred_macro_f1 | pred_edge_f1 | robust_edge_f1 | pred_count_mae | pred_z_mae |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        m = r.metrics
        md.append(
            f"| {r.group} | {r.best_epoch} | "
            f"{_metric_or_zero(m, 'peak_object_score'):.4f} | "
            f"{_metric_or_zero(m, 'pred_object_score'):.4f} | "
            f"{r.pred_gap:.4f} | "
            f"{_metric_or_zero(m, 'peak_center_type_acc'):.4f} | "
            f"{_metric_or_zero(m, 'pred_object_type_acc'):.4f} | "
            f"{_metric_or_zero(m, 'pred_object_macro_f1'):.4f} | "
            f"{_metric_or_zero(m, 'pred_object_edge_f1'):.4f} | "
            f"{_metric_or_zero(m, 'pred_object_edge_f1_robust'):.4f} | "
            f"{_metric_or_zero(m, 'pred_object_count_mae'):.4f} | "
            f"{_metric_or_zero(m, 'pred_object_z_mae'):.4f} |"
        )
    md.append("")

    if ref_metrics:
        md.append("## 三、相对 Full 的变化")
        md.append("| 组别 | d_pred_object | d_pred_type_acc | d_pred_macro_f1 | d_pred_edge_f1 | d_pred_count_mae | d_gap |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        ref_gap = _metric_or_zero(ref_metrics, "peak_object_score") - _metric_or_zero(ref_metrics, "pred_object_score")
        for r in results:
            if r.group == "full":
                continue
            m = r.metrics
            md.append(
                f"| {r.group} | "
                f"{_metric_or_zero(m, 'pred_object_score') - _metric_or_zero(ref_metrics, 'pred_object_score'):+.4f} | "
                f"{_metric_or_zero(m, 'pred_object_type_acc') - _metric_or_zero(ref_metrics, 'pred_object_type_acc'):+.4f} | "
                f"{_metric_or_zero(m, 'pred_object_macro_f1') - _metric_or_zero(ref_metrics, 'pred_object_macro_f1'):+.4f} | "
                f"{_metric_or_zero(m, 'pred_object_edge_f1') - _metric_or_zero(ref_metrics, 'pred_object_edge_f1'):+.4f} | "
                f"{_metric_or_zero(m, 'pred_object_count_mae') - _metric_or_zero(ref_metrics, 'pred_object_count_mae'):+.4f} | "
                f"{r.pred_gap - ref_gap:+.4f} |"
            )
        md.append("")

    md.append("## 四、判断")
    md.append("- `pred_object_score` 与 `peak->pred gap` 一起看，判断哪一组最能缩小闭环断层。")
    md.append("- `pred_object_type_acc`、`pred_object_macro_f1`、`pred_object_edge_f1` 一起看，判断 typed graph 闭环最依赖哪种监督。")
    md.append("- 如果 `no_pred_closure` 明显变差，说明 predicted-center closure 不是可有可无的附加项，而是主贡献之一。")
    md.append("- 如果 `no_curriculum` 明显变差，说明不是单纯多加损失，而是课程式接入时机本身起作用。")
    return summary, "\n".join(md) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--full_reference_ckpt", default=str(DEFAULT_FULL_REF_CKPT))
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["full", "gt_only", "peak_only", "pred_only", "no_curriculum", "no_pred_closure"],
    )
    parser.add_argument("--reuse_full_reference", action="store_true", default=True)
    args = parser.parse_args()

    base_config_path = Path(args.base_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = _resolve_auto_paths(_load_json(base_config_path))

    results: list[AblationResult] = []
    for group in args.groups:
        if group not in GROUP_SPECS:
            raise ValueError(f"unknown group: {group}")
        spec = GROUP_SPECS[group]
        config, config_path = _prepare_config(base_config, output_dir, group)
        ckpt_path = Path(config["save_dir"]) / "best_v19_object_joint.pt"
        log_path = output_dir / "logs" / f"{group}.log"

        if group == "full" and args.reuse_full_reference and Path(args.full_reference_ckpt).exists():
            ckpt_path = Path(args.full_reference_ckpt)
        else:
            _train_group(config_path, log_path)

        best_epoch, metrics = _load_checkpoint_metrics(ckpt_path)
        pred_gap = _metric_or_zero(metrics, "peak_object_score") - _metric_or_zero(metrics, "pred_object_score")
        results.append(
            AblationResult(
                group=group,
                description=spec["description"],
                checkpoint_path=str(ckpt_path),
                config_path=str(config_path),
                best_epoch=best_epoch,
                pred_gap=float(pred_gap),
                metrics={k: _metric_or_zero(metrics, k) for k in KEY_METRICS},
            )
        )

    summary, markdown = _summarize_results(results)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ablation_summary.md").write_text(markdown, encoding="utf-8")
    (reports_dir / "ablation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(output_dir),
        "report_md": str(reports_dir / "ablation_summary.md"),
        "report_json": str(reports_dir / "ablation_summary.json"),
        "groups": [r.group for r in results],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
