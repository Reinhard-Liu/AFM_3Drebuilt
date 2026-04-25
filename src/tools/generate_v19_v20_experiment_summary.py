from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/micro")
EXPERIMENTS_ROOT = ROOT / "experiments"
OUTPUT_PATH = ROOT / "docs" / "V19_V20实验总索引与总结.md"


FIELD_INFO: dict[str, tuple[str, str, str]] = {
    "atom_xy_mae": ("二维原子图平均绝对误差", "连续误差", "越低越好"),
    "atom_center_score_r3": ("原子中心半径3像素命中分数", "0-1", "越高越好"),
    "typed_center_score_r3": ("位置与类型同时正确的软分数", "0-1", "越高越好"),
    "atom_type_macro_f1_2d": ("二维类型图宏平均F1", "0-1", "越高越好"),
    "z_map_mae": ("二维z图平均绝对误差", "连续误差", "越低越好"),
    "atom_z_mae_r3": ("真实中心附近z平均绝对误差", "连续误差", "越低越好"),
    "gt_center_type_acc": ("GT-center条件原子类型准确率", "0-1", "越高越好"),
    "gt_center_macro_f1": ("GT-center条件原子类型宏平均F1", "0-1", "越高越好"),
    "gt_center_hetero_f1": ("GT-center条件杂原子F1", "0-1", "越高越好"),
    "gt_center_edge_precision": ("GT-center条件对象级边精确率", "0-1", "越高越好"),
    "gt_center_edge_recall": ("GT-center条件对象级边召回率", "0-1", "越高越好"),
    "gt_center_edge_f1": ("GT-center条件对象级边F1", "0-1", "越高越好"),
    "peak_center_type_acc": ("peak-center条件原子类型准确率", "0-1", "越高越好"),
    "peak_center_macro_f1": ("peak-center条件原子类型宏平均F1", "0-1", "越高越好"),
    "peak_center_hetero_f1": ("peak-center条件杂原子F1", "0-1", "越高越好"),
    "peak_center_edge_precision": ("peak-center条件对象级边精确率", "0-1", "越高越好"),
    "peak_center_edge_recall": ("peak-center条件对象级边召回率", "0-1", "越高越好"),
    "peak_center_edge_f1": ("peak-center条件对象级边F1", "0-1", "越高越好"),
    "peak_center_shift_px": ("peak-center相对真实中心平均偏移", "像素", "越低越好"),
    "gt_object_score": ("GT-center条件对象级总分", "0-1", "越高越好"),
    "peak_object_score": ("预测中心闭环对象总分", "0-1", "越高越好"),
    "pred_object_score": ("纯预测对象闭环对象级总分", "0-1", "越高越好"),
    "pred_object_3d_score": ("纯预测对象3D综合分", "0-1", "越高越好"),
    "pred_object_count_mae": ("纯预测对象原子数平均绝对误差", "个", "越低越好"),
    "pred_object_count_score": ("纯预测对象原子数相似度分数", "0-1", "越高越好"),
    "pred_object_center_score": ("纯预测对象proposal中心平均置信度", "0-1", "越高越好"),
    "pred_object_type_acc": ("纯预测对象原子类型准确率", "0-1", "越高越好"),
    "pred_object_macro_f1": ("纯预测对象原子类型宏平均F1", "0-1", "越高越好"),
    "pred_object_hetero_f1": ("纯预测对象杂原子F1", "0-1", "越高越好"),
    "pred_object_edge_f1": ("纯预测对象严格对象级边F1", "0-1", "越高越好"),
    "pred_object_edge_f1_robust": ("纯预测对象距离容忍后的稳健边F1", "0-1", "越高越好"),
    "pred_object_match_coverage_robust": ("稳健匹配覆盖率", "0-1", "越高越好"),
    "pred_object_graph_score": ("纯预测对象图结构综合分", "0-1", "越高越好"),
    "pred_object_heavy_rmsd": ("纯预测对象重原子RMSD", "归一化坐标", "越低越好"),
    "pred_object_heavy_rmsd_ang": ("纯预测对象重原子RMSD", "Å", "越低越好"),
    "pred_object_z_mae": ("纯预测对象z平均绝对误差", "连续误差", "越低越好"),
    "bond_map_mae": ("二维键图平均绝对误差", "0-1", "越低越好"),
    "type_map_mae": ("二维类型图平均绝对误差", "0-1", "越低越好"),
    "type_top1_local_acc_r3": ("半径3像素内局部Top1类型准确率", "0-1", "越高越好"),
    "ch_collapse_rate_2d": ("C/H塌缩率", "0-1", "越低越好"),
    "matched_type_acc_r3": ("匹配后对象类型准确率", "0-1", "越高越好"),
    "matched_macro_f1_r3": ("匹配后对象类型宏平均F1", "0-1", "越高越好"),
    "matched_hetero_f1_r3": ("匹配后杂原子F1", "0-1", "越高越好"),
    "matched_gt_node_coverage_r3": ("匹配后GT节点覆盖率", "0-1", "越高越好"),
    "matched_gt_bond_coverage_r3": ("匹配后GT键覆盖率", "0-1", "越高越好"),
    "edge_f1_xy_r3": ("XY半径3像素容忍下的边F1", "0-1", "越高越好"),
    "edge_gap_robust": ("稳健边F1与严格边F1之间的差值", "差值", "越低越好"),
    "edge_gap_xy_r3": ("XY容忍边F1与严格边F1之间的差值", "差值", "越低越好"),
    "mean_xy_match_px_r3": ("匹配对象的平均XY偏移", "像素", "越低越好"),
    "mean_z_abs_err_matched_ang": ("匹配对象z绝对误差均值", "Å", "越低越好"),
    "matched_pair_dist_mae_ang": ("匹配对象两两距离平均绝对误差", "Å", "越低越好"),
    "matched_bond_len_mae_ang": ("匹配对象键长平均绝对误差", "Å", "越低越好"),
    "gt_height_span_ang": ("GT高度跨度", "Å", "仅统计量"),
    "gt_nonplanarity_ang": ("GT非平面度", "Å", "仅统计量"),
    "pred_object_pair_dist_mae_r3": ("匹配对象两两距离MAE", "Å", "越低越好"),
    "pred_object_bond_len_mae_r3": ("匹配对象键长MAE", "Å", "越低越好"),
    "pred_object_nonplanarity_error_r3": ("非平面度误差", "Å", "越低越好"),
    "pred_object_z_corr_r3": ("匹配对象z相关系数", "-1到1", "越高越好"),
    "pair_dist_mae_r3_le_0p25_rate": ("两两距离误差≤0.25Å的样本占比", "0-1", "越高越好"),
    "bond_len_mae_r3_le_0p20_rate": ("键长误差≤0.20Å的样本占比", "0-1", "越高越好"),
    "z_corr_r3_ge_0p80_rate": ("z相关系数≥0.80的样本占比", "0-1", "越高越好"),
    "nonplanarity_error_r3_le_0p10_rate": ("非平面度误差≤0.10Å的样本占比", "0-1", "越高越好"),
    "top1": ("Top1命中率", "0-1", "越高越好"),
    "top3": ("Top3命中率", "0-1", "越高越好"),
    "top5": ("Top5命中率", "0-1", "越高越好"),
    "mrr": ("平均倒数排名", "0-1", "越高越好"),
    "mean_rank": ("平均排名", "名次", "越低越好"),
    "median_rank": ("中位排名", "名次", "越低越好"),
    "mean_pred_object_score": ("该分层中的平均pred_object_score", "0-1", "越高越好"),
    "mean_pred_object_type_acc": ("该分层中的平均pred_object_type_acc", "0-1", "越高越好"),
    "mean_pred_object_edge_f1": ("该分层中的平均pred_object_edge_f1", "0-1", "越高越好"),
    "mean_pred_object_z_mae": ("该分层中的平均pred_object_z_mae", "连续误差", "越低越好"),
    "mean_center_score": ("平均中心置信度", "0-1", "越高越好"),
    "ref_object_sim": ("参考结构对象级相似度", "0-1", "越高越好"),
    "ref_type_acc": ("参考结构对象级类型准确率", "0-1", "越高越好"),
    "ref_macro_f1": ("参考结构对象级类型宏平均F1", "0-1", "越高越好"),
    "ref_edge_f1": ("参考结构对象级边F1", "0-1", "越高越好"),
    "ref_coord_score": ("参考结构坐标相似度分数", "0-1", "越高越好"),
    "ref_count_score": ("参考结构原子数相似度分数", "0-1", "越高越好"),
}


CONFIG_INFO: dict[str, str] = {
    "data_root": "训练/评估数据集根目录",
    "save_dir": "实验输出目录",
    "param_key": "AFM参数配置键",
    "img_size": "输入AFM图像分辨率",
    "min_corrugation": "样本最低起伏阈值",
    "augment_rotation": "是否使用旋转增强",
    "require_ring": "是否只保留含环分子",
    "batch_size": "每步batch大小",
    "num_workers": "DataLoader并行worker数量",
    "max_samples": "训练集最大样本数上限",
    "val_size": "验证/测试子集样本数",
    "epochs": "训练轮数",
    "lr": "初始学习率",
    "weight_decay": "权重衰减",
    "min_lr": "学习率下限",
    "base_ch": "主干网络基础通道数",
    "warm_start_checkpoint": "warm start初始化checkpoint路径",
    "teacher_type_checkpoint": "教师类型模型checkpoint路径",
    "teacher_temperature": "教师蒸馏温度",
    "lambda_teacher_type_distill": "教师类型蒸馏损失权重",
    "lambda_teacher_type_pred_distill": "预测对象类型蒸馏损失权重",
    "lambda_center": "中心热图损失权重",
    "lambda_atom_aux_start": "原子辅助头起始权重",
    "lambda_atom_aux_final": "原子辅助头最终权重",
    "lambda_z_start": "z图辅助头起始权重",
    "lambda_z_final": "z图辅助头最终权重",
    "lambda_type_obj_gt": "GT-center对象级类型损失权重",
    "lambda_type_obj_peak_start": "peak-center对象级类型损失起始权重",
    "lambda_type_obj_peak_final": "peak-center对象级类型损失最终权重",
    "lambda_type_obj_pred_start": "pred-center对象级类型损失起始权重",
    "lambda_type_obj_pred_final": "pred-center对象级类型损失最终权重",
    "lambda_edge_obj_gt": "GT-center对象级边损失权重",
    "lambda_edge_obj_peak_start": "peak-center对象级边损失起始权重",
    "lambda_edge_obj_peak_final": "peak-center对象级边损失最终权重",
    "lambda_edge_obj_pred_start": "pred-center对象级边损失起始权重",
    "lambda_edge_obj_pred_final": "pred-center对象级边损失最终权重",
    "lambda_type_map_aux_start": "二维类型图辅助损失起始权重",
    "lambda_type_map_aux_final": "二维类型图辅助损失最终权重",
    "lambda_bond_map_aux_start": "二维键图辅助损失起始权重",
    "lambda_bond_map_aux_final": "二维键图辅助损失最终权重",
    "lambda_peak_consistency_start": "peak一致性损失起始权重",
    "lambda_peak_consistency_final": "peak一致性损失最终权重",
    "lambda_pred_type_consistency_start": "pred类型一致性损失起始权重",
    "lambda_pred_type_consistency_final": "pred类型一致性损失最终权重",
    "consistency_temperature": "一致性蒸馏温度",
    "aux_decay_epochs": "辅助分支衰减轮数",
    "loss_warmup_epochs": "损失预热轮数",
    "center_curriculum_alpha_start": "中心课程学习alpha起始值",
    "center_curriculum_alpha_final": "中心课程学习alpha最终值",
    "center_curriculum_warmup_epochs": "中心课程学习预热轮数",
    "center_search_radius": "中心搜索半径",
    "pred_train_match_radius_px": "pred对象训练匹配半径",
    "lambda_object_count": "对象数分类损失权重",
    "lambda_object_count_mae": "对象数MAE损失权重",
    "lambda_atom": "dense baseline原子图损失权重",
    "lambda_bond": "dense baseline键图损失权重",
    "lambda_type": "dense baseline类型图损失权重",
    "type_hidden_dim": "graph baseline类型头隐藏维度",
    "legacy_type_num_gnn_layers": "graph baseline类型头GNN层数",
    "legacy_type_num_heads": "graph baseline注意力头数",
    "legacy_type_bond_threshold": "graph baseline成键阈值",
    "legacy_type_token_grid_size": "graph baseline token网格尺寸",
    "type_label_smoothing": "类型标签平滑系数",
    "lambda_atom_aux": "graph baseline原子辅助损失权重",
    "lambda_z": "graph baseline z损失权重",
    "lambda_type_obj_pred": "graph baseline pred对象类型损失权重",
    "lambda_edge_obj_pred": "graph baseline pred对象边损失权重",
    "disable_z_for_object_heads": "是否关闭对象头中的z特征输入",
}


REPORT_FIELD_INFO: dict[str, str] = {
    "baseline_run": "对比基线实验名",
    "current_run": "当前实验名",
    "best_epoch": "最佳checkpoint对应epoch",
    "pred_gap": "`peak_object_score - pred_object_score` 的闭环断层",
    "sample_count": "评估样本数",
    "num_samples": "样本数",
    "num_queries": "检索查询数",
    "num_cases": "真实AFM case数",
    "split": "评估数据划分",
    "protocol": "检索协议",
    "candidate_pool_size": "检索候选池大小",
    "candidate_count": "候选结构数",
    "gt_rank": "GT或同身份候选在排序中的名次",
    "reciprocal_rank": "名次倒数",
    "gt_in_top3": "GT是否进入Top3",
    "top3_hit_count": "15样本中GT进入Top3的命中个数",
    "top3_rate": "15样本中GT进入Top3的命中率",
    "top1_hit": "Top1是否命中",
    "top3_hit": "Top3是否命中",
    "top5_hit": "Top5是否命中",
    "top1_cid": "Top1候选CID",
    "top1_label": "Top1候选分子标签",
    "top1_candidate_name": "Top1候选名称",
    "top1_sim": "Top1相似度",
    "top3_cids": "Top3候选CID列表",
    "top3_labels": "Top3候选分子标签列表",
    "top3_candidate_names": "Top3候选名称列表",
    "top3_sims": "Top3相似度列表",
    "gt_atom_count": "GT原子数",
    "pred_atom_count": "预测原子数",
    "ref_atom_count": "参考结构原子数",
    "gt_kind": "左侧参考结构类型（gt/reference）",
    "chosen_variant": "最终选用的真实AFM对比度变体",
    "contrast_variant": "真实AFM对比度变体",
    "molecule_name": "分子名称",
    "molecule_label": "分子标签",
    "tip": "探针类型",
    "checkpoint": "使用的checkpoint路径",
    "checkpoint_epoch": "checkpoint对应epoch",
    "summary_source": "可视化总结来源JSON",
    "figure_path": "主图路径",
    "compar_figure_path": "五列对比图路径",
    "main_figure": "主图路径",
    "compar_figure": "五列对比图路径",
    "gt_structure_compatible": "是否可做GT几何指标评估",
    "high_gap_ge_0p20": "严格/稳健边F1差值大于等于0.20的样本数",
    "high_gap_ge_0p20_ratio": "严格/稳健边F1差值大于等于0.20的样本占比",
    "robust_ge_0p90_and_strict_lt_0p70": "稳健边F1≥0.90且严格边F1<0.70的样本数",
    "robust_ge_0p90_and_strict_lt_0p70_ratio": "稳健边F1≥0.90且严格边F1<0.70的样本占比",
    "matched_type_gain_ge_0p10": "匹配后类型准确率提升至少0.10的样本数",
    "matched_type_gain_ge_0p10_ratio": "匹配后类型准确率提升至少0.10的样本占比",
}


RETRIEVAL_STRAT_ZH = {
    "atom_count": "按原子数分层",
    "hetero_count": "按杂原子数分层",
    "ring_count": "按环数分层",
    "pred_object_count_mae": "按预测对象数误差分层",
    "pred_object_score": "按纯预测对象总分分层",
    "pred_object_z_mae": "按纯预测对象z误差分层",
}


GEOM_STRAT_ZH = {
    "atom_count": "按原子数分层",
    "ring_count": "按环数分层",
    "height_span": "按GT高度跨度分层",
    "nonplanarity": "按GT非平面度分层",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _link(path: Path | str, label: str | None = None) -> str:
    p = Path(path)
    return f"[{label or p.name}]({p})"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.2f}"
        return f"{value:.4f}"
    return str(value)


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        if abs(value) >= 100:
            return f"{value:+.2f}"
        return f"{value:+.4f}"
    return str(value)


def _join_links(paths: list[Path]) -> str:
    if not paths:
        return "-"
    return "<br>".join(_link(p) for p in paths)


def _join_strings(items: list[Any]) -> str:
    if not items:
        return "-"
    return "; ".join(str(x) for x in items)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _metric_row(field: str, values: dict[str, Any]) -> list[Any]:
    zh, unit, direction = FIELD_INFO.get(field, (field, "-", "-"))
    return [field, zh, unit, direction] + [values[k] for k in values]


def _render_metric_dict(fields: list[str]) -> str:
    rows = []
    for field in fields:
        zh, unit, direction = FIELD_INFO.get(field, (field, "-", "-"))
        rows.append([field, zh, unit, direction])
    return _md_table(["字段名", "中文意义", "取值/单位", "趋势"], rows)


def _render_config_dict(fields: list[str]) -> str:
    rows = [[f, CONFIG_INFO.get(f, f)] for f in fields]
    return _md_table(["配置变量", "中文意义"], rows)


def _render_report_field_dict(fields: list[str]) -> str:
    rows = [[f, REPORT_FIELD_INFO.get(f, f)] for f in fields]
    return _md_table(["报告字段", "中文意义"], rows)


def _collect_experiment_roots() -> list[Path]:
    return sorted(
        [
            p
            for p in EXPERIMENTS_ROOT.iterdir()
            if p.is_dir() and (p.name.startswith("v19") or p.name.startswith("v20"))
        ],
        key=lambda x: x.name,
    )


def _experiment_kind(name: str) -> str:
    if name.startswith("v19_joint"):
        return "v19早期joint调试"
    if "stage1" in name:
        return "2D dense/stage1"
    if "compare" in name:
        return "对比分析"
    if "full15" in name:
        return "正式长训/全样本复盘"
    if "full6h" in name:
        return "夜间完整训练复盘"
    if "curriculum" in name:
        return "课程学习/闭环调试"
    if "typefocus" in name or "type_upper" in name:
        return "类型分支调试"
    if "type_closure" in name:
        return "类型闭环调试"
    if "ablate" in name:
        return "消融实验"
    if "dense" in name:
        return "dense baseline"
    if "graph" in name:
        return "graph baseline"
    if "exp01" in name:
        return "EXP-01 full-test"
    if "exp02" in name:
        return "EXP-02 retrieval"
    if "exp03" in name:
        return "EXP-03 strict/robust gap"
    if "exp04" in name:
        return "EXP-04 geometry diagnostics"
    if "sup03" in name:
        return "SUP-03 真实AFM"
    if "visual" in name:
        return "可视化结果"
    if "object_joint" in name:
        return "对象级联合训练"
    if "prededge" in name or "edge_refine" in name:
        return "边分支调试"
    return "其他"


def _scan_artifacts(roots: list[Path]) -> tuple[str, str, str]:
    model_rows: list[list[Any]] = []
    report_rows: list[list[Any]] = []
    visual_rows: list[list[Any]] = []
    for root in roots:
        ckdir = root / "checkpoints"
        if ckdir.exists():
            pt_files = sorted(ckdir.glob("*.pt"))
            history_files = sorted(ckdir.glob("history*.json"))
            preview_files = sorted(ckdir.glob("*.png"))
            model_rows.append(
                [
                    root.name,
                    _experiment_kind(root.name),
                    _link(root),
                    _link(ckdir, "checkpoints"),
                    _join_links(pt_files),
                    _join_links(history_files),
                    _join_links(preview_files),
                ]
            )
        report_dirs: list[Path] = []
        for candidate in [root / "reports", root / "review" / "reports"]:
            if candidate.exists():
                report_dirs.append(candidate)
        report_dirs.extend(sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("visual_reports_")]))
        if report_dirs:
            md_files: list[Path] = []
            json_files: list[Path] = []
            csv_files: list[Path] = []
            for d in report_dirs:
                md_files.extend(sorted(d.glob("*.md")))
                json_files.extend(sorted(d.glob("*.json")))
                csv_files.extend(sorted(d.glob("*.csv")))
            report_rows.append(
                [
                    root.name,
                    _experiment_kind(root.name),
                    _link(root),
                    "<br>".join(_link(d) for d in report_dirs),
                    _join_links(md_files),
                    _join_links(json_files),
                    _join_links(csv_files),
                ]
            )
        visual_dirs = [p for p in root.iterdir() if p.is_dir() and ("visual" in p.name or p.name in {"samples", "plots"} or p.name == "review")]
        if visual_dirs:
            visual_rows.append(
                [
                    root.name,
                    _experiment_kind(root.name),
                    _link(root),
                    "<br>".join(_link(d) for d in visual_dirs),
                ]
            )
    return (
        _md_table(["实验目录", "类别", "目录", "checkpoint目录", "模型文件(.pt)", "history/json", "预览图"], model_rows),
        _md_table(["实验目录", "类别", "目录", "报告目录", "Markdown报告", "JSON数据", "CSV数据"], report_rows),
        _md_table(["实验目录", "类别", "目录", "可视化/样本目录"], visual_rows),
    )


def _render_review_report(title: str, report_path: Path) -> str:
    data = _load_json(report_path)
    best = data["best_metrics"]
    epoch1 = data["epoch1_metrics"]
    rows = []
    for field in best:
        zh, unit, direction = FIELD_INFO.get(field, (field, "-", "-"))
        rows.append([field, zh, unit, direction, _fmt(epoch1.get(field)), _fmt(best.get(field)), _fmt_delta(best.get(field) - epoch1.get(field))])
    lines = [
        f"### {title}",
        "",
        f"- 报告：{_link(report_path.with_suffix('.md'))} / { _link(report_path) }",
        f"- `best_epoch`：`{data['best_epoch']}`",
        "",
        _md_table(["字段名", "中文意义", "单位/范围", "趋势", "Epoch1", "Best", "变化值"], rows),
        "",
    ]
    if data.get("improvements"):
        imp_rows = [[k, _fmt(v)] for k, v in data["improvements"].items()]
        lines.extend(["附加摘要变化：", "", _md_table(["字段", "变化值"], imp_rows), ""])
    return "\n".join(lines)


def _render_compare_report(title: str, report_path: Path) -> str:
    data = _load_json(report_path)
    rows = []
    for item in data["metrics"]:
        rows.append([
            item["field"],
            item["zh"],
            FIELD_INFO.get(item["field"], (item["zh"], "-", item["direction"]))[1],
            "越高越好" if item["direction"] == "higher" else "越低越好",
            _fmt(item["baseline"]),
            _fmt(item["current"]),
            _fmt_delta(item["delta"]),
            item["trend"],
        ])
    return "\n".join(
        [
            f"### {title}",
            "",
            f"- 报告：{_link(report_path.with_suffix('.md'))} / { _link(report_path) }",
            f"- 基线：`{data['baseline_run']}`，当前：`{data['current_run']}`",
            f"- `baseline_best_epoch={data['baseline_best_epoch']}`，`current_best_epoch={data['current_best_epoch']}`",
            "",
            _md_table(["字段名", "中文意义", "单位/范围", "趋势", "基线", "当前", "变化值", "结论"], rows),
            "",
        ]
    )


def _render_visual15_report(title: str, report_path: Path, is_real: bool = False) -> str:
    data = _load_json(report_path)
    lines = [f"### {title}", ""]
    lines.append(f"- 摘要：{_link(report_path.with_suffix('.md'))} / { _link(report_path) }")
    lines.append(f"- 使用checkpoint：`{data['checkpoint']}`")
    if "num_samples" in data:
        lines.append(f"- 样本数：`{data['num_samples']}`")
    if "num_cases" in data:
        lines.append(f"- Case数：`{data['num_cases']}`")
    if "gt_in_top3_count" in data:
        lines.append(f"- `gt_in_top3_count`：`{data['gt_in_top3_count']}`")
    lines.append("")
    if not is_real:
        rows = []
        for r in data["records"]:
            score_field = "pred_object_score" if "pred_object_score" in r else "peak_object_score"
            type_acc_field = "pred_object_type_acc" if "pred_object_type_acc" in r else "peak_center_type_acc"
            macro_f1_field = "pred_object_macro_f1" if "pred_object_macro_f1" in r else "peak_center_macro_f1"
            hetero_f1_field = "pred_object_hetero_f1" if "pred_object_hetero_f1" in r else "peak_center_hetero_f1"
            edge_f1_field = "pred_object_edge_f1" if "pred_object_edge_f1" in r else "peak_center_edge_f1"
            z_field = "pred_object_z_mae" if "pred_object_z_mae" in r else "atom_z_mae_r3"
            rows.append(
                [
                    r["dataset_index"],
                    r["gt_cid"],
                    r["gt_rank"],
                    "是" if r["gt_in_top3"] else "否",
                    r["pred_atom_count"],
                    r["gt_atom_count"],
                    _fmt(r[score_field]),
                    _fmt(r[type_acc_field]),
                    _fmt(r[macro_f1_field]),
                    _fmt(r[hetero_f1_field]),
                    _fmt(r[edge_f1_field]),
                    _fmt(r[z_field]),
                    _join_strings(r["top3_cids"]),
                    _join_strings([_fmt(x) for x in r["top3_sims"]]),
                    _link(Path(r["main_figure"]), "main"),
                    _link(Path(r["compar_figure"]), "compar"),
                ]
            )
        lines.append(
            _md_table(
                [
                    "样本编号",
                    "GT CID",
                    "GT Rank",
                    "GT在Top3",
                    "Pred原子数",
                    "GT原子数",
                    "pred_object_score",
                    "pred_object_type_acc",
                    "pred_object_macro_f1",
                    "pred_object_hetero_f1",
                    "pred_object_edge_f1",
                    "pred_object_z_mae",
                    "Top3 CID",
                    "Top3 sim",
                    "主图",
                    "五列图",
                ],
                rows,
            )
        )
    else:
        rows = []
        for r in data["records"]:
            mb = r["metric_block"]
            score_v = mb.get("ref_object_sim", mb.get("pred_object_score"))
            type_acc_v = mb.get("ref_type_acc", mb.get("pred_object_type_acc"))
            macro_f1_v = mb.get("ref_macro_f1", mb.get("pred_object_macro_f1"))
            edge_f1_v = mb.get("ref_edge_f1", mb.get("pred_object_edge_f1"))
            coord_v = mb.get("ref_coord_score", mb.get("pred_object_edge_f1_robust"))
            count_v = mb.get("ref_count_score", mb.get("pred_object_z_mae"))
            rows.append(
                [
                    r["case_id"],
                    r["molecule_label"],
                    r["tip"],
                    r["chosen_variant"],
                    r["gt_kind"],
                    r["gt_rank"],
                    r["pred_atom_count"],
                    r["ref_atom_count"],
                    _fmt(score_v),
                    _fmt(type_acc_v),
                    _fmt(macro_f1_v),
                    _fmt(edge_f1_v),
                    _fmt(coord_v),
                    _fmt(count_v),
                    _join_strings(r["top3_labels"]),
                    _join_strings([_fmt(x) for x in r["top3_sims"]]),
                    _link(Path(r["main_figure"]), "main"),
                    _link(Path(r["compar_figure"]), "compar"),
                ]
            )
        lines.append(
            _md_table(
                [
                    "Case",
                    "分子标签",
                    "tip",
                    "选用变体",
                    "参考类型",
                    "GT Rank",
                    "Pred原子数",
                    "参考原子数",
                    "ref_object_sim",
                    "ref_type_acc",
                    "ref_macro_f1",
                    "ref_edge_f1",
                    "ref_coord_score",
                    "ref_count_score",
                    "Top3标签",
                    "Top3 sim",
                    "主图",
                    "五列图",
                ],
                rows,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _render_fulltest_report(title: str, report_path: Path) -> str:
    data = _load_json(report_path)
    rows = []
    for field, mean_v in data["fulltest_mean_metrics"].items():
        zh, unit, direction = FIELD_INFO.get(field, (field, "-", "-"))
        rows.append(
            [
                field,
                zh,
                unit,
                direction,
                _fmt(mean_v),
                _fmt(data["fulltest_std_metrics"].get(field)),
                _fmt(data["validation_reference_metrics"].get(field)),
                _fmt_delta(data["fulltest_minus_validation"].get(field)),
            ]
        )
    cfg_rows = [[k, _fmt(v), CONFIG_INFO.get(k, k)] for k, v in data["config_snapshot"].items()]
    return "\n".join(
        [
            f"### {title}",
            "",
            f"- 报告：{_link(report_path.with_suffix('.md'))} / { _link(report_path) } / {_link(report_path.parent / 'fulltest_object_test_samples.csv')}",
            f"- `checkpoint_epoch={data['checkpoint_epoch']}`，`split={data['split']}`，`sample_count={data['sample_count']}`",
            "",
            "配置快照：",
            "",
            _md_table(["配置字段", "值", "中文意义"], cfg_rows),
            "",
            _md_table(["字段名", "中文意义", "单位/范围", "趋势", "Full-test均值", "Std", "Validation参考", "差值"], rows),
            "",
        ]
    )


def _render_retrieval_overall(title: str, report_path: Path, extra_compare_keys: list[str] | None = None, include_strat: bool = False) -> str:
    data = _load_json(report_path)
    lines = [f"### {title}", ""]
    lines.append(f"- 报告：{_link(report_path.with_suffix('.md'))} / { _link(report_path) }")
    meta_bits = []
    for key in ("protocol", "split", "candidate_pool_size"):
        if key in data:
            meta_bits.append(f"`{key}={data[key]}`")
    if meta_bits:
        lines.append("- " + "，".join(meta_bits))
    lines.append("")
    overall = data["overall"]
    rows = []
    for field, value in overall.items():
        if field == "num_queries":
            continue
        zh, unit, direction = FIELD_INFO.get(field, (field, "-", "-"))
        row = [field, zh, unit, direction, _fmt(value)]
        if extra_compare_keys:
            for extra_key in extra_compare_keys:
                extra_block = data.get(extra_key, {})
                row.append(_fmt(extra_block.get(field)))
                if field in extra_block:
                    row.append(_fmt_delta(extra_block[field]))
                else:
                    row.append("-")
        rows.append(row)
    headers = ["字段名", "中文意义", "单位/范围", "趋势", "overall"]
    if extra_compare_keys:
        for extra_key in extra_compare_keys:
            headers.extend([f"{extra_key}参考", f"相对{extra_key}变化"])
    lines.append(f"- `num_queries={overall.get('num_queries', '-')}`")
    lines.append("")
    lines.append(_md_table(headers, rows))
    lines.append("")
    if "fixed15_reference" in data:
        ref = data["fixed15_reference"]
        rows2 = [[k, REPORT_FIELD_INFO.get(k, FIELD_INFO.get(k, (k, "-", "-"))[0]), _fmt(v)] for k, v in ref.items()]
        lines.extend(["固定15样本参考：", "", _md_table(["字段", "中文意义", "值"], rows2), ""])
    if include_strat and "stratifications" in data:
        for strat_name, strat_block in data["stratifications"].items():
            rows3 = []
            for bin_name, stats in strat_block.items():
                rows3.append(
                    [
                        bin_name,
                        stats["num_queries"],
                        _fmt(stats["top1"]),
                        _fmt(stats["top3"]),
                        _fmt(stats["top5"]),
                        _fmt(stats["mrr"]),
                        _fmt(stats["mean_rank"]),
                        _fmt(stats["mean_pred_object_score"]),
                        _fmt(stats["mean_pred_object_type_acc"]),
                        _fmt(stats["mean_pred_object_edge_f1"]),
                        _fmt(stats["mean_pred_object_z_mae"]),
                    ]
                )
            lines.extend(
                [
                    f"分层：{RETRIEVAL_STRAT_ZH.get(strat_name, strat_name)}",
                    "",
                    _md_table(
                        ["分层区间", "查询数", "Top1", "Top3", "Top5", "MRR", "MeanRank", "MeanPredScore", "MeanTypeAcc", "MeanEdgeF1", "MeanZMae"],
                        rows3,
                    ),
                    "",
                ]
            )
    return "\n".join(lines)


def _render_gap_report(title: str, report_path: Path) -> str:
    data = _load_json(report_path)
    sections = [f"### {title}", "", f"- 报告：{_link(report_path.with_suffix('.md'))} / { _link(report_path) } / {_link(report_path.parent / 'gap_decomposition_test_records.csv')}", ""]
    mean_rows = []
    for field, value in data["mean_metrics"].items():
        zh, unit, direction = FIELD_INFO.get(field, (field, "-", "-"))
        mean_rows.append([field, zh, unit, direction, _fmt(value)])
    sections.extend(["平均指标：", "", _md_table(["字段名", "中文意义", "单位/范围", "趋势", "均值"], mean_rows), ""])
    count_rows = []
    for field, value in data["counts"].items():
        count_rows.append([field, REPORT_FIELD_INFO.get(field, FIELD_INFO.get(field, (field, "-", "-"))[0]), _fmt(value)])
    sections.extend(["计数统计：", "", _md_table(["字段", "中文意义", "值"], count_rows), ""])
    for block_name, zh_title in [
        ("high_gap_mean_metrics", "高gap子集均值"),
        ("robust_good_strict_bad_mean_metrics", "robust高但strict差子集均值"),
    ]:
        rows = []
        for field, value in data[block_name].items():
            zh, unit, direction = FIELD_INFO.get(field, (field, "-", "-"))
            rows.append([field, zh, unit, direction, _fmt(value)])
        sections.extend([f"{zh_title}：", "", _md_table(["字段名", "中文意义", "单位/范围", "趋势", "值"], rows), ""])
    return "\n".join(sections)


def _render_geom_report(title: str, report_path: Path) -> str:
    data = _load_json(report_path)
    sections = [f"### {title}", "", f"- 报告：{_link(report_path.with_suffix('.md'))} / { _link(report_path) } / {_link(report_path.parent / 'geom_diagnostics_test_records.csv')}", ""]
    mean_rows = []
    for field, value in data["mean_metrics"].items():
        zh, unit, direction = FIELD_INFO.get(field, (field, "-", "-"))
        mean_rows.append([field, zh, unit, direction, _fmt(value)])
    sections.extend(["平均指标：", "", _md_table(["字段名", "中文意义", "单位/范围", "趋势", "均值"], mean_rows), ""])
    for strat_name, strat_block in data["stratifications"].items():
        rows = []
        for bin_name, stats in strat_block.items():
            rows.append(
                [
                    bin_name,
                    stats["count"],
                    _fmt(stats["pred_object_score"]),
                    _fmt(stats["pred_object_type_acc"]),
                    _fmt(stats["pred_object_edge_f1"]),
                    _fmt(stats["pred_object_edge_f1_robust"]),
                    _fmt(stats["pred_object_z_mae"]),
                    _fmt(stats["pred_object_heavy_rmsd_ang"]),
                    _fmt(stats["pred_object_pair_dist_mae_r3"]),
                    _fmt(stats["pred_object_bond_len_mae_r3"]),
                    _fmt(stats["pred_object_z_corr_r3"]),
                    _fmt(stats["pred_object_nonplanarity_error_r3"]),
                ]
            )
        sections.extend(
            [
                f"分层：{GEOM_STRAT_ZH.get(strat_name, strat_name)}",
                "",
                _md_table(
                    ["分层区间", "样本数", "PredScore", "TypeAcc", "EdgeF1", "RobustEdgeF1", "ZMae", "HeavyRMSD(Å)", "PairDistMAE(Å)", "BondLenMAE(Å)", "ZCorr", "NonPlanarityErr(Å)"],
                    rows,
                ),
                "",
            ]
        )
    return "\n".join(sections)


def _render_baseline_fulltest(title: str, report_path: Path, compare_key: str, compare_label: str) -> str:
    data = _load_json(report_path)
    sections = [f"### {title}", "", f"- 报告：{_link(report_path.with_suffix('.md'))} / { _link(report_path) }", ""]
    cfg_rows = []
    for block_name in ("train_config_snapshot", "eval_config_snapshot"):
        if block_name in data:
            for k, v in data[block_name].items():
                cfg_rows.append([f"{block_name}.{k}", _fmt(v), CONFIG_INFO.get(k, k)])
    if cfg_rows:
        sections.extend(["配置快照：", "", _md_table(["配置字段", "值", "中文意义"], cfg_rows), ""])
    rows = []
    for field, value in data["mean_metrics"].items():
        zh, unit, direction = FIELD_INFO.get(field, (field, "-", "-"))
        rows.append(
            [
                field,
                zh,
                unit,
                direction,
                _fmt(value),
                _fmt(data["std_metrics"].get(field)),
                _fmt(data.get(compare_key, {}).get(field)),
                _fmt_delta(data.get(compare_label, {}).get(field) if compare_label in data else None),
            ]
        )
    delta_col = compare_label
    sections.append(_md_table(["字段名", "中文意义", "单位/范围", "趋势", "均值", "Std", f"{compare_key}参考", f"{delta_col}"], rows))
    sections.append("")
    return "\n".join(sections)


def _render_ablation(title: str, report_path: Path) -> str:
    data = _load_json(report_path)
    metric_order = [
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
    rows = []
    for g in data["groups"]:
        row = [g["group"], g["description"], g["best_epoch"], _fmt(g["pred_gap"])]
        for field in metric_order:
            row.append(_fmt(g["metrics"].get(field)))
        rows.append(row)
    return "\n".join(
        [
            f"### {title}",
            "",
            f"- 报告：{_link(report_path.with_suffix('.md'))} / { _link(report_path) }",
            "",
            _md_table(
                ["组别", "说明", "best_epoch", "pred_gap", "peak_score", "pred_score", "peak_type_acc", "pred_type_acc", "pred_macro_f1", "pred_edge_f1", "pred_edge_f1_robust", "pred_count_mae", "pred_z_mae"],
                rows,
            ),
            "",
        ]
    )


def _render_real_afm_old(title: str, report_path: Path) -> str:
    data = _load_json(report_path)
    rows = []
    for variant in ("normal", "inverted", "oracle_best"):
        if variant not in data:
            continue
        overall = data[variant]["overall"] if "overall" in data[variant] else data[variant]
        rows.append(
            [
                variant,
                overall.get("num_cases", "-"),
                _fmt(overall.get("top1")),
                _fmt(overall.get("top3")),
                _fmt(overall.get("top5")),
                _fmt(overall.get("mrr")),
                _fmt(overall.get("mean_rank")),
                _fmt(overall.get("mean_pred_object_score")),
                _fmt(overall.get("mean_pred_object_type_acc")),
                _fmt(overall.get("mean_pred_object_edge_f1")),
                _fmt(overall.get("mean_pred_object_z_mae")),
                _fmt(overall.get("mean_pred_object_count_mae")),
            ]
        )
    return "\n".join(
        [
            f"### {title}",
            "",
            f"- 报告：{_link(report_path.with_suffix('.md'))} / { _link(report_path) }",
            "- 说明：这一版仅包含4个GT兼容case，后续已被11-case扩展版替代。",
            "",
            _md_table(["变体", "case数", "Top1", "Top3", "Top5", "MRR", "MeanRank", "MeanPredScore", "MeanTypeAcc", "MeanEdgeF1", "MeanZMae", "MeanCountMAE"], rows),
            "",
        ]
    )


def _render_real_afm_expanded(title: str, report_path: Path) -> str:
    data = _load_json(report_path)
    sections = [f"### {title}", "", f"- 报告：{_link(report_path.with_suffix('.md'))} / { _link(report_path) }", ""]
    summary_rows = []
    for variant in ("normal", "inverted"):
        block = data[variant]
        ret_all = block["retrieval_all_cases"]
        ret_gt = block["retrieval_gt_compatible_subset"]
        gt = block["gt_metric_subset"]
        summary_rows.append(
            [
                variant,
                ret_all["num_cases"],
                _fmt(ret_all["top1"]),
                _fmt(ret_all["top3"]),
                _fmt(ret_all["top5"]),
                _fmt(ret_all["mrr"]),
                _fmt(ret_all["mean_rank"]),
                ret_gt["num_cases"],
                _fmt(ret_gt["top1"]),
                _fmt(gt["mean_pred_object_score"]),
                _fmt(gt["mean_pred_object_type_acc"]),
                _fmt(gt["mean_pred_object_edge_f1"]),
                _fmt(gt["mean_pred_object_edge_f1_robust"]),
                _fmt(gt["mean_pred_object_z_mae"]),
            ]
        )
    sections.extend(
        [
            _md_table(
                ["变体", "全部case数", "AllTop1", "AllTop3", "AllTop5", "AllMRR", "AllMeanRank", "GT兼容数", "GT兼容Top1", "GT子集PredScore", "GT子集TypeAcc", "GT子集EdgeF1", "GT子集RobustEdgeF1", "GT子集ZMae"],
                summary_rows,
            ),
            "",
        ]
    )
    record_rows = []
    inverted_index = {r["case_id"]: r for r in data["inverted"]["records"]}
    for normal_r in data["normal"]["records"]:
        inv_r = inverted_index[normal_r["case_id"]]
        record_rows.append(
            [
                normal_r["case_id"],
                normal_r["molecule_label"],
                normal_r["tip"],
                "是" if normal_r["gt_structure_compatible"] else "否",
                normal_r["gt_rank"],
                inv_r["gt_rank"],
                "是" if normal_r["top1_hit"] else "否",
                "是" if inv_r["top1_hit"] else "否",
                _fmt(normal_r["pred_object_score"]),
                _fmt(inv_r["pred_object_score"]),
                _fmt(normal_r["pred_object_type_acc"]),
                _fmt(inv_r["pred_object_type_acc"]),
                _fmt(normal_r["pred_object_edge_f1"]),
                _fmt(inv_r["pred_object_edge_f1"]),
                _fmt(normal_r["pred_object_z_mae"]),
                _fmt(inv_r["pred_object_z_mae"]),
                _link(Path(normal_r["figure_path"]), "normal图"),
                _link(Path(inv_r["figure_path"]), "inverted图"),
            ]
        )
    sections.extend(
        [
            "逐case对比：",
            "",
            _md_table(
                ["Case", "标签", "tip", "GT兼容", "Normal Rank", "Inverted Rank", "Normal Top1", "Inverted Top1", "Normal PredScore", "Inverted PredScore", "Normal TypeAcc", "Inverted TypeAcc", "Normal EdgeF1", "Inverted EdgeF1", "Normal ZMae", "Inverted ZMae", "Normal图", "Inverted图"],
                record_rows,
            ),
            "",
        ]
    )
    return "\n".join(sections)


def _render_graph_smoke(title: str, fulltest_path: Path, retrieval_path: Path) -> str:
    full_data = _load_json(fulltest_path)
    ret_data = _load_json(retrieval_path)
    rows1 = []
    for field, value in full_data["mean_metrics"].items():
        zh, unit, direction = FIELD_INFO.get(field, (field, "-", "-"))
        rows1.append([field, zh, unit, direction, _fmt(value)])
    rows2 = [
        [field, REPORT_FIELD_INFO.get(field, FIELD_INFO.get(field, (field, "-", "-"))[0]), _fmt(value)]
        for field, value in ret_data["overall"].items()
    ]
    return "\n".join(
        [
            f"### {title}",
            "",
            f"- Full-test：{_link(fulltest_path.with_suffix('.md'))} / {_link(fulltest_path)}",
            f"- Retrieval：{_link(retrieval_path.with_suffix('.md'))} / {_link(retrieval_path)}",
            "",
            "Smoke Full-test：",
            "",
            _md_table(["字段名", "中文意义", "单位/范围", "趋势", "值"], rows1),
            "",
            "Smoke Retrieval overall：",
            "",
            _md_table(["字段名", "中文意义", "值"], rows2),
            "",
        ]
    )


def build_summary() -> str:
    roots = _collect_experiment_roots()
    model_index, report_index, visual_index = _scan_artifacts(roots)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    used_metric_fields = sorted(FIELD_INFO.keys())
    used_config_fields = sorted(CONFIG_INFO.keys())
    used_report_fields = sorted(REPORT_FIELD_INFO.keys())

    sections: list[str] = []
    sections.extend(
        [
            "# V19 与 V20 实验总索引与总结",
            "",
            f"- 生成时间：`{generated_at}`",
            f"- 统计范围：`{len([r for r in roots if r.name.startswith('v19')])}` 个 `v19` 实验目录，`{len([r for r in roots if r.name.startswith('v20')])}` 个 `v20` 实验目录",
            "- 数值说明：本文件默认保留 JSON 原始标量数值；比例类字段通常位于 `0-1` 区间，可按百分比理解。",
            "- 完整逐样本 JSON / CSV、checkpoint、可视化图片均在下方索引中给出可点击路径。",
            "",
            "## 一、文件索引",
            "",
            "### 1. 模型文件索引",
            "",
            model_index,
            "",
            "### 2. 报告与结果数据文件索引",
            "",
            report_index,
            "",
            "### 3. 可视化与样本图索引",
            "",
            visual_index,
            "",
            "## 二、V19 结果总表",
            "",
            _render_review_report(
                "V19 Full6h 训练复盘",
                EXPERIMENTS_ROOT / "v19_object_joint_full6h" / "review" / "reports" / "review_summary.json",
            ),
            _render_review_report(
                "V19 Full15 All 训练复盘",
                EXPERIMENTS_ROOT / "v19_object_joint_full15_all" / "review" / "reports" / "review_summary.json",
            ),
            _render_compare_report(
                "V19 Full6h vs Full15 对比",
                EXPERIMENTS_ROOT / "v19_object_joint_compare_full6h_vs_full15" / "reports" / "compare_summary.json",
            ),
            _render_visual15_report(
                "V19 Full15 All 固定15样本可视化总结",
                EXPERIMENTS_ROOT / "v19_object_joint_full15_all" / "visual_reports_object15" / "summary.json",
            ),
            "## 三、V20 主线结果总表",
            "",
            _render_visual15_report(
                "V20 Epoch10 固定15样本可视化总结",
                EXPERIMENTS_ROOT / "v20_object_joint_medium10_epoch10_visual15" / "visual_reports_object15" / "summary.json",
            ),
            _render_fulltest_report(
                "EXP-01 Full-test 对象级评估",
                EXPERIMENTS_ROOT / "v20_object_joint_medium10_exp01_fulltest" / "reports" / "fulltest_object_test.json",
            ),
            _render_retrieval_overall(
                "EXP-02 Full-test Retrieval",
                EXPERIMENTS_ROOT / "v20_object_joint_medium10_exp02_retrieval_fulltest" / "reports" / "retrieval_fulltest_test.json",
                include_strat=True,
            ),
            _render_gap_report(
                "EXP-03 Strict / Robust Gap Decomposition",
                EXPERIMENTS_ROOT / "v20_object_joint_medium10_exp03_gap_decompose" / "reports" / "gap_decomposition_test.json",
            ),
            _render_geom_report(
                "EXP-04 Geometry Diagnostics",
                EXPERIMENTS_ROOT / "v20_object_joint_medium10_exp04_geom_diagnostics" / "reports" / "geom_diagnostics_test.json",
            ),
            "## 四、V20 对照 Baseline 结果总表",
            "",
            _render_baseline_fulltest(
                "EXP-05 Preliminary Dense Baseline（早期版，已被 SUP-01 取代）",
                EXPERIMENTS_ROOT / "v20_dense_baseline_exp05" / "reports" / "dense_baseline_fulltest.json",
                compare_key="v20_reference_metrics",
                compare_label="dense_minus_v20",
            ),
            _render_baseline_fulltest(
                "SUP-01 Fair Dense Baseline Full-test",
                EXPERIMENTS_ROOT / "v20_dense_stage1_medium10_sup01_fulltest" / "reports" / "dense_baseline_fulltest.json",
                compare_key="v20_reference_metrics",
                compare_label="dense_minus_v20",
            ),
            _render_retrieval_overall(
                "SUP-01 Fair Dense Baseline Retrieval",
                EXPERIMENTS_ROOT / "v20_dense_stage1_medium10_sup01_retrieval" / "reports" / "dense_retrieval_fulltest_test.json",
                extra_compare_keys=["v20_reference_overall", "dense_minus_v20"],
            ),
            _render_graph_smoke(
                "SUP-02 Graph Baseline Smoke 预跑",
                EXPERIMENTS_ROOT / "v20_graph_baseline_smoke_eval" / "reports" / "graph_baseline_fulltest.json",
                EXPERIMENTS_ROOT / "v20_graph_baseline_smoke_retrieval" / "reports" / "graph_retrieval_fulltest_test.json",
            ),
            _render_baseline_fulltest(
                "SUP-02 Graph Baseline Full-test",
                EXPERIMENTS_ROOT / "v20_graph_baseline_medium10_sup02_fulltest" / "reports" / "graph_baseline_fulltest.json",
                compare_key="v20_reference_metrics",
                compare_label="graph_minus_v20",
            ),
            _render_retrieval_overall(
                "SUP-02 Graph Baseline Retrieval",
                EXPERIMENTS_ROOT / "v20_graph_baseline_medium10_sup02_retrieval" / "reports" / "graph_retrieval_fulltest_test.json",
                extra_compare_keys=["v20_reference_overall", "graph_minus_v20"],
            ),
            "## 五、V20 消融实验总表",
            "",
            _render_ablation(
                "EXP-06 Curriculum / Closure 消融",
                EXPERIMENTS_ROOT / "v20_ablate_curriculum_debug" / "reports" / "ablation_summary.json",
            ),
            _render_ablation(
                "EXP-07 Type / Edge / Z / Teacher 消融",
                EXPERIMENTS_ROOT / "v20_ablate_type_edge_debug" / "reports" / "ablation_summary.json",
            ),
            "## 六、V20 真实 AFM 结果总表",
            "",
            _render_real_afm_old(
                "SUP-03 真实 AFM 初版（4 case，已被扩展版取代）",
                EXPERIMENTS_ROOT / "v20_object_joint_medium10_sup03_real_afm" / "reports" / "sup03_real_afm_summary.json",
            ),
            _render_real_afm_expanded(
                "SUP-03 真实 AFM 扩展版（11 case）",
                EXPERIMENTS_ROOT / "v20_object_joint_medium10_sup03_real_afm_expanded" / "reports" / "sup03_real_afm_summary.json",
            ),
            _render_visual15_report(
                "SUP-03 Real11 可视化总结",
                EXPERIMENTS_ROOT / "v20_object_joint_medium10_sup03_visual11" / "visual_reports_real11" / "summary.json",
                is_real=True,
            ),
            "## 七、字段名中文释义",
            "",
            "### 1. 常见配置变量",
            "",
            _render_config_dict(used_config_fields),
            "",
            "### 2. 常见报告结构字段",
            "",
            _render_report_field_dict(used_report_fields),
            "",
            "### 3. 常见评估指标字段",
            "",
            _render_metric_dict(used_metric_fields),
            "",
        ]
    )
    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    OUTPUT_PATH.write_text(build_summary(), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
