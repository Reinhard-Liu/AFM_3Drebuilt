# 文档索引(Documentation Index)

按主题分类,索引 `docs/` 下所有文档。

---

## 1. 项目设计与架构

| 文档 | 说明 |
|---|---|
| [`PROJECT_DESIGN_V15.md`](PROJECT_DESIGN_V15.md) | V15 项目架构定稿(对象级解码起点) |
| [`PROJECT_DESIGN_V16.md`](PROJECT_DESIGN_V16.md) | V16 环约束扩展 |
| [`V16b项目方案.md`](V16b项目方案.md) | V16b 项目方案 |
| [`V16c修复路线图.md`](V16c修复路线图.md) | V16c 修复路线图 |
| [`V18_two_stage_eval_and_architecture_plan.md`](V18_two_stage_eval_and_architecture_plan.md) | V18 两阶段评估与架构方案 |

## 2. V19 对象级联合学习方案

| 文档 | 说明 |
|---|---|
| [`V19_1_joint_afm_to_2d3d_plan.md`](V19_1_joint_afm_to_2d3d_plan.md) | V19-1 AFM→2D/3D 联合预测方案 |
| [`V19_2_object_joint_plan.md`](V19_2_object_joint_plan.md) | V19-2 对象级联合方案(核心创新) |
| [`V19_2d_then_z_plan.md`](V19_2d_then_z_plan.md) | V19 先 2D 后 Z 路径 |
| [`V19_3_overnight_full_joint_plan.md`](V19_3_overnight_full_joint_plan.md) | V19-3 完整长训方案 |
| [`V19_4_predicted_center_type_plan.md`](V19_4_predicted_center_type_plan.md) | V19-4 预测中心-类型联合方案 |

## 3. V20 前沿探索方案

| 文档 | 说明 |
|---|---|
| [`V20_pred_object_closure_plan.md`](V20_pred_object_closure_plan.md) | V20 预测对象闭环方案 |

## 4. 实验总索引与历史复盘

| 文档 | 说明 |
|---|---|
| **[`V19_V20实验总索引与总结.md`](V19_V20实验总索引与总结.md)** | **【最关键】V19/V20 全量实验索引,EXP-01~04 + SUP-01~03 全部入口** |
| [`V1-V6_RETROSPECTIVE.md`](V1-V6_RETROSPECTIVE.md) | V1–V6 早期版本复盘 |
| [`EXPERIMENT_PLAN_V1.md`](EXPERIMENT_PLAN_V1.md) ~ [`V14`](EXPERIMENT_PLAN_V14.md) | V1–V14 各阶段实验计划 |

## 5. V17 评估与检索

| 文档 | 说明 |
|---|---|
| [`V17_eval_retrieval_redesign.md`](V17_eval_retrieval_redesign.md) | V17 评估/检索体系重设计 |
| [`V17_ring_scaffold_plan.md`](V17_ring_scaffold_plan.md) | V17 环骨架方案 |

## 6. 使用指南(`docs/guides/`)

| 文档 | 说明 |
|---|---|
| [`guides/QUICK_START.md`](guides/QUICK_START.md) | 老版快速上手(参考新版 [`/QUICKSTART.md`](../QUICKSTART.md)) |
| [`guides/RDKIT_INSTALLATION.md`](guides/RDKIT_INSTALLATION.md) | RDKit 安装与版本验证清单(2025.09.6 已测通) |
| [`guides/COMMAND_COMPARISON.md`](guides/COMMAND_COMPARISON.md) | `run.sh` 各命令对照(含已知 bug 说明) |
| [`guides/RUN_SH_EXPLANATION.md`](guides/RUN_SH_EXPLANATION.md) | `run.sh` 各步骤逐段解析 |
| [`guides/RUN_SH_VERIFICATION.md`](guides/RUN_SH_VERIFICATION.md) | `run.sh` 验证流程 |
| [`guides/VISUALIZATION_GUIDE.md`](guides/VISUALIZATION_GUIDE.md) | 可视化使用说明 |
| [`guides/COMPLETE_OUTPUT_GUIDE.md`](guides/COMPLETE_OUTPUT_GUIDE.md) | 输出文件清单与解读 |
| [`guides/METRICS_FIX.md`](guides/METRICS_FIX.md) | 指标计算修复说明 |
| [`guides/PROJECT_STATUS.md`](guides/PROJECT_STATUS.md) | 项目状态快照 |
| [`guides/FINAL_VERIFICATION_SUMMARY.md`](guides/FINAL_VERIFICATION_SUMMARY.md) | 最终验证总结 |

## 7. 专题分析(`docs/analysis/`)

15 份分析报告,覆盖性能、设计取舍、检索方案与改进建议:

| 文档 | 说明 |
|---|---|
| [`analysis/ATOM_COUNT_ACCURACY_ANALYSIS.md`](analysis/ATOM_COUNT_ACCURACY_ANALYSIS.md) | 原子数量预测准确率分析 |
| [`analysis/COUNT_ACCURACY_EXPLANATION.md`](analysis/COUNT_ACCURACY_EXPLANATION.md) | 计数准确率解读 |
| [`analysis/BEFORE_AFTER_COMPARISON.md`](analysis/BEFORE_AFTER_COMPARISON.md) | 改进前后对比 |
| [`analysis/CURVES_EXPLANATION.md`](analysis/CURVES_EXPLANATION.md) | 训练曲线解读 |
| [`analysis/DESIGN_CRITIQUE.md`](analysis/DESIGN_CRITIQUE.md) | 设计批判性回顾 |
| [`analysis/EXPECTED_OUTPUT_FILES.md`](analysis/EXPECTED_OUTPUT_FILES.md) | 预期输出文件清单 |
| [`analysis/TRAINING_OUTPUT_FILES.md`](analysis/TRAINING_OUTPUT_FILES.md) | 训练输出文件说明 |
| [`analysis/MODIFICATION_COMPLETE_SUMMARY.md`](analysis/MODIFICATION_COMPLETE_SUMMARY.md) | 修改完成总结 |
| [`analysis/MODIFICATION_SUMMARY.md`](analysis/MODIFICATION_SUMMARY.md) | 修改摘要 |
| [`analysis/PROJECT_AUDIT_REPORT.md`](analysis/PROJECT_AUDIT_REPORT.md) | 项目审计报告 |
| [`analysis/stage_config_examples.md`](analysis/stage_config_examples.md) | 各阶段配置示例 |
| [`analysis/环检测问题解决方案.md`](analysis/环检测问题解决方案.md) | 环检测问题解决方案 |
| [`analysis/基于3D结构检索的方案分析.md`](analysis/基于3D结构检索的方案分析.md) | 3D 结构检索方案分析 |
| [`analysis/如何查看Top5相似分子.md`](analysis/如何查看Top5相似分子.md) | Top-5 相似分子查看说明 |
| [`analysis/样本0相似度分析报告.md`](analysis/样本0相似度分析报告.md) | 样本 0 相似度分析 |
| [`analysis/项目改进方案.md`](analysis/项目改进方案.md) | 项目改进方案汇总 |

---

## 推荐阅读顺序

**新读者** — [QUICKSTART](../QUICKSTART.md) → [README](../README.md) → [`V19_V20实验总索引与总结.md`](V19_V20实验总索引与总结.md)

**深入设计** — [`PROJECT_DESIGN_V15.md`](PROJECT_DESIGN_V15.md) → [`V19_2_object_joint_plan.md`](V19_2_object_joint_plan.md) → [`V20_pred_object_closure_plan.md`](V20_pred_object_closure_plan.md) → [`analysis/DESIGN_CRITIQUE.md`](analysis/DESIGN_CRITIQUE.md)

**复现实验** — [QUICKSTART](../QUICKSTART.md) → [README § 九](../README.md#九各模块运行指令) → [`guides/COMMAND_COMPARISON.md`](guides/COMMAND_COMPARISON.md)
