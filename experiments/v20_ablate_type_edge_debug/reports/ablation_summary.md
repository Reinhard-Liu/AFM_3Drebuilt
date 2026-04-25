# V20 EXP-07 Type / Edge Structure Ablation

## 一、实验组
- `full`：V20 full setting
- `old_type_head`：Replace current center-conditioned type head with legacy GNN-style type head
- `no_edge_head`：Remove object edge head and edge-related supervision
- `no_z_head`：Disable z supervision and strip z input from object heads
- `no_teacher_consistency`：Disable teacher distillation and type consistency losses

## 二、主表
| 组别 | best_epoch | peak_object | pred_object | peak->pred gap | peak_type_acc | pred_type_acc | pred_macro_f1 | pred_edge_f1 | robust_edge_f1 | pred_count_mae | pred_z_mae |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 2 | 0.7857 | 0.7223 | 0.0634 | 0.7586 | 0.7352 | 0.5488 | 0.7087 | 0.9618 | 3.9531 | 0.0862 |
| old_type_head | 3 | 0.6695 | 0.6206 | 0.0489 | 0.6738 | 0.6079 | 0.3557 | 0.8128 | 0.9896 | 3.3125 | 0.1350 |
| no_edge_head | 3 | 0.6205 | 0.5761 | 0.0444 | 0.7844 | 0.7256 | 0.5228 | 0.0000 | 0.0000 | 3.1406 | 0.1033 |
| no_z_head | 3 | 0.7830 | 0.7164 | 0.0665 | 0.7642 | 0.7224 | 0.5130 | 0.6930 | 0.9493 | 3.0391 | 0.1113 |
| no_teacher_consistency | 3 | 0.7929 | 0.7371 | 0.0558 | 0.8083 | 0.7583 | 0.5412 | 0.7031 | 0.9441 | 3.2969 | 0.0984 |

## 三、相对 Full 的变化
| 组别 | d_pred_object | d_pred_type_acc | d_pred_macro_f1 | d_pred_edge_f1 | d_robust_edge_f1 | d_pred_count_mae | d_pred_z_mae | d_gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old_type_head | -0.1017 | -0.1274 | -0.1931 | +0.1041 | +0.0277 | -0.6406 | +0.0489 | -0.0145 |
| no_edge_head | -0.1462 | -0.0096 | -0.0260 | -0.7087 | -0.9618 | -0.8125 | +0.0171 | -0.0189 |
| no_z_head | -0.0059 | -0.0128 | -0.0358 | -0.0157 | -0.0125 | -0.9141 | +0.0251 | +0.0032 |
| no_teacher_consistency | +0.0148 | +0.0230 | -0.0076 | -0.0056 | -0.0177 | -0.6562 | +0.0122 | -0.0076 |

## 四、判断
- `old_type_head` 用来判断当前中心条件类型头是否真的优于旧式图类型头。
- `no_edge_head` 用来判断对象级 edge 监督是否在帮助 typed graph 闭环，还是只影响 strict edge 指标本身。
- `no_z_head` 用来判断 z 分支是否在反向帮助 type/edge 闭环。
- `no_teacher_consistency` 用来判断 teacher + consistency 是否只是锦上添花，还是当前 v20 的关键稳定项。
