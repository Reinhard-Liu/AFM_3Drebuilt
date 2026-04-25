# V20 EXP-06 Curriculum + Closure Ablation

## 一、实验组
- `full`：V20 full setting
- `gt_only`：Only GT object supervision; disable peak/pred closure and curriculum
- `peak_only`：Only peak-centered object supervision; remove GT and pred closure losses
- `pred_only`：Only predicted-center closure supervision; remove GT and peak branches
- `no_curriculum`：Keep all branches but remove center/loss schedules
- `no_pred_closure`：Disable predicted-center closure losses and pred consistency

## 二、主表
| 组别 | best_epoch | peak_object | pred_object | peak->pred gap | peak_type_acc | pred_type_acc | pred_macro_f1 | pred_edge_f1 | robust_edge_f1 | pred_count_mae | pred_z_mae |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 3 | 0.7803 | 0.7132 | 0.0671 | 0.7822 | 0.7278 | 0.5407 | 0.7135 | 0.0000 | 2.9141 | 0.1075 |
| gt_only | 3 | 0.7387 | 0.6967 | 0.0420 | 0.7038 | 0.7053 | 0.5055 | 0.8194 | 0.9876 | 3.0859 | 0.1421 |
| peak_only | 3 | 0.7978 | 0.7232 | 0.0747 | 0.8159 | 0.7341 | 0.5262 | 0.7257 | 0.9495 | 3.2031 | 0.1054 |
| pred_only | 2 | 0.7645 | 0.7412 | 0.0233 | 0.7493 | 0.7593 | 0.5617 | 0.7239 | 0.9676 | 3.3750 | 0.0908 |
| no_curriculum | 3 | 0.7804 | 0.7079 | 0.0724 | 0.7832 | 0.7172 | 0.5017 | 0.6695 | 0.9104 | 3.7969 | 0.0836 |
| no_pred_closure | 2 | 0.7631 | 0.7271 | 0.0360 | 0.7369 | 0.7258 | 0.5433 | 0.7700 | 0.9686 | 3.4844 | 0.1133 |

## 三、相对 Full 的变化
| 组别 | d_pred_object | d_pred_type_acc | d_pred_macro_f1 | d_pred_edge_f1 | d_pred_count_mae | d_gap |
|---|---:|---:|---:|---:|---:|---:|
| gt_only | -0.0165 | -0.0225 | -0.0352 | +0.1059 | +0.1719 | -0.0251 |
| peak_only | +0.0100 | +0.0063 | -0.0145 | +0.0122 | +0.2891 | +0.0076 |
| pred_only | +0.0280 | +0.0316 | +0.0210 | +0.0104 | +0.4609 | -0.0438 |
| no_curriculum | -0.0053 | -0.0105 | -0.0390 | -0.0440 | +0.8828 | +0.0053 |
| no_pred_closure | +0.0139 | -0.0020 | +0.0026 | +0.0565 | +0.5703 | -0.0311 |

## 四、判断
- `pred_object_score` 与 `peak->pred gap` 一起看，判断哪一组最能缩小闭环断层。
- `pred_object_type_acc`、`pred_object_macro_f1`、`pred_object_edge_f1` 一起看，判断 typed graph 闭环最依赖哪种监督。
- 如果 `no_pred_closure` 明显变差，说明 predicted-center closure 不是可有可无的附加项，而是主贡献之一。
- 如果 `no_curriculum` 明显变差，说明不是单纯多加损失，而是课程式接入时机本身起作用。
