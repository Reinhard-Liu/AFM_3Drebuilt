# V19 对象级联合训练 6 小时完整训练复盘报告

## 一、最佳结果
- 字段名 `best_epoch`：最佳轮次 = `12`
- 字段名 `peak_object_score`：预测中心闭环对象级总分 = `0.7171`
- 字段名 `gt_object_score`：真实中心条件对象级总分 = `0.9477`
- 字段名 `atom_center_score_r3`：原子中心命中分数 = `0.9990`
- 字段名 `peak_center_type_acc`：预测中心原子类型准确率 = `0.7257`
- 字段名 `peak_center_macro_f1`：预测中心原子类型宏平均F1 = `0.5623`
- 字段名 `peak_center_hetero_f1`：预测中心杂原子F1 = `0.5674`
- 字段名 `peak_center_edge_f1`：预测中心对象级边F1 = `0.7981`
- 字段名 `peak_center_shift_px`：预测中心平均偏移(像素) = `2.0049`
- 字段名 `atom_z_mae_r3`：z轴平均绝对误差 = `0.0987`
- 字段名 `typed_center_score_r3`：位置与类型同时正确的软分数 = `0.4070`
- 字段名 `atom_type_macro_f1_2d`：稠密2D类型图宏平均F1 = `0.1842`

## 二、训练趋势
- 字段名 `peak_object_score`：`0.4763` -> `0.7171`
- 字段名 `peak_center_type_acc`：`0.3231` -> `0.7257`
- 字段名 `peak_center_macro_f1`：`0.2106` -> `0.5623`
- 字段名 `peak_center_hetero_f1`：`0.2757` -> `0.5674`
- 字段名 `peak_center_edge_f1`：`0.6799` -> `0.7981`
- 字段名 `peak_center_shift_px`：`2.8938` -> `2.0049`
- 字段名 `atom_z_mae_r3`：`0.1442` -> `0.0987`

## 三、核心结论
- 原子中心分支已经基本学稳，`atom_center_score_r3` 已接近饱和。
- 在预测中心闭环条件下，原子类型、杂原子、对象级边和 z 轴误差都出现了持续改善。
- `gt_object_score` 仍明显高于 `peak_object_score`，说明当前主瓶颈仍然是“预测中心到对象级类型/边的迁移损失”，而不是类型头和边头本身完全不会。
- 稠密 `2D` 类型图仍然偏弱，说明后续仍应优先发展对象级类型/边头，而不是继续把主要希望放在稠密 `type map` 上。

## 四、代表样本复盘
- 最佳样本：样本编号 `dataset_index=505`；`peak_object_score=0.9538`，`peak_center_type_acc=1.0000`，`peak_center_macro_f1=1.0000`，`peak_center_hetero_f1=1.0000`，`peak_center_edge_f1=0.8571`，`peak_center_shift_px=1.74`，`atom_z_mae_r3=0.0011`
- 中位样本：样本编号 `dataset_index=87`；`peak_object_score=0.7227`，`peak_center_type_acc=0.7857`，`peak_center_macro_f1=0.5350`，`peak_center_hetero_f1=0.3333`，`peak_center_edge_f1=0.8493`，`peak_center_shift_px=1.62`，`atom_z_mae_r3=0.0108`
- 最差样本：样本编号 `dataset_index=151`；`peak_object_score=0.4246`，`peak_center_type_acc=0.4667`，`peak_center_macro_f1=0.2487`，`peak_center_hetero_f1=0.0000`，`peak_center_edge_f1=0.5306`，`peak_center_shift_px=2.77`，`atom_z_mae_r3=0.3655`

## 五、指标说明
- 字段名 `best_epoch`：最佳轮次
- 字段名 `peak_object_score`：预测中心闭环对象级总分，越高越好
- 字段名 `gt_object_score`：真实中心条件下的对象级总分，表示当前上限参考，越高越好
- 字段名 `atom_center_score_r3`：原子中心命中分数，统计真实中心半径3像素内的预测响应，越高越好
- 字段名 `peak_center_type_acc`：预测中心条件下的原子类型准确率，越高越好
- 字段名 `peak_center_macro_f1`：预测中心条件下的原子类型宏平均F1，越高越好
- 字段名 `peak_center_hetero_f1`：预测中心条件下的杂原子F1，越高越好
- 字段名 `peak_center_edge_f1`：预测中心条件下的对象级边F1，越高越好
- 字段名 `peak_center_shift_px`：预测中心相对真实中心的平均偏移，单位像素，越低越好
- 字段名 `atom_z_mae_r3`：真实中心附近的z轴平均绝对误差，越低越好
- 字段名 `typed_center_score_r3`：半径3像素内位置与类型同时正确的软分数，越高越好
- 字段名 `atom_type_macro_f1_2d`：稠密2D类型图的原子类型宏平均F1，越高越好
- 字段名 `atom_xy_mae`：稠密2D原子图平均绝对误差，越低越好
- 字段名 `z_map_mae`：稠密z图平均绝对误差，越低越好
- 字段名 `gt_center_type_acc`：真实中心条件下的原子类型准确率，越高越好
- 字段名 `gt_center_macro_f1`：真实中心条件下的原子类型宏平均F1，越高越好
- 字段名 `gt_center_hetero_f1`：真实中心条件下的杂原子F1，越高越好
- 字段名 `gt_center_edge_f1`：真实中心条件下的对象级边F1，越高越好
