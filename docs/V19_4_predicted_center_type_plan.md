# V19.4 预测中心类型强化方案

## 目标

在已经基本学稳的对象级中心分支基础上，主攻 `predicted-center` 条件下的对象级原子类型分支，缩小：

- `gt_object_score`
- `peak_object_score`

之间的差距，重点提升：

- `peak_center_type_acc`：预测中心上的对象级原子类型准确率
- `peak_center_macro_f1`：预测中心上的对象级原子类型宏平均 F1
- `peak_center_hetero_f1`：预测中心上的对象级杂原子 F1

## 现状判断

完整训练已经证明三点：

1. `atom_center_score_r3` 已接近上限，中心检测不是主瓶颈。
2. `peak_center_edge_f1` 仍在上升，但已经明显好于早期版本，边分支不是当前第一优先级。
3. 主差距已经收缩到：**预测中心一旦有偏移，类型头的稳定性不够**。

因此下一版不再优先大改中心头或边头，而是直接增强类型头对预测中心偏移的鲁棒性。

## 设计调整

### 1. 类型头从“单点取样”改成“中心附近局部统计”

旧版类型头主要在给定坐标点做一次局部采样，这对 `predicted-center` 偏移比较敏感。  
新版改成同时看：

- 共享特征图在中心附近的 `center / mean / max`
- AFM 多高度堆栈在中心附近的 `center / mean / max`
- 中心响应图在中心附近的 `center / mean / max`

这样即使预测中心有 1 到 3 像素偏移，类型头也还能利用附近信息继续判断。

### 2. 加入粗分类辅助任务

原子类型细分类前，先学一个粗分类：

- `0`：碳氢类
- `1`：常见杂原子类（N/O/S/P）
- `2`：卤素类（F/Cl/Br/I）

作用：

- 减少“全部塌成 C/H”的趋势
- 先让模型学会大的元素组，再做细分类

### 3. 加入杂原子二值辅助任务

增加“是否为杂原子”这一支二值头，直接强化：

- `peak_center_hetero_f1`

它的目标不是取代细分类，而是帮助模型先分清：

- `H/C`
- 非 `H/C`

### 4. 训练重点从 dense type map 进一步转向对象级类型

当前稠密 `type map` 已证明不是主收益来源，因此下一版配置里会：

- 降低 dense `type_map` 的辅助权重
- 提高 `predicted-center` 对象级类型损失权重
- 保留类型教师蒸馏，但让它更多服务对象级类型头

## 对应代码调整

### [src/models/v19_center_type_head.py](/root/autodl-tmp/micro/src/models/v19_center_type_head.py)

具体修改：

- 从单点采样改为局部 `center / mean / max` 统计
- 新增粗分类头
- 新增杂原子二值头
- 细分类损失改成更偏难样本的 focal 形式

### [src/train_v19_object_joint.py](/root/autodl-tmp/micro/src/train_v19_object_joint.py)

具体修改：

- `GT center` 和 `predicted-center` 两条类型分支都显式接入 `center_map`
- 新增类型头超参数入口：
  - `type_hidden_dim`
  - `type_coarse_lambda`
  - `type_hetero_lambda`
  - `type_focal_gamma`
  - `type_afm_radius_px`
  - `type_feat_radius_px`
  - `type_center_radius_px`

### 新配置

新增一个 predicted-center type-focus quick config，用于验证：

- 提升是否主要发生在 `peak_center_type_acc`
- 宏平均 F1 和杂原子 F1 是否同步改善
- 是否不会明显伤到 `peak_center_edge_f1` 和 `atom_z_mae_r3`

## 这版的验收标准

quick debug 先看这些变化：

- `peak_center_type_acc` 明显高于当前 full6h 的 `0.7257`
- `peak_center_macro_f1` 明显高于当前 full6h 的 `0.5623`
- `peak_center_hetero_f1` 明显高于当前 full6h 的 `0.5674`
- `peak_center_edge_f1` 不明显下降
- `atom_z_mae_r3` 不明显恶化

如果 quick debug 趋势正确，再决定是否上新一轮长训练。
