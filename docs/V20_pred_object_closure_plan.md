# V20 纯预测对象闭环增强方案

## 一、方案定位

`V20` 不是推倒重来的大改版，也不是回到“稠密 2D 图 + 重型 3D 扩散”的旧路线。  
它是在当前 `V19 full15_all` 已经证明有效的对象级联合训练主线上，针对**纯预测对象闭环质量**做低风险、高收益增强。

核心目标只有一个：

**让模型在不依赖 GT center 的情况下，直接从 AFM 多高度堆栈中稳定恢复：**

- 原子中心
- 原子类型
- 对象级边
- 对象级 z / 3D 空间关系

并让这些结果与当前可视化中的 `Pred Object 2D / Pred Object 3D` 真正一致。

---

## 二、现状判断

### 1. 已经建立起来的能力

根据 [review_summary.md](/root/autodl-tmp/micro/experiments/v19_object_joint_full15_all/review/reports/review_summary.md)，当前最强结果为：

- 字段名 `peak_object_score`：预测中心闭环对象级总分 = `0.8016`
- 字段名 `peak_center_type_acc`：预测中心原子类型准确率 = `0.8185`
- 字段名 `peak_center_macro_f1`：预测中心原子类型宏平均F1 = `0.6564`
- 字段名 `peak_center_hetero_f1`：预测中心杂原子F1 = `0.8649`
- 字段名 `peak_center_edge_f1`：预测中心对象级边F1 = `0.8138`
- 字段名 `peak_center_shift_px`：预测中心平均偏移(像素) = `1.6854`
- 字段名 `atom_z_mae_r3`：z轴平均绝对误差 = `0.0893`

这说明：

1. 中心分支已经基本学稳。
2. 预测中心条件下的对象级类型、杂原子、边、z 均已进入可用区间。
3. 当前主问题已经不再是“模型完全不会”，而是“纯预测对象闭环下还有误差积累”。

### 2. 当前还没有真正解决的问题

从 `15` 样本对象级可视化和正式复盘可以看到，仍然有三类明显问题：

1. **纯预测对象的原子数和中心提案仍不够稳**
   - 预测中心图很强，但最终对象提案还依赖阈值与峰值筛选。
   - 一旦提案数量或局部峰位置不稳，后续类型和边会一起被拖坏。

2. **对象级类型在纯预测中心上仍弱于 GT-center 条件**
   - 当前密集 `type map` 指标仍偏低，说明稠密图不是主收益来源。
   - 当前真正强的是对象级类型头，但它还不够充分利用原始 AFM 局部多高度信息。

3. **3D / z 已经不差，但还缺少对象级约束和精修**
   - 现在的 `z_map` 回归已经有效，但更像“像素级高度图”。
   - 还没有把对象级边、局部平面性、相对深度关系真正编码进 3D 精修。

---

## 三、V20 的核心设计原则

### 原则 1：不做大改架构

不重新引入重型扩散主干，不回到“整团 3D 点云从噪声直接生成”的路线。  
继续保留当前对象级联合训练主线：

`AFM stack -> center -> object type/edge -> z -> object 3D`

### 原则 2：让评估口径与可视化口径完全一致

后续训练与分析优先使用“纯预测对象”指标，而不是只依赖 GT-center 附近的半闭环指标。

### 原则 3：先解决 2D 对象闭环，再做 3D 精修

当前最大瓶颈仍然是：
- 预测中心到对象级类型/边的迁移损失

所以 `3D/z` 的增强必须建立在对象级 2D 提案更稳的前提上。

---

## 四、V20 的四个核心改动

## 4.1 改动 A：中心提案从“阈值峰值”升级为“计数约束的对象解码”

### 目的

降低以下误差来源：

- 原子数偏多 / 偏少
- 中心局部重复提案
- 中心偏移后拖坏类型和边

### 具体设计

#### 新增计数头

在共享编码特征上新增一个对象计数头，预测：

- 字段名 `pred_object_count`
  - 中文含义：预测对象原子总数

训练使用：

- 字段名 `object_count_ce_loss`
  - 中文含义：对象数量分类损失

可选补充：

- 字段名 `object_count_mae_loss`
  - 中文含义：对象数量平均绝对误差损失

#### 推理阶段改成 count-conditioned top-K proposal

不再只依赖固定 `peak_threshold + NMS`。

改成：

1. 先由计数头预测对象数 `K`
2. 在 `center map` 上选取前 `K` 个最可信中心
3. 再做最小距离去重与局部修正

### 预期收益

- 降低 `Pred Object 2D` 中原子数错误
- 提高纯预测对象边和类型的上限
- 让 `CID top3` 检索更稳定

### 对应代码

- [v19_joint_model.py](/root/autodl-tmp/micro/src/models/v19_joint_model.py)
- [train_v19_object_joint.py](/root/autodl-tmp/micro/src/train_v19_object_joint.py)

---

## 4.2 改动 B：类型头升级为“预测中心上的局部 AFM 判别器”

### 目的

解决当前最明显的问题：

- `Pred Object 2D` 里原子类型仍不够稳
- 稠密 `type map` 偏弱，不适合作为主类型来源
- 共享 backbone 特征会损伤原始 AFM 类型信息

### 具体设计

#### 类型头输入从“共享特征单点采样”改成“双通道融合”

对每个 predicted-center，同时输入：

1. 共享 backbone / center 分支提取的对象级语义特征
2. 原始 `10` 层 AFM stack 在该中心附近裁出的局部 patch 特征

### 结构建议

类型头内部拆成三层：

1. **局部 AFM 编码器**
   - 对局部 `10 × h × w` patch 做轻量 3D Conv / 2.5D 编码

2. **对象上下文编码器**
   - 输入中心邻域、局部边上下文、center confidence

3. **融合分类器**
   - 输出细分类、粗分类、杂原子二值任务

### 保留现有辅助任务

继续保留：

- 粗分类头
- 杂原子二值头
- 类型教师蒸馏

但主任务从 dense `type_map` 转为 predicted-center 条件下的对象级类型分类。

### 预期收益

- 提升纯预测对象的原子类型准确率
- 提升宏平均 F1
- 提升杂原子稳定性
- 降低“图看起来类型不对，但半闭环指标还不错”的错位

### 对应代码

- [v19_center_type_head.py](/root/autodl-tmp/micro/src/models/v19_center_type_head.py)
- [train_v19_object_joint.py](/root/autodl-tmp/micro/src/train_v19_object_joint.py)

---

## 4.3 改动 C：边头从 pair 分类升级为轻量图细化

### 目的

解决当前对象级边“能用但不稳”的问题。

当前边 F1 已不低，但仍有：

- 中心轻微偏移后边判断掉点
- 类型变化后边判断受影响
- 局部 pair 独立判断缺少结构上下文

### 具体设计

#### 两阶段边预测

第一阶段：

- 用当前对象级边头得到初始边概率

第二阶段：

- 在 predicted-center 图上做 1 到 2 层轻量消息传递
- 让边预测参考：
  - 邻接中心的类型语义
  - 局部几何距离
  - center confidence
  - 初始边概率

### 预期收益

- 提升纯预测对象边 F1
- 提升 2D 分子图整体稳定性
- 给后续 3D 精修提供更可靠的拓扑输入

### 对应代码

- [v19_center_edge_head.py](/root/autodl-tmp/micro/src/models/v19_center_edge_head.py)
- [train_v19_object_joint.py](/root/autodl-tmp/micro/src/train_v19_object_joint.py)

---

## 4.4 改动 D：3D / z 从稠密图回归升级为对象级 z + 轻量几何精修

### 目的

提升：

- 3D 空间一致性
- 环平面性
- 相连原子相对深度关系
- 检索和可视化中的 3D 观感

### 具体设计

#### 对象级 z 头

在 predicted-center 条件下，对每个对象原子输出：

- 字段名 `pred_object_z`
  - 中文含义：对象级原子 z 预测值

训练增加：

- 字段名 `pred_object_z_mae_loss`
  - 中文含义：对象级 z 平均绝对误差损失

#### 相对深度约束

新增三个轻量几何损失：

- 字段名 `edge_delta_z_loss`
  - 中文含义：相连原子相对 z 差损失

- 字段名 `ring_planarity_loss`
  - 中文含义：环平面性损失

- 字段名 `local_normal_consistency_loss`
  - 中文含义：局部法向一致性损失

#### 轻量 3D refinement head

不回到重型扩散。  
只增加一个小的几何精修模块，对初始对象级 `3D` 做局部修正，目标是：

- 键长更合理
- 键角更合理
- 环更平
- 侧链扭转更自然

### 预期收益

- 降低对象级 z 误差
- 让 `Pred Object 3D` 更像分子，而不是“2D 对象外加 z”
- 提升检索质量和 3D 可视化质量

### 对应代码

- [v19_joint_model.py](/root/autodl-tmp/micro/src/models/v19_joint_model.py)
- [train_v19_object_joint.py](/root/autodl-tmp/micro/src/train_v19_object_joint.py)

---

## 五、V20 的评估体系重构

## 5.1 新增“纯预测对象”正式指标

这组指标必须和现在的可视化对象完全对齐。

### 纯预测对象 2D 指标

- 字段名 `pred_object_count_mae`
  - 中文含义：纯预测对象原子数平均绝对误差

- 字段名 `pred_object_type_acc`
  - 中文含义：纯预测对象原子类型准确率

- 字段名 `pred_object_macro_f1`
  - 中文含义：纯预测对象原子类型宏平均F1

- 字段名 `pred_object_hetero_f1`
  - 中文含义：纯预测对象杂原子F1

- 字段名 `pred_object_edge_f1`
  - 中文含义：纯预测对象边F1

- 字段名 `pred_object_graph_score`
  - 中文含义：纯预测对象图结构总分

### 纯预测对象 3D 指标

- 字段名 `pred_object_heavy_rmsd`
  - 中文含义：纯预测对象重原子三维均方根误差

- 字段名 `pred_object_z_mae`
  - 中文含义：纯预测对象 z 平均绝对误差

- 字段名 `bond_length_mae_3d`
  - 中文含义：三维键长平均绝对误差

- 字段名 `bond_angle_mae_3d`
  - 中文含义：三维键角平均绝对误差

- 字段名 `ring_planarity_mae`
  - 中文含义：环平面性平均偏差

### 新的核心总分

新增两个主指标：

- 字段名 `pred_object_score`
  - 中文含义：纯预测对象 2D 闭环总分

- 字段名 `pred_object_3d_score`
  - 中文含义：纯预测对象 3D 闭环总分

## 5.2 保留但降级的旧指标

以下指标保留，但不再作为唯一主排序依据：

- `gt_object_score`
- `peak_object_score`
- `typed_center_score_r3`
- `atom_type_macro_f1_2d`

原因：

- 它们仍有参考价值
- 但对“当前图里真实看到的 Pred Object 2D / 3D 到底好不好”解释力不够直接

---

## 六、V20 的训练策略

## 6.1 训练集使用策略

### 开发阶段

继续使用缩小训练集做快速迭代：

- `64k ~ 96k` 样本
- `4 ~ 6 epoch`

用于验证：

- center decoding
- predicted-center 局部类型头
- 轻量边 refinement
- 对象级 z 头

### 候选验证阶段

- `128k ~ 160k` 样本
- `6 ~ 8 epoch`

### 最终正式阶段

只对筛选出的最优 1 个版本使用全样本训练：

- 全样本
- 基于当前最强 checkpoint 热启动
- 再训练 `5 ~ 8 epoch`

### 判断

不建议每次小改动都上全样本 `15 epoch`。  
全样本训练应该只用于最终收敛，而不是结构搜索。

## 6.2 损失权重策略

建议分三阶段：

### 阶段 A：对象提案稳定化

重点：

- center loss
- count loss
- predicted-center type loss

### 阶段 B：图结构稳定化

提高：

- edge loss
- graph refinement loss

### 阶段 C：3D 精修

逐步提高：

- object z loss
- edge delta z loss
- ring planarity loss
- 轻量 refinement loss

---

## 七、V20 的验收标准

quick / medium 训练阶段，主看：

- `pred_object_count_mae`
- `pred_object_type_acc`
- `pred_object_macro_f1`
- `pred_object_hetero_f1`
- `pred_object_edge_f1`
- `pred_object_z_mae`
- `pred_object_score`
- `pred_object_3d_score`

预期方向：

1. 纯预测对象类型和边显著提升
2. 纯预测对象原子数误差下降
3. 3D / z 指标同步改善
4. `CID top3` 检索命中继续提高

---

## 八、优先级排序

### P0：必须先做

1. 纯预测对象正式指标
2. count-conditioned center decoding
3. predicted-center 局部 AFM 类型头

### P1：建议做

4. 边头轻量图细化
5. 对象级 z 头

### P2：在 P0/P1 验证有效后再做

6. 轻量 3D refinement
7. 更强的 3D 检索与几何评估

---

## 九、最终判断

`V20` 的策略不是“更大模型”或“更多全样本长训练”，而是：

**让当前已经有效的对象级主线，在纯预测对象闭环条件下真正对齐可视化、检索和 3D 重建目标。**

如果 `V20` 成功，最直接的变化会出现在三个地方：

1. `Pred Object 2D` 中原子数、原子类型和边关系更接近 GT
2. `Pred Object 3D` 的空间结构更自然、更稳定
3. `CID top3` 与肉眼判断的一致性进一步提高
