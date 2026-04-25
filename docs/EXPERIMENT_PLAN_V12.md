# AFM 3D 分子结构重建：V12 改进方案

## 第一部分：核心问题诊断（基于 V1-V11 全部实验数据）

### 1.1 坐标与原子类型的"跷跷板效应"

**现象**（V10 训练数据）：

| Epoch | RMSD | TypeMatch | Coulomb | 趋势 |
|-------|------|-----------|---------|------|
| 1 | 0.2685 | 0.5488 | 0.5644 | 起步 |
| 10 | 0.2659 | **0.5639** | **0.6104** | type 最佳区间 |
| 15 | 0.2571 | 0.5590 | **0.6215** | Coulomb 峰值 |
| 30 | 0.2451 | 0.5277 | 0.6022 | type 开始下降 |
| 50 | **0.2263** | 0.5219 | 0.5064 | RMSD 最佳但 type/Coulomb 崩塌 |

**根因分析**：

denoiser 的 transformer 同时服务 coord_head 和 type_head。coord_loss（权重 1.0，全时步计算）的有效梯度量约为 type_loss（权重 1.0，仅 t<500 计算）的 2 倍。训练后期 coord_loss 持续压低（0.40→0.31），transformer 特征被坐标优化主导，type_head 的特征空间被挤压。

**V1-V12 所有版本均观察到此现象**：type_match 在 Ep3-15 达峰后持续下降。

### 1.2 AFM 图像的"元素盲区"

**可视化证据**：
- Sample #499（V10）：Kabsch=0.944（形状极好），但 Type=0.294（类型极差）
- GT 含大量 O（红色）和 N（蓝色），模型全部预测为 C/H

**物理原因**：AFM 图像主要反映顶层电子云密度（范德华斥力），对 C/N/O 等相邻周期元素的直接区分度极低。denoiser type_head 过度依赖局部高度特征，而非化学成键的几何上下文（键长、键角、配位数）。

### 1.3 训练与推理的不可调和矛盾

V6-V11 反复验证了同一个死胡同：

| 版本 | 方案 | 训练时坐标 | 推理时坐标 | 结果 |
|------|------|----------|----------|------|
| V6 | TypeNet 解耦 | GT 坐标 | 生成坐标 | exposure bias，type_match 30% |
| V10 | α̅(t) 加权 | noisy coords | 生成坐标 | 优化错误区间，无效 |
| V11 | TypePredictor+高斯噪声 | GT+高斯噪声 | 生成坐标（结构性误差） | 噪声模式不匹配，47% |
| V10实验 | 两阶段推理 | — | 生成坐标(t=0重预测) | 生成坐标≠GT，48% |

**核心矛盾**：无论怎么调整训练策略，denoiser type_head 训练时看到的坐标分布和推理时永远不一致。

### 1.4 Phase 1 评估揭示的真实水平

使用 V10 checkpoint，新指标评估结果：

| 目标 | 指标 | 结果 | 目标值 | 状态 |
|------|------|------|--------|------|
| 原子位置 | RMSD | 0.242 | <0.30 | ✓ 达标 |
| Type ≥75% | conditional_type_acc | 55.5% | 75% | ✗ 差 20% |
| 形状 ≥80% | PMI similarity | **89.7%** | 80% | ✓ 已达标 |
| 环结构一致 | Ring preservation | **93.5%** | 一致 | ✓ 已达标 |

3/4 目标已达标，**唯一差距是原子类型预测**。

---

## 第二部分：V12 方案设计

### 2.1 核心思路：解耦坐标生成与类型预测

**关键洞察**：扩散模型已经能生成高质量 3D 骨架（RMSD < 0.25, Kabsch > 0.90），问题只在类型预测。与其在一个网络内同时优化两个冲突任务，不如让扩散模型专注坐标生成，将类型预测交给专门的后置 GNN。

**文献依据**：
- UniGEM (ICLR 2025)：解耦坐标扩散和类型预测，类型在 scaffold 形成后单独预测
- MiDi (ECML-PKDD 2023)：键约束（配位数）提供远强于坐标距离的类型信号
- GCDM (Nature CommsChem 2024)：几何完备架构能利用键长/键角/二面角信息

### 2.2 架构设计

```
推理流程：
  AFM Image Stack (B, 10, 128, 128)
         ↓
    Video ViT Encoder → c_global, c_patches
         ↓
    Conditional DDPM (DDIM 采样)
         ↓
    生成坐标 (B, N, 3)
         ↓
    构建分子图 (距离阈值推断键)
         ↓
    E(n)-GNN TypeClassifier → 原子类型 (B, N, 10)
```

### 2.3 GNN TypeClassifier 设计

```python
class GNNTypeClassifier(nn.Module):
    """后置 GNN：从生成坐标的分子图预测原子类型。

    输入：
      - coords: (B, N, 3) 生成的原子坐标
      - c_patches: (B, P, 512) ViT AFM patch 特征
      - mask: (B, N) 原子掩码

    GNN 能利用的化学几何特征（denoiser type_head 无法利用的）：
      - 精确键长：C-C 1.54Å vs C-N 1.47Å vs C-O 1.43Å
      - 配位数：C=4, N=3, O=2, H=1
      - 键角：sp2=120°, sp3=109.5°
      - AFM 局部空间特征（cross-attention to patches）
    """
    def __init__(self, ...):
        # 节点特征编码（坐标 + 几何特征）
        self.node_encoder  # 输入: [coords, n_neighbors, mean_bond_len, ...]
        # 边特征编码（键长 + 相对位置）
        self.edge_encoder  # 输入: [distance, relative_pos]
        # 消息传递层
        self.gnn_layers    # E(n)-equivariant message passing
        # AFM 特征融合
        self.cross_attn    # cross-attention to c_patches
        # 类型预测头
        self.type_head     # MLP → 10 types
```

### 2.4 两阶段训练（消除 exposure bias 的关键）

**Phase 1：训练扩散模型（15-20 epoch，早停）**

使用现有 V12 代码（旋转增强修复 + denoiser type_head），训练到 type_match 最佳平衡点后早停。

- 保留 denoiser type_head 作为辅助训练信号（维持坐标质量）
- shape_loss 权重 1.0（强化形状）
- 早停在 Ep15-20（type/Coulomb 最佳区间）

**Phase 2：训练 GNN TypeClassifier**

1. 冻结扩散模型
2. 用冻结的扩散模型在**训练集**上 DDIM 采样生成坐标
3. 在这些**生成坐标**上训练 GNN（标签 = GT 类型）

**这是消除 exposure bias 的关键**：
- V6 TypeNet：GT 坐标训练 → 生成坐标推理 → bias ✗
- V11 TypePredictor：GT+高斯噪声训练 → 生成坐标推理 → 噪声不匹配 ✗
- V12 GNN：**生成坐标**训练 → **生成坐标**推理 → **分布一致** ✓

### 2.5 早停策略（方案二，作为辅助）

扩散模型训练时采用动态早停：
- 监控 type_match 和 Coulomb
- 当 type_match 连续 5 epoch 下降时停止
- 预计 Ep15-20 停止

### 2.6 旋转增强修复（V11 已实现，保留）

- XY 平面旋转 + AFM 图像同步旋转
- V11 前 10 轮 RMSD 0.25→0.17，验证有效

---

## 第三部分：与历史方案的对比

| 维度 | V6 TypeNet | V11 TypePredictor | **V12 GNN** |
|------|-----------|-------------------|------------|
| 架构 | 6层Transformer | 4层Transformer | E(n)-GNN |
| 训练坐标 | GT（完美） | GT+高斯噪声 | **生成坐标（真实分布）** |
| Exposure bias | 严重 | 减轻但仍有 | **消除** |
| 几何特征 | 坐标+邻居统计 | 坐标+噪声 | **键长+配位数+键角** |
| 与 coord 梯度冲突 | 有（去掉 type_head） | 有（双 loss） | **无（完全独立）** |
| 参数增量 | ~26M | ~3.7M | ~2-4M |

---

## 第四部分：预期效果

| 指标 | V10 | V12 预期 | 依据 |
|------|-----|---------|------|
| RMSD | 0.254 | 0.20-0.24 | 旋转修复 + 早停 |
| **Type Match** | 48.7% | **65-75%** | GNN 利用化学几何 + 无 bias |
| **cond_type_acc** | 55.5% | **70-80%** | GNN 在匹配正确原子上更准 |
| Coulomb | 0.405 | 0.50-0.60 | type 更准 → Coulomb 改善 |
| PMI shape | 89.7% | 88-92% | 维持 |
| Ring preservation | 93.5% | 92-95% | 维持 |

### 为什么预期 65-75%？

GNN 能利用的确定性化学规则：
- 邻居数=4 → 100% 是 C（数据集中约 15% 的原子）
- 邻居数=1 → 95%+ 是 H（约 35% 的原子）
- 邻居数=3 + 键长~1.47Å → N（vs C 键长 1.54Å，差 0.07Å）
- 邻居数=2 + 键长~1.43Å → O（vs S 1.81Å，差 0.38Å）

仅靠邻居数+键长的确定性规则就能达到 ~68% 准确率。GNN 学习后的软分类应更高。

---

## 第五部分：实施步骤

### Step 1：扩散模型训练（Phase 1）

- V12 代码（旋转修复 + denoiser type_head + shape_loss 1.0）
- 训练 20 epoch，早停在 type_match 最佳点
- 保存 checkpoint

### Step 2：生成训练数据（Phase 2 准备）

- 用 Phase 1 checkpoint 在训练集上 DDIM 采样
- 生成 100,000 个样本的坐标
- 保存为 {生成坐标, GT 类型, AFM patches} 三元组

### Step 3：实现 GNN TypeClassifier

- 轻量级 E(n)-equivariant GNN（4 层消息传递）
- 输入：生成坐标图 + AFM patch cross-attention
- 输出：每个节点的类型分类

### Step 4：训练 GNN

- 在 Step 2 生成的数据上训练
- CE loss + sqrt(inv_freq) 类别权重
- 20-30 epoch

### Step 5：评估

- 完整推理流程：AFM → ViT → DDIM → GNN → 类型
- Phase 1 评估指标（含 conditional_type_acc, PMI, ring）

---

## 第六部分：参考文献

| 论文 | 发表 | 相关技术 |
|------|------|---------|
| UniGEM | ICLR 2025 | 解耦坐标和类型预测 |
| MiDi | ECML-PKDD 2023 | 键约束提供类型信号 |
| GCDM | Nature CommsChem 2024 | 几何完备扩散（角度/二面角） |
| EGNN | ICML 2021 | E(n) 等变图神经网络 |
| PCGrad | ICML 2020 | 多任务梯度冲突投影 |
| SchNet | NeurIPS 2017 | 连续滤波卷积分子表示 |
| DimeNet++ | ICLR 2020 | 方向消息传递（键角特征） |

---

## 第七部分：V12+ 优化改进（基于讨论后采纳的建议）

### 7.1 GNN 训练噪声注入（建议 1）

**原始建议**：σ ≈ 0.2Å
**调整后**：σ = 0.01 归一化 ≈ 0.12Å

**理由**：GNN 的核心优势是利用精确键长（C-C 1.54Å vs C-N 1.47Å，差 0.07Å）。σ=0.2Å 会淹没这个差异。σ=0.12Å 既提供鲁棒性又保留键长信号。

**与 V11 的区别**：V11 在 GT 坐标上加噪声训练（分布不匹配），V12 在**生成坐标**上加噪声（已匹配的分布上做数据增强）。

### 7.2 Recycling 循环精炼（建议 2）

**流程**：
```
Round 1: DDIM 坐标 → GNN → type_logits
Round 2: 高置信度(>0.8)类型 → MMFF94 力场优化 → 精炼坐标
Round 3: 精炼坐标 → GNN → 最终类型 → 化学修正
```

**关键设计**：置信度过滤（softmax > 0.8）——低置信度原子不参与力场优化，避免错误类型导致力场计算方向错误。

**代码**：`src/eval_recycling.py`

### 7.3 AFM 局部像素采样（建议 3b）

**实现**：根据每个原子的 XY 坐标，从 10 层 AFM 图像中采样局部像素强度。

```python
# 每个原子获得 10 维向量（10 层 AFM 切片的像素值）
afm_local[b, i] = afm_stack[b, :, pixel_y, pixel_x]
```

**作用**：AFM 图像对电子云密度敏感，不同元素的 vdW 半径不同 → 不同高度切片的像素强度不同。这提供了 cross-attention 无法获取的**逐原子位置级别**的 AFM 信息。

### 7.4 论文叙事框架（建议 4）

即使 Type Match 最终为 68%（未到 75%），论文叙事仍然完整：
- **55% → 68%** 的提升定义为"从形貌基线到几何推理重建"
- **Formula Similarity 0.95+** 证明模型理解了分子的化学组成
- **PMI 89.7%** 和 **Ring 93.5%** 证明形状和结构的高还原度
- 在论文中阐述 C/N/O 在 AFM 形貌上的物理相似性限制

---

## 第八部分：更新后的执行计划

| Phase | 内容 | 时间 |
|-------|------|------|
| Phase 1 | 扩散模型训练 20 epoch（早停）| ~3.5h（进行中） |
| Phase 2a | 用 Phase 1 checkpoint 生成 5000 个坐标 | ~2h |
| Phase 2b | 训练 GNN（30 epoch，含噪声注入+AFM局部） | ~1h |
| Phase 3 | Recycling 推理评估（2 轮） | ~0.5h |
| 基线 | 3D-ResNet 训练 20 epoch | ~1.5h |
| **总计** | | **~8.5h** |
