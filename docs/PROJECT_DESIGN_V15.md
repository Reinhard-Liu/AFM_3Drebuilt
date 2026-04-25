# AFM 3D 分子结构重建 — V15 完整项目设计方案

> 基于 V1-V14 全部历史改进 + 去除 SE(3) 等变性假设后的系统性重设计

---

## 第一部分：设计根基的转变

### 1.1 旧根基（V1-V14）

V1 方案将项目定位为"SE(3)-等变 3D 分子生成"，从通用分子生成领域（EDM、GeoDiff 等）借鉴了两个核心假设：

1. **SE(3) 等变性**：生成的 3D 结构应对旋转和平移不变
2. **旋转增强**：训练时随机旋转分子+AFM 图像，迫使模型学习旋转不变特征

这两个假设在通用分子生成（如药物设计）中完全合理——因为分子没有固定朝向。但 AFM 成像有**固定坐标系**：
- X/Y = 探针扫描平面，直接对应 AFM 图像像素坐标
- Z = 探针深度方向，对应 10 层深度切片的变化
- 分子平铺在基底表面上，朝向由实验确定

### 1.2 新根基（V15）

**核心原则：AFM 坐标系是固定的，不存在旋转模糊性。**

这一原则的推论：
- 不需要 SE(3) 等变性 → 可以直接使用空间特征
- 不需要旋转增强 → AFM 像素与分子坐标一一对应
- coord_head 可以接收 AFM 空间特征图 → 精确放置原子
- 评估指标不需要 Kabsch 旋转对齐 → raw RMSD 更有物理意义

### 1.3 旧设计中因 SE(3) 假设而做的设计决策清单

| 旧设计 | 来源版本 | 原因 | V15 处理 |
|--------|---------|------|----------|
| XY 旋转增强 | V1 | 提升旋转不变性 | **删除** |
| coord_head 不接 c_patches | V1-V14 | 空间特征与旋转冲突 | **修改**：加 cross-attention |
| SE3EquivariantDenoiser 命名 | V1 | 暗示等变设计 | **重命名** |
| Kabsch RMSD 作为主要评估 | V2 | 需要对齐才能比较 | **保留但降低权重**，raw RMSD 成为主指标 |
| 形状描述符（旋转不变） | V9 | 提供旋转不变的全局约束 | **保留**：仍有价值（编码分子整体形态） |
| 物理约束（键长/键角/平面性） | V1 | 本身旋转不变 | **保留并提前启用** |
| 环模板（标准 XY 平面） | V7 | Procrustes 对齐到标准朝向 | **保留**：AFM 坐标系下环确实近似平行于 XY 面 |

---

## 第二部分：V15 完整架构

### 2.1 系统总览

```
AFM 图像堆栈 (B, 10, 128, 128)
        ↓
┌────────────────────────────────┐
│     Video ViT 编码器            │
│  PatchEmbedding3D: (2,16,16)   │
│  8层 Transformer, dim=512      │
│  位置编码: learnable (321 tokens)│
└────────┬───────────┬───────────┘
         ↓           ↓
    c_global      c_patches
    (B, 512)     (B, 320, 512)
         ↓           ↓
┌────────┴───────────┴───────────────────┐
│         条件扩散去噪器                   │
│                                         │
│  x_t + global_bias → Transformer(6层)   │
│         ↓                               │
│         h (B, N, 256)                   │
│        ╱                  ╲             │
│   coord_head            type_head       │
│   + c_patches            + c_patches    │  ← V15: coord_head 也接空间特征
│   cross-attn             cross-attn     │
│       ↓                     ↓           │
│   eps_pred              type_logits     │
│   (B,N,3)               (B,N,10)       │
│                                         │
│  EDM*+γ: type_head 基于                 │
│  重建干净坐标 x_0_pred                   │
└─────────────────────────────────────────┘
         ↓
    ┌────┼────┬──────────────┐
    ↓    ↓    ↓              ↓
 AtomCount  Shape  Retrieval  Physics
   Head     Head    Head     Constraints
                            (Stage 2+)
```

### 2.2 与 V14 架构的差异

| 组件 | V14 | V15 | 改变原因 |
|------|-----|-----|---------|
| **数据增强** | XY 旋转（random） | **无旋转** | AFM 坐标系固定 |
| **coord_head 输入** | 仅 h（含 c_global bias） | **h + c_patches cross-attn** | 核心改动：让坐标预测看到空间信息 |
| **Stage 阈值** | 1-30/31-45/46-60 | **1-10/11-15/16-20** | 适配 20 epoch 训练 |
| **ring_bond_loss 阈值** | 0.18（2.16Å） | **0.15（1.80Å）** | 旧阈值过宽松，苯环键长 1.40Å |
| **类名** | SE3EquivariantDenoiser | **SpatialDenoiser**（别名） | 名实相符 |
| 其余 V14 改进 | — | **全部保留** | EDM*+γ/type_adapter/bottom_z_loss/ensemble |

### 2.3 模块详细规格

#### 2.3.1 Video ViT 编码器（不变）

| 参数 | 值 |
|------|-----|
| 输入 | (B, 10, 128, 128) AFM 深度切片堆栈 |
| 3D 卷积核 | (temporal=2, spatial=16, spatial=16) |
| Patch 数量 | 5 temporal × 8×8 spatial = 320 |
| Transformer | 8 层, dim=512, 8 heads |
| 位置编码 | learnable, (1, 321, 512) 含 CLS |
| 输出 c_global | (B, 512) — CLS token |
| 输出 c_patches | (B, 320, 512) — 保留空间位置的 patch 特征 |

**c_patches 的空间含义**：每个 patch token 对应 AFM 图像中一个 16×16 像素区域（8×8 网格）在某个深度范围（2 层一组，共 5 组）的特征。当 AFM 图像中某个位置有环形亮斑时，对应的 patch token 会编码"这里有环"的信息。

#### 2.3.2 去噪器（V15 核心改动）

**新增模块**（coord_head 的 cross-attention）：
```python
self.coord_patch_proj = nn.Linear(cond_dim=512, hidden_dim=256)  # 独立投影
self.coord_cross_attn = nn.MultiheadAttention(256, 8, dropout=0.1, batch_first=True)
self.coord_cross_norm = nn.LayerNorm(256)
```

**前向传播流程**：
```
输入: x_t(B,N,3), t(B,), c_global(B,512), c_patches(B,320,512), mask(B,N)

1. 特征提取
   h = coord_embed(x_t) + (time_emb + cond_proj(c_global) + shape_proj(shape))
   h = Transformer_blocks(h)  # 6 层自注意力

2. 坐标预测（V15 改动）
   p_coord = coord_patch_proj(c_patches)     # (B, 320, 256)
   h_norm = coord_cross_norm(h)              # (B, N, 256)
   cross_out = coord_cross_attn(Q=h_norm, K=p_coord, V=p_coord)
   h_for_coord = h + 0.1 * cross_out        # 残差权重 0.1
   eps_pred = coord_head(h_for_coord)        # (B, N, 3)

3. 类型预测（V14 EDM*+γ，保持不变）
   x_0_pred = reconstruct(x_t, eps_pred.detach())
   h_clean = coord_embed(x_0_pred) + global_bias
   h_for_type = h + h_clean
   h_for_type = h_for_type + 0.1 * type_cross_attn(h_for_type, c_patches)
   h_for_type = h_for_type + type_adapter(h_for_type)
   type_logits = type_head(h_for_type)       # (B, N, 10)
```

**为什么 coord 和 type 使用独立的 patch 投影层**：
- coord_patch_proj 学习"在哪里有原子"的空间注意力模式
- type 的 patch_proj 学习"这个位置的 AFM 对比度对应什么元素"
- 两者关注 c_patches 的不同方面，共享投影层会互相干扰

#### 2.3.3 训练损失函数

| 损失 | 公式 | 权重 | 生效阶段 | 说明 |
|------|------|------|----------|------|
| coord_loss | MSE(eps_pred, noise) × [1,1,2] Z 加权 | 1.0 | 全程 | Z 方向 2 倍权重 |
| type_loss | CE + SNR 软加权 + 类别平衡 | 1.0 | 全程 | EDM*+γ 干净坐标输入 |
| shape_loss | MSE(惯性张量特征值) | 0.5 | 全程 | 旋转不变 |
| ring_loss | ReLU(ring_bond_dist - **0.15**) + 0.1×count_diff | 0.3 | t < 500 | V15 收紧阈值 |
| bottom_z_loss | MSE(底部 30% 原子 Z 坐标) | 0.5 | t < 500 | M6 |
| count_loss | CE + MSE 双分支 | 1.0 | 全程 | |
| shape_pred_loss | MSE(pred_shape, gt_shape) | 0.5 | 全程 | |
| retrieval_loss | InfoNCE 对比 | 0.01 | 全程 | |
| constraint_loss | 键长 + 0.5×键角 + 0.3×平面性 | 0.1 | **Stage 2+（Ep11+）** | V15 提前启用 |

**V15 stage 阈值**：
```
Stage 1 (Ep 1-10):  基础训练，不加物理约束
Stage 2 (Ep 11-15): 键长/键角/平面性约束启用
Stage 3 (Ep 16-20): 底部原子 Z 深度加权
```

#### 2.3.4 推理流程

```
AFM 图像 → Video ViT → c_global + c_patches
    ↓
AtomCountHead → N 个原子
ShapeHead → 预测形状描述符
    ↓
DDIM 采样 (50 步):
  每步:
    1. 去噪器前向: eps_pred(含 coord cross-attn), type_logits
    2. 重建 x_0_pred = (x_t - √(1-ᾱ)·eps) / √ᾱ
    3. 物理引导 (t < 50%):
       - vdW 斥力
       - 连通性拉回 (0.6 比例, 2 次迭代)
    4. 环约束 (t < 40%):
       - 自动检测 5/6 元环
       - 平面投影 (blend 0.4)
       - 环间 Z 反重叠
    5. DDIM 步进
    ↓
输出坐标 + denoiser type_logits
    ↓
(可选) GNN TypeClassifier → gnn_type_logits
    ↓
Ensemble 融合: α·denoiser_probs + (1-α)·gnn_probs (α=0.6)
    ↓
化学类型修正 (化合价规则)
    ↓
最终输出: 坐标 + 类型 + 预测原子数 + 候选 CID
```

---

## 第三部分：需要删除的设计

### 3.1 XY 旋转增强

**位置**: `config.json` 第 8 行, `dataset.py` 第 331-335 行

**删除原因**: AFM 坐标系固定，旋转增强打破了像素-坐标的直接对应关系，迫使 coord_head 放弃空间特征。

**操作**: `config.json` 中 `"augment_rotation": false`。dataset.py 中的增强函数保留但不触发。

### 3.2 SE3EquivariantDenoiser 命名

**位置**: `diffusion.py` 第 67 行

**删除原因**: 类名暗示 SE(3) 等变性，但实际代码是标准 Transformer，无任何等变操作（无球谐函数、无 Wigner-D 矩阵、无相对位置编码）。V1-V14 的所有"等变性"实际来自数据增强，而非架构。

**操作**: 在类定义后添加别名 `SpatialDenoiser = SE3EquivariantDenoiser`。

### 3.3 旧的 Stage 阈值（30/45/60）

**位置**: `train.py` 第 267-279 行

**删除原因**: 为 60 epoch 设计，但 V13/V14 只训练 20 epoch，导致物理约束和 Z 深度加权永远不启用。

**操作**: 调整为 10/15/20 分界。

---

## 第四部分：需要修改的设计

### 4.1 coord_head 信息通路（核心修改）

**现状**: coord_head 只接收 c_global（512 维全局向量），AFM 空间特征 c_patches 只传给 type_head。

**修改**: 为 coord_head 添加对 c_patches 的 cross-attention，让每个原子 token 能查询 320 个 AFM patch 的空间信息。

**设计细节**:
- 使用独立的投影层 `coord_patch_proj`（与 type 的 `patch_proj` 分开）
- 残差权重 0.1（与 type_head 的 cross-attn 一致）
- 新增参数 ~329K（<1% 模型总量）
- 可从 V14 checkpoint warm-start（strict=False）

### 4.2 ring_bond_loss 阈值

**现状**: `F.relu(dist - 0.18)` — 阈值 0.18 归一化 = 2.16Å。苯环 C-C = 1.40Å，阈值允许 54% 拉伸无惩罚。在低噪声（t<100）时模型预测键距通常 < 0.16，全部无梯度。

**修改**: `F.relu(dist - 0.15)` — 阈值 0.15 = 1.80Å，允许 ~28% 余量。低噪声样本中部分键距超过 0.15 会产生有效梯度信号。

**只改 loss 阈值**，检测阈值（adjacency detection）保持 0.18 不变。

### 4.3 评估指标的解读方式

去掉旋转后，指标含义发生变化：

| 指标 | V14 解读 | V15 解读 |
|------|---------|---------|
| compute_rmsd (无 Kabsch) | 受旋转影响，需 Kabsch 对齐才公平 | **直接有效**：固定坐标系下 raw RMSD 就是真实误差 |
| Kabsch RMSD | 标准对齐后 RMSD | 仍可计算，但**不再是必需指标** |
| bottom_atom_recall | Z 坐标判断受旋转干扰 | **更准确**：Z 有确定的物理含义（深度） |
| conditional_type_acc | 欧式匹配可能因旋转偏差不准 | **更准确**：坐标直接对应 |
| PMI shape | 本身旋转不变 | 不变 |
| Ring preservation | 本身旋转不变 | 不变 |

**不需要修改评估代码**——指标计算逻辑不变，但结果更有意义。

---

## 第五部分：需要新增的设计

### 5.1 coord_head 的 AFM 空间 cross-attention（本文 2.3.2 已描述）

这是 V15 唯一的新架构组件。核心思想：让负责放置原子坐标的 coord_head 能够"看到" AFM 图像中的空间结构。

**预期效果**:
- 环位置精度提升：AFM Z-slice 0/2 中清晰可见的六元环亮斑位置信息直接传给 coord_head
- 整体 RMSD 下降：空间特征提供精确的 XY 坐标引导
- 环内原子分布均匀化：cross-attention 可以关注环心区域的多个 patch，获取更精细的空间定位

### 5.2 不需要新增的设计

以下方向经评估后决定**不在 V15 中引入**：

| 方向 | 不引入原因 |
|------|----------|
| 2D 环检测网络 | coord cross-attention 已能传递空间信息，不需要额外的 CNN 检测器 |
| 相对坐标编码 | 去掉旋转后绝对坐标更直接，相对编码反而丢信息 |
| 直接回归坐标（不用扩散） | 扩散模型的优势在于多步精炼，V14 RMSD=0.166 已证明有效 |
| 更大的 Transformer | 参数量不是瓶颈（48.6M 已足够），信息通路才是 |
| GradNorm 动态梯度平衡 | EDM*+γ 已缓解跷跷板效应，增加复杂度收益不确定 |

---

## 第六部分：数据与训练配置

### 6.1 数据集（不变）
- 数据源: QUAM-AFM K-1, 205,227 含环分子
- 过滤: corrugation >= 1.25Å + require_ring = true
- 分辨率: 128×128, 10 层深度切片
- 归一化: 坐标 ÷ 12.0 → [-1, 1]
- 划分: 100K train / 1K val / 1K test

### 6.2 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 20 | |
| batch_size | 128 | |
| lr | 1e-4 → 1e-6 | CosineAnnealing |
| optimizer | AdamW, weight_decay=1e-5 | |
| 混合精度 | BF16 | |
| augment_rotation | **false** | V15 核心改动 |
| save_dir | experiments/v15/checkpoints | |
| eval_ddim_steps | 50 | |
| eval_samples_per_epoch | 200 (quick) / 1000 (full, 每 5 ep) | |

### 6.3 三阶段训练策略（调整后）

| 阶段 | Epoch | 损失 | 学习率 | 目标 |
|------|-------|------|--------|------|
| Stage 1 基础 | 1-10 | coord + type + shape + ring + bottom_z + count + retrieval | 1e-4 → ~3e-5 | 学习基本重建 |
| Stage 2 约束 | 11-15 | +键长/键角/平面性约束 | ~3e-5 → ~1e-5 | 物理有效性 |
| Stage 3 底部 | 16-20 | +底部原子 3× Z 权重 | ~1e-5 → 1e-6 | 遮挡区精度 |

---

## 第七部分：评估体系

### 7.1 指标体系（保留）

| 指标 | 计算方式 | V15 权重 |
|------|---------|---------|
| RMSD ↓ | 匈牙利匹配后 L2 距离（无 Kabsch） | 核心 |
| Kabsch RMSD ↓ | 匈牙利 + SVD 旋转对齐 | 参考 |
| Type Match ↑ | 匹配原子的类型准确率 | 核心 |
| Cond Type Acc ↑ | 距离阈值内匹配原子的类型准确率 | 核心 |
| Coulomb ↑ | 库仑矩阵特征值余弦相似度 | 辅助 |
| PMI Shape ↑ | 惯性张量特征值比率相似度 | 辅助 |
| Ring Preserve ↑ | 环数量+尺寸分布匹配 | 核心 |
| Bond Valid ↑ | 合理键长比例 | 辅助 |
| Bottom Recall ↑ | 底部 30% 原子重建召回率 | 辅助 |
| Composite ↑ | 0.30×rmsd_score + 0.20×bottom + 0.15×bond + 0.15×ring + 0.10×count + 0.10×struct_sim | 综合 |

### 7.2 混淆矩阵分析（保留）
10×10 元素混淆矩阵 + Top-5 混淆对 + 每元素准确率。

### 7.3 Corrugation 分组评估（保留）
动态 P33/P67 分位数分组，每组报告 RMSD/Type/PMI/Ring。

---

## 第八部分：全流程执行计划

### Phase 1: 扩散模型训练（20 epoch, ~5h）
```bash
python3 -m src.train --config config.json
```

### Phase 1 Eval: 测试集评估 + 混淆矩阵（~15min）
```bash
python3 -m src.eval_phase1 --checkpoint best_diffusion.pt --num_samples 200
```

### Phase 2: GNN 训练（~1.5h）
```bash
python3 -m src.train_gnn --diffusion_checkpoint best_diffusion.pt --epochs 30
```

### Phase 3: Recycling + Ensemble 评估（~30min）
```bash
python3 -m src.eval_recycling --ensemble_alpha 0.6 --num_samples 256
```

### Phase 4: 可视化（Diff + Ensemble, ~30min）
```bash
python3 -m src.visualize_val --checkpoint best_diffusion.pt --num_samples 15
python3 -m src.visualize_val --checkpoint best_diffusion.pt --gnn_checkpoint best_gnn.pt --num_samples 15
```

---

## 第九部分：预期效果与风险

### 9.1 预期改善

| 指标 | V14 最佳 | V15 预期 | 改善来源 |
|------|----------|---------|---------|
| RMSD | 0.166 | **<0.14** | coord cross-attention 提供空间定位 |
| Ring Preserve | 0.939 | **>0.95** | 收紧 ring_loss + Stage 2 物理约束启用 |
| Bond Valid | 0.745 | **>0.80** | Stage 2 键长/键角约束 |
| TypeMatch | 0.580 | **>0.58** | 保持（非本版重点） |
| Bottom Recall | 0.139 | **>0.15** | Stage 3 Z 深度加权启用 |
| Composite | 0.664 | **>0.70** | 全面提升 |

### 9.2 风险评估

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| coord cross-attention 初期不稳定 | 中 | RMSD 波动 | 0.1 残差权重 + warm-start |
| Stage 2 约束过早导致优化冲突 | 低 | 训练不稳定 | 约束权重仅 0.1 |
| 去掉旋转后数据多样性下降 | 低 | 轻微过拟合 | 数据量 100K 足够 |
| ring_loss 阈值 0.15 过严 | 低 | ring_loss 过大 | 可调回 0.16 |

---

## 附录 A：完整文件修改清单

| 文件 | 修改内容 | 类型 |
|------|---------|------|
| `config.json` | augment_rotation→false, save_dir→v15 | 配置 |
| `src/models/diffusion.py` __init__ | 新增 coord_patch_proj/coord_cross_attn/coord_cross_norm | 架构 |
| `src/models/diffusion.py` forward() | coord_head 前加 cross-attention | 架构 |
| `src/models/diffusion.py` compute_loss() | ring_bond threshold 0.18→0.15 | 训练 |
| `src/models/diffusion.py` 类名 | 添加 SpatialDenoiser 别名 | 命名 |
| `src/train.py` get_training_stage() | 阈值 30/45→10/15 | 训练 |

## 附录 B：保留的 V14 改进（不修改）

| 改进 | 文件 | 说明 |
|------|------|------|
| EDM*+γ 分离噪声调度 | diffusion.py forward() | type_head 基于重建干净坐标 |
| type_adapter 层 | diffusion.py __init__ | 独立类型特征变换 |
| bottom_z_loss | diffusion.py compute_loss() | 底部原子 Z 坐标 MSE |
| SNR 软加权 | diffusion.py compute_loss() | 替代 t<500 硬阈值 |
| Composite 修复 | train.py validate_epoch() | ring_preservation 参与计算 |
| 混淆矩阵 | eval_phase1.py | 元素级错误分析 |
| Ensemble 推理 | eval_recycling.py | denoiser+GNN 加权融合 |
| 化学类型修正 | eval_phase1.py | 化合价规则后处理 |

## 附录 C：V5b-V15 设计演化脉络

```
V1:  基础框架（Video ViT + DDPM + SE(3)假设）
V2:  指标体系（6维结构相似度 + Composite）
V3:  Focal Loss + 形状约束
V5b: 关键 Bug 修复（噪声范围/类型loss/类别平衡）
V7:  分子级指标（Formula Sim + Ring Preserve）
V8:  AFM cross-attention 给 type_head（空间特征首次使用）
V9:  形状条件注入（惯性张量特征值）
V12: GNN 解耦（消除曝光偏差）
V13: 环数据集过滤 + Z轴加权 + 环预测loss
V14: EDM*+γ（缓解跷跷板）+ type_adapter + bottom_z_loss + Ensemble
     发现: coord_head 看不到空间信息 / SE(3)假设不必要
V15: 去除SE(3)假设 + coord cross-attention + 物理约束提前启用
     ← 你在这里
```
