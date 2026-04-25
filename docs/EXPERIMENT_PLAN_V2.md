# 实验方案 V2：3D 结构相似度评估 + 预测质量提升

## 概述

V2 对 V1 做了四项改进，目标是提升预测分子的原子数、原子类型、结构质量，并引入基于文献的 3D 结构相似度评估替代 Top-5 CID 检索。

**当前配置**（`config.json`）：
- 数据集：K-1（68,555 分子），`max_samples=100000`，`min_corrugation=1.25`
- 模型：Video ViT (depth=8, dim=512) + Conditional DDPM (1000步)
- 训练：60 epochs, batch_size=64, lr=1e-4, CosineAnnealing → 1e-6
- GPU：RTX 4080 SUPER (32GB)，预计总训练时间 ~26-28 小时

---

## 改进一：3D 结构相似度评估

> 文件：`src/utils/metrics.py`

### 动机

V1 使用 InfoNCE 对比学习做 Top-5 CID 检索，只能判断「最像哪个已知分子」，无法量化预测结构与真实结构的实际差距。V2 改为直接计算预测 vs 真实的 3D 结构多维相似度。

### 实现细节

新增 `compute_structure_similarity()` 函数及 5 个辅助函数，包含 6 个子指标：

| 子指标 | 函数/算法 | 文献来源 | 输入 |
|--------|----------|----------|------|
| **Kabsch RMSD Score** | `_kabsch_rmsd()` — 匈牙利匹配确定原子对应关系，SVD 求最优旋转矩阵，对齐后计算 RMSD，转换为 max(0, 1-RMSD/2.0) | EDM, GeoLDM, GeoDiff | 坐标 |
| **Atom-type Accuracy** | 匈牙利匹配后逐原子检查类型是否一致 | EDM (atom stability) | 坐标+类型 |
| **Coulomb Matrix Similarity** | `_coulomb_matrix_eigenvalues()` — 构建 Coulomb 矩阵 C_ij=Z_i*Z_j/r_ij，计算特征值向量（旋转不变），L2 距离归一化 | QM9 benchmark | 坐标+原子序数 |
| **Pairwise Distance JS Divergence** | `_pairwise_distance_histogram()` + `_js_divergence()` — 计算所有原子对距离分布直方图（50 bins, [0,2.0]），Jensen-Shannon 散度 | MolDiff, EDM | 坐标 |
| **Valence Validity** | `_valence_validity()` — 基于共价半径查找表（`_COVALENT_RADII`）+ 容差 0.0333 推断键，对照最大化合价表（`_MAX_VALENCE`）检查合理性 | EDM (molecule stability) | 坐标+类型 |
| **Atom Count Similarity** | 1 - \|N_pred - N_gt\| / max(N_pred, N_gt) | 通用 | 原子数 |

**综合评分公式**：
```
overall = 0.25 × kabsch_score + 0.20 × type_accuracy + 0.20 × coulomb_sim
        + 0.15 × (1 - JS_div) + 0.10 × valence_validity + 0.10 × count_sim
```

**常量表**：
- `_ATOMIC_NUMBERS`：类型索引 → 原子序数映射 (H=1, C=6, N=7, O=8, F=9, S=16, P=15, Cl=17, Br=35, I=53)
- `_MAX_VALENCE`：H≤1, C≤4, N≤3, O≤2, F≤1, S≤6, P≤5, Cl≤1, Br≤1, I≤1
- `_COVALENT_RADII`：归一化空间共价半径（Å/12.0）

### 调用链路

```
evaluate_generation()
  → compute_structure_similarity(pred_coords, gt_coords, pred_types, gt_types, mask)
  → 返回 6 个子指标 + overall_similarity
  → 传入 compute_composite_score(..., structure_similarity=overall)

save_predictions()
  → 为每个样本保存 7 维 structure_similarity 字典
```

`compute_composite_score()` 签名更新：`cid_accuracy` 参数改为 `structure_similarity`。

---

## 改进二：AtomCountHead 架构升级

> 文件：`src/models/prediction_heads.py`

### 动机

V1 的 AtomCountHead 仅 2 层 MLP (512→256→85) 做 85 类分类，能力不足。瓶颈不在深度，而在梯度流和训练策略。

### V1 vs V2 架构对比

```
V1:                                    V2:
┌─────────────────┐                    ┌─────────────────┐
│  c (512-dim)    │                    │  c (512-dim)    │
└────────┬────────┘                    └───┬─────────┬───┘
         │                                 │         │
    ┌────┴────┐                      shared_mlp   res_proj
    │         │                      (512→256    (512→256)
    ▼         ▼                       +GELU       linear)
cls_branch  reg_branch                +Dropout)    │
(512→256    (512→256                       │       │
 →85)        →1)                           └──+────┘
                                              │ ← 残差相加
                                         LayerNorm(256)
                                              │
                                      ┌───────┴───────┐
                                      ▼               ▼
                                 cls_branch      reg_branch
                                 (256→256→128    (256→128→1)
                                  →85)
```

### 关键改进

| 项目 | V1 | V2 |
|------|-----|-----|
| 分类有效深度 | 2 层 | **4 层**（shared 1 + cls 3） |
| 回归有效深度 | 2 层 | **3 层**（shared 1 + reg 2） |
| 梯度流 | 无跳接 | **残差连接** (`shared_mlp(c) + res_proj(c)`) + **LayerNorm** |
| 正则化 | 无 | **Dropout(0.1)** × 2 处 |
| 分类损失 | `F.cross_entropy` | `F.cross_entropy(label_smoothing=0.1)` |
| 推理策略 | `argmax(cls_logits) + 1` | **融合**：`round(0.7 × cls_pred + 0.3 × reg_value)` |
| 总损失权重 | 0.5 | **1.0** |

### 代码结构

```python
class AtomCountHead(nn.Module):
    shared_mlp: Linear(512,256) → GELU → Dropout(0.1)
    res_proj:   Linear(512,256)
    shared_norm: LayerNorm(256)
    cls_branch: Linear(256,256) → GELU → Dropout(0.1) → Linear(256,128) → GELU → Linear(128,85)
    reg_branch: Linear(256,128) → GELU → Dropout(0.1) → Linear(128,1)

    _shared_features(c):  return shared_norm(shared_mlp(c) + res_proj(c))
    forward(c):           h = _shared_features(c); return cls_branch(h), reg_branch(h)
    predict(c):           fused = 0.7 * argmax(cls) + 0.3 * clamp(reg); return round(fused)
    compute_loss(c, n):   cls_loss(label_smoothing=0.1) + 0.5 * smooth_l1_loss
```

---

## 改进三：原子类型损失优化

> 文件：`src/models/diffusion.py` — `ConditionalDDPM.compute_loss()`

### 动机

V1 的 type_loss 有两个问题：
1. 权重仅 0.1，被 coord_loss 完全主导（V1 训练 60 epoch type_loss 几乎不下降：1.259→1.188）
2. 无类别平衡，H/C 占数据集 80%+，稀有元素（F, S, P, Cl, Br, I）被忽略

### 实现

```python
# 每个 batch 动态计算逆频率权重
valid_types = types_flat[valid]
counts = torch.bincount(valid_types, minlength=num_classes).float().clamp(min=1.0)
class_weight = (valid_types.numel() / (num_classes * counts)).clamp(max=10.0)

type_loss = F.cross_entropy(logits[valid], valid_types, weight=class_weight)
```

- **动态权重**：每个 batch 根据实际元素频率计算，不需要预统计全局分布
- **上限钳制**：`clamp(max=10.0)` 防止极稀有元素权重爆炸
- **总损失权重**：0.1 → **0.3**（在 `train.py` 的 `AFM3DReconModel.forward()` 中）

**注意**：`diffusion.py` 内部的 `compute_loss()` 仍保留 `loss = coord_loss + 0.1 * type_loss`（局部组合），外层 `train.py` 中 type_loss 以 0.3 的系数参与总损失。

---

## 改进四：降低 InfoNCE 检索权重（保留正则化）

> 文件：`src/train.py` — `AFM3DReconModel.forward()`

### 动机

完全去掉 retrieval_loss 会丧失对 ViT encoder 的对比学习正则化压力，导致 encoder 表征缺乏全局分子辨识能力。

### 实现

retrieval_loss 系数从 0.05 降为 **0.01**：
- 足够低不干扰主要损失的优化
- 足够高为 encoder 提供「区分不同分子」的正则化信号
- `mol_embeddings` (Embedding(100000, 128)) 仍参与梯度更新

---

## 总损失公式

```python
loss = coord_loss
     + 0.3 * type_loss        # V1: 0.1
     + 1.0 * count_loss        # V1: 0.5
     + 0.01 * retrieval_loss   # V1: 0.05
     + 0.1 * constraint_loss   # Stage 2+ 不变
```

| 损失项 | V1 权重 | V2 权重 | 变化 |
|--------|---------|---------|------|
| coord_loss | 1.0 | 1.0 | — |
| type_loss | 0.1 | **0.3** | ×3 |
| count_loss | 0.5 | **1.0** | ×2 |
| retrieval_loss | 0.05 | **0.01** | ÷5 |
| constraint_loss | 0.1 | 0.1 | — |

---

## 三阶段训练策略（不变）

| 阶段 | Epoch | 额外特性 | 学习率范围 |
|------|-------|---------|-----------|
| Stage 1 基础训练 | 1-30 | coord + type + count + retrieval | 1e-4 → ~3e-5 |
| Stage 2 约束训练 | 31-45 | + constraint_loss (键长/键角/环) | ~3e-5 → ~1e-5 |
| Stage 3 底部聚焦 | 46-60 | + z_depth_weighting (底部原子 3× 权重) | ~1e-5 → 1e-6 |

学习率调度：`CosineAnnealingLR(T_max=60, eta_min=1e-6)`

Early stopping：epoch ≥ 60 且 RMSD < 1.0 时停止（当前设置下不太可能触发）。

---

## 评估指标体系

### 每 Epoch 评估（`evaluate_generation()`）

对验证集 1000 个样本，使用预测原子数（`use_gt_count=False`）端到端评估：

| 指标 | 函数 | 说明 |
|------|------|------|
| RMSD (mean ± std) | `compute_rmsd()` | 匈牙利匹配后均方根偏差 |
| Bottom Recall | `compute_bottom_atom_recall()` | 底部 30% 原子召回率 |
| Bottom RMSD | `compute_bottom_atom_rmsd()` | 底部原子专用 RMSD |
| Bond Validity | `compute_bond_validity()` | 化学键长合理性 |
| Count Accuracy (exact + MAE) | `compute_atom_count_accuracy()` | 原子数精确匹配率 |
| **Structure Similarity** (V2 新增) | `compute_structure_similarity()` | 6 维综合相似度 |
| — Kabsch Score | `_kabsch_rmsd()` | SVD 对齐后坐标相似度 |
| — Type Match Rate | 匈牙利匹配 | 原子类型正确率 |
| — Coulomb Similarity | `_coulomb_matrix_eigenvalues()` | 旋转不变电荷分布 |
| — Distance JS Divergence | `_pairwise_distance_histogram()` + `_js_divergence()` | 距离分布差异 |
| — Valence Validity | `_valence_validity()` | 化合价合理性 |
| Composite Score | `compute_composite_score()` | 加权综合评分 |

### Composite Score 公式

```
composite = 0.30 × max(0, 1 - RMSD/2.0)
          + 0.20 × bottom_recall
          + 0.15 × bond_validity
          + 0.15 × ring_preservation (TODO, 当前 = 0)
          + 0.10 × count_exact_match
          + 0.10 × structure_similarity
```

### 预测输出（`save_predictions()`）

每个样本保存：
```json
{
  "sample_id": 0,
  "coords": [[x, y, z], ...],
  "atom_types": [1, 0, 1, ...],
  "n_atoms_pred": 30,
  "n_atoms_gt": 32,
  "structure_similarity": {
    "overall": 0.65,
    "kabsch_score": 0.72,
    "type_match_rate": 0.80,
    "coulomb_similarity": 0.55,
    "distance_js_divergence": 0.12,
    "valence_validity": 0.90,
    "count_similarity": 0.94
  }
}
```

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/utils/metrics.py` | 新增 `_kabsch_rmsd()`, `_coulomb_matrix_eigenvalues()`, `_pairwise_distance_histogram()`, `_js_divergence()`, `_valence_validity()`, `compute_structure_similarity()`；更新 `compute_composite_score()` 签名 |
| `src/models/prediction_heads.py` | `AtomCountHead` 重构：共享特征层+残差+LayerNorm+Label Smoothing+推理融合 |
| `src/models/diffusion.py` | `ConditionalDDPM.compute_loss()` 中 type_loss 添加逆频率 class_weight |
| `src/train.py` | 总损失权重调整；`evaluate_generation()` 新增 structure_similarity 计算和输出；`save_predictions()` 改为保存结构相似度分数；评估打印和历史记录更新 |
| `src/quick_test.py` | `compute_composite_score()` 调用参数修正 |

---

## V1 基线对比（60 Epoch 训练完成）

| 指标 | V1 Epoch 60 | V1 最佳 Epoch | 问题 |
|------|------------|--------------|------|
| Train Loss | 1.312 | — | — |
| Val Loss | 1.289 | 1.074 (Ep31) | **过拟合 20%** |
| type_loss | 1.188 | 1.190 (Ep30) | **完全停滞** |
| Count Accuracy | 42.9% | 44.4% (Ep40) | 饱和 |
| RMSD | 1.830 | 0.657 (Ep45) | 波动大 |
| Bond Validity | 91.0% | — | 饱和 |
| Bottom Recall | 7.2% | 10.7% (Ep20) | 始终极低 |

---

## 验证

```bash
python3 -m src.quick_test  # 模块健全性检查 — 已通过 ✓
```

## 运行

```bash
cd /root/autodl-tmp/micro && bash run.sh
# 或后台运行：
nohup bash -c 'export PYTHONPATH="${PYTHONPATH}:$(pwd)" && cd /root/autodl-tmp && python3 -m src.train --config /root/autodl-tmp/micro/config.json' > micro/checkpoints/training_v2.log 2>&1 &
```

## 预期改善

1. **type_loss 下降**：class_weight + 权重 ×3 → 稀有元素召回提升，type_loss 不再停滞
2. **原子数预测**：残差+Label Smoothing+融合推理 + 权重 ×2 → MAE 降低
3. **结构评估**：6 维指标直接量化预测质量，替代间接的 CID 检索
4. **过拟合缓解**：retrieval 正则化保留，AtomCountHead Dropout + Label Smoothing
