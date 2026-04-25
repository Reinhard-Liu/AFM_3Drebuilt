# AFM 3D 分子结构重建：V7 改进方案

## 设计原则

基于 V1-V6 的系统性反思，V7 遵循三个原则：

1. **每次只改 1-2 个变量**：V6 同时改 7 个组件导致无法归因，V7 严格控制变量
2. **从"逐原子精确重建"转向"分子级特征匹配"**：新增分子式相似度、类型分布匹配等分子级指标和损失函数
3. **先修复再创新**：先恢复 V5b 的坐标质量基线，再叠加新功能

## 基线选择

**V7 基于 V5b 代码**，不基于 V6。原因：
- V5b RMSD=0.255 是历史最佳坐标质量
- V5b 参数量 25M，训练效率高
- V5b 的 denoiser type_head 提供有效的辅助梯度

---

## 改进 1：新增分子级评估指标与训练损失

### 1.1 Formula Similarity（分子式相似度）

**动机**：当前 type_match 要求逐原子对应正确，但用户需要的是分子整体元素组成正确。例如预测 C₆H₆O vs GT C₆H₅NO，元素计数向量的余弦相似度比逐原子对比更有意义。

**评估指标**：
```python
def formula_similarity(pred_types, gt_types, pred_mask, gt_mask):
    """比较预测和GT的元素组成向量"""
    pred_counts = torch.zeros(num_types)
    gt_counts = torch.zeros(num_types)
    for i in range(num_types):
        pred_counts[i] = (pred_types[pred_mask] == i).sum()
        gt_counts[i] = (gt_types[gt_mask] == i).sum()
    # 余弦相似度
    return F.cosine_similarity(pred_counts, gt_counts, dim=0)
```

**训练损失**（Formula Loss）：
```python
def formula_loss(type_logits, gt_types, mask):
    """分子级元素组成损失 —— 不要求逐原子对应"""
    # 预测的元素计数（软计数，可微）
    type_probs = F.softmax(type_logits, dim=-1)         # (B, N, 10)
    pred_counts = (type_probs * mask.unsqueeze(-1)).sum(dim=1)  # (B, 10)
    # GT 的元素计数
    gt_onehot = F.one_hot(gt_types, num_types).float()  # (B, N, 10)
    gt_counts = (gt_onehot * mask.unsqueeze(-1)).sum(dim=1)     # (B, 10)
    # MSE on counts
    return F.mse_loss(pred_counts, gt_counts)
```

**修改文件**：
- `src/utils/metrics.py` — 新增 `compute_formula_similarity()`
- `src/models/diffusion.py` — `compute_loss()` 中新增 formula_loss

### 1.2 Type Distribution Match（类型分布匹配度）

**评估指标**：
```python
def type_distribution_match(pred_types, gt_types, pred_mask, gt_mask):
    """比较预测和GT的类型比例分布"""
    pred_dist = bincount(pred_types[pred_mask], 10).float()
    pred_dist = pred_dist / pred_dist.sum().clamp(min=1)
    gt_dist = bincount(gt_types[gt_mask], 10).float()
    gt_dist = gt_dist / gt_dist.sum().clamp(min=1)
    return 1.0 - js_divergence(pred_dist, gt_dist)
```

**修改文件**：`src/utils/metrics.py`

---

## 改进 2：启用推理时环结构约束

**动机**：这是 V3 就计划但 6 个版本从未实现的功能。环结构（苯环、吡啶、呋喃等）在有机分子中占大量原子，Procrustes 对齐可以直接改善局部几何质量。更重要的是，环结构可以辅助 type 预测——苯环上一定是 C，吡啶有一个 N，呋喃有一个 O。

### 2.1 generate() 传入 ring_info

```python
# train.py: generate() 中增加环信息传递
def generate(self, batch, ...):
    ...
    ring_info = None
    if "ring_atom_indices" in batch and "ring_templates" in batch:
        ring_info = {
            "ring_atom_indices": batch["ring_atom_indices"],
            "ring_templates": batch["ring_templates"],
            "ring_valid": batch["ring_valid"],
        }
    coords, type_logits = self.ddpm.sample(
        c, n_atoms, max_atoms=MAX_ATOMS, ring_info=ring_info, ...
    )
```

### 2.2 推理时自动环检测（不依赖 GT）

当不提供 GT ring_info 时（真实推理场景），在 DDIM 采样的 t < 30% 阶段，对 x_0_pred 用距离阈值检测环：

```python
# diffusion.py: sample() 内部
if t_cur < int(self.timesteps * 0.3) and t_cur % re_detect_interval == 0:
    # 从当前预测坐标检测环
    detected_rings = detect_rings_from_coords(x_0_pred, threshold=0.18)
    # 执行平面性投影
    x_0_pred = project_rings_to_planar(x_0_pred, detected_rings)
```

### 2.3 环辅助类型推断

```python
def ring_type_prior(ring_info, type_logits):
    """利用环结构先验修正类型预测"""
    # 苯环：所有原子 → C
    # 吡啶：5个C + 1个N
    # 呋喃：4个C + 1个O
    # 将 ring_type 信息作为 type_logits 的 soft prior
```

### 2.4 ring_preservation 评估指标

```python
def compute_ring_preservation(pred_coords, gt_coords, pred_types, gt_types, mask):
    """检测GT中的环是否在预测结构中被保持"""
    gt_rings = detect_rings(gt_coords, gt_types, mask)
    pred_rings = detect_rings(pred_coords, pred_types, mask)
    # 比较环的数量、大小、类型
```

**修改文件**：
- `src/train.py` — `generate()` 传入 ring_info
- `src/models/diffusion.py` — `sample()` 启用环约束
- `src/utils/metrics.py` — 新增 `compute_ring_preservation()`

---

## 改进 3：修正评估指标体系

### 3.1 修复 compute_rmsd 的零坐标问题

```python
# 修复前：
p = pred_coords[b, :n]  # n = gt_mask.sum() = n_gt

# 修复后：
n_gt = mask[b].sum().item()
n_pred = n_atoms_pred[b].item() if n_atoms_pred is not None else n_gt
n_use = min(n_pred, n_gt)
p = pred_coords[b, :n_use]
g = gt_coords[b, :n_use]
```

### 3.2 放宽 Bottom Recall 阈值

```python
# 修复前：
distance_threshold = 0.1  # 1.2 Å，当前精度下几乎不可能达到

# 修复后：分级报告
thresholds = [0.1, 0.2, 0.3]  # 1.2Å, 2.4Å, 3.6Å
# 报告 Bottom Recall@1.2Å, @2.4Å, @3.6Å
```

### 3.3 Coulomb 指标改用中位数+分组

```python
# 修复前：
coulomb_mean = np.mean(all_coulombs)

# 修复后：
coulomb_median = np.median(all_coulombs)
coulomb_small = np.mean([c for c, n in zip(coulombs, n_atoms) if n <= 30])
coulomb_large = np.mean([c for c, n in zip(coulombs, n_atoms) if n > 30])
```

### 3.4 新的 overall score 组成

```python
overall = (
    0.20 * kabsch_score          # 形状
    + 0.20 * formula_similarity  # 分子式（新增）
    + 0.15 * type_distribution   # 类型分布（新增）
    + 0.15 * (1 - js_div)        # 距离分布
    + 0.10 * valence_validity    # 化学有效性
    + 0.10 * count_similarity    # 原子数
    + 0.10 * bond_validity       # 键合理性
)
```

**修改文件**：`src/utils/metrics.py`

---

## 损失函数

```python
# V7 总损失（基于 V5b，新增 formula_loss）
loss = (
    coord_loss                    # 1.0  坐标重建
    + 1.0 * type_loss             # 1.0  原子类型（标准CE, sqrt权重）
    + 0.5 * formula_loss          # 0.5  分子式相似度（新增）
    + 1.0 * count_loss            # 1.0  原子数
    + 0.5 * shape_loss            # 0.5  惯性张量形状
    + 0.01 * retrieval_loss       # 0.01 正则化
    + 0.1 * constraint_loss       # 0.1  键长/键角/平面性（Stage 2+）
)
```

与 V5b 相比只新增了 `formula_loss`（权重 0.5），其余完全不变。

---

## 实施步骤（严格按顺序）

### Phase 1：恢复基线 + 新指标（不改模型）

| 步骤 | 内容 | 改文件 | 风险 |
|------|------|--------|------|
| 1a | 恢复 V5b 代码为基线 | 回退 video_vit.py, diffusion.py, train.py | 低 |
| 1b | 新增 formula_similarity, type_distribution_match 评估指标 | metrics.py | 零（仅评估） |
| 1c | 修复 compute_rmsd 零坐标 bug | metrics.py | 低 |
| 1d | 修复 Bottom Recall 分级阈值 | metrics.py | 低 |
| 1e | Coulomb 改用中位数 + 分组报告 | metrics.py, train.py | 低 |
| 1f | 5 epoch 快速验证：确认 V5b 基线复现 | — | — |

### Phase 2：新增 Formula Loss（唯一的模型改动）

| 步骤 | 内容 | 改文件 | 风险 |
|------|------|--------|------|
| 2a | 新增 formula_loss（分子级元素组成损失） | diffusion.py, train.py | 低 |
| 2b | 5 epoch 快速验证：确认 formula_similarity 提升 | — | — |
| 2c | 50 epoch 完整训练 | — | — |

### Phase 3：启用环结构推理（Phase 2 训练完成后）

| 步骤 | 内容 | 改文件 | 风险 |
|------|------|--------|------|
| 3a | generate() 传入 ring_info | train.py | 低 |
| 3b | 新增 ring_preservation 评估指标 | metrics.py | 零 |
| 3c | 用 Phase 2 的 checkpoint 评估环约束效果 | — | — |
| 3d | 如果有效，加入推理时自动环检测 | diffusion.py | 中 |

---

## 预期结果

### Phase 1（基线复现 + 新指标）

| 指标 | V5b 值 | 预期 | 说明 |
|------|--------|------|------|
| RMSD | 0.255 | 0.255 | 复现基线 |
| Type Match（原有） | 48.5% | 48.5% | 复现基线 |
| **Formula Similarity** | — | **~0.75-0.85** | 新指标，分子式级别 |
| **Type Distribution** | — | **~0.70-0.80** | 新指标，类型分布 |

### Phase 2（+Formula Loss）

| 指标 | V5b 值 | 预期 | 说明 |
|------|--------|------|------|
| RMSD | 0.255 | 0.25-0.28 | 不应退化 |
| Type Match | 48.5% | 50-55% | 分子式监督间接提升 |
| **Formula Similarity** | — | **0.85-0.92** | 核心提升目标 |
| Count Accuracy | 30.8% | 32-38% | 间接改善 |
| Coulomb (median) | 0.60 | 0.60-0.70 | 类型改善带动 |

### Phase 3（+环约束推理）

| 指标 | Phase 2 | 预期 | 说明 |
|------|---------|------|------|
| Bond Validity | ~40% | **50-60%** | 环内键长/键角被约束 |
| Ring Preservation | — | **~0.5-0.7** | 新指标 |
| Valence Validity | 35% | **40-50%** | 环内原子化合价被约束 |

---

## 配置

```json
{
    "data_root": "auto",
    "param_key": "K-1",
    "img_size": 128,
    "num_frames": 10,
    "min_corrugation": 1.25,
    "augment_rotation": true,
    "batch_size": 128,
    "num_workers": 4,
    "prefetch_factor": 2,
    "persistent_workers": true,
    "max_samples": 100000,
    "val_size": 1000,

    "patch_size": 16,
    "temporal_patch_size": 2,
    "embed_dim": 512,
    "encoder_depth": 8,
    "num_heads": 8,
    "drop_rate": 0.1,

    "denoiser_hidden_dim": 256,
    "denoiser_depth": 6,
    "diffusion_steps": 1000,

    "lr": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 50,
    "save_dir": "experiments/v7/checkpoints",
    "log_interval": 1,

    "eval_ddim_steps": 50,
    "eval_samples_per_epoch": 200,
    "eval_full_interval": 5,

    "formula_loss_weight": 0.5,

    "model_type": "diffusion"
}
```

---

## 与历史版本改动对比

| 维度 | V5b | V6 (失败) | V7 |
|------|-----|-----------|-----|
| 模型基线 | V5 修复 | V5b | **V5b** |
| Denoiser type_head | ✓ 保留 | ✗ 移除 | **✓ 保留** |
| TypeNet | 无 | 新增（51M参数） | **无** |
| Cross-attention | 无 | 新增（6层） | **无** |
| 新增损失 | CE+sqrt权重 | dist_loss+valence_loss | **formula_loss** |
| 参数量 | ~25M | 51M | **~25M** |
| 同时改动数 | 2 | **7** | **1-2** |
| 新评估指标 | 无 | 无 | **formula_sim + type_dist + ring_pres** |
| 环约束推理 | 未启用 | 未启用 | **Phase 3 启用** |
| 指标修复 | 无 | 无 | **RMSD零坐标 + Bottom阈值 + Coulomb分组** |
