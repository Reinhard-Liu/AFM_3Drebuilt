# AFM 3D 分子结构重建：V5b 实验方案

## 概述

V5b 在 V5a 的基础上修复了原子类型预测的严重退化问题。V5a 修复了 DDIM 采样范围和数值稳定性，使 RMSD 从 1.08 降到 0.35，但 Type Match 仅 7.8%。V5b 定位到 Focal Loss + inverse-frequency 权重导致模型将所有原子预测为 O（氧），通过回归标准 CE 损失使 Type Match 恢复到 44.8%。

**硬件**：RTX 4080 SUPER (32GB)
**预计训练时间**：~6.5 小时（50 epoch）

---

## 第一部分：V5a/V5b 修复的 Bug 清单

### Bug 1：训练与推理噪声范围不一致（V5a 修复）

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| 训练 t 范围 | [0, 999] | **[0, 990]** |
| DDIM 起始 | t=100（仅 3% 噪声） | **t=990（完整噪声范围）** |

**文件**：`src/models/diffusion.py` — `compute_loss()` 第 277 行
```python
# 修复前
t = torch.randint(0, self.timesteps, (B,), device=device)
# 修复后
t_max = min(self.timesteps - 10, 990)
t = torch.randint(0, t_max + 1, (B,), device=device)
```

### Bug 2：DDPM/DDIM 采样数值爆炸（V5a 修复）

**问题**：cosine schedule 在 t>990 时 alpha_cumprod ≈ 0，DDPM 公式中 sqrt_recip_alpha[999]=31.6，放大误差 31 倍。

**修复**：
- DDIM：从 t=990 开始，使用 x_0 预测公式 + clamp(-3, 3)
- DDPM：同样改用 x_0 预测 + posterior mean 公式 + clamp

**文件**：`src/models/diffusion.py` — `sample()` 方法

### Bug 3：高噪声步的 type_loss 产生错误梯度（V5a 修复）

**问题**：t=999 时 x_t ≈ 纯噪声，denoiser 无法从中推断原子类型，但 type_loss 仍计算并反传。

**修复**：type_loss 仅在 t < 500 时计算。

**文件**：`src/models/diffusion.py` — `compute_loss()` 第 313-316 行
```python
t_per_atom = t.unsqueeze(1).expand_as(atom_types).reshape(-1)
low_noise = t_per_atom < 500
valid = (mask_flat > 0) & (types_flat >= 0) & low_noise
```

### Bug 4：type_logits 被 mask 清零（V5a 修复）

**问题**：`type_logits = type_logits * mask.unsqueeze(-1)` 破坏了 padding 位置的 logits 值。虽然训练时 loss 通过 valid mask 过滤，但推理时可能影响 attention 信息流。

**修复**：移除 type_logits 的 masking。

**文件**：`src/models/diffusion.py` — `SE3EquivariantDenoiser.forward()` 第 152 行

### Bug 5：Focal Loss + inverse-frequency 权重杀死 H/C 学习（V5b 修复）

**问题诊断**：

模型将所有原子预测为 O（氧），20 个样本统计：
```
GT分布:   H=178, C=230, N=76, O=27
Pred分布: N=46,  O=450, S=15     ← H和C完全消失
```

**根因**：inverse-frequency 权重 + Focal Loss 形成恶性循环：
```
H: inv_freq=0.21, Focal(γ=2.0) 进一步衰减 → 有效梯度 ≈ 0
C: inv_freq=0.29, Focal(γ=2.0) 进一步衰减 → 有效梯度 ≈ 0
O: inv_freq=2.30, Focal 不衰减 → 梯度最大
模型策略：全部预测为 O → 加权 loss 最低
```

**修复**：
1. 去掉 Focal Loss，回到标准 CrossEntropy
2. class_weight 改用 `sqrt(inverse_freq)`，clamp(max=3.0)
3. type_loss 外部权重从 0.3 提高到 1.0

```python
# 修复前 (V5a)
class_weight = (numel / (num_classes * counts)).clamp(max=10.0)
focal_weight = (1.0 - pt) ** 2.0
type_loss = (focal_weight * ce).mean()

# 修复后 (V5b)
inv_freq = numel / (num_classes * counts)
class_weight = torch.sqrt(inv_freq).clamp(max=3.0)
type_loss = F.cross_entropy(logits, targets, weight=class_weight)
```

新的 class_weight 分布（温和平衡）：
```
H: 0.54    C: 0.46    N: 0.88    O: 1.52    F: 3.0(clamp)
```

**文件**：
- `src/models/diffusion.py` — `compute_loss()` 第 317-328 行
- `src/train.py` — type_loss 权重 0.3 → 1.0（第 169 行）

### 改进 6：Count 回归损失增强（V5b）

**问题**：smooth_l1_loss 对大偏差（MAE>1）梯度恒为 1，收敛慢。

**修复**：
- smooth_l1 → MSE（对大偏差梯度更大）
- reg_loss 内部权重 0.5 → 1.0

**影响范围**：仅 count_head 内部参数，不影响 encoder 梯度分配，不影响 type_match。

**文件**：`src/models/prediction_heads.py` — `compute_loss()` 第 92-94 行

---

## 第二部分：最终损失函数

```python
# diffusion.py 内部 (仅 t<500 的样本计算 type_loss)
loss = coord_loss + 0.1 * type_loss + 0.5 * shape_loss

# train.py 最终损失 (覆盖 diffusion 内部 loss)
loss = (
    coord_loss                    # 1.0  坐标重建
    + 1.0 * type_loss             # 1.0  原子类型 (V5b: 标准CE, sqrt权重)
    + 1.0 * count_loss            # 1.0  原子数 (V5b: MSE回归)
    + 0.01 * retrieval_loss       # 0.01 正则化
    + 0.1 * constraint_loss       # 0.1  键长/键角 (Stage 2+)
    + 0.5 * shape_loss            # 0.5  惯性张量形状
)
```

**Encoder 梯度分配**：
```
coord: 1.0 (29%)  |  type: 1.0 (29%)  |  count: 1.0 (29%)  |  shape: 0.5 (14%)
```

---

## 第三部分：各版本损失配置对比

| 损失项 | V2 | V3 | V4 | V5b |
|--------|-----|-----|-----|------|
| coord_loss | 1.0 | 1.0 | 1.0 | 1.0 |
| type_loss | 0.3 CE+inv_freq | 0.3 Focal(γ=2)+inv_freq | 0.3 Focal(γ=3)+inv_freq | **1.0 CE+sqrt_inv_freq** |
| count_loss | 1.0 smooth_l1 | 1.0 smooth_l1 | 1.0 smooth_l1 | **1.0 MSE** |
| shape_loss | — | 0.5 | 0.5 | 0.5 |
| retrieval_loss | 0.01 | 0.01 | 0.01 | 0.01 |
| constraint_loss | 0.1 | 0.1 | 0.1 | 0.1 |

---

## 第四部分：配置

```json
{
    "batch_size": 128,
    "num_workers": 4,
    "prefetch_factor": 2,
    "persistent_workers": true,
    "max_samples": 100000,
    "epochs": 50,
    "lr": 1e-4,
    "eval_ddim_steps": 50,
    "eval_samples_quick": 200,
    "eval_samples_full": 500,
    "save_every": 10,
    "early_stopping_patience": 5
}
```

---

## 第五部分：5 Epoch 快速测试结果

### V5a vs V5b（均为 5 epoch）

| 指标 | V5a | V5b | 变化 |
|------|-----|-----|------|
| RMSD | **0.351** | 0.373 | +6% |
| Kabsch | **0.861** | 0.850 | -1% |
| Type Match | 7.8% | **44.8%** | **+474%** |
| Bond Valid | **33.6%** | 20.3% | -40% |
| Valence | 22.7% | **19.8%** | -13% |
| Count Acc | 27.2% | **26.6%** | -2% |
| Bottom Recall | 0.45% | **2.02%** | +349% |
| Struct Sim | 0.465 | **0.537** | +15% |
| Composite | **0.372** | 0.359 | -3% |

### V5b 逐 Epoch 指标

| Epoch | RMSD | Kabsch | Type | Bond | Count Acc | MAE | Composite |
|-------|------|--------|------|------|-----------|-----|-----------|
| 1 | 0.465 | 0.796 | 42.2% | 9.2% | 16.0% | 2.17 | 0.314 |
| 2 | 0.450 | 0.808 | 43.0% | 13.4% | 22.3% | 1.48 | 0.334 |
| 3 | 0.396 | 0.841 | 44.2% | 21.4% | 22.7% | 1.45 | 0.355 |
| 4 | 0.394 | 0.835 | 44.1% | 20.8% | 27.3% | 1.23 | 0.356 |
| 5 | 0.461 | 0.802 | 43.0% | 17.5% | 29.9% | 1.10 | 0.344 |
| **Test** | **0.373** | **0.850** | **44.8%** | **20.3%** | **26.6%** | **1.46** | **0.359** |

---

## 第六部分：50 Epoch 预估

### 时间

| 阶段 | Epoch | 平均耗时 | 小计 |
|------|-------|---------|------|
| Stage 1 基础训练 | 1-30 | 6.5 min | 3.2h |
| Stage 2 约束训练 | 31-45 | 9.7 min | 2.4h |
| Stage 3 底部聚焦 | 46-50 | 10.7 min | 0.9h |
| **总计** | | | **~6.5h** |

### 指标预估

| 指标 | V3 (60ep) | V4 (50ep) | V5b 预估 (50ep) |
|------|-----------|-----------|-----------------|
| RMSD | 1.038 | 1.076 | **0.25-0.35** |
| Kabsch | 0.509 | 0.492 | **0.88-0.92** |
| Type Match | 27.2% | 8.4% | **50-60%** |
| Count Acc | 36.3% | 35.9% | **40-50%** |
| Count MAE | 0.720 | 1.027 | **0.6-0.8** |
| Bond Valid | 8.6% | 5.6% | **30-40%** |
| Valence | 4.4% | 4.4% | **25-35%** |
| Composite | 0.236 | 0.220 | **0.40-0.50** |

---

## 第七部分：修改文件清单

| 文件 | 修改 | Bug |
|------|------|-----|
| `src/models/diffusion.py:152` | 移除 type_logits masking | #4 |
| `src/models/diffusion.py:277` | 训练 t 范围 [0,990] | #1 |
| `src/models/diffusion.py:313-316` | type_loss 仅 t<500 | #3 |
| `src/models/diffusion.py:317-328` | 标准 CE + sqrt(inv_freq) | #5 |
| `src/models/diffusion.py:528-565` | DDIM x_0 预测 + clamp | #2 |
| `src/models/diffusion.py:573-601` | DDPM x_0 预测 + clamp | #2 |
| `src/train.py:169` | type_loss 权重 0.3→1.0 | #5 |
| `src/models/prediction_heads.py:92-94` | count reg MSE + 权重 1.0 | #6 |

---

## 第八部分：运行命令

```bash
cd /root/autodl-tmp/micro

# 修改 config.json 中 epochs 为 50
# 启动训练
nohup python3 -m src.train --config config.json > checkpoints/training_v5b.log 2>&1 &
```
