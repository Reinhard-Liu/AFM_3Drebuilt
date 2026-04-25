# AFM 3D 分子结构重建：V5 修复方案

## 核心问题与修复

### Bug 1：训练与推理噪声范围不一致
- **问题**：训练 t∈[0,999]，推理 DDIM 只从 t=100 开始（仅3%噪声）
- **修复**：训练时限制 t∈[0,990]，推理 DDIM 从 t=990 开始（完整噪声范围）

### Bug 2：DDPM采样在cosine schedule尾部数值爆炸
- **问题**：t>990时 alpha_cumprod≈0，sqrt_recip_alpha=31.6，放大误差31倍
- **修复**：DDPM改用x_0预测公式+clamp(-3,3)；训练和推理都截断到 t_max=990

### Bug 3：Type prediction在高噪声步产生错误梯度
- **问题**：denoiser从纯噪声x_t预测type，高噪声步type_loss是噪声
- **修复**：type_loss仅在 t<500 时计算（低噪声步有足够结构信息）

### Bug 4：Focal Loss + inverse-freq weight 杀死了H/C的学习
- **问题**：H权重=0.21, C权重=0.29, 而稀有类权重=10.0；Focal Loss进一步压低H/C梯度；模型学到"全部预测为O"可获最低加权损失
- **修复**：去掉Focal Loss，使用标准CE；class_weight改用sqrt(inverse_freq)并clamp(max=3.0)；type_loss权重从0.3提高到1.0

### Bug 5：type_logits被mask清零
- **问题**：`type_logits = type_logits * mask` 破坏logits
- **修复**：移除type_logits的masking

---

## 代码修改

### diffusion.py

| 位置 | 修改 |
|------|------|
| `SE3EquivariantDenoiser.forward()` | 移除 `type_logits * mask` |
| `compute_loss()` t采样 | `[0,999]` → `[0,990]` |
| `compute_loss()` type_loss | 仅 t<500 计算；去掉Focal Loss；sqrt(inv_freq) clamp 3.0 |
| `sample()` DDIM | 从 t=990 开始，x_0预测+clamp |
| `sample()` DDPM | x_0预测公式+clamp，从 t=990 开始 |

### train.py

| 位置 | 修改 |
|------|------|
| `type_loss` 权重 | 0.3 → **1.0** |

---

## 5 Epoch快速测试结果

| 指标 | V4 (50ep) | V5 (5ep) | 变化 |
|------|-----------|----------|------|
| RMSD | 1.076 | **0.351** | -67% |
| Kabsch | 0.492 | **0.861** | +75% |
| Bond Valid | 5.6% | **33.6%** | +500% |
| Composite | 0.220 | **0.372** | +69% |
| Type Match | 8.4% | 7.8% | 待改善 |

## Type Match修复（第二轮）

问题：模型把所有原子预测为O（氧），不预测H和C
- Focal Loss的(1-pt)^γ权重让H/C梯度趋近于0
- inverse_freq权重让O/N的loss值是H/C的50倍
- 模型策略：全预测O → 加权loss最低

修复：
- 去掉Focal Loss → 标准CE
- sqrt(inv_freq), clamp(3.0) → H:0.54, C:0.46, N:0.88, O:1.52（温和平衡）
- type_loss权重 0.3 → 1.0
