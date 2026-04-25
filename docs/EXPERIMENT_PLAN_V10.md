# AFM 3D 分子结构重建：V10 改进方案

## 第一部分：V9 问题诊断（验证后修正）

### V9 的成就与问题

**成就（被 Final Test 掩盖了）**：
- t=0 type accuracy: V8=69.8% → V9=87.4%（+17.6%）
- shape conditioning 让模型从形状描述符中学到了区分 C/N/O 的能力
- 训练过程中（DDIM-50 评估）V9 和 V8 表现几乎相同

**问题（仅在 DDIM-100 Final Test 暴露）**：
- V9 Final Test TypeMatch=47.5% (V8=50.5%)
- V9 N+O 比例=9.0% (V8=20.2%)
- V9 Coulomb=0 比例=54% (V8=30%)

**验证后的根因**：
- shape guidance 在 DDIM-100 时有 ~70 步引导 vs DDIM-50 仅 35 步
- 更多引导步 = 累积更多坐标偏移 = type/Coulomb 退化
- 训练 DDIM-50 评估时差距极小：V8 RMSD=0.231, V9=0.234
- shape_head 预测质量良好（corr 0.83-0.91），不是分布偏移问题

### 核心洞察：87% 的能力如何释放到推理时

当前架构能力链：
```
t=0 (干净坐标): type acc = 87.4%  ← V9 shape conditioning 的贡献
t=30 (微噪声):  type acc = 62.8%  ← 推理坐标的等效噪声水平
t=50:            type acc = 53.5%
Final Test:      type_match = 47.5% ← 实际评估值
```

从 87% 到 47% 的落差有两个来源：
1. 推理坐标噪声（RMSD≈0.25 ≈ t=30-50）：87% → 53-63%
2. shape guidance 累积偏移：进一步降到 47%

---

## 第二部分：V10 改进方案

### 改动 1：移除 shape guidance（采样时不做梯度引导）

保留 V9 的 shape conditioning（训练时注入 GT shape_desc 到 denoiser），但完全移除采样时的 `_apply_shape_guidance()`。

```python
# sample() 中删除:
# if target_shape is not None and t_cur < int(self.timesteps * 0.7):
#     x_0_pred = self._apply_shape_guidance(x_0_pred, target_shape, mask)
```

### 改动 2：α̅(t) 加权 type_loss

将 type_loss 的硬阈值 `t < 500` 替换为连续的 α̅(t) 加权：

```python
# 当前 (V5-V9):
low_noise = t_per_atom < 500  # 硬阈值
valid = (mask_flat > 0) & (types_flat >= 0) & low_noise

# V10:
# 不再用硬阈值, 而是对每个样本的 CE loss 乘以 alpha_bar(t)
# alpha_bar(t=0) = 1.0 → 全权重
# alpha_bar(t=100) = 0.97 → 几乎全权重
# alpha_bar(t=500) = 0.49 → 半权重
# alpha_bar(t=900) = 0.02 → 几乎忽略
per_sample_weight = self.alphas_cumprod[t]  # (B,)
type_loss = weighted_CE(type_logits, atom_types, mask, per_sample_weight)
```

**依据**：
- 实验证明 t>100 后 type acc 平坦在 ~40%（接近先验）
- α̅(t=100) = 0.97，这些样本仍保留但权重合理
- α̅(t=500) = 0.49，自然衰减，无需硬截断
- 避免了硬阈值导致的梯度不连续

### 不改动的部分

- Shape conditioning（训练时注入 shape_desc）：✓ 保留
- AFM cross-attention for type_head：✓ 保留
- 排斥力引导 + 连通性投影：✓ 保留
- 自动环检测：✓ 保留
- 损失函数权重：与 V9 相同

---

## 第三部分：预期效果

| 指标 | V8 | V9 | V10 预期 | 依据 |
|------|-----|-----|---------|------|
| RMSD | 0.254 | 0.266 | 0.24-0.26 | 移除 shape guidance 恢复坐标质量 |
| Type Match | 50.5% | 47.5% | **58-65%** | shape cond (87% at t=0) + α̅ 加权 |
| Coulomb | 0.442 | 0.380 | 0.45-0.55 | 恢复到 V8 水平 + 更好的 type |
| N+O 比例 | 20.2% | 9.0% | 18-22% | 移除引导恢复 AFM cross-attn 效果 |
