# AFM 3D 分子结构重建：V11 改进方案

## 第一部分：问题诊断

### 核心瓶颈

V1-V10 共 10 个版本，Type Match 始终无法突破 60%：

| 版本 | 峰值 Type Match | 最终 Type Match | 干净坐标(t=0) Type Acc |
|------|---------------|----------------|----------------------|
| V5b | 49% | 49% | — |
| V7 | 56% | 56% | — |
| V8 | 57% | 51% | 69.8% |
| V9 | 57% | 48% | 87.4% |
| V10 | 58% | 49% | 88.2% |

**瓶颈已明确**：架构在干净坐标上能达 88%，但推理时卡在 50%。

原因链条：
1. denoiser 的 coord_head 和 type_head 共享 transformer → 训练越久 coord 越好但 type 被挤压
2. 推理时坐标有 RMSD≈0.25 的误差（等效 t≈30-50）→ type_head 在此噪声下只有 53-63% 准确率
3. 匈牙利匹配基于坐标 → 坐标匹配错位时 type 判定也错

### 为什么之前的方案都失败了

| 方案 | 版本 | 失败原因 |
|------|------|---------|
| 提高 type_loss 权重 | V5b | coord/type 梯度竞争不变 |
| TypeNet 解耦 | V6 | 只在 GT 坐标训练，推理时 exposure bias |
| formula_loss | V7 | 干扰了 type_loss，N/O 反而降到 6% |
| AFM cross-attn | V8 | 一次性提升 +12%，但之后饱和 |
| shape conditioning | V9 | t=0 从 70%→87%，但推理时释放不出来 |
| α̅(t) 加权 | V10 | type_loss 降了但 type_match 不升（优化了错误区间） |
| 两阶段推理 | V10实验 | 生成坐标≠GT坐标，denoiser 没见过这种误差模式 |

## 第二部分：V11 方案

### 核心改动：独立 TypePredictor + 噪声鲁棒训练

**与 V6 TypeNet 的关键区别**：

| 维度 | V6 TypeNet | V11 TypePredictor |
|------|-----------|-------------------|
| 训练输入坐标 | GT 坐标（完美） | GT 坐标 + **随机加噪**（模拟推理误差） |
| 噪声水平 | 无 | 等效 t=20~60（覆盖推理误差范围） |
| 与 denoiser 关系 | 替代 type_head（去掉了） | **保留** denoiser type_head 作辅助训练 |
| 推理时输入 | 生成坐标（exposure bias） | 生成坐标（但训练时见过类似噪声） |
| AFM 信息 | cross-attn to patches | cross-attn to patches（同 V8） |

```python
class TypePredictor(nn.Module):
    """独立的类型预测器，在带噪坐标上训练以匹配推理条件。"""

    def __init__(self, cond_dim=512, hidden_dim=256, num_types=10):
        # 坐标编码
        self.coord_encoder = MLP(3 → 256)
        # AFM patches cross-attention
        self.patch_proj = Linear(cond_dim → hidden_dim)
        self.cross_attn = MultiheadAttention(hidden_dim, 8)
        # Transformer layers（独立的，不与 denoiser 共享）
        self.layers = 4x TransformerEncoderLayer(256, 8)
        # Type head
        self.type_head = MLP(256 → num_types)

    def forward(self, coords, c_patches, mask):
        # coords 可以是干净的或加噪的
        h = self.coord_encoder(coords)
        p = self.patch_proj(c_patches)
        h_cross, _ = self.cross_attn(h, p, p)
        h = h + 0.3 * h_cross  # 更大的残差权重
        for layer in self.layers:
            h = layer(h)
        return self.type_head(h)
```

**训练策略**：
```python
# 在 forward() 中：
# 1. denoiser 正常训练（coord + type，保持 V10 的设置）
# 2. TypePredictor 在带噪 GT 坐标上训练
noise_level = torch.rand(B) * 0.3  # 归一化噪声 std 0~0.3（覆盖 RMSD 0~0.3 的范围）
noisy_coords = gt_coords + noise_level.unsqueeze(-1).unsqueeze(-1) * torch.randn_like(gt_coords)
type_pred_logits = self.type_predictor(noisy_coords, c_patches, mask)
type_pred_loss = CE(type_pred_logits, gt_types, weight=sqrt_inv_freq)
```

**推理时**：
```python
# DDIM 生成坐标（denoiser 照常工作）
coords, _ = self.ddpm.sample(c_global, c_patches, n_atoms, ...)
# TypePredictor 在生成坐标上预测类型
type_logits = self.type_predictor(coords, c_patches, mask)
```

### 其他改动

1. **修复旋转增强**（已完成）：XY 旋转 + AFM 图像同步旋转
2. **恢复 t<500 的 type_loss**：撤回 V10 的 α̅(t) 加权（验证无效）
3. **保留 shape conditioning**：训练时注入 shape_desc（V9 的有效改动）
4. **保留物理引导**：排斥力 + 连通性（V8 的有效改动）

### 不改动的部分

- denoiser 架构：保留 coord_head + type_head（type_head 作辅助训练信号）
- AFM cross-attention for denoiser type_head：保留
- Shape conditioning：保留

## 第三部分：预期效果

| 指标 | V10 | V11 预期 | 依据 |
|------|-----|---------|------|
| Type Match | 48.7% | **62-70%** | TypePredictor 在 RMSD≈0.25 等效噪声上训练 |
| RMSD | 0.254 | 0.24-0.26 | 旋转增强修复可能改善 |
| Coulomb | 0.405 | 0.40-0.50 | type 更准 → Coulomb 改善 |
| N+O 比例 | 24.1% | 18-22% | AFM cross-attn + 更准的 type 预测 |
| Composite | 0.493 | 0.52-0.56 | 综合提升 |

## 第四部分：实施步骤

| 步骤 | 内容 | 风险 |
|------|------|------|
| 1 | 恢复 denoiser type_loss 为 t<500 硬阈值 | 零（回到 V8 设置） |
| 2 | 实现 TypePredictor（独立模块） | 低 |
| 3 | forward() 中加入噪声鲁棒训练 | 低 |
| 4 | generate() 中用 TypePredictor 替代 denoiser type_logits | 低 |
| 5 | 3 epoch 快速验证 | — |
| 6 | 50 epoch 完整训练 | — |
