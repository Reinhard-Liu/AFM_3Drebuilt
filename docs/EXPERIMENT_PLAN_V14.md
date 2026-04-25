# AFM 3D 分子结构重建：V14 改进方案

> 基于 V13 全流程结果 + 可视化分析 + 历史版本趋势

---

## 第一部分：V13 成果与遗留问题

### 1.1 V13 达成情况

| 目标 | 指标 | V13-Diff | V13+GNN | 目标值 | 状态 |
|------|------|----------|---------|--------|------|
| 原子位置 | RMSD | 0.2095 | **0.2373** | <0.30 | ✓ 达标 |
| 类型≥75% | cond_type_acc | **0.5969** | 0.5707 | 75% | ✗ 差 15.3% |
| 形状≥80% | PMI shape | **0.8413** | 0.8331 | 80% | ✓ 达标 |
| 环结构 | Ring preserve | **0.9240** | 0.9131 | 一致 | ✓ 达标 |

### 1.2 V13 vs V12 核心改进

| 指标 | V12-Diff | V13-Diff | 变化 | V12+GNN | V13+GNN | 变化 |
|------|----------|----------|------|---------|---------|------|
| RMSD | 0.2050 | 0.2095 | +2.2% | 0.2828 | **0.2373** | **-16.1%** |
| Type Match | 0.4478 | **0.5834** | **+30.3%** | 0.5362 | 0.5463 | +1.9% |
| Cond Type Acc | 0.4694 | **0.5969** | **+27.2%** | 0.5300 | 0.5707 | +7.7% |
| Coulomb | 0.3921 | **0.5791** | +47.7% | 0.5284 | 0.5138 | -2.8% |
| Ring Preserve | 0.9074 | **0.9240** | +1.8% | 0.9088 | 0.9131 | +0.5% |
| Bond Valid | 0.7036 | **0.7978** | +13.4% | 0.7496 | 0.7637 | +1.9% |
| Bottom Recall | 0.1024 | 0.1011 | -1.3% | 0.1363 | 0.0922 | -32.4% |
| Count Acc | 0.0751 | 0.1914 | +154.9% | 0.2188 | 0.1758 | -19.7% |

### 1.3 V13 的成功改进

| 改进项 | 效果 | 验证依据 |
|--------|------|----------|
| 环数据集过滤 | Type Match +30.3% | 去掉 8,278 无环分子，模型专注于含环分子 |
| Z 轴加权 loss (z_weight=2.0) | Coulomb +47.7% | 更好的 Z 方向分辨，Coulomb 度量 3D 结构相似度 |
| 环预测 loss | Ring Preserve +1.8% | 环数量+键连接双重约束 |
| 去掉 MMFF94 | GNN 管线 RMSD -16.1% | 力场在 53% type 准确率下弊大于利 |
| 增强连通性修正 | Bond Valid +13.4% | 拉回 0.6 + 2 次迭代 |

### 1.4 V13 遗留的五大问题

| # | 问题 | 严重程度 | 数据证据 |
|---|------|----------|----------|
| P1 | **跷跷板效应**：TypeMatch 早期高、后期崩溃 | 高 | Ep1-4: 0.567-0.570 → Ep15-20: 0.477-0.488 |
| P2 | **GNN 管线反向退化**：Diff 比 Diff+GNN 更好 | 高 | Type Match: 0.583 vs 0.546，Cond Type: 0.597 vs 0.571 |
| P3 | **Bottom Recall 停滞不前** | 中 | V12: 0.102 → V13: 0.101（Z 轴加权未见效） |
| P4 | **Composite 计算不完整** | 中 | train.py 中 `ring_preservation=0.0` 硬编码；eval_recycling 无 Composite |
| P5 | **Type 准确率距离 75% 目标仍有 15% 差距** | 高 | cond_type_acc=0.597，受限于 C/N/O 物理相似性 |

---

## 第二部分：问题诊断与根因分析

### 2.1 P1：跷跷板效应（TypeMatch 早期高→后期崩溃）

**数据证据**（V13 Phase 1 epoch_metrics.md）：

| Epoch | TypeMatch | RMSD | Composite |
|-------|-----------|------|-----------|
| 1 | **0.5702** | 0.2085 | 0.4734 |
| 4 (best) | **0.5670** | 0.2451 | 0.4926 |
| 8 | 0.5226 | 0.2224 | **0.5006** |
| 15 | 0.4808 | **0.2101** | 0.4884 |
| 20 | 0.4773 | 0.2248 | 0.4798 |

**规律**：TypeMatch 在前 4 个 epoch 最高，然后单调下降约 10%。同时 RMSD 保持稳定（0.21-0.25 波动），Coulomb 在 Ep8 达到峰值 0.630。

**根因分析**：

1. **特征空间竞争**：SE(3)-等变 denoiser 的 Transformer 特征空间有限（512 维），coord_head 和 type_head 共享特征。随着训练深入，模型倾向于将容量分配给 coord_loss（因为 coord_loss 梯度信号更稳定），type 预测成为"牺牲品"。

2. **学习率调度不适配**：当前使用 cosine 退火，在后期学习率过低，无法维持 type 预测的更新动量。type 预测在早期快速学习了元素分布的 pattern，后期逐渐被 coord 优化覆盖。

3. **缺乏 type 专属正则化**：coord_loss 和 type_loss 使用相同的 backbone，没有独立的正则化策略。

**历史验证**：V12 也存在相同现象（V12 epoch 数据未详细记录，但 Phase 1 eval TypeMatch=0.448 低于训练早期），说明这是架构层面的问题，而非 V13 训练策略引入的新问题。

### 2.2 P2：GNN 管线反向退化

**数据证据**：

| 指标 | V13-Diff（denoiser type_head） | V13+GNN（GNN 替换 type） | 差异 |
|------|------|------|------|
| Type Match | **0.5834** | 0.5463 | GNN 损失 -3.7% |
| Cond Type Acc | **0.5969** | 0.5707 | GNN 损失 -2.6% |
| RMSD | **0.2095** | 0.2373 | GNN 管线 RMSD 恶化 |
| Coulomb | **0.5791** | 0.5138 | GNN 管线更差 |

**V12 对比**：

| 指标 | V12-Diff | V12+GNN | GNN 效果 |
|------|----------|---------|----------|
| Type Match | 0.4478 | **0.5362** | +8.8%（GNN 有益） |
| Cond Type Acc | 0.4694 | **0.5300** | +6.1%（GNN 有益） |

**根因分析**：

1. **V12 → V13 的力量对比翻转**：
   - V12：denoiser type_head 44.8% << GNN val_acc 67.8% → GNN 净增益
   - V13：denoiser type_head 58.3% 接近 GNN 在噪声坐标上的实际输出 54.6% → GNN 不再有增益

2. **GNN 的训练/推理环境不匹配**：
   - GNN 训练数据：5000 个 DDIM-50 生成样本 + σ=0.01 噪声（训练时 Val Type Acc=68.2%）
   - GNN 推理环境：测试集的 DDIM-50 生成坐标（RMSD~0.21，噪声分布与训练数据不同）
   - 这个 gap 导致 GNN 在测试集上只能达到 54.6%，低于 denoiser 的 58.3%

3. **GNN 替换而非融合**：当前代码 `pred_types = corrected`（`visualize_val.py:456`）是完全替换 denoiser 的 type 预测，而非融合。这意味着 GNN 的错误直接覆盖了 denoiser 的正确预测。

4. **V13 之所以之前预期 GNN 有帮助**：方案制定时基于 V12 数据（denoiser 44.8%），预期 V13 的 denoiser 只会略有提升。实际 V13 的环数据集+Z 轴 loss 带来了 +30% 的 TypeMatch 飞跃，超出预期。

### 2.3 P3：Bottom Recall 停滞

**数据证据**：

| 版本 | Bottom Recall | Z 轴策略 |
|------|---------------|----------|
| V5b | 0.039 | 无 |
| V8 | 0.040 | 无 |
| V12-Diff | 0.102 | corrugation 过滤 |
| V13-Diff | 0.101 | Z 轴加权 loss + corrugation 过滤 |
| V13+GNN | 0.092 | 同上 + GNN |

**根因分析**：

1. **AFM 物理极限**：Z 方向信息耦合在 10 层深度切片的对比度衰减中，不像 XY 方向有直接的空间映射。Bottom 原子（被上层原子遮挡）在所有切片中信号极弱。

2. **Z 轴加权 loss 的局限**：`z_weight=2.0` 只是让模型更关注 Z 方向的 MSE，但如果 Z 方向信息在输入中本来就很弱，加权也无法"创造"信息。Corrugation 分组数据也证实了这一点——高起伏分子的 RMSD（0.232）反而不是最差的，说明模型学到了一定的 Z 方向模式。

3. **Bottom 的定义问题**：Bottom 30% 原子通常是被上层原子遮挡最严重的，信号最弱。即使模型学到了 Z 方向趋势，精确重建被遮挡原子仍需要超越 AFM 信号的推理能力。

### 2.4 P4：Composite 计算不完整

**代码证据**：

`train.py:471`：
```python
composite = compute_composite_score(
    rmsd=rmsd_mean,
    bottom_atom_score=bottom_recall_mean,
    bond_validity=bond_valid_mean,
    ring_preservation=0.0,  # TODO: ring preservation hard-coded to 0
    atom_count_accuracy=count_exact_mean,
    structure_similarity=struct_sim_mean,
)
```

`eval_recycling.py`：完全没有调用 `compute_composite_score()`。

**影响**：
- V13-Diff Composite=0.4926 被低估：少了 `0.15 × ring_preserve` 的贡献
- 修正后 V13-Diff Composite ≈ **0.657**（使用 version_comparison 一致数据源重算）
- V13+GNN Composite=0.6195（手动计算）→ 一致数据源重算为 **0.641**
- **修正后 V13-Diff Composite (0.657) > V13+GNN (0.641)**，与各项单指标一致
- 这说明之前 version_comparison 中 Diff (0.493) < GNN (0.620) 的结论是不公平比较导致的假象

### 2.5 P5：Type 准确率与 75% 目标的差距

**当前最佳**：cond_type_acc = 0.5969（V13-Diff），距离 75% 差 15.3%。

**瓶颈分析**：

| 元素 | vdW 半径 (Å) | AFM 区分度 |
|------|-------------|------------|
| C | 1.70 | 基准 |
| N | 1.55 | C-N 差 0.15Å，AFM 难区分 |
| O | 1.52 | C-O 差 0.18Å，AFM 难区分 |
| H | 1.20 | 明显小于 C，可区分 |
| F | 1.47 | 与 O 接近 |
| S | 1.80 | 与 C 接近 |

C/N/O 在 AFM 形貌上的物理相似性（vdW 半径差 <0.2Å）是当前技术的理论瓶颈。在模型空间（÷12.0）中，这些差异仅为 0.012-0.015，接近坐标精度的噪声水平。

---

## 第三部分：V14 改进方案

### 3.1 改进总览

| # | 改进项 | 解决问题 | 类型 | 预期收益 |
|---|--------|----------|------|----------|
| M1 | Denoiser + GNN 类型集成 | P2 | 推理代码 | Type Acc +2-5% |
| ~~M2~~ | ~~type_loss 动态加权~~ | ~~P1~~ | ~~训练代码~~ | **已验证无效，被 M7 替代** |
| M3 | 分离 type_head 梯度路径（type_adapter） | P1 | 模型架构 | TypeMatch 稳定性 |
| M4 | 修复 Composite 计算 | P4 | 评估代码 | 公平比较 |
| M5 | Type 混淆矩阵分析 + 元素级策略 | P5 | 评估+训练 | 识别可提升元素 |
| M6 | 改进 Bottom Recall 的训练策略 | P3 | 训练代码 | Bottom Recall ↑ |
| **M7** | **EDM*+γ 分离噪声调度** | **P1** | **模型架构** | **消除跷跷板根因** |

### 3.2a M7：EDM*+γ 分离噪声调度（核心改进）

**动机**：V14 首轮实验证明 M2（动态加权）完全无效——V13 前 7ep TypeMatch 下降 7.6%，V14 下降 8.0%。训练 type_loss 下降 3.1% 但评估 TypeMatch 下降 8.0%，证明是**过拟合而非梯度不足**。

**理论依据**：
- EDM (Hoogeboom et al., ICML 2022): 坐标和类型使用统一扩散空间，但需 `normalize_factors` 平衡
- EDM*+γ (后续改进): 为坐标和类型使用**不同的噪声调度**(learned SNR per modality)
- GradNorm (Chen et al., ICML 2018): 固定/线性权重对多任务学习效果有限，需自适应梯度平衡

**核心问题**：当前 type_head 接收的是从**噪声坐标 x_t** 提取的特征 `h`。在高噪声 timestep（t>500），`h` 中的类型信号被噪声淹没，导致 type_head 只能学到训练集的统计偏差（过拟合）。

**解决方案**：让 type_head 基于**重建的干净坐标 x_0_pred** 工作（等价于 type 的 γ→0）：

```
x_t (noisy) → Transformer → h → coord_head → eps_pred
                                                ↓ detach
                                    x_0_pred = (x_t - √(1-ᾱ)·eps) / √ᾱ
                                                ↓
                                    coord_embed(x_0_pred) → h_clean
                                                ↓
                        h + h_clean → cross_attn → type_adapter → type_head
```

**关键设计**：
1. `eps_pred.detach()` 阻断 type_loss 对 coord backbone 的梯度干扰
2. `h + h_clean`：backbone 上下文特征 + 干净坐标特征的融合
3. 移除 t<500 硬阈值，改用 SNR 软加权：所有 timestep 都贡献 type_loss，但 SNR 高（低噪声）的样本权重更大
4. 移除 M2 动态加权（已验证无效），type_weight 固定为 1.0

**与旧方案的本质区别**：
- M2（动态加权）：同样的噪声输入，只是改变 loss 权重 → 加剧过拟合
- M7（EDM*+γ）：改变 type_head 的**输入**（干净坐标而非噪声坐标）→ 从根本上消除噪声干扰

### 3.2 M1：Denoiser + GNN 类型集成

**动机**：V13 中 denoiser type_head（58.3%）整体优于 GNN（54.6%），但 GNN 在验证集上达到 68.2%，说明两者在不同样本/元素上各有优势。当前代码是 GNN 完全替换 denoiser（`pred_types = corrected`），损失了 denoiser 的正确预测。

**方案**：

```python
# 推理时融合 denoiser 和 GNN 的 type logits
def ensemble_type_prediction(denoiser_logits, gnn_logits, alpha=0.6):
    """加权融合两个模型的类型预测。
    alpha: denoiser 权重（>0.5 因为 denoiser 在 V13 更强）
    """
    fused_logits = alpha * F.softmax(denoiser_logits, dim=-1) \
                 + (1 - alpha) * F.softmax(gnn_logits, dim=-1)
    return fused_logits.argmax(dim=-1)
```

**实现位置**：`eval_recycling.py` 和 `visualize_val.py` 中 GNN 预测后的融合。

**alpha 搜索**：在验证集上搜索 α ∈ {0.3, 0.4, 0.5, 0.6, 0.7}，选 cond_type_acc 最高的。

**预期效果**：
- 如果 denoiser 和 GNN 的错误不完全重叠，融合应优于任一单模型
- 保守估计 Type Match 提升 2-5%
- 无需重新训练，仅修改推理代码

### 3.3 M2：跷跷板效应对策——type_loss 动态加权

**动机**：TypeMatch 在 Ep1-4 达到 0.567-0.570，之后持续下降至 0.477。coord_loss 随训练逐渐主导了 backbone 的特征分配。

**方案 A：type_loss 权重递增**

```python
# 在 compute_loss 或 train loop 中
type_weight = min(1.0 + 0.5 * (epoch / total_epochs), 2.0)
# Ep1: weight=1.0, Ep10: weight=1.25, Ep20: weight=1.5
total_loss = coord_loss + type_weight * type_loss + ...
```

随训练推进，逐步增大 type_loss 的权重，抵消 coord_loss 对特征空间的侵蚀。

**方案 B：type_loss 梯度停止实验**

在 coord_head 的梯度中 detach type_head 共享的 backbone 特征：
```python
# 在 denoiser forward 中
shared_feat = self.backbone(x_t, t_emb, c)
coord_out = self.coord_head(shared_feat)
type_out = self.type_head(shared_feat.detach() + shared_feat - shared_feat.detach())
# type_head 的梯度不会影响 backbone 前向计算，但仍可回传
```

**推荐方案 A**（更安全，无架构修改）。

### 3.4 M3：分离 type_head 梯度路径

**动机**：coord_head 和 type_head 共享 SE(3)-等变 Transformer backbone。type 预测在后期下降说明 backbone 被 coord_loss 主导。

**方案**：为 type_head 添加独立的浅层特征提取器：

```python
class SE3EquivariantDenoiser(nn.Module):
    def __init__(self, ...):
        ...
        # 现有 backbone
        self.backbone = ...  # shared Transformer layers
        self.coord_head = ...

        # 新增：type 专属特征层（不与 coord 共享梯度）
        self.type_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.type_head = nn.Linear(hidden_dim, num_atom_types)

    def forward(self, ...):
        shared_feat = self.backbone(...)
        coord_out = self.coord_head(shared_feat)

        # type 使用 adapter 路径，梯度回传到 adapter 和 backbone
        # 但 adapter 提供 type 专属的特征变换
        type_feat = self.type_adapter(shared_feat)
        type_out = self.type_head(type_feat)
```

**预期效果**：type_adapter 为 type 预测提供独立的特征子空间，减少与 coord 优化的直接竞争。

**风险**：增加参数量（2 × hidden_dim² ≈ 2 × 512² = 524K 参数，占模型总量 <3%），风险低。

### 3.5 M4：修复 Composite 计算

**问题**：
1. `train.py:471` `ring_preservation=0.0` 硬编码
2. `eval_recycling.py` 没有调用 `compute_composite_score()`

**修复**：

**train.py** — 在 eval 循环中计算 ring_preservation 并传入：
```python
# 在 validate_epoch 中计算 ring_preservation
from src.utils.metrics import compute_ring_preservation
ring_result = compute_ring_preservation(coords_pred, coords_gt, pred_types, gt_types, mask)
ring_pres_mean = ring_result["ring_preservation_mean"]

composite = compute_composite_score(
    rmsd=rmsd_mean,
    bottom_atom_score=bottom_recall_mean,
    bond_validity=bond_valid_mean,
    ring_preservation=ring_pres_mean,  # 修复：使用实际值
    atom_count_accuracy=count_exact_mean,
    structure_similarity=struct_sim_mean,
)
```

**eval_recycling.py** — 在最终指标汇总处添加 Composite 计算：
```python
from src.utils.metrics import compute_composite_score
composite = compute_composite_score(
    rmsd=np.mean(all_rmsd),
    bottom_atom_score=np.mean(all_bottom),
    bond_validity=np.mean(all_bond),
    ring_preservation=np.mean(all_ring),
    atom_count_accuracy=np.mean(all_count_exact),
    structure_similarity=np.mean(all_struct_sim) if all_struct_sim else 0.0,
)
print(f"Composite Score: {composite:.4f}")
```

### 3.6 M5：Type 混淆矩阵分析

**动机**：cond_type_acc=59.7% 意味着 40% 的原子被预测为错误类型。需要分析错误集中在哪些元素对上。

**方案**：

1. **混淆矩阵生成**：在 eval_phase1 中收集逐原子的 (gt_type, pred_type)，生成 10×10 混淆矩阵

2. **元素级准确率**：
```python
# 期望输出示例
# H: 85%  C: 72%  N: 35%  O: 40%  F: 20%  S: 55%  ...
# 混淆热点：N→C (25%), O→C (20%), F→O (15%)
```

3. **针对性策略**：
   - 如果 N↔C 混淆严重：增加 N 原子的 type_loss 权重（类别不平衡对策）
   - 如果 H 准确率已经很高：H 预测基本靠几何（最小 vdW 半径），可作为 anchor 约束邻近原子类型

### 3.7 M6：Bottom Recall 改进

**方案 A：Bottom 原子 Z 位置回归 loss**

对 Bottom 30% 原子的 Z 坐标施加额外 MSE loss：
```python
# 在 compute_loss 中
z_gt = gt_coords[:, :, 2]  # (B, N)
bottom_mask = z_gt < torch.quantile(z_gt[mask.bool()], 0.3)
bottom_z_loss = F.mse_loss(
    x_0_pred[:, :, 2][bottom_mask & mask.bool()],
    gt_coords[:, :, 2][bottom_mask & mask.bool()]
)
# 叠加到总 loss
loss += 0.5 * bottom_z_loss
```

**方案 B：Corrugation 感知采样**

训练时对高 corrugation 分子（Z 方向起伏大）过采样：
```python
# 在 DataLoader 中使用 WeightedRandomSampler
weights = [2.0 if corrugation > median else 1.0 for corrugation in all_corrugations]
sampler = WeightedRandomSampler(weights, num_samples=len(weights))
```

---

## 第四部分：执行计划

### 4.1 优先级排序

| 优先级 | 改进项 | 理由 | 预计耗时 |
|--------|--------|------|----------|
| **P0** | M4: 修复 Composite | 不需要重训练，立即影响评估准确性 | 0.5h |
| **P0** | M1: Denoiser+GNN 集成 | 不需要重训练，可能立即提升 Type Acc | 1h |
| **P1** | M5: 混淆矩阵分析 | 诊断性工作，指导后续优化方向 | 1h |
| **P2** | M2: type_loss 动态加权 | 需要重训练，但改动小 | 训练 3.5h + 代码 0.5h |
| **P2** | M3: type_head 分离 | 需要重训练，架构改动 | 训练 3.5h + 代码 1h |
| **P3** | M6: Bottom Recall | 收益不确定，受限于 AFM 物理极限 | 训练 3.5h + 代码 0.5h |

### 4.2 分阶段执行

**阶段一（推理优化，无需重训练）**：
1. M4: 修复 train.py 和 eval_recycling.py 的 Composite 计算
2. M1: 实现 Denoiser+GNN 集成推理，α 搜索
3. M5: 生成混淆矩阵，分析元素级错误分布

**阶段二（训练优化，需要重训练）**：
4. M2: 根据混淆矩阵结果，实施 type_loss 动态加权
5. M3: 添加 type_adapter 层
6. 重新训练 Phase 1 (20 epoch)

**阶段三（评估+可视化）**：
7. Phase 2 GNN 训练（如果 M1 集成效果好，可跳过独立 GNN 训练）
8. Phase 3 Recycling 评估（使用集成推理）
9. 可视化 + 分组报告 + version_comparison 更新

### 4.3 评估要求

- 所有 Composite 计算必须使用相同的 `compute_composite_score()` 函数
- 对同一测试集评估 Diff-only、GNN-only、Ensemble 三个版本
- 混淆矩阵按元素报告准确率和 F1
- Corrugation 分组评估延续 V13 的 P33/P67 动态分位数方法

---

## 第五部分：验证与风险

### 5.1 方案验证（对照历史数据）

| 假设 | 验证依据 | 结论 |
|------|----------|------|
| GNN 集成能提升 Type | V12: GNN 帮助（44.8→53.6%），V13: GNN 损害（58.3→54.6%）。因为是完全替换而非融合 | ✓ 融合可能优于任一单模型 |
| 跷跷板效应来自特征竞争 | V13 TypeMatch Ep1=0.570 → Ep20=0.477，同期 RMSD 稳定 0.21-0.25 | ✓ coord_loss 稳定但 type 下降，说明特征空间被 coord 占据 |
| Bottom Recall 受限于 AFM | V5b-V13 Bottom Recall 始终 3.9-13.6%，与架构/训练策略关系不大 | ✓ 物理极限，优先级降低 |
| Composite 被低估 | V13-Diff Composite=0.4926，ring_preservation=0.0 而实际=0.924 | ✓ 修正后为 0.657，且 Diff > GNN |
| C/N/O 混淆是 Type 瓶颈 | vdW 半径 C=1.70, N=1.55, O=1.52 Å，AFM 分辨率不足 | 待混淆矩阵验证 |

### 5.2 风险评估

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| 集成 α 搜索无显著提升 | 中 | M1 失败 | 尝试元素级选择策略 |
| type_loss 动态加权导致 RMSD 上升 | 中 | RMSD 退步 | 控制 type_weight 上限 ≤2.0 |
| type_adapter 增加过拟合 | 低 | Type 训练集高、测试集低 | 加 dropout=0.1 |
| 混淆矩阵显示 C/N/O 混淆无法改善 | 高 | P5 不可解 | 接受物理极限，论文中如实报告 |

### 5.3 目标值

| 指标 | V13 最佳 | V14 目标 | 依据 |
|------|----------|----------|------|
| RMSD | 0.2095 | ≤0.21 | 保持不退步 |
| Cond Type Acc | 0.5969 | **≥0.65** | 集成+动态加权 |
| Type Match | 0.5834 | **≥0.62** | 集成+type_adapter |
| Ring Preserve | 0.9240 | ≥0.92 | 保持不退步 |
| PMI Shape | 0.8413 | ≥0.84 | 保持不退步 |
| Composite | 0.657 (修正) | **≥0.68** | 修复计算+指标提升 |

---

## 附录：V5b-V13 全版本指标趋势

| 指标 | V5b | V7 | V8 | V10 | V12-Diff | V12+GNN | V13-Diff | V13+GNN |
|------|-----|----|----|-----|----------|---------|----------|---------|
| RMSD ↓ | 0.269 | 0.262 | 0.254 | 0.254 | **0.205** | 0.283 | 0.210 | 0.237 |
| Type Match ↑ | 0.485 | 0.541 | 0.505 | 0.487 | 0.448 | 0.536 | **0.583** | 0.546 |
| Cond Type ↑ | — | — | — | — | 0.469 | 0.530 | **0.597** | 0.571 |
| Coulomb ↑ | 0.009 | 0.223 | 0.442 | 0.405 | 0.392 | 0.528 | **0.579** | 0.514 |
| Bond Valid ↑ | 0.404 | 0.766 | **0.798** | 0.783 | 0.704 | 0.750 | **0.798** | 0.764 |
| Ring Pres ↑ | — | — | — | — | 0.907 | 0.909 | **0.924** | 0.913 |
| Bottom ↑ | 0.039 | 0.058 | 0.040 | 0.049 | 0.102 | **0.136** | 0.101 | 0.092 |
| Count Acc ↑ | 0.308 | **0.359** | 0.337 | 0.350 | 0.075 | 0.219 | 0.191 | 0.176 |
| Formula ↑ | — | 0.949 | 0.964 | 0.961 | 0.895 | 0.942 | 0.937 | **0.951** |
