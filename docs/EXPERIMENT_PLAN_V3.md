# AFM 3D 分子结构重建：V3 改进方案


---

# 第一部分：V2 实验问题诊断

## 1.1 V2 训练结果（60 Epoch 完成）

| 指标 | V2 最佳值 | V1 最佳值 | 对比 |
|------|----------|----------|------|
| RMSD | 0.255 (Ep50) | 0.339 (Ep44) | V2 优 25% |
| type_loss | 1.456 (仍在降) | 1.188 (停滞) | V2 降幅 5.5x |
| Count Accuracy | 44.5% | 44.4% | 持平 |
| Bond Validity | 86.1% | 91.0% | V1 优 |
| Bottom Recall | 6.4% | 10.7% | V1 优 |
| Val Loss 过拟合 | +2% | +20% | V2 大幅优 |
| 训练时间 | 22 小时 | 23.5 小时 | — |

## 1.2 核心问题识别

通过对验证样本的逐一分析，发现以下四个关键问题：

### 问题 A：原子类型准确率不足
- Type Match ≈ 45%，近一半原子类型预测错误
- 非氢原子（N, O, F, S 等）尤其差，模型倾向于将它们预测为 C
- 已使用逆频率 class_weight，但效果有限

### 问题 B：3D 形状失真
- 样本 #777：真实分子 10.6×8.7×1.8 Å（扁平），预测 8.8×6.6×8.8 Å（球形）
- Kabsch Score = 0.94 但形状根本不对，因为 Kabsch 的标尺（24Å）太宽松
- 模型丢失了分子的整体形状各向异性信息

### 问题 C：化合价检查过于宽松
- 只检查化合价 ≤ max，不检查 ≥ min
- 化合价 = 0 的孤立原子被当作"有效"
- 无法检测断键和分子碎片化

### 问题 D：环约束推理时未生效
- `generate()` 调用 `ddpm.sample()` 时未传入 `ring_info`
- 推理时 `use_ring_constraints = False`，完整的环投影代码从未执行
- 无法在推理时检测预测结构中的环（因为不知道哪些原子构成环）

### 问题 E：训练时间过长
- 22 小时/60 epoch，其中评估占 ~6-8 小时
- 每 epoch 评估 1000 样本 × 1000 步扩散采样

---

# 第二部分：改进方案

## 改进 1：提升原子类型预测准确率

### 2.1.1 研究动机与文献支撑

#### 证据来源：UniGEM (ICLR 2025)

> 原子类型可以从分子骨架/几何结构推断，将离散类型从连续扩散过程中分离可降低先验分布误差。

**UniGEM 的核心发现**：联合扩散坐标和类型时，离散类型的高斯噪声不合理；解耦后类型预测显著改善。

- 论文：[UniGEM: A Unified Approach to Generation and Property Prediction for Molecules](https://arxiv.org/abs/2410.10516)

#### 证据来源：GFMDiff (AAAI 2024)

> Geometric-Facilitated Loss (GFLoss) 通过化合价一致性约束间接提升类型预测——不同元素的化合价不同（C=4, N=3, O=2），强制模型学习这种区分。

- 论文：[Geometric-Facilitated Denoising Diffusion Model for 3D Molecule Generation](https://arxiv.org/abs/2401.02683)

#### 证据来源：Focal Loss (Lin et al. 2017, 化学应用 PMC 2025)

> 对已经分类正确的高频类（H, C）自动降权，聚焦于难分类的低频类（N, O, F, S）。相比静态 class_weight，Focal Loss 提供动态聚焦效果。

### 2.1.2 我们的方法

**方法 A：Focal Loss 替换 CrossEntropy（立即实施）**

```
现有: F.cross_entropy(logits, targets, weight=class_weight)
改为: FocalLoss(logits, targets, alpha=class_weight, gamma=2.0)

公式: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
- gamma=2.0: 当 p_t=0.9（已分类正确的H/C），权重 = 0.01x
- gamma=2.0: 当 p_t=0.1（难分类的N/O/F），权重 = 0.81x
- 效果: 自动聚焦于难分类样本，与现有 class_weight 协同
```

**修改文件**：`src/models/diffusion.py` — `compute_loss()` 中 type_loss 计算

**方法 B：化合价辅助损失 GFLoss（中期实施）**

```
输入: 预测坐标 x_0, 预测类型概率 p_type
输出: 化合价一致性损失

步骤:
1. 从 p_type 获取每个原子的期望化合价:
   expected_valence = sum(p_type[i] * max_valence[type]) for each type
2. 从 x_0 推断实际键数:
   actual_valence = count(dist(i,j) < covalent_radii_sum + tol)
3. 损失 = MSE(expected_valence, actual_valence)

效果: N(化合价3) vs C(化合价4) vs O(化合价2) 产生不同梯度
      迫使模型通过几何关系学习区分元素类型
```

**修改文件**：`src/models/diffusion.py` — 新增 `valence_consistency_loss()`

---

## 改进 2：修复 3D 形状失真

### 2.2.1 研究动机与文献支撑

#### 证据来源：MLConformerGenerator (Digital Discovery, 2025)

> 使用惯性张量的主成分特征值作为分子形状描述符。扁平分子的最小特征值接近 0，球形分子三个特征值接近相等。

- 论文：[Moment of inertia as shape descriptor](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00318k)

#### 证据来源：DiffSMol (Nature Machine Intelligence, 2025)

> 在扩散采样的每个去噪步中，计算形状指标的梯度并修正样本方向，实现无需重训的形状引导。

- 论文：[Shape-conditioned diffusion models with guidance](https://www.nature.com/articles/s42256-025-01030-w)

### 2.2.2 我们的方法

**惯性张量特征值损失（训练时）**

```python
def shape_loss(pred_coords, gt_coords, mask):
    """惩罚预测分子与真实分子的形状差异。

    惯性张量特征值 = 三个主轴方向的扩展程度：
      扁平分子: eig ≈ [0.001, 0.5, 0.6]  (一个方向极小)
      球形分子: eig ≈ [0.3, 0.3, 0.3]    (三个方向相等)

    样本 #777 的问题直接被这个损失捕获:
      GT  eig = [小, 大, 大] → 扁平
      Pred eig = [大, 大, 大] → 球形 → MSE 很大
    """
    losses = []
    for b in range(B):
        valid = mask[b].bool()
        # 中心化
        p = pred_coords[b][valid] - pred_coords[b][valid].mean(0)
        g = gt_coords[b][valid] - gt_coords[b][valid].mean(0)
        # 协方差矩阵特征值 = 简化版惯性张量
        pred_eig = torch.linalg.eigvalsh(p.T @ p / valid.sum())
        gt_eig = torch.linalg.eigvalsh(g.T @ g / valid.sum())
        losses.append(F.mse_loss(pred_eig, gt_eig))
    return torch.stack(losses).mean()
```

**修改文件**：
- `src/models/diffusion.py` — `compute_loss()` 中新增 `shape_loss` 项
- `src/train.py` — 总损失中加入 shape_loss（权重 0.5）

---

## 改进 3：完善化合价与连通性检查

### 2.3.1 研究动机

#### 证据来源：EDM (ICML 2022), MiDi (ECML 2023)

> EDM 定义 "atom stability" 和 "molecule stability"：前者检查每个原子化合价是否在合理范围内（包括上下限），后者要求所有原子都稳定且分子为单一连通分量。MiDi 通过联合生成键矩阵将 molecule stability 从 6% 提升到 92%。

#### 证据来源：RDKit rdDetermineBonds

> `rdDetermineBonds.DetermineBonds()` 基于 xyz2mol 算法，从 3D 坐标推断完整键图（含键级），在 PubChem 分子上 >90% 完美匹配率。

### 2.3.2 我们的方法

**双向化合价检查 + 连通性检查（评估时）**

```
当前 (V2):
  valid = (valence <= max_valence)  → 化合价=0 也算有效

改进 (V3):
  valid = (min_valence <= valence <= max_valence)
  + 连通性检查: 分子必须是单一连通分量
  + 碎片率: fragment_count / expected_1 的比率

最小化合价表:
  H: ≥1, C: ≥2, N: ≥1, O: ≥1, F: ≥1, S: ≥1, P: ≥1, Cl: ≥1
```

**修改文件**：`src/utils/metrics.py` — 修改 `_valence_validity()` 和 `compute_structure_similarity()`

---

## 改进 4：推理时启用环约束

### 2.4.1 研究动机

#### 证据来源：Predict-Project-Renoise (PPR, arXiv 2026)

> 在每个去噪步中：(1) 预测 x₀，(2) 在 x₀ 上检测约束违反，(3) 投影到满足约束的空间，(4) 重新加噪。这个范式让约束可以在推理时执行而无需 GT 信息。

- 论文：[Predict-Project-Renoise](https://arxiv.org/html/2601.21033v1)

#### 证据来源：ConStruct (NeurIPS 2024)

> 将图扩散重构为约束生成问题，在每个反向步中投影到满足结构约束（连通性、平面性）的空间。

- 论文：[ConStruct](https://arxiv.org/abs/2406.17341)

### 2.4.2 我们的方法

**推理时自动环检测 + 平面投影**

```
当前 (V2):
  generate() → ddpm.sample(c, n_atoms)  # ring_info=None → 不执行

改进 (V3):
  在 ddpm.sample() 内部，不依赖 GT ring_info，而是：
  1. 每隔 200 步，对当前 predicted x₀ 用距离阈值推断键
  2. 用 BFS/DFS 检测 5/6 元环（复用已有的 ring_detection 模块）
  3. 对检测到的环原子执行平面性投影
  4. 继续去噪

  具体步骤（在去噪循环的 step 200-1000）:
    x₀_pred = denoiser(x_t, t, c)
    if t_idx in [200, 400, 600, 800]:
        rings = detect_rings_from_coords(x₀_pred, pred_types)
        x₀_pred = project_to_planar(x₀_pred, rings)
    x_t = renoise(x₀_pred, t-1)
```

**修改文件**：
- `src/models/diffusion.py` — 修改 `sample()` 方法，加入自动环检测逻辑
- `src/train.py` — `generate()` 不再需要传 ring_info

---

## 改进 5：训练时间压缩（22h → <10h）

### 2.5.1 研究动机

#### 证据来源：ELPD (ICLR GEM Workshop, 2024)

> 对 GeoLDM 应用渐进蒸馏，实现 7.5x 采样加速。关键发现：蒸馏到 125 步 + 随机 DDPM 采样（非 DDIM）给出最佳质量-速度平衡。

#### 证据来源：DPM-Solver++ (NeurIPS 2022)

> 高阶 ODE 求解器，15-20 步即可达到 1000 步 DDPM 的质量。可作为评估时采样器的直接替换。

#### 证据来源：EDM2 (Karras et al., 2024)

> EMA 是扩散模型不可或缺的组件。Post-hoc EMA 调优可以在不重训的情况下找到最优 EMA 衰减率。

### 2.5.2 我们的方法

**三项加速措施，预计 22h → 8.5h**

| 措施 | 实现 | 预计节省 | 风险 |
|------|------|---------|------|
| **A. 评估用 DDIM 100 步** | 仅修改评估采样器，训练不变 | ~4h | 低 |
| **B. 减少评估开销** | 每 epoch 200 样本，每 5 epoch 全量 1000 | ~1.5h | 低 |
| **C. BF16 混合精度训练** | ViT encoder + denoiser 用 BF16，坐标回归保持 FP32 | ~6h (吞吐量翻倍) | 低-中 |

```
时间估算:
  原始: 22h = 12h(训练) + 8h(评估) + 2h(其他)
  A:    22h - 4h = 18h        (评估 5min→30s/epoch)
  A+B:  18h - 1.5h = 16.5h    (评估量减少)
  A+B+C: 16.5h / 2 ≈ 8.5h    (BF16 吞吐翻倍)
```

**修改文件**：
- `src/models/diffusion.py` — `sample()` 中添加 DDIM 采样模式
- `src/train.py` — 评估函数使用 DDIM，减少评估样本数，添加混合精度

---

# 第三部分：总损失函数

```python
# V3 总损失
loss = (
    coord_loss                          # 坐标重建 (1.0)
    + 0.3 * focal_type_loss             # Focal Loss 原子类型 (改进1)
    + 1.0 * count_loss                  # 原子数预测
    + 0.01 * retrieval_loss             # 正则化
    + 0.1 * constraint_loss             # 键长/键角/平面性 (Stage 2+)
    + 0.5 * shape_loss                  # 惯性张量形状 (改进2, 新增)
    + 0.2 * valence_consistency_loss    # 化合价一致性 (改进1B, 新增)
)
```

| 损失项 | V1 | V2 | V3 | 变化 |
|--------|-----|-----|-----|------|
| coord_loss | 1.0 | 1.0 | 1.0 | — |
| type_loss | 0.1 CE | 0.3 CE+class_weight | **0.3 Focal+class_weight** | Focal Loss |
| count_loss | 0.5 | 1.0 | 1.0 | — |
| retrieval_loss | 0.05 | 0.01 | 0.01 | — |
| constraint_loss | 0.1 | 0.1 | 0.1 | — |
| **shape_loss** | — | — | **0.5** | 新增 |
| **valence_loss** | — | — | **0.2** | 新增 |

---

# 第四部分：修改文件清单

| 文件 | 修改内容 | 对应改进 |
|------|----------|---------|
| `src/models/diffusion.py` | Focal Loss、shape_loss、valence_consistency_loss、DDIM 采样 | 1A, 2, 5A |
| `src/utils/metrics.py` | 双向化合价检查、连通性检查、修正 Kabsch 标尺 | 3 |
| `src/train.py` | 新损失项集成、BF16 混合精度、评估加速 | 1, 2, 5 |

---

# 第五部分：实施优先级

| 优先级 | 改进项 | 实现难度 | 预期效果 |
|--------|--------|---------|---------|
| P0 | 评估加速 (DDIM + 减少样本) | 低 | 训练时间 -5.5h |
| P0 | BF16 混合精度 | 低 | 训练时间 ÷2 |
| P1 | Focal Loss | 低 | 类型准确率提升 |
| P1 | 惯性张量 shape_loss | 低 | 修复形状失真 |
| P1 | 双向化合价 + 连通性检查 | 低 | 评估更准确 |
| P2 | 推理时环检测 + 投影 | 中 | 环结构改善 |
| P2 | 化合价一致性 GFLoss | 中 | 类型准确率进一步提升 |

---

# 第六部分：参考文献

| 论文 | 发表 | 相关改进 |
|------|------|---------|
| UniGEM | ICLR 2025 | 解耦类型预测 |
| GFMDiff | AAAI 2024 | 化合价辅助损失 GFLoss |
| MiDi | ECML PKDD 2023 | 离散扩散 + 联合键生成 |
| DiGress | ICLR 2023 | 离散图扩散 |
| MLConformerGenerator | Digital Discovery 2025 | 惯性张量形状描述符 |
| DiffSMol | Nature Machine Intelligence 2025 | 形状引导采样 |
| GCDM | Communications Chemistry 2024 | 几何完备扩散架构 |
| ConStruct | NeurIPS 2024 | 约束图扩散（连通性） |
| PPR | arXiv 2026 | Predict-Project-Renoise 范式 |
| PDM | NeurIPS 2024 | 投影扩散模型 |
| ELPD | ICLR GEM Workshop 2024 | 渐进蒸馏加速 7.5x |
| EC-Conf | J. Cheminformatics 2024 | 一致性模型单步采样 |
| DPM-Solver++ | NeurIPS 2022 | 快速 ODE 求解器 |
| EDM2 | arXiv 2024 | EMA 优化训练动态 |
| EDM | ICML 2022 | 等变扩散基线 |
| MolDiff | ICML 2023 | 原子-键不一致性 |
| ACS Omega 2025 | ACS Omega | 3D 分子生成综合评测 |
| Focal Loss | ICCV 2017 | 类别不平衡聚焦损失 |
| rdDetermineBonds | RDKit | xyz→键图重建 |
