# V1-V6 系统性反思报告

## 一、各版本实验结果汇总

### 1.1 全版本关键指标对比（Final Test Set）

| 版本 | Epochs | RMSD | Type Match | Coulomb | Bottom Recall | Bond Valid | Count Acc | Composite |
|------|--------|------|-----------|---------|---------------|-----------|-----------|-----------|
| V1 | 60 | 1.830 | — | — | 7.2% | 91.0% | 42.9% | 0.219 |
| V2 | 60 | **0.255** | 43.6% | 0.375 | 5.0% | 86.1% | 44.5% | **0.514** |
| V3 | 60 | 1.038 | 27.2% | 0.249 | 0.1% | 8.6% | 36.3% | 0.236 |
| V4 | 50 | 1.076 | 8.4% | 0.218 | 0.1% | 5.6% | 35.9% | 0.220 |
| V5b | 50 | **0.269** | **48.5%** | 0.009 | **3.9%** | **40.4%** | 30.8% | 0.418 |
| V6 | 70 | 0.571 | 28.6% | 0.021 | 0.3% | — | 34.8% | 0.363 |

> 注：Coulomb 指标受评估方式影响巨大，见 §2.3 分析。

### 1.2 Per-Sample 分析揭示的真实水平（100 test samples, predictions 文件）

| 指标 | V2 | V5b | 说明 |
|------|-----|------|------|
| Coulomb mean | 0.505 | **0.513** | 远高于 Final Test 报告值 |
| Coulomb median | 0.579 | **0.599** | 中位数更高 |
| Type Match mean | 0.483 | **0.512** | 接近 50% |
| Type Match median | 0.484 | **0.500** | |
| Count MAE | **0.74** | 0.91 | V2 略好 |

**关键发现**：V5b 的 per-sample Coulomb 实际为 0.513，但 Final Test（1000 samples）报告为 0.009。原因见 §2.3。

---

## 二、各版本改进失效的根因分析

### 2.1 V2→V3：Focal Loss 灾难

**V3 方案**：引入 Focal Loss（γ=2.0）替代标准 CE，参考 UniGEM(ICLR 2025)、GFMDiff(AAAI 2024) 论文。

**结果**：Type Match 从 V2 的 43.6% **下降到 27.2%**，RMSD 从 0.255 **退化到 1.038**。

**失败原因（带数据）**：
```
数据集类型分布：H ≈ 35%, C ≈ 45%, N ≈ 11%, O ≈ 8%, 其他 < 1%
inverse_freq 权重：H=0.21, C=0.29, N=0.88, O=2.30, F=10.0
Focal Loss (γ=2.0) 对已分类正确的 H/C 额外衰减：
  H 有效梯度 = 0.21 × (1-0.9)^2 = 0.0021（几乎为零）
  O 有效梯度 = 2.30 × (1-0.1)^2 = 1.863（正常）
  → O 的梯度是 H 的 887 倍
```

引用论文中的数据集类别远比我们平衡（UniGEM 处理的是分子图生成，类别分布均匀）。我们的 H+C 占 80% 的极端不平衡下，Focal Loss 完全杀死了主要类别的学习。

**同时 V3 的 RMSD 退化到 1.038 不是 Focal Loss 导致的，而是 DDIM 采样 bug**：DDIM 从 t=100 开始（仅 3% 噪声），alpha_cumprod[100]≈0.97，x_0_pred 中除以 sqrt(0.97) 不产生爆炸，但从不完整的噪声范围去噪会导致坐标偏移。此 bug 在 V5 中被修复。

### 2.2 V3→V4：工程优化但未解决核心问题

**V4 方案**：数据加载优化（num_workers 0→4）、batch_size 增大（64→128）、DDIM 步数优化、早停策略。

**结果**：Type Match 从 27.2% **继续下降到 8.4%**，RMSD 维持 1.076。

**失败原因**：
- V4 没有修复 Focal Loss 问题（V3 引入），反而将 γ 从 2.0 提高到 3.0，**加剧了 H/C 梯度消失**
- V4 的工程优化（加载速度、batch size）本身是有效的，但核心模型问题未解决
- 课程学习（先训小分子）被提出但对 type 问题无帮助

**数据证据**：V4 Final Test Type Match = 8.4%，分析预测分布发现模型将所有原子预测为 O/N，H 和 C 完全消失——与 V5 中的诊断一致。

### 2.3 评估指标的系统性问题

#### 问题 A：Coulomb 指标的"大分子崩塌"

V5b per-sample 数据揭示了 Coulomb 指标的真实分布：

| 分子大小 | 样本数 | Coulomb mean | Type Match mean |
|---------|--------|-------------|----------------|
| n ≤ 15 | 6 | 0.549 | 0.404 |
| 15 < n ≤ 25 | 43 | 0.469 | 0.488 |
| 25 < n ≤ 40 | 51 | 0.546 | 0.544 |

predictions 中最大分子只有 38 个原子，Coulomb 均值 0.513。但 Final Test 涵盖 n>40 的大分子后，均值骤降到 0.009。

**原因**：Coulomb 矩阵 `C_ij = Z_i × Z_j / |r_i - r_j|` 对大分子有三重放大效应——原子数错→矩阵尺寸不匹配（零填充）；类型错→原子序数 Z 错（H=1 vs C=6 差 6 倍）；坐标错→距离分母错。大分子特征值向量更长，L2 距离自然更大。

**结论**：Coulomb 指标**不适合用平均值监控训练**。应分大/小分子报告，或改用中位数。

#### 问题 B：compute_rmsd 的零坐标 bug

```python
# metrics.py line 45-46
p = pred_coords[b, :n]   # n = gt_mask.sum() = n_gt
g = gt_coords[b, :n]
```

当 `n_pred < n_gt` 时，pred 的第 n_pred~n_gt 个位置是零坐标（采样时 `x_t * mask` 清零），匈牙利匹配将零坐标分配给真实 GT 原子，膨胀 RMSD。

**但此 bug 在所有版本中一致存在，版本间对比仍然公平。**

#### 问题 C：Bottom Recall 的阈值过于苛刻

```
distance_threshold = 0.1（归一化）= 1.2 Å（实际距离）
当前最佳 RMSD = 0.255（归一化）= 3.06 Å（实际距离）
```

要求预测原子在 1.2 Å 内**且类型正确**，而模型平均误差约 3 Å。即使 V5b 这个最好的模型，Bottom Recall 也只有 3.9%。此指标在当前精度下几乎无区分度。

#### 问题 D：type_match 是原子级指标，非分子级

当前 type_match 要求匈牙利匹配后**每个原子类型正确**。但：
1. C 和 N 在 3 邻居环境下坐标差异仅 0.006 归一化（0.07 Å），denoiser 无法从坐标区分
2. 即使坐标完美，仅靠邻居数推断类型的理论上限约 68%

**当前系统完全缺少分子级指标**：没有"分子式相似度"（如预测 C₆H₆O vs GT C₆H₅NO 的向量对比）、没有"类型分布匹配度"。

### 2.4 V5b→V6：过度工程化导致全面退化

**V6 同时引入 7 个改动**：

| 改动 | 预期效果 | 实际效果 | 原因 |
|------|---------|---------|------|
| TypeNet 解耦 | Type Match 提升 | RMSD 退化 0.25→0.52 | 去掉 denoiser type_head 削弱坐标质量 |
| Cross-attention patches | 更丰富的空间信息 | 未证实有效 | 增加训练负担，参数量翻倍 |
| Distance matrix loss | 保持键距 | 未证实有效 | 仅 t<300 生效，覆盖面有限 |
| Valence consistency loss | 类型-化合价一致性 | 微弱效果 | 0.2 权重太小 |
| Connectivity projection | 防止碎片化 | 仅采样末尾生效 | 10% 的步数太少 |
| Depth weight | 学习 AFM 层权重 | 未证实有效 | 参数极少，影响有限 |
| Bottom 5x weight | 改善底部原子 | Stage 3 后 RMSD 下降 | 但牺牲了顶部原子精度 |

**核心失误**：去掉 denoiser 的 type_head。

```
V5b denoiser: coord_embed → transformer → coord_head + type_head
              type_head 提供辅助梯度 → transformer 特征包含化学语义

V6 denoiser:  coord_embed → transformer → coord_head（仅此）
              无辅助梯度 → transformer 特征退化为纯几何信息
```

数据证据：
- V5b 参数量 ~25M，RMSD=0.255
- V6 参数量 51.5M（翻倍），RMSD=0.519（翻倍退化）
- 更大模型+同量数据+同量 epoch = 欠拟合

**TypeNet exposure bias**：
```
训练时：TypeNet(GT 坐标, ...) → type_loss   （看到完美坐标）
推理时：TypeNet(生成坐标, ...) → types       （看到 RMSD=0.52 的噪声坐标）
→ 经典的 train/test distribution shift
```

### 2.5 环结构：六个版本从未在推理中启用

| 组件 | 状态 | 位置 |
|------|------|------|
| detect_rings（BFS/DFS 检测 5/6 元环） | 已实现 ✓ | ring_detection.py |
| pad_ring_info（预处理） | 已实现 ✓ | dataset.py line 299-329 |
| planarity_penalty（训练损失） | Stage 2+ 启用 ✓ | constraints.py |
| _project_ring_constraints（Procrustes 对齐） | **已实现但从未调用** ✗ | diffusion.py line 543 |
| generate() 传入 ring_info | **从未实现** ✗ | train.py line 227 |
| ring_preservation 评估指标 | **硬编码 0.0** ✗ | train.py line 473 |

从 V3 方案中就计划了"推理时自动环检测 + 平面投影"，但 V3/V4/V5/V5b/V6 的 `generate()` 都没有传入 `ring_info`。这意味着 Procrustes 对齐——一个可以直接改善局部几何质量的强约束——在 6 个版本中完全是死代码。

---

## 三、Type Match 50% 天花板的根本原因

### 3.1 不是 RMSD 的问题

| 版本 | RMSD | Type Match |
|------|------|-----------|
| V2 | 0.255 | 43.6% |
| V5b | 0.255 | 48.5% |
| V6 | 0.519 | 28.6% |

V2 和 V5b 的 RMSD 几乎相同（0.255），但 Type Match 只差 5%。说明 RMSD 不是 Type Match 的决定性因素。

### 3.2 真正的瓶颈：信息不足+指标定义不匹配

**信息瓶颈**：denoiser 从噪声坐标 x_t + 全局条件向量 c 推断类型。在 C/N/O 的键长差异只有 0.07 Å（归一化 0.006）的情况下，仅靠坐标邻居数推断的理论上限约 68%：

```
4 邻居 → 必定是 C (sp3)                → 100% 准确
3 邻居 → C(sp2) 或 N(sp3)              → ~70% 准确（C 占多数）
2 邻居 → C(sp), N(sp2), O(sp3), S      → ~50% 准确
1 邻居 → H, F, Cl, Br, I, O(-OH)       → ~90% 准确（H 占绝大多数）
加权上限 ≈ 0.35×0.90 + 0.45×0.70 + 0.11×0.30 + 0.08×0.40 ≈ 0.68
```

**指标定义不匹配**：type_match 是**原子级**指标（匈牙利匹配后逐原子对比），但用户实际需要的是**分子级**指标——分子的整体元素组成、各类型分布、化学键拓扑是否相似。

### 3.3 缺失的指标维度

| 用户关心的 | 现有指标 | 问题 |
|-----------|---------|------|
| 分子形状相似 | Kabsch RMSD | ✓ 可用 |
| 元素组成正确 | **无** | 没有分子式对比 |
| 各类型分布合理 | **无** | 没有类型分布匹配 |
| 化学键有效 | Bond Validity | 阈值可能偏紧 |
| 化合价合理 | Valence Validity | ✓ 可用 |
| 原子数正确 | Count Accuracy | ✓ 可用 |

---

## 四、核心结论

1. **有效的改进只有三个**：V2 的结构化评估 + 增强 AtomCountHead；V5 的噪声范围一致性修复 + x_0 prediction；V5b 的 sqrt(inv_freq) 类别权重修复

2. **有害的改进有两个**：V3 的 Focal Loss（杀死 H/C 学习）；V6 的去除 denoiser type_head（削弱坐标质量）

3. **从未生效的功能有一个**：推理时环结构约束（6 个版本都是死代码）

4. **评估体系存在三个系统性问题**：Coulomb 对大分子崩塌；Bottom Recall 阈值过苛；缺少分子级指标

5. **Type Match 的真正天花板不在 RMSD，而在信息源和指标定义**：仅靠坐标的理论上限约 68%，需要新的信息源（AFM 像素级特征、环结构先验）和新的评估方式（分子式相似度、类型分布匹配）
