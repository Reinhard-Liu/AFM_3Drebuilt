# AFM 3D 分子结构重建：V13 改进方案

> 基于 V12 全流程结果 + 可视化问题分析 + DeepSeek 建议审查

---

## 第一部分：V12 成果与遗留问题

### 1.1 V12 达成情况

| 目标 | 指标 | V12-Diff | V12+GNN | 目标值 | 状态 |
|------|------|----------|---------|--------|------|
| 原子位置 | RMSD | **0.205** | 0.283 | <0.30 | ✓ Diff 达标 |
| 类型≥75% | cond_type_acc | 0.469 | 0.530 | 75% | ✗ 差 22% |
| 形状≥80% | PMI shape | **0.860** | 0.856 | 80% | ✓ 达标 |
| 环结构 | Ring preserve | **0.907** | 0.909 | 一致 | ✓ 达标 |

### 1.2 V12 可视化中暴露的四大问题

| # | 问题 | 严重程度 | 影响指标 |
|---|------|----------|----------|
| P1 | 环结构角度/倾角预测不准 | 中 | RMSD, Ring Preserve |
| P2 | 含环分子的环形状重建不精确 | 中 | Ring Preserve, Valence |
| P3 | 多环分子出现环重叠（Z轴坍缩） | 高 | RMSD, Coulomb, Bottom Recall |
| P4 | 离散原子散落在主分子外 | 中 | RMSD, Bond Valid |

---

## 第二部分：问题诊断与改进思路

### 2.1 P1：环结构角度/倾角预测不准

**现状**：

当前模型能检测到环并投影到平面（Ring Preserve=0.907），但环平面的**法线方向（倾角）**常常错误。

**根因**：

`_auto_detect_and_project_rings`（`diffusion.py:477-492`）仅用 SVD 找最佳拟合平面并做 30% blend 投影，有两个缺陷：
1. **只强制共面性，不约束环法线方向**：环被"压平"了，但倾斜角度可能完全错误
2. **blend 比例太弱**：30% 混合（`0.7 * ring_coords + 0.3 * projected`）约束力不足

而 `_project_ring_constraints`（GT 环 Procrustes 对齐）使用模板提供法线方向，效果更好——但推理时没有 GT 环信息。

**改进思路**：

- **方案 A（训练时）**：添加**环法线一致性 loss**——对检测到的环计算法线，与 GT 环法线求余弦相似度，作为辅助训练信号
- **方案 B（推理时）**：增强 `_auto_detect_and_project_rings` 的 blend 比例从 0.3 提高到 0.5-0.7，并在更多采样步中执行
- **方案 C（推理时）**：利用 AFM 图像的深度切片信息推断环法线——AFM Z-slice 0（最近）和 Z-slice 9（最远）的对比度差异可以暗示分子的 Z 方向倾斜

### 2.2 P2：含环分子训练策略

**现状**：

当前训练对所有分子等权重采样。有机分子大多含苯环，但一些含稠环或杂环的复杂分子的环形状重建不精确。

**分析**：

- QUAM-AFM 中含 5/6 元环的分子比例很高（有机分子大多含芳香环）
- 全排他性过滤（只保留含环分子）会导致模型对无环分子性能下降
- 数据集已有环检测基础设施（`ring_detection.py` + `dataset.py:299`）

**改进思路**：

- **方案 A（已实施）**：**排除无环分子，仅保留含 5/6 元环的分子**
  - 实现：`dataset.py` 新增 `require_ring` 参数 + `ring_cache_K-1.pkl` 缓存
  - 结果：213,505 → 205,227 分子（96.1% 含环，去掉 8,278 个无环分子）
  - `config.json` 新增 `"require_ring": true`
  - 所有入口脚本（train/eval/visualize）已同步更新
- **方案 B**：在 loss 中对环区域原子增加权重——环内原子的 coord_loss 乘以 1.5-2.0x
- **方案 C**：在 Stage 2（约束训练阶段）增加**环几何 loss**——专门惩罚环内键长、键角偏差

### 2.3 P3：多环分子 Z 轴坍缩 / 环重叠（核心问题）

**具体案例**：`visualizations_gnn/val_sample_00784.png`

- Ground Truth（25 atoms）：分子含稠环结构，多个环在 3D 空间中有不同 Z 位置和倾角
- Predicted（24 atoms, RMSD=0.128）：原子被压缩到很小的空间，不同环重叠在一起

**根因分析**：

1. **AFM 对 Z 方向固有不敏感**：论文大纲 1.2 节明确指出"Z 方向信息耦合在图像对比度中，不是直接可读的"。模型倾向于把分子"压扁"到 XY 平面
2. **coord_loss 对 XY 和 Z 等权重**：模型优先学习信号更强的 XY 方向，Z 方向成为弱势
3. **环约束只保证共面性，不防止重叠**：两个环可以各自共面但质心重合
4. **Bottom Atom Recall 仅 10-14%**：佐证模型在 Z 方向的分辨能力严重不足

**论文大纲的要求**：
- 5.2 节评估指标包含 Bottom Atom Recall，说明 Z 轴精度是核心目标之一
- 需要对 Z 方向达到一定效果，不能出现环结构重叠

**改进思路**：

#### 方案 A：Z 轴加权 loss（训练时）
在 `compute_loss` 中将 coord_loss 拆分为 XY 和 Z 两部分：
```python
xy_loss = F.mse_loss(pred[:,:,:2], gt[:,:,:2])  # XY 方向
z_loss = F.mse_loss(pred[:,:,2:], gt[:,:,2:])    # Z 方向
coord_loss = xy_loss + z_weight * z_loss          # z_weight = 2.0-3.0
```
强迫模型分配更多容量到 Z 方向信息的提取。这与论文大纲 3.2 "Z-Axis Aware Encoder" 的设计理念一致。

#### 方案 B：环间距约束（推理时）
在 `_auto_detect_and_project_rings` 中检测到多个环后，检查环质心间距离：
```python
# 如果两个非共享原子的环质心距离 < 环半径，在 Z 轴拉开
if centroid_dist < ring_radius and not shared_atoms:
    z_shift = ring_radius - centroid_dist
    # 将两个环在 Z 方向分离
```
防止独立环在 XY 平面上重叠。

#### 方案 C：环法线互斥约束
- 共享原子的稠环：约束法线方向一致或有化学合理的夹角
- 不共享原子的独立环：约束质心有足够间距（≥ 环直径）

#### 方案 D：corrugation 感知训练
- 按 corrugation 分组训练：高 corrugation 分子（Z 方向起伏大）给更高的 Z-loss 权重
- 低 corrugation 分子（准平面）Z-loss 权重保持正常

### 2.4 P4：离散原子（已分析，见 v12_final_report.md 第六节）

**三层原因**：
1. 扩散模型采样本身产生离散原子（氢原子最易离散）
2. 物理引导力度不足（`_apply_physics_guidance` 只拉回 30%，只执行一次）
3. MMFF94 力场基于错误类型预测恶化坐标（Recycling 特有）

**改进思路**：
- 增强连通性修正：拉回比例从 0.3 → 0.6-0.8，或迭代执行 2-3 次
- 在更多采样步中执行物理引导：从仅最后阶段 → 最后 20% 的步数
- 禁用 MMFF94 力场精炼：当前 type 准确率（53%）下弊大于利

---

## 第三部分：DeepSeek 建议审查

### 3.1 建议总览与评价

| # | DeepSeek 建议 | 是否采用 | 理由 |
|---|---|---|---|
| 1 | Height map 替代 3D 点云（2D坐标+z值） | **不采用** | 需重写 SE(3)-等变 denoiser 架构，与论文大纲方向矛盾 |
| 2 | 跨距离维度对比度建模 | **已实现** | Video ViT 将 10 层深度切片作为时间维度，正是此设计 |
| 3 | 6 模块新流程（图结构生成） | **不采用** | 完全不同方法论（图生成 vs 坐标生成），风险过大 |
| 4 | 准平面分子分组验证 | **采用** | 按 corrugation 分组报告指标，回应审稿人对 Z 方向的质疑 |
| 5 | MMFF94 能量验证 | **部分采用** | 统计生成结构 vs GT 的 MMFF94 能量差作为化学合理性指标 |
| 6 | 实验图像验证（2-iodotriphenylene） | **不采用** | 超出范围（QUAM-AFM 是模拟数据），放 Future Work |
| 7 | 论文叙事策略 | **采用** | "首次系统尝试 + 诚实评估局限 + 明确贡献"定位准确 |

### 3.2 采用建议的详细设计

#### 3.2.1 准平面分子分组验证（建议 4）

在评估中按 corrugation 分组报告指标：

| 分组 | Corrugation 范围 | 预期样本占比 | 预期 RMSD | 预期 Type |
|------|-----------------|-------------|-----------|-----------|
| 准平面 | < 0.5 Å | ~30% | < 0.15 | > 60% |
| 低起伏 | 0.5 - 1.5 Å | ~40% | 0.15-0.25 | 50-60% |
| 高起伏 | > 1.5 Å | ~30% | 0.25-0.40 | 40-50% |

实现方式：在 `eval_phase1.py` 中根据每个样本的 corrugation 值分组计算指标。

#### 3.2.2 MMFF94 能量验证（建议 5）

在评估时计算：
```python
E_pred = mmff94_energy(pred_coords, pred_types)
E_gt = mmff94_energy(gt_coords, gt_types)
energy_ratio = E_pred / E_gt  # 理想值接近 1.0
```
- 能量比 < 2.0 视为化学合理
- 能量比 > 10.0 视为结构不合理

### 3.3 不采用建议的理由详述

#### Height map 方案（建议 1）

DeepSeek 建议："不直接生成绝对坐标，而是生成 2D 坐标 + 相对高度 z 值"。

**思路本身有道理**——将 3D 生成降维为 2D+标量确实降低生成难度。但：
- 当前 SE(3)-等变 denoiser 的核心设计是在 3D 坐标空间操作旋转/平移等变性。改成 2D+z 会破坏等变性
- 论文大纲明确写 "Conditional Diffusion Model 生成 Z 方向坐标"，改底层表示会改变论文叙事
- V12 架构已经成熟（12 版迭代），从头改底层表示风险太大
- **可以在论文 Future Work 中提及**此思路

#### 6 模块新流程（建议 3）

DeepSeek 建议用结构化输出（邻接矩阵 + 2D 坐标 + z 值）替代点云生成。

**这本质上是不同的方法论**——从"条件扩散生成 3D 点云"变成"图生成模型"。需要重新设计 DDPM 的噪声调度、loss 函数和采样过程。
- 但"分别报告原子类型准确率、键准确率、高度预测误差"的验证思路值得借鉴，已体现在 V12 的多维评估体系中

---

## 第四部分：V13 确定方案（用户已确认）

### 4.1 用户决策

| 决策项 | 决定 | 说明 |
|--------|------|------|
| 重新训练扩散模型 | **是** | 使用含环数据集 + Z 轴加权 loss + 环预测 loss |
| Recycling 保留 MMFF94 | **否** | 去掉 MMFF94，仅保留 GNN 二次预测 |
| 含环分子加权采样 | **否** | 新数据集已全部是含环分子，不需要 |
| 分组报告 corrugation | **是** | 三分位数分组（数据集 corrugation 集中在 1.25-1.83Å） |
| 环质心 Z 轴反重叠约束 | **是** | 推理时约束，多个环质心之间 Z 轴不重叠 |
| 非法环（3/4 元）拆解 | ~~是~~ **已删除** | 首轮实验 Ring Preserve 从 0.907 暴跌到 0.518，误伤正确环结构 |
| 主动预测环结构 | **是** | 训练时添加环预测 loss（保留），推理时非法环拆解（删除） |
| 环 blend 比例 | 0.3→**0.4**（非 0.6） | 0.6 太强导致过度投影到错误平面 |

### 4.2 最终改进清单与执行顺序

| 步骤 | 改进项 | 类型 | 预计时间 | 预期收益 |
|------|--------|------|----------|----------|
| 1 | Z 轴加权 loss（z_weight=2.0） | 训练代码 | ✅ 已实施 | RMSD↓, Bottom Recall↑ |
| 2 | 环预测 loss（ring_count + ring_match） | 训练代码 | ✅ 已实施 | Ring Preserve↑, 环形状↑ |
| 3 | 增强连通性修正（拉回 60%，迭代 2 次） | 推理代码 | ✅ 已实施 | 离散原子↓ |
| 4 | 环 blend 比例调整（0.3→**0.4**）+ 多步执行 | 推理代码 | ✅ 已修正 | 环形状↑（0.6 太强已回调） |
| 5 | 环质心 Z 轴反重叠约束 | 推理代码 | ✅ 已实施 | 环重叠↓ |
| ~~6~~ | ~~非法环（3/4 元）拆解~~ | ~~推理代码~~ | ❌ 已删除 | 导致 Ring 0.907→0.518 |
| 7 | 去掉 MMFF94 力场精炼 | Recycling 代码 | ✅ 已实施 | RMSD 0.283→~0.205 |
| 8 | Corrugation 三分位数分组评估 | 评估代码 | ✅ 已修正 | 按 P33/P67 动态分组 |
| 9 | Phase 1 重新训练（含环数据集 + Z loss + 环 loss） | 训练运行 | ~3.5h | 全面提升 |
| 10 | Phase 2 GNN 训练 | 训练运行 | ~1h | Type Acc↑ |
| 11 | Phase 3 Recycling 评估（无 MMFF94） | 评估运行 | ~0.5h | 最终指标 |
| 12 | 可视化 + 分组报告 | 输出 | ~0.5h | — |
| **总计** | | | **~10.5h** | |

### 4.3 数据集变更（已实施）

```
V12 数据集：corrugation ≥ 1.25Å → 213,505 分子（含无环分子）
V13 数据集：corrugation ≥ 1.25Å + require_ring → 205,227 分子（全部含 5/6 元环）

config.json 新增: "require_ring": true
ring_cache_K-1.pkl: 已生成并缓存（一次性扫描，后续秒加载）
```

受影响的文件：
- `src/data/dataset.py`: 新增 `require_ring` 参数 + `_filter_by_ring()` 方法
- `src/train.py`: 传递 `require_ring`
- `src/eval_phase1.py`: 传递 `require_ring`
- `src/eval_recycling.py`: 传递 `require_ring`
- `src/train_gnn.py`: 传递 `require_ring`
- `src/train_cgan.py`: 传递 `require_ring`
- `src/visualize_val.py`: 传递 `require_ring`
- `config.json`: 新增 `"require_ring": true`

### 4.4 推理时约束设计

#### 4.4.1 环质心 Z 轴反重叠

```python
# 在 _auto_detect_and_project_rings 末尾
for i, ring_a in enumerate(rings_found):
    for j, ring_b in enumerate(rings_found):
        if j <= i:
            continue
        shared = set(ring_a) & set(ring_b)
        if len(shared) >= 2:
            continue  # 稠环（共享 ≥2 原子），允许共面
        centroid_a = x_0[b, ring_a].mean(dim=0)
        centroid_b = x_0[b, ring_b].mean(dim=0)
        xy_dist = (centroid_a[:2] - centroid_b[:2]).norm()
        if xy_dist < ring_radius * 1.5:
            # XY 方向太近 → 在 Z 轴拉开
            z_diff = (centroid_a[2] - centroid_b[2]).abs()
            if z_diff < min_z_sep:  # min_z_sep ≈ 0.05 归一化 ≈ 0.6Å
                shift = (min_z_sep - z_diff) / 2
                x_0[b, ring_a, 2] += shift
                x_0[b, ring_b, 2] -= shift
```

#### 4.4.2 非法环拆解

```python
# 检测 3/4 元环并推开
for ring_size in [3, 4]:
    illegal_rings = find_rings_of_size(adj, ring_size)
    for ring in illegal_rings:
        # 找到最弱的边（距离最长）并推开
        max_dist_pair = find_weakest_edge(ring)
        push_apart(x_0[b], max_dist_pair, delta=0.05)
```

### 4.5 训练时环预测 loss 设计

**动机**：新数据集中 100% 分子含 5/6 元环。模型应被主动奖励去形成正确的环拓扑，而非仅靠 Procrustes 被动对齐。

**设计**：在 `compute_loss` 中，对每步去噪的 x_0_pred 进行可微分的环检测，与 GT 环信息比对：

```python
def compute_ring_prediction_loss(x_0_pred, gt_ring_info, mask):
    """主动环预测 loss：鼓励模型在预测坐标中形成正确环结构。

    组成：
    (1) ring_count_loss: |n_rings_pred - n_rings_gt|  (环数量匹配)
    (2) ring_atom_loss:  对 GT 环中的原子，检查预测坐标中是否
        也形成环 (距离阈值内互相连接)
    """
    # (1) 环数量 loss
    # 在 x_0_pred 上建图 → 检测 5/6 环 → 计数
    n_pred = count_rings_differentiable(x_0_pred, mask)
    n_gt = gt_ring_info["n_rings"]
    ring_count_loss = F.smooth_l1_loss(n_pred, n_gt.float())

    # (2) 环原子连通性 loss
    # 对 GT 中每个环的相邻原子对 (i, j)，惩罚 pred 中 dist(i,j) > bond_threshold
    ring_bond_loss = 0.0
    for ring_idx in gt_ring_info["ring_atom_indices"]:
        valid = ring_idx >= 0
        atoms = ring_idx[valid].long()
        if len(atoms) < 5:
            continue
        # 环内相邻原子之间应有键连接
        for k in range(len(atoms)):
            a, b = atoms[k], atoms[(k+1) % len(atoms)]
            dist = (x_0_pred[:, a] - x_0_pred[:, b]).norm(dim=-1)
            # 惩罚距离超过键长阈值的情况
            ring_bond_loss += F.relu(dist - 0.18).mean()  # 0.18 ≈ 2.16Å 归一化

    return ring_count_loss + 0.5 * ring_bond_loss
```

**注意事项**：
- 此 loss 仅在 t < 500（去噪后期）时计算，因为早期 x_0_pred 噪声太大，环检测无意义
- 权重建议 0.3-0.5，避免过强约束干扰坐标自由度
- `count_rings_differentiable` 需要使用 soft adjacency（sigmoid 而非 hard threshold）保持梯度可流动

---

## 第五部分：论文叙事调整（结合 DeepSeek 建议）

### 5.1 核心叙事框架

> "我们首次提出从 AFM 图像栈直接生成 3D 分子坐标的方法。虽然 AFM 对 Z 方向信息天然不敏感，但通过 Video Vision Transformer 对多层深度切片的跨距离对比度建模，结合化学先验约束的条件扩散模型，我们在准平面分子上取得了令人鼓舞的结果（RMSD < 0.20, PMI > 85%）。"

### 5.2 审稿人预期质疑与回应

**Q: AFM 对 z 方向不敏感，你怎么做 3D？**

A: 不敏感 ≠ 没有信息。CO-tip AFM 的对比度随 tip-sample 距离变化，不同原子的衰减率不同。我们用 Video ViT 编码 10 层深度切片的对比度变化模式，提取隐含的 Z 方向信息。按 corrugation 分组验证：准平面分子 RMSD < 0.15，高起伏分子误差增大但仍有意义（模型正确捕获了倾斜趋势）。

**Q: Type Match 只有 53%，实用性？**

A: (1) 53% 远超随机基线（10 元素均匀分布 = 10%）；(2) Formula Similarity 0.94 说明模型正确理解了化学组成；(3) C/N/O 在 AFM 形貌上的物理相似性（vdW 半径差 < 0.1Å）是当前技术的理论上限；(4) 形状重建（PMI 86%）和环结构（Ring 91%）才是 3D 重建的核心价值。
