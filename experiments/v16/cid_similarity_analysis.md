# CID 相似度计算问题分析

**样本**: Val Sample 356
**GT CID**: 83464（23 atoms: 4O, **3N**, 9C, 7H）
**模型**: V16, Epoch 25（best_diffusion.pt, Val Loss=13.87）

---

## 1. GT / Prediction / Top-3 CID 3D 结构对比

| | CID | 原子数 | 元素组成 | sim |
|---|---|---|---|
| **GT** | 83464 | 23 | 4O, **3N**, 9C, 7H | — |
| **Prediction** | — | 23 | 8O, **0N**, 15C, 7H | — |
| **Top-1** | 83532103 | 19 | 1S, 3N, 8C, 5H | **1.000** |
| **Top-2** | 83535668 | 19 | 1S, 2N, 8C, 6H | **1.000** |
| **Top-3** | 83914637 | 20 | 3N, 9C, 6H | **1.000** |

**GT rank: ~2800+**

> **核心问题**：GT 有 **3 个 N 原子**，Prediction 预测出 **0 个 N**，Top-3 分子中有 2-3 个 N 原子，但 sim 全部为 1.000。

---

## 2. Root Cause：`density=True` 归一化 Bug

### Bug 位置
`src/visualize_val.py` 第 38-70 行

```python
h1, _ = np.histogram(d1, bins=bins, density=True)  # Bug
h2, _ = np.histogram(d2, bins=bins, density=True)
```

### Bug 原理

```
density=True: h = count / (total_count × bin_width)  → 积分恒等于 1.0

不同原子数 → 不同数量的 pairwise distances
Prediction (23 atoms): 276 pairs  → 约 13.8 pairs/bin
Top-1   (19 atoms): 171 pairs  → 约  8.6 pairs/bin
Top-2   (19 atoms): 171 pairs  → 约  8.6 pairs/bin
Top-3   (20 atoms): 190 pairs  → 约  9.5 pairs/bin

不同原子数 → 相同的归一化直方图形状 → cos_sim ≈ 1.000
```

### 验证数据

```
Prediction（23 atoms）直方图: [1.536, 1.884, 0.551, 0.029, 0, 0, ...]
Top-1    （19 atoms）直方图: [1.542, 1.882, 0.549, 0.026, 0, 0, ...]
Max diff: 0.006  ← 仅 0.6% 差异！

cosine similarity = 0.999996 → 四舍五入 = 1.000
```

---

## 3. 为什么 GT 排名低？

| 原因 | 说明 |
|------|------|
| **① 原子数不匹配** | 预测 23 vs GT 23 atoms → 相同数量，所以这部分没问题 |
| **② 原子类型错误** | Prediction 预测 0N vs GT 3N → 3D 结构不对 |
| **③ 度量局限性** | `distance histogram sim` 只看全局形状，不区分原子类型 |
| **④ density Bug** | 不同原子数 → 相同归一化分布 → sim≈1.000 |

**更根本的问题**：模型 **完全没有预测出 N 原子**，这需要改进模型的 atom type prediction 能力。

---

## 4. 两个图不一致的原因

`val_sample_00356.png`（原始可视化）和 `cid_top3_3d_comparison.png`（重新生成）使用不同的随机采样种子，导致预测分子结构不同。但 **bug 完全一致**：不同原子数 → sim=1.000。

---

## 5. 修复方案

### 方案 1：限制同原子数比较（最小修复）

```python
def compute_distance_histogram_similarity(...):
    n1 = int((mask1 > 0).sum())
    n2 = int((mask2 > 0).sum())

    # Bug fix: 不同原子数的分子不能比较
    if abs(n1 - n2) > 2:  # 允许 ±2 容差
        return 0.0

    # ... 其余代码不变
```

### 方案 2：改用 RMSD + Kabsch 对齐（推荐）

```
1. 按原子数过滤（|n_pred - n_db| ≤ 2）
2. Kabsch 对齐后计算 RMSD
3. 综合考虑原子类型匹配率
```

---

## 6. 结论

| 问题 | 严重性 | 修复难度 |
|------|--------|----------|
| `density=True` 导致 sim=1.000 给不同原子数分子 | **Bug** | 简单修复 |
| 模型预测 0 个 N 原子（GT 有 3 个） | **模型缺陷** | 需改进模型 |
| 当前 retrieval 度量无效 | **需重新设计** | 中等复杂度 |
