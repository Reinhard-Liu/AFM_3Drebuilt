# V16 CID 检索评估报告（最终版）

**日期**: 2026-03-28
**数据集**: QUAM-AFM, min_corrugation=1.25, require_ring=True, val_size=1000
**模型**: V16 best_diffusion.pt (Epoch 25, Val Loss=13.87)
**方法**: Hungarian 匹配 + Kabsch RMSD + 元素组成相似度（已集成到 `visualize_val.py`）

---

## 1. 新相似度计算方法

### 核心改进（已修复 `density=True` Bug）

```python
def compute_molecular_similarity_for_retrieval(pred_coords, pred_types, pred_mask,
                                               db_coords, db_types, db_mask) -> dict:
```

**7 步计算**：

| 步骤 | 指标 | 原理 | 权重 |
|------|------|------|------|
| 1 | **Hungarian 匹配** | 空间距离 + 元素类型成本矩阵，最优 1-1 原子配对 | — |
| 2 | **Kabsch RMSD** | 匹配原子 SVD 对齐后计算 RMSD | 0.25 |
| 3 | **原子类型准确率** | 匹配后同类型原子比例 | 0.20 |
| 4 | **Coulomb 特征值相似度** | 旋转不变分子描述符（特征值 cosine） | 0.20 |
| 5 | **元素组成相似度** | 10 种元素计数向量 cosine | 0.15 |
| 6 | **原子数相似度** | 1 - \|n_pred - n_db\| / max(n_pred, n_db) | 0.10 |
| 7 | **综合得分** | 加权平均 → [0, 1] | 0.10 |

**关键修复**：OLD 方法使用 `density=True` → 不同原子数归一化直方图完全相同 → sim≈1.000。NEW 方法通过 Hungarian 匹配 + 元素类型成本彻底解决这个问题。

---

## 2. 15 样本评估结果

| Idx | GT CID | RMSD | Overall Sim | Top-1 CID | GT Rank | GT Elements | Pred Elements | Type Match | Elem Sim |
|-----|--------|------|------------|-----------|---------|-------------|---------------|------------|----------|
| 0 | ~96799461 | 0.110 | 0.740 | 97735266 | **104** | CHNO | CHNO | 0.667 | 0.985 |
| 71 | ~96855893 | 0.105 | 0.803 | 97300481 | **23** | CHNO | CHNO | 0.571 | 1.000 |
| 142 | ~968948 | 0.119 | 0.801 | 969440 | **14** | CHNO | CHO | 0.474 | 0.943 |
| 214 | ~97058107 | 0.125 | 0.681 | 972595 | **141** | CFHNOS | CHS | 0.138 | 0.632 |
| 285 | ~97180587 | 0.124 | 0.672 | 97301107 | **261** | CHNO | CHNO | 0.381 | 1.000 |
| 356 | ~97302621 | 0.117 | 0.678 | 97301098 | **426** | CHN | CHN | 0.333 | 0.933 |
| 428 | ~97610 | 0.106 | 0.706 | 97264 | **585** | CHN | CH | 0.286 | 0.800 |
| 499 | ~97702881 | 0.104 | 0.737 | 97301068 | **272** | CHNO | CHN | 0.412 | 0.933 |
| 570 | ~97705683 | 0.107 | 0.800 | 98040849 | **89** | CHNO | CHNO | 0.520 | 1.000 |
| 642 | ~97758095 | 0.107 | 0.623 | 96861809 | **368** | CHNOS | CHNOS | 0.286 | 1.000 |
| 713 | ~97776379 | 0.124 | 0.739 | 9794422 | **425** | CHNO | CHNO | 0.452 | 1.000 |
| 784 | ~9796278 | 0.119 | 0.782 | 98030476 | **33** | CHNO | CHNO | 0.528 | 1.000 |
| 856 | ~98011567 | 0.111 | 0.650 | 97301105 | **399** | CHOS | CClHNOS | 0.174 | 0.632 |
| 927 | ~98020601 | 0.108 | 0.712 | 97030357 | **226** | CFHNOS | CHNOS | 0.333 | 0.849 |
| 999 | ~9813074 | 0.074 | 0.787 | 98011581 | **38** | CHNOS | CHOS | 0.500 | 0.971 |

---

## 3. 汇总统计

| 指标 | 值 |
|------|-----|
| RMSD 均值 | 0.1106 ± 0.0120 |
| RMSD 范围 | [0.074, 0.125] |
| GT Rank 均值 | 226.9 |
| GT Rank 中位数 | 226.0 |
| GT Rank 范围 | [14, 585] |
| Overall Sim 均值 | 0.727 |
| GT Rank ≤ 50 | **4/15 (27%)** |
| GT Rank ≤ 100 | **5/15 (33%)** |
| GT Rank ≤ 200 | **7/15 (47%)** |

---

## 4. 典型案例

### 案例 1：Sample 142（GT Rank=14，最佳案例）

- GT: 38 atoms, CHNO | Pred: CHO（缺 N）
- Type Match=0.474 | Elem Sim=0.943
- **GT Rank=14**：即使缺少 N 原子，Hungarian 匹配 + Kabsch 仍能准确定位

### 案例 2：Sample 356（GT Rank=426，原子类型错误）

- GT: 24 atoms, CHN（3N）| Pred: CHN（0N）
- Type Match=0.333 | Elem Sim=0.933
- **GT Rank=426**：缺少 N 原子导致 Type Match 仅 33.3%，正确惩罚

### 案例 3：Sample 214（GT Rank=141，元素缺失严重）

- GT: 29 atoms, CFHNOS | Pred: CHS（缺 F, N, O）
- Type Match=0.138 | Elem Sim=0.632
- **GT Rank=141**：FNO 全部缺失，Type Match 极低（13.8%）

---

## 5. 与 OLD 方法对比

> **注意**：OLD 报告使用顺序索引（0-14），NEW 使用均匀采样（0,71,142,...），样本不完全相同。此对比仅供参考。

| 指标 | OLD（Buggy） | NEW（Corrected） |
|------|-------------|------------------|
| 数据集参数 | min_corrugation=0.0 ❌ | min_corrugation=1.25 ✅ |
| 相似度方法 | density=True → sim≈1.000 | Hungarian + Kabsch |
| GT Rank 均值 | ~312 | **227** |
| 综合 Sim 可信度 | 不可信 | **可信** |

---

## 6. 原子类型预测问题（与相似度方法无关）

**6/15 样本存在原子类型缺失/多余问题**：

| Sample | GT Elements | Pred Elements | 问题 |
|--------|------------|---------------|------|
| 142 | CHNO | CHO | GT 有 N，Pred 无 N |
| 214 | CFHNOS | CHS | GT 有 FNO，Pred 无 |
| 356 | CHN(3N) | CHN(0N) | GT 有 3N，Pred 无 N |
| 428 | CHN | CH | GT 有 N，Pred 无 N |
| 499 | CHNO | CHN | GT 有 O，Pred 无 O |
| 856 | CHOS | CClHNOS | Pred 多 Cl（GT 无） |

**这是模型能力问题**，需要从模型架构或训练策略改进。

---

## 7. 可视化输出

生成文件（15 张）位于：`experiments/v16/visualizations_corrected/`

每张图像包含：
- 3D GT vs Predicted 坐标对比
- Kabsch RMSD + 匈牙利匹配信息
- **NEW 子指标面板**：Kabsch Score, Type Match, Coulomb Sim, Element Comp, Overall Sim
- **NEW Top-3 检索结果**：CID + Overall Sim + 子指标分解 + GT Rank

---

## 8. 结论

1. **NEW 方法正确集成**：Hungarian 匹配 + Kabsch RMSD + 元素组成相似度已完整实现到 `src/visualize_val.py`
2. **消除 density=True Bug**：不同原子数分子不再被错误赋予 sim≈1.000
3. **GT Rank 均值 227**：验证集 1000 分子中，GT 分子平均排在前 22.7%
4. **子指标透明**：Type Match、Coulomb Sim、Elem Sim 等分解显示预测质量
5. **主要瓶颈**：模型原子类型预测（N/F/Cl 等重元素）仍需改进

---

## 9. 输出文件

| 文件 | 路径 |
|------|------|
| 可视化图像 | `experiments/v16/visualizations_corrected/val_sample_*.png` (15 张) |
| 本报告 | `experiments/v16/cid_retrieval_final_report.md` |
| 新相似度代码 | `src/visualize_val.py` (第 45-238 行) |
