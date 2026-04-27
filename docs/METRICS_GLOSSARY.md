# 指标术语表(Metrics Glossary)

> 本文定义本项目**所有**评估指标:含义、范围、加权公式、对应代码行号、与已发布的真实数字。报告解读请见 [`RESULT_INTERPRETATION.md`](RESULT_INTERPRETATION.md)。
>
> 配套阅读:[`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md) §四–§五 介绍指标背后的网络头。

---

## 一、6 维评估框架

| 编号 | 名称 | 评估脚本 | 协议 | 主要输出 |
|---|---|---|---|---|
| EXP-01 | 对象级 Full-test | `src/v20_eval_fulltest_object.py` | 闭环推理(pred 中心) | `fulltest_object_test.{md,json,csv}`、`samples/*.png` |
| EXP-02 | 闭集检索 | `src/v20_eval_retrieval_full.py` | closed_world_test_pool | `retrieval_fulltest_test.{md,json}`、`plots/*.png` |
| EXP-03 | 缝隙诊断 | `src/v20_eval_gap_decompose.py` | edge_match_radius_px=3.0 | `gap_decomposition_test.{json,csv}` |
| EXP-04 | 几何诊断 | `src/v20_eval_geom_diagnostics.py` | xy_match_radius_px=3.0 | `geom_diagnostics_test.{json,csv}` |
| SUP-01 | Dense baseline | `src/v20_eval_dense_baseline.py` | 2D dense → argmax 解码 | `dense_baseline_fulltest.{md,json,csv}` |
| SUP-02 | Graph baseline | `src/v20_eval_graph_baseline.py` | GNN type head 替换 | `graph_baseline_fulltest.{md,json,csv}` |

所有评估默认 `--split test`,`val_size=512`,故 test set 为 512 样本。

---

## 二、综合分公式(主指标)

### 2.1 `pred_object_score` — V20 主线核心数

`src/utils/metrics.py:555-563`

```python
pred_object_score = (
    0.15 * atom_count_score             # 原子数
  + 0.25 * atom_position_score          # 重原子位置 RMSD
  + 0.20 * atom_semantic_score          # 类型 + 杂原子
  + 0.15 * ring_integrity_score         # 环结构
  + 0.10 * connectivity_score           # 连接性
  + 0.10 * local_chem_score             # 化学合理性(键长)
  + 0.05 * global_shape_aux_score       # 全原子形状
)
```

权重和为 1.00。每个分量值域 [0, 1],综合分值域 [0, 1]。

### 2.2 `pred_object_3d_score` — 3D 综合

与 `pred_object_score` 同公式,但 `atom_position_score` 与 `global_shape_aux_score` 用 **包含 z 的全 3D RMSD** 计算(而非 xy 投影)。3D 系统性比 2D 高 ~0.10。

### 2.3 `peak_object_score` — peak 路径综合

完全相同的 7 项加权,但所有原子量取自 **peak-center 路径**(训练 curriculum 切换前的"标准"通路)。V19 主线最重要的对外数字。

---

## 三、各分量定义

### 3.1 `atom_count_score`(0.15)

`metrics.py:339-342`

```python
atom_count_exact = (n_pred == n_gt)       # 0/1
atom_count_abs_error = |n_pred - n_gt|
count_norm = max(max(n_pred, n_gt), 1)
atom_count_score = 0.7 * exact + 0.3 * max(0, 1 - abs_error / count_norm)
```

- 完全命中 0.7 分;线性衰减 0.3 分
- V20 实测均值 **0.9711**(test, n=512),exact rate 高于 70%

### 3.2 `atom_position_score`(0.25)

`metrics.py:407-408`

```python
atom_position_score = max(0.0, 1.0 - matched_heavy_atom_rmsd / 2.0)
global_shape_aux_score = max(0.0, 1.0 - matched_atom_rmsd / 2.0)
```

- 归一化常数 **2.0 Å**(超过 2 Å 视为 0 分)
- 仅匹配的重原子(无 H)进入 `matched_heavy_atom_rmsd`
- V20 实测 `pred_object_heavy_rmsd` 均值 **0.0659 Å**(因为是归一化空间,实际 Å ≈ 0.79)

### 3.3 `atom_semantic_score`(0.20)

`metrics.py:418`

```python
atom_semantic_score = 0.65 * atom_type_acc + 0.35 * hetero_f1
```

权重:类型精度(0.65)+ 杂原子 F1(0.35)。

#### `atom_type_acc`

`metrics.py:364`

```python
atom_type_acc = (pred_match_types == gt_match_types).mean()  # 仅匹配原子
```

V20 实测 **0.6942**(test)。

#### `macro_f1`

`metrics.py:150-176`

```python
classes = sorted(set(pred_all_types | gt_all_types))    # 排除 -1 (padding)
f1_list = []
for cls in classes:
    tp = ((pred_match == cls) & (gt_match == cls)).sum()
    fp = (pred_all == cls).sum() - tp
    fn = (gt_all == cls).sum() - tp
    f1_list.append(safe_f1(tp, fp, fn))
return mean(f1_list)
```

- 类别集合:**所有出现过的类型**(预测或真实),不限于主元素
- V20 实测 **0.5345**;偏低因为含罕见 P / Br / I 等(样本极少 → F1 = 0)

#### `hetero_f1`(杂原子 F1)

`metrics.py:410-417`

```python
is_hetero = lambda t: t not in (0, 1)        # 排除 H, C
pred_match_hetero = is_hetero(pred_match)    # 匹配原子的杂原子标记
gt_match_hetero   = is_hetero(gt_match)
pred_all_hetero   = is_hetero(pred_all)
gt_all_hetero     = is_hetero(gt_all)
tp = (pred_match_hetero & gt_match_hetero).sum()
fp = pred_all_hetero.sum() - tp
fn = gt_all_hetero.sum() - tp
hetero_f1 = safe_f1(tp, fp, fn)
```

- 二分类:H/C(0、1)→ 负;其他 8 种 → 正
- V20 实测 **0.7434**(对外宣传"杂原子 F1 0.74")

### 3.4 `ring_integrity_score`(0.15)

`metrics.py:496-500`

```python
ring_integrity_score = (
    0.30 * ring_count_exact        # 环数完全一致 0/1
  + 0.40 * ring_complete_rate      # = 0.75 * scaffold_local_edge_recall + 0.25 * approx_ring_complete_rate
  + 0.30 * scaffold_local_edge_f1
)
```

判定环结构是否被复原(基于 RDKit 推断的 scaffold 边)。

### 3.5 `connectivity_score`(0.10)

`metrics.py:540`

```python
connectivity_score = 0.6 * attachment_edge_f1 + 0.4 * scaffold_local_edge_f1
```

- `attachment_edge_f1`:scaffold 与 sidechain 之间的连接边
- `scaffold_local_edge_f1`:scaffold 内部边

### 3.6 `local_chem_score`(0.10)

主要由 `bond_validity` 决定(`metrics.py:793-856`):

```python
ideal = IDEAL_BOND_LENGTHS[t_i, t_j]            # 理想键长查表
max_dist = ideal * 1.3                          # 候选键阈值
n_bonds = pairs with dist < max_dist
n_valid = pairs with |dist - ideal| / ideal < 0.25
bond_validity = n_valid / max(n_bonds, 1)
```

- 容差 ±25%(键长偏差)
- 候选键阈值 1.3 ×ideal

辅助:`valence_validity`(`metrics.py:1018-1078`,价态合法性 + 连通分量数,权重 0.7 / 0.3),用于诊断而非综合分。

### 3.7 `global_shape_aux_score`(0.05)

同 `atom_position_score` 公式,但用 `matched_atom_rmsd`(包含 H,即所有匹配原子)。

---

## 四、对象级类型 / 边指标

### 4.1 `pred_object_type_acc / macro_f1 / hetero_f1`

由 `CenterConditionedTypeHead` 在 pred-中心路径输出,经 Hungarian 与 GT 匹配后逐原子比较。前缀 `pred_` 表示"使用预测中心",对比项有:
- `peak_object_*`:使用 peak-center
- `gt_object_*`:使用 GT-center(上界)

| V20 实测(test, n=512) | 值 |
|---|---|
| `pred_object_type_acc` | 0.6942 |
| `pred_object_macro_f1` | 0.5345 |
| `pred_object_hetero_f1` | 0.7434 |
| `peak_object_score` | 0.8338 |
| `gt_object_score` | 0.8279 |

### 4.2 `pred_object_edge_f1` 与 `pred_object_edge_f1_robust`

`metrics.py:509-512`

#### Strict edge F1

```python
pred_edges = _infer_bond_edges(pred_coords, pred_types)
gt_edges   = _infer_bond_edges(gt_coords,   gt_types)
# 用 Hungarian map 把 pred 索引转到 GT 索引
mapped_pred_edges = {sorted(map[i], map[j]) for (i,j) in pred_edges if both in map}
tp = |mapped_pred ∩ gt|
fp = |mapped_pred − gt|
fn = |gt − mapped_pred|
strict_f1 = safe_f1(tp, fp, fn)
```

`_infer_bond_edges`(`metrics.py:179-196`)依据 `MAX_BOND_DIST = IDEAL_BOND_LENGTHS × 1.3` 自动推断邻接。

#### Robust edge F1

`v20_eval_gap_decompose.py`:`edge_match_radius_px=3.0`(归一化空间约 0.46 Å)。
- 任何预测原子到某真实原子距离 ≤ 3px 即视作匹配,允许中心略偏
- robust 衡量"拓扑能力",strict 衡量"精确定位 + 拓扑"

V20 实测:
- `pred_object_edge_f1`(strict 类) = **0.6358**
- `pred_object_edge_f1_robust` = **0.9138**
- 二者差 = **edge_gap_robust = 0.2780**(EXP-03 关键数字)

### 4.3 Hungarian 匹配实现

`metrics.py:78-86`

```python
from scipy.optimize import linear_sum_assignment

def hungarian_match_numpy(pred_coords, gt_coords):
    diff = pred_coords[:, None, :] - gt_coords[None, :, :]
    cost = sqrt((diff ** 2).sum(-1))           # 欧氏距离矩阵
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind, cost
```

- 任意尺寸(`n_pred ≠ n_gt` 也可)
- 距离单位 = 归一化坐标,需乘 `COORD_SCALE=12` 才是 Å

---

## 五、几何诊断(EXP-04)

`v20_eval_geom_diagnostics.py`,匹配半径 3 px。

| 字段 | V20 实测 | 含义 |
|---|---|---|
| `gt_height_span_ang` | 1.3544 Å | GT 分子 z 跨度均值 |
| `gt_nonplanarity_ang` | 0.2533 Å | GT 非平面性(z 标准差) |
| `pred_object_heavy_rmsd_ang` | 0.7912 Å | 重原子 RMSD(以 Å 计) |
| `pred_object_pair_dist_mae_r3` | 0.1976 Å | 配对距离 MAE |
| `pred_object_bond_len_mae_r3` | 0.1867 Å | 键长 MAE |
| `pred_object_z_corr_r3` | 0.3166 | 预测 z 与 GT z 的相关系数 |
| `pred_object_nonplanarity_error_r3` | 0.0723 Å | 非平面性误差 |

### 通过率指标

| 字段 | V20 实测 | 阈值含义 |
|---|---|---|
| `pair_dist_mae_r3_le_0p25_rate` | **93.16%** | 配对 MAE ≤ 0.25 Å 的样本比例 |
| `bond_len_mae_r3_le_0p20_rate` | 64.26% | 键长 MAE ≤ 0.20 Å |
| `z_corr_r3_ge_0p80_rate` | 25.59% | z 相关 ≥ 0.80 |
| `nonplanarity_error_r3_le_0p10_rate` | 66.60% | 非平面误差 ≤ 0.10 Å |

**论文话术**:"V20 在 93.2% 样本上配对距离 MAE ≤ 0.25 Å"。

### 分层解读

按原子数分组(EXP-04 报告):

| 范围 | pair@0.25 | bond@0.20 | z_corr≥0.8 | nonplan≤0.10 |
|---|---|---|---|---|
| ≥35 | 94.5% | 67.2% | 27.3% | 59.4% |
| 29-34 | 93.7% | 66.7% | 22.5% | 60.4% |
| 23-28 | 92.8% | 61.1% | 18.0% | 61.7% |
| ≤22 | 91.5% | 64.2% | 17.0% | 74.5% |

观察:**配对距离稳定** 跨规模,**z 相关性**随原子数增加而提高(大分子 z 跨度大,信号强)。

---

## 六、检索指标(EXP-02)

`v20_eval_retrieval_full.py`,protocol = closed_world_test_pool(候选库 = 全 512 测试样本)。

### 6.1 公式

```python
embed_query, embed_pool = forward(...)             # 全局 embedding (B, 512)
sim = cosine_similarity(query, pool)               # (n_q, n_pool)
ranks = argsort(-sim, axis=1)
top_k_correct = (ranks[:, :K] == gt_idx).any(axis=1).mean()
mrr = mean(1 / (rank_of_correct + 1))
```

### 6.2 V20 实测

| 字段 | 数值 |
|---|---|
| `num_queries` | 512 |
| `top1` | **0.7422**(379/512) |
| `top3` | **0.8633**(442/512) |
| `top5` | **0.9023**(462/512) |
| `mrr` | **0.8118** |
| `mean_rank` | 5.533 |
| `median_rank` | 1.0 |

### 6.3 分层结果

按原子数:

| 范围 | n | top1 | top3 | top5 | mrr |
|---|---|---|---|---|---|
| ≥35 | 128 | 0.8047 | 0.8906 | 0.9453 | 0.8593 |
| 29-34 | 111 | 0.7477 | 0.8649 | 0.9189 | 0.8165 |
| 23-28 | 167 | 0.7126 | 0.8323 | 0.8503 | 0.7806 |
| ≤22 | 106 | 0.7075 | 0.8774 | 0.9151 | 0.7988 |

观察:大分子 top1 显著更高(embedding 更独特);小分子虽 top1 低但 top5 不差(同分子家族近似)。

---

## 七、缝隙诊断(EXP-03)

`v20_eval_gap_decompose.py`,匹配半径 3 px。核心字段:

| 字段 | V20 实测 | 解读 |
|---|---|---|
| `matched_gt_node_coverage_r3` | 0.6957 | 真实原子 ≤ 3px 内有预测覆盖的比例 |
| `matched_gt_bond_coverage_r3` | 0.5791 | 真实键有匹配的比例 |
| `edge_f1_xy_r3` | 0.9104 | xy 平面边 F1(放宽到 3 px) |
| `edge_gap_robust` | **0.2780** | robust − strict edge F1 |
| `edge_gap_xy_r3` | 0.2746 | xy − robust |
| `matched_type_acc_r3` | 0.8334 | 匹配原子的类型精度 |
| `matched_macro_f1_r3` | 0.6635 | 匹配原子 macro_f1 |
| `matched_hetero_f1_r3` | 0.8735 | 匹配原子 hetero_f1 |
| `mean_xy_match_px_r3` | 1.1444 | 平均 xy 匹配偏移(像素) |
| `matched_pair_dist_mae_ang` | 0.1976 Å | 配对距离 MAE |
| `matched_bond_len_mae_ang` | 0.1867 Å | 键长 MAE |

### 样本分类

| 类别 | 比例 | 含义 |
|---|---|---|
| `high_gap_ge_0p20` | 80.7% | edge_gap ≥ 0.20 的样本 |
| `robust_ge_0p90_and_strict_lt_0p70` | 45.3% | "可恢复型":拓扑对了但定位差 |
| `matched_type_gain_ge_0p10` | 72.3% | 类型在放宽匹配下显著改善 |

**核心结论**:80% 样本的拓扑结构基本正确,**瓶颈在亚像素中心定位**,而非图结构能力。下一步优化方向:亚像素 head(soft-argmax / heatmap regression)。

---

## 八、SUP-01 / SUP-02 对比表

### 8.1 V20 vs Dense (SUP-01) vs Graph (SUP-02),test split,n=512

| 指标 | V20 (Object) | Dense (SUP-01) | Graph (SUP-02) |
|---|---|---|---|
| `pred_object_score` | **0.7141** | 0.2936 | 0.5414 |
| `pred_object_3d_score` | **0.8112** | 0.5333 | 0.5425 |
| `pred_object_type_acc` | **0.6942** | 0.3458 | 0.4874 |
| `pred_object_macro_f1` | **0.5345** | 0.1131 | 0.2999 |
| `pred_object_hetero_f1` | **0.7434** | 0.1028 | 0.4476 |
| `pred_object_edge_f1` | **0.6358** | 0.3324 | 0.5905 |
| `pred_object_edge_f1_robust` | 0.9138 | 0.6745 | **0.9708** |
| `pred_object_z_mae` | **0.0946** | 0.1490 | 0.3494 |
| `pred_object_count_mae` | **0.9434** | 34.143 | — |
| `pred_object_center_score` | **0.9866** | — | 0.6235 |

观察:
- **Dense baseline 完全失败** 在原子数预测(MAE 34 vs 0.94)。说明只学 dense 图无法判断"图里有几个原子"
- **Graph baseline 在 robust edge** 反而最高(0.97),说明 GNN 的局部图表达强,但 z 维与类型语义大幅落后
- V20 综合分约为 Dense 的 **× 2.4**,Graph 的 **× 1.3**

---

## 九、训练 history 字段

`history_v19_object_joint.json` 每 epoch 一项:

```jsonc
{
  "epoch": 0,
  "train_loss": 1.234,
  "val_loss": 1.111,
  "val_metrics": {
    "atom_center_score_r3": 0.97,
    "pred_object_score": 0.45,
    "peak_object_score": 0.50,
    "gt_object_score": 0.65,
    "pred_object_count_mae": 1.4,
    ...
  },
  "lambda_snapshot": { "lambda_type_obj_pred": 0.25, ... },
  "alpha_snapshot": 0.0,                              // center curriculum α
  "lr": 0.00008,
  "best_so_far": false
}
```

V19 主线复盘示例:

| Epoch | peak_object_score | peak_center_type_acc | peak_center_hetero_f1 | peak_center_edge_f1 | atom_z_mae_r3 |
|---|---|---|---|---|---|
| 1 | 0.4849 | 0.2538 | 0.2201 | 0.7681 | 0.1003 |
| 15 | **0.8016** | 0.8185 | 0.8649 | 0.8138 | 0.0893 |

---

## 十、`peak_*` / `gt_*` / `pred_*` 三种前缀

| 前缀 | 含义 | 用法 |
|---|---|---|
| `gt_*` | 用 GT 中心 + GT 类型/边喂入条件头 | 上界,对应 teacher 路径 |
| `peak_*` | 用 peak detection 出的中心,GT 类型对齐 | 训练主路径,V19 主指标 |
| `pred_*` | 完全闭环:pred 中心 + pred 类型/边 | 部署路径,V20 主指标 |

理论关系:`gt > peak > pred`(实际 V20 的 peak 与 gt 已经接近)。

EXP-01 报告同时给三套数,分析 train-deploy gap。

---

## 十一、消融数字摘要(EXP-06 / EXP-07)

`experiments/v20_ablate_*/reports/ablation_summary.json`:

| 变体 | pred_object_score | Δ |
|---|---|---|
| baseline | 0.7141 | — |
| `no_curriculum` | 0.5612 | −0.15 |
| `no_teacher_consistency` | 0.7929 | (注:用了不同 split,不可直接比较) |
| `no_object_count` | 0.6520 | −0.06 |
| `no_z_head` | 0.7164 | ≈ 0 |
| `no_edge_head` | 0.5761 | −0.14 |

**注意**:V20 ablation 在缩减子集 + 不同 epoch 上跑,绝对数字与主线不可比,只看相对差距。

---

## 十二、相关文档

- 报告解读 — [`RESULT_INTERPRETATION.md`](RESULT_INTERPRETATION.md)
- 配置 lambda 调参 — [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)
- 实现细节 — [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md)
- FAQ 含真实数字解读 — [`FAQ_EXTENDED.md`](FAQ_EXTENDED.md)
