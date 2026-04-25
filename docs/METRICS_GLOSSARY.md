# 指标术语表(Metrics Glossary)

> 本文件定义本项目所有评估指标。每条包含**含义、范围、计算方式、对应代码**。复盘报告解读请见 [`RESULT_INTERPRETATION.md`](RESULT_INTERPRETATION.md)。

---

## 总目录

- [一、综合分(Composite Scores)](#一综合分composite-scores)
- [二、原子级(Atom-level)](#二原子级atom-level)
- [三、对象级类型(Object-level Type)](#三对象级类型object-level-type)
- [四、边/键(Edge / Bond)](#四边键edge--bond)
- [五、3D 几何](#五3d-几何)
- [六、检索(Retrieval)](#六检索retrieval)
- [七、缝隙分解(Gap Decomposition)](#七缝隙分解gap-decomposition)
- [八、`peak_*` vs `gt_*` 两套对照](#八peak_-vs-gt_-两套对照)

---

## 一、综合分(Composite Scores)

V20 主线两个最关键的对外数字。所有子项均为 0–1。

### 1.1 `pred_object_score`(2D 综合分)

```
pred_object_score = 0.25 * type_acc
                  + 0.20 * macro_f1
                  + 0.15 * hetero_f1
                  + 0.20 * edge_f1
                  + 0.15 * count_score      # 1 - normalize(count_mae)
                  + 0.05 * center_score
```

V20 medium10 测试集均值 = **0.7141 ± 0.086**(EXP-01,512 样本)。

### 1.2 `pred_object_3d_score`(3D 综合分)

```
pred_object_3d_score = 0.35 * coord_score   # 1 - normalize(heavy_rmsd / xy_mae)
                     + 0.20 * z_score        # 1 - normalize(z_mae)
                     + 0.15 * edge_f1
                     + 0.15 * type_acc
                     + 0.10 * count_score
                     + 0.05 * center_score
```

V20 medium10 测试集均值 = **0.8112 ± 0.056**。

> **3D 分一般高于 2D 分**:因为 coord/z 分量在 V19 起就接近上限(`atom_xy_mae < 0.011`、`atom_z_mae < 0.10`),拉高了 3D 分基线。

---

## 二、原子级(Atom-level)

度量"找到原子位置"这一步的质量,只看中心、不看类型。

| 字段 | 含义 | 范围 | V20 值 |
|---|---|---|---|
| `atom_center_score_r3` | 3 px 半径内中心命中率 | 0–1 | **0.9991** |
| `atom_xy_mae` | 命中原子的 (x, y) 平均误差(归一化坐标) | ≥0 | **0.01091** |
| `atom_z_mae_r3` | 命中原子的 z 误差(3 px 半径内,归一化) | ≥0 | **0.09095** |
| `atom_count_mae` | 全局原子数 MAE(峰检测后) | ≥0 | 0.94 |

`atom_xy_mae=0.011` 在 128×128 网格上 ≈ 0.011 × 128 = 1.4 像素(即 ≈ 0.14 Å)。

---

## 三、对象级类型(Object-level Type)

度量"识别原子是什么元素"。

### 3.1 `pred_object_type_acc`

匹配后预测元素与 GT 元素一致的比例。

V20 = **0.6942**(全部 11 类微平均)。

### 3.2 `pred_object_macro_f1`

11 类的宏平均 F1。**对小类敏感**:即使大量正确预测 C/H,杂原子表现差也会拉低这个数。

V20 = **0.5345**。

### 3.3 `pred_object_hetero_f1`

杂原子(N/O/F/P/S/Cl/Br/I)子集的 F1,排除最常见的 C 和 H。**这是论文里最关键的"困难场景"指标**。

V20 = **0.7434**;Dense baseline = 0.2080(× 3.57)。

---

## 四、边/键(Edge / Bond)

### 4.1 `pred_object_edge_f1`(strict)

严格匹配:预测边端点中心 与 GT 边端点中心 距离 ≤ 3 像素。

V20 = **0.6358**。

### 4.2 `pred_object_edge_f1_robust`

放宽匹配:用 Hungarian 把预测中心匹到 GT 中心 后,在**逻辑邻接矩阵**上算 F1。

V20 = **0.9138**。

### 4.3 缝隙(Gap)

```
edge_gap_robust = pred_object_edge_f1_robust - pred_object_edge_f1
```

V20 平均 = **0.2780**。**缝隙大** = 模型"勉强对齐"成功:中心稍偏但拓扑对了。详见 [§ 七](#七缝隙分解gap-decomposition)。

---

## 五、3D 几何

### 5.1 `pred_object_heavy_rmsd`

排除 H 的重原子 RMSD(Å)。Hungarian 匹配后计算。

V20 = **0.066 Å**(EXP-04)。换算:在 12 Å 归一化范围下 = 0.066 / 12 ≈ 0.0055。报告中也常以归一化值呈现。

### 5.2 `pred_object_z_mae`

匹配原子的 z 平均误差(归一化)。

V20 = **0.0946**。

### 5.3 `pred_object_nonplanarity_mae`

预测平面性偏离 GT 的差。归一化分子主轴坐标系下计算。

V20 = ~0.05。

### 5.4 `pred_object_bond_len_mae`

按预测边重建键后,键长平均偏差(Å)。

V20 = **0.187 Å**(EXP-04)。

### 5.5 `pair_dist_pass_rate@0.25`

两两原子距离 |d_pred - d_gt| ≤ 0.25 Å 的样本占比。

V20 = **93.16%**。

---

## 六、检索(Retrieval)

把测试样本的 embedding 与 K-1 闭集所有分子比对,看 GT 排第几。

| 字段 | 含义 | V20 值 |
|---|---|---|
| `top1` | Top-1 命中率 | 0.7422 |
| `top3` | Top-3 命中率 | 0.8633 |
| `top5` | Top-5 命中率 | 0.9023 |
| `mrr` | Mean Reciprocal Rank | 0.8118 |
| `mean_rank` | 平均排名 | 5.533 |
| `median_rank` | 中位排名 | 1.0 |

**分层(EXP-02 输出)**:

- 大分子(≥35 原子)Top-1 = 0.8047
- 小分子(≤22 原子)Top-1 = 0.7075
- 杂原子多(≥3 个杂原子)Top-1 一般高于纯碳氢

---

## 七、缝隙分解(Gap Decomposition)

**EXP-03** 专门研究 strict 与 robust 之间的 gap。

### 7.1 `edge_gap_robust` 直方图

V20 EXP-03 的核心产出:`experiments/v20_object_joint_medium10_exp03_gap_decompose/plots/edge_gap_robust_hist.png`(已收录到 `assets/figures/v20_exp03_edge_gap_robust_hist.png`)。

`80.66%` 样本 gap ≥ 0.20 — 模型在"拓扑对、像素稍偏"的状态下大量样本停留;表明**进一步提升 strict 性能的瓶颈是中心定位精度**(亚像素 / sub-pixel),而非拓扑能力。

### 7.2 关联性

- `gap_vs_z_mae.png` — gap 与 z 误差的相关
- `gap_vs_bond_len_mae.png` — gap 与键长误差的相关
- `gap_vs_node_coverage.png` — gap 与节点覆盖率的相关

详见 [`RESULT_INTERPRETATION.md § 4`](RESULT_INTERPRETATION.md#四exp-03-缝隙诊断怎么读)。

---

## 八、`peak_*` vs `gt_*` 两套对照

| 前缀 | 含义 | 用途 |
|---|---|---|
| `gt_object_*` | 用 GT centers 作为对象级头输入 | 评估"上界",验证 type / edge head 自身能力 |
| `peak_object_*` | 用峰检测 centers 作为对象级头输入 | 评估"实际部署"性能 |
| `pred_object_*` | V20 新引入,用预测中心采样 + 双输入头 | V20 主线评估指标 |

**Gap 缩小的路径**:
- V15 — 没有 peak 路径,只有 gt 路径
- V19 — peak 路径 + curriculum,gap 显著收窄
- V20 — pred 路径 + 双输入头,完全闭环

---

## 九、配置变量与权重影响

详见 [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)。简表:

| 想提升的指标 | 调高对应权重 |
|---|---|
| `pred_object_type_acc` | `lambda_type_obj_pred` |
| `pred_object_hetero_f1` | `lambda_teacher_type_distill`、`focal_gamma` |
| `pred_object_edge_f1` | `lambda_edge_obj_pred` |
| `atom_count_mae` | `lambda_object_count`、`lambda_object_count_mae` |
| `atom_xy_mae` | `lambda_center` 与 `center_curriculum_warmup_epochs` |

---

## 十、对应代码

`src/utils/metrics.py`(1370 行)是所有指标的实现。关键函数:

- `compute_atom_metrics()` — § 二
- `compute_object_metrics()` — § 三、四
- `compute_3d_metrics()` — § 五
- `compute_retrieval_metrics()` — § 六
- `decompose_gap()` — § 七
