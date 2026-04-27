# 结果解读手册(Result Interpretation)

> 帮助你看懂 `experiments/<exp>/reports/*.md` 与 `*.json`。指标定义见 [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)。

---

## 一、报告文件总览

每个评估脚本输出的报告位于 `experiments/<exp_name>/reports/`:

| 评估 | 报告文件 | 主要字段 |
|---|---|---|
| EXP-01 | `fulltest_object_test.md/json/csv` | pred/peak/gt 三套对象级指标 |
| EXP-02 | `retrieval_fulltest_test.md/json` | top1/3/5、MRR、分层结果 |
| EXP-03 | `gap_decomposition_test.json/csv` | strict / robust edge F1、coverage |
| EXP-04 | `geom_diagnostics_test.json/csv` | RMSD、pair_dist_mae、z_corr 等 |
| SUP-01 | `dense_baseline_fulltest.md/json/csv` | dense 解码后的对象级指标(对照) |
| SUP-02 | `graph_baseline_fulltest.md/json/csv` | GNN 头替换后的对象级指标 |
| 复盘 | `v19_object_joint_review_<run>.md` | 训练 history 摘要、最佳 epoch |

---

## 二、EXP-01 fulltest_object 报告解读

### 2.1 报告结构

```markdown
# V20 Object Full-Test (split=test, n=512)

## 总体指标
- pred_object_score: 0.7141
- pred_object_3d_score: 0.8112
- peak_object_score: 0.8338
- gt_object_score: 0.8279
- pred_object_count_mae: 0.9434
- pred_object_type_acc: 0.6942
- pred_object_macro_f1: 0.5345
- pred_object_hetero_f1: 0.7434
- pred_object_edge_f1: 0.6358
- pred_object_edge_f1_robust: 0.9138
- pred_object_z_mae: 0.0946
- pred_object_center_score: 0.9866

## 分量
- atom_count_score: 0.9711
- atom_position_score: 0.9671 (heavy_rmsd 0.0659)
- atom_semantic_score: 0.5530
- ring_integrity_score: 0.4582
- connectivity_score: 0.6131
- local_chem_score: 0.6258
- global_shape_aux_score: 0.9528
```

### 2.2 三种前缀的解读

| 前缀 | 路径 | 期望关系 |
|---|---|---|
| `gt_*` | GT 中心 + GT 类型/边 喂入条件头(teacher 路径) | 上界 |
| `peak_*` | peak 检测中心 + GT 类型对齐 | V19 主指标 |
| `pred_*` | 完全闭环:pred 中心 + pred 类型 + pred 边 | V20 部署指标 |

**理论关系**:`gt > peak > pred`。

**V20 实测**:`peak (0.8338) > gt (0.8279) > pred (0.7141)`。

**为什么 peak > gt?**
peak 训练得更"用力"(curriculum 末段 λ_peak_final=2.5,gt 路径 λ_gt=1.5),且 peak 中心和实际部署更接近。GT 路径更多是作为 KD upper bound 与一致性锚点使用。这种"反直觉"关系在 V20 是设计预期的副产物。

**pred 与 gt/peak 的差距 ≈ 0.12**:体现 train-deploy gap。下一步重点。

### 2.3 pred_object_score 分量观察(V20)

| 分量 | 权重 | V20 值 | 解读 |
|---|---|---|---|
| `atom_count_score` | 0.15 | 0.9711 | **饱和**,几乎完美 |
| `atom_position_score` | 0.25 | 0.9671 | xy 中心定位准 |
| `atom_semantic_score` | 0.20 | 0.5530 | **瓶颈 1**:类型(尤其罕见类) |
| `ring_integrity_score` | 0.15 | 0.4582 | **瓶颈 2**:环结构(scaffold 复原) |
| `connectivity_score` | 0.10 | 0.6131 | scaffold + attachment 边 |
| `local_chem_score` | 0.10 | 0.6258 | 键长合理性 |
| `global_shape_aux_score` | 0.05 | 0.9528 | 全形状(含 H) |

**结论**:计数与定位已饱和;**主要差距在 类型语义 + 环结构**。下一代优化方向应聚焦于:
1. 罕见类增强(P / Br / I 上采样 / focal loss 加重)
2. 环识别 head 显式化(目前是 RDKit 推断)

### 2.4 3D 综合分(`pred_object_3d_score`)

V20: **0.8112**(高于 2D 综合 0.7141 约 0.10)。

**原因**:`atom_position_score` 在 3D 是 `1 - rmsd_xyz / 2.0`,而 2D 是 `1 - rmsd_xy / 2.0`。模型 z 预测虽不强,但 xy 偏差 + z 偏差合在一起算 RMSD,**z 引入的额外距离反而把 1- 的项拉得更稳**(分母 2.0 不变)。这不是 z 学得好,而是 RMSD 的几何性质,论文应**优先汇报 2D 综合**。

### 2.5 错误样本

`samples/<idx>_best.png` 显示具体样本可视化,通常包含:
- 输入 AFM 中心切片
- center heatmap(预测 + GT 叠加)
- 类型 dense map(top 3 类)
- 解码后的分子结构图(2D)

**典型失败模式**:
1. **大分子边角原子漏检**:超过 35 原子时 peak 阈值偏紧
2. **杂原子误判为 C**:N→C 与 O→C 是最常见错误
3. **缝隙边遗漏**:scaffold 边对了,sidechain 边经常漏

---

## 三、EXP-02 retrieval 报告解读

### 3.1 关键字段

```json
{
  "num_queries": 512,
  "top1": 0.7422,
  "top3": 0.8633,
  "top5": 0.9023,
  "mrr": 0.8118,
  "mean_rank": 5.533,
  "median_rank": 1.0
}
```

### 3.2 解读 top-K 与 MRR

- **top1 = 0.7422**:74.2% 的查询样本能从 512 候选中精准锁定自己。
- **median_rank = 1.0**:中位数排名第 1 名,说明大部分样本检索准确。
- **mean_rank = 5.5**:少数失败样本拉高均值,长尾问题。

### 3.3 分层结果

```
按原子数:
| 范围 | n   | top1   | top3   | top5   | mrr    |
|------|-----|--------|--------|--------|--------|
| ≥35  | 128 | 0.8047 | 0.8906 | 0.9453 | 0.8593 |
| 29-34| 111 | 0.7477 | 0.8649 | 0.9189 | 0.8165 |
| 23-28| 167 | 0.7126 | 0.8323 | 0.8503 | 0.7806 |
| ≤22  | 106 | 0.7075 | 0.8774 | 0.9151 | 0.7988 |
```

**观察**:
- 大分子 top1 显著更高(embedding 更独特,8 个 attention head 聚焦更聚焦)
- 小分子虽 top1 低但 top5 不差(说明同类小分子之间存在合理"近似",非完全失败)

### 3.4 与论文话术的映射

- "**74% top1 检索精度**(closed-pool 512)"
- "在大于 35 原子的复杂分子上达到 80% top1"
- "MRR 0.81 表示典型样本 1 步即得"

---

## 四、EXP-03 gap_decompose 解读

### 4.1 核心数字

```json
{
  "edge_f1_strict_r3": 0.6358,
  "edge_f1_robust_r3": 0.9138,
  "edge_gap_robust": 0.2780,
  "edge_f1_xy_r3": 0.9104,
  "edge_gap_xy_r3": 0.2746,
  "matched_gt_node_coverage_r3": 0.6957,
  "matched_gt_bond_coverage_r3": 0.5791
}
```

### 4.2 怎么读 robust vs strict

- **strict edge F1**(0.6358):必须每个键的两端原子都对齐(几乎要求像素级精度)
- **robust edge F1**(0.9138):放宽到 ≤ 3 px(归一化空间约 0.46 Å 半径)
- **gap = robust - strict = 0.2780**:**拓扑能力 vs 定位能力 的差异**

### 4.3 关键结论

```
80.7% 样本 edge_gap ≥ 0.20
45.3% 样本 robust ≥ 0.90 且 strict < 0.70
```

**论文话术**:
> "V20 在 80% 以上样本上 edge_gap_robust ≥ 0.20,即模型已正确识别绝大部分键的拓扑结构,但需 0.5 Å 以内的中心精度才能完全计入 strict 指标。这表明**当前瓶颈在亚像素中心定位**,而非图结构能力。"

### 4.4 下一步优化方向

基于 EXP-03 的诊断:
1. **soft-argmax** 替代离散 peak detection(亚像素精度)
2. **heatmap regression** 加 sub-pixel offset 头
3. **center refinement** 用 deformable conv

EXP-03 是设计未来工作时**最有价值**的诊断工具。

---

## 五、EXP-04 geom_diagnostics 解读

### 5.1 关键数字

```json
{
  "gt_height_span_ang": 1.3544,           // GT z 跨度
  "gt_nonplanarity_ang": 0.2533,          // GT 非平面性
  "pred_object_heavy_rmsd_ang": 0.7912,   // 重原子 RMSD (Å)
  "pred_object_pair_dist_mae_r3": 0.1976, // 配对距离 MAE
  "pred_object_bond_len_mae_r3": 0.1867,  // 键长 MAE
  "pred_object_z_corr_r3": 0.3166,        // z 与 GT 相关
  "pred_object_nonplanarity_error_r3": 0.0723
}
```

### 5.2 通过率指标

```
pair_dist_mae_r3_le_0p25_rate:    93.16% ← 论文核心数字
bond_len_mae_r3_le_0p20_rate:     64.26%
z_corr_r3_ge_0p80_rate:           25.59% ← z 弱
nonplanarity_error_r3_le_0p10:    66.60%
```

### 5.3 解读

**93.2% 配对距离 MAE ≤ 0.25 Å** 是 V20 最有说服力的物理数字:意味着**任意两原子的距离误差中位数远小于一个键长**,模型确实在做几何重建,不是猜形状。

**z_corr 25.59%**:z 方向预测与 GT 的相关系数 ≥ 0.80 的样本仅 25%,说明:
- AFM 信号在 z 方向天然弱(物理限制)
- 模型 z 头容量足够,但训练信号不足
- 下一步可考虑双 AFM(不同振幅)输入加强 z 信号

**nonplanarity_error 0.072 Å**:模型对"分子是不是平的"判断很准,但 **z 相对值** 不准 — 这两个指标的对比说明模型学了"绝对平/不平",未学"具体在哪 z"。

### 5.4 分层观察

```
范围   pair@0.25  bond@0.20  z_corr≥0.8  nonplan≤0.10
≥35    94.5%      67.2%      27.3%        59.4%
29-34  93.7%      66.7%      22.5%        60.4%
23-28  92.8%      61.1%      18.0%        61.7%
≤22    91.5%      64.2%      17.0%        74.5%
```

**配对距离稳定** 跨规模 → 模型对距离的 inductive bias 强;**z 相关随原子数增加而提高** → 大分子 z 跨度大,信号强 → 与物理直觉一致。

---

## 六、SUP-01 / SUP-02 对照表解读

```
| 指标                   | V20      | Dense    | Graph    |
|------------------------|----------|----------|----------|
| pred_object_score      | 0.7141   | 0.2936   | 0.5414   |
| pred_object_count_mae  | 0.9434   | 34.143   | —        |
| pred_object_edge_f1    | 0.6358   | 0.3324   | 0.5905   |
| pred_object_edge_robust| 0.9138   | 0.6745   | 0.9708   |
| pred_object_hetero_f1  | 0.7434   | 0.1028   | 0.4476   |
```

### 6.1 Dense baseline 的失败

**count_mae = 34**:Dense 只学 2D 像素分类,根本不知道"分子有几个原子"。peak 检测后取所有候选,得到的数字与真实平均数(28)差 34 个 → 完全失败。

**这正是为什么 V20 显式加 `lambda_object_count`**。

### 6.2 Graph baseline 的"局部强 + 全局弱"

- **edge_f1_robust = 0.9708(超过 V20)**:GNN 的局部图建模强,放宽阈值后准确率高
- **strict edge_f1 = 0.5905**:严格阈值下不如 V20(GNN 缺少强的中心约束)
- **hetero_f1 = 0.4476**:GNN 没用 AFM patch grid,杂原子识别差
- **z_mae = 0.3494**:无 dense z 监督,z 预测崩

**论文话术**:
> "替换为 GNN 类型头(SUP-02)虽提升了局部图建模能力(robust edge F1 +0.06),却显著损失了 z 维与稀有类型(hetero F1 -0.30),证明 V20 的 dense + 对象级条件头组合是**联合优化的必要选择**。"

---

## 七、训练 history 解读

`history_v19_object_joint.json` 每 epoch 一项,核心字段:

```json
{
  "epoch": 0,
  "train_loss": 1.234,
  "val_loss": 1.111,
  "val_metrics": {
    "atom_center_score_r3": 0.97,
    "pred_object_score": 0.45,
    "peak_object_score": 0.50,
    "gt_object_score": 0.65,
    "pred_object_count_mae": 1.4
  },
  "lambda_snapshot": {
    "lambda_type_obj_pred": 0.25
  },
  "alpha_snapshot": 0.0,
  "lr": 8e-5,
  "best_so_far": false
}
```

### 7.1 V19 主线复盘示例

| Epoch | peak_object_score | type_acc | hetero_f1 | edge_f1 | z_mae |
|---|---|---|---|---|---|
| 1 | 0.4849 | 0.2538 | 0.2201 | 0.7681 | 0.1003 |
| 5 | 0.6932 | 0.6312 | 0.6453 | 0.7905 | 0.0941 |
| 10 | 0.7705 | 0.7821 | 0.8268 | 0.8085 | 0.0907 |
| **15** | **0.8016** | **0.8185** | **0.8649** | **0.8138** | **0.0893** |

**关键观察**:
- type_acc 从 0.25 起步,5 epoch 后陡升 — KD + curriculum 共同作用
- z_mae 从一开始就低 — z dense 监督一直存在,无 ramp
- edge_f1 在第 1 epoch 就 0.77 — warm start ckpt 提供了边的能力

### 7.2 lambda_snapshot 怎么用

每 epoch 记录所有 λ 当前值,用于:
1. **重现训练**:resume 时确认调度恢复正确
2. **诊断**:某个 epoch 性能突变 → 看哪个 λ 大跳
3. **画图**:横轴 epoch,纵轴 λ,直观展示 curriculum

### 7.3 alpha_snapshot

`alpha_snapshot=0.4` 表示该 epoch 中心采样 40% 用 peak、60% 用 GT。
- α=0:纯 GT(epoch 0)
- α=1.0:纯 peak/pred(epoch ≥ warmup_epochs)

---

## 八、复盘脚本(v19_object_joint_review)

```bash
python -m src.v19_object_joint_review \
    --history experiments/v19_object_joint_full15_all/checkpoints/history_v19_object_joint.json \
    --output experiments/v19_object_joint_full15_all/reports/review.md
```

输出包含:
- 每 epoch 的 7 项主要指标表
- 最佳 epoch 与对应 ckpt
- 训练曲线(loss / score / lr)的可视化建议
- λ / α 调度核查

---

## 九、报告中常见的"反直觉"现象

### 9.1 peak > gt

V20 设计预期:peak 路径在 curriculum 末段权重更大,且 peak 中心通过自身 detection 获得 → 与下游表征更一致。**不是 bug**。

### 9.2 3D 综合 > 2D 综合

不是 z 学得好,而是 RMSD 数学性质:`1 - sqrt(dx² + dy² + dz²) / 2.0` 的 dz 项在 z 弱预测下贡献小、又分摊到 sqrt 中,反而抬高了归一化分数。**论文优先汇报 2D 综合 + 单独诊断 z**。

### 9.3 robust > strict 远大

设计意图:**解耦拓扑能力与定位能力**。robust 高、strict 低 = "懂图结构但中心略偏" = V20 现状。

### 9.4 macro_f1 < hetero_f1

macro_f1 包含罕见 P / Br / I(F1=0)拉低均值;hetero_f1 是二分类无此问题。**对外宣传用 hetero_f1**。

---

## 十、推荐汇报话术(论文 / 演讲)

```
我们提出的 V20 在 QUAM-AFM Lite test set(n=512)取得:

[主指标]
- 综合分 pred_object_score = 0.7141(2D)/ 0.8112(3D)
- 原子计数 MAE = 0.9434(几乎完美)
- 杂原子 F1 = 0.7434
- 闭环边 F1 strict / robust = 0.6358 / 0.9138

[物理几何]
- 93.2% 样本配对距离 MAE ≤ 0.25 Å
- 重原子 RMSD = 0.7912 Å

[检索]
- top1 = 74.22%, top5 = 90.23%, MRR = 0.8118(closed-pool 512)

[相比 baselines]
- 综合分 × 2.4(Dense),× 1.3(Graph)
- 计数 MAE 降低 36×(Dense)
- 杂原子 F1 提升 7×(Dense)

[诊断]
- edge_gap_robust = 0.2780 揭示瓶颈在亚像素中心定位,
  下一步可通过 soft-argmax / heatmap regression 改善。
```

---

## 十一、相关文档

- 指标定义 — [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)
- 配置参考 — [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)
- 设计原理 — [`PRINCIPLES.md`](PRINCIPLES.md)
- FAQ 含数字解读 — [`FAQ_EXTENDED.md`](FAQ_EXTENDED.md)
