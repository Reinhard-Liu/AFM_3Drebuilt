# 结果解读手册(Result Interpretation)

> 帮助你看懂 `experiments/<exp>/reports/*.md` 与 `*.json`。指标定义见 [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)。

---

## 一、训练 history(`history_v19_object_joint.json`)怎么读

### 1.1 字段结构

```jsonc
[
  {
    "epoch": 0,
    "train_loss": 1.234,
    "val_loss": 1.111,
    "val_metrics": {
      "atom_center_score_r3": 0.97,
      "pred_object_score": 0.45,
      ...
    },
    "lambda_snapshot": { "lambda_type_obj_pred": 0.25, ... },
    "alpha_snapshot": 0.0,
    "lr": 0.00008,
    "best_so_far": false
  },
  ...
]
```

### 1.2 关键判断

- **train_loss 单调降但 val 卡** — 过拟合;调高 `weight_decay` 或减 epoch
- **val_loss 降但 score 卡** — 损失函数与评估指标不一致;检查权重
- **alpha 仍是 0** — curriculum 还没启动,等到 epoch >= warmup_epochs
- **某个 lambda 突然变 0** — 配置里 final = 0(如 `lambda_atom_aux_final=0.05`)

### 1.3 用 `v19_object_joint_review.py` 做可视化

```bash
python3 -m src.v19_object_joint_review \
    --checkpoint $CKPT \
    --history experiments/v19_object_joint_full15_all/checkpoints/history_v19_object_joint.json \
    --output_dir experiments/v19_object_joint_full15_all/review
```

输出 `review/plots/`:
- `loss_curves.png` — train vs val loss
- `score_curves.png` — 每个指标的 epoch 曲线
- `lambda_curves.png` — λ 与 α 调度

---

## 二、EXP-01 Full-test 报告(`fulltest_object_test.md`)

### 2.1 顶部表

```
| 指标 | 测试集 | val 参考 |
|---|---|---|
| pred_object_score | 0.7141 | 0.7335 |
| pred_object_type_acc | 0.6942 | 0.7104 |
...
```

**val 参考** = 训练日志最后一 epoch 的 val 数字。**test 数字必然略低**(0.01–0.03),正常。

### 2.2 分布字段

```
| 字段 | mean | std | min | p25 | p50 | p75 | max |
```

- `std / mean > 0.2` — 性能不均衡,有一批"差样本"拖后腿
- p50 接近 mean — 分布对称
- p50 << mean — 长尾差样本

### 2.3 `samples/` 子目录

`best_sample_<idx>.png`:综合分最高的样本(模型最得意的)
`median_sample_<idx>.png`:综合分中位
`worst_sample_<idx>.png`:综合分最低(看模型为什么败)

每张图 9 宫格:AFM 切片(top)/ GT(中)/ pred(bottom),分别画中心图、类型图、3D。

---

## 三、EXP-02 检索报告

### 3.1 总体

```
Top-1: 0.7422
Top-3: 0.8633
Top-5: 0.9023
MRR:   0.8118
```

### 3.2 分层

`plots/atom_count_stratification.png` — 按原子数分组的 Top-K
`plots/pred_object_score_stratification.png` — 按对象级综合分分组的检索表现

**典型解读**:
- 大分子 Top-1 高于小分子 — 大分子 embedding 更独特
- 杂原子多的子集表现好 — 杂原子是强 feature

---

## 四、EXP-03 缝隙诊断怎么读

### 4.1 核心数字

```
edge_gap_robust_mean: 0.2780
samples_with_gap_ge_0.2: 80.66%
```

含义:**80%+ 样本的拓扑都几乎对了,只是中心定位有亚像素偏差**。

### 4.2 关联图

| 图 | X 轴 | Y 轴 | 解读 |
|---|---|---|---|
| `gap_vs_z_mae.png` | edge_gap | z_mae | 强相关 — z 误差是 gap 的主因 |
| `gap_vs_bond_len_mae.png` | edge_gap | bond_len_mae | 中等相关 — 键长越偏 gap 越大 |
| `gap_vs_node_coverage.png` | edge_gap | coverage | 弱相关 — gap 不是漏检导致 |
| `edge_gap_robust_hist.png` | edge_gap | freq | 双峰?长尾?分布形状揭示问题分布 |

**结论** — gap 来源是"亚像素中心定位",而非"图结构能力"。下一步优化方向:亚像素 head(soft-argmax / heatmap regression)。

---

## 五、EXP-04 几何诊断

### 5.1 关键数字

```
heavy_rmsd_mean: 0.066 Å      (重原子 RMSD)
z_mae_mean: 0.0946            (归一化 z 误差)
bond_len_mae_mean: 0.187 Å    (键长 MAE)
nonplanarity_mae_mean: 0.05   (平面性偏差)
pair_dist_pass_rate@0.25: 93.16%
```

### 5.2 图解读

- `coverage_vs_pair_dist.png` — coverage = 1 时距离误差小;coverage 不全时拉散
- `height_span_vs_z_mae.png` — z 跨度大的分子 z 误差大(直觉)
- `gt_nonplanarity_vs_error.png` — 非平面分子综合误差比平面分子高 ~30%

---

## 六、SUP-01 / SUP-02 Baseline 对照

### 6.1 比较表(SUP-01 vs V20)

| 指标 | V20(Object) | Dense(2D) | 比例 |
|---|---|---|---|
| pred_object_score | 0.7141 | 0.2986 | × 2.39 |
| pred_object_type_acc | 0.6942 | 0.1221 | × 5.68 |
| pred_object_macro_f1 | 0.5345 | 0.0521 | × 10.25 |
| pred_object_hetero_f1 | 0.7434 | 0.2080 | × 3.57 |
| atom_count_mae | 0.94 | 19.86 | ÷ 21.1 |

**这是 V20 对外最强的对照** — Dense baseline 与 V20 唯一架构差异 = 没有对象级头。差距全部归因于对象级头 + 计数闭环。

### 6.2 SUP-02 Graph baseline

GNN 看不到原始 AFM 图像,只接 dense 中心 + 类型图作为节点 feature。**比 Dense 强但远不如 V20** —— 证明端到端学比"先 2D 再图"好。

---

## 七、消融报告(EXP-06 / EXP-07)

V20 ablation 的 `experiments/v20_ablate_*/reports/ablation_summary.json` 含每个变体的指标:

```jsonc
{
  "baseline":              { "pred_object_score": 0.7141, ... },
  "no_curriculum":         { "pred_object_score": 0.5612, ... },  // -0.15
  "no_teacher_consistency":{ "pred_object_score": 0.7929, ... },  // 注意:V20 缩减集变体
  "no_object_count":       { "pred_object_score": 0.6520, ... },  // -0.06
  "no_z_head":             { "pred_object_score": 0.7164, ... },  // 几乎不变
  "no_edge_head":          { "pred_object_score": 0.5761, ... }   // -0.14
}
```

> **注意**:V20 ablation 变体在缩减子集上跑,部分数字看起来比主线高(如 no_teacher_consistency 0.7929)是因为 ablation 用了不同的 epoch / split。**只看相对差距,不直接和主线比**。

详细解读见 `experiments/v20_ablate_*/reports/ablation_summary.md`。

---

## 八、可视化样本图怎么读

### 8.1 9 宫格图(`sample_*.png`)

```
+-----------+-----------+-----------+
|  AFM 切片 |  GT 中心  |  pred 中心|
+-----------+-----------+-----------+
|     -     |  GT 类型  | pred 类型 |
+-----------+-----------+-----------+
|     -     |  GT 3D    | pred 3D   |
+-----------+-----------+-----------+
```

### 8.2 关键看点

- **中心图** — 红色 = 高响应。GT 是高斯热斑,pred 应大致重合
- **类型图** — 颜色对应元素(C 灰、H 白、N 蓝、O 红、F 绿、...)。GT vs pred 的颜色差异 = 类型错分
- **3D** — 球棍模型。Hungarian 匹配后取最优 RMSD 视角

### 8.3 worst sample 怎么诊断?

通常 4 大失败模式:

| 失败模式 | 中心图 | 类型图 | 3D |
|---|---|---|---|
| 漏检 | pred 缺 peak | 缺位置 | 缺原子 |
| 误检 | pred 多 peak | 多杂色 | 多原子 |
| 类型错 | OK | pred 颜色变 | 类型变 |
| z 偏 | OK | OK | 高度偏 |

---

## 九、检索 Top-5 图(`*_5mol.png`)

```
+--------+--------+--------+--------+--------+--------+
|  GT    | Top-1  | Top-2  | Top-3  | Top-4  | Top-5  |
| 分子图 | 候选1  | 候选2  | 候选3  | 候选4  | 候选5  |
+--------+--------+--------+--------+--------+--------+
```

绿色边框 = Top-K 命中 GT。

---

## 十、`v19_v20_experiment_summary.md`(总索引)

由 `python3 -m src.tools.generate_v19_v20_experiment_summary` 重新生成。

包含:
- 所有 ckpt 路径
- 所有报告路径
- 所有可视化目录
- V19 / V20 主要数字汇总表
- 消融对照
- 字段中文释义

**这是你跨实验对比的总入口**。

---

## 十一、相关文档

- 指标定义 — [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)
- 配置 lambda 调参 — [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)
- 总实验索引 — [`V19_V20实验总索引与总结.md`](V19_V20实验总索引与总结.md)
