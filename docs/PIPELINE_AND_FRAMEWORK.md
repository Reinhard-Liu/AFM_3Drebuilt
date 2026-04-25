# 流程与框架(Pipeline & Framework)

> 本文件用 ASCII / mermaid 风格图表呈现项目**整体流程、网络架构、数据流、模块依赖**。源码细节请见 [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md)。

---

## 一、整体训练-评估流程

```
┌────────────────────────────────────────────────────────────────────┐
│                          准备阶段                                  │
├────────────────────────────────────────────────────────────────────┤
│ [1] 环境       conda + CUDA + PyTorch + (可选) RDKit               │
│ [2] 数据       下载 QUAM-AFM (K-1) → /path/to/K-1/                 │
│ [3] 自检       python3 -m src.quick_test       (~1 分钟 smoke)     │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                          训练阶段                                  │
├────────────────────────────────────────────────────────────────────┤
│ V19 主线   bash scripts/launchers/run_v19_object_joint_full15_all.sh│
│             ├─ 全样本 68,555 分子 × 15 epoch                        │
│             ├─ A100 单卡 ~36h                                       │
│             └─ 输出 best_v19_object_joint.pt + history             │
│                                                                    │
│ V20 主线   bash scripts/launchers/run_v20_object_joint_medium10.sh │
│             ├─ 缩减集 65k 分子 × 10 epoch                           │
│             ├─ A100 单卡 ~10h                                       │
│             └─ 新增对象计数头、双输入类型头、curriculum 提速        │
│                                                                    │
│ 监控       watch_*.sh / monitor_*.sh / supervise_*.sh              │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                       评估阶段(test split)                       │
├────────────────────────────────────────────────────────────────────┤
│ EXP-01  对象级 Full-test       v20_eval_fulltest_object.py         │
│ EXP-02  闭集检索               v20_eval_retrieval_full.py          │
│ EXP-03  Strict/Robust 缝隙诊断 v20_error_decompose.py              │
│ EXP-04  3D 几何诊断            v20_geom_diagnostics.py             │
│ SUP-01  Dense baseline 对照    v20_eval_dense_baseline.py          │
│ SUP-02  Graph baseline 对照    v20_eval_graph_baseline.py          │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                       可视化 / 复盘                                │
├────────────────────────────────────────────────────────────────────┤
│ v19_visualize_test15.py         AFM/GT/pred 并排,15 样本          │
│ visualize_5mol.py               Top-5 候选分子并排                 │
│ v19_object_joint_review.py      训练曲线 + 最优样本                │
│ tools/generate_v19_v20_*.py     重建总索引 markdown 表             │
└────────────────────────────────────────────────────────────────────┘
```

---

## 二、网络架构(Forward 数据流)

```
                        AFM 图像栈
                  (B, 10, 128, 128)
                          │
                          ▼  add channel dim
                   (B, 1, 10, 128, 128)
                          │
                          ▼  PatchEmbedding3D
                   Conv3d(1, 512, kernel=(2,16,16), stride=(2,16,16))
                          │
                          ▼  flatten
                   (B, 320, 512)        ← 5 frames × 8×8 spatial
                          │
                          ▼  + cls_token + positional embed
                   (B, 321, 512)
                          │
                          ▼  × 8 Transformer blocks
                          │     spatial-attn → temporal-attn → MLP
                          │
                          ▼
                   (B, 321, 512)
                          │
                          ▼  drop cls,reshape
                   (B, 512, 5, 8, 8)
                          │
                          ▼  upsample bridge
                   共享特征图 (B, 64, 128, 128)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Center Head        Type Head           Z Head
   (B,1,128,128)      (B,11,128,128)    (B,1,128,128)
        │                 │                 │
        │  σ              │ argmax           │
        ▼                 ▼                 ▼
   peak detect ──► peak_centers (N_obj, 2) ─► z_pred[N_obj]
        │                                     │
        │                                     │
        │           ┌────────────────────────┘
        │           │
        ▼           ▼
   ┌─────────────────────────────────┐
   │     对象级条件头(V19 主创新)  │
   │ Peak-Type Head     ───►  type_logits (N_obj, 11)
   │ Peak-Edge Head     ───►  edge_logits (N_obj, N_obj)
   │ Object Count Head  ───►  count_logits (B, 80) [V20 新增]
   └─────────────────────────────────┘
                          │
                          ▼  V20: 双输入采样、闭环
                          │   pred_type_head ◄── 预测类型图采样
                          │   gt_type_head   ◄── GT 类型图采样
                          │   一致性 KL
                          │
                          ▼  解码 + 后处理
                   {coords, atom_types, bonds}
                          │
                          ▼  RDKit MMFF94/UFF (可选, ≤0.3 Å 弛豫)
                   最终 3D 分子结构
```

---

## 三、训练损失图

```
                    forward 一次 (afm_stack)
                            │
        ┌───────────────────┼─────────────────────┐
        ▼                   ▼                     ▼
   center_logits       type_logits            z_pred
        │                   │                     │
        ▼                   ▼                     ▼
  L_center (BCE+Dice)  L_type (Focal CE)    L_z (L1)
        ▼                   ▼                     ▼
    λ_center             λ_type                 λ_z
                            │
        ┌───────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
  GT-center 路径                          peak-center 路径
        │                                         │
        ▼                                         ▼
  type/edge head                          type/edge head
  (gt_centers)                            (peak_centers)
        │                                         │
        ▼                                         ▼
  L_type_obj (×0.25)                    L_type_obj_peak (curriculum)
  L_edge_obj (×0.25)                    L_edge_obj_peak (curriculum)
                            │
                            └─► curriculum α(epoch) 控制权重切换
                                     │
                                     ▼ V20 新增
                             pred-center 路径
                                     │
                                     ▼
                           L_type_obj_pred (curriculum start 0.25 → final 2.0)
                           L_edge_obj_pred (curriculum start 0.20 → final 1.50)
                           L_pred_consistency (KL: gt-pred 双路一致, λ 0.10→0.50)
                                     │
                                     ▼ V20 新增
                             object count head
                                     │
                                     ▼
                           L_object_count_ce  (× 1.0)
                           L_object_count_mae (× 0.15)

                                     ▼
                            ─────────────────
                            蒸馏 (Type Upper Teacher)
                            L_distill = KL(student/T || teacher/T) × T²
                            (× 1.0 V19 / × 0.5 V20)
                            ─────────────────

                                     ▼
                          total_loss = Σ λ_i * L_i
                                     │
                                     ▼
                              AdamW.backward()
```

---

## 四、模块依赖图

```
src/data/dataset.py ───────────────────────────┐
src/data/afm_stack.py                           │
src/data/augment.py                             │
                                                ▼
src/models/video_vit.py ─────────────► src/models/v19_joint_model.py
src/models/v19_center_type_head.py ──────────► (backbone)
src/models/v19_center_edge_head.py
src/models/v19_object_count_head.py            │
src/models/upsample_bridge.py                   ▼
                                       src/train_v19_object_joint.py
                                                │
                          ┌─────────────────────┼─────────────────────┐
                          ▼                     ▼                     ▼
                src/utils/metrics.py    src/utils/peak_detect.py  src/utils/visualize.py
                          │                     │                     │
                          ▼                     │                     ▼
                src/v20_eval_*.py ◄─────────────┘            src/v19_visualize_test15.py
                                                              src/visualize_5mol.py
                                                              src/visualize_val.py

src/models/postprocess.py (RDKit MMFF/UFF, 可选)
src/utils/mol2d.py (2D 分子绘图,Top-5 用)
src/tools/generate_v19_v20_experiment_summary.py (索引表生成)
```

---

## 五、文件路径约定

```
experiments/
└── <exp_name>/
    ├── checkpoints/
    │   ├── best_v19_object_joint.pt         # 最优 ckpt
    │   ├── latest_v19_object_joint.pt       # 最新 ckpt(用于 resume)
    │   ├── history_v19_object_joint.json    # 每 epoch 训练 / 验证指标
    │   └── best_preview.png                 # 训练时随手存的 best 预览
    ├── logs/
    │   └── train.log                        # nohup 输出
    ├── reports/                             # 评估脚本输出
    │   ├── *.md, *.json, *.csv
    │   └── *_records.json (per-sample)
    ├── samples/                             # best/median/worst 三联图
    ├── plots/                               # EXP-02/03/04 直方图
    ├── visualizations_object15/             # 15 样本固定可视化
    ├── visual_compar_object15/              # 同上 + 同分子 5mol Top-5
    └── review/                              # v19_object_joint_review.py 复盘产物
```

---

## 六、数据集 split

K-1 数据集的 split:

| split | 比例 | 数量 | 用途 |
|---|---|---|---|
| train | ~80% | ~54,000 | 训练 |
| val | ~10% | ~6,800 | 训练时早停参考 |
| test | ~10% | ~6,800 | 投稿数字(`v20_eval_fulltest_object.py --split test`) |

切分方式 — 按分子 SMILES 哈希 mod 10。**永远不在 train 上调超参,在 test 上报数**。

---

## 七、相关文档

- 实现细节 — [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md)
- 配置参数 — [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)
- 指标定义 — [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)
- 项目原理 — [`PRINCIPLES.md`](PRINCIPLES.md)
