# 流程与框架(Pipeline & Framework)

> 本文用 ASCII 图表呈现项目**整体流程、网络架构、数据流、模块依赖**。源码细节请见 [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md)。

---

## 一、端到端流程总览

```
QUAM-AFM Lite 数据集
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 1. Dataset (src/data/dataset.py)                          │
 │    ├─ 读 CID 排序后的 split (train/val/test)              │
 │    ├─ 加载 10 张 AFM 切片 (128×128, float32)              │
 │    ├─ 加载真实坐标(归一化 ÷12.0)、元素 index、键索引     │
 │    └─ 双层缓存:samples + ring                            │
 └──────────────────────────────────────────────────────────┘
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 2. Video ViT 主干 (models/video_vit.py)                   │
 │    AFM (B,10,128,128) → tubelet 投影                      │
 │    → (B,320,512) tokens + cls                             │
 │    → 8 层 TransformerBlock                                │
 │    → cls_feat (B,512), patch_feat (B,320,512)             │
 └──────────────────────────────────────────────────────────┘
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 3. V19JointUNet (models/v19_joint_model.py)               │
 │    UNet 5 层编码器 + 三分支解码器                         │
 │    输出 pred (B,13,128,128):                             │
 │      [0]   atom_map(中心热图,sigmoid)                   │
 │      [1]   bond_map(键密度,Tanh)                        │
 │      [2:12] type_map × 10(每元素一个 sigmoid map)        │
 │      [12]  z_map(归一化高度,Tanh)                       │
 │    + features (B,64,128,128) 共享特征                     │
 │    + count_logits (B,86) 全局计数                         │
 └──────────────────────────────────────────────────────────┘
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 4. 对象级条件头(关键)                                   │
 │   ┌──────────────────────────────────────────────┐       │
 │   │ peak_detect(atom_map) → 中心 (N_peaks, 2)    │       │
 │   └──────────────────────────────────────────────┘       │
 │              │                                            │
 │   ┌──────────┴──────────┬──────────────┐                  │
 │   ▼                     ▼              ▼                  │
 │ GT-中心             peak-中心      pred-中心              │
 │ (训练监督)          (训练主路径)    (V20 部署路径)        │
 │   │                     │              │                  │
 │   ▼                     ▼              ▼                  │
 │ CenterConditionedTypeHead → type logits (B,N,10)          │
 │ CenterConditionedEdgeHead → edge logits (B,N,N)           │
 └──────────────────────────────────────────────────────────┘
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 5. 损失合计(`train_v19_object_joint.py:1278-1480`)       │
 │   ├─ L_center  (BCE+Dice,λ=20)                            │
 │   ├─ L_atom_aux (curriculum: 5 → 1)                       │
 │   ├─ L_z       (curriculum: 4 → 8)                        │
 │   ├─ L_type_obj_{gt, peak, pred}                          │
 │   ├─ L_edge_obj_{gt, peak, pred}                          │
 │   ├─ L_type_map_aux / L_bond_map_aux                      │
 │   ├─ L_peak_consistency / L_pred_type_consistency         │
 │   ├─ L_teacher_distill (KD)                               │
 │   └─ L_object_count + L_object_count_mae (V20)            │
 └──────────────────────────────────────────────────────────┘
       │
       ▼ AdamW + Cosine LR + grad_clip 1.0,FP32
   反向传播 → 下一 epoch
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 6. 最佳 ckpt 选择(13-tuple 字典序)                      │
 │   best_v19_object_joint.pt + history.json                │
 └──────────────────────────────────────────────────────────┘
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 7. 评估族 (6 维)                                          │
 │    EXP-01 fulltest_object | EXP-02 retrieval              │
 │    EXP-03 gap_decompose   | EXP-04 geom_diagnostics       │
 │    SUP-01 dense_baseline  | SUP-02 graph_baseline         │
 │    输出 reports/*.{md,json,csv} + plots                   │
 └──────────────────────────────────────────────────────────┘
       │
       ▼
   RDKit 后处理(MMFF94 → UFF fallback,位移 cap 0.3 Å)
       │
       ▼
   弛豫后坐标 + .mol / .sdf / 可视化
```

---

## 二、网络架构(V20 完整图)

```
                    ┌─────────────────────────┐
                    │ AFM (B, 10, 128, 128)    │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
            ┌─────▼────┐    ┌────▼─────┐   ┌────▼────┐
            │ Video ViT │   │ UNet enc  │   │ 直供后续 │
            │ (主干)    │   │ (5 层)    │   │ heads    │
            └─────┬────┘    └────┬─────┘   └────┬────┘
                  │              │              │
        ┌─────────┴─────────┐    │              │
        ▼                   ▼    │              │
    cls_feat            patch_feat              │
    (B, 512)           (B, 320, 512)            │
        │                   │                   │
   ┌────┴────────┐          │                   │
   │ 计数头      │          │                   │
   │ 86 类       │          │                   │
   └─────────────┘          │                   │
                            │                   │
                ┌───────────┼───────────┬───────┴────────┐
                ▼           ▼           ▼                ▼
          ┌──────────┐ ┌──────────┐ ┌──────────┐
          │ Center   │ │ 2D 结构  │ │ Z 高度   │
          │ 解码 → 1ch│ │ 解码 → 12ch│ │ 解码 → 1ch│
          │ (sigmoid)│ │ (Tanh)   │ │ (Tanh)   │
          └────┬─────┘ └────┬─────┘ └────┬─────┘
               │            │            │
               └────────────┼────────────┘
                            │
                  pred (B, 13, 128, 128)
                  features (B, 64, 128, 128)  共享特征供条件头使用
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          GT-中心       peak-中心       pred-中心
              │             │             │
              └─────────────┼─────────────┘
                            ▼
              ┌──────────────────────────────┐
              │ CenterConditionedTypeHead    │
              │ - 采样 shared_feat / AFM      │
              │ - 三级分类 (粗 3 / 异质 / 精 10)│
              │ - focal-CE + label_smoothing │
              └──────────────────────────────┘
                            ▼
              ┌──────────────────────────────┐
              │ CenterConditionedEdgeHead    │
              │ - 节点 MLP (82 → 128)         │
              │ - 边 MLP (263 → 1)            │
              │ - 图精化(零初始化暖启动)    │
              │ - BCEWithLogits + pos_weight │
              └──────────────────────────────┘
                            │
                            ▼
                       loss summation
```

---

## 三、数据集流程

```
QUAM-AFM Lite (1755 分子)
    │
    │  按 CID 字符串排序
    ▼
sorted_cids = [cid_001, cid_002, ..., cid_1755]
    │
    │  线性切分(确定性,no random shuffle):
    │  test  = sorted_cids[:val_size=512]
    │  val   = sorted_cids[512:1024]
    │  train = sorted_cids[1024:]
    ▼
QUAMAFMDataset
    │
    │  __getitem__(idx):
    │    1. 读 .npy / .h5(取决于格式)
    │    2. 10 张 128×128 AFM 拼成 (10,128,128)
    │    3. coords (n_atoms, 3) ÷ 12.0 → 归一化空间
    │    4. atom_types (n_atoms,) ∈ {0..9}
    │    5. 候选键由距离 ≤ ideal × 1.2 推断
    │    6. mask (MAX_ATOMS=85,) padding 标记
    │    7. (可选) ring 标签 / V17-bridge 标签
    │    8. (可选) augment_rotation:0-360° XY 旋转
    ▼
DataLoader (batch_size=8, num_workers=8)
    │
    │  collate:
    │    AFM stack (B, 10, 128, 128)
    │    coords     (B, 85, 3)
    │    atom_types (B, 85)
    │    masks      (B, 85)
    │    edges      (B, 85, 85)
    │    其他辅助标签
    ▼
训练 / 评估
```

**确定性切分**:无 random / hash,保证不同机器结果一致。

---

## 四、对象级头采样细节

```
                pred (B, 13, 128, 128)
                features (B, 64, 128, 128)
                AFM (B, 10, 128, 128)
                center_coords (B, N, 2/3)  [from GT/peak/pred]
                                       │
            ┌──────────────────────────┘
            │
            ▼
  对每个中心点 (xi, yi):
    ┌─────────────────────────────────────┐
    │ 1. 共享特征采样(64 ch × 3 stat):     │
    │    - center sample (双线性插值)      │
    │    - 邻域 mean (radius 1.0 px)       │
    │    - 邻域 max  (radius 1.0 px)       │
    │  → 192 维                            │
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │ 2. AFM 采样(10 ch × 3 stat):         │
    │    - center sample (radius 2.0)      │
    │    - 邻域 mean / max (radius 2.0)    │
    │  → 30 维                             │
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │ 3. center 高斯统计(3 维):           │
    │    [center_val, mean_2px, max_2px]   │
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │ 4. 坐标特征(3 维):  [x, y, z]       │
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │ 5. 环境特征(5 维,基于近邻):       │
    │    [n_neighbors/6, mean, min, max,   │
    │     var of neighbour distances]      │
    │    邻域阈值 = 0.20(归一化空间)      │
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │ 6. Patch grid 5×5(供精分类):       │
    │    在中心周围 ±2px 取 25 个点,        │
    │    每点采样 10 类 type_map → 250 维   │
    │    经 patch encoder → 192 维          │
    └─────────────────────────────────────┘
            │
            ▼
   主干 MLP: 233 维 → 192 → 192
            │
   ┌────────┼─────────┬─────────┐
   ▼        ▼         ▼         ▼
 粗 3 类   异质 2     精分类(192+patch=196 → 192 → 10)
 (Linear) (Linear)   focal-CE + label_smoothing 0.02
```

---

## 五、训练循环时序图

```
for epoch in range(epochs):
    ┌─────────────────────────────────────────────────────┐
    │ 计算 epoch 级 lambdas                              │
    │   lambda_type_obj_peak = scheduled_weight(...)     │
    │   alpha = scheduled_weight(...)                     │
    │   λ_pred_type_consistency = ...                     │
    └─────────────────────────────────────────────────────┘
          │
          ▼
    for batch in train_loader:
       ┌────────────────────────────────────────────────┐
       │ 1. forward V19JointUNet(afm)                  │
       │     → pred, features, count_logits             │
       │ 2. peak_detect(pred[:,0]) → peak coords        │
       │ 3. CenterConditionedTypeHead(GT center)        │
       │ 4. CenterConditionedTypeHead(peak center)      │
       │ 5. CenterConditionedTypeHead(pred center)[V20] │
       │ 6. CenterConditionedEdgeHead(同上 3 路径)       │
       │ 7. teacher_encoder/classifier(GT center)        │
       │ 8. 计算 loss(各项加权累加)                    │
       │ 9. loss.backward() + clip_grad_norm_(1.0)      │
       │ 10. optimizer.step() + scheduler.step()(per-step)│
       └────────────────────────────────────────────────┘
          │
          ▼
    评估 val:
       ┌────────────────────────────────────────────────┐
       │ for batch in val_loader:                       │
       │   forward + peak detect + 对象头三路径          │
       │   计算所有 6-dim 指标                          │
       │ 聚合: mean(metric) over val set                │
       └────────────────────────────────────────────────┘
          │
          ▼
    更新最佳 ckpt(13-tuple 字典序)
       ┌────────────────────────────────────────────────┐
       │ key = (pred_object_score, pred_object_3d_score,│
       │        -count_mae, macro_f1, edge_f1,           │
       │        peak_object_score, peak_center_edge_f1, │
       │        ...) (13 keys)                          │
       │ if key > best_key:                             │
       │   save best.pt                                 │
       └────────────────────────────────────────────────┘
       save latest.pt(总是)
       追加 history.json(逐 epoch)
```

---

## 六、评估族(6 维)流程

```
ckpt + config
     │
     ├──▶ EXP-01 fulltest_object
     │      ├─ 协议:闭环推理(pred 中心 + pred 类型/边)
     │      ├─ 计算所有 pred_* / peak_* / gt_* 指标
     │      └─ 输出:reports/fulltest_object_test.{md,json,csv}
     │             samples/<idx>_best.png(可视化)
     │
     ├──▶ EXP-02 retrieval_full
     │      ├─ 协议:closed_world_test_pool(候选 = 全 512 测试)
     │      ├─ 全局 embedding cosine sim → top-K + MRR
     │      └─ 输出:retrieval_fulltest_test.{md,json}
     │             plots/cosine_distribution.png
     │
     ├──▶ EXP-03 gap_decompose
     │      ├─ 协议:edge_match_radius=3.0 px
     │      ├─ strict / robust edge F1 + matched coverage
     │      └─ 输出:gap_decomposition_test.{json,csv}
     │
     ├──▶ EXP-04 geom_diagnostics
     │      ├─ 协议:xy_match_radius=3.0 px,Å 单位
     │      ├─ pair_dist_mae / bond_len_mae / z_corr / nonplanarity
     │      └─ 输出:geom_diagnostics_test.{json,csv}
     │
     ├──▶ SUP-01 dense_baseline
     │      ├─ 协议:LegacyDenseHead + peak detect + argmax 解码
     │      ├─ 仅作 V20 对照
     │      └─ 输出:dense_baseline_fulltest.{md,json,csv}
     │
     └──▶ SUP-02 graph_baseline
            ├─ 协议:LegacyGNNTypeHeadAdapter(GNN 替换条件头)
            ├─ 仅作 V20 对照
            └─ 输出:graph_baseline_fulltest.{md,json,csv}
```

每个评估脚本均接 `--checkpoint best.pt --config config.json --split test --output_dir reports/`。

---

## 七、代码模块依赖图

```
src/train_v19_object_joint.py
   │
   ├─→ src/data/dataset.py
   │     QUAMAFMDataset
   │     build_dataloaders()
   │
   ├─→ src/models/v19_joint_model.py
   │     V19JointUNet
   │     ├─→ models/video_vit.py
   │     │      PatchEmbedding3D
   │     │      VideoViTEncoder
   │     │      TransformerBlock
   │     └─→ models/prediction_heads.py
   │            AtomCountHead
   │
   ├─→ src/models/v19_center_type_head.py
   │     CenterConditionedTypeHead
   │
   ├─→ src/models/v19_center_edge_head.py
   │     CenterConditionedEdgeHead
   │
   ├─→ src/models/v20_ablation_heads.py
   │     LegacyGNNTypeHeadAdapter
   │     ZeroEdgeHead
   │
   ├─→ src/utils/peak_detect.py
   │     peak detection(局部最大)
   │
   ├─→ src/utils/metrics.py
   │     hungarian_match_numpy
   │     compute_object_metrics
   │     bond_validity / valence_validity
   │     atom_position_score / atom_semantic_score / ...
   │
   └─→ src/models/postprocess.py(评估时)
         coords_to_mol
         rdkit_relaxation

src/v20_eval_*.py(评估族)
   ├─→ 共享上述全部模块
   ├─→ src/models/ring_detection.py(EXP-04 部分指标)
   └─→ scipy.optimize.linear_sum_assignment(Hungarian)

scripts/launchers/run_*.sh
   ├─→ python -m src.train_v19_object_joint --config <json>
   ├─→ tee <save_dir>/train.log

scripts/launchers/{watch,monitor,supervise}_*.sh
   ├─→ bash run_*.sh(失败 / 卡顿时)
```

---

## 八、配置文件 → 训练流程映射

```
config.json
    │
    │  json.load(...)
    ▼
cfg = {...}
    │
    ├─ 数据相关 ─────▶ build_dataloaders(cfg)
    │                    └─→ QUAMAFMDataset(...)
    │
    ├─ 模型相关 ─────▶ V19JointUNet(base_ch=cfg["base_ch"], ...)
    │                  CenterConditionedTypeHead(...)
    │                  CenterConditionedEdgeHead(...)
    │
    ├─ 训练超参 ─────▶ optim.AdamW(lr, weight_decay)
    │                  optim.lr_scheduler.CosineAnnealingLR(T_max=epochs)
    │
    ├─ 教师 ─────────▶ build_type_teacher(teacher_ckpt)
    │
    ├─ Curriculum ──▶ scheduled_weight(epoch, start, final, warmup)
    │                  → 每 epoch 重算所有 lambdas
    │
    └─ 加载 / Resume ▶ load_state_dict(warm_start) [strict=False]
                       或 load_state_dict(resume) [strict=True]
                       含 optimizer + scheduler + history
```

---

## 九、Launcher 三层监控协作

```
   user
    │ tmux / screen
    ▼
┌──────────────────────────┐
│ supervise_<exp>.sh       │
│  while true:             │
│    run_once (前台同步)   │
│    if exit_code == 0:    │
│      check epoch ≥ target│
│      → exit if done      │
│    else:                 │
│      sleep 10            │
└──────┬───────────────────┘
       │
       │ 启动训练
       ▼
┌──────────────────────────┐                ┌─────────────────────────┐
│ python -m src.train_*    │ ◀─── kill ──── │ monitor_<exp>.sh        │
│   nohup,写 train.log    │                │  每 60s 检查 ckpt mtime │
└──────────────────────────┘                │  > STALL_SECONDS → kill │
                                            │  V19=1200s, V20=1800s   │
                                            └─────────────────────────┘
                                            (供 supervise 重启)
┌──────────────────────────┐
│ watch_<exp>.sh(可选)    │
│  pgrep -f "src.train_*" │
│  无进程 → bash run.sh   │
└──────────────────────────┘
```

通常**三选其一**或两两组合即可,无需全开。

---

## 十、Workflow 全景:从克隆到论文数字

```
1. git clone https://github.com/Reinhard-Liu/AFM_3Drebuilt.git
   cd AFM_3Drebuilt

2. conda env create -f environment.yml
   conda activate afm

3. 下载 QUAM-AFM Lite → /root/autodl-tmp/K-1/

4. (可选)V19 暖启动 + Type Upper Teacher 短训
   bash scripts/launchers/run_v19_object_joint_full6h.sh
   bash scripts/launchers/run_v19_type_upper_debug.sh

5. V19 主线 15 epoch 训练
   bash scripts/launchers/run_v19_object_joint_full15_all.sh
   tail -f experiments/v19_object_joint_full15_all/checkpoints/train.log

6. V20 主线 10 epoch 训练(继承 V19)
   bash scripts/launchers/run_v20_object_joint_medium10.sh

7. 6 维评估
   for exp in fulltest_object retrieval_full gap_decompose geom_diagnostics \
              dense_baseline graph_baseline; do
     python -m src.v20_eval_${exp} --checkpoint .../best.pt ...
   done

8. 复盘
   python -m src.v19_object_joint_review --checkpoint .../best.pt
   python -m src.tools.generate_v19_v20_experiment_summary

9. 论文表 / 图直接引用 reports/*.{md,json,csv}
```

---

## 十一、相关文档

- 实现细节 — [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md)
- 配置参考 — [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)
- 设计原理 — [`PRINCIPLES.md`](PRINCIPLES.md)
- 指标定义 — [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)
- 结果解读 — [`RESULT_INTERPRETATION.md`](RESULT_INTERPRETATION.md)
- 排错 — [`RUNTIME_TROUBLESHOOTING.md`](RUNTIME_TROUBLESHOOTING.md)
