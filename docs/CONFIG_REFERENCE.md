# 配置参考(Config Reference)

> 本文枚举 `configs/` 下**所有 JSON 配置**的字段含义、默认值、V19 vs V20 差异、与对应训练循环的耦合点。所有事实来自源码,带 `file:line` 引用。
>
> 实现层信息见 [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md);设计动机见 [`PRINCIPLES.md`](PRINCIPLES.md)。

---

## 一、Configs 目录清单

`configs/` 下共 10 份 JSON:

| 文件 | 用途 |
|---|---|
| `config.json` | 历史扩散主线占位(V1–V6) |
| `config_v17_bridge_b_eval.json` | V17 Bridge 评估(过期) |
| `config_v17_bridge_token_comp_eval.json` | V17 Token-comp 评估 |
| `config_v17_pred_count_eval.json` | V17 计数评估 |
| **`config_v19_object_joint_full15_all.json`** | **V19 主线训练**(15 epoch 全 K-1) |
| `config_v19_object_joint_full6h.json` | V19 6 小时短训(用于 warm start teacher) |
| `config_v19_object_joint_medium.json` | V19 medium 子集快速验证 |
| `config_v20_dense_stage1_medium10.json` | SUP-01 dense baseline |
| `config_v20_graph_baseline_medium10.json` | SUP-02 graph baseline |
| **`config_v20_object_joint_medium10.json`** | **V20 主线训练**(10 epoch medium) |

主线为加粗两份。其余 baselines / 旁支与主线共享同一个 `train_v19_object_joint.py` 入口,仅切配置文件即可。

---

## 二、V19 主线配置完整字段表

**文件**:`configs/config_v19_object_joint_full15_all.json`

### 2.1 基础数据 / 训练参数

| Key | 值 | 含义 / 耦合点 |
|---|---|---|
| `data_root` | `"auto"` | 自动尝试 `/root/autodl-tmp/K-1/` 与 `/root/autodl-tmp/micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM`;不存在则报错 |
| `save_dir` | `"…/experiments/v19_object_joint_full15_all/checkpoints"` | ckpt + history 输出目录 |
| `param_key` | `"K-1"` | 选用 QUAM-AFM 第 K-1 参数组(振幅 0.4 Å × CO 0.4 N/m 等) |
| `img_size` | 128 | AFM 切片缩放后的 H,W |
| `min_corrugation` | 0.0 Å | Z 高差下限,过滤过分平坦样本 |
| `augment_rotation` | true | 训练集启用 0–360° XY 旋转增广 |
| `require_ring` | false | 不要求样本必含 5/6 元环 |
| `batch_size` | 8 | 单卡训练 batch |
| `num_workers` | 8 | DataLoader worker 数 |
| `max_samples` | 0 | 0 = 无上限 |
| `val_size` | 512 | val 与 test 各 512 样本(均匀切分) |
| `epochs` | 15 | 总训练轮数 |
| `lr` | 1.5e-4 | AdamW 初始学习率 |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `min_lr` | 1e-5 | CosineAnnealingLR 下限 |
| `base_ch` | 64 | UNet 基础通道(三分支共享) |

### 2.2 暖启动与教师

| Key | 值 | 用法 |
|---|---|---|
| `warm_start_checkpoint` | `…/v19_object_joint_full6h/.../best_v19_object_joint.pt` | 非严格加载(strict=False),仅 model + heads,**不**继承 optimizer/scheduler |
| `teacher_type_checkpoint` | `…/v19_type_upper_debug/.../best_v19_type_upper.pt` | 类型 upper teacher,使用 GT 中心训练完成 |
| `teacher_temperature` | 1.5 | KL 蒸馏温度 |
| `lambda_teacher_type_distill` | 1.0 | GT-中心 student 与 teacher 的 KL 权重 |

### 2.3 损失权重(每分支)

| Key | 值 | 影响 |
|---|---|---|
| `lambda_center` | 20.0 | 中心高斯 BCE+Dice 强约束(若误,所有下游头崩) |
| `lambda_atom_aux_start / final` | 5.0 / 1.0 | dense atom map 辅助分支 |
| `lambda_z_start / final` | 4.0 / 8.0 | z 回归 L1(随训练加重) |
| `lambda_type_obj_gt` | 1.5 | GT-中心条件头分类 |
| `lambda_type_obj_peak_start / final` | 0.25 / 2.5 | peak-中心条件头(curriculum 主路径) |
| `lambda_edge_obj_gt` | 1.5 | GT-中心条件边 |
| `lambda_edge_obj_peak_start / final` | 0.25 / 2.5 | peak-中心条件边 |
| `lambda_type_map_aux_start / final` | 1.0 / 0.15 | 类型 dense map 辅助(逐渐弱化) |
| `lambda_bond_map_aux_start / final` | 1.0 / 0.15 | 键 dense map 辅助 |
| `lambda_peak_consistency_start / final` | 0.0 / 0.5 | GT vs peak 路径一致性蒸馏 |

### 2.4 Curriculum 调度

| Key | 值 | 含义 |
|---|---|---|
| `aux_decay_epochs` | 12 | atom_aux / type_map_aux / bond_map_aux 衰减周期 |
| `loss_warmup_epochs` | 12 | type/edge peak 头 ramp-up 周期 |
| `center_curriculum_alpha_start` | 0.0 | 起始:100% 用 GT 中心 |
| `center_curriculum_alpha_final` | 1.0 | 终止:100% 用 peak 中心 |
| `center_curriculum_warmup_epochs` | 12 | α 线性插值长度 |
| `consistency_temperature` | 1.5 | 一致性 KL 温度 |
| `center_search_radius` | 3 | 局部 peak 检测半径(像素) |

### 2.5 Resume 类(可选)

`resume_from_checkpoint`:严格加载完整状态(model + optimizer + scheduler + history)。CLI `--resume_checkpoint` 也可指定,优先级高于 config(`train_v19_object_joint.py:1589-1596`)。

---

## 三、V20 主线配置 — V19 增量 diff

**文件**:`configs/config_v20_object_joint_medium10.json`

### 3.1 修改的字段(V19 → V20)

| Key | V19 | V20 | 备注 |
|---|---|---|---|
| `max_samples` | 0(无限) | **65536** | medium 子集上限 |
| `epochs` | 15 | **10** | 缩短训练 |
| `lr` | 1.5e-4 | **8e-5** | 更稳定的低 lr |
| `aux_decay_epochs` | 12 | **5** | 加速辅助衰减 |
| `loss_warmup_epochs` | 12 | **5** | 加速 ramp-up |
| `center_curriculum_warmup_epochs` | 12 | **5** | 加速 curriculum |

### 3.2 V20 新增字段

| Key | 值 | 含义 |
|---|---|---|
| `lambda_teacher_type_pred_distill` | 0.0 | pred-中心二次蒸馏(默认关) |
| `lambda_type_obj_pred_start / final` | 0.25 / **2.0** | pred-中心条件头分类(V20 新路径) |
| `lambda_edge_obj_pred_start / final` | 0.20 / **1.50** | pred-中心条件边 |
| `lambda_pred_type_consistency_start / final` | 0.10 / **0.50** | gt vs pred 类型 KL 一致性 |
| `pred_train_match_radius_px` | 4.0 | pred-中心与 GT-中心配对的有效半径(像素) |
| `lambda_object_count` | 1.0 | 全局计数头 CE 显式权重(V19 不显式加权) |
| `lambda_object_count_mae` | 0.15 | 计数回归 MAE 权重 |

### 3.3 不变的字段

`batch_size=8`、`num_workers=8`、`weight_decay=1e-4`、`min_lr=1e-5`、`base_ch=64`、`teacher_temperature=1.5`、`lambda_teacher_type_distill=1.0`、`lambda_center=20.0` 等保持一致。

V20 通过这些**增量**实现"对象级闭环":全局计数 → peak 采样 → pred-center 类型/边 → 与 gt-path 一致性蒸馏 → 反馈到全局计数。

---

## 四、训练循环关键耦合(`src/train_v19_object_joint.py`)

### 4.1 优化器 / 调度器

`train_v19_object_joint.py:1157-1163`

```python
optimizer = optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(cfg["epochs"], 1), eta_min=cfg["min_lr"]
)
```

- 默认 `betas=(0.9, 0.999)`、`eps=1e-8`(PyTorch 默认)
- `T_max = epochs` → 全程一个 cosine 周期
- 不使用 warmup(因为多个 lambda 已经做了 curriculum)

### 4.2 梯度裁剪

`train_v19_object_joint.py:1481`

```python
torch.nn.utils.clip_grad_norm_(
    list(model.parameters()) + list(type_head.parameters()) + list(edge_head.parameters()),
    max_norm=1.0
)
```

### 4.3 Mixed precision

代码**不使用** `torch.cuda.amp.GradScaler` / `autocast`。`mixed_precision` 字段(若 config 出现)目前为占位,FP32 全程。

### 4.4 Best ckpt 选择

`train_v19_object_joint.py:1532-1563` 13 维元组按字典序比较:

```python
current_key = (
    pred_object_score,           # 主指标
    pred_object_3d_score,        # 3D 综合
    -pred_object_count_mae,      # 越小越好故取负
    pred_object_macro_f1,
    pred_object_edge_f1,
    peak_object_score,
    peak_center_edge_f1,
    ...                          # 共 13 个
)
if best_key is None or current_key > best_key:
    best_key = current_key
    torch.save(state, save_dir / "best_v19_object_joint.pt")
```

### 4.5 λ / α 调度公式

`train_v19_object_joint.py:87-95`:

```python
def scheduled_weight(epoch, final, start, warmup_epochs):
    if warmup_epochs <= 1: return final
    if epoch <= 1:         return start
    if epoch >= warmup_epochs: return final
    alpha = (epoch - 1) / max(warmup_epochs - 1, 1)
    return start + alpha * (final - start)
```

**线性插值**(非 cosine、非 step)。每个 `lambda_*_start/final` 与 `center_curriculum_alpha_*` 都用相同公式。

### 4.6 蒸馏调用

`train_v19_object_joint.py:1377-1397`:

```python
with torch.no_grad():
    _, teacher_patches = teacher_encoder(afm)
    teacher_logits = teacher_classifier(coords_obj, teacher_patches, mask, afm_stack=afm)

T = teacher_temperature
student = gt_center_logits[valid] / T
teacher = teacher_logits[valid]   / T
distill_loss = F.kl_div(
    F.log_softmax(student, -1),
    F.softmax(teacher, -1),
    reduction="batchmean"
) * (T ** 2) * lambda_teacher_type_distill
```

**T² 补偿**:KD 标准做法,使梯度尺度独立于温度。

---

## 五、Launcher / 监控脚本配置

### 5.1 启动器命名约定

`scripts/launchers/` 下成对出现:

| 前缀 | 进程 | 用途 |
|---|---|---|
| `run_*.sh` | 单次启动(nohup) | 训练入口 |
| `watch_*.sh` | 看门狗 | 进程不存在则调用 `run_*.sh` |
| `monitor_*.sh` | 卡顿检测 | 检查 ckpt/log mtime,长时间无更新 → kill |
| `supervise_*.sh` | 自动恢复 | 前台同步执行,失败延迟 10s 重启 |

### 5.2 关键参数

V19 主线 `monitor_v19_object_joint_full15_all.sh`:

```bash
STALL_SECONDS=1200    # 20 分钟无更新 → 触发 kill
CHECK_INTERVAL=60     # 每 60 秒检查一次
```

V20 主线 `monitor_v20_object_joint_medium10.sh`:

```bash
STALL_SECONDS=1800    # V20 容忍度更长(30 分钟)
CHECK_INTERVAL=60
```

`supervise_*.sh` 的核心循环:

```bash
target_epochs=$(读取 config.json 的 epochs)
while true; do
  latest_epoch=$(读取 latest_*.pt 中的 epoch 字段)
  if [[ $latest_epoch -ge $target_epochs ]]; then
    exit 0   # 训练完成
  fi
  run_once   # 前台同步,捕获 exit code
  sleep 10   # 失败重试间隔
done
```

`watch_*.sh` 监视进程是否存活:

```bash
while true; do
  if ! pgrep -f "src.train_v19_object_joint --config ..." ; then
    bash run_*.sh   # 不存活 → 重启
  fi
  sleep 30
done
```

### 5.3 三层监控的协作

```
supervise(主进程,前台同步)
    ├── 失败/退出 → 自动重启
monitor(独立进程)
    ├── 检测到长时间无更新 → kill 主进程
    └── supervise 接住后重启
watch(可选守护)
    └── 进程缺失 → 用 nohup 拉起
```

通常 **三选其一**:训练在 `tmux/screen` 里跑,日常用 `monitor`(检测卡顿)+ `supervise`(自动恢复)即可。`watch` 用在希望完全无人值守时。

---

## 六、SUP-01 / SUP-02 配置摘要

### 6.1 SUP-01 Dense baseline(`config_v20_dense_stage1_medium10.json`)

与 V20 主线相同的 `train_v19_object_joint.py` 入口,但:
- 使用 `LegacyDenseHead`(只输出 dense type/center map,无对象头)
- 评估通过 `v20_eval_dense_baseline.py` 调 peak 检测 + dense argmax 解码
- CLI 关键参数:`--peak_threshold 0.45 --min_distance_px 2 --max_objects 64 --bond_line_mean_threshold 0.18 --bond_line_peak_threshold 0.35 --bond_length_scale 1.35`

### 6.2 SUP-02 Graph baseline(`config_v20_graph_baseline_medium10.json`)

- 使用 `LegacyGNNTypeHeadAdapter`(`v20_ablation_heads.py:14-90`)
- `num_gnn_layers=4`、`bond_threshold=0.20`、`token_grid_size=16`
- 边头替换为 `ZeroEdgeHead` 或仍用 V19 边头(取决于消融变体)
- 对比意义:验证"端到端学比先 2D 再图好"

---

## 七、ablation / 旁支配置变体

`experiments/v20_ablate_*/` 下若干变体,主要差异在 lambda 设置:

| 变体 | 关键改动 |
|---|---|
| `no_curriculum` | `center_curriculum_alpha_start=alpha_final=0`(始终 GT 中心) |
| `no_teacher_consistency` | `lambda_teacher_type_distill=0` |
| `no_object_count` | `lambda_object_count=lambda_object_count_mae=0`(V20 专属) |
| `no_z_head` | 移除 z 分支 / `lambda_z_start=lambda_z_final=0` |
| `no_edge_head` | 替换为 `ZeroEdgeHead` |

具体数字见 [`RESULT_INTERPRETATION.md`](RESULT_INTERPRETATION.md) §七。

---

## 八、调参经验与陷阱

### 8.1 哪些不要轻易动

| 字段 | 原因 |
|---|---|
| `img_size=128` | 下游 head 与 patch grid 硬编码,改了会形状错配 |
| `lambda_center=20.0` | center 是所有下游的源头,弱化它 → peak 检测失败 → 整链塌 |
| `temporal_patch_size=2` / `patch_size=16` | 决定 token 数 320,改了 pos_embed 重训 |
| `COORD_SCALE=12.0`(后处理) | 与 dataset 的归一化耦合,不一致会导致 RDKit 键判定错 |

### 8.2 OOM 优先级

1. `batch_size: 8 → 4 → 2`
2. `num_workers: 8 → 4`
3. 关 `augment_rotation`(略损精度)
4. **不要**改 `img_size`

### 8.3 调 λ 的次序

1. 先确认 `lambda_center` 让 peak 能稳定检测
2. 看 `loss_type_obj_gt` 是否下降(GT 路径)
3. 调 `lambda_type_obj_peak_final`(2.5 → 5.0)激进些 → 看 `peak_center_type_acc`
4. 蒸馏权重 `lambda_teacher_type_distill`(1.0 → 0.5)若 student 与 teacher 差距大
5. V20 路径只动 `lambda_*_pred_*` 与 `lambda_pred_type_consistency_*`,先关一致性看 pred 单独表现

### 8.4 Curriculum 经验

- `center_curriculum_warmup_epochs` 不能 **小于 5**:peak 头要 5 epoch 才能稳定学到中心
- V19 用 12 偏保守,V20 缩到 5 是因为 warm start 自 V19,可以更激进
- 完全关 curriculum(α 始终为 1)→ pred_object_score 掉 0.15 左右(EXP-06 ablation)

---

## 九、配置 JSON Schema(简化)

```jsonc
{
  // 数据
  "data_root":             "auto" | "/abs/path",
  "save_dir":              "/abs/path",
  "param_key":             "K-1",
  "img_size":              128,
  "min_corrugation":       0.0,
  "require_ring":          false,
  "augment_rotation":      true,

  // 训练
  "batch_size":            8,
  "num_workers":           8,
  "max_samples":           0,
  "val_size":              512,
  "epochs":                10 | 15,
  "lr":                    1e-4 (V19=1.5e-4, V20=8e-5),
  "weight_decay":          1e-4,
  "min_lr":                1e-5,
  "base_ch":               64,

  // Resume / Warm start / Teacher
  "warm_start_checkpoint": "/abs/path/best.pt",
  "teacher_type_checkpoint": "/abs/path/teacher.pt",
  "teacher_temperature":   1.5,
  "resume_from_checkpoint": "/abs/path/latest.pt",  // 可选

  // Curriculum
  "loss_warmup_epochs":          12 | 5,
  "aux_decay_epochs":            12 | 5,
  "center_curriculum_alpha_start": 0.0,
  "center_curriculum_alpha_final": 1.0,
  "center_curriculum_warmup_epochs": 12 | 5,
  "consistency_temperature":     1.5,
  "center_search_radius":        3,

  // Lambdas (V19 + V20 共有)
  "lambda_center":                       20.0,
  "lambda_atom_aux_start":               5.0,
  "lambda_atom_aux_final":               1.0,
  "lambda_z_start":                      4.0,
  "lambda_z_final":                      8.0,
  "lambda_type_obj_gt":                  1.5,
  "lambda_type_obj_peak_start":          0.25,
  "lambda_type_obj_peak_final":          2.5,
  "lambda_edge_obj_gt":                  1.5,
  "lambda_edge_obj_peak_start":          0.25,
  "lambda_edge_obj_peak_final":          2.5,
  "lambda_type_map_aux_start":           1.0,
  "lambda_type_map_aux_final":           0.15,
  "lambda_bond_map_aux_start":           1.0,
  "lambda_bond_map_aux_final":           0.15,
  "lambda_peak_consistency_start":       0.0,
  "lambda_peak_consistency_final":       0.5,
  "lambda_teacher_type_distill":         1.0,

  // V20 新增
  "lambda_type_obj_pred_start":          0.25,
  "lambda_type_obj_pred_final":          2.0,
  "lambda_edge_obj_pred_start":          0.20,
  "lambda_edge_obj_pred_final":          1.50,
  "lambda_pred_type_consistency_start":  0.10,
  "lambda_pred_type_consistency_final":  0.50,
  "lambda_teacher_type_pred_distill":    0.0,
  "pred_train_match_radius_px":          4.0,
  "lambda_object_count":                 1.0,
  "lambda_object_count_mae":             0.15
}
```

---

## 十、相关文档

- 实现细节 — [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md)
- 设计原理 — [`PRINCIPLES.md`](PRINCIPLES.md)
- 流程框架 — [`PIPELINE_AND_FRAMEWORK.md`](PIPELINE_AND_FRAMEWORK.md)
- 指标定义 — [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)
- 排错 — [`RUNTIME_TROUBLESHOOTING.md`](RUNTIME_TROUBLESHOOTING.md)
