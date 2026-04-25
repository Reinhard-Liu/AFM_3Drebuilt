# 配置参考(Config Reference)

> 本文件枚举 `configs/config_v*_object_joint_*.json` 中**所有可调参数**:含义、默认值、V19 vs V20 差异、影响指标。配置示例直接读 `configs/`。

---

## 一、基础参数

| 字段 | 类型 | V19 默认 | V20 默认 | 含义 |
|---|---|---|---|---|
| `data_root` | str | `"auto"` | `"auto"` | K-1 数据根。`"auto"` 自动尝试 `/root/autodl-tmp/K-1/` 及备路径 |
| `save_dir` | str | exp 目录 | exp 目录 | checkpoint 与 history 输出目录 |
| `model_type` | str | `"v19_object_joint"` | `"v19_object_joint"` | 训练入口 dispatcher 用 |
| `epochs` | int | 15 | **10** | 训练总轮数 |
| `batch_size` | int | 8 | 8 | mini-batch |
| `lr` | float | 1.5e-4 | **8e-5** | AdamW 初始学习率 |
| `weight_decay` | float | 1e-4 | 1e-4 | L2 正则 |
| `warmup_epochs` | int | 1 | 1 | 学习率 warmup |
| `aux_decay_epochs` | int | 12 | **5** | 辅助损失权重从 `_start` 衰减到 `_final` 的区间 |
| `num_workers` | int | 8 | 8 | DataLoader workers |
| `mixed_precision` | bool | true | true | `torch.cuda.amp` |
| `seed` | int | 42 | 42 | torch + numpy 随机种子 |

---

## 二、Lambda 权重清单

> 训练总损失:`L = Σ λ_i(epoch) * L_i`。每个 lambda 都可以是标量或 `{start, final}` 字典(curriculum 调度)。

### 2.1 三解码头权重

| 字段 | V19 默认 | V20 默认 | 监督目标 |
|---|---|---|---|
| `lambda_center` | 1.0 | 1.0 | 中心高斯热图(BCE+Dice) |
| `lambda_type` | 1.0 | 1.0 | dense 类型分割(Focal CE) |
| `lambda_z` | 1.0 | 1.0 | z 高度 L1 |
| `lambda_atom_aux_start` / `_final` | 0.5 → 0.05 | 0.5 → 0.05 | 辅助原子图,逐步衰减 |

### 2.2 对象级头权重(V19 主创新)

| 字段 | V19 default | V20 default | 含义 |
|---|---|---|---|
| `lambda_type_obj_peak_start` | 0.25 | 0.25 | peak-center 类型头权重起点 |
| `lambda_type_obj_peak_final` | 2.5 | 2.5 | 终点 |
| `lambda_edge_obj_peak_start` | 0.25 | 0.25 | peak-center 边头权重起点 |
| `lambda_edge_obj_peak_final` | 2.5 | 2.5 | 终点 |
| `lambda_type_obj` | 0.25 | 0.25 | gt-center 类型头(辅) |
| `lambda_edge_obj` | 0.25 | 0.25 | gt-center 边头(辅) |

### 2.3 V20 新增预测对象闭环

| 字段 | V20 default | 含义 |
|---|---|---|
| `lambda_type_obj_pred_start` | **0.25** | 预测中心采样的类型头权重起点 |
| `lambda_type_obj_pred_final` | **2.0** | 终点(主权重) |
| `lambda_edge_obj_pred_start` | **0.20** | 预测中心采样的边头权重起点 |
| `lambda_edge_obj_pred_final` | **1.50** | 终点 |
| `lambda_pred_type_consistency_start` | **0.10** | gt-pred 双路一致性正则起点 |
| `lambda_pred_type_consistency_final` | **0.50** | 终点 |
| `lambda_object_count` | **1.0** | 对象计数 CE 头 |
| `lambda_object_count_mae` | **0.15** | 对象计数 MAE 平滑头 |

V19 这些字段缺省 — 等价于 V20 一组关掉 lambda_*_pred、关掉 object_count head。

### 2.4 Type Upper Teacher 蒸馏

| 字段 | V19 default | V20 default | 含义 |
|---|---|---|---|
| `lambda_teacher_type_distill` | 1.0 | **0.5** | KL 蒸馏权重 |
| `consistency_temperature` | 1.5 | 1.5 | softmax 温度 |
| `teacher_checkpoint` | path | path | 上界 teacher 权重路径 |

### 2.5 Focal CE

| 字段 | default | 含义 |
|---|---|---|
| `focal_gamma` | 1.5 | type / edge focal 强度 |
| `focal_alpha` | 0.25 | 不同类的 prior |

---

## 三、Curriculum 调度

V19 主创新 — peak vs gt center curriculum。

| 字段 | V19 default | V20 default | 含义 |
|---|---|---|---|
| `center_curriculum_alpha_start` | 0.0 | 0.0 | 起初 100% 用 GT-center 监督 |
| `center_curriculum_alpha_final` | 1.0 | 1.0 | 终点 100% 用 peak-center 监督 |
| `center_curriculum_warmup_epochs` | 12 | **5** | 在 N epoch 内线性切换 |

`alpha = min(1.0, epoch / warmup_epochs)`,然后:

```
loss_obj = α * loss_peak + (1 - α) * loss_gt
```

---

## 四、Peak 检测参数

只在评估时用,但训练循环也内部调用一次。

| 字段 | default | 含义 |
|---|---|---|
| `peak_threshold` | 0.45 | sigmoid(center_logits) 阈值 |
| `min_distance_px` | 2 | 局部极大值最小间距 |
| `center_search_radius` | 3 | 匹配半径(px) |

---

## 五、数据增强 / 过滤

| 字段 | default | 含义 |
|---|---|---|
| `augment_rotation` | true | 旋转增强(90°/180°/270° + 任意角 with tilt < 30°) |
| `augment_noise_sigma` | 0.01 | AFM 切片高斯噪声 σ(归一化后) |
| `min_corrugation` | 0.05 | AFM 切片最大-最小差阈,过滤"完全平躺"样本 |
| `require_ring` | false | 只保留含环分子 |
| `max_samples` | -1 | -1 表示全量 |
| `img_size` | 128 | AFM 输入分辨率(硬编码,不建议改) |
| `n_layers` | 10 | AFM 层数(硬编码) |

---

## 六、模型架构参数

| 字段 | V19 default | V20 default | 含义 |
|---|---|---|---|
| `embed_dim` | 512 | 512 | ViT 隐藏维度 |
| `depth` | 8 | 8 | Transformer 层数 |
| `num_heads` | 8 | 8 | 注意力头数 |
| `mlp_ratio` | 4.0 | 4.0 | MLP 扩展倍数 |
| `patch` | 16 | 16 | spatial patch 大小 |
| `temporal_patch` | 2 | 2 | temporal patch 大小 |
| `base_ch` | 64 | 64 | 解码器基础通道 |
| `disable_z_for_object_heads` | false | false | 关闭对象头 z 输入(消融用) |

---

## 七、断点续训 / Warm Start

| 字段 | default | 含义 |
|---|---|---|
| `warm_start_checkpoint` | null | 从该 ckpt 加载权重(不加载 optimizer) |
| `warm_start_strict` | false | 是否要求权重完全匹配 |
| `--resume_checkpoint <path>` (CLI) | null | 加载权重 + optimizer + history,从下一 epoch 继续 |

`scripts/launchers/run_v*.sh` 自动检查 `latest_v19_object_joint.pt`,存在则 `--resume_checkpoint`。

---

## 八、消融常用配置组合

```jsonc
// 关闭 peak curriculum(回到 V15 风格)
"center_curriculum_alpha_start": 0.0,
"center_curriculum_alpha_final": 0.0

// 关闭 teacher 蒸馏
"lambda_teacher_type_distill": 0.0

// 关闭对象计数
"lambda_object_count": 0.0,
"lambda_object_count_mae": 0.0

// 关闭 z 输入(消融)
"disable_z_for_object_heads": true,
"lambda_z": 0.0
```

V20 EXP-06 / EXP-07 ablation 报告完整结果见 [`V19_V20实验总索引与总结.md § 五`](V19_V20实验总索引与总结.md#五v20-消融实验总表)。

---

## 九、写自己的 config

最小写法:

```jsonc
{
  "model_type": "v19_object_joint",
  "data_root": "/path/to/K-1",
  "save_dir": "experiments/my_run",
  "epochs": 10,
  "batch_size": 8,
  "lr": 8e-5,
  "warm_start_checkpoint": "experiments/v20_object_joint_medium10/checkpoints/best_v19_object_joint.pt"
}
```

未指定的字段沿用 [`src/train_v19_object_joint.py`](../src/train_v19_object_joint.py) 内 `_default_config()` 的默认值。

---

## 十、相关文档

- 损失公式与代码位置 — [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md)
- 指标与 lambda 对应 — [`METRICS_GLOSSARY.md § 9`](METRICS_GLOSSARY.md#九配置变量与权重影响)
- 历史 lambda 调参轨迹 — [`V19_2_object_joint_plan.md`](V19_2_object_joint_plan.md)
