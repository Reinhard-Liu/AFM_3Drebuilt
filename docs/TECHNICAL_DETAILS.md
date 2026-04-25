# 技术详情(Technical Details)

> 本文件是**实现层**的参考手册:每个组件的张量形状、超参、损失公式、推理路径。问题动机请见 [`PRINCIPLES.md`](PRINCIPLES.md);完整参数清单请见 [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)。

---

## 一、Video Vision Transformer 编码器

**源文件** — [`src/models/video_vit.py`](../src/models/video_vit.py)

### 1.1 PatchEmbedding3D

```python
self.proj = nn.Conv3d(
    in_channels = 1,
    out_channels = embed_dim,        # 默认 512(可配置 64~512)
    kernel_size = (temporal_patch, patch, patch),   # (2, 16, 16)
    stride = (temporal_patch, patch, patch),
)
```

- 输入 `(B, 10, 128, 128)` → 加通道维 `(B, 1, 10, 128, 128)`
- 时空 patchify:沿 z 步长 2,沿 (x, y) 步长 16
- 输出 token 数:`(10/2) × (128/16) × (128/16) = 5 × 8 × 8 = 320`
- 加 1 个 `cls_token`(`nn.Parameter(randn(1, 1, embed_dim) * 0.02)`)→ 共 **321 tokens**

### 1.2 时空联合自注意力

每个 Transformer block(共 `depth=8` 层)做:

```
spatial-attn   ─►  per-frame self-attn(每个时间步内的 spatial tokens)
temporal-attn  ─►  per-position self-attn(每个 spatial 位置的 temporal tokens)
MLP            ─►  expansion × 4
```

`einops.rearrange` 在两种维度间切换。详见 `video_vit.py:62-118`。

### 1.3 输出与解码桥接

ViT 输出 token 序列 → reshape 回 `(B, embed_dim, 5, 8, 8)` → 上采样桥接(`upsample_bridge` 模块)到 **共享特征图 `(B, base_ch=64, 128, 128)`**,供三个 UNet 解码头共享。

---

## 二、三解码头(Center / Type / Z)

**源文件** — [`src/models/v19_joint_model.py`](../src/models/v19_joint_model.py)

### 2.1 Center Head

```python
out.center_logits  # (B, 1, 128, 128)
```

监督 — 高斯热图(σ=2 像素,峰值 1.0),BCE-with-logits + Dice 混合损失。

### 2.2 Type Head

```python
out.type_logits    # (B, 11, 128, 128)
```

11 类 = 10 元素(C/H/N/O/F/P/S/Cl/Br/I)+ 1 背景。

监督 — 在每个 GT 原子位置 5×5 邻域填 one-hot;背景位置填类 0;Focal CE,`focal_gamma=1.5`。

### 2.3 Z Head

```python
out.z_pred         # (B, 1, 128, 128)
```

监督 — 仅在 `center_map > 0.5` 的位置计算 L1 损失。归一化 z = 实际 z / `Z_NORM_RANGE`(默认 12 Å)。

---

## 三、对象级条件头(V19 主创新)

**源文件** — [`src/models/v19_center_type_head.py`](../src/models/v19_center_type_head.py)、[`src/models/v19_center_edge_head.py`](../src/models/v19_center_edge_head.py)

### 3.1 输入

- 共享特征图 `(B, 64, 128, 128)`
- 一组中心坐标 — 训练:`gt_centers` 或 `peak_centers`(从 `center_logits.sigmoid()` 取阈值化峰);评估:始终 `peak_centers`

### 3.2 处理

每个中心点 (x, y) 取 16×16 局部 crop 特征 → 通过 MLP/ConvBlock → 输出对象级 logits:

- **类型分类** — `(N_objects, 11)` per sample
- **边连接** — `(N_objects, N_objects)`(对称矩阵)

### 3.3 训练时双前向

```python
gt_out   = head(features, gt_centers)
peak_out = head(features, peak_centers)

loss = α * peak_loss + (1 - α) * gt_loss
# α = curriculum_alpha(epoch)
```

`curriculum_alpha(epoch)` 由 `center_curriculum_alpha_start` (默认 0.0) → `final` (1.0) 在 `warmup_epochs`(V19 默认 12,V20 默认 5)内线性增长,之后保持 1.0。

代码 — [`src/train_v19_object_joint.py`](../src/train_v19_object_joint.py) 训练循环 lines 800–950。

---

## 四、Type Upper Teacher 蒸馏

**Teacher 训练** — [`src/train_v19_type_upper.py`](../src/train_v19_type_upper.py)

- Backbone:同主模型 Video ViT
- Heads:**只接 GT-center**,跳过中心检测分支
- 输出:每个 GT 原子位置的 11 类 logits

**Student 蒸馏 loss**:

```python
T = consistency_temperature  # 默认 1.5
log_p_s = log_softmax(student_logits / T, dim=-1)
p_t     = softmax(teacher_logits / T, dim=-1)
distill_loss = KLDiv(log_p_s, p_t) * T * T
```

权重 `lambda_teacher_type_distill=1.0`(V19) / `0.5`(V20)。

---

## 五、Peak 检测(推理路径关键步)

**源文件** — [`src/utils/peak_detect.py`](../src/utils/peak_detect.py)

### 5.1 算法

1. `center_map = sigmoid(center_logits)` → `(B, 1, 128, 128)`
2. 局部最大:scipy `maximum_filter` with `min_distance_px`(默认 2)
3. 阈值过滤:`peak_value > peak_threshold`(默认 0.45)
4. 取 top-N(由 `object_count_head` 给出预测 N)— V20 新引入

### 5.2 关键超参

| 参数 | 默认 | 影响 |
|---|---|---|
| `peak_threshold` | 0.45 | 太高会漏检;太低误报多 |
| `min_distance_px` | 2 | 太小会重复;太大会合并相邻原子 |
| `object_count_top_n` | 由计数头给 | V20 起替代固定 top-N |

---

## 六、对象计数头(V20 新增)

**源文件** — [`src/models/v19_object_count_head.py`](../src/models/v19_object_count_head.py)

### 6.1 架构

- 输入:Video ViT 的 cls_token feature 或全局池化 feature `(B, 512)`
- 输出 1:CE logits `(B, 80)` — 1 到 80 个原子的 80 类分类
- 输出 2:MAE 标量 `(B,)` — 直接回归原子数量

### 6.2 损失

```python
ce_loss   = CrossEntropy(count_logits, gt_n - 1)        # 1-indexed → 0-indexed
mae_loss  = L1(count_pred_scalar, gt_n.float())
total = lambda_object_count * ce_loss + lambda_object_count_mae * mae_loss
```

V20 默认 `lambda_object_count=1.0`,`lambda_object_count_mae=0.15`。

### 6.3 推理

argmax CE → 整数 N → peak detection 取 top-N。CE 与 MAE 不一致时以 CE 为主、MAE 做平滑监督。

---

## 七、双输入类型头(V20 收尾)

V20 让对象级类型头同时消费两组中心:

| 输入 | 损失项 | 默认权重 |
|---|---|---|
| `peak_centers`(老路径) | `lambda_type_obj_peak` | start 0.25 → final 2.5 |
| `pred_centers` 来自 GT 类型图 sample | `lambda_type_obj` | 0.25 |
| `pred_centers` 来自预测类型图 sample | `lambda_type_obj_pred` | 2.0 |
| 两路一致性 | `lambda_pred_type_consistency` | 0.50 |

**意义** — 评估时使用的"采样自预测类型图"那一路在训练阶段也参与梯度,显著提升 deployment-time 表现(EXP-01 vs V19 baseline 在杂原子 F1 上 +5pp)。

---

## 八、损失总和

完整 loss 是上述所有项的加权和。每一项都有 `_start` / `_final` curriculum 调度,在 `aux_decay_epochs` 内线性变化:

```
total_loss = Σᵢ λᵢ(epoch) * lossᵢ
```

详见 [`CONFIG_REFERENCE.md § 2`](CONFIG_REFERENCE.md#二lambda-权重清单)。

---

## 九、推理后处理(可选)

**源文件** — [`src/models/postprocess.py`](../src/models/postprocess.py)

### 9.1 RDKit MMFF94 / UFF 弛豫

1. 从预测原子坐标 + 类型构造 `RDKit Mol`
2. 用预测的 `pred_object_edges` 推断键
3. 调用 `MMFFOptimizeMolecule` 或 `UFFOptimizeMolecule`
4. **位移 cap**:每个原子 ≤0.3 Å(防止"修过头"破坏结构)

### 9.2 失败保护

- 无 RDKit:`RDKIT_AVAILABLE=False`,跳过
- MMFF 失败:fallback 到 UFF
- 都失败:返回原始预测

`COORD_SCALE=12.0`(归一化常数),处理时除/乘回真实 Å 单位。

---

## 十、硬编码常数索引

| 常数 | 位置 | 数值 | 含义 |
|---|---|---|---|
| `IMG_SIZE` | dataset.py | 128 | AFM 输入分辨率 |
| `N_LAYERS` | dataset.py | 10 | AFM 切片层数 |
| `Z_NORM_RANGE` | metrics.py | 12.0 Å | z 归一化范围 |
| `COORD_SCALE` | postprocess.py | 12.0 | 反归一化缩放 |
| `MATCH_RADIUS_PX` | metrics.py | 3 | 严格 vs 稳健边匹配半径 |
| `GAUSSIAN_SIGMA` | dataset.py | 2.0 px | 中心图高斯展宽 |

修改其中任意一个**都需要同步下游代码**,不推荐随意改动。

---

## 十一、扩展阅读

- 整体架构图 — [`PIPELINE_AND_FRAMEWORK.md`](PIPELINE_AND_FRAMEWORK.md)
- 完整参数 — [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)
- 评估指标 — [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)
- 历史路线复盘 — [`V19_2_object_joint_plan.md`](V19_2_object_joint_plan.md)、[`V20_pred_object_closure_plan.md`](V20_pred_object_closure_plan.md)
