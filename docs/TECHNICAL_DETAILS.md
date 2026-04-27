# 技术实现细节(Technical Details)

> 本文记录 V19 / V20 主线在源码层面的**全部实现细节**:网络模块、张量形状、损失函数、采样、curriculum、蒸馏、对象计数、后处理、与硬编码常量。所有事实均带 `file:line` 引用,便于定位与复现。
>
> 配套阅读:
> - 设计动机 — [`PRINCIPLES.md`](PRINCIPLES.md)
> - 配置参数 — [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)
> - 指标公式 — [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)
> - 流程框架 — [`PIPELINE_AND_FRAMEWORK.md`](PIPELINE_AND_FRAMEWORK.md)

---

## 一、源码目录速查

```
src/
├── data/                                # 数据管线(单文件实现)
│   ├── __init__.py                      # 空文件,不导出
│   └── dataset.py                       # QUAMAFMDataset + DataLoader 工厂
├── models/                              # 网络模块
│   ├── __init__.py                      # 空文件
│   ├── video_vit.py                     # Video ViT 主干 (PatchEmbedding3D + VideoViTEncoder)
│   ├── v19_joint_model.py               # V19 三分支 UNet + 计数头
│   ├── v19_center_type_head.py          # 中心条件原子类型头
│   ├── v19_center_edge_head.py          # 中心条件边预测头(V20 增量改造)
│   ├── prediction_heads.py              # 计数 / 环 / Site / 检索 等独立预测头
│   ├── postprocess.py                   # RDKit MMFF94 / UFF 弛豫
│   ├── ring_detection.py                # 环识别 + 环系统标签生成
│   ├── upsample_bridge.py               # 多尺度 token bridge
│   ├── v20_ablation_heads.py            # 消融用替代头
│   └── ...
├── utils/
│   ├── metrics.py                       # 6 维评估指标
│   ├── peak_detect.py                   # 局部极值检测
│   └── ...
├── train_v19_object_joint.py            # V19/V20 训练主入口
├── v19_eval_*.py / v20_eval_*.py        # 评估脚本族
└── ...
```

---

## 二、Video ViT 主干

### 2.1 `PatchEmbedding3D` — 时空 Tubelet 切块

`src/models/video_vit.py:15-67`

| 参数 | 默认 | 含义 |
|---|---|---|
| `img_size` | 128 | AFM 切片宽高 |
| `num_frames` | 10 | AFM 高度通道数 |
| `patch_size` | 16 | 空间 patch 大小 |
| `temporal_patch_size` | 2 | 时间 tubelet 大小 |
| `in_channels` | 1 | 单通道(灰度) |
| `embed_dim` | 512 | 输出 token 维度 |

派生量(`__init__` 内计算,`video_vit.py:38-48`):

```
num_spatial_patches  = (128 / 16)² = 64
num_temporal_patches = 10 / 2     = 5
num_patches          = 5 × 64     = 320
```

**Conv3d 配置** (`video_vit.py:43-48`):

```python
self.proj = nn.Conv3d(
    in_channels=1,
    out_channels=512,
    kernel_size=(2, 16, 16),
    stride=(2, 16, 16),
    padding=0,
)
```

**forward 逐步形状** (`video_vit.py:50-67`):

```
x: (B, 10, 128, 128) [灰度堆栈]
→ unsqueeze(1): (B, 1, 10, 128, 128)         # 加 channel 维
→ Conv3d:       (B, 512, 5, 8, 8)             # tubelet 投影
→ flatten(2):   (B, 512, 320)                 # 时空展平
→ transpose:    (B, 320, 512)                 # token-major
```

### 2.2 `VideoViTEncoder` — 时空 ViT 主干

`src/models/video_vit.py:70-155`

| 参数 | 默认 | 备注 |
|---|---|---|
| `embed_dim` | 512 | token 维度 |
| `depth` | 8 | TransformerBlock 层数 |
| `num_heads` | 8 | 注意力头数 |
| `mlp_ratio` | 4.0 | FFN 隐藏维度倍数 |
| `drop_rate` | 0.1 | Attention/FFN dropout |

**初始化细节**:
- `cls_token`: `torch.randn(1, 1, 512) * 0.02` (`video_vit.py:103`)
- `pos_embed`: `torch.randn(1, 321, 512) * 0.02` (`video_vit.py:106-108`)
- token-level dropout `p=0.1` (`video_vit.py:110`)
- `nn.LayerNorm(512)` 收尾 (`video_vit.py:122`)

**forward 流程** (`video_vit.py:125-155`):

```
x: (B, 10, 128, 128)
→ PatchEmbedding3D:           (B, 320, 512)
→ concat([cls, x]):           (B, 321, 512)
→ + pos_embed + dropout:      (B, 321, 512)
→ × 8 TransformerBlock:       (B, 321, 512)
→ LayerNorm:                  (B, 321, 512)
→ split:
  cls_feat:                   (B, 512)        # 全局表征
  patch_feat:                 (B, 320, 512)   # 时空 token
```

### 2.3 `TransformerBlock`

`src/models/video_vit.py:158-190`

```
PreNorm → MultiheadAttention(dropout=0.1)  ── + (skip)
PreNorm → Linear(d → d×4) → GELU → Dropout → Linear(d×4 → d) → Dropout  ── + (skip)
```

激活函数为 `GELU` (`video_vit.py:176`)。无 stochastic depth、无 LayerScale。

---

## 三、V19 主模型 — `V19JointUNet`

`src/models/v19_joint_model.py:14-200`

### 3.1 输入与编码器

| 参数 | 默认 | 来源 |
|---|---|---|
| `in_channels` | 10 | AFM 切片层数 |
| `base_ch` | 64 | UNet 基础通道 |
| `max_objects` | 85 | `MAX_ATOMS` |

编码器 5 层 (`v19_joint_model.py:28-33`):

| 层 | 输入 ch | 输出 ch | 备注 |
|---|---|---|---|
| `enc1` | 10 | 64 | **不**使用 BatchNorm (`normalize=False`) |
| `enc2` | 64 | 128 | Conv2d(4×2, stride=2) + BN + LeakyReLU(0.2) |
| `enc3` | 128 | 256 | 同上 |
| `enc4` | 256 | 512 | 同上 |
| `bottleneck` | 512 | 512 | 最深层 |

### 3.2 三分支解码器

每个分支由 `up + skip-fusion` 块组成,共享相同模板 (`v19_joint_model.py:93-106`):

```
ConvTranspose2d(stride=2) → BN → ReLU            # up
Conv2d(3 ×) → BN → ReLU                          # 与对应 skip 拼接后融合
```

| 分支 | 末层 | 输出形状 | 含义 |
|---|---|---|---|
| Center | `ConvTranspose2d(32→1)` | (B, 1, 128, 128) | 原子中心高斯热图(sigmoid 后) |
| 2D 结构 | `ConvTranspose2d(32→12) + Tanh` | (B, 12, 128, 128) | 10 个类型 map + 2 个键 / 形状 map |
| Z | `ConvTranspose2d(32→1) + Tanh` | (B, 1, 128, 128) | z 高度回归(范围 [-1, 1]) |

### 3.3 全局计数头

`v19_joint_model.py:75-82`

```
AdaptiveAvgPool2d(1) → Flatten
→ Linear(512 → 256) → GELU → LayerNorm
→ Linear(256 → 86)
```

**输出 86 类**:0 ~ 85 个原子,index = 0 表示空分子。

### 3.4 forward 总输出

`v19_joint_model.py:152-163`

`pred` 形状 `(B, 13, 128, 128)`,通道拆分:

| 通道 | 含义 |
|---|---|
| `[0:1]` | atom_map(中心热图) |
| `[1:2]` | bond_map(键密度辅助) |
| `[2:12]` | type_map × 10 类(每类一个 sigmoid map) |
| `[12:13]` | z_map(归一化 z) |

`features` = bottleneck 之后的共享特征图 `(B, 64, 128, 128)`,供条件头采样。

---

## 四、对象级条件头(V19 主创新)

### 4.1 `CenterConditionedTypeHead` — 类型分类

`src/models/v19_center_type_head.py:19-303`

**采样半径(像素)**:
- `afm_radius_px=2.0`(原图采样)
- `feat_radius_px=1.0`(共享特征采样)
- `center_radius_px=2.0`(中心邻域)
- `afm_patch_radius_px=2.0` + `afm_patch_grid_size=5` → 5×5 = 25 个 patch 点

**输入维度** (`v19_center_type_head.py:52`):

```
in_dim = shared_feat × 3 (center+mean+max)            # 64×3 = 192
       + afm × 3        (center+mean+max)             # 10×3 = 30
       + center_stats × 3                             # 3
       + coords × 3      (xyz)                        # 3
       + env_features × 5                             # 5
                                                      # = 233
```

**主干** (`v19_center_type_head.py:53-60`):

```
Linear(233 → 192) → GELU → LayerNorm
→ Linear(192 → 192) → GELU → LayerNorm
```

**三级分类头**:
- 粗 3 类:`Linear(192 → 3)`(类映射:0=C/H,1=N/O/S/P,2=F/Cl/Br/I)
- 异质性二分类:`Linear(192 → 1)`
- Patch 编码:`Linear(250 → 192) → GELU → LayerNorm → Linear(192 → 192)`,250 维 = 10 类 × 5×5 grid
- 精分类:`Linear(196 → 192) → GELU → Linear(192 → 10)`

**损失加权** (`v19_center_type_head.py:258-303`):

```python
loss = fine_loss + 0.35 * coarse_loss + 0.25 * hetero_loss
```

精损失为 **focal-CE**(`v19_center_type_head.py:213`):

```python
ce_focal = (1 - p_t) ** 1.5 * ce        # focal_gamma=1.5
loss = mean(ce_focal * class_weight)    # 带 class_weight,reduction=mean
```

异质性损失 = `BCEWithLogitsLoss`,正样本权重 = `max(neg/pos, 1.0)`,标签平滑 0.02。

**环境特征 5 维** (`v19_center_type_head.py:162-195`):

```
[n_neighbors / 6.0, mean_dist, min_dist, max_dist, var_dist]
```

邻域阈值 0.20(归一化空间),配位数上限假定 6。

### 4.2 `CenterConditionedEdgeHead` — 边预测 + 图精化

`src/models/v19_center_edge_head.py:15-192`

**节点 MLP** (`v19_center_edge_head.py:23-30`):
- 输入维度 82 = `shared(64) + afm(10) + coords(3) + env(5)`
- `Linear(82 → 128) → GELU → LayerNorm → Linear(128 → 128) → GELU`

**边 MLP** (`v19_center_edge_head.py:32-40`):
- 输入维度 263 = `h_i(128) + h_j(128) + delta_xyz(3) + abs_delta_xyz(3) + dist(1)`
- `Linear(263 → 128) → GELU → LayerNorm → Linear(128 → 64) → GELU → Linear(64 → 1)`

**V20 图精化分支** (`v19_center_edge_head.py:42-76`,**全部零初始化** 保证暖启动稳定):

```
msg_mlp:        Linear(263 → 128) → GELU → LayerNorm → Linear(128 → 128)
refine_gate:    Linear(263 → 64)  → GELU → Linear(64 → 1)
node_refine:    Linear(256 → 128) → GELU → LayerNorm → Linear(128 → 128)
refine_edge_mlp: 与基础边 MLP 同结构
```

forward 主流程 (`v19_center_edge_head.py:119-164`):

```
base_logits = edge_mlp(pair_feat)
base_prob   = sigmoid(base_logits)
weighted_msg = msg_mlp(pair_feat) * gate * base_prob * mask
msg_mean = mean(weighted_msg, neighbours)
node_refined = node_embed + node_refine([node_embed, msg_mean])
final_logits = base_logits + refine_edge_mlp([node_refined_i, node_refined_j, deltas])
```

**损失** (`v19_center_edge_head.py:166-192`):
- BCEWithLogits,`pos_weight = max(neg/pos, 1.0)`
- reduction = mean

---

## 五、对象计数头(V20 显式损失)

`src/models/prediction_heads.py:18-108`(类 `AtomCountHead`)

```
共享:  Linear(512 → 256) → GELU → Dropout(0.1)
残差:  Linear(512 → 256)                 # 维度对齐
分类:  Linear(256 → 256) → GELU → Dropout → Linear(256 → 128) → GELU → Linear(128 → 85)
回归:  Linear(256 → 128) → GELU → Dropout → Linear(128 → 1)
```

**推理融合** (`prediction_heads.py:77-83`):

```python
n_pred = 0.7 * (argmax(cls_logits) + 1) + 0.3 * clamp(reg_pred, 1, 85)
```

**损失**:
- 分类:CE with `label_smoothing=0.1`
- 回归:MSE
- 总:`L_count_cls + 1.0 * L_count_mae`(代码 `prediction_heads.py:85-108`)

V19 主模型的全局计数头同样存在(86 类,见 §3.3),但 V20 在 train loss 中**额外加权**:`lambda_object_count=1.0`、`lambda_object_count_mae=0.15`,使全局计数与对象级头形成 **count → object → count 闭环**。

---

## 六、Type Upper Teacher 蒸馏

`src/train_v19_object_joint.py:1149-1424`

**加载阶段** (`train_v19_object_joint.py:1149-1155`):

```python
if config.get("teacher_type_checkpoint"):
    teacher_encoder, teacher_classifier = build_type_teacher(
        teacher_ckpt_path, device
    )
```

teacher 使用 GT 中心作为输入,理论上代表"上界"。

**推断 + 蒸馏损失** (`train_v19_object_joint.py:1377-1424`):

```python
with torch.no_grad():
    _, teacher_patches = teacher_encoder(afm)
    teacher_logits = teacher_classifier(coords_obj, teacher_patches, mask, afm_stack=afm)

T = teacher_temperature                   # 1.5
student = gt_center_logits[valid] / T
teacher = teacher_logits[valid]      / T
distill = (
    F.kl_div(F.log_softmax(student, -1), F.softmax(teacher, -1), reduction="batchmean")
    * (T ** 2)
    * lambda_teacher_type_distill        # V19=1.0, V20=同 1.0
)
```

V20 还预留 `lambda_teacher_type_pred_distill`(默认 0.0),指向 pred-center 路径的二次蒸馏。

---

## 七、Curriculum 与 λ 调度公式

`src/train_v19_object_joint.py:87-95`

```python
def scheduled_weight(epoch, final, start, warmup_epochs):
    if warmup_epochs <= 1:
        return float(final)
    if epoch <= 1:
        return float(start)
    if epoch >= warmup_epochs:
        return float(final)
    alpha = (epoch - 1) / max(warmup_epochs - 1, 1)
    return start + alpha * (final - start)
```

**线性插值**(非 cosine、非阶梯)。所有 `lambda_*_start / final` 均经此函数计算,在 `train_v19_object_joint.py:1278-1292` 实例化。

`center_curriculum_alpha` 同样:从 `alpha_start=0`(纯 GT 中心)线性升到 `alpha_final=1.0`(纯 peak/pred 中心),warmup 长度由 `center_curriculum_warmup_epochs` 控制(V19=12,V20=5)。

---

## 八、训练循环关键实现

`src/train_v19_object_joint.py`

| 项 | 实现 | 行号 |
|---|---|---|
| 优化器 | `AdamW(lr, weight_decay)`(默认 betas=(0.9, 0.999)) | 1157-1163 |
| 调度器 | `CosineAnnealingLR(T_max=epochs, eta_min=min_lr)` | 同上 |
| Mixed precision | **未启用** GradScaler / autocast,FP32 全程 | 1258-1262 |
| Gradient clip | `clip_grad_norm_(params, max_norm=1.0)` | 1481 |
| Best 选择 | 13 维 `(pred_object_score, pred_object_3d_score, -count_mae, ...)` 元组按字典序 | 1532-1563 |
| Resume | `--resume_checkpoint` 严格加载 model + optimizer + scheduler + history | 1174-1190, 1589-1596 |
| Warm start | `warm_start_checkpoint` 非严格加载,仅 model/heads,不动 optimizer | 18, 193-195 |

输出文件:
- `best_v19_object_joint.pt`:仅当当前指标元组 > 历史最高时存档
- `latest_v19_object_joint.pt`:每 epoch 结束都覆盖,用于 resume
- `history_v19_object_joint.json`:逐 epoch 的 train/val 指标列表
- `best_preview.png`:最佳样本快速预览

---

## 九、对象计数 / 双输入 / 一致性(V20 闭环)

V20 在 V19 基础上增加三处闭环:

### 9.1 双输入类型头

V19 仅在 GT 中心 + peak 中心两条路径计算类型损失。V20 补充 **pred 中心** 路径:

```
gt-center logits:    L_type_obj_gt    (λ=1.5)
peak-center logits:  L_type_obj_peak  (λ: 0.25 → 2.5,curriculum)
pred-center logits:  L_type_obj_pred  (λ: 0.25 → 2.0,curriculum,V20 新增)
```

预测中心由 model 在线 peak detection 拿到,匹配半径 `pred_train_match_radius_px=4.0`。

### 9.2 GT-pred 类型一致性(KL)

`config_v20_object_joint_medium10.json`:

```jsonc
"lambda_pred_type_consistency_start": 0.10,
"lambda_pred_type_consistency_final": 0.50,
"consistency_temperature": 1.5
```

实现思路:对相同(可对齐)中心位置,要求 gt-path 与 pred-path 的 logits 经温度软化后 KL 接近(类似自蒸馏)。

### 9.3 计数头硬权重

V20 显式加 `lambda_object_count=1.0` + `lambda_object_count_mae=0.15`,使全局计数与对象级头形成闭环:计数头先估计原子数,采样头按这个数取 top-K peak,再反过来由对象头检验。

---

## 十、后处理 — RDKit MMFF94 / UFF 弛豫

`src/models/postprocess.py:38-235`

### 10.1 关键常量

| 常量 | 值 | 单位 | 含义 |
|---|---|---|---|
| `COORD_SCALE` | **12.0** | Å / unit | 归一化坐标 → 真实 Å |
| `_BOND_TOLERANCE` | **1.3** | — | 候选键判定 = `dist < ideal × 1.3` |
| `max_displacement` | **0.3** | Å | 弛豫单原子位移上限(超出按比例缩回) |
| `max_iters` | **200** | — | MMFF / UFF 最大迭代步数 |

`ATOM_TYPES = ["H","C","N","O","F","S","P","Cl","Br","I"]`(10 类,与 dataset 一致)。

### 10.2 主流程

`coords_to_mol(coords, atom_types, mask)` (`postprocess.py:72-140`):

1. 反归一化:`xyz_ang = coords * 12.0`
2. 过滤有效原子(mask > 0,类型在 0-9 之内)
3. 用 RDKit `RWMol` 顺序加原子
4. 双层循环判定键:`if dist[i,j] < bond_max[t_i, t_j] * 1.3: AddBond(SINGLE)`
5. `Conformer` 写入 3D 坐标

`rdkit_relaxation(coords, atom_types, mask, max_iters=200, max_displacement=0.3)` (`postprocess.py:145-235`):

1. 逐 batch 调用 `coords_to_mol`
2. 优先 MMFF94(`AllChem.MMFFOptimizeMolecule`),失败 fallback UFF
3. 计算 per-atom 位移;若最大位移 > 0.3 Å,**整体按比例缩回**:

```python
scale = max_displacement / max_dist
disp_clipped = disp * scale
```

4. 把弛豫坐标重新归一化(除以 12.0)写回 tensor。

### 10.3 失败模式与降级

- MMFF94 setup 失败(常见原因:孤立原子 / 缺键 / 异常价态)→ **静默** 转 UFF
- UFF 也失败 → 跳过该样本,保留模型原始坐标
- **不影响** 整体指标(评估默认在弛豫前 / 后均算一次,见 `metrics.py`)

---

## 十一、环识别与环系统标签(V17-Bridge 副产物)

`src/models/ring_detection.py`

**核心常量**:
- `MAX_RINGS = 10`,`MAX_RING_SIZE = 6`
- `_BOND_TOLERANCE = 1.2`(注意比 postprocess 的 1.3 略严)
- 9 类环类型:`benzene / pyridine / pyrimidine / furan / thiophene / cyclopentane / cyclohexane / other_5 / other_6`

**主入口** `detect_rings(coords_norm, elements, normalized=True)` (`ring_detection.py:343`):

```
build_molecular_graph  →  find_rings (BFS+DFS, 5/6 元)  →  classify_ring  →  compute_ring_rigid_body
```

输出:`ring_centers (n_rings, 3)`、`ring_normals` (法向量,SVD 最小奇异值方向)、`ring_templates` 等,供数据集 `__getitem__` 与训练时 ring head 使用。

**环系统(V17-Bridge)** `compute_ring_system_scaffold_labels` (`ring_detection.py:603+`):
- `MAX_RING_SYSTEMS=10`,`MAX_SYSTEM_SIDECHAIN_EDGES=128`
- 原子角色 `scaffold_core / attachment_anchor / sidechain` 编码 0/1/2
- 输出 80+ 个标签字段(scaffold_*),用于 V17 旁支 site graph parser

V19/V20 主线**不**强制使用,仅在 `return_v17_bridge_labels=True` 时由 dataset 加载。

---

## 十二、`peak_detect` 与中心采样

`src/utils/peak_detect.py`(局部极值)

主函数语义:对 sigmoid 后的 center heatmap,以 `center_search_radius=3` 像素半径在邻域内取局部最大,按阈值过滤后得到 `(N_peaks, 2)` 像素坐标。

V19/V20 训练时,peak 与 GT 中心的对齐通过 Hungarian(`scipy.optimize.linear_sum_assignment`)在像素空间完成;V20 的 pred-path 进一步用 `pred_train_match_radius_px=4.0` 作为有效配对阈值。

---

## 十三、消融用替代头

`src/models/v20_ablation_heads.py`

| 类 | 用途 | 备注 |
|---|---|---|
| `LegacyGNNTypeHeadAdapter` | 替代 V19 类型头,改用 GNN(SUP-02 graph baseline) | `num_gnn_layers=4`,`bond_threshold=0.20`,`token_grid_size=16` |
| `ZeroEdgeHead` | 永远预测无边,用于消融 | 输出全零 logits |

---

## 十四、关键设计要点速查

| 编号 | 设计 | 实现位置 |
|---|---|---|
| 1 | 时空 Tubelet (2,16,16) | `video_vit.py:43-48` |
| 2 | CLS / pos 初始化 N(0, 0.02²) | `video_vit.py:103-108` |
| 3 | LeakyReLU(0.2) + Conv 4×2 stride=2 编码器 | `v19_joint_model.py:28-33` |
| 4 | enc1 不用 BN | `v19_joint_model.py:29` |
| 5 | 三分支共享 bottleneck,独立 skip | `v19_joint_model.py:35-72` |
| 6 | center 头:`ConvTranspose2d → 1`,sigmoid | `v19_joint_model.py:44` |
| 7 | 2D 头输出 12 通道(10 type + 2 aux) + Tanh | `v19_joint_model.py:56-58` |
| 8 | z 头:`ConvTranspose2d → 1` + Tanh | `v19_joint_model.py:70-72` |
| 9 | 全局计数头 86 类(0~85) | `v19_joint_model.py:75-82` |
| 10 | focal-CE 精分类 (γ=1.5) + label_smoothing 0.02 | `v19_center_type_head.py:213` |
| 11 | 三级分类:粗 3 / 异质 2 / 精 10 | `v19_center_type_head.py:61-74` |
| 12 | Patch grid 5×5 = 25 点,250 维输入 | `v19_center_type_head.py:63` |
| 13 | 边精化分支零初始化暖启动 | `v19_center_edge_head.py:42-76` |
| 14 | 边特征:hi+hj+delta+abs_delta+dist = 263 | `v19_center_edge_head.py:32-40` |
| 15 | KD KL with `T²` 补偿 | `train_v19_object_joint.py:1385-1392` |
| 16 | curriculum 线性插值 | `train_v19_object_joint.py:87-95` |
| 17 | 优化器 AdamW + Cosine LR | `train_v19_object_joint.py:1157-1163` |
| 18 | Grad clip 1.0 | `train_v19_object_joint.py:1481` |
| 19 | 13-key best 元组 | `train_v19_object_joint.py:1532-1563` |
| 20 | 后处理 COORD_SCALE=12.0,位移 cap 0.3 Å | `postprocess.py:38, 231-233` |
| 21 | 候选键判定 1.3 ×ideal | `postprocess.py:42` |
| 22 | 环识别键容差 1.2 ×ideal | `ring_detection.py:38` |
| 23 | 邻域阈值 0.20(归一化) | `v19_center_type_head.py:172` |
| 24 | pred 训练匹配半径 4.0 px(V20) | `config_v20_*.json` |
| 25 | dataset 双层缓存(samples + ring) | `dataset.py:138-150, 218-250` |

---

## 十五、与论文常规做法的差异

1. **不使用 SE(3)-equivariant 网络**。等变性由 XY 旋转增广(训练时)与坐标对齐(评估时)显式获得,代价更小。
2. **dense 网格 + 对象级条件头** 而非端到端 set prediction(DETR 式)。代价更小、更稳,缺点是依赖 peak detection。
3. **fixed coords supervision**(128×128 dense 监督)而非直接回归 (x,y,z)。Hungarian 仅在评估期使用,不进入训练 loss。
4. **focal-CE + KD + curriculum 三件套**:focal 解决类不平衡,KD 把"GT-center 上界"传给"peak-center 学生",curriculum 平滑路径切换。
5. **后处理 RDKit 弛豫带位移上限 0.3 Å**:防止力场把好预测拉坏。论文同类工作通常无此 cap。
6. **图精化零初始化暖启动**:V19→V20 升级期可加载 V19 ckpt 不损精度。

---

## 十六、参考实现入口

| 任务 | 入口脚本 |
|---|---|
| 训练 | `python -m src.train_v19_object_joint --config <config.json>` |
| EXP-01 评估 | `python -m src.v20_eval_fulltest_object --checkpoint <pt> ...` |
| EXP-02 检索 | `python -m src.v20_eval_retrieval_full ...` |
| EXP-03 缝隙 | `python -m src.v20_eval_gap_decompose ...` |
| EXP-04 几何 | `python -m src.v20_eval_geom_diagnostics ...` |
| 复盘 | `python -m src.v19_object_joint_review ...` |
| 索引重建 | `python -m src.tools.generate_v19_v20_experiment_summary` |

详细 CLI 参数见 [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md) 与 [`PIPELINE_AND_FRAMEWORK.md`](PIPELINE_AND_FRAMEWORK.md)。
