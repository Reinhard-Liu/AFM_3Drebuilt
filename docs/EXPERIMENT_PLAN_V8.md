# AFM 3D 分子结构重建：V8 改进方案

## 第一部分：V7 问题诊断（带数据）

### 问题 A：Type Match 在 Epoch 5 后停滞，N/O 预测几乎消失

**数据**：
```
V7 预测类型分布 (100 test samples):
  H: 44.4%  C: 49.2%  N: 3.0%  O: 3.0%    ← N+O 仅 6%
V5b 预测类型分布:
  H: 34.8%  C: 44.5%  N: 11.4%  O: 8.0%   ← N+O = 19.5%
GT 真实分布:
  H: ~35%   C: ~45%   N: ~11%   O: ~8%    ← N+O ≈ 19%
```

**原因分析**：

1. **denoiser type_head 从噪声坐标推断类型的根本局限**。type_loss 仅在 t<500 时计算，但 t=400 时坐标仍有大量噪声（SNR≈1.5），C-N 键长差异 0.07Å（归一化 0.006）被噪声淹没。模型学到"不确定时猜 H/C"是 loss 最优策略。

2. **formula_loss 反而有害**。V7 新增的 formula_loss (MSE on element counts) 权重 0.5，分散到 10 个类别后每类仅 0.05 的梯度。远不足以对抗 CE type_loss 中 H/C 80% 样本量带来的主导梯度。V5b 没有 formula_loss 反而有 19.5% 的 N/O。

3. **缺少 AFM 图像的像素级类型信息**。AFM 图像中不同元素产生不同的力曲线形状（van der Waals 半径不同），但当前架构仅用全局 CLS token 做条件向量，丢失了空间像素级信息。

**文献支撑**：
- **UniGEM (ICLR 2025)**：解耦坐标扩散和类型预测。仅在低噪声步（scaffold 形成后）激活类型预测。
- **MiDi (ECML-PKDD 2023)**：联合扩散原子类型和键矩阵，键约束提供比坐标距离更强的类型信号（C 4 键 vs N 3 键 vs O 2 键）。
- **CGAN for AFM (npj Computational Materials, 2023)**：直接从 AFM 图像堆栈识别元素类型，证明多高度 AFM 图像包含元素特异性信息。

### 问题 B：化学键断裂和分子碎片化

**数据**：
```
100 个 test 样本碎片化统计：
  碎片数=1（完整分子）: 78/100
  碎片数=2: 18/100
  碎片数≥3: 4/100
  含孤立原子(nn>3.6Å): 4/100
```

**原因**：训练时只有坐标 MSE 和 type CE，没有显式的键图/连通性监督。扩散采样时原子独立去噪，没有机制确保最终结构是连通分子。

**文献支撑**：
- **MolDiff (ICML 2023)**：原子-键不一致性问题。提出两阶段逆过程：先从部分去噪位置恢复键，再调整原子位置使其与键一致。采样时添加键长引导梯度。
- **ConStruct (NeurIPS 2024)**：将图扩散重构为约束生成问题，在每个逆步中投影到满足连通性约束的空间。

### 问题 C：原子坐标聚集（挤成一团）

**数据**：Sample #716 (可视化)——20 个原子挤在一个极小的空间内，GT 结构是展开的环状分子，但预测结构是一个致密球团。Kabsch=0.923、TypeMatch=0.65 看似很好，但视觉上完全错误。

**原因**：
1. 扩散采样的高噪声步去噪不充分，原子向质心收缩
2. 缺少排斥力约束——原子之间无最小距离保证
3. 评估指标盲区——Kabsch、JS-Div 无法检测致密聚集

**文献支撑**：
- **NucleusDiff (PNAS 2025)**：van der Waals 流形正则化，每个原子周围维护一个 vdW 半径球面，确保原子间最小距离。碰撞率降低 100%。

### 问题 D：评估指标与视觉观感严重不一致

**数据**：Sample #716——Overall=0.824, Kabsch=0.923, Type=0.65, Coulomb=0.80, 1-JS=0.964，但视觉上原子挤成一团、无 N/O 原子、结构与 GT 完全不同。

**原因**：
1. **Kabsch**：匈牙利匹配后对齐 RMSD 低，但匹配把预测的密集原子团和 GT 局部对齐了
2. **Type Match**：H+C 占 GT ~80%，全猜 H/C 就能拿 65%
3. **1-JS Div**：密集原子间距离分布碰巧和 GT 相似
4. **缺少空间展开度/密度检测指标**

### 问题 E：环结构推理从未真正启用

**数据**：
- 评估时 `generate()` 传入了 GT ring_info（Procrustes 投影生效），但这是用了真实答案，不公平
- 真实推理时没有 GT ring_info，环约束完全不生效
- 计划中的"推理时自动环检测"从 V3 到 V7 始终未实现

**文献支撑**：
- **SubgDiff (NeurIPS 2024)**：子图感知扩散，环结构作为一致单元去噪，保持内部几何。
- **GCDM (Nature Comms Chemistry, 2024)**：几何完备扩散模型，传播角度/二面角信息而非仅传播距离。

---

## 第二部分：V8 改进方案

### 设计原则

1. **只改 2 个核心组件**：不重蹈 V6 同时改 7 个的覆辙
2. **先修后加**：先修复 V7 的 formula_loss 有害问题，再加新功能
3. **解决最影响视觉质量的问题**：类型预测和碎片化/聚集

### 改动 1：AFM 空间特征直接辅助类型预测（替代 formula_loss）

**动机**：当前 type_head 仅从 noisy coords + 全局条件 c 预测类型。但 AFM 图像的多高度切片包含元素特异性信息（不同元素的 vdW 半径产生不同的力曲线）。CGAN for AFM (npj Comp Materials 2023) 已证明直接从 AFM 图像预测元素类型的可行性。

**方案**：在 denoiser 内部增加对 AFM patch 特征的 cross-attention（轻量版，不是 V6 那种每层都加）。

```python
class SE3EquivariantDenoiser:
    def __init__(self, ...):
        ...
        # V8: 仅在最后 2 层加 cross-attention to AFM patches
        # 而非 V6 的每层都加（V6 参数翻倍导致欠拟合）
        self.patch_proj = nn.Linear(cond_dim, hidden_dim)
        self.type_cross_attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.type_cross_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_t, t, c_global, c_patches, mask):
        ...
        # 前 4 层：标准 self-attention（和 V5b 相同）
        for i, layer in enumerate(self.layers):
            h = layer(h, attn_mask)

        # 最后 type_head 前：cross-attend to AFM patches
        p = self.patch_proj(c_patches)
        h_norm = self.type_cross_norm(h)
        cross_out, _ = self.type_cross_attn(h_norm, p, p)
        h_for_type = h + 0.1 * cross_out  # 小残差，不影响 coord

        eps_pred = self.coord_head(h)            # 坐标噪声（不受 cross-attn 影响）
        type_logits = self.type_head(h_for_type)  # 类型预测（有 AFM 空间信息辅助）
        ...
```

**关键设计差异 vs V6**：
- V6：每层都加 cross-attention（6 层 × patch_proj），参数从 25M→51M
- V8：仅在 type_head 前加一层 cross-attention，参数增量 < 1M
- V6：cross-attention 影响 coord_head 和 type_head
- V8：cross-attention 只影响 type_head，coord_head 不受干扰

**ViT 修改**：恢复 V6 的 `(c_global, c_patches)` 返回值，但不加 depth_weight（V6 的无效改动）。

**文献依据**：
- CGAN for AFM (npj Comp Materials 2023)：AFM 图像堆栈含元素特异性信息
- AFM Fingerprint (J. Cheminformatics 2024)：从 AFM 图像提取分子指纹，隐式学习元素特征

### 改动 2：采样时添加排斥力引导 + 连通性投影

**动机**：解决化学键断裂、孤立原子和坐标聚集三个问题。不修改训练流程，仅在推理采样时添加物理约束引导。

```python
@torch.no_grad()
def sample(self, c_global, c_patches, n_atoms, ...):
    ...
    for i in range(ddim_steps):
        eps_pred, type_logits = self.denoiser(x_t, t, c_global, c_patches, mask)
        x_0_pred = ...  # 标准 x_0 预测

        # V8: 物理约束引导（仅在 t < 50% 时启用）
        if t_cur < int(self.timesteps * 0.5):
            x_0_pred = self._apply_physics_guidance(x_0_pred, type_logits, mask)

        ...

def _apply_physics_guidance(self, x_0, type_logits, mask):
    """三合一物理约束投影"""
    B, N, _ = x_0.shape

    # (1) 排斥力：防止原子挤成一团
    #     原子间距离不得小于 vdW 半径之和的 0.7 倍
    types = type_logits.argmax(dim=-1)
    vdw_radii = torch.tensor([0.025, 0.064, 0.061, 0.055, 0.053, 0.087,
                               0.089, 0.083, 0.095, 0.111]).to(x_0.device)  # 归一化
    dists = torch.cdist(x_0, x_0)  # (B, N, N)
    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
    eye = torch.eye(N, device=x_0.device).unsqueeze(0)
    pair_mask = pair_mask * (1 - eye)

    # 最小允许距离 = 0.7 * (r_i + r_j)
    r_i = vdw_radii[types].unsqueeze(2)  # (B, N, 1)
    r_j = vdw_radii[types].unsqueeze(1)  # (B, 1, N)
    min_dist = 0.7 * (r_i + r_j)

    # 排斥力方向
    diff = x_0.unsqueeze(2) - x_0.unsqueeze(1)  # (B, N, N, 3)
    violations = F.relu(min_dist - dists) * pair_mask  # 距离不足的部分
    # 沿连线方向推开
    push_dir = diff / (dists.unsqueeze(-1) + 1e-8)  # 单位方向
    push = (violations.unsqueeze(-1) * push_dir * 0.5).sum(dim=2)  # (B, N, 3)
    x_0 = x_0 + push * mask.unsqueeze(-1)

    # (2) 连通性：拉回孤立原子
    #     对最近邻距离 > 0.22 (2.64Å) 的原子，拉向最近邻
    dists_valid = dists + (1 - pair_mask) * 1e6
    min_nn_dist, min_nn_idx = dists_valid.min(dim=-1)  # (B, N)
    isolated = (min_nn_dist > 0.22) & (mask > 0)
    if isolated.any():
        for b in range(B):
            iso = isolated[b].nonzero(as_tuple=True)[0]
            for a in iso:
                nn = min_nn_idx[b, a]
                direction = x_0[b, nn] - x_0[b, a]
                x_0[b, a] = x_0[b, a] + 0.3 * direction

    # (3) 自动环检测 + 平面投影（不依赖 GT）
    #     检测 5/6 元环并投影到平面
    x_0 = self._auto_detect_and_project_rings(x_0, types, mask)

    return x_0
```

**文献依据**：
- NucleusDiff (PNAS 2025)：vdW 流形正则化，碰撞率降低 100%
- MolDiff (ICML 2023)：键长引导梯度，防止碎片化
- SubgDiff (NeurIPS 2024)：子图（环）感知扩散

### 其他修复

**移除 formula_loss**：V7 实验证明有害（N+O 从 19.5% 降到 6%），直接删除。

**新增评估指标**：
```python
def compute_spatial_spread(pred_coords, gt_coords, mask, n_atoms_pred):
    """检测坐标聚集/挤压：比较预测和 GT 的空间展开度"""
    # 用协方差矩阵特征值比较空间分布
    # 如果预测特征值远小于 GT → 挤成一团
```

---

## 第三部分：实施步骤

### Phase 1：修复 + AFM 类型辅助

| 步骤 | 内容 | 风险 |
|------|------|------|
| 1a | 移除 formula_loss | 零 |
| 1b | ViT 恢复返回 (c_global, c_patches) | 低 |
| 1c | Denoiser type_head 前加单层 cross-attn to patches | 低 |
| 1d | 5 epoch 快速验证：确认 N/O 比例恢复到 V5b 水平 | — |

### Phase 2：采样时物理约束引导

| 步骤 | 内容 | 风险 |
|------|------|------|
| 2a | 实现排斥力引导（vdW 最小距离） | 低 |
| 2b | 实现连通性投影（拉回孤立原子） | 低 |
| 2c | 实现自动环检测 + 平面投影 | 中 |
| 2d | 新增 spatial_spread 评估指标 | 零 |
| 2e | 用 Phase 1 的 checkpoint 直接评估（无需重训） | — |

### Phase 3：完整训练

| 步骤 | 内容 |
|------|------|
| 3a | 50 epoch 完整训练 |
| 3b | 生成可视化 + 全版本对比报告 |

---

## 第四部分：预期效果

| 指标 | V5b | V7 | V8 预期 | 改善来源 |
|------|-----|-----|---------|---------|
| RMSD | 0.269 | 0.262 | 0.25-0.27 | coord_head 不受干扰 |
| Type Match | 48.5% | 54.1% | **58-65%** | AFM patches 辅助 type_head |
| N+O 预测比例 | 19.5% | 6.0% | **15-20%** | 移除 formula_loss + AFM 信息 |
| Bond Validity | 40.4% | 76.6% | **80-88%** | 排斥力 + 连通性约束 |
| 碎片数=1 | — | 78% | **90%+** | 连通性投影 |
| 聚集问题 | — | 存在 | **基本消除** | vdW 排斥力 |
| Formula Sim | — | 0.949 | 0.94-0.96 | 保持 |
| Composite | 0.418 | 0.488 | **0.52-0.56** | 综合提升 |

---

## 第五部分：与历史版本对比

| 维度 | V5b | V6 (失败) | V7 | V8 |
|------|-----|-----------|-----|-----|
| 基线 | V5 | V5b | V5b | **V7（修复后）** |
| Type 信息源 | noisy coords | TypeNet (GT coords) | noisy coords | **noisy coords + AFM patches** |
| ViT 返回值 | 单 c | (c, patches) | 单 c | **(c, patches)** |
| Cross-attn | 无 | 6层全加 (51M) | 无 | **仅 type_head 前 1 层 (~26M)** |
| Formula loss | 无 | 无 | 0.5 (有害) | **移除** |
| 排斥力/连通性 | 无 | connectivity_proj | 无 | **vdW 排斥 + 连通性** |
| 环约束推理 | 未启用 | 未启用 | GT ring_info | **自动检测** |
| 同时改动数 | 2 | **7** | 2 | **2** |

---

## 第六部分：损失函数

```python
# V8 总损失（与 V5b 相同，移除了 V7 的 formula_loss）
# diffusion.py 内部:
loss = coord_loss + 1.0 * type_loss + 0.5 * shape_loss

# train.py 最终:
loss = (
    coord_loss                    # 1.0
    + 1.0 * type_loss             # 标准 CE, sqrt(inv_freq)
    + 1.0 * count_loss            # 原子数
    + 0.01 * retrieval_loss       # 正则化
    + 0.1 * constraint_loss       # Stage 2+
    + 0.5 * shape_loss            # 惯性张量
)
```

**与 V7 的唯一区别**：移除了 formula_loss（V7 中权重 0.5，实验证明有害）。

---

## 第七部分：参考文献

| 论文 | 发表 | 解决的问题 | 引用的关键技术 |
|------|------|-----------|---------------|
| UniGEM | ICLR 2025 | 类型预测精度 | 解耦类型和坐标扩散 |
| MiDi | ECML-PKDD 2023 | 类型+键联合预测 | 离散扩散 on types & bonds |
| MolDiff | ICML 2023 | 碎片化 | 键长引导梯度 |
| NucleusDiff | PNAS 2025 | 原子碰撞 | vdW 流形正则化 |
| CGAN for AFM | npj Comp Mat 2023 | AFM→元素类型 | 多高度 AFM 图像含元素特异性信息 |
| AFM Fingerprint | J. Cheminformatics 2024 | AFM→分子指纹 | ECFP4 from AFM images |
| SubgDiff | NeurIPS 2024 | 环结构保持 | 子图感知扩散 |
| GCDM | Nature CommsChem 2024 | 几何完备性 | 角度/二面角信息传播 |
| ConStruct | NeurIPS 2024 | 连通性约束 | 投影到连通空间 |
| DiffLinker | Nature MI 2024 | 碎片连接 | 条件扩散生成连接原子 |

---

## 第八部分：配置

```json
{
    "data_root": "auto",
    "param_key": "K-1",
    "img_size": 128,
    "num_frames": 10,
    "min_corrugation": 1.25,
    "augment_rotation": true,
    "batch_size": 128,
    "num_workers": 4,
    "prefetch_factor": 2,
    "persistent_workers": true,
    "max_samples": 100000,
    "val_size": 1000,

    "patch_size": 16,
    "temporal_patch_size": 2,
    "embed_dim": 512,
    "encoder_depth": 8,
    "num_heads": 8,
    "drop_rate": 0.1,

    "denoiser_hidden_dim": 256,
    "denoiser_depth": 6,
    "diffusion_steps": 1000,

    "lr": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 50,
    "save_dir": "experiments/v8/checkpoints",
    "log_interval": 1,

    "eval_ddim_steps": 50,
    "eval_samples_per_epoch": 200,
    "eval_full_interval": 5,

    "use_afm_type_cross_attn": true,
    "physics_guidance": true,
    "auto_ring_detection": true,

    "model_type": "diffusion"
}
```
