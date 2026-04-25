# AFM 3D 分子结构重建：V9 改进方案

## 第一部分：V8 问题诊断（带数据和可视化证据）

### 问题 A：结构压缩——所有版本最严重的共性问题

**现象**：预测的分子结构始终呈"致密球团"形态，而 GT 是展开的平面/链状/环状分子。

**可视化证据**：
- Sample #232：GT 是长条型分子（含蓝色 N 和红色 O），预测挤成一团（全黑白）
- Sample #0：GT 扁平环状，预测是压缩球
- Sample #393：GT 有环结构展开，预测致密聚集

**数据**：30/100 样本 Coulomb=0，这些样本的平均原子数更大（27.6 vs 25.1），说明大分子更容易压缩。但 TypeMatch 反而不低（0.517），证明 Coulomb=0 是坐标压缩导致距离矩阵偏差，而非类型错误。

**根因**：
1. 扩散去噪过程中原子自然向质心收缩——噪声分布是各向同性的高斯，去噪时若无形状引导，原子趋向中心
2. shape_loss（惯性张量特征值 MSE）提供了形状信号但梯度太弱（权重 0.5，且在高噪声步骤权重衰减很大）
3. 全局条件向量 c 不包含显式的形状描述信息（长/宽/高比例等）

**文献**：
- **"Improving Structural Plausibility in Diffusion-Based 3D Molecule Generation via Property-Conditioned Training with Distorted Molecules"** (Digital Discovery, RSC, 2025)：向训练集加入扭曲分子，附带质量标签，条件模型学会区分和生成高质量构象。
- **DiffSMol** (Nature Machine Intelligence, 2025)：形状引导采样，在每个去噪步计算形状梯度并修正。

### 问题 B：Type Match 50% 仍是天花板

**数据**：V8 全局类型分布已恢复正确（N+O=20.2% vs GT 19%），但 TypeMatch 仍为 50.5%。

原因：类型分布正确≠每个原子位置的类型正确。denoiser 在 noisy coords 上预测类型，C/N/O 键长差异仅 0.006 归一化，噪声环境下无法逐原子区分。

**文献**：
- **MiDi** (ECML-PKDD 2023)：联合扩散 atom types + bond matrix。键约束（C=4 键, N=3 键, O=2 键）提供远强于坐标距离的类型信号。
- **MUDiff** (LoG/PMLR 2024)：统一 2D 图结构和 3D 几何的扩散。同时对原子连续特征加连续噪声、对边离散特征加离散噪声。

### 问题 C：环检测推理效果微弱

**数据**：Bond Validity 仅从 V7 的 76.6% 提升到 V8 的 78.6%（+2%）。

**原因**：
1. `_auto_detect_and_project_rings` 的键距阈值 0.18 在压缩坐标上大量误检
2. 投影力度 blend=0.3 太弱
3. 仅 t < 30% 时启用，覆盖步数少

### 问题 D：30% 样本 Coulomb=0

**数据**：Coulomb=0 的样本特征——原子数偏大、Kabsch 很高（0.93+）、TypeMatch 不低。说明匈牙利匹配能找到局部对齐，但全局距离矩阵因坐标压缩而特征值完全偏离。

---

## 第二部分：V9 改进方案

### 核心思路

V8 解决了 N/O 类型比例问题（AFM cross-attention），V9 需要解决**最后也是最严重的问题：结构压缩**。

### 改动 1：Shape-Conditioned Training（形状条件训练）

**动机**：当前模型的全局条件 c 只编码 AFM 图像信息，不包含分子的形状描述符。模型不知道目标分子是扁平的还是球形的、是长条的还是紧凑的。

**方案**：在条件向量中注入分子形状描述符，训练时用 GT 形状，推理时从 AFM 图像预测形状。

```python
# 训练时：
# 1. 从 GT 坐标计算形状描述符
shape_desc = compute_shape_descriptors(gt_coords, mask)  # [asphericity, acylindricity, relative_shape_anisotropy]
# shape_desc 是 3 维向量，编码分子的各向异性程度

# 2. 注入到 denoiser 的全局条件中
shape_emb = self.shape_proj(shape_desc)  # Linear(3 -> hidden_dim)
global_bias = t_emb + c_emb + shape_emb  # 加入到全局偏置
```

```python
def compute_shape_descriptors(coords, mask):
    """计算分子形状描述符（回转张量的三个特征值衍生量）"""
    # 回转张量 S = (1/N) * sum_i (r_i - r_cm)(r_i - r_cm)^T
    # 特征值 lambda_1 >= lambda_2 >= lambda_3
    #
    # asphericity = lambda_1 - 0.5*(lambda_2 + lambda_3)
    #   → 0 = 球形, 大 = 非球形
    # acylindricity = lambda_2 - lambda_3
    #   → 0 = 圆柱形, 大 = 非圆柱形
    # relative_shape_anisotropy = 1 - 3*(l1*l2+l1*l3+l2*l3)/(l1+l2+l3)^2
    #   → 0 = 球形, 1 = 线性
```

**推理时**：用 AFM 图像预测形状描述符：
```python
# 在 ViT encoder 上加一个 shape_head
self.shape_head = nn.Sequential(
    nn.Linear(embed_dim, 64),
    nn.GELU(),
    nn.Linear(64, 3),  # 预测 [asphericity, acylindricity, anisotropy]
)
# 训练时：shape_loss = MSE(shape_head(c_global), gt_shape_desc)
```

**文献依据**：
- "Improving Structural Plausibility" (Digital Discovery 2025)：property-conditioned training 框架
- DiffSMol (Nature MI 2025)：形状引导采样

### 改动 2：形状引导采样（Shape Guidance）

**动机**：即使训练时有形状条件，采样时仍需要显式引导防止收缩。

**方案**：在 DDIM 每步应用形状引导梯度，强制预测结构的惯性张量特征值接近目标值。

```python
def _apply_shape_guidance(self, x_0_pred, target_shape, mask, strength=0.1):
    """在 DDIM 每步引导 x_0_pred 的形状接近 target_shape"""
    B, N, _ = x_0_pred.shape
    x = x_0_pred.clone().requires_grad_(True)

    # 计算当前形状描述符
    current_shape = compute_shape_descriptors(x, mask)

    # 形状损失
    shape_loss = F.mse_loss(current_shape, target_shape)

    # 梯度引导
    grad = torch.autograd.grad(shape_loss, x, create_graph=False)[0]
    x_0_guided = x_0_pred - strength * grad

    return x_0_guided.detach()
```

**注意**：这需要在 `@torch.no_grad()` 的 sample 函数中临时开启梯度计算，仅对 shape guidance 步骤。

### 不改动的部分

- **AFM cross-attention for type_head**：V8 验证有效（N+O 恢复到 20%），保留
- **排斥力引导 + 连通性投影**：V8 验证有效（碎片率 78%→91%），保留
- **损失函数**：与 V8 相同（不加 formula_loss）
- **环检测**：暂时不改（形状引导比环检测更优先）

---

## 第三部分：实施步骤

### Phase 1：Shape-Conditioned Training

| 步骤 | 内容 | 风险 |
|------|------|------|
| 1a | 实现 `compute_shape_descriptors()` | 零 |
| 1b | Denoiser 加入 shape_emb 到 global_bias | 低 |
| 1c | ViT 加 shape_head（预测形状描述符） | 低 |
| 1d | 训练时用 GT 形状条件，推理时用预测形状 | 低 |
| 1e | 3 epoch 快速验证 | — |

### Phase 2：Shape Guidance during Sampling

| 步骤 | 内容 | 风险 |
|------|------|------|
| 2a | 实现 `_apply_shape_guidance()` | 中（需要开梯度） |
| 2b | 在 DDIM 每步 t < 70% 应用形状引导 | 低 |
| 2c | 用 Phase 1 checkpoint 评估效果 | — |

### Phase 3：完整训练

| 步骤 | 内容 |
|------|------|
| 3a | 50 epoch 完整训练 |
| 3b | 可视化 + 全版本对比 |

---

## 第四部分：预期效果

| 指标 | V8 | V9 预期 | 改善来源 |
|------|-----|---------|---------|
| RMSD | 0.254 | 0.24-0.26 | 形状引导减少系统偏差 |
| Kabsch | 0.908 | 0.91-0.93 | 各向异性形状更匹配 |
| Type Match | 50.5% | 52-56% | 形状正确 → 匹配更准 |
| Coulomb | 0.442 | **0.55-0.65** | 距离矩阵不再因压缩偏移 |
| Coulomb=0 比例 | 30% | **<15%** | 消除结构压缩的极端情况 |
| Bond Validity | 78.6% | 80-85% | 展开结构 → 键距更合理 |
| Valence | 46.9% | 48-52% | 同上 |
| Structure Sim | 0.700 | **0.73-0.78** | 综合提升 |

---

## 第五部分：参考文献

| 论文 | 发表 | 解决的问题 |
|------|------|-----------|
| Improving Structural Plausibility | Digital Discovery (RSC) 2025 | 形状条件训练+扭曲分子增强 |
| DiffSMol | Nature MI 2025 | 形状引导采样（61.4% shape-matching） |
| ShapeMol | arXiv 2023 | 形状条件等变扩散 |
| HierDiff | ICML 2023 | 粗到细层次扩散（先形状后细节） |
| MD3MD | Science Advances 2024 | 多尺度图等变扩散 |
| DecompDiff | ICLR 2024 | 分解先验（scaffold+arms） |
| Reducing Atomic Clashes | ICLR 2024 workshop | 采样时近端正则化约束 |
| MiDi | ECML-PKDD 2023 | 联合 type+bond 离散扩散 |
| MUDiff | LoG/PMLR 2024 | 统一离散+连续扩散 |
| GCDM | Nature CommsChem 2024 | 几何完备扩散（角度/二面角） |
| ACS Omega Benchmark | ACS Omega 2025 | 所有模型都有 3D 结构偏差问题 |

**关键发现**：ACS Omega 2025 的综合基准测试表明，结构压缩/球团化是**所有**扩散分子生成模型的共性问题，并非我们独有。这验证了 V9 聚焦于形状引导的方向是正确的。

---

## 第六部分：与历史版本对比

| 维度 | V5b | V7 | V8 | V9 |
|------|-----|-----|-----|-----|
| 核心问题 | 基线 | Type分布 | N/O消失+碎片化 | **结构压缩** |
| 新增组件 | — | formula_loss(有害) | AFM cross-attn + 物理引导 | **形状条件 + 形状引导** |
| 参数增量 | — | ~0 | <1M | **<0.5M** |
| 改模型 | — | 1处 | 1处 | **2处（shape_emb + shape_head）** |
| 改采样 | — | — | 排斥力+连通性+环检测 | **+形状引导** |
