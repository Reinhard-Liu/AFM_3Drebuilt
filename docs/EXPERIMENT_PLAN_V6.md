# AFM 3D 分子结构重建：V6 架构升级方案

---

## 第一部分：V5b 问题总结与V6目标

### V5b 剩余问题

| 问题 | 现状 | 根因 |
|------|------|------|
| Coulomb ≈ 0 | 0.009 | Type错误(48.5%)导致Z错误 + 坐标聚团导致r偏小 + Count不匹配导致零填充 |
| Type Match低 | 48.5% | 仅靠坐标几何区分C/N/O几乎不可能（键长差异0.006仅为RMSD的2%） |
| 底部原子 | Recall=3.9% | AFM只看到表面，模型缺少推断遮挡区域的能力 |
| 分子碎片化 | Sample#333: 6个碎片 | 扩散模型回归均值，XY压缩+Z拉伸 |
| 分子聚团 | XY范围压缩50% | 缺少原子间距离约束 |

### V6 目标

| 指标 | V5b | V6目标 |
|------|-----|--------|
| Type Match | 48.5% | **70-80%** |
| Coulomb | 0.009 | **> 0.15** |
| Bottom Recall | 3.9% | **> 15%** |
| Bond Valid | 40.4% | **> 60%** |
| RMSD | 0.269 | **< 0.22** |
| Composite | 0.418 | **> 0.55** |

---

## 第二部分：Type Match 达到 70-80% 的方案

### 2.1 为什么仅靠坐标不够

```
C-N键长差异: 0.006 (归一化空间)
V5b RMSD:    0.269
比值:        RMSD / 键长差异 = 45倍

→ 坐标误差是C/N键长差异的45倍
→ 纯靠坐标几何区分C和N几乎不可能
```

### 2.2 TypeNet：坐标 + AFM条件 + 配位环境

**核心思想**：类型预测不仅依赖坐标几何，还要利用AFM图像中的元素对比度差异和化学配位规则。

```python
class TypeNet(nn.Module):
    """从3D坐标 + AFM条件 + 配位环境推断原子类型

    三路输入融合:
    1. 坐标几何: 原子间距离、角度 → 区分H(1键) vs 重原子(多键)
    2. AFM条件:  不同元素的AFM响应不同 → 区分C/N/O/S
    3. 配位环境: 邻居数量和距离分布 → C(4配位) vs N(3) vs O(2) vs H(1)
    """

    def __init__(self, coord_dim=3, cond_dim=512, hidden_dim=256,
                 num_types=10, num_layers=6):
        super().__init__()

        # 坐标特征: 原子对距离 + 角度特征
        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 配位环境特征: 每个原子的邻居统计
        # 输入: [邻居数, 平均距离, 最近距离, 最远距离, 距离方差] = 5维
        self.env_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # AFM条件注入: 多尺度交叉注意力
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)

        # 融合 + Transformer
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim*4,
                                       dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])

        # 分类头
        self.type_head = nn.Linear(hidden_dim, num_types)

    def compute_coordination_features(self, coords, mask):
        """计算每个原子的配位环境特征"""
        B, N, _ = coords.shape
        # 原子对距离矩阵
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B,N,N,3)
        dist = diff.norm(dim=-1)  # (B,N,N)

        # 用mask屏蔽无效原子
        large = 1e6
        dist = dist + (1 - mask.unsqueeze(1)) * large
        dist = dist + (1 - mask.unsqueeze(2)) * large
        dist.diagonal(dim1=1, dim2=2).fill_(large)

        # 键距阈值内的邻居
        bond_thresh = 0.15  # 归一化空间约1.8Å
        is_neighbor = (dist < bond_thresh).float()

        n_neighbors = is_neighbor.sum(dim=-1)                           # (B,N)
        # 邻居距离统计（仅计算有邻居的）
        neighbor_dist = dist * is_neighbor + (1-is_neighbor) * large
        mean_dist = (dist * is_neighbor).sum(-1) / n_neighbors.clamp(min=1)
        min_dist = neighbor_dist.min(dim=-1).values.clamp(max=1.0)
        max_dist_vals = (dist * is_neighbor).max(dim=-1).values
        dist_var = ((dist - mean_dist.unsqueeze(-1))**2 * is_neighbor).sum(-1) / n_neighbors.clamp(min=1)

        # (B, N, 5)
        env = torch.stack([n_neighbors, mean_dist, min_dist, max_dist_vals, dist_var], dim=-1)
        return env

    def forward(self, coords, c_global, c_patches, mask):
        """
        Args:
            coords: (B, N, 3) 去噪后的坐标
            c_global: (B, cond_dim) 全局条件向量
            c_patches: (B, P, cond_dim) patch-level条件
            mask: (B, N) 原子mask
        Returns:
            type_logits: (B, N, num_types)
        """
        # 1. 坐标特征
        h_coord = self.coord_encoder(coords)  # (B, N, hidden)

        # 2. 配位环境特征
        env = self.compute_coordination_features(coords, mask)
        h_env = self.env_encoder(env)  # (B, N, hidden)

        # 3. AFM条件特征 (交叉注意力)
        c_proj = self.cond_proj(c_patches)  # (B, P, hidden)
        h_afm, _ = self.cross_attn(h_coord, c_proj, c_proj)  # (B, N, hidden)

        # 4. 三路融合
        h = self.fusion(torch.cat([h_coord, h_env, h_afm], dim=-1))

        # 5. Transformer自注意力
        attn_mask = (mask == 0)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=attn_mask)

        # 6. 分类
        return self.type_head(h)
```

### 2.3 化合价一致性损失（GFLoss）

**作用**：在TypeNet训练时，添加化合价一致性约束，迫使类型预测与几何结构一致。

```python
def valence_consistency_loss(type_probs, coords, mask):
    """
    从预测类型概率和坐标推断期望化合价,
    与实际配位数比较

    C→4键, N→3键, O→2键, H→1键, S→2键, F→1键
    如果模型预测某原子为C(4键)但它只有2个邻居 → 惩罚
    """
    # 期望化合价 (按类型概率加权)
    max_valence = torch.tensor([1,4,3,2,1,2,5,1,1,1], device=coords.device).float()
    expected = (type_probs.softmax(-1) * max_valence).sum(-1)  # (B, N)

    # 实际邻居数
    dist = torch.cdist(coords, coords)
    actual = ((dist < 0.15) & (dist > 0.01)).float().sum(-1) * mask

    # MSE损失
    return F.mse_loss(expected * mask, actual)
```

### 2.4 Type Match 70-80% 可行性

| 原子 | 占比 | 区分线索 | 预期准确率 |
|------|------|---------|-----------|
| H | 34.4% | 1配位 + 最短键长 + AFM信号弱 | **90%** |
| C | 42.0% | 4配位 + 中等键长 + 最常见 | **85%** |
| N | 12.1% | 3配位 + 略短键长 + AFM对比度 | **60%** |
| O | 6.7% | 2配位 + 最短非H键 + AFM对比度 | **55%** |
| 其他 | 4.8% | S大原子/卤素等特殊 | **50%** |
| **加权** | | | **79%** |

---

## 第三部分：Coulomb 指标修复

### 3.1 Coulomb为0的三个来源

```
Coulomb = max(0, 1 - L2(eig_pred, eig_gt) / norm(eig_gt))

L2过大的原因:
  (1) Type错误 → Z错误 → C_ij = Z_i*Z_j/r 偏差巨大
  (2) 坐标聚团 → r偏小 → C_ij偏大 → 特征值偏大
  (3) 原子数不匹配 → 特征值维度不同 → 零填充加大L2
```

### 3.2 修复路径

| 来源 | 修复措施 | 对应改进 |
|------|---------|---------|
| Type错误 | TypeNet解耦 + 配位特征 → Type 70-80% | §2.2, §2.3 |
| 坐标聚团 | 距离矩阵损失 → 原子间距准确 | §4.2 |
| 原子数不匹配 | Count MAE改进 → < 0.8 | §5 |

**当Type Match > 70%且Bond Valid > 60%时，Coulomb预计 > 0.15**

---

## 第四部分：结构质量改进

### 4.1 多尺度条件注入（解决信息瓶颈）

**问题**：512维向量编码不了完整的3D分子信息。

**方案**：ViT输出多尺度特征，在去噪器每层注入。

```python
# video_vit.py 修改
class VideoViTEncoder(nn.Module):
    def forward(self, x):
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens)

        c_global = self.norm(tokens[:, 0])     # (B, 512) CLS token
        c_patches = self.norm(tokens[:, 1:])   # (B, 320, 512) patch tokens

        return c_global, c_patches  # 返回两种条件

# diffusion.py 修改
class SE3EquivariantDenoiser(nn.Module):
    def __init__(self, ...):
        # 每层添加交叉注意力
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, x_t, t, c_global, c_patches, mask):
        h = self.coord_embed(x_t) + t_emb + c_emb  # 全局条件

        for layer, cross_attn in zip(self.layers, self.cross_attn_layers):
            h = layer(h, attn_mask)
            # 每层注入AFM空间细节
            c_proj = self.patch_proj(c_patches)
            h_cross, _ = cross_attn(h, c_proj, c_proj)
            h = h + 0.1 * h_cross  # 残差连接，0.1系数防止主导

        return self.coord_head(h)  # 只输出坐标噪声
```

### 4.2 距离矩阵损失（解决聚团和碎片化）

```python
def distance_matrix_loss(pred_coords, gt_coords, mask):
    """约束原子间距离关系，防止聚团和碎片化"""
    # 预测距离矩阵
    D_pred = torch.cdist(pred_coords, pred_coords)  # (B,N,N)
    D_gt = torch.cdist(gt_coords, gt_coords)

    # mask: 只计算valid原子对
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B,N,N)

    # 分两部分:
    # (1) 键距保持: 近邻原子对距离准确（权重高）
    bond_mask = (D_gt < 0.15) & (D_gt > 0.01)
    bond_loss = ((D_pred - D_gt)**2 * bond_mask * pair_mask).sum() / bond_mask.sum().clamp(min=1)

    # (2) 全局距离分布: 所有原子对（权重低）
    global_loss = ((D_pred - D_gt)**2 * pair_mask).sum() / pair_mask.sum().clamp(min=1)

    return 2.0 * bond_loss + 0.5 * global_loss
```

### 4.3 连通性投影（解决碎片化）

```python
def connectivity_projection(coords, mask, bond_threshold=0.15):
    """在DDIM采样后期，将碎片分子投影回连通状态"""
    B, N, _ = coords.shape
    for b in range(B):
        valid = mask[b] > 0
        c = coords[b, valid]
        n = valid.sum().item()
        if n < 2:
            continue

        # 构建邻接图
        dist = torch.cdist(c.unsqueeze(0), c.unsqueeze(0))[0]
        adj = (dist < bond_threshold) & (dist > 0.01)

        # BFS找连通分量
        components = []
        visited = set()
        for start in range(n):
            if start in visited:
                continue
            comp = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                for nb in range(n):
                    if adj[node, nb] and nb not in visited:
                        queue.append(nb)
            components.append(comp)

        if len(components) <= 1:
            continue

        # 找最大分量，将其他分量拉向它
        largest = max(components, key=len)
        largest_center = c[largest].mean(dim=0)

        for comp in components:
            if comp is largest:
                continue
            comp_center = c[comp].mean(dim=0)
            # 计算需要移动的方向和距离
            direction = largest_center - comp_center
            # 移动到使最近原子对距离 = bond_threshold
            min_dist = torch.cdist(c[comp].unsqueeze(0), c[largest].unsqueeze(0))[0].min()
            move = direction * (1.0 - bond_threshold / min_dist.clamp(min=0.01))
            move = move.clamp(-0.3, 0.3)  # 限制最大移动量
            c[comp] += move

        coords[b, valid] = c
    return coords
```

---

## 第五部分：底部原子预测改进

### 5.1 问题分析

```
AFM探针从上方扫描 → 只看到表面原子
底部30%原子被完全遮挡 → Bottom Recall = 3.9%

但AFM深度切片(Z-slice 6-9)包含部分底部信息:
  - Z-slice 6-9 的图像对比度反映底部结构的间接影响
  - 分子整体高度可推断是否有底部原子
```

### 5.2 深度感知注意力

在ViT编码器中，对不同Z-slice赋予不同权重，让深层切片获得更多关注：

```python
# video_vit.py 修改
class VideoViTEncoder(nn.Module):
    def __init__(self, ...):
        # 深度权重: 让模型学习每层切片的重要性
        self.depth_weight = nn.Parameter(torch.ones(num_frames))

    def forward(self, x):
        # x: (B, 10, 128, 128)
        # 对每层切片加权
        w = self.depth_weight.softmax(dim=0)  # (10,)
        x = x * w.view(1, -1, 1, 1)
        ...
```

### 5.3 底部聚焦的Z轴条件

在去噪器中注入Z轴位置信息，帮助模型区分上下原子：

```python
# 在denoiser中添加Z轴位置编码
z_pos = x_t[:, :, 2:3]  # 当前预测的Z坐标
z_embed = self.z_encoder(z_pos)  # 编码Z位置信息
h = h + z_embed  # 注入到原子特征中
```

### 5.4 Stage 3 底部聚焦增强

在训练Stage 3中，对底部原子的坐标损失权重从3x提高到5x：

```python
# Stage 3: 底部原子5x权重
z_weight = 1.0 + 4.0 * (1.0 - z_ratio)  # bottom=5x, top=1x
```

---

## 第六部分：原子数预测改进

### 6.1 Mixture Density Network

替换当前的分类+回归双分支，用混合密度网络直接建模原子数的条件分布：

```python
class AtomCountHead(nn.Module):
    def predict(self, c):
        cls_logits, reg_value = self.forward(c)

        # 分类分支给出概率分布
        cls_probs = F.softmax(cls_logits, dim=-1)

        # 取概率最高的3个候选
        top3_probs, top3_idx = cls_probs.topk(3, dim=-1)
        top3_counts = top3_idx + 1

        # 回归值作为偏移修正
        offset = (reg_value - top3_counts[:, 0].float()).clamp(-2, 2)

        # 最终预测: 分类最大概率 + 回归偏移
        pred = (top3_counts[:, 0].float() + 0.3 * offset).round().long()
        return pred.clamp(1, self.max_count)
```

---

## 第七部分：V6 总损失函数

```python
# 阶段1 (Epoch 1-20): 基础训练
loss = (
    coord_loss                        # 1.0  坐标噪声预测
    + 0.5 * dist_loss                 # 0.5  距离矩阵 (t<300)
    + 1.0 * type_loss                 # 1.0  TypeNet (独立网络)
    + 0.2 * valence_consistency_loss  # 0.2  化合价一致性
    + 1.0 * count_loss                # 1.0  原子数
    + 0.5 * shape_loss                # 0.5  惯性张量
    + 0.01 * retrieval_loss           # 0.01 正则化
)

# 阶段2 (Epoch 21-35): 约束训练
loss += 0.1 * constraint_loss         # 键长/键角

# 阶段3 (Epoch 36-50): 底部聚焦
coord_loss: 底部原子5x权重
```

---

## 第八部分：架构总览

```
V5b架构:
AFM (10,128,128) → ViT → c(512) → Denoiser(c, x_t, t) → eps + types
                                                            ↑ 耦合

V6架构:
AFM (10,128,128) → ViT → c_global(512) + c_patches(320,512)
                              ↓                    ↓
                     Denoiser(每层cross-attn) → eps_pred → x_0_pred
                                                              ↓
                                                    TypeNet(x_0, c_global, c_patches)
                                                              ↓
                                                         type_logits
                              ↓
                     connectivity_projection(x_0_pred)
                              ↓
                         最终输出: coords + types
```

---

## 第九部分：预期效果

### 各改进的预期影响

| 改进 | 影响指标 | 预期提升 | 关键机制 |
|------|---------|---------|---------|
| TypeNet解耦+配位特征+AFM条件 | Type Match | 48%→**75%** | 三路特征融合，不受噪声干扰 |
| 化合价一致性损失 | Type, Valence | Type+5%, Valence+10% | 全局化学约束 |
| 距离矩阵损失 | Bond, RMSD | Bond+20%, RMSD-15% | 直接约束原子间距 |
| 多尺度条件注入 | RMSD, Kabsch | RMSD-10%, Kabsch+3% | 更丰富空间信息 |
| 连通性投影 | Bond, Valence | 碎片率→0 | 后处理保证连通 |
| 底部聚焦增强 | Bottom Recall | 4%→**15-20%** | 5x权重+深度注意力 |
| 综合: Type+Bond提升 | Coulomb | 0.01→**0.15-0.25** | Z准确+r准确 |

### 指标预估

| 指标 | V5b | V6 预估 |
|------|-----|---------|
| **Type Match** | 48.5% | **70-80%** |
| **Coulomb** | 0.009 | **0.15-0.25** |
| **Bottom Recall** | 3.9% | **15-20%** |
| **Bond Valid** | 40.4% | **60-70%** |
| **RMSD** | 0.269 | **0.18-0.22** |
| **Valence** | 35.4% | **50-60%** |
| **Count MAE** | 1.17 | **0.7-0.9** |
| **Kabsch** | 0.900 | **0.93-0.96** |
| **Composite** | 0.418 | **0.55-0.65** |

---

## 第十部分：训练配置

```json
{
    "batch_size": 128,
    "num_workers": 4,
    "epochs": 50,
    "lr": 1e-4,
    "eval_ddim_steps": 50,

    "encoder_depth": 8,
    "embed_dim": 512,
    "denoiser_depth": 6,
    "denoiser_hidden_dim": 256,

    "typenet_depth": 6,
    "typenet_hidden_dim": 256,

    "use_multiscale_cond": true,
    "use_typenet": true,
    "use_dist_loss": true,
    "use_valence_loss": true,
    "use_connectivity_projection": true,

    "dist_loss_weight": 0.5,
    "valence_loss_weight": 0.2,
    "type_loss_weight": 1.0,
    "bottom_weight_multiplier": 5.0
}
```

**预计训练时间**：~8小时（RTX 4080 SUPER）
- TypeNet: +1小时（独立网络）
- 多尺度条件: +0.5小时（交叉注意力）
- 距离矩阵损失: +0.3小时

**参数量**：44M → ~52M (+18%)

---

## 第十一部分：实施优先级

| 优先级 | 改进 | 难度 | 预期效果 | 文件 |
|--------|------|------|---------|------|
| **P0** | TypeNet解耦+配位特征+AFM条件 | 高 | Type 48→75% | 新type_net.py, 改diffusion/train |
| **P0** | 化合价一致性损失 | 低 | Type+5%, Valence+10% | 改diffusion.py |
| **P0** | 距离矩阵损失 | 低 | Bond+20%, 防碎片 | 改diffusion.py |
| **P1** | 多尺度条件注入 | 中 | RMSD-10% | 改video_vit.py, diffusion.py |
| **P1** | 连通性投影 | 低 | 碎片率→0 | 改diffusion.py |
| **P1** | 底部聚焦增强 | 低 | Bottom 4→15% | 改video_vit.py, train.py |

---

## 第十二部分：修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/models/type_net.py` | **新增** TypeNet (坐标+AFM+配位→类型) |
| `src/models/diffusion.py` | denoiser只输出eps; 添加交叉注意力; 距离矩阵损失; 化合价一致性损失; 连通性投影 |
| `src/models/video_vit.py` | 输出c_global + c_patches; 深度权重 |
| `src/train.py` | 集成TypeNet; 解耦训练流程; 新损失项; Stage3底部5x |
| `config.json` | V6配置项 |

---

## 第十三部分：参考文献

| 论文 | 发表 | 采纳的技术 |
|------|------|-----------|
| UniGEM | ICLR 2025 | 类型预测与坐标扩散解耦 |
| EMDS | 2024 | 分离扩散过程 |
| GFMDiff | AAAI 2024 | 化合价一致性损失 (GFLoss) |
| MD3MD | Science Advances 2024 | 多尺度条件注入 |
| GCDM | Nature Comm. Chem. 2024 | 距离矩阵约束 |
| MiDi | ECML 2023 | 图-几何联合生成, 配位特征 |
| ConStruct | NeurIPS 2024 | 连通性约束投影 |
| JODO | 2024 | 关系注意力, 2D-3D联合 |
| GeoLDM | 2023 | 几何潜空间扩散 |
| DiffSMol | Nature MI 2025 | 形状引导采样 |
