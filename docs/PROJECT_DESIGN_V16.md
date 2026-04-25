# V16: AFM 环检测 + 模板放置

> 基于 V15（RMSD=0.108, coord cross-attention, 无旋转增强）+ 环几何改进

## 问题

V15 可视化发现：模型将原子放在了正确的环位置（RMSD=0.108），但**15/15 个样本都没有形成规则的六/五边形环**——原子聚成团而非排成环。根因是 per-atom MSE loss 不编码"6 个原子等距排成正六边形"这个拓扑约束。

## 解决方案

从 AFM 图像直接检测环 → 用标准化学模板放置 → 扩散模型只负责精调非环原子。

## 架构改动

### 1. RingDetectionHead（新增模块，prediction_heads.py，~140 行）

从 c_global + c_patches 预测环信息，三分支架构：

| 分支 | 输入 | 输出 | 损失 |
|------|------|------|------|
| 环数量 | c_global → shared MLP + 残差 | (B, 11) 分类 + (B,) 回归 | CE + MSE |
| 环中心 | 10 learned queries cross-attend c_patches（temporal mean-pool → 64 spatial） | (B, 10, 2) XY 回归（Z=0） | MSE + 匈牙利匹配 |
| 环类型 | 复用 query 特征 → MLP | (B, 10, 9) 分类 | CE（匹配后） |

- 参数量：795K（占模型总量 1.6%）
- 匈牙利匹配解决环槽位无序性（scipy.linear_sum_assignment）
- 中心 loss 权重 5.0（最重要输出），类型 loss 权重 1.0

### 2. 模板 snap（diffusion.py 新增方法，~60 行）

`_snap_to_ring_templates(x_0_pred, predicted_rings, mask, blend)`:
- 对每个有效预测环：从 `RING_TEMPLATES` 查标准模板（正六/五边形）
- 平移到预测环中心位置
- 匈牙利匹配找 x_0_pred 中最近原子（距离 < 0.25 才 snap）
- blend 向模板位置：`new = blend * template + (1-blend) * current`
- 已分配原子标记为不可再分配（防止环间冲突）

在 DDIM 采样中 t < 70%T 时激活，blend 从 0 渐增到 0.9。保留 auto_detect 作为 fallback。

### 3. 训练集成（train.py，~30 行改动）

- `AFM3DReconModel.__init__`：添加 `self.ring_head = RingDetectionHead(embed_dim)`
- `forward()`：计算 ring_det_loss，权重 0.5 加入总 loss
- `generate()`：用 `ring_head.predict()` 替代 GT ring_info，传给 `sample(predicted_rings=...)`
- `train_epoch()`：totals dict 添加 ring_det_loss 键
- `main()`：warm-start 逻辑（`strict=False` 加载 V15 checkpoint）

### 4. 训练监控指标（train.py 打印改进）

每 epoch 打印新增：
- **ring_det_loss**：训练时环检测损失（确认 ring_head 在收敛）
- **constraint_loss**：Stage 2+ 打印（确认物理约束启用）
- **Ring Preservation**：eval 时环保持分数（确认环几何是否改善）

## 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 30 | |
| warm_start | experiments/v15/checkpoints/best_diffusion.pt | V15 Ep39 |
| save_dir | experiments/v16/checkpoints | |
| batch_size | 128 | |
| lr | 1e-4 → 1e-6 | CosineAnnealing |
| augment_rotation | false | V15 已去除 |

### 三阶段训练（30 epoch 适配）

| Stage | Epoch | 激活的组件 | 新增组件 |
|-------|-------|----------|---------|
| 1 基础 | 1-12 | coord + type + shape + ring_bond + bottom_z + count + retrieval + **ring_det** | ring_det_loss 全程活跃 |
| 2 约束 | 13-22 | + **键长/键角/平面性约束** | constraint_loss 从零变非零 |
| 3 底部 | 23-30 | + **底部原子 3× Z 权重** | z_depth_weighting 启用 |

## 保留的 V15 改进
全部保留：coord cross-attention、无旋转增强、EDM*+γ、type_adapter、bottom_z_loss、SNR 软加权、混淆矩阵、Composite 修复

## 全流程执行

```bash
# run_v16.sh 自动执行：
1. 训练 30 epoch（warm-start from V15）
2. Phase 1 Eval（DDIM-50, 200 samples）
3. 可视化（15 samples）
```

预计总耗时 ~16.5h（训练 ~16h + eval+vis ~30min）

## 关键复用

| 已有组件 | 文件 | 复用方式 |
|---------|------|---------|
| `RING_TEMPLATES` | ring_detection.py | _snap_to_ring_templates 查模板坐标 |
| `RING_TYPE_TO_IDX` | ring_detection.py | 类型索引映射 |
| ring_centers/types/valid/n_rings | dataset.py batch | GT 标签（已在数据加载中） |
| `_project_ring_constraints()` | diffusion.py | Kabsch 对齐 fallback |
| AtomCountHead 模式 | prediction_heads.py | RingDetectionHead 架构参考 |

## 预期效果
- Ring Preserve > 0.95（环几何由模板保证）
- Bond Valid > 0.85（环内键长精确）
- RMSD 保持 ~0.11（坐标能力来自 V15 warm-start）
- 可视化中应能看到**清晰的正六/五边形环结构**（V15 的核心短板）

## 验证清单

| # | 检查项 | 状态 |
|---|--------|------|
| 1 | py_compile 所有修改文件 | ✅ |
| 2 | RingDetectionHead forward 形状正确 | ✅ |
| 3 | 梯度流过 ring_queries | ✅ |
| 4 | ring_det_loss 非零 | ✅ |
| 5 | predict() 输出格式正确 | ✅ |
| 6 | _snap_to_ring_templates 修改坐标 | ✅ |
| 7 | DDIM 采样 with predicted_rings | ✅ |
| 8 | Config 正确 | ✅ |
| 9 | Stage 阈值适配 30 epoch | ✅ |
| 10 | ring_det_loss 打印到训练日志 | ✅ |
| 11 | Ring Preservation 打印到 eval 日志 | ✅ |
| 12 | constraint_loss 打印到训练日志 | ✅ |

## 修改文件清单

| 文件 | 改动类型 |
|------|---------|
| `src/models/prediction_heads.py` | 新增 RingDetectionHead 类 |
| `src/models/diffusion.py` | 新增 _snap_to_ring_templates + 修改 sample() |
| `src/train.py` | 集成 ring_head + warm-start + 打印 ring_det/constraint/ring_pres |
| `config.json` | epochs=30, save_dir=v16, warm_start 字段 |
| `run_v16.sh` | 训练+eval+vis 脚本 |
| `docs/PROJECT_DESIGN_V16.md` | 本文档 |
