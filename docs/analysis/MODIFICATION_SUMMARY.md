# 训练配置修改说明

## 修改时间
2026-03-12

## 修改目的
确保模型能够完成至少60轮训练，充分利用三阶段训练策略，包括物理约束和底部原子优化。

## 修改内容

### 1. 训练轮次调整 (`config.json`)
```diff
- "epochs": 50
+ "epochs": 60
```

### 2. 早停机制修改 (`src/train.py`)

**修改前：**
- 只要验证集 RMSD < 1.0 就立即停止训练
- 导致在第27轮就早停，未能进入Stage 2和Stage 3

**修改后：**
- 只有在完成至少60轮训练后，才允许早停
- 确保三个训练阶段都能完整执行

```python
# 旧代码（第646-649行）
if config["model_type"] == "diffusion" and rmsd_mean < 1.0:
    print(f"[Early Stop] RMSD {rmsd_mean:.4f} < 1.0, stopping training.")
    early_stop = True
    break

# 新代码
if config["model_type"] == "diffusion" and epoch >= 60 and rmsd_mean < 1.0:
    print(f"[Early Stop] RMSD {rmsd_mean:.4f} < 1.0 after epoch {epoch}, stopping training.")
    early_stop = True
    break
```

### 3. 训练阶段划分调整 (`src/train.py`)

**修改前：**
- Stage 1: 1-30轮
- Stage 2: 31-70轮
- Stage 3: 71+轮

**修改后：**
- Stage 1: 1-30轮（基础训练）
- Stage 2: 31-45轮（约束训练）
- Stage 3: 46-60轮（底部聚焦）

```python
def get_training_stage(epoch: int) -> int:
    """Determine training stage from epoch number.

    Stage 1 (epochs 1-30): base training
    Stage 2 (epochs 31-45): constraint training
    Stage 3 (epochs 46-60): bottom atom focus
    """
    if epoch <= 30:
        return 1
    elif epoch <= 45:
        return 2
    else:
        return 3
```

### 4. 文档更新

同步更新了以下文档：
- `/root/autodl-tmp/CLAUDE.md` - 训练策略表格
- `/root/autodl-tmp/项目改进方案.md` - 三阶段训练策略表格

## 三阶段训练策略详解

| 阶段 | 轮次 | 损失组成 | 启用功能 | 目标 |
|------|------|---------|---------|------|
| **Stage 1** | 1-30 | coord + type + count + retrieval | 基础预测 | 学习基本的结构重建能力 |
| **Stage 2** | 31-45 | Stage 1 + 物理约束 | 键长/键角/平面性约束 + 环一致性 | 提升物理有效性 |
| **Stage 3** | 46-60 | Stage 2 + 底部加权 | Z轴深度感知加权（底部原子3x） | 提升遮挡区域精度 |

## 预期效果

### 修改前问题：
1. ✗ 训练在27轮早停，仅完成Stage 1
2. ✗ 物理约束（问题2、6）未启用
3. ✗ 环结构约束（问题5）未启用
4. ✗ 底部原子优化（问题7）未启用

### 修改后预期：
1. ✓ 保证完成60轮完整训练
2. ✓ Stage 2 启用物理约束（键长、键角、平面性）
3. ✓ Stage 2 启用环结构一致性约束
4. ✓ Stage 3 启用底部原子3x权重优化
5. ✓ 解决改进方案中的问题2、5、6、7

## 验证结果

### 训练阶段划分验证：
```
Epoch   1 -> Stage 1  ✓
Epoch  30 -> Stage 1  ✓
Epoch  31 -> Stage 2  ✓
Epoch  45 -> Stage 2  ✓
Epoch  46 -> Stage 3  ✓
Epoch  60 -> Stage 3  ✓
```

### 配置文件验证：
```
训练轮次: 60  ✓
模型类型: diffusion  ✓
```

### 模块导入验证：
```
✓ src.data.dataset
✓ src.models.video_vit
✓ src.models.diffusion
✓ src.models.prediction_heads
✓ src.models.constraints
✓ src.models.ring_detection
✓ src.train
```

## 使用方法

重新开始训练：
```bash
cd /root/autodl-tmp/micro
bash run.sh
```

训练将自动：
1. 执行60轮完整训练（除非在60轮后RMSD < 1.0才早停）
2. 在31-45轮启用物理约束
3. 在46-60轮启用底部原子优化
4. 生成所有输出文件（checkpoints、predictions、visualizations）

## 注意事项

1. **早停触发条件**：现在只有在第60轮之后，如果RMSD < 1.0才会触发早停
2. **训练时长**：完整60轮训练预计需要更长时间（约为之前的2-3倍）
3. **显存使用**：Stage 2和Stage 3启用更多约束，可能需要更多显存
4. **检查点保存**：最佳模型仍然在验证损失最低时保存

## 相关文件

- `config.json` - 训练配置
- `src/train.py` - 训练主循环和阶段划分
- `CLAUDE.md` - 项目文档
- `项目改进方案.md` - 改进方案说明
