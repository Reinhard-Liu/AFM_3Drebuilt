# 项目代码审查报告

**审查时间**: 2026-03-12
**审查范围**: 确保 `bash run.sh` 能完整执行并生成所有预期输出
**审查人**: Claude Code

---

## 执行流程概览

```
bash run.sh
    ↓
[1/3] python3 -m src.train --config config.json
    ↓
    生成输出：
    - checkpoints/best_diffusion.pt (模型权重)
    - checkpoints/epoch_X_diffusion.pt (定期检查点)
    - checkpoints/history_diffusion.json (训练损失历史)
    - checkpoints/metrics_diffusion.json (评估指标历史)
    - checkpoints/predictions_diffusion.json (模型预测)
    - checkpoints/training.log (训练日志)
    ↓
[3/5] 生成训练曲线可视化
    ↓
    生成输出：
    - visualizations/curves_diffusion.png
    ↓
[4/5] 生成分子 3D 可视化
    ↓
    生成输出：
    - visualizations/molecules_diffusion/sample_XXX.png (10个样本)
    ↓
[5/5] 生成总结报告
    ↓
    生成输出：
    - checkpoints/training_summary.txt
    ↓
打印总结到终端
```

---

## ✅ 关键文件审查

### 1. 主训练脚本 (`src/train.py`)

#### 1.1 训练日志 ✅

**代码位置**: `train.py:535`
```python
sys.stdout = Logger(log_file)
```

**输出文件**: `checkpoints/training.log`

**验证**:
- ✅ Logger 类正确实现 (line 37-50)
- ✅ 同时输出到终端和文件
- ✅ 包含所有 print 输出

---

#### 1.2 模型权重保存 ✅

**代码位置**: `train.py:637-644` (best), `train.py:648-654` (periodic)

**输出文件**:
- `checkpoints/best_diffusion.pt` - 验证集最佳模型
- `checkpoints/epoch_{10,20,30,40,50,60}_diffusion.pt` - 每10轮保存

**保存内容**:
```python
{
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "val_loss": val_metrics["loss"],
    "config": config,
}
```

**验证**:
- ✅ 保存条件正确（验证集损失改善时）
- ✅ 包含完整的模型状态和优化器状态
- ✅ 定期检查点每10轮保存

---

#### 1.3 训练损失历史 ✅

**代码位置**: `train.py:693-707`

**输出文件**: `checkpoints/history_diffusion.json`

**内容格式**:
```json
{
  "train": [
    {
      "loss": 浮点数,
      "coord_loss": 浮点数,
      "type_loss": 浮点数,
      "count_loss": 浮点数,
      "retrieval_loss": 浮点数,
      "constraint_loss": 浮点数  // Stage 2+ 有值
    },
    ...
  ],
  "val": [...]
}
```

**验证**:
- ✅ 包含所有 6 个损失组件（修改后）
- ✅ train_epoch() 返回所有损失 (line 237-238)
- ✅ validate() 返回所有损失 (line 282-283)
- ✅ Tensor 转换为 float (line 695-706)

---

#### 1.4 评估指标历史 ✅

**代码位置**: `train.py:687-690`

**输出文件**: `checkpoints/metrics_diffusion.json`

**内容格式**:
```json
[
  {
    "epoch": 整数,
    "rmsd_mean": 浮点数,
    "rmsd_std": 浮点数,
    "bottom_recall_mean": 浮点数,
    "bottom_rmsd_mean": 浮点数,
    "bond_validity_mean": 浮点数,
    "count_exact_match": 浮点数,  // 使用预测值（修改后）
    "count_mae": 浮点数,
    "composite_score": 浮点数
  },
  ...
]
```

**验证**:
- ✅ 包含所有 6 维评估指标
- ✅ evaluate_generation() 使用 use_gt_count=False (line 331)
- ✅ 每个 epoch 追加新记录

---

#### 1.5 模型预测 ✅

**代码位置**: `train.py:395-461`

**输出文件**: `checkpoints/predictions_diffusion.json`

**内容格式**:
```json
{
  "num_samples": 100,
  "fields": [
    "coords (3D atomic coordinates)",
    "atom_types (predicted atom types)",
    "n_atoms_pred (predicted number of atoms)",
    "retrieval_cid_indices (Top-5 candidate molecule CID indices)"
  ],
  "predictions": [
    {
      "sample_id": 整数,
      "coords": [[x,y,z], ...],  // 仅有效原子
      "atom_types": [类型索引, ...],  // 仅有效原子
      "n_atoms_pred": 整数,
      "retrieval_cid_indices": [CID索引, ...],  // Top-5
      "retrieval_scores": [分数, ...]  // 可选
    },
    ...
  ]
}
```

**验证**:
- ✅ save_predictions() 正确实现
- ✅ 包含所有 4 个必需字段
- ✅ 使用 use_gt_count=False (line 416)
- ✅ 保存 100 个样本
- ✅ 只保存有效原子（前 n_atoms_pred 个）

---

### 2. 可视化脚本 (`src/utils/visualize.py`)

#### 2.1 训练曲线图 ✅

**代码位置**: `visualize.py:109-241`

**输出文件**: `visualizations/curves_diffusion.png`

**内容**:
- 2×3 子图布局
- Row 1: Total Loss, Coord Loss, Type Loss
- Row 2: Count Loss, Retrieval Loss, Summary

**验证**:
- ✅ plot_training_curves() 正确实现
- ✅ **向后兼容**（修改后）：支持旧格式历史文件
- ✅ 使用 `.get()` 安全读取字段
- ✅ 自动检测是否有新损失字段
- ✅ 路径 bug 已修复（空目录名处理）

**向后兼容行为**:
- 新格式（有 count_loss, retrieval_loss）→ 显示所有 5 个损失
- 旧格式（只有 3 个损失）→ 显示 3 个损失 + 灰色提示

---

#### 2.2 分子 3D 可视化 ✅

**代码位置**: `src/visualize_val.py`

**输出文件**: `visualizations/molecules_diffusion/sample_XXX.png` (10个)

**验证**:
- ✅ run.sh 调用正确 (line 56-60)
- ✅ 传递参数：checkpoint, num_samples, output_dir
- ✅ 使用 best_diffusion.pt

---

### 3. 运行脚本 (`run.sh`)

#### 3.1 路径设置 ✅

**验证**:
- ✅ `set -e` - 遇错即停
- ✅ `PYTHONPATH` 正确设置
- ✅ `CONFIG` 指向 micro/config.json
- ✅ `SAVE_DIR` 指向 micro/checkpoints

---

#### 3.2 步骤 1: 训练主模型 ✅

**命令**: `python3 -m src.train --config "$CONFIG"`

**验证**:
- ✅ 使用 `-m` 模块导入方式
- ✅ 传递配置文件路径

---

#### 3.3 步骤 3: 生成训练曲线 ✅

**代码位置**: `run.sh:36-50`

**验证**:
- ✅ Python inline 脚本
- ✅ 检查文件存在性
- ✅ 调用 plot_training_curves()
- ✅ 输出到 visualizations/

**潜在问题**:
- ⚠️ 同时检查 diffusion 和 resnet3d，但实际只训练 diffusion

---

#### 3.4 步骤 4: 生成分子可视化 ✅

**代码位置**: `run.sh:54-69`

**验证**:
- ✅ 检查模型文件存在性
- ✅ 传递正确参数
- ✅ 输出到 visualizations/molecules_diffusion/

---

#### 3.5 步骤 5: 生成总结报告 ✅

**代码位置**: `run.sh:74-167`

**验证**:
- ✅ 生成 training_summary.txt
- ✅ 列出所有输出文件
- ✅ 显示最终评估指标
- ✅ 包含文件大小和绝对路径

---

## ✅ 训练阶段功能验证

### Stage 1 (Epochs 1-30): 基础训练 ✅

**验证**:
- ✅ get_training_stage(1-30) 返回 1
- ✅ train_epoch() 检测 stage = 1
- ✅ enable_constraints = False
- ✅ z_depth_weighting = False
- ✅ constraint_loss = 0.0

### Stage 2 (Epochs 31-45): 约束训练 ✅

**验证**:
- ✅ get_training_stage(31-45) 返回 2
- ✅ train_epoch() 检测 stage = 2
- ✅ enable_constraints = True
- ✅ z_depth_weighting = False
- ✅ model.forward() 接受 enable_constraints 参数
- ✅ compute_all_constraints() 被调用
- ✅ constraint_loss 加入总损失（权重 0.1）

### Stage 3 (Epochs 46-60): 底部聚焦 ✅

**验证**:
- ✅ get_training_stage(46-60) 返回 3
- ✅ train_epoch() 检测 stage = 3
- ✅ enable_constraints = True
- ✅ z_depth_weighting = True
- ✅ 底部原子权重 ×3

---

## ✅ 早停机制验证

**代码位置**: `train.py:679-683`

**条件**:
```python
if config["model_type"] == "diffusion" and epoch >= 60 and rmsd_mean < 1.0:
```

**验证**:
- ✅ 三个条件 AND 组合
- ✅ **确保至少 60 轮训练**（epoch >= 60）
- ✅ 只有在性能极好时才提前停止（RMSD < 1.0）
- ✅ Epoch 1-59 绝不会触发早停

---

## ✅ 配置文件验证 (`config.json`)

**关键配置**:
```json
{
  "model_type": "diffusion",
  "epochs": 60,
  "save_dir": "micro/checkpoints",
  ...
}
```

**验证**:
- ✅ epochs = 60（修改后）
- ✅ model_type = "diffusion"
- ✅ save_dir 正确

---

## 📊 预期输出文件清单

### A. 模型文件

| 文件 | 路径 | 大小估计 | 生成时机 |
|------|------|---------|---------|
| 最佳模型 | `checkpoints/best_diffusion.pt` | ~100-200 MB | 验证集损失改善时 |
| Epoch 10 | `checkpoints/epoch_10_diffusion.pt` | ~100-200 MB | 第 10 轮 |
| Epoch 20 | `checkpoints/epoch_20_diffusion.pt` | ~100-200 MB | 第 20 轮 |
| Epoch 30 | `checkpoints/epoch_30_diffusion.pt` | ~100-200 MB | 第 30 轮 |
| Epoch 40 | `checkpoints/epoch_40_diffusion.pt` | ~100-200 MB | 第 40 轮 |
| Epoch 50 | `checkpoints/epoch_50_diffusion.pt` | ~100-200 MB | 第 50 轮 |
| Epoch 60 | `checkpoints/epoch_60_diffusion.pt` | ~100-200 MB | 第 60 轮 |

### B. 训练数据文件

| 文件 | 路径 | 内容 | 格式 |
|------|------|------|------|
| 训练历史 | `checkpoints/history_diffusion.json` | 所有 6 个损失 × 60 轮 | JSON |
| 评估指标 | `checkpoints/metrics_diffusion.json` | 6 维指标 × 60 轮 | JSON |
| 模型预测 | `checkpoints/predictions_diffusion.json` | 100 个测试样本预测 | JSON |
| 训练日志 | `checkpoints/training.log` | 完整训练过程文本输出 | TXT |
| 训练总结 | `checkpoints/training_summary.txt` | 所有文件列表和最终指标 | TXT |

### C. 可视化文件

| 文件 | 路径 | 内容 | 格式 |
|------|------|------|------|
| 训练曲线 | `visualizations/curves_diffusion.png` | 2×3 布局，5 个损失曲线 | PNG |
| 分子可视化 | `visualizations/molecules_diffusion/sample_000.png` ~ `sample_009.png` | 10 个 GT vs Pred 对比图 | PNG |

---

## ⚠️ 潜在问题和建议

### 问题 1: run.sh 检查 resnet3d 文件（低优先级）

**位置**: `run.sh:45-48, 63-68`

**问题**: 脚本检查并尝试处理 resnet3d 相关文件，但实际上只训练 diffusion 模型。

**影响**: 不会导致错误，但会输出不相关的检查信息。

**建议**: 可选修复
```bash
# 删除或注释掉 resnet3d 相关检查
```

**优先级**: 🟡 低（不影响功能）

---

### 问题 2: visualize_val.py 路径（需验证）

**位置**: `run.sh:56`

**问题**: 未检查 `src/visualize_val.py` 是否存在。

**验证**: 让我检查

---

## 🔍 额外验证检查

让我检查 visualize_val.py 是否存在...
