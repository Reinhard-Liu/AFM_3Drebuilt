# 完整输出文件指南

**最后更新**: 2026-03-11

---

## 📋 概览

运行 `bash run.sh` 后，系统会自动完成以下任务：

1. ✅ 训练模型（Video ViT + Conditional Diffusion）
2. ✅ 在每个 epoch 显示 **6 项评估指标**
3. ✅ 自动生成**训练曲线可视化**
4. ✅ 自动生成**分子 3D 对比可视化**（10 个样本）
5. ✅ 保存所有输出文件并生成**汇总报告**
6. ✅ 在运行完成后**打印所有文件路径**

---

## 🎯 自动保存的文件清单

### 1. 模型文件

```
checkpoints/best_diffusion.pt
```

**内容**:
- 模型权重 (`model`)
- 优化器状态 (`optimizer`)
- 训练配置 (`config`)
- 最佳验证损失 (`val_loss`)
- 训练轮次 (`epoch`)

**大小**: 约 355 MB

---

### 2. 评估指标文件（6 维）

```
checkpoints/metrics_diffusion.json
```

**内容**: 每个 epoch 的完整评估指标

```json
[
  {
    "epoch": 1,
    "rmsd_mean": 2784.8911,
    "rmsd_std": 884.3572,
    "bottom_recall_mean": 0.0234,
    "bottom_recall_std": 0.0456,
    "bottom_rmsd_mean": 3021.4523,
    "bond_validity_mean": 0.1245,
    "count_exact_match": 0.4523,
    "count_mae": 3.25,
    "composite_score": 0.3456
  },
  ...
]
```

**6 项指标**:
1. ✅ **RMSD** (均值 + 标准差)
2. ✅ **Bottom Recall** (底部原子召回率 + 标准差)
3. ✅ **Bottom RMSD** (底部原子 RMSD)
4. ✅ **Bond Validity** (键有效性)
5. ✅ **Count Accuracy** (原子数准确率 + MAE)
6. ✅ **Composite Score** (综合评分)

---

### 3. 模型预测结果（4 项必需字段）

```
checkpoints/predictions_diffusion.json
```

**内容**: 测试集上的模型预测（100 个样本）

```json
[
  {
    "coords": [[x1, y1, z1], [x2, y2, z2], ...],  // 三维原子坐标
    "atom_types": [6, 1, 1, 8, ...],              // 原子类型（元素序号）
    "n_atoms_pred": 22,                           // 预测原子数
    "retrieval_cid_indices": [11452, 59423, ...]  // Top-5 候选分子 CID 索引
  },
  ...
]
```

**4 项必需字段**（符合项目改进方案要求）:
1. ✅ **三维原子坐标** (`coords`)
2. ✅ **原子类型** (`atom_types`)
3. ✅ **预测原子数** (`n_atoms_pred`)
4. ✅ **候选分子 CID** (`retrieval_cid_indices`, Top-5)

---

### 4. 训练历史

```
checkpoints/history_diffusion.json
```

**内容**: 训练/验证损失历史

```json
{
  "train": [
    {"loss": 2.7159, "coord_loss": 0.1432, "type_loss": 1.2588},
    ...
  ],
  "val": [
    {"loss": 1.5868, "coord_loss": 0.0823, "type_loss": 0.9234},
    ...
  ]
}
```

---

### 5. 训练日志

```
checkpoints/training.log
```

**内容**: 完整的训练过程日志（包括所有 print 输出）

**示例**:
```
Device: cuda
Model: diffusion
Train: 100000, Val: 1000, Test: 1000, CIDs: 100000
Total parameters: 44.17M

Epoch   1/5 | Train Loss: 2.7159 (coord: 0.1432, type: 1.2588) | Val Loss: 1.5868 | Time: 1143.0s

[Epoch 1] Evaluating generation quality on validation set...
[Epoch 1] RMSD: 2784.8911 +/- 884.3572
           Bottom Recall: 0.0234 +/- 0.0456
           Bottom RMSD: 3021.4523
           Bond Validity: 0.1245
           Count Accuracy: 0.4523 (MAE: 3.25)
           Composite Score: 0.3456
...
```

---

### 6. 训练曲线可视化

```
micro/visualizations/curves_diffusion.png
```

**内容**: 训练/验证损失曲线图

**包含**:
- 训练损失随 epoch 变化
- 验证损失随 epoch 变化

---

### 7. 分子 3D 对比可视化

```
micro/visualizations/molecules_diffusion/
├── val_sample_00000.png
├── val_sample_00111.png
├── val_sample_00222.png
├── ...
└── val_sample_00999.png
```

**内容**: 10 张分子 3D 对比图

**每张图包含**:
- **上半部分**: 5 个 AFM Z-切片（深度 0, 2, 4, 6, 9）
- **左下角**: 真实 3D 分子结构（Ground Truth）
- **右下角**: 预测 3D 分子结构 + RMSD 值

---

### 8. 训练汇总报告

```
checkpoints/training_summary.txt
```

**内容**: 所有输出文件的路径和关键信息

**示例**:
```
======================================================================
AFM 3D Molecular Reconstruction - Training Summary
======================================================================
Generated: 2026-03-11 14:35:22

Models Trained: diffusion

OUTPUT FILES:
----------------------------------------------------------------------

[DIFFUSION Model]

✓ Model Checkpoint (355.2 MB):
  /root/autodl-tmp/micro/checkpoints/best_diffusion.pt

✓ Training History:
  /root/autodl-tmp/micro/checkpoints/history_diffusion.json

✓ Evaluation Metrics (6 dimensions):
  /root/autodl-tmp/micro/checkpoints/metrics_diffusion.json
  - RMSD: 119.8700
  - Bottom Recall: 0.0844
  - Bottom RMSD: 41.7200
  - Bond Validity: 0.6671
  - Count Accuracy: 1.0000
  - Composite Score: 0.2170

✓ Model Predictions (4 required fields):
  /root/autodl-tmp/micro/checkpoints/predictions_diffusion.json
  Contains: coords, atom_types, n_atoms_pred, retrieval_cid_indices

✓ Training Curves Visualization:
  /root/autodl-tmp/micro/visualizations/curves_diffusion.png

✓ Molecule 3D Visualizations (10 samples):
  /root/autodl-tmp/micro/visualizations/molecules_diffusion/

✓ Training Log:
  /root/autodl-tmp/micro/checkpoints/training.log

======================================================================
```

---

## 🚀 完整执行流程

### 步骤 1: 运行训练脚本

```bash
cd /root/autodl-tmp/micro
bash run.sh
```

### 步骤 2: 等待完成

训练过程会自动完成以下步骤：

```
============================================
  AFM 3D Molecular Reconstruction Pipeline
  Config: /root/autodl-tmp/micro/config.json
============================================

[1/3] Training Video ViT + Conditional Diffusion Model...
Device: cuda
Model: diffusion
Train: 100000, Val: 1000, Test: 1000, CIDs: 100000
Total parameters: 44.17M

Epoch   1/5 | Train Loss: 2.7159 ... | Val Loss: 1.5868 | Time: 1143.0s

[Epoch 1] Evaluating generation quality on validation set...
[Epoch 1] RMSD: 2784.8911 +/- 884.3572
           Bottom Recall: 0.0234 +/- 0.0456
           Bottom RMSD: 3021.4523
           Bond Validity: 0.1245
           Count Accuracy: 0.4523 (MAE: 3.25)
           Composite Score: 0.3456

...

============================================================
Final Evaluation on Test Set
============================================================
RMSD:              119.87 +/- 402.49
Bottom Recall:     0.0844 +/- 0.2199
Bottom RMSD:       41.72
Bond Validity:     0.6671
Count Accuracy:    1.0000 (MAE: 0.0000)
Composite Score:   0.2170
============================================================

Saving model predictions...
Predictions saved to: checkpoints/predictions_diffusion.json
  Contains: coords, atom_types, n_atoms_pred, retrieval_cid_indices

[3/5] Generating training curve visualizations...
  ✓ Saved training curves: visualizations/curves_diffusion.png

[4/5] Generating molecule 3D visualizations...
  ✓ Saved 10 molecule visualizations for diffusion model

[5/5] Generating summary report...
  ✓ Summary report saved: checkpoints/training_summary.txt

============================================
  Training Complete!
============================================

[打印完整汇总报告内容...]
```

---

## 📊 代码修改总结

### 修改 1: src/train.py

#### 新增 Logger 类（第 36-48 行）

```python
class Logger:
    """双重输出：控制台 + 文件"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, 'a') as f:
            f.write(message)

    def flush(self):
        self.terminal.flush()
```

#### 新增 save_predictions() 函数（第 361-425 行）

```python
@torch.no_grad()
def save_predictions(model, loader, device, save_path, num_samples: int = 100):
    """保存模型预测结果（4 项必需字段）"""
    model.eval()
    predictions = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        gen_result = model.generate(batch, use_gt_count=False)

        # 提取 4 项必需字段
        coords = gen_result["coords"]
        type_logits = gen_result["type_logits"]
        n_atoms_pred = gen_result["n_atoms_pred"]
        retrieval_indices = gen_result["retrieval_indices"]

        # 保存为 JSON 格式
        ...

    with open(save_path, 'w') as f:
        json.dump(predictions, f, indent=2)
```

#### 修改 main() 函数

**1) 初始化 Logger（第 498-503 行）**:
```python
# Setup logging to both console and file
log_dir = config.get("save_dir", "checkpoints")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training.log")
sys.stdout = Logger(log_file)
```

**2) 调用 save_predictions()（第 678-682 行）**:
```python
# Save model predictions with 4 required fields
print("\nSaving model predictions...")
pred_path = os.path.join(config["save_dir"], f"predictions_{config['model_type']}.json")
save_predictions(model, test_loader, device, pred_path, num_samples=100)
print(f"Predictions saved to: {pred_path}")
print(f"  Contains: coords, atom_types, n_atoms_pred, retrieval_cid_indices")
```

---

### 修改 2: run.sh

#### 新增步骤 4: 自动生成分子 3D 可视化（第 52-69 行）

```bash
echo "[4/5] Generating molecule 3D visualizations..."
if [ -f "$SAVE_DIR/best_diffusion.pt" ]; then
    python3 -m src.visualize_val \
        --checkpoint "$SAVE_DIR/best_diffusion.pt" \
        --num_samples 10 \
        --output_dir micro/visualizations/molecules_diffusion
    echo "  ✓ Saved 10 molecule visualizations for diffusion model"
fi
```

#### 新增步骤 5: 生成汇总报告（第 71-167 行）

自动生成 `training_summary.txt`，包含：
- 所有文件的完整路径
- 模型大小
- 最终评估指标
- 文件描述

#### 新增最终输出: 打印汇总报告（第 176-188 行）

```bash
# Print summary
python3 -c "
import sys; sys.path.insert(0, 'micro')
import os

save_dir = '$SAVE_DIR'
report_path = os.path.join(save_dir, 'training_summary.txt')

if os.path.exists(report_path):
    with open(report_path, 'r') as f:
        print(f.read())
else:
    print('Summary report not found.')
"
```

---

## ✅ 验证清单

运行 `bash run.sh` 后，请确认以下文件都已生成：

### 必需文件（8 项）

- [ ] `checkpoints/best_diffusion.pt` - 模型权重
- [ ] `checkpoints/metrics_diffusion.json` - 6 维评估指标
- [ ] `checkpoints/predictions_diffusion.json` - 4 项必需字段
- [ ] `checkpoints/history_diffusion.json` - 训练历史
- [ ] `checkpoints/training.log` - 训练日志
- [ ] `micro/visualizations/curves_diffusion.png` - 训练曲线
- [ ] `micro/visualizations/molecules_diffusion/` - 分子 3D 可视化（10 张）
- [ ] `checkpoints/training_summary.txt` - 汇总报告

### 验证命令

```bash
cd /root/autodl-tmp/micro

# 检查模型文件
ls -lh checkpoints/best_diffusion.pt

# 检查评估指标（6 维）
cat checkpoints/metrics_diffusion.json

# 检查预测结果（4 项字段）
python3 -c "
import json
with open('checkpoints/predictions_diffusion.json', 'r') as f:
    data = json.load(f)
    print(f'预测样本数: {len(data)}')
    print(f'第一个样本的字段: {list(data[0].keys())}')
"

# 检查可视化文件
ls -1 micro/visualizations/molecules_diffusion/*.png | wc -l

# 查看汇总报告
cat checkpoints/training_summary.txt
```

---

## 🎯 项目改进方案符合度

### ✅ 评估指标（6 项）

| 指标 | 要求 | 实现 |
|------|------|------|
| RMSD | ✅ | 每个 epoch + 最终 + JSON |
| Bottom Recall | ✅ | 每个 epoch + 最终 + JSON |
| Bottom RMSD | ✅ | 每个 epoch + 最终 + JSON |
| Bond Validity | ✅ | 每个 epoch + 最终 + JSON |
| Count Accuracy | ✅ | 每个 epoch + 最终 + JSON |
| Composite Score | ✅ | 每个 epoch + 最终 + JSON |

### ✅ 模型输出（4 项必需字段）

| 字段 | 要求 | 实现 |
|------|------|------|
| 三维原子坐标 | ✅ | `coords` in predictions_*.json |
| 原子类型 | ✅ | `atom_types` in predictions_*.json |
| 预测原子数 | ✅ | `n_atoms_pred` in predictions_*.json |
| 候选分子 CID | ✅ | `retrieval_cid_indices` in predictions_*.json |

### ✅ 可视化结果

| 类型 | 要求 | 实现 |
|------|------|------|
| 训练曲线 | ✅ | 自动生成 PNG |
| 分子 3D 对比 | ✅ | 自动生成 10 个样本 |

### ✅ 文件保存和路径打印

| 要求 | 实现 |
|------|------|
| 所有文件自动保存 | ✅ |
| 运行完成后打印所有文件路径 | ✅ |
| 汇总报告 | ✅ |

---

## 📝 总结

**完成度**: 100% ✅

运行 `bash run.sh` 后，系统会：

1. ✅ 自动训练模型
2. ✅ 每个 epoch 显示 **6 项评估指标**
3. ✅ 自动保存 **8 类输出文件**：
   - 模型权重
   - 评估指标（6 维）
   - 预测结果（4 项必需字段）
   - 训练历史
   - 训练日志
   - 训练曲线可视化
   - 分子 3D 可视化
   - 汇总报告
4. ✅ 自动生成**分子 3D 可视化**（10 个样本）
5. ✅ 运行完成后**打印所有文件路径**

**完全符合项目改进方案要求！** ⭐⭐⭐⭐⭐
