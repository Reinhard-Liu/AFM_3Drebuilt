# 预期输出文件清单

运行 `bash run.sh` 后将生成以下所有文件。

---

## 📁 模型权重文件（checkpoints/）

### 最佳模型
```
checkpoints/best_diffusion.pt
```
- **大小**: ~100-200 MB
- **内容**: 验证集损失最低时的模型状态
- **包含**:
  - model.state_dict() - 模型权重
  - optimizer.state_dict() - 优化器状态
  - epoch - 对应轮次
  - val_loss - 验证集损失
  - config - 训练配置

### 定期检查点（每10轮）
```
checkpoints/epoch_10_diffusion.pt
checkpoints/epoch_20_diffusion.pt
checkpoints/epoch_30_diffusion.pt  ← Stage 1 结束
checkpoints/epoch_40_diffusion.pt
checkpoints/epoch_50_diffusion.pt
checkpoints/epoch_60_diffusion.pt  ← 最终模型
```
- **大小**: 每个 ~100-200 MB
- **总计**: ~700-1400 MB
- **用途**: 可以从任意检查点恢复训练

---

## 📊 训练数据文件（checkpoints/）

### 1. 训练损失历史
```
checkpoints/history_diffusion.json
```

**内容格式**:
```json
{
  "train": [
    {
      "loss": 2.7363,
      "coord_loss": 0.1434,
      "type_loss": 1.2591,
      "count_loss": 3.7059,
      "retrieval_loss": 12.2814,
      "constraint_loss": 0.0000  // Stage 1: 0.0, Stage 2+: 有值
    },
    ... // 60 个 epoch
  ],
  "val": [
    ... // 60 个 epoch
  ]
}
```

**包含指标**（每轮）:
- ✅ Total Loss
- ✅ Coordinate Loss (MSE)
- ✅ Atom Type Loss (CE)
- ✅ Count Loss（分类+回归）
- ✅ Retrieval Loss (InfoNCE)
- ✅ Constraint Loss（Stage 2+ 激活）

**大小**: ~10-50 KB

---

### 2. 评估指标历史
```
checkpoints/metrics_diffusion.json
```

**内容格式**:
```json
[
  {
    "epoch": 1,
    "rmsd_mean": 83.4250,
    "rmsd_std": 934.4512,
    "bottom_recall_mean": 0.0843,
    "bottom_recall_std": 0.0123,
    "bottom_rmsd_mean": 10.9870,
    "bond_validity_mean": 0.7654,
    "count_exact_match": 0.44,  // 真实预测准确率（不再是 100%）
    "count_mae": 2.15,
    "composite_score": 0.1234
  },
  ... // 60 个 epoch
]
```

**包含指标**（每轮）:
- ✅ RMSD (mean ± std) - 几何精度
- ✅ Bottom Atom Recall - 底部原子召回率
- ✅ Bottom RMSD - 底部原子几何精度
- ✅ Bond Validity - 化学键有效率
- ✅ Count Accuracy - 原子数精确匹配率
- ✅ Count MAE - 原子数平均绝对误差
- ✅ Composite Score - 综合评分

**大小**: ~5-20 KB

**重要变化**: `count_exact_match` 现在使用 **预测值** 进行计算，反映真实性能（~44%），而非自比对的 100%。

---

### 3. 模型预测
```
checkpoints/predictions_diffusion.json
```

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
      "sample_id": 0,
      "coords": [
        [0.123, -0.456, 0.789],
        [0.234, 0.567, -0.890],
        ...
      ],  // 仅有效原子（前 n_atoms_pred 个）
      "atom_types": [1, 1, 0, 2, ...],  // 仅有效原子
      "n_atoms_pred": 15,
      "retrieval_cid_indices": [123, 456, 789, 101, 112],  // Top-5 CID
      "retrieval_scores": [0.98, 0.87, 0.76, 0.65, 0.54]  // 可选
    },
    ... // 100 个测试样本
  ]
}
```

**包含字段**（4个必需 + 1个可选）:
- ✅ coords - 3D 原子坐标（仅有效原子）
- ✅ atom_types - 原子类型预测（仅有效原子）
- ✅ n_atoms_pred - 预测的原子数
- ✅ retrieval_cid_indices - Top-5 候选分子 CID
- 🔹 retrieval_scores - Top-5 检索分数（可选）

**样本数**: 100 个测试样本
**大小**: ~1-5 MB

---

### 4. 训练日志
```
checkpoints/training.log
```

**内容**: 完整的训练过程文本输出

**包含信息**:
- 每轮训练/验证损失
- RMSD 和 6 维评估指标
- 模型保存通知
- 早停检查（如触发）
- 阶段切换标识（S1 → S2 → S3）

**示例片段**:
```
Train [31/60] S2: 100%|██████| 50/50 [00:45<00:00, loss=1.4567]
                ^^^
              Stage 2 标识

[Epoch 31] Train Loss: 1.4567, Val Loss: 1.3456
[Epoch 31] RMSD: 5.6789 +/- 0.1234
[Epoch 31] Bottom Recall: 0.1567
[Epoch 31] Bond Validity: 0.8234 ← Stage 2 约束效果
[Saved] Best model: checkpoints/best_diffusion.pt
```

**大小**: ~100-500 KB（取决于详细度）

---

### 5. 训练总结报告
```
checkpoints/training_summary.txt
```

**内容**: 所有输出文件列表 + 最终指标

**示例**:
```
======================================================================
AFM 3D Molecular Reconstruction - Training Summary
======================================================================
Generated: 2026-03-12 15:30:00

Models Trained: diffusion

OUTPUT FILES:
----------------------------------------------------------------------

[DIFFUSION Model]

✓ Model Checkpoint (156.7 MB):
  /root/autodl-tmp/micro/checkpoints/best_diffusion.pt

✓ Training History:
  /root/autodl-tmp/micro/checkpoints/history_diffusion.json

✓ Evaluation Metrics (6 dimensions):
  /root/autodl-tmp/micro/checkpoints/metrics_diffusion.json
  - RMSD: 5.6789
  - Bottom Recall: 0.1567
  - Bottom RMSD: 4.3210
  - Bond Validity: 0.8234
  - Count Accuracy: 0.4500
  - Composite Score: 0.5678

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

**大小**: ~5-10 KB

---

## 📈 可视化文件（visualizations/）

### 1. 训练曲线图
```
visualizations/curves_diffusion.png
```

**布局**: 2×3 子图

```
┌─────────────────┬─────────────────┬─────────────────┐
│ Total Loss      │ Coordinate Loss │ Atom Type Loss  │
│ (Train + Val)   │ (MSE)           │ (CE)            │
├─────────────────┼─────────────────┼─────────────────┤
│ Count Loss      │ Retrieval Loss  │ Summary:        │
│ (weight=0.5)    │ (weight=0.05)   │ Loss Formula +  │
│                 │                 │ Final Stats     │
└─────────────────┴─────────────────┴─────────────────┘
```

**显示内容**:
- ✅ 所有 5 个损失组件的训练/验证曲线
- ✅ 损失公式说明
- ✅ 最终 epoch 的统计数据

**向后兼容**: 如果使用旧格式历史文件（只有 3 个损失），会显示前 3 个子图 + 灰色提示。

**大小**: ~500 KB
**分辨率**: 2700×1500 (150 DPI)

---

### 2. 分子 3D 可视化
```
visualizations/molecules_diffusion/
├── sample_000.png  ← GT vs Predicted 对比
├── sample_001.png
├── sample_002.png
├── sample_003.png
├── sample_004.png
├── sample_005.png
├── sample_006.png
├── sample_007.png
├── sample_008.png
└── sample_009.png
```

**每个图像**:
- 左侧: Ground Truth (真实结构)
- 右侧: Predicted (模型预测)
- 原子颜色编码:
  - H (白色), C (深灰), N (蓝色), O (红色)
  - F (绿色), S (黄色), P (橙色), Cl (亮绿), Br (棕色), I (紫色)

**数量**: 10 个样本
**单个大小**: ~500 KB
**总大小**: ~5 MB
**分辨率**: 2100×900 (150 DPI)

---

## 📦 完整文件树

```
micro/
├── checkpoints/
│   ├── best_diffusion.pt                   ~150 MB
│   ├── epoch_10_diffusion.pt               ~150 MB
│   ├── epoch_20_diffusion.pt               ~150 MB
│   ├── epoch_30_diffusion.pt               ~150 MB
│   ├── epoch_40_diffusion.pt               ~150 MB
│   ├── epoch_50_diffusion.pt               ~150 MB
│   ├── epoch_60_diffusion.pt               ~150 MB
│   ├── history_diffusion.json              ~20 KB
│   ├── metrics_diffusion.json              ~10 KB
│   ├── predictions_diffusion.json          ~2 MB
│   ├── training.log                        ~300 KB
│   └── training_summary.txt                ~8 KB
│
└── visualizations/
    ├── curves_diffusion.png                ~500 KB
    └── molecules_diffusion/
        ├── sample_000.png                  ~500 KB
        ├── sample_001.png                  ~500 KB
        ├── ...
        └── sample_009.png                  ~500 KB

总计: ~1.2 GB
```

---

## 🎯 关键输出验证清单

运行 `bash run.sh` 后，使用以下命令验证所有文件都已生成：

```bash
# 检查模型文件
ls -lh checkpoints/*.pt

# 检查数据文件
ls -lh checkpoints/*.json checkpoints/*.log checkpoints/*.txt

# 检查可视化文件
ls -lh visualizations/*.png
ls -lh visualizations/molecules_diffusion/*.png

# 查看训练总结
cat checkpoints/training_summary.txt

# 验证 history 包含所有 6 个损失
cat checkpoints/history_diffusion.json | jq '.train[0] | keys'
# 预期输出: ["loss", "coord_loss", "type_loss", "count_loss", "retrieval_loss", "constraint_loss"]

# 验证 metrics 包含真实 count_accuracy
cat checkpoints/metrics_diffusion.json | jq '.[-1].count_exact_match'
# 预期输出: ~0.40-0.50（不再是 1.0）
```

---

## 📊 预期最终指标（参考范围）

基于 60 轮完整训练（3个阶段）:

| 指标 | Epoch 1 | Epoch 30<br>(Stage 1 结束) | Epoch 45<br>(Stage 2 结束) | Epoch 60<br>(最终) |
|------|---------|--------------------------|--------------------------|------------------|
| **RMSD** | 80-100 Å | 5-15 Å | 5-12 Å | **5-10 Å** |
| **Bottom Recall** | 0.05-0.10 | 0.08-0.15 | 0.12-0.18 | **0.15-0.25** ✨ |
| **Bottom RMSD** | 10-15 Å | 8-12 Å | 6-10 Å | **4-8 Å** ✨ |
| **Bond Validity** | 0.70-0.80 | 0.75-0.82 | **0.82-0.90** ✨ | 0.85-0.90 |
| **Count Accuracy** | 0.30-0.40 | 0.40-0.50 | 0.42-0.52 | **0.45-0.55** |
| **Composite Score** | 0.10-0.20 | 0.30-0.45 | 0.40-0.55 | **0.45-0.60** |

✨ = 对应阶段的重点优化指标

**注意**:
- Stage 2 主要提升 **Bond Validity**（物理约束效果）
- Stage 3 主要提升 **Bottom Recall/RMSD**（底部原子聚焦效果）
- Count Accuracy 是**真实预测准确率**（~44%），不再是 100%

---

## 🔍 如何读取输出文件

### Python 读取示例

```python
import json
import torch

# 1. 读取训练历史
with open('checkpoints/history_diffusion.json', 'r') as f:
    history = json.load(f)

final_loss = history['train'][-1]['loss']
print(f"最终训练损失: {final_loss:.4f}")

# 2. 读取评估指标
with open('checkpoints/metrics_diffusion.json', 'r') as f:
    metrics = json.load(f)

final_metrics = metrics[-1]
print(f"最终 RMSD: {final_metrics['rmsd_mean']:.2f} Å")
print(f"最终 Count Accuracy: {final_metrics['count_exact_match']:.2%}")

# 3. 读取模型预测
with open('checkpoints/predictions_diffusion.json', 'r') as f:
    predictions = json.load(f)

first_pred = predictions['predictions'][0]
print(f"第1个样本预测的原子数: {first_pred['n_atoms_pred']}")
print(f"Top-1 检索 CID: {first_pred['retrieval_cid_indices'][0]}")

# 4. 加载模型权重
checkpoint = torch.load('checkpoints/best_diffusion.pt')
print(f"最佳模型来自 Epoch {checkpoint['epoch']}")
print(f"验证集损失: {checkpoint['val_loss']:.4f}")

# 恢复模型
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
```

---

## ⚠️ 注意事项

### 磁盘空间
- **最少需要**: 2 GB 可用空间
- **推荐**: 5 GB 可用空间（留有余量）

### 训练时间
- **GPU**: RTX 4080 SUPER
- **预计时长**: 15-20 小时（60 轮）
- **可中断**: 可以 Ctrl+C 中断，从最近的检查点恢复

### 文件保留
- `best_diffusion.pt` - **必须保留**（最佳模型）
- `epoch_60_diffusion.pt` - **建议保留**（最终模型）
- 其他 `epoch_X_diffusion.pt` - 可删除以节省空间（~1 GB）

### 恢复训练
如果需要从检查点恢复训练，可以修改 `src/train.py` 加载检查点：
```python
# 在 main() 函数中添加
if os.path.exists('checkpoints/epoch_30_diffusion.pt'):
    checkpoint = torch.load('checkpoints/epoch_30_diffusion.pt')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
```

---

**生成时间**: 2026-03-12
**基于配置**: config.json (model_type=diffusion, epochs=60)
