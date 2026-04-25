# 训练输出文件完整清单

生成时间：2026-03-12
训练状态：27轮早停（修改前配置）

---

## 📁 目录结构

```
/root/autodl-tmp/micro/
├── checkpoints/              # 模型权重和训练历史
│   ├── *.pt                  # 模型权重文件
│   ├── *.json                # 训练历史和预测结果
│   └── training.log          # 训练日志
│
├── visualizations/           # 可视化图片
│   ├── molecules_diffusion/  # 验证集可视化
│   └── test_predictions/     # 测试集可视化
│
└── [文档和脚本]              # 说明文档和验证工具

```

---

## 1️⃣ Checkpoints - 模型权重

### 1.1 主要模型文件

| 文件路径 | 大小 | 说明 |
|---------|------|------|
| `checkpoints/best_diffusion.pt` | 506 MB | **最佳扩散模型**（Epoch 25，验证损失最低） |
| `checkpoints/best_resnet3d.pt` | 104 MB | 最佳ResNet3D基线模型 |

### 1.2 定期保存的检查点

| 文件路径 | 大小 | 说明 |
|---------|------|------|
| `checkpoints/epoch_10_diffusion.pt` | 506 MB | 第10轮检查点 |
| `checkpoints/epoch_20_diffusion.pt` | 506 MB | 第20轮检查点 |
| `checkpoints/epoch_30_diffusion.pt` | 355 MB | 第30轮检查点（注：大小不同，可能配置变化） |
| `checkpoints/epoch_40_diffusion.pt` | 355 MB | 第40轮检查点 |

**注意：** 训练在第27轮早停，epoch_30和epoch_40是之前训练遗留的文件。

### 1.3 检查点内容

每个 `.pt` 文件包含：
```python
{
    "epoch": int,           # 训练轮次
    "model": state_dict,    # 模型参数
    "optimizer": state_dict,# 优化器状态
    "config": dict,         # 训练配置
    "history": dict,        # 训练历史（可选）
}
```

加载方法：
```python
checkpoint = torch.load("checkpoints/best_diffusion.pt")
model.load_state_dict(checkpoint["model"])
```

---

## 2️⃣ 训练历史和评估数据

### 2.1 训练历史（JSON格式）

| 文件路径 | 大小 | 说明 |
|---------|------|------|
| `checkpoints/history_diffusion.json` | 11 KB | 扩散模型每轮的训练/验证损失 |
| `checkpoints/history_resnet3d.json` | 289 B | ResNet3D基线模型历史 |

**内容结构：**
```json
{
  "train": [
    {"loss": 1.234, "coord_loss": 0.56, "type_loss": 0.12, ...},
    ...
  ],
  "val": [
    {"loss": 1.123, "coord_loss": 0.54, "type_loss": 0.11, ...},
    ...
  ]
}
```

### 2.2 评估指标（JSON格式）

| 文件路径 | 大小 | 说明 |
|---------|------|------|
| `checkpoints/metrics_diffusion.json` | 9.9 KB | **完整的6维评估指标**（每轮） |
| `checkpoints/rmsd_diffusion.json` | 946 B | RMSD历史记录（简化版） |
| `checkpoints/rmsd_resnet3d.json` | 195 B | ResNet3D的RMSD |

**metrics_diffusion.json 包含：**
```json
[
  {
    "epoch": 1,
    "rmsd_mean": 4.66,
    "rmsd_std": 84.13,
    "bottom_recall_mean": 0.0732,
    "bottom_rmsd_mean": 1.5796,
    "bond_validity_mean": 0.8319,
    "count_exact_match": 1.0000,
    "count_mae": 0.0000,
    "composite_score": 0.2394
  },
  ...
]
```

### 2.3 训练日志

| 文件路径 | 大小 | 说明 |
|---------|------|------|
| `checkpoints/training.log` | 12 KB | **完整的终端输出日志** |

**包含内容：**
- 每轮训练的损失和指标
- 验证集评估结果
- 早停触发信息
- 检查点保存记录
- 最终测试集评估结果

---

## 3️⃣ 预测结果

### 3.1 测试集预测（JSON格式）

| 文件路径 | 大小 | 说明 |
|---------|------|------|
| `checkpoints/predictions_diffusion.json` | 358 KB | **100个测试样本的完整预测** |

**文件结构：**
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
      "coords": [[x, y, z], ...],           // (n_atoms, 3)
      "atom_types": [0, 1, 2, ...],         // (n_atoms,)
      "n_atoms_pred": 35,
      "retrieval_cid_indices": [14956, 2877, 6438, 99451, 234],
      "retrieval_scores": [0.89, 0.76, 0.65, 0.54, 0.43]
    },
    ...
  ]
}
```

**关键信息：**
- 使用 `use_gt_count=False`（真实端到端预测）
- 包含改进方案中要求的4个字段
- 坐标为归一化空间（需×12.0还原为埃）

**使用示例：**
```python
import json
with open('checkpoints/predictions_diffusion.json', 'r') as f:
    data = json.load(f)

sample = data['predictions'][0]
coords = np.array(sample['coords']) * 12.0  # 还原为埃
n_atoms = sample['n_atoms_pred']
```

---

## 4️⃣ 可视化图片

### 4.1 验证集可视化（10张）

**目录：** `visualizations/molecules_diffusion/`

| 文件名 | 验证集索引 | 说明 |
|--------|-----------|------|
| `val_sample_00000.png` | 0 | 第1个验证样本 |
| `val_sample_00111.png` | 111 | 第112个验证样本 |
| `val_sample_00222.png` | 222 | 第223个验证样本 |
| `val_sample_00333.png` | 333 | 第334个验证样本 |
| `val_sample_00444.png` | 444 | 第445个验证样本 |
| `val_sample_00555.png` | 555 | 第556个验证样本 |
| `val_sample_00666.png` | 666 | 第667个验证样本 |
| `val_sample_00777.png` | 777 | 第778个验证样本 |
| `val_sample_00888.png` | 888 | 第889个验证样本 |
| `val_sample_00999.png` | 999 | 第1000个验证样本 |

**采样策略：** 均匀间隔（linspace）

**图片内容：**
- 上半部分：5个AFM深度切片（Z-slice 0, 2, 4, 6, 9）
- 下半部分左：真实3D结构
- 下半部分右：预测3D结构（标注RMSD）

### 4.2 测试集可视化（10张）

**目录：** `visualizations/test_predictions/`

| 文件名 | 测试集索引 | 真实原子数 | 预测原子数 | RMSD (Å) |
|--------|-----------|-----------|-----------|----------|
| `test_sample_00000.png` | 0 | 37 | 35 | 0.39 |
| `test_sample_00001.png` | 1 | 31 | 32 | 0.24 |
| `test_sample_00002.png` | 2 | 21 | 21 | 0.25 |
| `test_sample_00003.png` | 3 | 15 | 15 | 0.21 |
| `test_sample_00004.png` | 4 | 19 | 19 | 0.17 |
| `test_sample_00005.png` | 5 | 21 | 20 | 0.13 |
| `test_sample_00006.png` | 6 | 22 | 22 | 0.24 |
| `test_sample_00007.png` | 7 | 27 | 26 | 0.20 |
| `test_sample_00008.png` | 8 | 24 | 24 | 0.20 |
| `test_sample_00009.png` | 9 | 21 | 23 | 0.25 |

**采样策略：** 前10个测试样本（索引0-9）

**索引映射文件：** `visualizations/test_predictions/index_mapping.json`

```json
[
  {
    "visualization_file": "test_sample_00000.png",
    "test_set_index": 0,
    "predictions_json_index": 0,
    "rmsd": 0.3937,
    "n_atoms": 37
  },
  ...
]
```

---

## 5️⃣ 文档和工具（本次会话创建）

### 5.1 说明文档

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `MODIFICATION_SUMMARY.md` | 4.1 KB | 训练配置修改说明（早停机制、三阶段） |
| `BEFORE_AFTER_COMPARISON.md` | 5.2 KB | 修改前后详细对比 |
| `ATOM_COUNT_ACCURACY_ANALYSIS.md` | 6.8 KB | 原子数准确率技术分析报告 |
| `COUNT_ACCURACY_EXPLANATION.md` | 7.3 KB | 原子数准确率问题解释（含流程图） |
| `TRAINING_OUTPUT_FILES.md` | 本文件 | 训练输出文件完整清单 |

### 5.2 验证脚本

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `verify_modifications.sh` | 3.7 KB | 验证训练配置修改是否成功 |
| `check_real_count_accuracy.py` | 6.4 KB | 检查真实的原子数预测准确率 |

**使用方法：**
```bash
# 验证配置修改
bash verify_modifications.sh

# 检查真实准确率
python3 check_real_count_accuracy.py
```

---

## 6️⃣ 其他重要文件

### 6.1 配置文件

| 文件路径 | 说明 |
|---------|------|
| `/root/autodl-tmp/micro/config.json` | **训练配置**（已修改为60轮） |
| `/root/autodl-tmp/CLAUDE.md` | 项目文档（已更新三阶段） |
| `/root/autodl-tmp/项目改进方案.md` | 改进方案说明（已更新） |

### 6.2 源代码（已修改）

| 文件路径 | 修改内容 |
|---------|---------|
| `src/train.py` | 训练阶段划分（30/45/60）+ 早停机制（epoch>=60） |

---

## 7️⃣ 文件访问路径

### 绝对路径

所有文件的根目录：`/root/autodl-tmp/micro/`

**最重要的文件：**
```
/root/autodl-tmp/micro/checkpoints/best_diffusion.pt
/root/autodl-tmp/micro/checkpoints/predictions_diffusion.json
/root/autodl-tmp/micro/checkpoints/training.log
/root/autodl-tmp/micro/visualizations/test_predictions/
```

### 相对路径（在micro目录下）

```bash
cd /root/autodl-tmp/micro

# 查看训练日志
cat checkpoints/training.log

# 查看预测结果
cat checkpoints/predictions_diffusion.json | jq '.' | less

# 查看可视化
ls visualizations/test_predictions/*.png
```

---

## 8️⃣ 文件大小统计

| 类别 | 总大小 | 文件数 |
|------|--------|--------|
| 模型权重 (*.pt) | ~2.2 GB | 6个 |
| JSON数据 (*.json) | ~380 KB | 7个 |
| 可视化图片 (*.png) | ~15 MB | 20个 |
| 日志文件 (*.log) | 12 KB | 1个 |
| 文档和脚本 (*.md, *.sh, *.py) | ~40 KB | 7个 |

**总计：** 约 2.2 GB

---

## 9️⃣ 快速访问命令

```bash
# 进入项目目录
cd /root/autodl-tmp/micro

# 查看所有checkpoints
ls -lh checkpoints/

# 查看所有可视化
ls -R visualizations/

# 查看所有文档
ls -lh *.md

# 查看训练日志最后50行
tail -50 checkpoints/training.log

# 统计预测结果
cat checkpoints/predictions_diffusion.json | jq '.num_samples'

# 查看第一个预测样本
cat checkpoints/predictions_diffusion.json | jq '.predictions[0]'
```

---

## 🔟 注意事项

### 10.1 文件版本

- 当前文件对应**修改前**的训练（27轮早停）
- 新的60轮训练将覆盖这些文件
- 建议备份重要结果

### 10.2 备份建议

```bash
# 备份当前训练结果
mkdir -p backup_27epochs
cp -r checkpoints/ backup_27epochs/
cp -r visualizations/ backup_27epochs/
```

### 10.3 文件对应关系

```
predictions_diffusion.json  ←→  test_predictions/*.png
     (100个预测)                    (前10个可视化)
          ↓
  index_mapping.json
    (映射关系)
```

---

## 📊 数据来源和生成

| 文件类型 | 生成时机 | 数据来源 |
|---------|---------|---------|
| `*.pt` | 训练过程 | 模型参数自动保存 |
| `history_*.json` | 训练过程 | 每轮损失自动记录 |
| `metrics_*.json` | 训练过程 | 每轮评估自动记录 |
| `predictions_*.json` | 训练结束 | `save_predictions()` |
| `molecules_diffusion/*.png` | 训练结束 | `visualize_val.py` |
| `test_predictions/*.png` | 手动运行 | `visualize_test_predictions.py` |
| 文档和脚本 | 本次会话 | Claude Code生成 |

---

**文档结束**

如有疑问，查看各个说明文档或运行验证脚本。
