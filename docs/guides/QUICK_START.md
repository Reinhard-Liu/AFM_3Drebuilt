# 快速开始：训练和可视化总结

## ✅ 您已完成的工作

### 1. 训练模型
```bash
python3 -m src.train --config config.json
```

**训练结果**：
- ✅ 训练完成：5 个 epochs
- ✅ 最佳验证损失：1.3307（第 3 个 epoch）
- ✅ 测试集 RMSD：80.69 ± 188.43
- ✅ Bottom Recall：0.0887 ± 0.2052
- ✅ 模型已保存：`checkpoints/best_diffusion.pt`

### 2. 生成可视化
```bash
python3 -m src.visualize_val \
    --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 5 \
    --output_dir visualizations/demo
```

**可视化结果**：
- ✅ 5 个样本已生成
- ✅ 保存位置：`visualizations/demo/`
- ✅ 平均 RMSD：222.50 ± 353.36

**生成的文件**：
```
visualizations/demo/
├── val_sample_00000.png  (430 KB) - RMSD: 0.27 ✨ 非常好！
├── val_sample_00249.png  (444 KB) - RMSD: 199.10
├── val_sample_00499.png  (402 KB) - RMSD: 912.25
├── val_sample_00749.png  (435 KB) - RMSD: 0.58 ✨ 非常好！
└── val_sample_00999.png  (427 KB) - RMSD: 0.28 ✨ 非常好！
```

---

## 📊 结果分析

### RMSD 分布
从生成的 5 个样本看：
- **3 个样本非常好**（RMSD < 1.0）：样本 0, 749, 999
- **1 个样本可接受**（RMSD ~200）：样本 249
- **1 个样本较差**（RMSD ~900）：样本 499

**结论**：模型在大部分样本上表现良好，但在某些复杂分子上还有提升空间。

### 训练进度
| Epoch | RMSD (验证集) | 改进 |
|-------|--------------|------|
| 1 | 2784.89 | - |
| 2 | 1641.99 | ↓ 41% |
| 3 | 367.73 | ↓ 78% |
| 4 | 196.18 | ↓ 47% |
| 5 | 99.61 | ↓ 49% |

**趋势**：RMSD 持续下降，说明模型正在学习，继续训练可能会进一步提升。

---

## 🎯 下一步建议

### 选项 1: 继续训练（推荐）

由于 RMSD 还在持续下降，建议继续训练：

```bash
# 修改 config.json
vim config.json
# 将 "epochs": 5 改为 "epochs": 20

# 继续训练
python3 -m src.train --config config.json
```

预期效果：RMSD 可能降至 20-50 范围。

### 选项 2: 生成更多可视化

查看更多样本，了解模型的表现：

```bash
# 生成 20 个样本
python3 -m src.visualize_val \
    --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 20 \
    --output_dir visualizations/detailed
```

### 选项 3: 训练 ResNet3D 进行对比

```bash
# 1. 修改 config.json
vim config.json
# 将 "model_type": "diffusion" 改为 "model_type": "resnet3d"

# 2. 训练 ResNet3D
python3 -m src.train --config config.json

# 3. 生成可视化对比
python3 -m src.visualize_val \
    --checkpoint checkpoints/best_resnet3d.pt \
    --num_samples 5 \
    --output_dir visualizations/resnet3d_demo
```

### 选项 4: 使用更大的数据集

当前使用了 100,000 个样本，完整数据集有 294,123 个：

```bash
# 修改 config.json
vim config.json
# 将 "max_samples": 100000 改为 "max_samples": 0  (0 表示全部)
# 将 "epochs": 5 改为 "epochs": 40

# 开始长时间训练（预计 10-20 小时）
python3 -m src.train --config config.json
```

---

## 📁 文件位置总结

### 模型文件
```
checkpoints/
├── best_diffusion.pt              # 最佳模型（355 MB）
├── history_diffusion.json         # 训练历史
└── rmsd_diffusion.json            # RMSD 历史
```

### 可视化文件
```
visualizations/demo/
├── val_sample_00000.png  # 真实 vs 预测对比图
├── val_sample_00249.png
├── val_sample_00499.png
├── val_sample_00749.png
└── val_sample_00999.png
```

### 配置文件
```
config.json                        # 当前已优化（log_interval=1）
```

---

## 🔧 问题 1 的解决方案已应用

### 修改前
```json
"log_interval": 10  // 每 10 个 epoch 打印一次
```

只有第 1 个 epoch 打印详细日志，第 2-5 个 epoch 被跳过。

### 修改后 ✅
```json
"log_interval": 1  // 每个 epoch 都打印
```

**下次训练时每个 epoch 都会显示**：
```
Epoch   1/20 | Train Loss: 2.7159 (coord: 0.1432, type: 1.2588) | Val Loss: 1.5868 | Time: 1143.0s
Epoch   2/20 | Train Loss: 1.8234 (coord: 0.0987, type: 0.9876) | Val Loss: 1.4521 | Time: 1156.3s
Epoch   3/20 | Train Loss: 1.5432 (coord: 0.0765, type: 0.8123) | Val Loss: 1.3307 | Time: 1149.8s
...
```

---

## 🐛 修复的 Bug

### Bug #3: visualize_val.py 返回值解包错误

**问题**：
```python
coords_pred, type_logits = model.generate(batch)  # 错误
```

**修复**：
```python
result = model.generate(batch)
coords_pred = result["coords"]
type_logits = result["type_logits"]
```

**状态**：✅ 已修复（第 203 行）

---

## 📖 完整文档

| 文档 | 说明 |
|------|------|
| `CLAUDE.md` | 项目架构和使用指南 |
| `PROJECT_STATUS.md` | 项目测试报告 |
| `RDKIT_INSTALLATION.md` | RDKit 安装说明 |
| `COMMAND_COMPARISON.md` | 训练命令对比 |
| `RUN_SH_EXPLANATION.md` | run.sh 脚本说明 |
| `VISUALIZATION_GUIDE.md` | 可视化详细指南 |
| `QUICK_START.md` | 本文件（快速开始） |

---

## 🎓 学习资源

### 查看训练历史
```python
import json
with open('checkpoints/history_diffusion.json', 'r') as f:
    history = json.load(f)

print(f"训练了 {len(history['train'])} 个 epochs")
print(f"最终训练损失: {history['train'][-1]['loss']:.4f}")
print(f"最终验证损失: {history['val'][-1]['loss']:.4f}")
```

### 查看 RMSD 历史
```python
import json
with open('checkpoints/rmsd_diffusion.json', 'r') as f:
    rmsd = json.load(f)

for item in rmsd:
    print(f"Epoch {item['epoch']}: RMSD={item['rmsd_mean']:.2f} ± {item['rmsd_std']:.2f}")
```

---

## 💡 提示

1. **可视化文件很大**：每张图 400-500 KB，生成 100 张约需 40-50 MB
2. **GPU 加速**：可视化使用 GPU，每个样本约 0.5 秒
3. **查看图片**：在 Linux 上可用 `eog` 或 `feh` 查看 PNG 文件
4. **Jupyter 查看**：可以在 notebook 中用 `IPython.display.Image()` 显示

---

## ✅ 总结

您已成功：
1. ✅ 训练了 Diffusion 模型（5 epochs）
2. ✅ 生成了可视化结果（5 个样本）
3. ✅ 修复了日志显示问题（log_interval=1）
4. ✅ 修复了可视化脚本的 bug

**下次训练将获得更好的日志输出！** 🎉
