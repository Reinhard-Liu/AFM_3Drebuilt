# curves_diffusion.png 曲线说明

## 快速回答

**curves_diffusion.png 中的 loss 曲线是基于训练损失（Training Loss），而不是 RMSD。**

- RMSD 是**评估指标**（用于测试模型性能）
- Loss 是**训练目标**（用于优化模型参数）

---

## 详细说明

### 1. 图片内容

`curves_diffusion.png` 包含 **3 个子图**：

| 子图位置 | 标题 | Y轴 | 说明 |
|---------|------|-----|------|
| 左 | **Total Loss** | Total Loss | 总损失（所有损失的加权和） |
| 中 | **Coordinate Loss** | Coord Loss (MSE) | 坐标损失（均方误差） |
| 右 | **Atom Type Loss** | Type Loss (CE) | 原子类型损失（交叉熵） |

每个子图都包含两条曲线：
- 🔵 **Train** - 训练集损失
- 🟠 **Val** - 验证集损失

### 2. 数据来源

**读取文件：** `checkpoints/history_diffusion.json`

**文件内容：**
```json
{
  "train": [
    {
      "loss": 2.7363,           // 总损失
      "coord_loss": 0.1434,     // 坐标损失
      "type_loss": 1.2591,      // 类型损失
      "count_loss": 3.7059,     // 原子数损失
      "retrieval_loss": 12.2814 // 检索损失
    },
    ...
  ],
  "val": [...]
}
```

### 3. 生成代码

**位置：** `src/utils/visualize.py:109-159`

**核心逻辑：**
```python
def plot_training_curves(history_path: str, save_path: str = None):
    """Plot training and validation loss curves."""
    with open(history_path, "r") as f:
        history = json.load(f)

    # 提取损失数据
    train_loss = [m["loss"] for m in history["train"]]      # 总损失
    val_loss = [m["loss"] for m in history["val"]]
    train_coord = [m["coord_loss"] for m in history["train"]]  # 坐标损失
    val_coord = [m["coord_loss"] for m in history["val"]]
    train_type = [m["type_loss"] for m in history["train"]]    # 类型损失
    val_type = [m["type_loss"] for m in history["val"]]

    # 绘制3个子图
    # 1. Total Loss
    # 2. Coordinate Loss (MSE)
    # 3. Atom Type Loss (CE)
```

**调用位置：** `run.sh:48`
```bash
plot_training_curves(hist, os.path.join(vis_dir, f'curves_{model_type}.png'))
```

---

## 4. Loss 的组成

### 总损失（Total Loss）计算公式

在 `src/train.py:142-147` 中定义：

```python
losses["loss"] = (
    losses["coord_loss"]        # 坐标损失（权重 1.0）
    + 0.1 * losses["type_loss"]     # 类型损失（权重 0.1）
    + 0.5 * losses["count_loss"]    # 原子数损失（权重 0.5）
    + 0.05 * losses["retrieval_loss"]  # 检索损失（权重 0.05）
)
```

**各部分说明：**

| 损失类型 | 权重 | 计算方式 | 说明 |
|---------|------|---------|------|
| `coord_loss` | 1.0 | MSE（均方误差） | 预测坐标与真实坐标的差异 |
| `type_loss` | 0.1 | CE（交叉熵） | 预测原子类型的分类损失 |
| `count_loss` | 0.5 | 分类CE + 回归L1 | 原子数预测损失 |
| `retrieval_loss` | 0.05 | InfoNCE（对比学习） | 分子检索对比损失 |

### 各损失的具体含义

#### (1) Coordinate Loss (坐标损失)

- **目的**: 最小化预测坐标和真实坐标之间的距离
- **计算**: 扩散模型预测的噪声与真实噪声的MSE
- **影响**: 直接影响3D重建的几何精度

#### (2) Atom Type Loss (原子类型损失)

- **目的**: 正确预测每个原子的类型（H, C, N, O等）
- **计算**: 分类交叉熵损失
- **影响**: 影响原子种类的识别准确率

#### (3) Count Loss (原子数损失)

- **目的**: 预测分子的原子总数
- **计算**: 分类分支（85类）+ 回归分支
- **权重**: 0.5（较高，因为原子数很重要）

#### (4) Retrieval Loss (检索损失)

- **目的**: 学习将AFM图像映射到分子嵌入空间，用于检索相似分子
- **计算**: InfoNCE对比学习损失
- **权重**: 0.05（较低，辅助任务）

---

## 5. Loss vs RMSD 的区别

### Loss（训练损失）

**作用：** 训练过程中优化的目标函数

**特点：**
- 在训练过程中计算（每个batch）
- 用于梯度反向传播
- 越小越好，但不直接代表模型性能
- 可能包含正则化项、辅助损失等

**curves_diffusion.png 显示的就是这个**

### RMSD（评估指标）

**作用：** 评估模型重建质量的几何指标

**特点：**
- 在验证/测试时计算（完整生成后）
- 不参与训练优化
- 直接反映重建精度（单位：埃）
- 使用匈牙利匹配对齐原子

**计算方式：**
```python
# 1. 生成完整的分子结构
coords_pred = model.generate(afm_image)

# 2. 使用匈牙利算法匹配原子
cost_matrix = pairwise_distance(coords_pred, coords_gt)
row_ind, col_ind = hungarian_matching(cost_matrix)

# 3. 计算匹配后的RMSD
rmsd = sqrt(mean((coords_pred[row_ind] - coords_gt[col_ind])**2))
```

### 对比表

| 项目 | Loss | RMSD |
|------|------|------|
| 用途 | 训练优化目标 | 性能评估指标 |
| 计算时机 | 每个batch | 验证/测试时 |
| 梯度反传 | ✓ 是 | ✗ 否 |
| 单位 | 无量纲 | 埃 (Å) |
| 是否直观 | 不直观 | 直观（几何距离） |
| curves_diffusion.png | ✓ 显示 | ✗ 不显示 |

---

## 6. 如何查看 RMSD 曲线？

RMSD 数据保存在 `checkpoints/metrics_diffusion.json` 中。

### 方法1: 读取metrics文件

```python
import json
import matplotlib.pyplot as plt

with open('checkpoints/metrics_diffusion.json', 'r') as f:
    metrics = json.load(f)

epochs = [m['epoch'] for m in metrics]
rmsd_mean = [m['rmsd_mean'] for m in metrics]

plt.plot(epochs, rmsd_mean)
plt.xlabel('Epoch')
plt.ylabel('RMSD (Å)')
plt.title('RMSD vs Epoch')
plt.grid(True)
plt.savefig('rmsd_curve.png')
```

### 方法2: 查看训练日志

```bash
grep "RMSD:" checkpoints/training.log
```

输出示例：
```
[Epoch 1] RMSD: 83.4250 +/- 934.4512
[Epoch 2] RMSD: 59.6091 +/- 670.9166
...
[Epoch 27] RMSD: 0.5642 +/- 8.5925
```

### RMSD 数据内容

`metrics_diffusion.json` 包含完整的评估指标：

```json
[
  {
    "epoch": 1,
    "rmsd_mean": 83.425,
    "rmsd_std": 934.451,
    "bottom_recall_mean": 0.0843,
    "bottom_rmsd_mean": 10.987,
    "bond_validity_mean": 0.7654,
    "count_exact_match": 1.0000,
    "count_mae": 0.0000,
    "composite_score": 0.1234
  },
  ...
]
```

---

## 7. 典型的训练曲线模式

### 正常训练模式

从 `history_diffusion.json` 可以看到的典型模式：

```
Epoch 1:  Train Loss = 2.74,  Val Loss = 1.63  (初始阶段，损失较高)
Epoch 10: Train Loss = 1.20,  Val Loss = 1.09  (快速下降)
Epoch 20: Train Loss = 1.08,  Val Loss = 1.07  (逐渐收敛)
Epoch 27: Train Loss = 1.05,  Val Loss = 1.09  (趋于稳定)
```

**观察要点：**
- Train Loss 持续下降 → 模型在学习
- Val Loss 也下降 → 没有过拟合
- Train/Val 差距小 → 泛化良好
- 后期波动小 → 收敛稳定

### 异常模式

❌ **过拟合**: Train Loss ↓ 但 Val Loss ↑
❌ **欠拟合**: Train Loss 和 Val Loss 都很高
❌ **不收敛**: 损失震荡剧烈

---

## 8. 总结

### curves_diffusion.png 显示的是：

✓ **训练损失**（Training Loss）
✓ 3个子图：Total Loss, Coord Loss, Type Loss
✓ 数据来源：`history_diffusion.json`
✓ 目的：监控训练过程，判断模型是否收敛

### curves_diffusion.png 不显示：

✗ RMSD（评估指标）
✗ Bottom Recall（底部原子召回率）
✗ Bond Validity（键有效率）
✗ 其他性能指标

### 如果想看 RMSD 曲线：

需要自己从 `metrics_diffusion.json` 生成，或查看训练日志。

---

## 9. 快速验证

```bash
# 查看curves_diffusion.png
open visualizations/curves_diffusion.png

# 查看history数据
cat checkpoints/history_diffusion.json | jq '.'

# 查看RMSD数据
cat checkpoints/metrics_diffusion.json | jq '.'

# 查看训练日志中的RMSD
grep "RMSD:" checkpoints/training.log
```

---

**结论：curves_diffusion.png 中的曲线是基于训练损失（Loss），而不是 RMSD。**

RMSD 是性能评估指标，不直接参与训练优化，需要从 `metrics_diffusion.json` 或训练日志中查看。
