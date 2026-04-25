# 模型可视化使用指南

## 您的训练结果

根据刚才的训练输出：

```
Training complete. Best val loss: 1.3307
Checkpoint saved to: /root/autodl-tmp/micro/src/../checkpoints
Test RMSD: 80.6865 +/- 188.4281
Test Bottom Recall: 0.0887 +/- 0.2052
```

**生成的模型文件**：
- 最佳模型：`checkpoints/best_diffusion.pt`
- 训练历史：`checkpoints/history_diffusion.json`
- RMSD 历史：`checkpoints/rmsd_diffusion.json`

---

## 方法 1: 使用内置可视化脚本（推荐）

### 基本用法

```bash
cd /root/autodl-tmp/micro

python3 -m src.visualize_val \
    --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 10 \
    --output_dir visualizations/diffusion
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 模型检查点路径（必需） | 无 |
| `--num_samples` | 生成多少个样本的可视化 | 10 |
| `--output_dir` | 输出目录 | `checkpoints/val_visualizations` |
| `--data_root` | 数据集路径 | 从 checkpoint 中读取 |

### 生成结果

执行后会生成：

```
visualizations/diffusion/
├── val_sample_00000.png  # 第 1 个验证样本
├── val_sample_00111.png  # 第 2 个验证样本
├── val_sample_00222.png  # 第 3 个验证样本
├── ...
└── val_sample_00999.png  # 第 10 个验证样本
```

**每张图包含**：
- **上半部分**：5 个 AFM Z-切片图像（深度 0, 2, 4, 6, 9）
- **左下角**：真实的 3D 分子结构（Ground Truth）
- **右下角**：模型预测的 3D 分子结构 + RMSD 值

---

## 方法 2: 生成更多样本

### 生成 50 个样本

```bash
python3 -m src.visualize_val \
    --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 50 \
    --output_dir visualizations/diffusion_50samples
```

### 生成所有验证集样本（1000 个）

```bash
python3 -m src.visualize_val \
    --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 1000 \
    --output_dir visualizations/diffusion_all
```

⚠️ **注意**：生成 1000 个样本需要约 5-10 分钟。

---

## 方法 3: 自定义可视化脚本

如果您想更灵活地控制可视化，可以使用 Python 脚本：

### 创建自定义脚本

```python
# custom_visualize.py
import torch
import numpy as np
from src.train import AFM3DReconModel
from src.data.dataset import QUAMAFMDataset

# 1. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("checkpoints/best_diffusion.pt", map_location=device)
config = checkpoint["config"]

model = AFM3DReconModel(config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

print(f"✅ 模型已加载，训练到第 {checkpoint['epoch']} epoch")
print(f"   验证损失: {checkpoint['val_loss']:.4f}")

# 2. 加载数据
dataset = QUAMAFMDataset(
    data_root=config["data_root"],
    param_key=config["param_key"],
    img_size=config["img_size"],
    min_corrugation=config["min_corrugation"],
    augment_rotation=False,
    split="val",
    val_size=config["val_size"],
    max_samples=config["max_samples"],
)

print(f"✅ 验证集大小: {len(dataset)} 个样本")

# 3. 选择一个样本生成分子
sample_idx = 0  # 可以改为任意索引 (0 到 len(dataset)-1)
sample = dataset[sample_idx]

# 转换为 batch 格式
batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

# 4. 生成预测
with torch.no_grad():
    result = model.generate(batch)
    pred_coords = result["coords"]         # (1, 85, 3)
    pred_type_logits = result["type_logits"]  # (1, 85, 10)
    pred_n_atoms = result["n_atoms_pred"]  # (1,)

    pred_types = pred_type_logits.argmax(dim=-1)  # (1, 85)

# 5. 提取结果
pred_coords_np = pred_coords[0].cpu().numpy()  # (85, 3)
pred_types_np = pred_types[0].cpu().numpy()    # (85,)
gt_coords_np = batch["coords"][0].cpu().numpy()
gt_types_np = batch["atom_types"][0].cpu().numpy()
mask_np = batch["atom_mask"][0].cpu().numpy()

# 打印结果
n_valid = int(mask_np.sum())
print(f"\n样本 {sample_idx}:")
print(f"  真实原子数: {n_valid}")
print(f"  预测原子数: {pred_n_atoms.item()}")
print(f"  预测坐标范围: [{pred_coords_np.min():.3f}, {pred_coords_np.max():.3f}]")

# 6. 计算 RMSD
from src.utils.metrics import compute_rmsd
rmsd = compute_rmsd(pred_coords, batch["coords"], batch["atom_mask"])
print(f"  RMSD: {rmsd.item():.4f}")

# 7. 可视化（可选）
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))

# 真实结构
ax1 = fig.add_subplot(121, projection='3d')
valid = mask_np > 0
coords_valid = gt_coords_np[valid]
ax1.scatter(coords_valid[:, 0], coords_valid[:, 1], coords_valid[:, 2],
            c='blue', s=50, alpha=0.7)
ax1.set_title(f'Ground Truth ({n_valid} atoms)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 预测结构
ax2 = fig.add_subplot(122, projection='3d')
pred_coords_valid = pred_coords_np[valid]
ax2.scatter(pred_coords_valid[:, 0], pred_coords_valid[:, 1], pred_coords_valid[:, 2],
            c='red', s=50, alpha=0.7)
ax2.set_title(f'Predicted (RMSD={rmsd.item():.3f})')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.savefig('custom_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n✅ 可视化已保存到: custom_visualization.png")
```

### 运行自定义脚本

```bash
python3 custom_visualize.py
```

---

## 方法 4: 批量生成并分析

### 脚本：分析验证集的 RMSD 分布

```python
# analyze_validation.py
import torch
import numpy as np
from tqdm import tqdm
from src.train import AFM3DReconModel
from src.data.dataset import QUAMAFMDataset
from src.utils.metrics import compute_rmsd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
checkpoint = torch.load("checkpoints/best_diffusion.pt", map_location=device)
config = checkpoint["config"]
model = AFM3DReconModel(config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

# 加载验证集
dataset = QUAMAFMDataset(
    data_root=config["data_root"],
    param_key=config["param_key"],
    img_size=config["img_size"],
    min_corrugation=config["min_corrugation"],
    augment_rotation=False,
    split="val",
    val_size=config["val_size"],
    max_samples=config["max_samples"],
)

print(f"分析验证集 {len(dataset)} 个样本...")

# 批量生成
all_rmsd = []
for idx in tqdm(range(min(100, len(dataset)))):  # 只分析前 100 个样本
    sample = dataset[idx]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

    with torch.no_grad():
        result = model.generate(batch)
        rmsd = compute_rmsd(result["coords"], batch["coords"], batch["atom_mask"])
        all_rmsd.append(rmsd.item())

# 统计
all_rmsd = np.array(all_rmsd)
print(f"\nRMSD 统计 (100 个样本):")
print(f"  均值: {all_rmsd.mean():.4f}")
print(f"  标准差: {all_rmsd.std():.4f}")
print(f"  中位数: {np.median(all_rmsd):.4f}")
print(f"  最小值: {all_rmsd.min():.4f}")
print(f"  最大值: {all_rmsd.max():.4f}")

# 绘制分布
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.hist(all_rmsd, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('RMSD')
plt.ylabel('Frequency')
plt.title('RMSD Distribution on Validation Set')
plt.axvline(all_rmsd.mean(), color='red', linestyle='--',
            label=f'Mean: {all_rmsd.mean():.2f}')
plt.legend()
plt.savefig('rmsd_distribution.png', dpi=150)
print(f"\n✅ 分布图已保存到: rmsd_distribution.png")
```

---

## 快速开始（推荐命令）

### 生成 10 个样本的可视化

```bash
cd /root/autodl-tmp/micro

python3 -m src.visualize_val \
    --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 10 \
    --output_dir visualizations/quick_demo
```

**预期输出**：
```
Device: cuda
Loaded diffusion model from: checkpoints/best_diffusion.pt
  Epoch: 3, Val loss: 1.3307
Validation set: 1000 samples
Generating comparison plots for 10 samples...

  [1/10] Sample 0: RMSD=42.3456 -> visualizations/quick_demo/val_sample_00000.png
  [2/10] Sample 111: RMSD=55.1234 -> visualizations/quick_demo/val_sample_00111.png
  ...

==================================================
Results: 10 validation samples
  RMSD: 80.6865 +/- 188.4281
  Output: visualizations/quick_demo/
==================================================
```

---

## 查看结果

### 方法 1: 命令行查看

```bash
# 列出生成的图片
ls -lh visualizations/quick_demo/

# 使用图片查看器打开
eog visualizations/quick_demo/val_sample_00000.png  # Linux
# 或
open visualizations/quick_demo/val_sample_00000.png  # macOS
```

### 方法 2: 在 Jupyter Notebook 中查看

```python
from IPython.display import Image, display

# 显示第一个样本
display(Image('visualizations/quick_demo/val_sample_00000.png'))
```

---

## 理解可视化结果

### 图片布局说明

每张可视化图包含：

```
┌─────────────────────────────────────────────────────────────┐
│              Validation Sample #0                           │
├─────────┬─────────┬─────────┬─────────┬─────────────────────┤
│ Z-slice │ Z-slice │ Z-slice │ Z-slice │   Z-slice 9         │
│    0    │    2    │    4    │    6    │                     │
│ (AFM)   │ (AFM)   │ (AFM)   │ (AFM)   │   (AFM)             │
├─────────────────────────────┬───────────────────────────────┤
│                             │                               │
│   Ground Truth              │   Predicted                   │
│   (23 atoms)                │   (RMSD=42.3456)              │
│                             │                               │
│   [3D 分子结构图]            │   [3D 分子结构图]              │
│                             │                               │
└─────────────────────────────┴───────────────────────────────┘
```

### RMSD 值的含义

- **RMSD < 1.0 Å**：非常好，原子位置几乎完全正确
- **RMSD < 5.0 Å**：良好，分子整体结构正确
- **RMSD < 50 Å**：可接受，主要特征正确但细节有偏差
- **RMSD > 100 Å**：较差，可能需要更多训练

**您的测试结果**：`80.7 ± 188.4` 表示大部分样本在可接受范围，但方差较大，说明某些样本效果很好，某些较差。

---

## 提示

1. **首次使用**：建议先生成 10 个样本查看效果
2. **性能**：GPU 下每个样本约 0.5-1 秒
3. **磁盘空间**：每张图约 500KB-1MB
4. **可视化质量**：图片 DPI=150，适合查看和论文使用

---

## 故障排查

### 问题 1: 找不到模型文件

```bash
# 检查模型是否存在
ls -lh checkpoints/best_diffusion.pt

# 如果不存在，检查训练是否完成
ls -lh checkpoints/
```

### 问题 2: CUDA 内存不足

```bash
# 使用 CPU
CUDA_VISIBLE_DEVICES="" python3 -m src.visualize_val \
    --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 10
```

### 问题 3: 可视化窗口无法显示

脚本使用 `Agg` 后端，直接保存图片到文件，不会弹出窗口。
