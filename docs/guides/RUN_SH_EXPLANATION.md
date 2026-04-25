# run.sh 脚本详细说明

## 跳过的步骤

### 📌 跳过的是：[2/3] 训练 ResNet3D Baseline 模型

在 `run.sh` 第 26-31 行：

```bash
# ---- 2. Train Baseline (3D-ResNet) ----
# To train resnet3d, change "model_type" in config.json to "resnet3d" and run again,
# or uncomment below with a separate config file:
# echo ""
# echo "[2/3] Training 3D-ResNet Baseline..."
# python3 -m src.train --config "$SCRIPT_DIR/config_resnet3d.json"
```

**所有代码都被 `#` 注释掉了，因此会被跳过。**

---

## 为什么会跳过这个步骤？

### 原因 1: 设计冲突

**原始设计意图**：
```
[1/3] 自动训练 Diffusion 模型
[2/3] 自动训练 ResNet3D 模型
[3/3] 为两个模型生成可视化对比
```

**实际问题**：
- 步骤 [1/3] 使用的是 `config.json`，由其中的 `model_type` 决定训练哪个模型
- 步骤 [2/3] 想训练另一个模型，需要一个单独的 `config_resnet3d.json`
- 但项目中**不存在** `config_resnet3d.json` 文件

### 原因 2: 避免重复训练

如果不注释掉步骤 [2/3]：
```bash
# 假设 config.json 设置 "model_type": "diffusion"

[1/3] 训练 Diffusion    # 使用 config.json
[2/3] 训练 ResNet3D     # 使用 config_resnet3d.json（不存在！）→ 报错
[3/3] 生成可视化
```

**结果**：脚本会因为找不到 `config_resnet3d.json` 而报错退出。

### 原因 3: 当前项目的使用模式

项目实际使用模式是：
```bash
# 通过修改 config.json 的 model_type 来切换模型
# 而不是同时训练两个模型
```

因此步骤 [2/3] 被注释掉，让用户**手动控制**训练哪个模型。

---

## 当前 run.sh 的实际执行流程

### 执行 `bash run.sh` 时：

```
┌─────────────────────────────────────────────────┐
│ [1/3] 训练模型                                   │
│   → 读取 config.json                            │
│   → 根据 model_type 训练 Diffusion 或 ResNet3D  │
│   → 保存到 checkpoints/                         │
└─────────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────┐
│ [2/3] ⏭️  跳过（被注释）                        │
│   → 不执行任何操作                              │
└─────────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────┐
│ [3/3] 生成可视化                                │
│   → 查找 checkpoints/history_diffusion.json     │
│   → 查找 checkpoints/history_resnet3d.json      │
│   → 如果存在，生成对应的曲线图                   │
│   → 保存到 micro/visualizations/                │
└─────────────────────────────────────────────────┘
```

---

## 对项目的影响

### ✅ 无负面影响

跳过步骤 [2/3] **不会对项目造成任何负面影响**：

| 方面 | 影响 | 说明 |
|------|------|------|
| **功能完整性** | ✅ 无影响 | 项目所有功能正常 |
| **模型训练** | ✅ 无影响 | 可以训练任意模型 |
| **对比实验** | ✅ 无影响 | 可以手动训练两个模型对比 |
| **训练效率** | ✅ 提升 | 避免重复训练不需要的模型 |
| **灵活性** | ✅ 提升 | 用户自主选择训练哪个模型 |

### 📊 实际使用场景

#### 场景 1: 只训练 Diffusion 模型

```bash
# config.json: "model_type": "diffusion"
bash run.sh

# 结果：
# ✅ [1/3] 训练 Diffusion 模型
# ⏭️  [2/3] 跳过
# ✅ [3/3] 生成 diffusion 可视化（如果存在历史文件）
```

#### 场景 2: 只训练 ResNet3D 模型

```bash
# config.json: "model_type": "resnet3d"
bash run.sh

# 结果：
# ✅ [1/3] 训练 ResNet3D 模型
# ⏭️  [2/3] 跳过
# ✅ [3/3] 生成 resnet3d 可视化（如果存在历史文件）
```

#### 场景 3: 训练两个模型进行对比

**方法 1：手动执行两次**
```bash
# 第一次：训练 Diffusion
vim config.json  # 设置 "model_type": "diffusion"
bash run.sh

# 第二次：训练 ResNet3D
vim config.json  # 设置 "model_type": "resnet3d"
bash run.sh

# 结果：
# ✅ checkpoints/best_diffusion.pt
# ✅ checkpoints/best_resnet3d.pt
# ✅ visualizations/curves_diffusion.png
# ✅ visualizations/curves_resnet3d.png
```

**方法 2：使用独立的配置文件**
```bash
# 创建两个配置文件
cp config.json config_diffusion.json
cp config.json config_resnet3d.json

# 修改各自的 model_type
vim config_diffusion.json  # "model_type": "diffusion"
vim config_resnet3d.json   # "model_type": "resnet3d"

# 分别训练
python3 -m src.train --config config_diffusion.json
python3 -m src.train --config config_resnet3d.json

# 手动生成可视化
python3 -c "
from src.utils.visualize import plot_training_curves
import os
os.makedirs('visualizations', exist_ok=True)
plot_training_curves('checkpoints/history_diffusion.json', 'visualizations/curves_diffusion.png')
plot_training_curves('checkpoints/history_resnet3d.json', 'visualizations/curves_resnet3d.png')
"
```

---

## 如果想启用步骤 [2/3]

如果您确实想让 `run.sh` 自动训练两个模型，需要以下修改：

### 步骤 1: 创建第二个配置文件

```bash
cd /root/autodl-tmp/micro
cp config.json config_resnet3d.json
```

### 步骤 2: 修改 config_resnet3d.json

```json
{
  "model_type": "resnet3d",  // 修改这里
  // 其他配置保持不变...
}
```

### 步骤 3: 取消注释 run.sh 的第 29-31 行

```bash
# 修改前（被注释）：
# echo ""
# echo "[2/3] Training 3D-ResNet Baseline..."
# python3 -m src.train --config "$SCRIPT_DIR/config_resnet3d.json"

# 修改后（取消注释）：
echo ""
echo "[2/3] Training 3D-ResNet Baseline..."
python3 -m src.train --config "$SCRIPT_DIR/config_resnet3d.json"
```

### 步骤 4: 修改第一步的描述（可选）

为了更清晰，可以修改步骤 [1/3] 的描述：

```bash
# 修改前：
echo "[1/3] Training Video ViT + Conditional Diffusion Model..."

# 修改后：
echo "[1/3] Training model from config.json..."
```

### 完整修改后的 run.sh

这样修改后，执行 `bash run.sh` 会：

1. **[1/3]** 训练 `config.json` 指定的模型（通常是 Diffusion）
2. **[2/3]** 训练 `config_resnet3d.json` 指定的 ResNet3D 模型
3. **[3/3]** 为两个模型生成可视化对比

---

## 当前建议

### ⭐ 推荐保持现状（跳过步骤 [2/3]）

**理由：**

1. ✅ **更灵活**：用户自主决定训练哪个模型
2. ✅ **避免浪费**：不会意外训练不需要的模型
3. ✅ **更清晰**：执行流程简单明了
4. ✅ **节省时间**：训练两个模型需要 15-30 小时（完整数据集）

### 如果需要对比两个模型

使用**方法 2**（独立配置文件）更好：
- 可以并行训练（如果有多个 GPU）
- 配置独立，不会互相干扰
- 更容易管理和追踪

---

## 总结

| 问题 | 答案 |
|------|------|
| **跳过的是什么？** | [2/3] 训练 ResNet3D Baseline 模型 |
| **为什么跳过？** | 避免重复训练，让用户自主选择训练哪个模型 |
| **有什么影响？** | ✅ **无负面影响**，反而提升了灵活性和效率 |
| **需要修复吗？** | ❌ **不需要**，这是合理的设计选择 |
| **如何训练两个模型？** | 手动修改 config.json 执行两次，或使用独立配置文件 |

**结论**：跳过步骤 [2/3] 是合理的设计，不会影响项目功能，反而提供了更好的灵活性。
