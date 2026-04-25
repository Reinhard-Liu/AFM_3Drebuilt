# 训练命令对比说明

## 两条命令的区别

### 命令 1: `python3 -m src.train --config config.json`
**直接训练命令**

### 命令 2: `bash run.sh`
**脚本封装命令**

---

## 详细对比

### 1. `python3 -m src.train --config config.json`

#### 执行内容
```
只执行一次训练 → 根据 config.json 中的 model_type 训练对应模型
```

#### 流程
1. 读取 `config.json`
2. 根据 `"model_type"` 字段决定训练哪个模型：
   - `"diffusion"` → 训练 Video ViT + Conditional DDPM
   - `"resnet3d"` → 训练 3D-ResNet Baseline
3. 训练完成，保存检查点到 `checkpoints/` 目录
4. 保存训练历史到 `history_{model_type}.json`
5. 保存 RMSD 评估结果到 `rmsd_{model_type}.json`
6. **结束**

#### 输出文件
```
checkpoints/
├── best_{model_type}.pt          # 最优模型
├── history_{model_type}.json     # 训练历史
└── rmsd_{model_type}.json        # RMSD 评估结果
```

#### 优点
- ✅ 简单直接
- ✅ 可以精确控制训练哪个模型
- ✅ 适合调试和实验
- ✅ 不会重复训练

#### 适用场景
- 只想训练一个模型
- 测试超参数
- 快速实验
- 断点续训

---

### 2. `bash run.sh`

#### 当前执行内容（存在问题）
```bash
[1/3] 训练模型（根据 config.json 的 model_type）
[2/3] 跳过（已注释）
[3/3] 生成可视化曲线（⚠️ 存在 bug）
```

#### 当前流程
1. 设置环境变量 `PYTHONPATH`
2. **[1/3]** 执行 `python3 -m src.train --config config.json`
3. **[2/3]** 已注释掉，不执行
4. **[3/3]** 尝试生成可视化
   - ⚠️ **问题**：使用了未定义的 `$SAVE_DIR` 变量
   - 实际上这部分会失败

#### 当前存在的问题

**Bug #1: `$SAVE_DIR` 变量未定义**

`run.sh` 第 40 行：
```bash
save_dir = '$SAVE_DIR'  # ⚠️ $SAVE_DIR 未定义，会是空字符串
```

这会导致可视化脚本找不到正确的文件路径。

**Bug #2: 脚本设计不完整**

原始设计意图是：
1. 先训练 Diffusion 模型
2. 再训练 ResNet3D 模型
3. 对两个模型都生成可视化

但目前第 2 步被注释掉了，第 3 步有 bug。

#### 原始设计流程（如果完全实现）
```
[1/3] 训练 Diffusion 模型
       ↓
[2/3] 训练 ResNet3D 模型（需要额外配置）
       ↓
[3/3] 为两个模型生成训练曲线可视化
       ↓
输出：checkpoints/ + visualizations/
```

---

## 实际效果对比

### 场景 1: config.json 设置 `"model_type": "diffusion"`

| 命令 | 训练 Diffusion | 训练 ResNet3D | 生成可视化 | 结果 |
|------|----------------|---------------|------------|------|
| `python3 -m src.train` | ✅ | ❌ | ❌ | 只训练 Diffusion，无可视化 |
| `bash run.sh` | ✅ | ❌ | ⚠️ 失败 | 只训练 Diffusion，可视化失败 |

### 场景 2: config.json 设置 `"model_type": "resnet3d"`

| 命令 | 训练 Diffusion | 训练 ResNet3D | 生成可视化 | 结果 |
|------|----------------|---------------|------------|------|
| `python3 -m src.train` | ❌ | ✅ | ❌ | 只训练 ResNet3D，无可视化 |
| `bash run.sh` | ❌ | ✅ | ⚠️ 失败 | 只训练 ResNet3D，可视化失败 |

---

## 推荐使用方式

### 方式 1: 直接使用 Python 命令（推荐）

**训练 Diffusion 模型：**
```bash
# 1. 修改 config.json
vim config.json  # 设置 "model_type": "diffusion"

# 2. 训练
python3 -m src.train --config config.json
```

**训练 ResNet3D 模型：**
```bash
# 1. 修改 config.json
vim config.json  # 设置 "model_type": "resnet3d"

# 2. 训练
python3 -m src.train --config config.json
```

**生成可视化（训练后）：**
```bash
python3 -m src.utils.visualize
# 或使用可视化脚本：
python3 -m src.visualize_val --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 100 --output_dir visualizations/
```

### 方式 2: 修复后的 run.sh（需要修改）

如果您希望使用 `run.sh` 来自动化训练两个模型，需要修复脚本。

---

## 当前建议

### ✅ 推荐：使用 `python3 -m src.train --config config.json`

**理由：**
1. 更可控：明确知道训练哪个模型
2. 无 bug：不依赖未定义的变量
3. 更灵活：可以随时切换模型类型
4. 更清晰：输出直接，易于调试

### 使用步骤

**步骤 1: 选择模型**

编辑 `config.json`，修改 `model_type`：
```json
{
  "model_type": "diffusion"   // 或 "resnet3d"
}
```

**步骤 2: 开始训练**
```bash
cd /root/autodl-tmp/micro
python3 -m src.train --config config.json
```

**步骤 3: 查看结果**
```bash
# 检查点
ls checkpoints/

# 训练历史
cat checkpoints/history_diffusion.json  # 或 history_resnet3d.json

# RMSD 结果
cat checkpoints/rmsd_diffusion.json
```

---

## 如何训练两个模型

如果您想训练两个模型进行对比：

### 方法 1: 手动切换（推荐）

```bash
# 1. 训练 Diffusion 模型
# 修改 config.json: "model_type": "diffusion"
python3 -m src.train --config config.json

# 2. 训练 ResNet3D 模型
# 修改 config.json: "model_type": "resnet3d"
python3 -m src.train --config config.json

# 3. 对比结果
ls checkpoints/
# best_diffusion.pt, history_diffusion.json
# best_resnet3d.pt, history_resnet3d.json
```

### 方法 2: 使用两个配置文件

```bash
# 创建两个配置文件
cp config.json config_diffusion.json
cp config.json config_resnet3d.json

# 修改 config_diffusion.json: "model_type": "diffusion"
# 修改 config_resnet3d.json: "model_type": "resnet3d"

# 分别训练
python3 -m src.train --config config_diffusion.json
python3 -m src.train --config config_resnet3d.json
```

---

## 总结

| 对比项 | `python3 -m src.train` | `bash run.sh` |
|--------|------------------------|---------------|
| **功能** | 训练单个模型 | 训练单个模型 + 尝试可视化 |
| **是否有 bug** | ✅ 无 | ⚠️ 有（$SAVE_DIR 未定义）|
| **可控性** | ✅ 高 | ⚠️ 低 |
| **适用场景** | 所有场景 | 无（需修复）|
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐ |

**结论**：目前推荐使用 `python3 -m src.train --config config.json`，这是最可靠和灵活的方式。
