# 训练阶段配置示例

## 当前配置（默认）

```python
# src/train.py - get_training_stage()
if epoch <= 30:    # Stage 1: epochs 1-30
    return 1
elif epoch <= 45:  # Stage 2: epochs 31-45
    return 2
else:              # Stage 3: epochs 46-60
    return 3
```

```json
// config.json
{
  "epochs": 60
}
```

---

## 示例 1: 快速测试配置（30 轮）

适合快速验证各阶段功能是否正常。

```python
# src/train.py - get_training_stage()
if epoch <= 10:    # Stage 1: epochs 1-10
    return 1
elif epoch <= 20:  # Stage 2: epochs 11-20
    return 2
else:              # Stage 3: epochs 21-30
    return 3
```

```json
// config.json
{
  "epochs": 30
}
```

**轮次分配**: 10 + 10 + 10 = 30 轮

---

## 示例 2: 延长 Stage 1 基础训练（80 轮）

如果发现基础重建能力不足，可以延长 Stage 1。

```python
# src/train.py - get_training_stage()
if epoch <= 50:    # Stage 1: epochs 1-50 (延长基础训练)
    return 1
elif epoch <= 65:  # Stage 2: epochs 51-65
    return 2
else:              # Stage 3: epochs 66-80
    return 3
```

```json
// config.json
{
  "epochs": 80
}
```

**轮次分配**: 50 + 15 + 15 = 80 轮

---

## 示例 3: 重点约束训练（100 轮）

如果化学有效性（Bond Validity）提升缓慢，可以延长 Stage 2。

```python
# src/train.py - get_training_stage()
if epoch <= 30:    # Stage 1: epochs 1-30
    return 1
elif epoch <= 80:  # Stage 2: epochs 31-80 (延长约束训练)
    return 2
else:              # Stage 3: epochs 81-100
    return 3
```

```json
// config.json
{
  "epochs": 100
}
```

**轮次分配**: 30 + 50 + 20 = 100 轮

---

## 示例 4: 重点底部原子优化（100 轮）

如果 Bottom Recall 提升不足，可以延长 Stage 3。

```python
# src/train.py - get_training_stage()
if epoch <= 30:    # Stage 1: epochs 1-30
    return 1
elif epoch <= 50:  # Stage 2: epochs 31-50
    return 2
else:              # Stage 3: epochs 51-100 (延长底部聚焦)
    return 3
```

```json
// config.json
{
  "epochs": 100
}
```

**轮次分配**: 30 + 20 + 50 = 100 轮

---

## 示例 5: 均衡长期训练（150 轮）

均衡分配三个阶段的训练时间。

```python
# src/train.py - get_training_stage()
if epoch <= 50:    # Stage 1: epochs 1-50
    return 1
elif epoch <= 100: # Stage 2: epochs 51-100
    return 2
else:              # Stage 3: epochs 101-150
    return 3
```

```json
// config.json
{
  "epochs": 150
}
```

**轮次分配**: 50 + 50 + 50 = 150 轮

---

## 示例 6: 跳过 Stage 3（仅测试约束效果）

如果只想测试物理约束的效果，可以设置较短的 Stage 3。

```python
# src/train.py - get_training_stage()
if epoch <= 30:    # Stage 1: epochs 1-30
    return 1
elif epoch <= 55:  # Stage 2: epochs 31-55 (主要阶段)
    return 2
else:              # Stage 3: epochs 56-60 (仅验证)
    return 3
```

```json
// config.json
{
  "epochs": 60
}
```

**轮次分配**: 30 + 25 + 5 = 60 轮

---

## 如何选择配置

### 根据训练目标选择

| 目标 | 推荐配置 | 总轮次 |
|------|---------|--------|
| 快速验证功能 | 示例 1 | 30 |
| 提升基础重建 | 示例 2 | 80 |
| 提升化学有效性 | 示例 3 | 100 |
| 提升底部原子精度 | 示例 4 | 100 |
| 均衡完整训练 | 示例 5 | 150 |

### 根据观察到的问题调整

**如果观察到**:
- RMSD 下降缓慢 → 延长 Stage 1
- Bond Validity 低 → 延长 Stage 2
- Bottom Recall 低 → 延长 Stage 3

### 根据 GPU 时间预算

假设每轮 ~15 分钟（单 V100）：

| 总轮次 | 预计时长 | 适合场景 |
|--------|---------|---------|
| 30 | ~7.5 小时 | 快速测试 |
| 60 | ~15 小时 | 标准训练 |
| 100 | ~25 小时 | 充分训练 |
| 150 | ~37.5 小时 | 完整训练 |

---

## 修改后的验证

修改配置后，运行验证脚本确保边界正确：

```bash
python3 -c "
from src.train import get_training_stage

# 测试边界
epochs_to_test = [1, 30, 31, 45, 46, 60]  # 修改为您的边界
for e in epochs_to_test:
    stage = get_training_stage(e)
    print(f'Epoch {e:3d} → Stage {stage}')
"
```

预期输出（当前配置）：
```
Epoch   1 → Stage 1
Epoch  30 → Stage 1
Epoch  31 → Stage 2
Epoch  45 → Stage 2
Epoch  46 → Stage 3
Epoch  60 → Stage 3
```

---

## 注意事项

1. **确保总轮次一致**: config.json 的 epochs 应该 >= Stage 3 的最大轮次
2. **早停机制**: 早停检查要求 `epoch >= 总轮次`，需要手动调整（搜索 "early stopping"）
3. **Stage 1 至少 10 轮**: 确保基础训练有足够轮次
4. **每个阶段至少 5 轮**: 太短的阶段可能看不到效果

---

## 快速修改模板

```bash
# 1. 修改阶段边界
nano src/train.py  # 找到 get_training_stage() 函数

# 2. 修改总轮次
nano config.json   # 修改 "epochs" 字段

# 3. 验证配置
python3 verify_all_modifications.py

# 4. 开始训练
bash run.sh
```
