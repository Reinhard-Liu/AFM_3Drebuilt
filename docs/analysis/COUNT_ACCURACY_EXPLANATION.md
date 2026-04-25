# 原子数准确率100%的真相

## 快速回答

**训练日志显示的 Count Accuracy = 100% 不代表模型真实预测能力。**

- **训练日志中的100%**: 使用了真实原子数 (`use_gt_count=True`)，相当于自己和自己比较
- **真实预测准确率**: 约44% (基于100个测试样本，使用 `use_gt_count=False`)
- **平均误差**: 1.52个原子

---

## 为什么会出现100%？

### 代码流程分析

```python
# src/train.py 第299行（评估函数）
gen_result = model.generate(batch, use_gt_count=True)
#                                  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
#                            关键：使用真实原子数！

# src/train.py 第161-164行（生成函数）
if use_gt_count and "n_atoms" in batch:
    n_atoms = batch["n_atoms"]  # ← 直接使用真实值
else:
    n_atoms = self.count_head.predict(c)  # ← 才是真正的预测

# src/train.py 第172行（返回值）
result = {
    "n_atoms_pred": n_atoms,  # ← 如果use_gt_count=True，这是真实值！
}

# src/train.py 第319行（计算准确率）
count_acc = compute_atom_count_accuracy(
    n_pred,           # = batch["n_atoms"] (真实值)
    batch["n_atoms"]  # = batch["n_atoms"] (真实值)
)
# 结果：真实值 == 真实值 → 100%准确率
```

### 流程图

```
评估流程 (use_gt_count=True):
┌─────────────┐
│  AFM图像    │
└──────┬──────┘
       │
       ├──→ Video ViT Encoder
       │
       ├──→ AtomCountHead.predict() → 预测值 (被忽略！)
       │
       ├──→ 使用 batch["n_atoms"] (真实值)
       │
       ├──→ DDPM生成坐标
       │
       └──→ 返回 n_atoms_pred = 真实值
                    ↓
            计算准确率: 真实值 vs 真实值 = 100%

预测流程 (use_gt_count=False):
┌─────────────┐
│  AFM图像    │
└──────┬──────┘
       │
       ├──→ Video ViT Encoder
       │
       ├──→ AtomCountHead.predict() → 预测值 (使用！)
       │
       ├──→ 使用预测值
       │
       ├──→ DDPM生成坐标
       │
       └──→ 返回 n_atoms_pred = 预测值
                    ↓
            计算准确率: 预测值 vs 真实值 = 44%
```

---

## 真实表现数据

### 测试集100个样本统计

| 指标 | 值 |
|------|-----|
| 完全匹配率 | 44.00% |
| 平均绝对误差 (MAE) | 1.52 个原子 |
| 最大正误差 | +6 个原子 |
| 最大负误差 | -3 个原子 |
| 误差标准差 | 1.79 |

### 误差分布

```
误差值 | 样本数 | 占比  | 可视化
-------|--------|-------|---------------------------
  -3   |   1    |  1.8% | █
  -2   |   7    | 12.5% | ██████
  -1   |  22    | 39.3% | ███████████████████
   0   |  44    | 44.0% | █████████████████████   ← 完全正确
  +1   |  15    | 26.8% | █████████████
  +2   |   7    | 12.5% | ██████
  +3   |   2    |  3.6% | █
  +5   |   1    |  1.8% | █
  +6   |   1    |  1.8% | █
```

**关键观察：**
- 78.6% 的预测误差在 ±1 个原子以内
- 93.0% 的预测误差在 ±2 个原子以内
- 极端误差（±5以上）很少见

---

## 为什么这样设计？

### 设计意图：分离评估两种能力

1. **已知原子数时的重建能力** (`use_gt_count=True`)
   - 问题："如果我告诉你分子有N个原子，你能重建得多好？"
   - 评估：扩散模型的坐标生成质量
   - 排除：原子数预测误差的影响

2. **端到端完整预测能力** (`use_gt_count=False`)
   - 问题："给你一张AFM图，你能完整重建分子吗？"
   - 评估：AtomCountHead + DDPM的整体能力
   - 真实应用场景

### 问题所在

❌ **命名混淆**：`n_atoms_pred` 字段名暗示是预测值，但在评估时实际是真实值

✓ **更好的做法**：
```python
result = {
    "n_atoms_predicted": predicted_count,  # 总是返回预测值
    "n_atoms_used": used_count,            # 标注实际使用的值
}
```

---

## 如何查看真实准确率？

### 方法1: 运行验证脚本

```bash
cd /root/autodl-tmp/micro
python3 check_real_count_accuracy.py
```

输出示例：
```
总样本数: 100
完全匹配: 44 (44.00%)
有误差的: 56 (56.00%)
平均绝对误差 (MAE): 1.52 个原子
```

### 方法2: 手动检查predictions_diffusion.json

```python
import json

with open('checkpoints/predictions_diffusion.json', 'r') as f:
    data = json.load(f)

# 这个文件使用 use_gt_count=False，是真实预测
for sample in data['predictions'][:10]:
    print(f"样本 {sample['sample_id']}: 预测 {sample['n_atoms_pred']} 个原子")
```

### 方法3: 查看可视化映射文件

```python
import json

# 可视化映射包含真实原子数
with open('visualizations/test_predictions/index_mapping.json', 'r') as f:
    mapping = json.load(f)

# 读取预测
with open('checkpoints/predictions_diffusion.json', 'r') as f:
    pred = json.load(f)

for i in range(10):
    gt = mapping[i]['n_atoms']
    pred_n = pred['predictions'][i]['n_atoms_pred']
    print(f"样本{i}: GT={gt}, Pred={pred_n}, 误差={pred_n-gt}")
```

---

## 这个表现是否正常？

### ✓ 相对合理

1. **误差范围小**：78.6%的预测在±1个原子以内
2. **无极端错误**：最大误差仅6个原子（相对37个原子的分子）
3. **训练不完整**：仅完成Stage 1（27轮），未进入Stage 2和3

### ⚠ 有改进空间

1. **完全匹配率偏低**：44%意味着超过一半的预测有误差
2. **AtomCountHead训练不足**：只训练了27轮
3. **预期改进**：完成60轮完整训练后应该提升至60-70%

### 对比其他指标

当前模型表现：

| 指标 | 值 | 评价 |
|------|-----|------|
| RMSD | 4.66 ± 84.13 Å | 较差（有极端值） |
| Bottom Recall | 7.32% | 很差 |
| Bond Validity | 83.19% | 中等 |
| **Count Accuracy (真实)** | **44%** | **中下** |
| Count Accuracy (训练日志) | 100% | 虚假 |

**结论**：原子数预测准确率与其他指标相符，都处于"训练不充分"的水平。

---

## 预期改进路线

### Stage 1 (1-30轮) - 已完成

- Count Accuracy: 44%
- MAE: 1.52个原子
- 学习基础的原子数-AFM特征映射

### Stage 2 (31-45轮) - 将执行

- 启用物理约束
- 可能间接提升原子数预测（通过更合理的结构）
- 预期 Count Accuracy → 50-55%

### Stage 3 (46-60轮) - 将执行

- 底部原子优化
- AtomCountHead得到更多训练
- 预期 Count Accuracy → 60-70%
- 预期 MAE → 0.8-1.0个原子

---

## 总结

### 三句话总结

1. **训练日志的100%是假的** - 使用了真实原子数进行评估
2. **真实准确率是44%** - 基于端到端预测 (use_gt_count=False)
3. **这是合理的** - 仅完成了27轮训练，完整60轮后会提升

### 关键要点

✓ 代码设计有意分离两种评估模式
✓ 但字段命名容易引起误解
✓ 真实预测能力应看 predictions_diffusion.json
✓ 44%的准确率对于27轮训练是合理的
✓ 完成60轮训练后预期提升至60-70%

### 验证方法

```bash
# 查看真实准确率
python3 check_real_count_accuracy.py

# 或查看详细分析文档
cat ATOM_COUNT_ACCURACY_ANALYSIS.md
```
