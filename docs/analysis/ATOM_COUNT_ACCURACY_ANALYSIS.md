# 原子数预测准确率分析报告

## 问题发现

训练日志显示 **Count Accuracy = 1.0000 (100%)**，但这个数字**不代表模型真实的预测能力**。

---

## 问题根源

### 1. 评估代码的问题

在 `src/train.py` 的 `evaluate_generation()` 函数中（第299行）：

```python
gen_result = model.generate(batch, use_gt_count=True)  # ← 使用真实原子数！
```

### 2. generate() 函数的实现

在 `src/train.py` 的 `AFM3DReconModel.generate()` 方法中（第161-164行）：

```python
# Predict atom count
if use_gt_count and "n_atoms" in batch:
    n_atoms = batch["n_atoms"]  # ← 直接使用真实原子数
else:
    n_atoms = self.count_head.predict(c)  # ← 才是真正的预测
```

### 3. 返回值的问题

第172行将使用的原子数（无论是真实的还是预测的）都赋值给 `n_atoms_pred`：

```python
result = {
    "coords": coords,
    "type_logits": type_logits,
    "n_atoms_pred": n_atoms,  # ← 如果use_gt_count=True，这里是真实值！
}
```

### 4. 准确率计算

第319行计算准确率时：

```python
n_pred = gen_result["n_atoms_pred"]  # 如果use_gt_count=True，这是真实值
count_acc = compute_atom_count_accuracy(n_pred, batch["n_atoms"])
# 相当于比较 batch["n_atoms"] 和 batch["n_atoms"]，当然是100%！
```

---

## 实际预测能力验证

### 测试集前10个样本的真实表现

使用 `use_gt_count=False` 保存的 `predictions_diffusion.json` 文件分析：

| 样本ID | 真实原子数 | 预测原子数 | 误差 | 准确? |
|--------|-----------|-----------|------|------|
| 0 | 37 | 35 | -2 | ✗ |
| 1 | 31 | 32 | +1 | ✗ |
| 2 | 21 | 21 | 0 | ✓ |
| 3 | 15 | 15 | 0 | ✓ |
| 4 | 19 | 19 | 0 | ✓ |
| 5 | 21 | 20 | -1 | ✗ |
| 6 | 22 | 22 | 0 | ✓ |
| 7 | 27 | 26 | -1 | ✗ |
| 8 | 24 | 24 | 0 | ✓ |
| 9 | 21 | 23 | +2 | ✗ |

**统计结果：**
- 总样本数: 10
- 完全匹配: 5 (50.0%)
- 有误差的: 5 (50.0%)
- 平均绝对误差 (MAE): 1.40原子
- 最大误差: 2个原子

---

## 为什么会有两种模式？

### use_gt_count=True（训练日志中的评估）

**目的**: 评估模型在**已知正确原子数**时的重建能力
- 相当于"如果我告诉你这个分子有多少个原子，你能重建得多好？"
- 排除了原子数预测误差对坐标重建的影响
- 用于评估扩散模型的纯重建能力

**位置**: `evaluate_generation()` 函数（第299行）

### use_gt_count=False（predictions_diffusion.json）

**目的**: 评估模型的**端到端完整能力**
- 包括原子数预测 + 坐标重建
- 这才是真实应用场景
- 用于保存最终预测结果

**位置**: `save_predictions()` 函数（第383行）

---

## 问题的影响

### 1. 训练日志中的"Count Accuracy = 1.0"是误导性的

这个100%的准确率是**虚假的**，因为：
- 使用了真实原子数进行生成
- 然后比较"真实原子数"和"真实原子数"
- 相当于自己和自己比较

### 2. 实际预测准确率

根据 `predictions_diffusion.json` 的分析：
- **真实准确率约50%**（前10个样本）
- 平均误差约1.4个原子
- 大部分误差在±2个原子范围内

### 3. 这是一个命名问题

`n_atoms_pred` 这个字段名具有误导性：
- 在 `use_gt_count=True` 时，它不是"预测"，而是"使用的真实值"
- 在 `use_gt_count=False` 时，它才是真正的"预测值"

---

## 如何修正这个问题

### 方案1：修改evaluate_generation()函数

```python
@torch.no_grad()
def evaluate_generation(model, loader, device, num_samples: int = 50):
    # ...

    # 方案1A: 改为使用预测的原子数（更真实）
    gen_result = model.generate(batch, use_gt_count=False)

    # 或方案1B: 同时评估两种模式
    gen_gt = model.generate(batch, use_gt_count=True)
    gen_pred = model.generate(batch, use_gt_count=False)

    # 分别计算准确率
    count_acc_with_gt = compute_atom_count_accuracy(
        gen_gt["n_atoms_pred"], batch["n_atoms"]
    )  # 这个会是100%，但没意义

    count_acc_real = compute_atom_count_accuracy(
        gen_pred["n_atoms_pred"], batch["n_atoms"]
    )  # 这才是真实能力
```

### 方案2：修改generate()函数的返回值

```python
def generate(self, batch: dict, use_gt_count: bool = False) -> dict:
    afm = batch["afm_stack"]
    c = self.encoder(afm)

    # 总是预测原子数
    n_atoms_predicted = self.count_head.predict(c)

    # 根据标志选择使用哪个
    if use_gt_count and "n_atoms" in batch:
        n_atoms_used = batch["n_atoms"]
    else:
        n_atoms_used = n_atoms_predicted

    coords, type_logits = self.ddpm.sample(c, n_atoms_used, max_atoms=MAX_ATOMS)

    return {
        "coords": coords,
        "type_logits": type_logits,
        "n_atoms_pred": n_atoms_predicted,  # 总是返回预测值
        "n_atoms_used": n_atoms_used,       # 标注实际使用的值
    }
```

### 方案3：添加两个独立的评估指标

```python
# 1. 原子数预测准确率（AtomCountHead的能力）
count_prediction_accuracy = compute_atom_count_accuracy(
    model.count_head.predict(condition_vector),
    batch["n_atoms"]
)

# 2. 完整生成的RMSD（在已知原子数时的重建能力）
reconstruction_rmsd = evaluate_with_gt_count(...)
```

---

## 建议

### 短期（不破坏现有代码）

1. **在文档中明确说明**：
   - 训练日志中的 "Count Accuracy" 使用了 `use_gt_count=True`
   - 这个指标评估的是"已知原子数时的重建能力"
   - 真实的原子数预测准确率需要查看 `predictions_diffusion.json`

2. **添加额外的评估指标**：
   - 在训练日志中同时输出两种模式的结果
   - 明确标注哪个使用了真实原子数，哪个是端到端预测

### 长期（改进代码）

1. 修改 `evaluate_generation()` 使用 `use_gt_count=False`
2. 修改 `generate()` 的返回值结构，区分预测值和使用值
3. 分离两个评估目标：
   - AtomCountHead的准确率
   - 完整生成流程的RMSD

---

## 结论

**训练日志中显示的 Count Accuracy = 1.0000 (100%) 不是错误，但也不代表模型的真实预测能力。**

- **100%准确率的含义**: 在已知正确原子数时，模型能正确使用这个数字进行生成
- **实际预测能力**: 约50%的完全匹配率，平均误差约1.4个原子
- **这是正常的**: 因为评估代码设计为测试"已知原子数时的重建能力"，而不是"端到端预测能力"

如需了解模型的真实原子数预测能力，应该：
1. 查看 `predictions_diffusion.json` 文件
2. 与测试集的真实原子数比较
3. 计算真实的准确率和MAE

**当前模型在前10个测试样本上的真实表现：**
- 完全匹配率: 50%
- 平均绝对误差: 1.4个原子
- 误差范围: -2 到 +2个原子

这个表现是**相对合理的**，但仍有改进空间。完成60轮完整训练后，预期这个准确率会提升。
