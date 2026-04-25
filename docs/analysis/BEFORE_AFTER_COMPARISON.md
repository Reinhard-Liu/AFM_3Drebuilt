# 训练配置修改前后对比

## 修改概览

| 配置项 | 修改前 | 修改后 | 说明 |
|--------|--------|--------|------|
| **总训练轮次** | 50 | 60 | 确保完成三个阶段 |
| **Stage 1** | 1-30 | 1-30 | 基础训练（不变） |
| **Stage 2** | 31-70 | 31-45 | 约束训练（缩短） |
| **Stage 3** | 71+ | 46-60 | 底部聚焦（提前） |
| **早停条件** | RMSD < 1.0 立即停止 | epoch >= 60 且 RMSD < 1.0 | 确保至少60轮 |

---

## 详细对比

### 1. 配置文件 (config.json)

```diff
{
    ...
-   "epochs": 50,
+   "epochs": 60,
    ...
}
```

### 2. 训练阶段划分 (src/train.py)

**函数 `get_training_stage()`:**

```diff
def get_training_stage(epoch: int) -> int:
    """Determine training stage from epoch number.

-   Stage 1 (epochs 1-30): base training
-   Stage 2 (epochs 31-70): constraint training
-   Stage 3 (epochs 71+): bottom atom focus
+   Stage 1 (epochs 1-30): base training
+   Stage 2 (epochs 31-45): constraint training
+   Stage 3 (epochs 46-60): bottom atom focus
    """
    if epoch <= 30:
        return 1
-   elif epoch <= 70:
+   elif epoch <= 45:
        return 2
    else:
        return 3
```

### 3. 早停机制 (src/train.py)

**训练主循环中的早停逻辑:**

```diff
-   if config["model_type"] == "diffusion" and rmsd_mean < 1.0:
-       print(f"[Early Stop] RMSD {rmsd_mean:.4f} < 1.0, stopping training.")
+   # Early stopping: only allow after completing at least 60 epochs
+   if config["model_type"] == "diffusion" and epoch >= 60 and rmsd_mean < 1.0:
+       print(f"[Early Stop] RMSD {rmsd_mean:.4f} < 1.0 after epoch {epoch}, stopping training.")
        early_stop = True
        break
```

---

## 实际训练效果对比

### 修改前（实际运行结果）

| 阶段 | 计划轮次 | 实际执行 | 启用功能 | 状态 |
|------|---------|---------|---------|------|
| Stage 1 | 1-30 | 1-27 | 基础训练 | ✓ 部分完成（第27轮早停） |
| Stage 2 | 31-70 | - | 物理约束 | ✗ 未执行 |
| Stage 3 | 71+ | - | 底部优化 | ✗ 未执行 |

**结果：**
- 早停轮次：第27轮
- 触发原因：RMSD 0.56 < 1.0
- 未解决问题：问题2、5、6、7（物理约束和底部优化未启用）

### 修改后（预期效果）

| 阶段 | 计划轮次 | 预期执行 | 启用功能 | 状态 |
|------|---------|---------|---------|------|
| Stage 1 | 1-30 | 1-30 | 基础训练 | ✓ 完整执行 |
| Stage 2 | 31-45 | 31-45 | 物理约束 + 环约束 | ✓ 将完整执行 |
| Stage 3 | 46-60 | 46-60 | 底部原子3x权重 | ✓ 将完整执行 |

**预期结果：**
- 最早早停轮次：第60轮
- 触发条件：完成60轮后 RMSD < 1.0
- 将解决：问题2、5、6、7全部启用

---

## 训练时长预估

| 项目 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| 实际训练轮次 | 27 | 60 | +33 轮 |
| 预估训练时长 | ~8.5小时 | ~19-22小时 | 约2.2-2.6倍 |
| Stage 2约束计算 | 无 | 15轮 | +约10%时间 |
| Stage 3底部加权 | 无 | 15轮 | +约5%时间 |

**注意：** Stage 2和Stage 3启用额外约束会增加单轮训练时间，但幅度不大（约10-15%）。

---

## 预期改进效果

根据改进方案，修改后的完整60轮训练预期将解决以下问题：

### Stage 1 (1-30轮) - 已解决

✓ **问题1**：原子数预测（Count Accuracy = 1.0）
✓ **问题3**：分子ID识别（retrieval_cid_indices字段）
✓ **问题4**：6维评估体系

### Stage 2 (31-45轮) - 将解决

⚠ → ✓ **问题2**：物理结构合理性（启用键长/键角/平面性约束）
⚠ → ✓ **问题5**：环结构刚性先验（启用环一致性损失）
⚠ → ✓ **问题6**：结构不变性约束（启用物理约束模块）

### Stage 3 (46-60轮) - 将解决

⚠ → ✓ **问题7**：底部原子精度（启用Z轴深度感知3x权重）

---

## 验证方法

执行验证脚本确认修改成功：

```bash
cd /root/autodl-tmp/micro
bash verify_modifications.sh
```

预期输出：

```
✓ epochs = 60 (>= 60)
✓ 训练阶段划分正确
✓ 早停机制已修改为 epoch >= 60
✓ 所有核心模块导入成功
✓ CLAUDE.md 已更新
✓ 项目改进方案.md 已更新
```

---

## 重新训练

如需使用新配置重新训练：

```bash
cd /root/autodl-tmp/micro

# 清理旧的checkpoints（可选）
# rm -rf checkpoints/*

# 启动训练
bash run.sh
```

训练将：
1. 执行完整60轮（除非60轮后RMSD < 1.0触发早停）
2. 在第31-45轮启用物理约束
3. 在第46-60轮启用底部原子优化
4. 生成完整的预测和可视化结果

---

## 文档更新

以下文档已同步更新：

1. `/root/autodl-tmp/CLAUDE.md` - 训练策略表格
2. `/root/autodl-tmp/项目改进方案.md` - 三阶段训练策略
3. `config.json` - epochs配置
4. `src/train.py` - 训练阶段和早停逻辑

---

## 总结

通过本次修改：

1. **确保完整性**：至少完成60轮训练，覆盖三个训练阶段
2. **启用约束**：Stage 2启用物理约束和环约束
3. **优化底部**：Stage 3启用底部原子3x权重
4. **解决问题**：预期解决改进方案中的7个问题

**关键变化**：早停机制从"RMSD < 1.0立即停止"改为"完成60轮后RMSD < 1.0才停止"，确保三阶段训练策略得以完整执行。
