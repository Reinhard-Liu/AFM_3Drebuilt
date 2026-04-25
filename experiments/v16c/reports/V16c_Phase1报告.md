# V16c Phase 1 评估报告

## 配置
- Checkpoint: `experiments/v16b/checkpoints/best_diffusion.pt` (epoch 30)
- 采样器: DDIM-50（V16c 修正公式）
- 原子数: **GT count**
- Physics guidance: **关闭**
- Ring snap: **关闭**
- Shape guidance: **关闭**
- 未重训，直接用 V16b 现有权重

## 1. 核心指标对比

| 指标 | V16b (旧采样器) | V16c Phase1 (修正采样器) | 变化 |
|------|----------------|------------------------|------|
| **RMSD** | 0.4546 | **0.2034** | **-55.2%** |
| **Bond Validity** | 0.2204 | **0.4378** | **+98.6%** |
| Count Exact | 0.3867 | **1.0000** | (用了GT count) |
| Count MAE | 0.7734 | **0.0000** | (用了GT count) |
| Ring Preservation | 0.8154 | **0.8555** | +4.9% |
| PMI Shape Sim | 0.8163 | 0.7877 | -3.5% |
| type_match (raw) | 0.4307 | 0.2874 | -33.3% |
| conditional_type_acc | 0.4284 | 0.3199 | -25.3% |

## 2. 坍缩诊断

| 指标 | V16b | V16c | GT |
|------|------|------|----|
| Median pair distance | ~0.05 | **~0.20** | ~0.37 |
| Fraction(pair < 0.05) | ~0.80 | **~0.18** | 0.00 |
| Pred/GT spatial ratio | ~0.13 | **~0.55** | 1.00 |
| Z 轴范围 | ~0.02 | **~0.015** | ~0.15 |

### 关键发现
1. **大规模坍缩已解除**：V16b 的 median pair distance ~0.05，V16c 恢复到 ~0.20（GT 的 55%）
2. **XY 平面展布恢复**：预测结构在 XY 方向的空间范围从 V16b 的 ~0.1 恢复到 ~0.27（GT ~0.7 的 38%）
3. **Z 轴仍然严重压扁**：预测 Z 范围 ~0.015，GT Z 范围 ~0.15，压缩比 ~10:1
4. **仍有 ~18% 原子对距离 < 0.05**（GT 为 0%），说明局部仍有聚团

## 3. 可视化分析

从 5-molecule 对比图可以清楚看到：
- **Predicted 结构不再是一团**，已经展开为可辨识的分子形状
- **XY 平面内的整体拓扑大致正确**（与 GT 和 Top-1/2/3 有相似的空间分布）
- **Z 轴方向几乎完全扁平**：所有原子被压在一个 ~0.015 厚的薄层上
- **原子类型准确率下降**：这可能是因为 V16b 的 denoiser 在错误采样器下训练，类型头适配了错误的输入分布

## 4. Type Match 下降的分析

type_match 从 0.43 降到 0.29，可能原因：
1. **V16b 的 type_match 是虚高的**：在坍缩状态下，匈牙利匹配会将密集团中的原子随机分配到 GT 原子，碰巧有 ~40% 类型一致
2. **V16c 的结构展开后，匹配更精确**，但类型预测本身可能不准确——denoiser 的类型头是在错误采样器下训练的
3. 需要重训后才能真正评估类型能力

## 5. 结论

### 修采样器后是否不再大规模坍缩？
**是的，大规模坍缩已解除。** Median pair distance 从 ~0.05 恢复到 ~0.20（GT 的 55%），结构从"一个点"恢复为可辨识的分子形状。

### 当前 denoiser 权重是否已失效？
**部分失效。** 具体表现：
- **XY 坐标重建能力存在**（RMSD 0.20 比 V16b 的 0.45 好很多）
- **Z 轴重建能力几乎没有**（Z 范围被压缩 10 倍）
- **类型预测能力存疑**（type_match 下降）

### 是否需要重训？
**是的，Phase 2 重训是必要的。** 原因：
1. Z 轴压扁说明 denoiser 没有学到正确的 Z 方向展布
2. 类型头在错误采样器下训练，与正确采样流程不匹配
3. 训练时没有 CoM 归零，但推理时有，存在分布偏移

### 是否满足进入 Phase 2 的条件？
**是的。** 理由：
1. 采样器修复经过数学验证（单元测试全部通过）
2. 生成主链已恢复——不重训就能看到 RMSD -55%、Bond Validity +99%
3. 剩余问题（Z 压扁、类型偏差）需要在修正后的采样器下重训才能解决

## 6. 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `src/models/diffusion.py` | 新增 `_build_node_mask`, `_remove_mean_with_mask`, `_get_ddim_time_pairs`, `_ddim_step`, `_ddpm_step`；完全重写 `sample()` |
| `src/train.py` | `generate()` 增加 `disable_guidance`, `disable_ring_snap`, `sampler` 参数 |
| `src/eval_phase1.py` | 增加 `--use_gt_count`, `--disable_guidance`, `--disable_ring_snap`, `--sampler` CLI 参数 |
| `src/visualize_val.py` | 同上 CLI 参数 |
| `src/visualize_5mol.py` | 同上 CLI 参数 |
| `tests/test_v16c_sampler.py` | 新增 6 个单元测试 |
| `docs/V16c修复路线图.md` | Phase 0 文档 |
| `run_v16c_phase1.sh` | Phase 1 评估脚本 |
