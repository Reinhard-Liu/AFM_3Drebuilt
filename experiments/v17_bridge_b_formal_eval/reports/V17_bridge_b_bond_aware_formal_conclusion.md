# V17 Bridge-B Bond-Aware 正式评估结论

评估日期：2026-04-10  
代码提交：`e12aec5` (`v17-bridge: add bond-aware scaffold edge correction`)  
代码标签：`v17-bridge-bond-aware`  
评估 checkpoint：`experiments/v17_bridge_b_debug/checkpoints/best_gen.pt`  
评估配置：`config_v17_bridge_b_debug.json`

## 1. 本次改动内容

本轮在 `Bridge-B` 的 soft scaffold constraint 中新增了 **bond-aware local edge correction**：

- 保留已有的 `position pull`
- 保留已有的 `plane projection`
- 新增 GT scaffold 局部边 `scaffold_local_edges`
- 对这些局部边执行软边长回正 `edge_scale`

当前正式评估配置：

- `constraint_time_threshold = 150`
- `constraint_scale = 0.08`
- `plane_scale = 0.04`
- `edge_scale = 0.12`

## 2. 正式结果

### 2.1 Val Set

来源：`experiments/v17_bridge_b_formal_eval/reports/val_test_metrics.json`

- RMSD: `0.5822 +/- 0.3757`
- Bottom Recall: `0.0722 +/- 0.2066`
- Bottom RMSD: `0.5096`
- Bond(gt_mask): `0.5520`
- Bond(pred_mask): `0.5509`
- Count Accuracy: `0.2812` (MAE `1.3516`)
- Type Match: `0.4030`
- Ring Preservation: `0.9234`
- Composite Score: `0.5201`

### 2.2 Test Set

官方 `eval_only` 日志：`experiments/v17_bridge_b_formal_eval/logs/test_eval_official.log`

- RMSD: `0.4560 +/- 0.3076`
- Bottom Recall: `0.0774 +/- 0.2054`
- Bottom RMSD: `0.4603`
- Bond(gt_mask): `0.5423`
- Bond(pred_mask): `0.5324`
- Count Accuracy: `0.3398` (MAE `1.0664`)
- Type Match: `0.4015`
- Composite Score: `0.5445`

同配置二次 full test 复跑：`experiments/v17_bridge_b_formal_eval/reports/val_test_metrics.json`

- RMSD: `0.4866 +/- 0.3296`
- Bottom Recall: `0.0665 +/- 0.2018`
- Bottom RMSD: `0.4694`
- Bond(gt_mask): `0.5173`
- Bond(pred_mask): `0.5052`
- Count Accuracy: `0.3398` (MAE `1.0664`)
- Type Match: `0.3909`
- Ring Preservation: `0.9188`
- Composite Score: `0.5325`

## 3. 与旧版 Bridge-B 的比较

旧版 `Bridge-B` 同一 checkpoint、旧约束逻辑下的 test 结果来自：
`experiments/v17_bridge_b_debug/checkpoints/training.log`

- RMSD: `0.4186`
- Bottom Recall: `0.0563`
- Bond Validity: `0.4480`
- Count Accuracy: `0.3398`
- Type Match: `0.3839`
- Composite Score: `0.5275`

本轮 bond-aware 改动后的主要变化：

- `Bond` 明显上升  
  从 `0.4480` 提升到约 `0.517 - 0.542`
- `Bottom Recall` 上升  
  从 `0.0563` 提升到约 `0.066 - 0.077`
- `Type Match` 小幅上升  
  从 `0.3839` 提升到约 `0.391 - 0.402`
- `Composite` 小幅上升  
  从 `0.5275` 提升到约 `0.533 - 0.545`
- `RMSD` 没有同步改善，反而略有波动或回落

## 4. 如何解释这次结果

结论是明确的：

1. `bond-aware edge correction` 是有效的。  
   这次已经实证修复了旧版 Bridge-B 的核心问题：  
   以前是 `Bottom / Ring / Composite` 上升，但 `Bond` 下滑；  
   现在 `Bond` 已经可以和 `Bottom / Ring` 一起上升。

2. 当前改动主要改善的是 **局部化学合理性**，不是全局几何拟合。  
   因此 `Bond`、`Type`、`Bottom` 改善明显，而 `RMSD` 不一定同步改善。

3. `Bridge-B` 的结构层方向被进一步确认。  
   当前不再是“scaffold 只会拉坏 bond”，而是：
   - 只做几何拉回时，bond 会受损
   - 加入局部边长回正后，bond 可以被救回来

## 5. 关于评估波动

两次 full test 的绝对值存在波动，尤其体现在 `RMSD` 和 `Bond` 上。

这说明当前 full evaluation 仍然受采样随机性影响：

- 官方 `eval_only` test：Composite `0.5445`
- 二次复跑 full test：Composite `0.5325`

因此更稳妥的结论应写成：

- 本轮改动在 test 上把 Composite 稳定推到 **约 `0.53 - 0.54`**
- Bond(gt) 稳定推到 **约 `0.52 - 0.54`**
- 方向稳定优于旧版 `0.4480` Bond baseline

## 6. 正式判断

当前可以正式确认：

- `Bridge-B` 不是错误方向
- `bond-aware local edge correction` 是必须组件
- `Bridge-B` 已经从“会伤 Bond 的几何先验”升级为“能同时改善 Bond 与结构层指标的可用方案”

但当前还不能宣称它已经解决整体结构重建问题，因为：

- `RMSD` 仍不稳定
- `Count` 没改善
- `Bottom Recall` 虽提升但仍低
- `Type` 只小幅改善

## 7. 下一步建议

下一步不建议退回旧版 Bridge-B，也不建议直接跳到 predicted scaffold generator。

建议顺序：

1. 固定当前 `edge_scale = 0.12` 作为新的 Bridge-B 默认配置
2. 做一轮 **固定随机种子 / 多次重复** 的 val/test 评估，量化采样方差
3. 专门分析 `RMSD` 为什么没有随 Bond 同步变好
4. 再决定是否进入真正的 predicted scaffold generator

一句话总结：

**这次改动已经证明：Bridge-B 的正确补强方式不是更强的几何拉回，而是把 scaffold 约束补成 bond-aware 的局部结构层。**
