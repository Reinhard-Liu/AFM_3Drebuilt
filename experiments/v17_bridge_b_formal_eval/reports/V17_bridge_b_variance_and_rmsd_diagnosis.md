# V17 Bridge-B 方差评估与 RMSD 诊断

评估对象：`experiments/v17_bridge_b_debug/checkpoints/best_gen.pt`  
评估配置：`config_v17_bridge_b_eval.json`  
Bridge-B 默认评估参数：

- `bridge_eval_scaffold_constraint_time_threshold = 150`
- `bridge_eval_scaffold_constraint_scale = 0.08`
- `bridge_eval_scaffold_plane_scale = 0.04`
- `bridge_eval_scaffold_edge_scale = 0.12`
- `eval_ddim_steps = 100`

## 1. 多次重复评估

固定种子：`11, 22, 33, 44, 55`  
结果文件：`experiments/v17_bridge_b_formal_eval/reports/repeat_eval_summary.json`

### 1.1 Val 波动

- RMSD: `0.5844 +/- 0.0061`
- Bond(gt): `0.5391 +/- 0.0120`
- Type Match: `0.4064 +/- 0.0068`
- Bottom Recall: `0.0716 +/- 0.0108`
- Ring Preservation: `0.9197 +/- 0.0031`
- Composite: `0.5167 +/- 0.0039`

### 1.2 Test 波动

- RMSD: `0.4842 +/- 0.0199`
- Bond(gt): `0.5282 +/- 0.0148`
- Bond(pred): `0.5208 +/- 0.0174`
- Type Match: `0.3889 +/- 0.0047`
- Bottom Recall: `0.0592 +/- 0.0182`
- Ring Preservation: `0.9199 +/- 0.0023`
- Composite: `0.5328 +/- 0.0076`

### 1.3 结论

- `Bond / Type / Ring / Composite` 的波动都存在，但幅度可控。
- `RMSD` 和 `Bottom Recall` 的波动最明显，说明当前 test 分数确实受采样随机性影响。
- 对外汇报时更稳妥的写法应是区间，而不是单点：
  - Test `Bond(gt)` 约在 `0.51 - 0.55`
  - Test `Composite` 约在 `0.52 - 0.54`
  - Test `RMSD` 约在 `0.46 - 0.52`

## 2. RMSD 诊断

诊断脚本：`src/v17_bridge_rmsd_diagnosis.py`  
诊断文件：`experiments/v17_bridge_b_formal_eval/reports/rmsd_diagnosis_test_seed42.json`  
对比对象：

- Baseline：guided，无 scaffold soft constraint
- Bridge-B：GT scaffold tokens + soft constraint + `edge_scale=0.12`

诊断 split：`test`  
固定种子：`42`

### 2.1 平均 delta（Bridge-B 相对 Baseline）

- RMSD: `-0.0313`
- Bond(gt): `+0.0332`
- Bottom Recall: `-0.0028`
- Type Match: `-0.0477`
- Ring Preservation: `+0.0463`
- Scaffold RMSD: `-0.0875`
- Non-scaffold RMSD: `-0.0029`
- Attachment RMSD: `-0.0840`
- Scaffold edge MAE: `-0.3971`

### 2.2 中位数 delta

- RMSD: `+0.0204`
- Bond(gt): `+0.0107`
- Scaffold RMSD: `-0.0587`
- Non-scaffold RMSD: `+0.0523`
- Attachment RMSD: `-0.0496`
- Scaffold edge MAE: `-0.1750`

### 2.3 相关性

`delta_rmsd` 与以下量的相关系数：

- `delta_non_scaffold_rmsd`: `0.9797`
- `delta_scaffold_rmsd`: `0.6424`
- `delta_attachment_rmsd`: `0.6328`
- `delta_edge_mae`: `0.5827`
- `delta_bond`: `-0.1793`

### 2.4 最关键的结论

1. `Bond` 的提升是真实的，而且主要来自 scaffold 局部边和 attachment 区域。  
   证据：
   - `scaffold edge MAE` 大幅下降
   - `scaffold RMSD` 和 `attachment RMSD` 都明显改善

2. `RMSD` 为什么没有稳定同步提升，主因不是 scaffold 区域，而是 **non-scaffold 原子**。  
   最强证据：
   - `delta_rmsd` 与 `delta_non_scaffold_rmsd` 的相关系数达到 `0.9797`
   - 中位数上 `non-scaffold RMSD` 仍然是变差的（`+0.0523`）

3. 也就是说，Bridge-B 现在已经能把环骨架和 attachment 区域拉到更合理的位置，但 **侧链 / 非 scaffold 原子** 并没有跟着一起改善，反而经常拖累整体 RMSD。

4. `Bond` 提升但 `RMSD` 变差的样本占比不低。  
   - `bond_up_rmsd_worse_fraction = 0.2656`
   - 这类样本里：
     - `mean_delta_non_scaffold_rmsd = +0.3086`
     - `mean_delta_scaffold_rmsd = +0.0237`
     - `mean_delta_edge_mae = -0.2042`

   这说明在这些样本里，局部边长确实变好了，但 overall RMSD 被非 scaffold 区域拉坏了。

5. `Count` 仍然是硬瓶颈。  
   - `exact_count_fraction = 0.3398`
   - 对 `exact_count` 样本，`mean_rmsd_delta_exact = +0.0146`
   - 对 `count` 不准确样本，`mean_rmsd_delta_inexact = -0.0549`

   这说明 `Bridge-B` 不是通过修 count 来改善 Bond；count 问题仍独立存在，也继续限制 RMSD。

## 3. 最终判断

当前可以明确判断：

- `edge_scale = 0.12` 应固定为新的 Bridge-B 默认评估配置。
- `Bridge-B` 现在已经不再是“改善 scaffold 但伤害 Bond”的方案，而是能同时改善：
  - `Bond`
  - `Ring`
  - scaffold / attachment 局部几何
- 当前 `RMSD` 不能稳定同步改善，不是因为 scaffold 方向错了，而是因为 **非 scaffold 原子和 count 头仍是主瓶颈**。

## 4. 下一步建议

接下来不该退回旧 Bridge-B，也不该马上跳 predicted scaffold generator。

更合理的顺序是：

1. 保持当前 `Bridge-B + edge_scale=0.12`
2. 专门补一层 **sidechain / non-scaffold 约束或条件化**
3. 并行继续处理 `count` 问题
4. 等 non-scaffold 区域不再拖 RMSD 后，再进入 predicted scaffold generator
