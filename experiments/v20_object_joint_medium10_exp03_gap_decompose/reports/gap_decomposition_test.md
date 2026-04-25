# V20 EXP-03 Strict vs Robust Gap Decomposition

## 一、实验设置
- checkpoint：`/root/autodl-tmp/micro/experiments/v20_object_joint_medium10/checkpoints/latest_v19_object_joint.pt`
- split：`test`
- 样本数：`512`
- `matched-r3` 半径：`3.0` 像素

## 二、核心均值
- `pred_object_edge_f1`：严格对象级边F1 = `0.6358`
- `pred_object_edge_f1_robust`：现有稳健边F1 = `0.9138`
- `edge_f1_xy_r3`：按XY半径3像素门限后的边F1 = `0.9104`
- `edge_gap_robust`：稳健边F1与严格边F1的差值 = `0.2780`
- `edge_gap_xy_r3`：XY-R3边F1与严格边F1的差值 = `0.2746`
- `pred_object_type_acc`：全局纯预测对象类型准确率 = `0.6942`
- `matched_type_acc_r3`：matched-r3 类型准确率 = `0.8334`
- `pred_object_macro_f1`：全局纯预测对象类型宏平均F1 = `0.5345`
- `matched_macro_f1_r3`：matched-r3 类型宏平均F1 = `0.6635`
- `matched_gt_node_coverage_r3`：matched-r3 GT节点覆盖率 = `0.6957`
- `matched_gt_bond_coverage_r3`：matched-r3 GT键覆盖率 = `0.5791`
- `mean_xy_match_px_r3`：matched-r3 平均XY匹配误差(像素) = `1.1444`
- `pred_object_z_mae`：全局纯预测对象z误差 = `0.0946`
- `mean_z_abs_err_matched_ang`：matched-r3 平均z绝对误差(Å) = `0.0426`
- `matched_pair_dist_mae_ang`：matched-r3 成对距离MAE(Å) = `0.1976`
- `matched_bond_len_mae_ang`：matched-r3 键长MAE(Å) = `0.1867`

## 三、计数型判断
- `edge_gap_robust >= 0.20` 的样本数：`413` / `512` = `80.66%`
- `pred_object_edge_f1 < 0.70` 且 `pred_object_edge_f1_robust >= 0.90` 的样本数：`232` / `512` = `45.31%`
- `matched_type_acc_r3 - pred_object_type_acc >= 0.10` 的样本数：`370` / `512` = `72.27%`

## 四、高 gap 子集均值
- 高 gap 子集数量：`413`
- `pred_object_edge_f1`：严格边F1 = `0.6134`
- `pred_object_edge_f1_robust`：稳健边F1 = `0.9223`
- `matched_gt_node_coverage_r3`：matched-r3 GT节点覆盖率 = `0.6826`
- `matched_type_acc_r3`：matched-r3 类型准确率 = `0.8357`
- `matched_macro_f1_r3`：matched-r3 类型宏平均F1 = `0.6696`
- `matched_bond_len_mae_ang`：matched-r3 键长MAE(Å) = `0.1855`
- `pred_object_z_mae`：纯预测对象z误差 = `0.0970`

## 五、Top Gap 样本
- `idx=28` `cid=99494789` `strict=0.4571` `robust=1.0000` `xy_r3=1.0000` `gap=0.5429` `coverage=0.5500` `matched_type=0.8182` `bond_len_mae=0.1246`
- `idx=180` `cid=9971773` `strict=0.3294` `robust=0.8571` `xy_r3=0.8462` `gap=0.5277` `coverage=0.5882` `matched_type=0.9500` `bond_len_mae=0.1073`
- `idx=443` `cid=9991875` `strict=0.4324` `robust=0.9524` `xy_r3=0.9474` `gap=0.5199` `coverage=0.5333` `matched_type=0.9375` `bond_len_mae=0.2245`
- `idx=163` `cid=99683` `strict=0.4815` `robust=1.0000` `xy_r3=1.0000` `gap=0.5185` `coverage=0.5909` `matched_type=1.0000` `bond_len_mae=0.2022`
- `idx=482` `cid=9993903` `strict=0.4946` `robust=1.0000` `xy_r3=1.0000` `gap=0.5054` `coverage=0.6000` `matched_type=0.8571` `bond_len_mae=0.1534`

## 六、结论
- 如果 `pred_object_edge_f1_robust` 和 `edge_f1_xy_r3` 明显高于 `pred_object_edge_f1`，说明严格对象级边F1被对象错位显著放大。
- 如果 `matched_type_acc_r3` 明显高于全局 `pred_object_type_acc`，说明一部分类型错误来自错对象匹配，而不是类型头本身完全失效。
- 如果 `matched_bond_len_mae_ang` 和 `matched_pair_dist_mae_ang` 仍较小，则说明在正确对齐的对象子集上，局部几何已经明显优于 strict graph 指标给出的印象。
