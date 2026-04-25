# V20 EXP-05 Dense Structured-Map Baseline

## 一、实验设置
- baseline checkpoint：`/root/autodl-tmp/micro/experiments/v19_stage1_2d_debug/checkpoints/best_v19_stage1.pt`
- baseline config：`/root/autodl-tmp/micro/config_v19_stage1_2d_debug.json`
- 当前 baseline 训练规模：`train=128` `val=16` `epochs=1`
- 当前评估样本数：`512`
- 说明：这是当前仓库中唯一现成的 stage1 dense baseline checkpoint，因此结果属于 `debug baseline`，不是最终公平训练版。

## 二、Dense Baseline Full-Test 主结果
- `atom_xy_mae`：均值 = `0.3746`；标准差 = `0.0175`
- `bond_map_mae`：均值 = `0.3821`；标准差 = `0.0119`
- `type_map_mae`：均值 = `0.4536`；标准差 = `0.0023`
- `atom_center_score_r3`：均值 = `0.6304`；标准差 = `0.0339`
- `typed_center_score_r3`：均值 = `0.2726`；标准差 = `0.0244`
- `type_top1_local_acc_r3`：均值 = `0.1184`；标准差 = `0.0685`
- `atom_type_macro_f1_2d`：均值 = `0.0646`；标准差 = `0.0410`
- `ch_collapse_rate_2d`：均值 = `0.0000`；标准差 = `0.0000`
- `pred_object_score`：均值 = `0.2986`；标准差 = `0.0335`
- `pred_object_3d_score`：均值 = `0.6272`；标准差 = `0.0431`
- `pred_object_count_mae`：均值 = `19.8555`；标准差 = `10.4001`
- `pred_object_count_score`：均值 = `0.6155`；标准差 = `0.1434`
- `pred_object_center_score`：均值 = `0.5734`；标准差 = `0.0349`
- `pred_object_type_acc`：均值 = `0.1221`；标准差 = `0.0716`
- `pred_object_macro_f1`：均值 = `0.0521`；标准差 = `0.0330`
- `pred_object_hetero_f1`：均值 = `0.2080`；标准差 = `0.0878`
- `pred_object_edge_f1`：均值 = `0.5273`；标准差 = `0.0795`
- `pred_object_edge_f1_robust`：均值 = `0.7623`；标准差 = `0.2014`
- `pred_object_match_coverage_robust`：均值 = `0.3624`；标准差 = `0.1349`
- `pred_object_graph_score`：均值 = `0.3905`；标准差 = `0.0383`
- `pred_object_heavy_rmsd`：均值 = `0.0828`；标准差 = `0.0293`
- `pred_object_z_mae`：均值 = `0.1384`；标准差 = `0.1160`

## 三、与 V20 主模型对比
| 字段名 | Dense Baseline | V20 Full-Test | Dense - V20 |
|---|---:|---:|---:|
| pred_object_score | 0.2986 | 0.7141 | -0.4155 |
| pred_object_type_acc | 0.1221 | 0.6942 | -0.5721 |
| pred_object_macro_f1 | 0.0521 | 0.5345 | -0.4823 |
| pred_object_hetero_f1 | 0.2080 | 0.7434 | -0.5353 |
| pred_object_edge_f1 | 0.5273 | 0.6358 | -0.1085 |
| pred_object_edge_f1_robust | 0.7623 | 0.9138 | -0.1516 |
| pred_object_count_mae | 19.8555 | 0.9434 | +18.9121 |
| pred_object_z_mae | 0.1384 | 0.0946 | +0.0439 |

## 四、代表样本
- `best`：`idx=371` `cid=9989780` `score=0.4160` `type_acc=0.3000` `edge_f1=0.4857` `count_mae=7.0000`
- `median`：`idx=189` `cid=9972418` `score=0.2976` `type_acc=0.1765` `edge_f1=0.5812` `count_mae=30.0000`
- `worst`：`idx=202` `cid=99769876` `score=0.2099` `type_acc=0.0000` `edge_f1=0.6122` `count_mae=45.0000`

## 五、结论
- 该 baseline 的优势主要体现在把 AFM 转成可视化的稠密 2D 图；但它没有对象级 z 分支，也没有 center-conditioned type/edge closure。
- 因此如果它在 `pred_object_score`、`pred_object_type_acc`、`pred_object_edge_f1`、`pred_object_z_mae` 上明显落后于 V20，这更说明当前对象级路线的价值。
- 但因为当前 baseline 只训练了 debug 规模，论文正文里应把它明确写成 `preliminary dense baseline`，后续最好再补一个正式训练版。
