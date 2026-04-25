# V20 EXP-05 Dense Structured-Map Baseline

## 一、实验设置
- baseline checkpoint：`/root/autodl-tmp/micro/experiments/v20_dense_stage1_medium10/checkpoints/best_v19_stage1.pt`
- baseline config：`/root/autodl-tmp/micro/config_v20_dense_stage1_medium10.json`
- 当前 baseline 训练规模：`train=65536` `val=512` `epochs=10`
- 当前评估样本数：`512`
- 说明：这是当前仓库中唯一现成的 stage1 dense baseline checkpoint，因此结果属于 `debug baseline`，不是最终公平训练版。

## 二、Dense Baseline Full-Test 主结果
- `atom_xy_mae`：均值 = `0.0718`；标准差 = `0.0158`
- `bond_map_mae`：均值 = `0.0466`；标准差 = `0.0104`
- `type_map_mae`：均值 = `0.0122`；标准差 = `0.0022`
- `atom_center_score_r3`：均值 = `0.6396`；标准差 = `0.2899`
- `typed_center_score_r3`：均值 = `0.2663`；标准差 = `0.1528`
- `type_top1_local_acc_r3`：均值 = `0.3346`；标准差 = `0.1070`
- `atom_type_macro_f1_2d`：均值 = `0.1627`；标准差 = `0.0711`
- `ch_collapse_rate_2d`：均值 = `0.1068`；标准差 = `0.2273`
- `pred_object_score`：均值 = `0.2936`；标准差 = `0.1203`
- `pred_object_3d_score`：均值 = `0.5333`；标准差 = `0.1717`
- `pred_object_count_mae`：均值 = `34.1426`；标准差 = `8.7311`
- `pred_object_count_score`：均值 = `0.3947`；标准差 = `0.1855`
- `pred_object_center_score`：均值 = `0.8673`；标准差 = `0.3347`
- `pred_object_type_acc`：均值 = `0.3458`；标准差 = `0.1616`
- `pred_object_macro_f1`：均值 = `0.1131`；标准差 = `0.0662`
- `pred_object_hetero_f1`：均值 = `0.1028`；标准差 = `0.1051`
- `pred_object_edge_f1`：均值 = `0.3324`；标准差 = `0.1569`
- `pred_object_edge_f1_robust`：均值 = `0.6745`；标准差 = `0.3292`
- `pred_object_match_coverage_robust`：均值 = `0.2319`；标准差 = `0.1402`
- `pred_object_graph_score`：均值 = `0.2582`；标准差 = `0.1043`
- `pred_object_heavy_rmsd`：均值 = `0.2534`；标准差 = `0.2887`
- `pred_object_z_mae`：均值 = `0.1490`；标准差 = `0.1588`

## 三、与 V20 主模型对比
| 字段名 | Dense Baseline | V20 Full-Test | Dense - V20 |
|---|---:|---:|---:|
| pred_object_score | 0.2936 | 0.7141 | -0.4205 |
| pred_object_type_acc | 0.3458 | 0.6942 | -0.3483 |
| pred_object_macro_f1 | 0.1131 | 0.5345 | -0.4214 |
| pred_object_hetero_f1 | 0.1028 | 0.7434 | -0.6405 |
| pred_object_edge_f1 | 0.3324 | 0.6358 | -0.3034 |
| pred_object_edge_f1_robust | 0.6745 | 0.9138 | -0.2394 |
| pred_object_count_mae | 34.1426 | 0.9434 | +33.1992 |
| pred_object_z_mae | 0.1490 | 0.0946 | +0.0545 |

## 四、代表样本
- `best`：`idx=131` `cid=9964700` `score=0.4909` `type_acc=0.6000` `edge_f1=0.3793` `count_mae=34.0000`
- `median`：`idx=53` `cid=99524078` `score=0.3309` `type_acc=0.3902` `edge_f1=0.3483` `count_mae=23.0000`
- `worst`：`idx=32` `cid=99498480` `score=0.0037` `type_acc=0.0000` `edge_f1=0.0000` `count_mae=51.0000`

## 五、结论
- 该 baseline 的优势主要体现在把 AFM 转成可视化的稠密 2D 图；但它没有对象级 z 分支，也没有 center-conditioned type/edge closure。
- 因此如果它在 `pred_object_score`、`pred_object_type_acc`、`pred_object_edge_f1`、`pred_object_z_mae` 上明显落后于 V20，这更说明当前对象级路线的价值。
- 但因为当前 baseline 只训练了 debug 规模，论文正文里应把它明确写成 `preliminary dense baseline`，后续最好再补一个正式训练版。
