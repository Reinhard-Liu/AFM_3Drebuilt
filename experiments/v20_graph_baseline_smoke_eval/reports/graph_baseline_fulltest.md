# SUP-02 Graph Baseline Full-Test Report

## 一、实验设置
- checkpoint：`/root/autodl-tmp/micro/experiments/v20_graph_baseline_smoke/checkpoints/best_v20_graph_baseline.pt`
- config：`/root/autodl-tmp/micro/config_v20_graph_baseline_medium10.json`
- full-test 样本数：`64`

## 二、对象级主结果
- `pred_object_score`：`0.3497`
- `pred_object_3d_score`：`0.4531`
- `pred_object_count_mae`：`4.8281`
- `pred_object_count_score`：`0.8183`
- `pred_object_center_score`：`0.5231`
- `pred_object_type_acc`：`0.5055`
- `pred_object_macro_f1`：`0.1724`
- `pred_object_hetero_f1`：`0.0000`
- `pred_object_edge_f1`：`0.1999`
- `pred_object_edge_f1_robust`：`0.1558`
- `pred_object_match_coverage_robust`：`0.0576`
- `pred_object_graph_score`：`0.3090`
- `pred_object_heavy_rmsd`：`0.2855`
- `pred_object_z_mae`：`0.4917`

## 三、与 V20 对比
| 字段名 | Graph | V20 | Graph - V20 |
|---|---:|---:|---:|

## 四、与 SUP-01 Dense 对比
| 字段名 | Graph | Dense | Graph - Dense |
|---|---:|---:|---:|
| pred_object_score | 0.3497 | 0.2936 | +0.0562 |
| pred_object_3d_score | 0.4531 | 0.5333 | -0.0801 |
| pred_object_count_mae | 4.8281 | 34.1426 | -29.3145 |
| pred_object_count_score | 0.8183 | 0.3947 | +0.4236 |
| pred_object_center_score | 0.5231 | 0.8673 | -0.3442 |
| pred_object_type_acc | 0.5055 | 0.3458 | +0.1597 |
| pred_object_macro_f1 | 0.1724 | 0.1131 | +0.0593 |
| pred_object_hetero_f1 | 0.0000 | 0.1028 | -0.1028 |
| pred_object_edge_f1 | 0.1999 | 0.3324 | -0.1325 |
| pred_object_edge_f1_robust | 0.1558 | 0.6745 | -0.5187 |
| pred_object_match_coverage_robust | 0.0576 | 0.2319 | -0.1743 |
| pred_object_graph_score | 0.3090 | 0.2582 | +0.0508 |
| pred_object_heavy_rmsd | 0.2855 | 0.2534 | +0.0322 |
| pred_object_z_mae | 0.4917 | 0.1490 | +0.3426 |

## 五、代表样本
- `best_sample`：idx=`32` cid=`9990935` score=`0.4140` type_acc=`0.6667` edge_f1=`0.2120` figure=`/root/autodl-tmp/micro/experiments/v20_graph_baseline_smoke_eval/samples/graph_baseline_sample_0032.png`
- `median_sample`：idx=`23` cid=`9990756` score=`0.3502` type_acc=`0.4286` edge_f1=`0.2414` figure=`/root/autodl-tmp/micro/experiments/v20_graph_baseline_smoke_eval/samples/graph_baseline_sample_0023.png`
- `worst_sample`：idx=`52` cid=`9991701` score=`0.2382` type_acc=`0.3333` edge_f1=`0.0000` figure=`/root/autodl-tmp/micro/experiments/v20_graph_baseline_smoke_eval/samples/graph_baseline_sample_0052.png`
