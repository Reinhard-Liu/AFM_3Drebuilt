# SUP-02 Graph Baseline Full-Test Report

## 一、实验设置
- checkpoint：`/root/autodl-tmp/micro/experiments/v20_graph_baseline_medium10/checkpoints/best_v20_graph_baseline.pt`
- config：`/root/autodl-tmp/micro/config_v20_graph_baseline_medium10.json`
- full-test 样本数：`512`

## 二、对象级主结果
- `pred_object_score`：`0.5414`
- `pred_object_3d_score`：`0.5425`
- `pred_object_count_mae`：`1.4766`
- `pred_object_count_score`：`0.9543`
- `pred_object_center_score`：`0.6235`
- `pred_object_type_acc`：`0.4874`
- `pred_object_macro_f1`：`0.2999`
- `pred_object_hetero_f1`：`0.4476`
- `pred_object_edge_f1`：`0.5905`
- `pred_object_edge_f1_robust`：`0.9708`
- `pred_object_match_coverage_robust`：`0.4957`
- `pred_object_graph_score`：`0.5947`
- `pred_object_heavy_rmsd`：`0.2757`
- `pred_object_z_mae`：`0.3494`

## 三、与 V20 对比
| 字段名 | Graph | V20 | Graph - V20 |
|---|---:|---:|---:|

## 四、与 SUP-01 Dense 对比
| 字段名 | Graph | Dense | Graph - Dense |
|---|---:|---:|---:|
| pred_object_score | 0.5414 | 0.2936 | +0.2478 |
| pred_object_3d_score | 0.5425 | 0.5333 | +0.0092 |
| pred_object_count_mae | 1.4766 | 34.1426 | -32.6660 |
| pred_object_count_score | 0.9543 | 0.3947 | +0.5596 |
| pred_object_center_score | 0.6235 | 0.8673 | -0.2438 |
| pred_object_type_acc | 0.4874 | 0.3458 | +0.1416 |
| pred_object_macro_f1 | 0.2999 | 0.1131 | +0.1868 |
| pred_object_hetero_f1 | 0.4476 | 0.1028 | +0.3448 |
| pred_object_edge_f1 | 0.5905 | 0.3324 | +0.2581 |
| pred_object_edge_f1_robust | 0.9708 | 0.6745 | +0.2963 |
| pred_object_match_coverage_robust | 0.4957 | 0.2319 | +0.2638 |
| pred_object_graph_score | 0.5947 | 0.2582 | +0.3365 |
| pred_object_heavy_rmsd | 0.2757 | 0.2534 | +0.0224 |
| pred_object_z_mae | 0.3494 | 0.1490 | +0.2004 |

## 五、代表样本
- `best_sample`：idx=`443` cid=`9942181` score=`0.8372` type_acc=`0.7500` edge_f1=`0.8667` figure=`/root/autodl-tmp/micro/experiments/v20_graph_baseline_medium10_sup02_fulltest/samples/graph_baseline_sample_0443.png`
- `median_sample`：idx=`465` cid=`9942412` score=`0.5313` type_acc=`0.4348` edge_f1=`0.6222` figure=`/root/autodl-tmp/micro/experiments/v20_graph_baseline_medium10_sup02_fulltest/samples/graph_baseline_sample_0465.png`
- `worst_sample`：idx=`427` cid=`9940735` score=`0.2638` type_acc=`0.1000` edge_f1=`0.3077` figure=`/root/autodl-tmp/micro/experiments/v20_graph_baseline_medium10_sup02_fulltest/samples/graph_baseline_sample_0427.png`
