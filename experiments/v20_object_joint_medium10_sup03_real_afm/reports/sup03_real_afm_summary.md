# SUP-03 真实 AFM 零样本迁移实验报告

## 一、实验设置
- checkpoint：`/root/autodl-tmp/micro/experiments/v20_object_joint_medium10/checkpoints/latest_v19_object_joint.pt`
- prepared real-AFM root：`/root/autodl-tmp/micro/real_afm/edafm_sup03_cases`
- EDAFM source root：`/root/autodl-tmp/real_afm_datasets/edafm_zenodo_10609676/edafm-data/edafm-data`
- 候选分子池大小：`5`
- 候选分子：`BCB, NCM, PTCDA, PTH, TTF-TDZ`
- 因 `MAX_ATOMS=85` 被排除的候选：`Water(341)`
- 评估协议：`zero_shot_real_afm_edafm`
- 说明：本实验不做任何真实 AFM 微调，只做零样本迁移。
- 因 `MAX_ATOMS=85` 被排除的真实 case：`edafm_Water_CO_exp(341), edafm_Water_Xe_exp(341)`

## 二、contrast 版本对比
| 设置 | case数 | Top1 | Top3 | Top5 | MRR | mean_rank | pred_object_score | type_acc | macro_f1 | edge_f1 | robust_edge_f1 | z_mae | count_mae | center_score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4 | 50.00% | 50.00% | 100.00% | 0.6125 | 2.75 | 0.4169 | 0.5888 | 0.1873 | 0.4523 | 0.1250 | 0.1264 | 23.0000 | 0.9273 |
| inverted | 4 | 50.00% | 75.00% | 100.00% | 0.6333 | 2.50 | 0.4659 | 0.7336 | 0.1514 | 0.5031 | 0.1000 | 0.0659 | 20.0000 | 0.9111 |
| oracle_best | 4 | 25.00% | 50.00% | 100.00% | 0.4458 | 3.25 | 0.5087 | 0.7961 | 0.2279 | 0.5669 | 0.2250 | 0.0793 | 23.0000 | 0.9558 |

## 三、normal case 结果

| case_id | 分子 | tip | GT rank | Top1 | Top3 | Pred atoms | GT atoms | pred_score | type_acc | macro_f1 | edge_f1 | robust_edge_f1 | z_mae | Top3 names |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| edafm_BCB_CO_exp | BCB | CO | 5 | N | N | 48 | 12 | 0.2201 | 0.2500 | 0.0195 | 0.3750 | 0.0000 | 0.1549 | PTH, TTF-TDZ, PTCDA |
| edafm_BCB_Xe_exp | BCB | Xe | 4 | N | N | 48 | 12 | 0.5443 | 1.0000 | 0.4000 | 0.6364 | 0.5000 | 0.0539 | PTH, NCM, TTF-TDZ |
| edafm_PTCDA_CO_exp | PTCDA | CO | 1 | Y | Y | 48 | 38 | 0.4337 | 0.5263 | 0.1253 | 0.4301 | 0.0000 | 0.1933 | PTCDA, PTH, NCM |
| edafm_PTCDA_Xe_exp | PTCDA | Xe | 1 | Y | Y | 48 | 38 | 0.4696 | 0.5789 | 0.2045 | 0.3678 | 0.0000 | 0.1034 | PTCDA, NCM, PTH |

## 三、inverted case 结果

| case_id | 分子 | tip | GT rank | Top1 | Top3 | Pred atoms | GT atoms | pred_score | type_acc | macro_f1 | edge_f1 | robust_edge_f1 | z_mae | Top3 names |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| edafm_BCB_CO_exp | BCB | CO | 1 | Y | Y | 48 | 12 | 0.4951 | 1.0000 | 0.2264 | 0.6000 | 0.0000 | 0.0356 | BCB, TTF-TDZ, PTCDA |
| edafm_BCB_Xe_exp | BCB | Xe | 1 | Y | Y | 36 | 12 | 0.3731 | 0.7500 | 0.0938 | 0.3810 | 0.0000 | 0.0003 | BCB, PTH, TTF-TDZ |
| edafm_PTCDA_CO_exp | PTCDA | CO | 3 | N | Y | 48 | 38 | 0.5256 | 0.6053 | 0.1775 | 0.5060 | 0.0000 | 0.0914 | PTH, BCB, PTCDA |
| edafm_PTCDA_Xe_exp | PTCDA | Xe | 5 | N | N | 48 | 38 | 0.4697 | 0.5789 | 0.1078 | 0.5253 | 0.4000 | 0.1362 | BCB, PTH, TTF-TDZ |

## 四、结论口径
- 该实验是 `真实 AFM 零样本迁移`，目标是判断当前对象级 3D 路线在实验图像上是否还能给出可用结构假设。
- `normal` 与 `inverted` 的差异用于诊断真实 AFM 对比度不确定性，而不是新的训练技巧。
- `oracle_best` 只作为上限诊断，不能当部署口径。
