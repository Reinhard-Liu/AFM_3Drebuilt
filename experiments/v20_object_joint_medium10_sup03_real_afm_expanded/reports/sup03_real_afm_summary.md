# SUP-03 扩展真实 AFM 实验报告

## 一、实验设置
- checkpoint：`/root/autodl-tmp/micro/experiments/v20_object_joint_medium10/checkpoints/latest_v19_object_joint.pt`
- real AFM roots：`/root/autodl-tmp/micro/real_afm/edafm_sup03_cases, /root/autodl-tmp/micro/real_afm/camphor_sup03_cases`
- EDAFM source root：`/root/autodl-tmp/real_afm_datasets/edafm_zenodo_10609676/edafm-data/edafm-data`
- camphor structure root：`/root/autodl-tmp/real_afm_datasets/camphor_adsorbate_4710346/structures`
- 检索按 `分子身份` 统计，不按单个候选构型文件统计。

## 二、normal 结果

- 全部真实 case retrieval：`n=11`，`Top1=9.09%`，`Top3=18.18%`，`Top5=27.27%`，`MRR=0.2738`，`mean_rank=5.27`
- GT 兼容子集 retrieval：`n=4`，`Top1=25.00%`，`Top3=50.00%`，`Top5=75.00%`，`MRR=0.4792`，`mean_rank=3.25`
- GT 兼容子集几何指标：`n=4`，`pred_object_score=0.4169`，`type_acc=0.5888`，`macro_f1=0.1873`，`edge_f1=0.4523`，`robust_edge_f1=0.1250`，`z_mae=0.1264`

| case_id | label | tip | GT-compatible | GT rank | Top1 | Top3 | Top5 | Pred atoms | GT atoms | top3 labels |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| edafm_BCB_CO_exp | BCB | CO | Y | 6 | N | N | N | 48 | 12 | Water, PTH, TTF-TDZ |
| edafm_BCB_Xe_exp | BCB | Xe | Y | 4 | N | N | Y | 48 | 12 | PTH, NCM, TTF-TDZ |
| edafm_PTCDA_CO_exp | PTCDA | CO | Y | 1 | Y | Y | Y | 48 | 38 | PTCDA, PTH, NCM |
| edafm_PTCDA_Xe_exp | PTCDA | Xe | Y | 2 | N | Y | Y | 48 | 38 | Water, PTCDA, NCM |
| edafm_Water_CO_exp | Water | CO | N | 7 | N | N | N | 48 |  | BCB, TTF-TDZ, PTH |
| edafm_Water_Xe_exp | Water | Xe | N | 6 | N | N | N | 48 |  | TTF-TDZ, BCB, NCM |
| camphor_exp_1 | camphor | unknown | N | 6 | N | N | N | 16 |  | PTCDA, PTH, TTF-TDZ |
| camphor_exp_3 | camphor | unknown | N | 7 | N | N | N | 16 |  | PTCDA, Water, BCB |
| camphor_exp_4 | camphor | unknown | N | 6 | N | N | N | 18 |  | TTF-TDZ, BCB, PTH |
| camphor_exp_6 | camphor | unknown | N | 7 | N | N | N | 48 |  | PTH, TTF-TDZ, NCM |
| camphor_exp_7 | camphor | unknown | N | 6 | N | N | N | 48 |  | BCB, NCM, PTCDA |

## 二、inverted 结果

- 全部真实 case retrieval：`n=11`，`Top1=18.18%`，`Top3=45.45%`，`Top5=72.73%`，`MRR=0.4054`，`mean_rank=3.82`
- GT 兼容子集 retrieval：`n=4`，`Top1=50.00%`，`Top3=75.00%`，`Top5=100.00%`，`MRR=0.6333`，`mean_rank=2.50`
- GT 兼容子集几何指标：`n=4`，`pred_object_score=0.4659`，`type_acc=0.7336`，`macro_f1=0.1514`，`edge_f1=0.5031`，`robust_edge_f1=0.1000`，`z_mae=0.0659`

| case_id | label | tip | GT-compatible | GT rank | Top1 | Top3 | Top5 | Pred atoms | GT atoms | top3 labels |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| edafm_BCB_CO_exp | BCB | CO | Y | 1 | Y | Y | Y | 48 | 12 | BCB, TTF-TDZ, PTCDA |
| edafm_BCB_Xe_exp | BCB | Xe | Y | 1 | Y | Y | Y | 36 | 12 | BCB, PTH, TTF-TDZ |
| edafm_PTCDA_CO_exp | PTCDA | CO | Y | 3 | N | Y | Y | 48 | 38 | PTH, BCB, PTCDA |
| edafm_PTCDA_Xe_exp | PTCDA | Xe | Y | 5 | N | N | Y | 48 | 38 | BCB, PTH, TTF-TDZ |
| edafm_Water_CO_exp | Water | CO | N | 7 | N | N | N | 48 |  | BCB, PTCDA, camphor |
| edafm_Water_Xe_exp | Water | Xe | N | 2 | N | Y | Y | 10 |  | PTCDA, Water, BCB |
| camphor_exp_1 | camphor | unknown | N | 2 | N | Y | Y | 48 |  | BCB, camphor, TTF-TDZ |
| camphor_exp_3 | camphor | unknown | N | 6 | N | N | N | 48 |  | BCB, PTCDA, TTF-TDZ |
| camphor_exp_4 | camphor | unknown | N | 5 | N | N | Y | 48 |  | BCB, TTF-TDZ, PTH |
| camphor_exp_6 | camphor | unknown | N | 4 | N | N | Y | 48 |  | BCB, PTH, TTF-TDZ |
| camphor_exp_7 | camphor | unknown | N | 6 | N | N | N | 48 |  | BCB, PTH, NCM |

