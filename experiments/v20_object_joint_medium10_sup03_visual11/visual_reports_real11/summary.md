# V20 Real11 Visualization Summary

- checkpoint: `/root/autodl-tmp/micro/experiments/v20_object_joint_medium10/checkpoints/latest_v19_object_joint.pt`
- num_cases: `11`

| case_id | label | tip | variant | kind | gt_rank | pred_atoms | ref_atoms | top3 labels |
|---|---|---|---|---|---:|---:|---:|---|
| camphor_exp_1 | camphor | unknown | normal | reference | 6 | 16 | 27 | PTCDA, PTH, TTF-TDZ |
| camphor_exp_3 | camphor | unknown | normal | reference | 7 | 16 | 27 | PTCDA, Water, BCB |
| camphor_exp_4 | camphor | unknown | normal | reference | 6 | 18 | 27 | TTF-TDZ, BCB, PTH |
| camphor_exp_6 | camphor | unknown | normal | reference | 7 | 48 | 27 | PTH, TTF-TDZ, NCM |
| camphor_exp_7 | camphor | unknown | normal | reference | 6 | 48 | 27 | BCB, NCM, PTCDA |
| edafm_BCB_CO_exp | BCB | CO | normal | gt | 6 | 48 | 12 | Water, PTH, TTF-TDZ |
| edafm_BCB_Xe_exp | BCB | Xe | normal | gt | 4 | 48 | 12 | PTH, NCM, TTF-TDZ |
| edafm_PTCDA_CO_exp | PTCDA | CO | inverted | gt | 3 | 48 | 38 | PTH, BCB, PTCDA |
| edafm_PTCDA_Xe_exp | PTCDA | Xe | inverted | gt | 5 | 48 | 38 | BCB, PTH, TTF-TDZ |
| edafm_Water_CO_exp | Water | CO | inverted | gt | 7 | 48 | 341 | BCB, PTCDA, camphor |
| edafm_Water_Xe_exp | Water | Xe | normal | gt | 6 | 48 | 341 | TTF-TDZ, BCB, NCM |

- `kind=gt` 表示左侧参考面板使用真实结构。
- `kind=reference` 表示左侧参考面板使用同身份最佳参考构型。
