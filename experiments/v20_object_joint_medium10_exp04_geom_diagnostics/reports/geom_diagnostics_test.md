# V20 EXP-04 Geometry / 3D Diagnostics

## 一、实验设置
- checkpoint：`/root/autodl-tmp/micro/experiments/v20_object_joint_medium10/checkpoints/latest_v19_object_joint.pt`
- split：`test`
- 样本数：`512`
- `matched-r3` 半径：`3.0` 像素

## 二、几何主指标
- `pred_object_heavy_rmsd_ang`：纯预测对象重原子RMSD(Å，3D Hungarian) = `0.7912`
- `pred_object_z_mae`：纯预测对象z平均绝对误差(Å) = `0.0946`
- `pred_object_pair_dist_mae_r3`：matched-r3 成对距离MAE(Å) = `0.1976`
- `pred_object_bond_len_mae_r3`：matched-r3 键长MAE(Å) = `0.1867`
- `pred_object_z_corr_r3`：matched-r3 z 相关系数 = `0.3166`
- `pred_object_nonplanarity_error_r3`：matched-r3 非平面度误差(Å) = `0.0723`
- `matched_gt_node_coverage_r3`：matched-r3 GT节点覆盖率 = `0.6957`
- `matched_gt_bond_coverage_r3`：matched-r3 GT键覆盖率 = `0.5791`
- `gt_height_span_ang`：GT 高度起伏(Å) = `1.3544`
- `gt_nonplanarity_ang`：GT 非平面度(Å) = `0.2533`

## 三、通过率型统计
- `pair_dist_mae_r3_le_0p25_rate` = `0.9316`
- `bond_len_mae_r3_le_0p20_rate` = `0.6426`
- `z_corr_r3_ge_0p80_rate` = `0.2559`
- `nonplanarity_error_r3_le_0p10_rate` = `0.6660`

## 四、复杂度分层
### atom_count

| 分层 | 样本数 | heavy_rmsd(Å) | z_mae(Å) | pair_dist_mae_r3(Å) | bond_len_mae_r3(Å) | z_corr_r3 | nonplanarity_err_r3(Å) | coverage_r3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| >=35 | 128 | 0.8133 | 0.0878 | 0.2015 | 0.1871 | 0.3769 | 0.0811 | 0.6855 |
| <=22 | 106 | 0.7324 | 0.1110 | 0.1951 | 0.1893 | 0.1978 | 0.0658 | 0.7032 |
| 29-34 | 111 | 0.8058 | 0.0893 | 0.1949 | 0.1794 | 0.2614 | 0.0813 | 0.6956 |
| 23-28 | 167 | 0.8019 | 0.0928 | 0.1980 | 0.1897 | 0.3823 | 0.0635 | 0.6989 |

### ring_count

| 分层 | 样本数 | heavy_rmsd(Å) | z_mae(Å) | pair_dist_mae_r3(Å) | bond_len_mae_r3(Å) | z_corr_r3 | nonplanarity_err_r3(Å) | coverage_r3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| >=3 | 289 | 0.8092 | 0.0761 | 0.1970 | 0.1853 | 0.3038 | 0.0617 | 0.6963 |
| 2 | 136 | 0.7534 | 0.0952 | 0.1958 | 0.1833 | 0.3123 | 0.0799 | 0.7075 |
| 0-1 | 87 | 0.7906 | 0.1548 | 0.2024 | 0.1969 | 0.3658 | 0.0952 | 0.6753 |

### height_span

| 分层 | 样本数 | heavy_rmsd(Å) | z_mae(Å) | pair_dist_mae_r3(Å) | bond_len_mae_r3(Å) | z_corr_r3 | nonplanarity_err_r3(Å) | coverage_r3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| >=1.75A | 281 | 0.8036 | 0.1204 | 0.1980 | 0.1875 | 0.2119 | 0.1037 | 0.6851 |
| <1.20A | 128 | 0.7983 | 0.0188 | 0.1860 | 0.1777 | 0.2647 | 0.0086 | 0.7420 |
| 1.20-1.75A | 103 | 0.7486 | 0.1183 | 0.2110 | 0.1957 | 0.6667 | 0.0654 | 0.6671 |

### nonplanarity

| 分层 | 样本数 | heavy_rmsd(Å) | z_mae(Å) | pair_dist_mae_r3(Å) | bond_len_mae_r3(Å) | z_corr_r3 | nonplanarity_err_r3(Å) | coverage_r3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20-0.35A | 257 | 0.7930 | 0.0906 | 0.1966 | 0.1868 | 0.2649 | 0.0855 | 0.7024 |
| <0.20A | 122 | 0.7984 | 0.0112 | 0.1858 | 0.1771 | 0.2109 | 0.0088 | 0.7450 |
| >=0.35A | 133 | 0.7812 | 0.1786 | 0.2104 | 0.1954 | 0.5133 | 0.1048 | 0.6375 |

## 五、最差样本
- `worst_z_samples`：按 `pred_object_z_mae` 降序
  - `idx=89` `cid=9964074` `z_mae=0.6084` `pair=0.1743` `nonplanarity_err=0.2092`
  - `idx=502` `cid=9996089` `z_mae=0.5677` `pair=0.2187` `nonplanarity_err=0.3698`
  - `idx=345` `cid=9989291` `z_mae=0.4481` `pair=0.3193` `nonplanarity_err=0.3981`
  - `idx=340` `cid=9989228` `z_mae=0.4362` `pair=0.2041` `nonplanarity_err=0.1434`
  - `idx=346` `cid=9989311` `z_mae=0.3477` `pair=0.2095` `nonplanarity_err=0.2874`
- `worst_nonplanarity_samples`：按 `pred_object_nonplanarity_error_r3` 降序
  - `idx=345` `cid=9989291` `gt_nonplanarity=0.6330` `err=0.3981` `z_corr=-0.2540`
  - `idx=339` `cid=9989220` `gt_nonplanarity=0.5445` `err=0.3958` `z_corr=-0.4779`
  - `idx=502` `cid=9996089` `gt_nonplanarity=0.6264` `err=0.3698` `z_corr=-0.2343`
  - `idx=392` `cid=9990348` `gt_nonplanarity=0.4647` `err=0.3311` `z_corr=0.0539`
  - `idx=280` `cid=99826546` `gt_nonplanarity=0.3745` `err=0.3077` `z_corr=-0.0264`

## 六、结论
- 如果 `pair_dist_mae_r3`、`bond_len_mae_r3` 保持较低，同时 `z_corr_r3` 明显为正，说明当前模型的 3D 价值不止体现在单一 `z_mae`。
- 如果复杂度分层中 `height_span` 和 `nonplanarity` 升高时几何误差同步升高，说明当前模型的主要 3D 短板集中在高度起伏更强、非平面度更高的分子。
