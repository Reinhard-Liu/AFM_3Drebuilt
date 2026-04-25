# V20 EXP-01 全测试集对象级正式 Benchmark 报告

## 一、实验设置
- checkpoint：`/root/autodl-tmp/micro/experiments/v20_object_joint_medium10/checkpoints/latest_v19_object_joint.pt`
- checkpoint 轮次：`10`
- split：`test`
- full-test 样本数：`512`
- 参数域：`K-1`
- 图像尺寸：`128`
- 训练时 max_samples：`65536`
- 训练时 val_size：`512`

## 二、Full-Test 主结果
- 字段名 `pred_object_score`：纯预测对象闭环对象级总分，越高越好；full-test 均值 = `0.7141`；标准差 = `0.0858`
- 字段名 `pred_object_3d_score`：纯预测对象3D综合分，越高越好；full-test 均值 = `0.8112`；标准差 = `0.0564`
- 字段名 `pred_object_count_mae`：纯预测对象原子数平均绝对误差，越低越好；full-test 均值 = `0.9434`；标准差 = `1.4667`
- 字段名 `pred_object_count_score`：纯预测对象原子数相似度分数，越高越好；full-test 均值 = `0.9711`；标准差 = `0.0404`
- 字段名 `pred_object_center_score`：纯预测对象proposal中心平均置信度，越高越好；full-test 均值 = `0.9866`；标准差 = `0.0522`
- 字段名 `pred_object_type_acc`：纯预测对象原子类型准确率，越高越好；full-test 均值 = `0.6942`；标准差 = `0.0960`
- 字段名 `pred_object_macro_f1`：纯预测对象原子类型宏平均F1，越高越好；full-test 均值 = `0.5345`；标准差 = `0.1726`
- 字段名 `pred_object_hetero_f1`：纯预测对象杂原子F1，越高越好；full-test 均值 = `0.7434`；标准差 = `0.2035`
- 字段名 `pred_object_edge_f1`：纯预测对象严格对象级边F1，越高越好；full-test 均值 = `0.6358`；标准差 = `0.0982`
- 字段名 `pred_object_edge_f1_robust`：纯预测对象距离容忍后的稳健边F1，越高越好；full-test 均值 = `0.9138`；标准差 = `0.0698`
- 字段名 `pred_object_match_coverage_robust`：稳健匹配覆盖率，越高越好；full-test 均值 = `0.6910`；标准差 = `0.0994`
- 字段名 `pred_object_graph_score`：纯预测对象图结构综合分，越高越好；full-test 均值 = `0.7209`；标准差 = `0.0845`
- 字段名 `pred_object_heavy_rmsd`：纯预测对象重原子RMSD，越低越好；full-test 均值 = `0.0659`；标准差 = `0.0366`
- 字段名 `pred_object_z_mae`：纯预测对象z平均绝对误差，越低越好；full-test 均值 = `0.0946`；标准差 = `0.0778`
- 字段名 `peak_object_score`：peak-center条件对象级总分，越高越好；full-test 均值 = `0.8338`；标准差 = `0.0787`
- 字段名 `gt_object_score`：GT-center条件对象级总分，表示上限参考，越高越好；full-test 均值 = `0.8279`；标准差 = `0.0779`
- 字段名 `atom_center_score_r3`：真实原子中心半径3像素内中心命中分数，越高越好；full-test 均值 = `0.9989`；标准差 = `0.0010`
- 字段名 `typed_center_score_r3`：真实原子中心半径3像素内位置与类型同时正确的软分数，越高越好；full-test 均值 = `0.4098`；标准差 = `0.0732`
- 字段名 `atom_type_macro_f1_2d`：稠密2D类型图宏平均F1，越高越好；full-test 均值 = `0.3085`；标准差 = `0.0903`
- 字段名 `atom_xy_mae`：稠密2D原子图平均绝对误差，越低越好；full-test 均值 = `0.0109`；标准差 = `0.0059`
- 字段名 `z_map_mae`：稠密z图平均绝对误差，越低越好；full-test 均值 = `0.0010`；标准差 = `0.0013`
- 字段名 `atom_z_mae_r3`：真实中心附近z平均绝对误差，越低越好；full-test 均值 = `0.0910`；标准差 = `0.0739`
- 字段名 `peak_center_type_acc`：peak-center条件原子类型准确率，越高越好；full-test 均值 = `0.8190`；标准差 = `0.0987`
- 字段名 `peak_center_macro_f1`：peak-center条件原子类型宏平均F1，越高越好；full-test 均值 = `0.7182`；标准差 = `0.1698`
- 字段名 `peak_center_hetero_f1`：peak-center条件杂原子F1，越高越好；full-test 均值 = `0.8841`；标准差 = `0.1670`
- 字段名 `peak_center_edge_f1`：peak-center条件对象级边F1，越高越好；full-test 均值 = `0.8979`；标准差 = `0.0692`
- 字段名 `peak_center_shift_px`：peak-center相对真实中心平均偏移像素，越低越好；full-test 均值 = `1.7182`；标准差 = `0.1255`
- 字段名 `gt_center_type_acc`：GT-center条件原子类型准确率，越高越好；full-test 均值 = `0.8178`；标准差 = `0.0935`
- 字段名 `gt_center_macro_f1`：GT-center条件原子类型宏平均F1，越高越好；full-test 均值 = `0.6854`；标准差 = `0.1520`
- 字段名 `gt_center_hetero_f1`：GT-center条件杂原子F1，越高越好；full-test 均值 = `0.5861`；标准差 = `0.2497`
- 字段名 `gt_center_edge_f1`：GT-center条件对象级边F1，越高越好；full-test 均值 = `0.9988`；标准差 = `0.0047`

## 三、与当前 Validation 口径对照
- 字段名 `pred_object_score`：validation=`0.7129`，full-test=`0.7141`，差值=`+0.0012`
- 字段名 `pred_object_type_acc`：validation=`0.6937`，full-test=`0.6942`，差值=`+0.0005`
- 字段名 `pred_object_macro_f1`：validation=`0.5326`，full-test=`0.5345`，差值=`+0.0018`
- 字段名 `pred_object_hetero_f1`：validation=`0.7340`，full-test=`0.7434`，差值=`+0.0094`
- 字段名 `pred_object_edge_f1`：validation=`0.6416`，full-test=`0.6358`，差值=`-0.0058`
- 字段名 `pred_object_edge_f1_robust`：validation=`0.9194`，full-test=`0.9138`，差值=`-0.0056`
- 字段名 `pred_object_count_mae`：validation=`1.0605`，full-test=`0.9434`，差值=`-0.1172`
- 字段名 `pred_object_z_mae`：validation=`0.0938`，full-test=`0.0946`，差值=`+0.0008`
- 字段名 `peak_object_score`：validation=`0.8170`，full-test=`0.8338`，差值=`+0.0168`
- 字段名 `peak_center_type_acc`：validation=`0.8190`，full-test=`0.8190`，差值=`+0.0000`
- 字段名 `peak_center_macro_f1`：validation=`0.6447`，full-test=`0.7182`，差值=`+0.0734`
- 字段名 `peak_center_hetero_f1`：validation=`0.8829`，full-test=`0.8841`，差值=`+0.0011`
- 字段名 `peak_center_edge_f1`：validation=`0.8943`，full-test=`0.8979`，差值=`+0.0036`
- 字段名 `peak_center_shift_px`：validation=`1.7268`，full-test=`1.7182`，差值=`-0.0086`
- 字段名 `atom_z_mae_r3`：validation=`0.0890`，full-test=`0.0910`，差值=`+0.0020`

## 四、代表样本
- 最佳样本：`dataset_index=277`，`cid=99825228`，`pred_object_score=0.9675`，`pred_object_type_acc=0.9600`，`pred_object_macro_f1=0.9787`，`pred_object_edge_f1=0.9091`，`pred_object_edge_f1_robust=0.9804`，`pred_object_z_mae=0.0158`，`peak_object_score=0.9504`，`peak_center_type_acc=0.9600`，`peak_center_edge_f1=1.0000`
- 中位样本：`dataset_index=255`，`cid=99801751`，`pred_object_score=0.7184`，`pred_object_type_acc=0.6857`，`pred_object_macro_f1=0.5613`，`pred_object_edge_f1=0.6585`，`pred_object_edge_f1_robust=0.9714`，`pred_object_z_mae=0.0536`，`peak_object_score=0.8511`，`peak_center_type_acc=0.8286`，`peak_center_edge_f1=0.8916`
- 最差样本：`dataset_index=198`，`cid=9975270`，`pred_object_score=0.4781`，`pred_object_type_acc=0.4762`，`pred_object_macro_f1=0.2354`，`pred_object_edge_f1=0.5607`，`pred_object_edge_f1_robust=0.9630`，`pred_object_z_mae=0.1610`，`peak_object_score=0.7398`，`peak_center_type_acc=0.7381`，`peak_center_edge_f1=0.7679`

## 五、核心判断
- 这个 full-test 报告以 `pred_object_score` 作为样本排序主指标，因为它最贴近真实闭环推理条件。
- `peak_*` 指标表示以 peak-center 为锚点的上限式对象级能力，`pred_object_*` 指标表示纯预测对象闭环能力。
- 如果 `pred_object_edge_f1_robust` 明显高于 `pred_object_edge_f1`，说明局部邻接恢复能力强于严格对象对应表现。
- 当前最重要的 gap 仍是 `peak/gt` 到 `pred-object` 的迁移损失，而不是中心、z 或局部邻接完全失效。

## 六、字段说明
- 字段名 `pred_object_score`：纯预测对象闭环对象级总分，越高越好
- 字段名 `pred_object_3d_score`：纯预测对象3D综合分，越高越好
- 字段名 `pred_object_count_mae`：纯预测对象原子数平均绝对误差，越低越好
- 字段名 `pred_object_count_score`：纯预测对象原子数相似度分数，越高越好
- 字段名 `pred_object_center_score`：纯预测对象proposal中心平均置信度，越高越好
- 字段名 `pred_object_type_acc`：纯预测对象原子类型准确率，越高越好
- 字段名 `pred_object_macro_f1`：纯预测对象原子类型宏平均F1，越高越好
- 字段名 `pred_object_hetero_f1`：纯预测对象杂原子F1，越高越好
- 字段名 `pred_object_edge_f1`：纯预测对象严格对象级边F1，越高越好
- 字段名 `pred_object_edge_f1_robust`：纯预测对象距离容忍后的稳健边F1，越高越好
- 字段名 `pred_object_match_coverage_robust`：稳健匹配覆盖率，越高越好
- 字段名 `pred_object_graph_score`：纯预测对象图结构综合分，越高越好
- 字段名 `pred_object_heavy_rmsd`：纯预测对象重原子RMSD，越低越好
- 字段名 `pred_object_z_mae`：纯预测对象z平均绝对误差，越低越好
- 字段名 `peak_object_score`：peak-center条件对象级总分，越高越好
- 字段名 `gt_object_score`：GT-center条件对象级总分，表示上限参考，越高越好
- 字段名 `atom_center_score_r3`：真实原子中心半径3像素内中心命中分数，越高越好
- 字段名 `typed_center_score_r3`：真实原子中心半径3像素内位置与类型同时正确的软分数，越高越好
- 字段名 `atom_type_macro_f1_2d`：稠密2D类型图宏平均F1，越高越好
- 字段名 `atom_xy_mae`：稠密2D原子图平均绝对误差，越低越好
- 字段名 `z_map_mae`：稠密z图平均绝对误差，越低越好
- 字段名 `atom_z_mae_r3`：真实中心附近z平均绝对误差，越低越好
- 字段名 `peak_center_type_acc`：peak-center条件原子类型准确率，越高越好
- 字段名 `peak_center_macro_f1`：peak-center条件原子类型宏平均F1，越高越好
- 字段名 `peak_center_hetero_f1`：peak-center条件杂原子F1，越高越好
- 字段名 `peak_center_edge_f1`：peak-center条件对象级边F1，越高越好
- 字段名 `peak_center_shift_px`：peak-center相对真实中心平均偏移像素，越低越好
- 字段名 `gt_center_type_acc`：GT-center条件原子类型准确率，越高越好
- 字段名 `gt_center_macro_f1`：GT-center条件原子类型宏平均F1，越高越好
- 字段名 `gt_center_hetero_f1`：GT-center条件杂原子F1，越高越好
- 字段名 `gt_center_edge_f1`：GT-center条件对象级边F1，越高越好
