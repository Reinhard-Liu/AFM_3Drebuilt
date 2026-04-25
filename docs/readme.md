  项目总览                                                                                                                                                     
                                                                                                                                                               
  这是一个从 AFM 图像重建 3D 分子结构的项目，核心是 Video ViT 编码器 + 条件扩散模型（DDPM）。                                                                  

  执行流程                                                                                                                                                     
                                                                                                                                                               
  入口：run.sh                                                                                                                                                 

  依次执行三个步骤：

  [1/3] 训练扩散模型 (Video ViT + Conditional DDPM)
  [2/3] 训练基线模型 (3D-ResNet)
  [3/3] 生成可视化曲线

  详细流程

  run.sh
    │
    ├─→ python3 -m src.train --model_type diffusion
    ├─→ python3 -m src.train --model_type resnet3d
    └─→ 绘制训练曲线

  每次 src.train 内部的流程：

  1. 解析参数 + 加载配置
          ↓
  2. 数据加载 (dataset.py)
     ├─ 扫描 XYZ_FILES/ 和 K-1/ 目录，匹配有效样本
     ├─ pkl 缓存加速（首次扫描后保存）
     ├─ corrugation 过滤 → 排序 → 80/10/10 划分
     └─ 输出: afm_stack(10,128,128) + coords(85,3) + atom_types(85)
          ↓
  3. 创建模型
     ├─ diffusion 模式:
     │   ├─ VideoViTEncoder: AFM图像 → 条件向量 c (512维)
     │   └─ ConditionalDDPM: 条件扩散去噪，预测坐标+原子类型
     └─ resnet3d 模式:
         └─ ResNet3DRegression: 直接回归坐标+类型
          ↓
  4. 训练循环 (100 epochs)
     ├─ 前向传播 → 计算 loss (coord_loss + 0.1 × type_loss)
     ├─ 反向传播 + 梯度裁剪 (max_norm=1.0)
     ├─ AdamW 优化器 + CosineAnnealing 调度器
     ├─ 每 epoch 验证，保存最优 checkpoint
     └─ 每 10 epochs 打印日志
          ↓
  5. 保存结果
     ├─ best_diffusion.pt / best_resnet3d.pt (模型权重)
     ├─ history_diffusion.json (训练历史)
     └─ 扩散模型额外做生成评估 → RMSD + bottom-atom recall

  关键模块对应关系
  ┌────────────┬─────────────────────────┐
  │    步骤    │          文件           │
  ├────────────┼─────────────────────────┤
  │ 入口脚本   │ run.sh                  │
  ├────────────┼─────────────────────────┤
  │ 训练主程序 │ src/train.py            │
  ├────────────┼─────────────────────────┤
  │ 数据加载   │ src/data/dataset.py     │
  ├────────────┼─────────────────────────┤
  │ ViT 编码器 │ src/models/video_vit.py │
  ├────────────┼─────────────────────────┤
  │ 扩散模型   │ src/models/diffusion.py │
  ├────────────┼─────────────────────────┤
  │ 基线模型   │ src/models/baselines.py │
  ├────────────┼─────────────────────────┤
  │ 评估指标   │ src/utils/metrics.py    │
  ├────────────┼─────────────────────────┤
  │ 可视化     │ src/utils/visualize.py  │
  ├────────────┼─────────────────────────┤
  │ 快速测试   │ src/quick_test.py       │
  └────────────┴─────────────────────────┘

  # 修改

  1. 所有训练参数都通过 config.json 控制，不需要再改命令行。需要训练 resnet3d baseline 时，把 config.json 里的
  "model_type" 改成 "resnet3d" 再跑一次即可。

  2. 可以在 config.json 里加一个 "max_samples" 字段，设为 0                          
  用全量数据，设为具体数字就只用那么多条。    

  3. 你可以在 config.json 里设置 min_corrugation 来只保留起伏大的样本，比如：

  "min_corrugation": 2.0

  这样只会保留 Z 轴起伏 >= 2.0 Å 的分子。当前设为 0.0，即不过滤。

  4. 绘图代码

  python -m src.visualize_val \
        --checkpoint /root/autodl-tmp/micro/checkpoints/epoch_40_diffusion.pt \
        --num_samples 100 \
        --output_dir micro/val_diffusion