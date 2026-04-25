#!/bin/bash
# 生成更多可视化图片

# 生成 50 个样本的可视化
python3 -m src.visualize_val \
    --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 50 \
    --output_dir visualizations/molecules_diffusion_50

echo "已生成 50 个样本的可视化图片到: visualizations/molecules_diffusion_50/"

# 或者生成全部 100 个
# python3 -m src.visualize_val \
#     --checkpoint checkpoints/best_diffusion.pt \
#     --num_samples 100 \
#     --output_dir visualizations/molecules_diffusion_100
