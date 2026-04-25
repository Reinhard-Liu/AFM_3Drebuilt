#!/bin/bash
# 生成测试集的可视化，对应 predictions_diffusion.json

echo "=========================================="
echo "  生成测试集可视化（对应 predictions.json）"
echo "=========================================="
echo ""

# 生成前10个测试样本的可视化
python3 visualize_test_predictions.py \
    --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 10 \
    --output_dir visualizations/test_predictions

echo ""
echo "完成！查看文件："
echo "  - 可视化图片: visualizations/test_predictions/"
echo "  - 索引映射: visualizations/test_predictions/index_mapping.json"
