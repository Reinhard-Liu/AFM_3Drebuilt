#!/bin/bash
# 生成更有代表性的测试集可视化样本

echo "=========================================="
echo "  生成代表性测试集可视化"
echo "=========================================="
echo ""

# 方案1: 随机采样10个样本（更有代表性）
echo "[方案1] 随机采样10个测试样本"
python3 << 'EOF'
import numpy as np
np.random.seed(42)
indices = sorted(np.random.choice(1000, 10, replace=False))
print("随机选择的索引:", indices)

# 保存到文件供后续使用
import json
with open('random_test_indices.json', 'w') as f:
    json.dump(indices.tolist(), f)
EOF

echo ""
echo "✓ 随机索引已保存到 random_test_indices.json"
echo ""

# 方案2: 分层采样（好、中、差各几个）
echo "[方案2] 如需分层采样，请根据predictions_diffusion.json中的RMSD值手动选择"
echo "  - 建议选择: RMSD < 1Å (优秀), 1-10Å (中等), >10Å (困难) 各3-4个样本"

echo ""
echo "=========================================="
echo "  建议："
echo "=========================================="
echo ""
echo "1. 学术用途: 展示test_predictions + 报告完整统计"
echo "2. 演示用途: 可以展示前10个优秀案例，但要注明"
echo "3. 全面评估: 使用方案1的随机采样结果"
echo ""
