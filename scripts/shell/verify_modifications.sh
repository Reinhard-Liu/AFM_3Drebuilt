#!/bin/bash
# 验证训练配置修改是否成功

echo "=========================================="
echo "  训练配置修改验证脚本"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 验证配置文件
echo "[1/5] 验证 config.json..."
EPOCHS=$(python3 -c "import json; print(json.load(open('config.json'))['epochs'])")
if [ "$EPOCHS" -ge 60 ]; then
    echo -e "${GREEN}✓ epochs = $EPOCHS (>= 60)${NC}"
else
    echo -e "${RED}✗ epochs = $EPOCHS (< 60)${NC}"
    exit 1
fi
echo ""

# 2. 验证训练阶段划分
echo "[2/5] 验证训练阶段划分..."
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from src.train import get_training_stage

# 测试关键轮次
test_cases = [
    (30, 1, "Stage 1 边界"),
    (31, 2, "Stage 2 开始"),
    (45, 2, "Stage 2 边界"),
    (46, 3, "Stage 3 开始"),
    (60, 3, "Stage 3 结束")
]

all_pass = True
for epoch, expected, desc in test_cases:
    result = get_training_stage(epoch)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_pass = False
    print(f"{status} Epoch {epoch}: Stage {result} (期望 {expected}) - {desc}")

sys.exit(0 if all_pass else 1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 训练阶段划分正确${NC}"
else
    echo -e "${RED}✗ 训练阶段划分错误${NC}"
    exit 1
fi
echo ""

# 3. 验证早停机制
echo "[3/5] 验证早停机制代码..."
if grep -q "epoch >= 60 and rmsd_mean < 1.0" src/train.py; then
    echo -e "${GREEN}✓ 早停机制已修改为 epoch >= 60${NC}"
else
    echo -e "${RED}✗ 早停机制未正确修改${NC}"
    exit 1
fi
echo ""

# 4. 验证模块导入
echo "[4/5] 验证核心模块导入..."
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

modules = [
    ("src.data.dataset", "QUAMAFMDataset"),
    ("src.models.video_vit", "VideoViTEncoder"),
    ("src.models.diffusion", "ConditionalDDPM"),
    ("src.models.prediction_heads", "AtomCountHead"),
    ("src.models.constraints", "compute_all_constraints"),
    ("src.models.ring_detection", "detect_rings"),
    ("src.train", "AFM3DReconModel"),
]

all_pass = True
for module_name, class_name in modules:
    try:
        exec(f"from {module_name} import {class_name}")
        print(f"✓ {module_name}")
    except Exception as e:
        print(f"✗ {module_name}: {e}")
        all_pass = False

sys.exit(0 if all_pass else 1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 所有核心模块导入成功${NC}"
else
    echo -e "${RED}✗ 模块导入失败${NC}"
    exit 1
fi
echo ""

# 5. 验证文档更新
echo "[5/5] 验证文档更新..."
if grep -q "31-45" /root/autodl-tmp/CLAUDE.md && \
   grep -q "46-60" /root/autodl-tmp/CLAUDE.md; then
    echo -e "${GREEN}✓ CLAUDE.md 已更新${NC}"
else
    echo -e "${YELLOW}⚠ CLAUDE.md 可能未更新${NC}"
fi

if grep -q "31~45" /root/autodl-tmp/项目改进方案.md && \
   grep -q "46~60" /root/autodl-tmp/项目改进方案.md; then
    echo -e "${GREEN}✓ 项目改进方案.md 已更新${NC}"
else
    echo -e "${YELLOW}⚠ 项目改进方案.md 可能未更新${NC}"
fi
echo ""

# 总结
echo "=========================================="
echo -e "${GREEN}  ✓ 所有验证通过！${NC}"
echo "=========================================="
echo ""
echo "修改总结："
echo "  - 训练轮次: 50 -> 60"
echo "  - Stage 1: 1-30 轮 (基础训练)"
echo "  - Stage 2: 31-45 轮 (约束训练)"
echo "  - Stage 3: 46-60 轮 (底部聚焦)"
echo "  - 早停条件: epoch >= 60 且 RMSD < 1.0"
echo ""
echo "可以开始训练："
echo "  cd /root/autodl-tmp/micro"
echo "  bash run.sh"
echo ""
