#!/usr/bin/env python3
"""测试旧格式历史文件的兼容性"""

import json
import os

# 创建模拟的旧格式历史文件（只有 3 个损失）
old_history = {
    "train": [
        {"loss": 2.74, "coord_loss": 0.14, "type_loss": 1.26},
        {"loss": 1.63, "coord_loss": 0.08, "type_loss": 0.75},
        {"loss": 1.20, "coord_loss": 0.06, "type_loss": 0.54},
    ],
    "val": [
        {"loss": 1.89, "coord_loss": 0.12, "type_loss": 0.87},
        {"loss": 1.45, "coord_loss": 0.09, "type_loss": 0.68},
        {"loss": 1.15, "coord_loss": 0.07, "type_loss": 0.51},
    ]
}

# 保存为临时文件
temp_history_path = "test_old_history.json"
with open(temp_history_path, "w") as f:
    json.dump(old_history, f, indent=2)

print("=" * 70)
print("测试旧格式历史文件")
print("=" * 70)
print()
print("创建模拟的旧格式历史文件（只有 loss, coord_loss, type_loss）")
print()

# 测试绘图
from src.utils.visualize import plot_training_curves

try:
    test_output = "test_old_format_curves.png"
    plot_training_curves(temp_history_path, test_output)

    if os.path.exists(test_output):
        print(f"✅ 成功生成旧格式图像: {test_output}")
        print()
        print("图像说明:")
        print("  - 前 3 个子图显示 Total Loss, Coord Loss, Type Loss")
        print("  - 后 2 个子图显示 '(Not available in old history file)'")
        print("  - Summary 显示旧格式说明")
        print()
        print(f"请查看图像: {test_output}")
    else:
        print("✗ 未能生成图像")
except Exception as e:
    print(f"✗ 绘图失败: {e}")
    import traceback
    traceback.print_exc()
finally:
    # 清理临时文件
    if os.path.exists(temp_history_path):
        os.remove(temp_history_path)

print()
print("=" * 70)
print("测试完成")
print("=" * 70)
