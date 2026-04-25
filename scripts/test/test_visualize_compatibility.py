#!/usr/bin/env python3
"""测试 visualize.py 的向后兼容性"""

import json
import os
import sys

def test_backward_compatibility():
    """测试旧格式历史文件的兼容性"""

    print("=" * 70)
    print("测试 visualize.py 向后兼容性")
    print("=" * 70)
    print()

    # 检查当前历史文件格式
    history_path = "checkpoints/history_diffusion.json"

    if not os.path.exists(history_path):
        print(f"⚠️  历史文件不存在: {history_path}")
        print("   这是正常的，如果还没有训练过的话")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    if not history.get("train"):
        print(f"⚠️  历史文件为空")
        return

    # 检查第一个 epoch 的数据
    first_epoch = history["train"][0]

    print(f"检查历史文件: {history_path}")
    print(f"训练轮次数: {len(history['train'])}")
    print()

    # 检查字段
    has_loss = "loss" in first_epoch
    has_coord = "coord_loss" in first_epoch
    has_type = "type_loss" in first_epoch
    has_count = "count_loss" in first_epoch
    has_retrieval = "retrieval_loss" in first_epoch

    print("数据字段检查:")
    print(f"  loss:           {'✓' if has_loss else '✗'}")
    print(f"  coord_loss:     {'✓' if has_coord else '✗'}")
    print(f"  type_loss:      {'✓' if has_type else '✗'}")
    print(f"  count_loss:     {'✓' if has_count else '✗'}")
    print(f"  retrieval_loss: {'✓' if has_retrieval else '✗'}")
    print()

    if has_count and has_retrieval:
        print("✅ 这是新格式历史文件（包含所有 5 个损失）")
        print("   visualize.py 将显示完整的 2×3 布局")
    else:
        print("⚠️  这是旧格式历史文件（只有 3 个损失）")
        print("   visualize.py 将:")
        print("   - 显示前 3 个损失（Total, Coord, Type）")
        print("   - 后 2 个子图显示 '(Not available in old history file)'")
        print("   - Summary 显示旧格式说明")
    print()

    # 测试绘图
    print("测试绘图功能...")
    try:
        from src.utils.visualize import plot_training_curves

        test_output = "test_curves.png"
        plot_training_curves(history_path, test_output)

        if os.path.exists(test_output):
            print(f"✅ 成功生成测试图像: {test_output}")
            # 清理
            os.remove(test_output)
        else:
            print("✗ 未能生成图像")
    except Exception as e:
        print(f"✗ 绘图失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 70)
    print("向后兼容性测试完成")
    print("=" * 70)
    print()

    if not has_count or not has_retrieval:
        print("💡 建议:")
        print("   要生成包含所有 5 个损失的新格式历史文件，")
        print("   请运行新的训练:")
        print()
        print("   bash run.sh")
        print()

    return True


if __name__ == "__main__":
    success = test_backward_compatibility()
    sys.exit(0 if success else 1)
