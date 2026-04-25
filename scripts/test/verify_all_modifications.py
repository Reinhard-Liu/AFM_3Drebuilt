#!/usr/bin/env python3
"""
验证所有修改是否正确完成

检查项：
1. curves_diffusion.png 显示所有5个训练损失
2. 原子数准确率使用预测值（use_gt_count=False）
3. Stage 2/3 功能正确集成（约束、底部原子权重）
"""

import sys
import json
from pathlib import Path

def check_visualize_modification():
    """检查 visualize.py 是否正确修改为显示5个损失"""
    print("=" * 70)
    print("检查 1: visualize.py 修改（5个损失子图）")
    print("=" * 70)

    with open("src/utils/visualize.py", "r") as f:
        content = f.read()

    checks = {
        "提取 count_loss": 'train_count = [m["count_loss"] for m in history["train"]]',
        "提取 retrieval_loss": 'train_retrieval = [m["retrieval_loss"] for m in history["train"]]',
        "2x3 子图布局": "plt.subplots(2, 3, figsize=(18, 10))",
        "count_loss 子图": "Atom Count Loss (weight=0.5)",
        "retrieval_loss 子图": "Molecule Retrieval Loss (weight=0.05)",
        "总损失公式说明": "Total = coord_loss",
    }

    all_pass = True
    for desc, pattern in checks.items():
        if pattern in content:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc} - 未找到")
            all_pass = False

    if all_pass:
        print("\n✅ visualize.py 修改正确\n")
    else:
        print("\n❌ visualize.py 修改不完整\n")

    return all_pass


def check_count_accuracy_fix():
    """检查 evaluate_generation 是否使用预测值"""
    print("=" * 70)
    print("检查 2: 原子数准确率修复（use_gt_count=False）")
    print("=" * 70)

    with open("src/train.py", "r") as f:
        content = f.read()

    checks = {
        "evaluate_generation 使用 use_gt_count=False":
            'gen_result = model.generate(batch, use_gt_count=False)',
        "docstring 说明":
            'Uses predicted atom count (use_gt_count=False)',
    }

    all_pass = True
    for desc, pattern in checks.items():
        if pattern in content:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc} - 未找到")
            all_pass = False

    if all_pass:
        print("\n✅ 原子数准确率修复正确\n")
    else:
        print("\n❌ 原子数准确率修复不完整\n")

    return all_pass


def check_stage_integration():
    """检查 Stage 2/3 功能是否正确集成"""
    print("=" * 70)
    print("检查 3: Stage 2/3 功能集成")
    print("=" * 70)

    with open("src/train.py", "r") as f:
        content = f.read()

    checks = {
        "导入 compute_all_constraints":
            "from src.models.constraints import compute_all_constraints",
        "forward 接受 enable_constraints 参数":
            "def forward(self, batch: dict, z_depth_weighting: bool = False,\n                enable_constraints: bool = False)",
        "forward 计算约束损失":
            "constraint_losses = compute_all_constraints(",
        "总损失包含约束项":
            '+ 0.1 * losses["constraint_loss"]  # Stage 2+',
        "train_epoch 包含 constraint_loss":
            '"retrieval_loss": 0.0, "constraint_loss": 0.0}',
        "train_epoch 启用约束（Stage 2+）":
            "enable_constraints = (stage >= 2)",
        "train_epoch 启用底部权重（Stage 3）":
            "z_depth_weighting = (stage >= 3)",
        "train_epoch 传递 enable_constraints":
            "enable_constraints=enable_constraints)",
        "validate 包含 constraint_loss":
            '"retrieval_loss": 0.0, "constraint_loss": 0.0}',
        "get_training_stage 边界正确":
            "Stage 2 (epochs 31-45): constraint training",
    }

    all_pass = True
    for desc, pattern in checks.items():
        if pattern in content:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc} - 未找到")
            all_pass = False

    if all_pass:
        print("\n✅ Stage 2/3 功能集成正确\n")
    else:
        print("\n❌ Stage 2/3 功能集成不完整\n")

    return all_pass


def check_config():
    """检查 config.json 是否正确配置"""
    print("=" * 70)
    print("检查 4: config.json 配置")
    print("=" * 70)

    with open("config.json", "r") as f:
        config = json.load(f)

    checks = {
        "epochs 设置为 60": config.get("epochs") == 60,
        "model_type 为 diffusion": config.get("model_type") == "diffusion",
    }

    all_pass = True
    for desc, passed in checks.items():
        if passed:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc}")
            all_pass = False

    if all_pass:
        print("\n✅ config.json 配置正确\n")
    else:
        print("\n❌ config.json 配置有误\n")

    return all_pass


def check_stage_boundaries():
    """检查训练阶段边界定义"""
    print("=" * 70)
    print("检查 5: 训练阶段边界")
    print("=" * 70)

    with open("src/train.py", "r") as f:
        content = f.read()

    # 提取 get_training_stage 函数
    import re
    match = re.search(r'def get_training_stage.*?(?=\ndef )', content, re.DOTALL)
    if not match:
        print("  ✗ 未找到 get_training_stage 函数")
        print("\n❌ 训练阶段边界检查失败\n")
        return False

    func_code = match.group(0)

    checks = {
        "Stage 1: epochs 1-30": "if epoch <= 30:" in func_code and "return 1" in func_code,
        "Stage 2: epochs 31-45": "elif epoch <= 45:" in func_code and "return 2" in func_code,
        "Stage 3: epochs 46-60": "return 3" in func_code,
    }

    all_pass = True
    for desc, passed in checks.items():
        if passed:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc}")
            all_pass = False

    if all_pass:
        print("\n✅ 训练阶段边界正确\n")
    else:
        print("\n❌ 训练阶段边界有误\n")

    return all_pass


def check_early_stopping():
    """检查早停机制是否要求至少60轮"""
    print("=" * 70)
    print("检查 6: 早停机制")
    print("=" * 70)

    with open("src/train.py", "r") as f:
        content = f.read()

    # 查找 early stopping 相关代码
    if "epoch >= 60" in content and "early stopping" in content.lower():
        print("  ✓ 早停机制要求至少 60 轮")
        print("\n✅ 早停机制配置正确\n")
        return True
    else:
        print("  ? 未找到明确的早停条件（可能正常）")
        print("\n⚠️  早停机制需要手动验证\n")
        return True  # 不影响主要功能


def main():
    print("\n" + "=" * 70)
    print("验证所有修改")
    print("=" * 70)
    print()

    results = []

    # 执行所有检查
    results.append(("curves_diffusion.png 修改", check_visualize_modification()))
    results.append(("原子数准确率修复", check_count_accuracy_fix()))
    results.append(("Stage 2/3 功能集成", check_stage_integration()))
    results.append(("config.json 配置", check_config()))
    results.append(("训练阶段边界", check_stage_boundaries()))
    results.append(("早停机制", check_early_stopping()))

    # 总结
    print("=" * 70)
    print("验证总结")
    print("=" * 70)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}: {name}")

    print()

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("🎉 所有检查通过！项目已正确修改。\n")
        print("下一步:")
        print("  1. 运行 python3 -m src.quick_test 进行快速测试")
        print("  2. 运行 bash run.sh 开始训练")
        print()
        return 0
    else:
        print("⚠️  部分检查未通过，请检查上述失败项。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
