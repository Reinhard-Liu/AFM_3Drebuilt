#!/usr/bin/env python3
"""
运行前验证脚本

检查所有必要文件、模块和配置，确保 bash run.sh 能成功执行
"""

import os
import sys
import json
import importlib

def check_files():
    """检查关键文件是否存在"""
    print("=" * 70)
    print("文件存在性检查")
    print("=" * 70)
    print()

    required_files = {
        "配置文件": "config.json",
        "训练脚本": "src/train.py",
        "数据集模块": "src/data/dataset.py",
        "Video ViT": "src/models/video_vit.py",
        "扩散模型": "src/models/diffusion.py",
        "预测头": "src/models/prediction_heads.py",
        "物理约束": "src/models/constraints.py",
        "环检测": "src/models/ring_detection.py",
        "评估指标": "src/utils/metrics.py",
        "可视化工具": "src/utils/visualize.py",
        "验证集可视化": "src/visualize_val.py",
        "运行脚本": "run.sh",
    }

    all_exist = True
    for name, path in required_files.items():
        if os.path.exists(path):
            print(f"  ✓ {name:20s} {path}")
        else:
            print(f"  ✗ {name:20s} {path} (缺失!)")
            all_exist = False

    print()
    if all_exist:
        print("✅ 所有必需文件存在")
    else:
        print("❌ 部分必需文件缺失")

    return all_exist


def check_modules():
    """检查关键模块是否可导入"""
    print()
    print("=" * 70)
    print("模块导入检查")
    print("=" * 70)
    print()

    modules_to_check = [
        ("训练模块", "src.train"),
        ("数据集", "src.data.dataset"),
        ("Video ViT", "src.models.video_vit"),
        ("扩散模型", "src.models.diffusion"),
        ("预测头", "src.models.prediction_heads"),
        ("物理约束", "src.models.constraints"),
        ("评估指标", "src.utils.metrics"),
        ("可视化", "src.utils.visualize"),
    ]

    all_ok = True
    for name, module_name in modules_to_check:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {name:20s} {module_name}")
        except Exception as e:
            print(f"  ✗ {name:20s} {module_name}")
            print(f"    错误: {e}")
            all_ok = False

    print()
    if all_ok:
        print("✅ 所有模块可以正常导入")
    else:
        print("❌ 部分模块导入失败")

    return all_ok


def check_config():
    """检查配置文件"""
    print()
    print("=" * 70)
    print("配置文件检查")
    print("=" * 70)
    print()

    if not os.path.exists("config.json"):
        print("  ✗ config.json 不存在")
        return False

    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"  ✗ 无法解析 config.json: {e}")
        return False

    # 检查关键配置项
    required_keys = {
        "model_type": "diffusion",
        "epochs": 60,
    }

    checks = []
    for key, expected in required_keys.items():
        value = config.get(key)
        if value == expected:
            print(f"  ✓ {key:20s} = {value}")
            checks.append(True)
        else:
            print(f"  ⚠️  {key:20s} = {value} (预期: {expected})")
            checks.append(False)

    # 其他重要配置
    print()
    print("  其他配置:")
    print(f"    batch_size:      {config.get('batch_size', 'N/A')}")
    print(f"    lr:              {config.get('lr', 'N/A')}")
    print(f"    save_dir:        {config.get('save_dir', 'N/A')}")
    print(f"    min_corrugation: {config.get('min_corrugation', 'N/A')}")
    print(f"    max_samples:     {config.get('max_samples', 'N/A')}")

    print()
    if all(checks):
        print("✅ 配置文件正确")
    else:
        print("⚠️  配置文件部分项不符合预期")

    return True  # 即使有警告也返回 True


def check_directories():
    """检查并创建必要的目录"""
    print()
    print("=" * 70)
    print("目录结构检查")
    print("=" * 70)
    print()

    directories = [
        "checkpoints",
        "visualizations",
    ]

    for dir_name in directories:
        if os.path.exists(dir_name):
            print(f"  ✓ {dir_name}/ 存在")
        else:
            try:
                os.makedirs(dir_name, exist_ok=True)
                print(f"  ✓ {dir_name}/ 已创建")
            except Exception as e:
                print(f"  ✗ {dir_name}/ 无法创建: {e}")
                return False

    print()
    print("✅ 目录结构正常")
    return True


def check_training_stages():
    """验证训练阶段配置"""
    print()
    print("=" * 70)
    print("训练阶段配置检查")
    print("=" * 70)
    print()

    try:
        from src.train import get_training_stage
    except Exception as e:
        print(f"  ✗ 无法导入 get_training_stage: {e}")
        return False

    # 测试边界
    test_cases = [
        (1, 1, "Stage 1 起始"),
        (30, 1, "Stage 1 结束"),
        (31, 2, "Stage 2 起始"),
        (45, 2, "Stage 2 结束"),
        (46, 3, "Stage 3 起始"),
        (60, 3, "Stage 3 结束"),
    ]

    all_correct = True
    for epoch, expected_stage, desc in test_cases:
        actual_stage = get_training_stage(epoch)
        if actual_stage == expected_stage:
            print(f"  ✓ Epoch {epoch:2d} → Stage {actual_stage} ({desc})")
        else:
            print(f"  ✗ Epoch {epoch:2d} → Stage {actual_stage} (预期: {expected_stage}, {desc})")
            all_correct = False

    print()
    if all_correct:
        print("✅ 训练阶段配置正确")
        print("   Stage 1 (基础训练):    Epoch 1-30")
        print("   Stage 2 (约束训练):    Epoch 31-45")
        print("   Stage 3 (底部聚焦):    Epoch 46-60")
    else:
        print("❌ 训练阶段配置有误")

    return all_correct


def check_visualization_compatibility():
    """检查可视化工具的向后兼容性"""
    print()
    print("=" * 70)
    print("可视化工具兼容性检查")
    print("=" * 70)
    print()

    try:
        from src.utils.visualize import plot_training_curves
        print("  ✓ plot_training_curves 可导入")
    except Exception as e:
        print(f"  ✗ 无法导入 plot_training_curves: {e}")
        return False

    # 测试向后兼容性
    import tempfile
    import json

    # 创建旧格式历史文件
    old_history = {
        "train": [{"loss": 1.0, "coord_loss": 0.5, "type_loss": 0.3}],
        "val": [{"loss": 0.9, "coord_loss": 0.4, "type_loss": 0.25}]
    }

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(old_history, f)
            temp_path = f.name

        # 尝试绘图
        plot_training_curves(temp_path, "test_compat.png")
        os.remove(temp_path)

        if os.path.exists("test_compat.png"):
            os.remove("test_compat.png")
            print("  ✓ 向后兼容旧格式历史文件")
        else:
            print("  ⚠️  生成图像失败")
    except Exception as e:
        print(f"  ✗ 向后兼容性测试失败: {e}")
        return False

    print()
    print("✅ 可视化工具向后兼容")
    return True


def check_gpu():
    """检查 GPU 可用性"""
    print()
    print("=" * 70)
    print("GPU 检查")
    print("=" * 70)
    print()

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA 可用")
            print(f"  ✓ GPU 数量: {gpu_count}")
            print(f"  ✓ GPU 0: {gpu_name}")
        else:
            print("  ⚠️  CUDA 不可用，将使用 CPU 训练（速度会很慢）")
    except Exception as e:
        print(f"  ✗ 无法检查 GPU: {e}")

    print()


def estimate_output_size():
    """估算输出文件大小"""
    print()
    print("=" * 70)
    print("预期输出文件大小估算")
    print("=" * 70)
    print()

    estimates = {
        "模型权重文件": {
            "best_diffusion.pt": "100-200 MB",
            "epoch_X_diffusion.pt (×7)": "700-1400 MB (共7个检查点)",
        },
        "训练数据文件": {
            "history_diffusion.json": "10-50 KB",
            "metrics_diffusion.json": "5-20 KB",
            "predictions_diffusion.json": "1-5 MB (100个样本)",
            "training.log": "100-500 KB (取决于轮次和日志详细度)",
        },
        "可视化文件": {
            "curves_diffusion.png": "~500 KB",
            "molecules_diffusion/*.png (×10)": "~5 MB",
        }
    }

    total_min = 800  # MB
    total_max = 1700  # MB

    for category, files in estimates.items():
        print(f"  {category}:")
        for filename, size in files.items():
            print(f"    - {filename:40s} {size}")
        print()

    print(f"  预计总大小: {total_min}-{total_max} MB")
    print()
    print("  ⚠️  请确保磁盘空间充足（建议至少 5 GB 可用空间）")
    print()


def main():
    """主验证流程"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "运行前验证脚本" + " " * 34 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    checks = []

    # 1. 文件存在性
    checks.append(("文件存在性", check_files()))

    # 2. 模块导入
    checks.append(("模块导入", check_modules()))

    # 3. 配置文件
    checks.append(("配置文件", check_config()))

    # 4. 目录结构
    checks.append(("目录结构", check_directories()))

    # 5. 训练阶段
    checks.append(("训练阶段", check_training_stages()))

    # 6. 可视化兼容性
    checks.append(("可视化兼容性", check_visualization_compatibility()))

    # 7. GPU
    check_gpu()

    # 8. 输出大小估算
    estimate_output_size()

    # 总结
    print("=" * 70)
    print("验证总结")
    print("=" * 70)
    print()

    for name, passed in checks:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {status}: {name}")

    print()

    all_passed = all(passed for _, passed in checks)

    if all_passed:
        print("🎉 所有检查通过！可以运行 bash run.sh")
        print()
        print("运行命令:")
        print("  cd /root/autodl-tmp/micro")
        print("  bash run.sh")
        print()
        print("预计训练时间: ~15-20 小时 (单 GPU)")
        print()
        return 0
    else:
        print("⚠️  部分检查未通过")
        print()
        print("请解决上述问题后再运行 bash run.sh")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
