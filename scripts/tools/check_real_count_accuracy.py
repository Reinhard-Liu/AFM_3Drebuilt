#!/usr/bin/env python3
"""
检查原子数预测的真实准确率

这个脚本读取predictions_diffusion.json和测试集数据，
计算模型真实的原子数预测准确率（不使用ground truth）。
"""

import json
import sys
import os
sys.path.insert(0, '.')

import numpy as np
import torch
from src.data.dataset import QUAMAFMDataset


def load_test_dataset(config_path="config.json"):
    """加载测试集以获取真实原子数"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_root = config.get("data_root")
    if data_root is None or data_root == "auto":
        data_root = os.path.join(os.path.dirname(__file__),
                                 "dataverse_files", "SUBMIT_QUAM-AFM", "QUAM")

    dataset = QUAMAFMDataset(
        data_root=data_root,
        param_key=config.get("param_key", "K-1"),
        img_size=config.get("img_size", 128),
        min_corrugation=config.get("min_corrugation", 0.0),
        augment_rotation=False,
        split="test",
        val_size=config.get("val_size", 0),
        max_samples=config.get("max_samples", 0),
    )
    return dataset


def main():
    print("=" * 80)
    print("  原子数预测真实准确率检查")
    print("=" * 80)
    print()

    # 读取预测结果
    pred_path = "checkpoints/predictions_diffusion.json"
    if not os.path.exists(pred_path):
        print(f"错误: 找不到预测文件 {pred_path}")
        print("请先运行训练生成预测结果")
        return

    with open(pred_path, 'r') as f:
        pred_data = json.load(f)

    predictions = pred_data['predictions']
    num_samples = len(predictions)
    print(f"预测样本数: {num_samples}")

    # 尝试加载测试集
    try:
        test_dataset = load_test_dataset()
        print(f"测试集样本数: {len(test_dataset)}")
        print()
    except Exception as e:
        print(f"无法加载测试集: {e}")
        print("将使用可视化映射文件作为备选")
        print()

        # 使用index_mapping.json作为备选
        mapping_path = "visualizations/test_predictions/index_mapping.json"
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)

            num_to_check = min(len(mapping), num_samples)
            print(f"检查前 {num_to_check} 个样本:")
            print("-" * 80)
            print(f"{'ID':>4} | {'真实':>6} | {'预测':>6} | {'误差':>6} | {'准确?':>6}")
            print("-" * 80)

            exact = 0
            errors = []
            for i in range(num_to_check):
                gt = mapping[i]['n_atoms']
                pred = predictions[i]['n_atoms_pred']
                diff = pred - gt
                match = "✓" if diff == 0 else "✗"
                if diff == 0:
                    exact += 1
                else:
                    errors.append(diff)
                print(f"{i:4d} | {gt:6d} | {pred:6d} | {diff:6d} | {match:>6}")

            print("-" * 80)
            print(f"\n完全匹配: {exact}/{num_to_check} ({exact/num_to_check*100:.1f}%)")
            if errors:
                print(f"平均绝对误差: {np.mean(np.abs(errors)):.2f}")
        else:
            print("找不到真实数据，无法验证")
        return

    # 使用完整测试集验证
    print("逐样本对比:")
    print("-" * 80)
    print(f"{'样本ID':>6} | {'真实原子数':>10} | {'预测原子数':>10} | {'误差':>6} | {'准确?':>6}")
    print("-" * 80)

    exact_matches = 0
    total_checked = min(num_samples, len(test_dataset))
    all_errors = []

    for i in range(total_checked):
        sample = test_dataset[i]
        gt_n_atoms = sample['n_atoms'].item()
        pred_n_atoms = predictions[i]['n_atoms_pred']

        error = pred_n_atoms - gt_n_atoms
        is_exact = (error == 0)

        if is_exact:
            exact_matches += 1
            mark = "✓"
        else:
            mark = "✗"
            all_errors.append(error)

        # 只显示前20个样本，避免输出过长
        if i < 20:
            print(f"{i:6d} | {gt_n_atoms:10d} | {pred_n_atoms:10d} | {error:6d} | {mark:>6}")
        elif i == 20:
            print("  ... (省略中间样本，仅显示统计结果)")

    print("-" * 80)
    print()

    # 统计结果
    print("=" * 80)
    print("  统计结果")
    print("=" * 80)
    print(f"总样本数: {total_checked}")
    print(f"完全匹配: {exact_matches} ({exact_matches/total_checked*100:.2f}%)")
    print(f"有误差的: {total_checked - exact_matches} ({(total_checked-exact_matches)/total_checked*100:.2f}%)")

    if all_errors:
        all_errors_abs = [abs(e) for e in all_errors]
        print()
        print("误差分析:")
        print(f"  平均绝对误差 (MAE): {np.mean(all_errors_abs):.2f} 个原子")
        print(f"  最大正误差: +{max(all_errors)} 个原子")
        print(f"  最大负误差: {min(all_errors)} 个原子")
        print(f"  误差标准差: {np.std(all_errors):.2f}")

        # 误差分布
        error_counts = {}
        for e in all_errors:
            error_counts[e] = error_counts.get(e, 0) + 1

        print()
        print("误差分布:")
        for err in sorted(error_counts.keys()):
            count = error_counts[err]
            pct = count / len(all_errors) * 100
            bar = "█" * int(pct / 2)
            print(f"  {err:+3d} 个原子: {count:4d} ({pct:5.1f}%) {bar}")

    print()
    print("=" * 80)
    print("  结论")
    print("=" * 80)
    print()
    accuracy_pct = exact_matches / total_checked * 100

    if accuracy_pct > 80:
        print("✓ 原子数预测准确率很高 (>80%)")
    elif accuracy_pct > 50:
        print("○ 原子数预测准确率中等 (50-80%)")
    else:
        print("⚠ 原子数预测准确率较低 (<50%)")

    print(f"  当前准确率: {accuracy_pct:.2f}%")

    if all_errors:
        mae = np.mean(all_errors_abs)
        if mae < 1.0:
            print(f"✓ 平均误差较小 (<1个原子): {mae:.2f}")
        elif mae < 2.0:
            print(f"○ 平均误差中等 (1-2个原子): {mae:.2f}")
        else:
            print(f"⚠ 平均误差较大 (>2个原子): {mae:.2f}")

    print()
    print("注意: 这是模型端到端预测的真实表现 (use_gt_count=False)")
    print("      训练日志中的 100% 准确率使用了 use_gt_count=True")
    print()


if __name__ == "__main__":
    main()
