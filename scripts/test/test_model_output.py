#!/usr/bin/env python3
"""
测试模型输出是否包含项目改进方案要求的所有字段
"""

import torch
from src.train import AFM3DReconModel
from src.data.dataset import QUAMAFMDataset, ATOM_TYPES

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("测试模型输出完整性")
    print("="*60)

    # 1. 加载模型
    print("\n1. 加载训练好的模型...")
    checkpoint = torch.load('checkpoints/best_diffusion.pt', map_location=device)
    config = checkpoint['config']
    model = AFM3DReconModel(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"   ✓ 模型已加载 (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")

    # 2. 加载验证数据
    print("\n2. 加载验证数据...")
    dataset = QUAMAFMDataset(
        data_root=config['data_root'],
        param_key=config['param_key'],
        img_size=config['img_size'],
        min_corrugation=config['min_corrugation'],
        augment_rotation=False,
        split='val',
        val_size=config['val_size'],
        max_samples=config['max_samples'],
    )
    print(f"   ✓ 验证集大小: {len(dataset)} 个样本")

    # 3. 选择一个样本进行测试
    print("\n3. 测试样本 #0...")
    sample = dataset[0]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

    # 4. 生成预测
    print("\n4. 执行推理...")
    with torch.no_grad():
        result = model.generate(batch, use_gt_count=False)

    print("   ✓ 推理完成")

    # 5. 检查输出字段
    print("\n5. 检查输出字段:")
    print("-" * 60)

    required_fields = {
        "coords": "三维原子坐标",
        "type_logits": "原子类型 logits",
        "n_atoms_pred": "预测原子数",
        "retrieval_indices": "候选分子 CID (Top-5)"
    }

    all_present = True
    for field, desc in required_fields.items():
        if field in result:
            value = result[field]
            if isinstance(value, torch.Tensor):
                print(f"   ✓ {field:20s} ({desc})")
                print(f"     Shape: {value.shape}, dtype: {value.dtype}")
            else:
                print(f"   ✓ {field:20s} ({desc}): {type(value)}")
        else:
            print(f"   ✗ {field:20s} ({desc}): NOT FOUND")
            all_present = False

    # 6. 显示具体预测结果
    print("\n6. 预测结果示例:")
    print("-" * 60)

    # 预测原子数
    n_pred = result["n_atoms_pred"].item()
    n_gt = batch["n_atoms"].item()
    print(f"   预测原子数: {n_pred}")
    print(f"   真实原子数: {n_gt}")
    print(f"   准确性: {'✓ 完全正确' if n_pred == n_gt else f'✗ 误差 {abs(n_pred - n_gt)}'}")

    # 预测原子类型
    pred_types = result["type_logits"].argmax(dim=-1)[0]
    gt_types = batch["atom_types"][0]
    mask = batch["atom_mask"][0]

    print(f"\n   前 5 个原子的预测类型:")
    for i in range(min(5, n_pred)):
        if mask[i] > 0:
            pred_elem = ATOM_TYPES[pred_types[i].item()]
            gt_elem = ATOM_TYPES[gt_types[i].item()]
            match = "✓" if pred_elem == gt_elem else "✗"
            print(f"     原子 {i}: 预测={pred_elem:2s}, 真实={gt_elem:2s} {match}")

    # 坐标范围
    coords_pred = result["coords"][0]
    coords_gt = batch["coords"][0]
    print(f"\n   预测坐标范围: [{coords_pred.min():.3f}, {coords_pred.max():.3f}]")
    print(f"   真实坐标范围: [{coords_gt.min():.3f}, {coords_gt.max():.3f}]")

    # Top-5 候选 CID
    if "retrieval_indices" in result:
        top5_indices = result["retrieval_indices"][0].cpu().tolist()
        print(f"\n   Top-5 候选分子 CID 索引:")
        for rank, idx in enumerate(top5_indices, 1):
            score = result["retrieval_scores"][0, rank-1].item()
            print(f"     #{rank}: CID索引={idx}, 相似度={score:.4f}")

    # 7. 总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)

    if all_present:
        print("✅ 所有必需字段都存在")
        print("✅ 模型输出符合项目改进方案要求")
        print()
        print("输出字段包括:")
        print("  1. 三维原子坐标 (coords)")
        print("  2. 原子类型 (type_logits)")
        print("  3. 预测原子数 (n_atoms_pred)")
        print("  4. 候选分子 CID (retrieval_indices, Top-5)")
    else:
        print("❌ 缺少部分必需字段")
        print("请检查模型配置和训练过程")

    print("="*60)

if __name__ == "__main__":
    main()
