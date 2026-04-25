#!/usr/bin/env python3
"""
查看预测分子的Top-5相似分子CID

用法：
    python 查看Top5相似分子.py                    # 查看前10个样本
    python 查看Top5相似分子.py --sample_id 5      # 查看指定样本
    python 查看Top5相似分子.py --num_samples 20   # 查看前20个样本
    python 查看Top5相似分子.py --save results.txt # 保存到文件
"""

import sys
import json
import argparse
import torch

sys.path.insert(0, '/root/autodl-tmp/micro')
from src.data.dataset import QUAMAFMDataset


def load_cid_mapping(config):
    """加载CID索引到真实CID的映射"""
    print("正在加载CID映射...", end=' ', flush=True)

    train_ds = QUAMAFMDataset(
        data_root=config['data_root'],
        param_key=config['param_key'],
        img_size=config['img_size'],
        min_corrugation=config['min_corrugation'],
        augment_rotation=False,
        split='train',
        val_size=config['val_size'],
        max_samples=0,
    )

    # 构建反向映射
    idx_to_cid = {idx: cid for cid, idx in train_ds.cid_to_idx.items()}
    print(f"完成 (共{len(idx_to_cid)}个CID)")

    return idx_to_cid


def display_top5_cids(sample, idx_to_cid, sample_id, output_file=None):
    """显示单个样本的Top-5相似分子"""
    cid_indices = sample['retrieval_cid_indices']
    scores = sample['retrieval_scores']
    n_atoms = sample['n_atoms_pred']

    lines = []
    lines.append("=" * 70)
    lines.append(f"测试样本 #{sample_id}")
    lines.append("=" * 70)
    lines.append(f"预测原子数: {n_atoms}\n")
    lines.append("Top-5 相似分子（按相似度降序）:")
    lines.append(f"{'排名':<6} {'PubChem CID':<20} {'内部索引':<12} {'相似度':<10}")
    lines.append("-" * 70)

    for rank, (idx, score) in enumerate(zip(cid_indices, scores), 1):
        real_cid = idx_to_cid.get(idx, f"Unknown(idx={idx})")
        lines.append(f"{rank:<6} {str(real_cid):<20} {idx:<12} {score:<10.4f}")

    lines.append("")

    output = "\n".join(lines)
    print(output)

    if output_file:
        output_file.write(output + "\n")

    return cid_indices, scores


def main():
    parser = argparse.ArgumentParser(description='查看预测分子的Top-5相似分子CID')
    parser.add_argument('--sample_id', type=int, default=None,
                        help='查看指定样本ID（默认：显示前10个）')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='查看的样本数量（默认：10）')
    parser.add_argument('--predictions', type=str,
                        default='/root/autodl-tmp/micro/checkpoints/predictions_diffusion.json',
                        help='预测结果文件路径')
    parser.add_argument('--checkpoint', type=str,
                        default='/root/autodl-tmp/micro/checkpoints/best_diffusion.pt',
                        help='模型检查点路径')
    parser.add_argument('--save', type=str, default=None,
                        help='保存结果到文件')

    args = parser.parse_args()

    # 加载配置
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    # 加载CID映射
    idx_to_cid = load_cid_mapping(config)

    # 加载预测结果
    print(f"正在加载预测结果: {args.predictions}")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)

    total_samples = len(predictions['predictions'])
    print(f"预测结果中共有 {total_samples} 个测试样本\n")

    # 打开输出文件（如果指定）
    output_file = None
    if args.save:
        output_file = open(args.save, 'w', encoding='utf-8')
        output_file.write("预测分子的Top-5相似分子CID查询结果\n")
        output_file.write("=" * 70 + "\n\n")

    # 显示结果
    if args.sample_id is not None:
        # 查看指定样本
        if args.sample_id >= total_samples:
            print(f"错误：样本ID {args.sample_id} 超出范围 (0-{total_samples-1})")
            return

        sample = predictions['predictions'][args.sample_id]
        display_top5_cids(sample, idx_to_cid, args.sample_id, output_file)
    else:
        # 查看前N个样本
        num_to_show = min(args.num_samples, total_samples)

        for i in range(num_to_show):
            sample = predictions['predictions'][i]
            display_top5_cids(sample, idx_to_cid, i, output_file)

    # 关闭输出文件
    if output_file:
        output_file.close()
        print(f"\n结果已保存到: {args.save}")

    # 提供PubChem查询链接示例
    print("\n" + "=" * 70)
    print("如何查看分子详细信息")
    print("=" * 70)
    print("在 PubChem 网站查看分子结构和属性：")

    sample_0 = predictions['predictions'][0]
    cid_example = idx_to_cid[sample_0['retrieval_cid_indices'][0]]
    print(f"  https://pubchem.ncbi.nlm.nih.gov/compound/{cid_example}")
    print("\n将上述URL中的CID替换为你想查询的CID即可")
    print("\n在数据集中查找对应的XYZ文件：")
    print(f"  data_root/K-1/CID_{cid_example}_*/*.xyz")
    print(f"  data_root/K-1/CID_{cid_example}_*/*_df_*.jpg  (AFM图像)")


if __name__ == "__main__":
    main()
