#!/usr/bin/env python3
"""
绘制 RMSD 曲线（与 curves_diffusion.png 对比）

curves_diffusion.png 显示的是训练损失（Loss）
本脚本生成 RMSD 曲线，展示评估指标的变化
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def plot_rmsd_curve():
    """从 metrics_diffusion.json 绘制 RMSD 曲线"""

    # 读取评估指标
    with open('checkpoints/metrics_diffusion.json', 'r') as f:
        metrics = json.load(f)

    if not metrics:
        print("错误: metrics_diffusion.json 为空")
        return

    # 提取数据
    epochs = [m['epoch'] for m in metrics]
    rmsd_mean = [m['rmsd_mean'] for m in metrics]
    rmsd_std = [m['rmsd_std'] for m in metrics]
    bottom_recall = [m['bottom_recall_mean'] for m in metrics]
    bottom_rmsd = [m['bottom_rmsd_mean'] for m in metrics]
    bond_validity = [m['bond_validity_mean'] for m in metrics]
    composite_score = [m['composite_score'] for m in metrics]

    # 创建图表 (2行3列)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. RMSD (主要指标)
    ax = axes[0, 0]
    ax.plot(epochs, rmsd_mean, 'b-', linewidth=2, label='RMSD Mean')
    ax.fill_between(epochs,
                     [m - s for m, s in zip(rmsd_mean, rmsd_std)],
                     [m + s for m, s in zip(rmsd_mean, rmsd_std)],
                     alpha=0.2, color='b', label='±1 Std')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('RMSD (Å)', fontsize=10)
    ax.set_title('Root Mean Square Deviation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Bottom Atom Recall
    ax = axes[0, 1]
    ax.plot(epochs, bottom_recall, 'g-', linewidth=2, label='Bottom Recall')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Recall', fontsize=10)
    ax.set_title('Bottom Atom Recall (遮挡区域)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 3. Bottom RMSD
    ax = axes[0, 2]
    ax.plot(epochs, bottom_rmsd, 'r-', linewidth=2, label='Bottom RMSD')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('RMSD (Å)', fontsize=10)
    ax.set_title('Bottom Atom RMSD', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Bond Validity
    ax = axes[1, 0]
    ax.plot(epochs, bond_validity, 'm-', linewidth=2, label='Bond Validity')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Validity', fontsize=10)
    ax.set_title('Bond Validity (键有效率)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 5. Composite Score
    ax = axes[1, 1]
    ax.plot(epochs, composite_score, 'c-', linewidth=2, label='Composite Score')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Composite Score (综合评分)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 6. 统计表格
    ax = axes[1, 2]
    ax.axis('off')

    # 最终epoch的统计
    final = metrics[-1]
    stats_text = f"""
    最终评估结果 (Epoch {final['epoch']}):

    • RMSD: {final['rmsd_mean']:.2f} ± {final['rmsd_std']:.2f} Å
    • Bottom Recall: {final['bottom_recall_mean']:.4f}
    • Bottom RMSD: {final['bottom_rmsd_mean']:.2f} Å
    • Bond Validity: {final['bond_validity_mean']:.4f}
    • Count Accuracy: {final['count_exact_match']:.4f}
    • Count MAE: {final['count_mae']:.2f}
    • Composite Score: {final['composite_score']:.4f}

    训练轮次: {len(metrics)} epochs
    """

    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')

    plt.suptitle('RMSD 和评估指标变化曲线', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 保存图片
    output_path = 'visualizations/rmsd_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✓ RMSD 曲线已保存到: {output_path}')

    # 打印摘要
    print()
    print('=' * 70)
    print('RMSD 变化趋势:')
    print('=' * 70)
    for i in [0, len(metrics)//4, len(metrics)//2, 3*len(metrics)//4, -1]:
        if i >= len(metrics):
            i = -1
        m = metrics[i]
        print(f"Epoch {m['epoch']:2d}: RMSD = {m['rmsd_mean']:7.2f} ± {m['rmsd_std']:7.2f} Å")
    print('=' * 70)

    # 对比 Loss vs RMSD
    print()
    print('Loss vs RMSD 对比:')
    print('-' * 70)
    print('Loss (curves_diffusion.png):')
    print('  • 显示训练损失（coord_loss, type_loss等）')
    print('  • 每个batch计算，用于梯度优化')
    print('  • 越小越好，但不直接代表性能')
    print()
    print('RMSD (rmsd_curves.png):')
    print('  • 显示评估指标（几何精度）')
    print('  • 每个epoch完整生成后计算')
    print('  • 直接反映重建质量（单位: Å）')
    print('-' * 70)


if __name__ == '__main__':
    print('绘制 RMSD 和评估指标曲线...')
    print()
    plot_rmsd_curve()
    print()
    print('提示: 对比以下两个图:')
    print('  1. visualizations/curves_diffusion.png  - 训练损失（Loss）')
    print('  2. visualizations/rmsd_curves.png       - 评估指标（RMSD等）')
