# V6 Experiment

## Changes from V5b
1. **TypeNet** — 独立的原子类型预测器，使用坐标+环境特征+AFM patch cross-attention
2. **Multi-scale condition** — ViT 返回 (c_global, c_patches)，denoiser 每层 cross-attend patches
3. **Distance matrix loss** — 低噪声时步 (t<300) 约束键距和全局距离
4. **Valence consistency loss** — 预测类型与邻居数一致性约束
5. **Connectivity projection** — 采样最后 10% 步拉回孤立原子
6. **Bottom atom focus** — 底部权重从 3x 提升到 5x
7. **Depth weighting** — ViT 学习 AFM 各层切片的权重

## V5b Baseline
- RMSD: 0.269
- Type Match: 48.5%
- Coulomb: 0.009
- Bottom Recall: 3.9%

## Training Config
- Epochs: 70
- Batch size: 128
- Max samples: 100,000
- 3-stage training: base(1-30) → constraints(31-45) → bottom focus(46-70)

## Loss Weights
coord + 0.5*dist + 1.0*type + 0.2*valence + 1.0*count + 0.5*shape + 0.01*retrieval + 0.1*constraint
