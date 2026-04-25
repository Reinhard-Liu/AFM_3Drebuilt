# V13 Phase 2 GNN TypeClassifier - Training Metrics

> 30 epochs, 5000 generated samples, noise σ=0.01

| Epoch | Val Type Acc | 备注 |
|-------|-------------|------|
| 10 | 0.6799 | |
| 20 | 0.6839 | |
| 27 | **0.6817** | Best checkpoint |
| 30 | 0.6817 | |

### 分析
- GNN Val Type Acc 在 Ep10 即达 0.680，后续提升有限（0.680-0.684）
- Best epoch=27，val_acc=0.6817（与 V12 的 0.6777 持平）
- 训练数据：5000 个 DDIM-50 生成样本 + σ=0.01 噪声注入
