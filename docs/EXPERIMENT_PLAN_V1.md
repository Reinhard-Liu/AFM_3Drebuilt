# AFM 三维分子结构重建 - 实验方案

## 1. 研究目标

从原子力显微镜（AFM）图像堆栈中重建三维分子结构，包括原子坐标、原子类型和原子数量，并从参考数据库中检索最相似的分子。

## 2. 技术路线

### 2.1 整体框架

```
                    AFM 三维分子重建系统
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   输入: AFM 图像堆栈 (10层深度切片, 128x128)                     │
│         K=40 pN/nm, Amplitude=40 pm                             │
│                                                                 │
│   ┌──────────────────────────────┐                              │
│   │    Video ViT 编码器           │                              │
│   │  - PatchEmbedding3D          │                              │
│   │    卷积核=(2,16,16)          │                              │
│   │  - 8层 Transformer blocks    │                              │
│   │    dim=512, heads=8          │                              │
│   │  - CLS token 全局池化        │                              │
│   └──────────┬───────────────────┘                              │
│              │                                                  │
│              ▼                                                  │
│     条件向量 c (512维)                                           │
│              │                                                  │
│   ┌──────────┼──────────┬────────────────┐                      │
│   ▼          ▼          ▼                ▼                      │
│ 原子数      条件DDPM    分子检索         物理约束                 │
│ 预测头     (1000步)     预测头          模块                     │
│ 分类+回归  SE(3)-等变   InfoNCE         键长/键角/环             │
│           去噪网络     对比学习                                  │
│   │          │          │                │                      │
│   ▼          ▼          ▼                │                      │
│ 原子数N    坐标       Top-5 CID      在扩散采样                  │
│ (1-85)    + 类型     检索结果        过程中施加                   │
│   │          │          │                                       │
│   └──────────┴──────────┘                                       │
│              │                                                  │
│              ▼                                                  │
│   [可选] RDKit MMFF94 力场弛豫（位移上限 0.3 A）                  │
│              │                                                  │
│              ▼                                                  │
│   输出: 三维结构 + 原子类型 + Top-5 候选分子                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键模块说明

| 模块 | 实现文件 | 功能 |
|------|---------|------|
| Video ViT 编码器 | `src/models/video_vit.py` | 将10层AFM深度切片视为时间维度，通过3D卷积生成 tubelet patches，经8层Transformer编码为512维全局条件向量 |
| 条件DDPM | `src/models/diffusion.py` | SE(3)-等变去噪网络，1000步 cosine schedule 扩散过程，同时生成原子坐标和原子类型 |
| 原子数预测头 | `src/models/prediction_heads.py` | 分类（85类）+ 回归双分支，预测分子原子数（1-85） |
| 分子检索头 | `src/models/prediction_heads.py` | 将条件向量投影到128维归一化嵌入空间，通过InfoNCE对比学习与训练集分子嵌入匹配 |
| 物理约束 | `src/models/constraints.py` | 键长惩罚（C-C 1.54 A, C-H 1.09 A 等）、键角惩罚（sp3: 109.5, sp2: 120）、平面性约束 |
| 环结构约束 | `src/models/ring_detection.py` | BFS/DFS 搜索5/6元环，Procrustes 对齐到标准刚体模板 |
| RDKit 后处理 | `src/models/postprocess.py` | MMFF94/UFF 分子力场能量最小化，仅推理时使用 |

## 3. 数据集

| 属性 | 规格 |
|------|------|
| 数据来源 | QUAM-AFM 数据集（K-1 参数集） |
| 分子总数 | 68,555 个 |
| 元素种类 | H, C, N, O, F, S, P, Cl, Br, I（共10种） |
| 最大原子数 | 85 个/分子 |
| AFM 参数 | 弹性常数 K=40 pN/nm, 振幅 Amplitude=40 pm |
| 图像分辨率 | 128x128 像素，10层深度切片 |
| Corrugation 过滤 | >= 1.25 A（Z轴起伏阈值） |
| 数据划分 | 训练集 90% / 验证集 5% / 测试集 5%（val_size=1000） |
| 数据存储路径 | `dataverse_files/SUBMIT_QUAM-AFM/QUAM/` |

### 数据流水线

```
原始数据（K-1 目录，68,555个分子）
    ↓
QUAMAFMDataset（src/data/dataset.py）
  - 解析 XYZ 坐标文件 + AFM 图像堆栈
  - corrugation 过滤（Z轴起伏 >= 1.25 A）
  - 按 CID 排序，保证可复现性
  - 首次加载生成 pkl 缓存加速后续读取
  - 3D 旋转增强（仅训练集，tilt > 30 时触发）
  - 80/10/10 train/val/test 划分
    ↓
批次输出：(afm_stack, coords, atom_types, atom_mask, n_atoms, cid_idx, ring_info)
```

## 4. 评估指标

| 指标 | 定义 | 目标值 |
|------|------|--------|
| **RMSD** | 匈牙利匹配后的均方根偏差（A） | < 2.0 |
| **底部原子召回率** | Z坐标最低30%原子的重建精度（%） | > 30% |
| **原子数精确匹配率** | 预测原子数与真实原子数完全一致的比例（%） | > 60% |
| **原子数MAE** | 原子数预测的平均绝对误差 | < 2.0 |
| **键有效性** | 预测键长在合理范围内的比例（%） | > 85% |
| **环保持分数** | 环结构（5/6元环）的保持程度 | > 0.5 |
| **CID Top-1/Top-5** | 检索命中正确分子CID的比率 | > 10% / > 30% |
| **综合评分** | 以上6个指标的加权综合得分 | > 0.3 |

### 综合评分公式

```
Composite = 0.3  * (1 - RMSD/10)        # RMSD 贡献最大
          + 0.2  * BottomRecall          # 底部原子召回
          + 0.15 * BondValidity          # 键有效性
          + 0.15 * AtomCountAccuracy     # 原子数精确匹配
          + 0.1  * RingPreservation      # 环保持
          + 0.1  * CIDAccuracy           # 检索准确率
```

## 5. 训练策略（三阶段渐进式训练）

### 阶段一：基础训练（Epoch 1-30）

| 参数 | 设置 |
|------|------|
| 学习率 | 1e-4 → 1e-6（余弦退火） |
| 损失函数 | coord_loss + 0.1*type_loss + 0.5*count_loss + 0.05*retrieval_loss |
| 物理约束 | 关闭 |
| Z轴加权 | 均匀权重 |
| 目标 | 学习基本的三维重建能力 |

### 阶段二：约束训练（Epoch 31-45）

| 参数 | 设置 |
|------|------|
| 学习率 | 5e-5 → 1e-6 |
| 损失函数 | 阶段一损失 + 0.1*constraint_loss |
| 物理约束 | 键长约束 + 键角约束 + 平面性约束 |
| Z轴加权 | 均匀权重 |
| 目标 | 提升化学有效性 |

### 阶段三：底部聚焦训练（Epoch 46-60）

| 参数 | 设置 |
|------|------|
| 学习率 | 2e-5 → 1e-6 |
| 损失函数 | 阶段二损失（加入Z深度加权） |
| 物理约束 | 开启 |
| Z轴加权 | 底部原子 3x 权重，顶部原子 1x 权重 |
| 目标 | 提升遮挡区域（底部）原子的重建精度 |

## 6. 实验流程

### 第一步：数据准备与验证

```bash
# 1. 运行模块健全性检查（测试所有组件）
cd /root/autodl-tmp/micro
python3 -m src.quick_test

# 2. corrugation >= 1.25 A 的分子将被纳入训练
#    过滤后约 213,505 个有效分子
```

### 第二步：模型训练

```bash
# 完整训练（三阶段，60个epoch）
cd /root/autodl-tmp/micro
bash scripts/shell/run.sh

# 或使用自定义配置
python3 -m src.train --config config.json
```

**核心配置参数**（config.json）：
```json
{
    "model_type": "diffusion",
    "data_root": "auto",
    "param_key": "K-1",
    "img_size": 128,
    "num_frames": 10,
    "min_corrugation": 1.25,
    "max_samples": 100000,
    "val_size": 1000,
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 60,
    "embed_dim": 512,
    "encoder_depth": 8,
    "diffusion_steps": 1000
}
```

### 第三步：评估与分析

```bash
# 1. 训练过程中自动评估（每个epoch）
#    -> checkpoints/metrics_diffusion.json   （各指标数值）
#    -> checkpoints/history_diffusion.json   （损失历史）

# 2. 验证集可视化（GT vs 预测 对比图）
python3 -m src.visualize_val \
    --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 100 \
    --output_dir outputs/val_diffusion

# 3. 训练曲线自动生成
#    -> outputs/curves/curves_diffusion.png
```

### 第四步：预测输出与结果保存

```bash
# 训练结束后自动保存预测结果：
#    -> checkpoints/predictions_diffusion.json
#
# 每个样本包含以下字段：
#   coords:               三维原子坐标
#   atom_types:            预测的原子类型
#   n_atoms_pred:          预测的原子数量
#   retrieval_cid_indices: Top-5 候选分子CID索引
#   retrieval_scores:      Top-5 相似度分数
```

## 7. 基线对比

| 模型 | 架构 | 用途 |
|------|------|------|
| **Diffusion（主模型）** | Video ViT + 条件DDPM | 完整三维重建 |
| **ResNet3D（基线模型）** | 3D ResNet 直接回归 | 坐标直接回归对比 |

两个模型在相同的测试集上使用相同的评估指标进行评估。

## 8. 坐标空间归一化

所有模型操作在归一化坐标空间中进行：

| 物理量 | 真实值（A） | 归一化值（/12.0） |
|--------|-----------|------------------|
| 归一化系数 | 12.0 A | 1.0 |
| C-C 单键 | 1.54 A | 0.128 |
| C-H 键 | 1.09 A | 0.091 |
| 键长容差 | 0.15 A | 0.0125 |
| 苯环半径 | 1.40 A | 0.117 |

## 9. 输出文件说明

```
checkpoints/                          # 模型权重与训练记录
├── best_diffusion.pt                 #   最佳模型权重
├── epoch_{10,20,...,60}_diffusion.pt  #   各阶段检查点
├── history_diffusion.json            #   训练损失历史
├── metrics_diffusion.json            #   逐epoch评估指标
├── predictions_diffusion.json        #   100个样本的预测结果
├── training.log                      #   完整训练日志
└── training_summary.txt              #   训练总结

outputs/                              # 可视化输出
├── curves/                           #   训练曲线图
├── val_diffusion/                    #   验证集可视化（GT vs 预测）
├── molecules_diffusion/              #   三维分子结构对比
├── test_predictions/                 #   测试集预测结果
└── sample_analysis/                  #   样本级分析（Top-5 AFM对比等）
```

## 10. 依赖环境

| 库 | 用途 |
|----|------|
| PyTorch（CUDA） | 深度学习框架 |
| NumPy, SciPy | 数值计算 |
| PIL (Pillow) | 图像读取 |
| einops | 张量变换 |
| tqdm | 进度条 |
| Matplotlib | 可视化绘图 |
| RDKit（可选） | 分子力场后处理 |
