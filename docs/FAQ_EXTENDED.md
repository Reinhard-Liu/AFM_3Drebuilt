# 扩展 FAQ

> 涵盖**专业技术问题、运行问题、结果解读问题**。简版 FAQ 见 [README § 十二](../README.md#十二常见问题答疑faq)。

---

## A. 专业技术问题

### A1. 为什么用 Video ViT 而非 2D ViT 或 3D CNN?

10 层 AFM 切片在 z 方向有强相关性 — 同一原子在不同高度产生**响应曲线**而非孤立信号。

- **2D ViT 单帧** — 丢失层间关系,无法估 z
- **3D CNN(C3D / I3D)** — 局部感受野不够,长程依赖建模弱
- **Video ViT** — 时空联合自注意力,既看 (x, y) 平面又看 z 轴上下文

实验中 Video ViT 比 ResNet3D baseline 在 `pred_object_score` 上 +0.42(详见 [`V1-V6_RETROSPECTIVE.md`](V1-V6_RETROSPECTIVE.md))。

### A2. 为什么不直接回归 3D 坐标?

V1–V6 试过,详见复盘:

- 直接回归 + Hungarian matching loss 收敛慢
- 缺乏 SE(3) 等变性,旋转后预测不一致
- 类型与坐标耦合,小类别完全压不出来

V15 起改为 **fixed-coordinate 监督** — 在 128×128 网格上做 dense 预测,(x, y) 由位置确定,z 单独回归。SE(3) 等变性用旋转增广替代。

### A3. peak-center curriculum 真的有用吗?有消融吗?

EXP-06 消融对比(详见 `experiments/v20_ablate_curriculum_debug/reports/`):

| 配置 | pred_object_score |
|---|---|
| baseline(curriculum on) | 0.7141 |
| `alpha_start = alpha_final = 0` (no curriculum, only GT) | 0.5612(↓ 21%) |
| `alpha_start = alpha_final = 1` (always peak) | 0.6803(↓ 5%) |

curriculum 必须有 — **慢启动**让模型先学到 GT-center 下的"理想路径",再切换到 peak-center 才能收敛得稳。

### A4. Type Upper Teacher 蒸馏的副作用?

- **正面** — 杂原子 F1 +52pp
- **负面** — 主类(C / H)F1 微降 ~1pp(模型被"教得"软化)
- **配置成本** — 需先训完一个 teacher(~6h on V19)再训 student

V19/V20 都默认开启,需要纯净 baseline 时关掉 `lambda_teacher_type_distill`。

### A5. V19 vs V20 谁更好,为什么 V19 是主线?

| 维度 | V19 Full15 | V20 Medium10 |
|---|---|---|
| 数据规模 | 全样本 68,555 | 缩减集 65k(实际更小) |
| 训练时长 | ~36h | ~10h |
| `peak_object_score` | 0.8016 | 0.7141(`pred_*` 名字) |
| 架构创新 | object-joint 头 | + 计数头 + 双输入头 + 闭环 |
| 状态 | 投稿就绪 | 架构原型,验证可行性 |

V19 = 数据规模 + 稳定性的胜利;V20 = 架构方向的胜利(在缩减集上完成闭环验证)。论文主图用 V19,创新点用 V20 演示。

### A6. 为什么 Strict 与 Robust 边 F1 差这么多(0.27)?

`strict` 要求中心在 ≤3 px 内对齐才计入正例;`robust` 用 Hungarian 把所有中心匹完再算邻接矩阵 F1。

> Gap 大 = 模型"拓扑拓出来了,中心稍微偏" — **瓶颈是亚像素定位,而非图结构能力**。

EXP-03 的 `gap_vs_*` 图证实:gap 与 z_mae 强相关,与 node_coverage 弱相关。

### A7. 数据增广包括什么?会不会破坏物理意义?

- **旋转** — 90° / 180° / 270° 安全;任意角度有 `tilt < 30°` 限制(避免分子被"放倒")
- **翻转** — 水平 / 垂直
- **AFM 切片高斯噪声** — σ=0.01(归一化后)
- **不会破坏** — 因为 AFM 物理本身在 (x, y) 平面下旋转对称(扫描方向无关)

### A8. 如何处理"分子边缘超出 128×128 视野"的情况?

dataset.py 加载时用 `min_corrugation` 滤掉"完全平躺、与背景无差异"的样本。对于"局部超出视野"的分子,目前不做特殊处理,模型靠注意力关注主区。

### A9. 模型对哪些分子最难?

EXP-04 几何诊断的 `height_span_vs_z_mae.png` 表明:

- z 跨度 > 3 Å 的非平面分子 z_mae 显著高
- 含三键、芳环融合的复杂分子 type_acc 低
- ≤5 原子的小分子 hetero_f1 反而低(样本数少,噪声主导)

### A10. 训练数据为什么用 K-1 而不是更大的 QUAM-AFM 子集?

K-1 是 Pérez group 公开的"质量统一"子集。其他子集(如 K-2)在 AFM 仿真参数(K, Amplitude)上有差异,直接混合会引入新的分布偏移。

---

## B. 运行 / 工程问题

### B1. 单 GPU 显存最低多少?

A100 40GB 当前 batch_size=8 占 ~28GB。降到 batch_size=4 可在 V100 24GB 跑。

### B2. CPU-only 能训吗?

理论可以,但 ViT + 10 层 + 128×128 在 CPU 上一个 epoch 估算 > 100h。**不实用**。

### B3. RDKit 安装失败?

详见 [`guides/RDKIT_INSTALLATION.md`](guides/RDKIT_INSTALLATION.md)。

最稳路径:`conda install -c conda-forge rdkit`,Python 必须 ≥ 3.10。

### B4. PyTorch 版本兼容性?

| 组件 | 要求 |
|---|---|
| Python | 3.10–3.12 |
| PyTorch | ≥ 1.10(推荐 ≥ 2.0 用 `torch.compile`) |
| CUDA | 11.8 / 12.x |

### B5. 怎么从 checkpoint 直接做推理?

```bash
python3 -m src.v20_eval_fulltest_object \
    --checkpoint <best.pt> \
    --output_dir my_output \
    --split test --batch_size 8
```

---

## C. 结果解读问题

### C1. 训练日志显示 `val_loss` 还在降但 `val_score` 已经停了?

这是常态。`val_loss` 是损失函数加权和,`val_score` 是离散指标(F1 / 准确率)。**优先看 score**。

### C2. EXP-01 报告里 `pred_object_score = 0.7141`,但训练日志最高是 `0.74`?

训练日志 `val` 在 **val split**(随机 ~512 样本),EXP-01 在 **test split**(独立)。差 0.02–0.03 正常。

### C3. `pred_object_3d_score` 比 `pred_object_score` 高,什么意思?

3D 综合分加大了 coord/z 分量(35% + 20% = 55%),而 V19 起 atom_xy_mae 和 z_mae 都接近上限,所以 3D 分系统性偏高。这是**评估口径**的差异,不是质量真的更好。

### C4. EXP-04 的 `coverage_vs_pair_dist.png` 怎么读?

X 轴 = 节点覆盖率(预测 N / GT N);Y 轴 = 两两距离误差。

- 覆盖率 = 1.0 时点云密集,误差小 — 模型识对了 N
- 覆盖率 < 1 时误差散开 — 漏检原子拉高 RMSD

V20 的 93.16% 通过率(误差 ≤ 0.25 Å)是论文里给"几何精度通过率"的关键数字。

### C5. SUP-01 (Dense baseline) 为什么差这么多?

Dense baseline 没有对象级头,只能 dense 预测后 argmax → peak。**没法学"图里有几个原子"**,所以计数 MAE 19.86(V20: 0.94)。

### C6. 检索 Top-1 = 74%,大分子比小分子高,反直觉?

不反直觉:大分子(原子越多)embedding 越独特,容易精确匹配;小分子(< 22 原子)在 K-1 闭集里有大量"同构异构体",Top-1 高度耦合。

### C7. EXP-03 缝隙 0.28 算大还是小?

中等。同类工作 baseline 通常 0.4+,V19 缩到 0.3,V20 0.28。再降需要亚像素 head,V21 探索方向。

### C8. 蒸馏 teacher 与 student 的差距能量化吗?

teacher 在杂原子 F1 上达 0.91(GT-center 输入);student 用 peak-center 拿到 0.74。差 0.17 是 "**center 检测误差** + **采样噪声**" 的组合,EXP-03 试图分解。

---

## D. 实验复现问题

### D1. 我改了 `lambda_*` 重训,值对不上?

V19/V20 训练的 stochastic 性来源:

- DataLoader shuffle(`seed=42` 控制,但 worker 内还有抖动)
- AMP 累加顺序
- CUDA non-deterministic ops

**建议** — 同 config 跑 3 次取平均。论文数字是单次最优,差 ±0.005 属正常。

### D2. 用更大数据集会更好吗?

V19 已用全 K-1。下一步要扩,需要新数据集(QUAM-AFM 其他子集 / 自合成)。期待社区贡献。

### D3. 为什么 baseline 与论文上某些参数不一致?

`configs/config_v20_object_joint_medium10.json` 是**最终发布版**;实验过程中的 V20a / V20b / 早期 medium10 略有差异(详见 `docs/V19_3_overnight_full_joint_plan.md`)。

---

## E. 拓展方向

### E1. 想加新元素(如 Si)?

1. 改 `src/data/dataset.py` 中的 `ELEMENT_TO_IDX`
2. 改 `n_classes` 从 11 → 12(在 `v19_joint_model.py`)
3. 重新训练(无法 warm start,新增的 type 通道权重未学过)

### E2. 想换 backbone?

主干在 `src/models/video_vit.py`。理论可换 SwinTransformer / MaxViT 等,但需保持输出 `(B, embed_dim, T', H', W')` 的形状供 upsample bridge。

### E3. 想做 conditional generation(给 SMILES 生成 AFM)?

老版扩散主线 `src/train.py` 是反向任务(AFM → 3D);conditional generation 需要新的 decoder + DDPM,不在本项目范围。

---

## F. 投稿与引用

### F1. 应该引用哪些文献?

- 数据集 — `dataverse_files/readme.txt` 给出 QUAM-AFM 完整 cite
- 主架构 — Video ViT 原论文(Arnab et al., ICCV 2021)
- 蒸馏 — Hinton 2015 KD;FocalLoss(Lin 2017)

### F2. 投稿能用什么数字?

V19 Full15:`peak_object_score 0.8016`,Top-3 检索 86.33%。
V20 Medium10 + EXP-01~04:`pred_object_score 0.7141`,杂原子 F1 0.7434,缝隙 0.2780。

---

## G. 没在这里答到的问题?

提 [GitHub Issue](https://github.com/Reinhard-Liu/AFM_3Drebuilt/issues)。模板:

- 错误堆栈 / 日志末 50 行
- 复现命令
- 环境(`python -V`、`pip freeze`、CUDA 版本)
- 期望 vs 实际表现
