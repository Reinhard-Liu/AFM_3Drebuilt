# 项目原理(Principles)

> 本文件介绍 AFM 3D 重建项目的**核心问题、关键洞察、设计哲学**。代码细节请见 [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md);指标定义请见 [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)。

---

## 一、核心问题与挑战

### 1.1 任务定义

输入:**10 张** AFM(Atomic Force Microscopy)切片图像,每张 `128×128`,对应不同针尖-样品距离(z 方向 Δ=0.05 Å,共跨越 ~0.5 Å)。

输出:分子的 3D 结构 — 每个原子的:
1. **3D 坐标** `(x, y, z)`(归一化到 `[-1, 1]`,真实尺度 = ×12 Å)
2. **元素类型** ∈ {H, C, N, O, F, S, P, Cl, Br, I}(10 类)
3. **键(邻接关系)**:同分子原子两两间是否成键(无键序细分,只判 0/1)

数据集:**QUAM-AFM Lite**(Hapala 团队 2022),共 1,755 个分子,严格按 CID 排序后线性切分为 train / val / test。

### 1.2 与传统 AFM 重建的差异

经典 AFM 反演通常用 **probe particle simulation**(模拟 → 优化匹配)或 **CNN 分类**(像素级元素图)。本项目采用端到端学习 + 对象级显式建模:

| 经典做法 | 本项目 V19/V20 |
|---|---|
| 像素级类型分类 → 后处理聚类 | dense map + **对象级条件头**(在 peak/pred 中心采样) |
| z 方向手工力场反演 | dense z 回归 + 全 3D RMSD 监督 |
| 键由距离阈值决定 | 独立 `CenterConditionedEdgeHead` 预测,可学非键长依赖 |
| 单阶段训练 | curriculum + KD + warm start |

### 1.3 两个核心难点

1. **AFM 信号弱 → 元素稀疏类难分**(P / Br / I 在 QUAM-AFM Lite 中只占百分之零点几)
2. **z 方向信号系统性弱**:AFM 高度差小(~0.5 Å),且像差与振幅耦合强 → 直接回归 z 误差大

V19 / V20 的设计针对性解决这两点。

---

## 二、关键洞察

### 2.1 洞察 1:AFM 切片 = "伪视频"

**事实**:同一物质的 10 张 AFM 是相同分子在不同 z 高度的"扫描帧",像帧序列一样具备:
- 强空间相关(同一原子在多帧投影位置接近)
- 强时间相关(随 z 变化连续)
- 平稳背景(衬底纹理几乎不变)

**对应做法**:用 **Video ViT**(`PatchEmbedding3D` 用 `(2,16,16)` tubelet),不是普通 2D ViT 累通道,也不是 3D CNN 全连接。

**为什么 tubelet 而非 ConvLSTM**:
- 计算复杂度 O(N²) 但 N=320 = 5×64 已小到可以 attention 全连接
- attention 显式跨帧建模 z 维变化,对 z-shape 重建至关重要
- 训练稳定,无 RNN 梯度问题

### 2.2 洞察 2:AFM 信号天然分解为三个物理通道

任何一张 AFM 像 = "**位置 × 元素特性 × 高度**"的某种乘积:
- **位置**(x, y):像素中心
- **元素特性**(电子云形状、半径):像素强度的形状细节
- **高度**(z):像素强度随帧的变化模式

**对应做法**:UNet 三分支解码器:
- Center 分支(高斯热图 + sigmoid)→ 学位置
- 2D 结构分支(12 通道:10 type + 2 aux + Tanh)→ 学元素特性 + 键
- Z 分支(1 通道 + Tanh)→ 学高度

三分支共享主干 + 独立 skip,**任务相关的特征聚合在最后一层**。

### 2.3 洞察 3:对象级条件头 = train-deploy 一致性

**问题**:dense map 只能"画图",真正的下游任务(原子级 RMSD、键 F1)是**对象级**的。如果训练只优化 dense loss,得到的预测在 peak detection 后:
- peak 数错(MAE 可能 > 5)
- peak 位置略偏(每个偏 1-3 px)
- 解码出的分子形状对,但类型、键大幅错

**洞察**:把"在中心位置采样并分类"作为**显式的 head 与 loss**,而不是后处理。

`CenterConditionedTypeHead` / `CenterConditionedEdgeHead` 接受 `(coords, shared_feat, afm)` 三元组,**完全 differentiable**:
- 训练时:gt 中心 → 已知的真实位置喂入,作为 upper bound
- 训练时:peak 中心 → 模型自己检测的位置,作为 deploy 路径
- 评估时:pred 中心 → 完全闭环

Train-deploy gap 由 **center curriculum** 显式调度从 0(全 GT)到 1(全 pred),平滑过渡。

### 2.4 洞察 4:全局计数是闭环的"锚"

仅靠 peak detection 不知道"应该有几个原子",阈值难调。**全局计数头**(86 类 0~85)在 bottleneck 处端到端预测分子规模,作用:

1. 推理时按计数 top-K peak,避免 false-peak 污染
2. 训练时与对象头形成闭环:计数错 → 取错 K → 对象头分类损失大 → 反传到计数头
3. V20 显式加 `lambda_object_count_mae=0.15`,让回归头辅助分类头

V20 实测 `count_mae = 0.94`(几乎完美),Dense baseline 没这条路径,MAE = 34 → 直接说明这一设计的关键作用。

### 2.5 洞察 5:分类 = 粗 + 细 + 异质 三级

仅做 10 类 fine 分类 → P / Br / I 永远 F1=0(样本极少)。

**做法**(`v19_center_type_head.py:61-74`):
- **粗 3 类**(C/H = 0,N/O/S/P = 1,F/Cl/Br/I = 2):学元素**化学性质族**
- **异质 2 分类**(H/C 与其他):学"是不是杂原子"
- **细 10 类**:focal-CE(γ=1.5) + label_smoothing 0.02

总损失 `0.35 × coarse + 0.25 × hetero + fine`(`v19_center_type_head.py:298`)。

效果:hetero_f1=0.7434(对外宣传数字),通过粗分支兜底罕见类。

### 2.6 洞察 6:Type Upper Teacher 蒸馏

GT-中心是物理上的上界:**如果连中心都告诉你,类型理应能学到 ~95%**。但实际 V20 `gt_object_score=0.8279`,说明类型本身就不容易。

**做法**:先训一个"GT 中心 + GT 类型监督"的 teacher,得到上界 logits;再让 V19/V20 student 在 GT 路径上 KL 蒸馏 teacher。**温度 1.5 + T² 补偿**(KD 标准技巧)使梯度尺度独立于温度。

效果:student 的 GT 路径与 teacher 接近,**peak / pred 路径通过 curriculum 与 consistency 进一步逼近**。

### 2.7 洞察 7:RDKit 弛豫的位移上限

后处理用 MMFF94 / UFF 力场弛豫坐标。**风险**:模型预测可能某些键长略偏,力场会以"远离最近 minimum"的代价拉拢,反而拉坏 RMSD。

**做法**(`postprocess.py:225-235`):

```python
if max_disp > max_displacement (0.3 Å):
    scale = 0.3 / max_disp
    disp_clipped = disp * scale
```

整体按比例缩回,确保力场只**微调**模型预测,不能完全主宰几何。论文同类工作通常无此 cap,造成"弛豫后反而更差"的常见问题。

---

## 三、设计哲学

### 3.1 显式胜于隐式

每一个**关键的中间状态**都要有显式监督:
- 中心位置 → `lambda_center=20.0`(最重)
- 原子计数 → `lambda_object_count=1.0` + MAE
- 类型(三级)→ 三个分支独立 loss
- 边 → 独立 head + BCE
- 一致性(GT vs pred)→ KL 蒸馏

**为什么不交给端到端学**:dense loss 太多自由度,梯度难以聚焦关键路径。每加一个显式 loss,模型就少一个失败模式。

### 3.2 Curriculum 平滑路径切换

GT → peak → pred 三条路径在物理上等价,但**特征分布**不同(GT 来自精确标注,peak 有 1-3 px 抖动)。突然从 GT 切到 peak,对类型头是"分布外"。

**Curriculum 解法**:
1. 类型 / 边 head 的 peak / pred 路径权重从 0.25 ramp 到 2.5(主)/ 2.0(V20 pred)
2. center_curriculum_alpha 从 0(纯 GT)线性到 1(纯 peak/pred)
3. Aux dense loss 同步衰减(start=5.0 → final=1.0),把容量"挪给"对象头

12 epoch 跨度(V19),V20 缩到 5 epoch(因为 warm start 起点高)。

### 3.3 Warm start + 增量改造

V19 → V20 不重训,而是:
1. `warm_start_checkpoint=V19_best.pt` 非严格加载
2. 边头新增的 `msg_mlp / refine_gate / refine_edge_mlp` **零初始化**(`v19_center_edge_head.py:42-76`)
3. 新增 lambdas 从小值 ramp 起来

零初始化保证暖启动时新分支输出 = 0,完全等价于 V19,然后逐步学。

### 3.4 后处理与训练解耦

RDKit 弛豫**只在评估时用**,不进入训练 loss。原因:
- RDKit 不可微,要 STE 或 RL,引入大量噪声
- 训练阶段已经有 `bond_validity` (键长偏差容差 25%),已隐式约束化学合理性
- 弛豫只做 ≤ 0.3 Å 的微调,不改变任务定义

### 3.5 评估指标的双层结构

每个核心能力**同时**用 strict + robust 两个版本(EXP-03):
- strict edge F1:必须键的两端原子位置都对(严格)
- robust edge F1:允许中心偏 ≤ 3 px(放宽)
- 二者差 = `edge_gap_robust`(0.2780)= "拓扑能力 vs 定位能力" 的差异

这种解耦让我们看清"V20 已经懂图结构,瓶颈是亚像素中心定位",直接指导下一步优化方向(soft-argmax / heatmap regression)。

---

## 四、与同类工作的差异

| 维度 | 本项目 | 经典 AFM 反演 | 通用图重建(DETR 式) |
|---|---|---|---|
| 主干 | Video ViT(tubelet) | CNN 或 ConvLSTM | Transformer set predictor |
| 监督 | dense + 对象级条件头(混合) | 像素级分类 | 端到端 set loss(Hungarian) |
| 计数 | 显式 86 类头 + 闭环 | 后处理阈值 | DETR 的 no-object class |
| 类型 | 三级(粗/细/异质)+ KD | 单一 fine | 单一 fine |
| 中心 | curriculum(GT → pred) | 后处理 peak | 直接坐标回归 |
| 后处理 | RDKit + 位移 cap | 力场 / 退火 | 无 |
| z 监督 | dense + 全 3D RMSD | 手工反演 | 直接 (x,y,z) |
| 等变性 | 数据增广 + 评估对齐 | SE(3) 网络 | data augment 或 equivariant |

本项目走的是**"重设计 + 轻数学"**:用工程级技巧(curriculum、KD、零初始化、位移 cap)换取**简单稳定**的结构,而不引入等变层 / 神经 ODE / diffusion 等重数学组件。

---

## 五、为什么这套设计有效:消融证据

EXP-06 / EXP-07 ablation:

| 移除项 | pred_object_score 下降 | 说明 |
|---|---|---|
| Curriculum | −0.15 | 直接从 epoch 0 用 peak/pred 中心,collapse |
| Edge head | −0.14 | 完全失去键预测,scaffold 重建崩 |
| Object count loss | −0.06 | 计数 MAE 翻倍,top-K 取偏导致下游错 |
| Z head | ≈ 0 | z 只贡献 5% 综合分,但影响 3D 综合 |
| Teacher consistency | 不可比(用了不同 split) | — |

每项都说明对应设计是**必要**的,而非装饰。

---

## 六、从 V18 → V19 → V20 的演化

| 版本 | 核心改动 | 解决的问题 |
|---|---|---|
| V17 Bridge | dense + 检索 + ring | 引入站点、site graph |
| V18 | 加入 z 头 | 让 dense 学 3D |
| **V19** | **对象级条件头**(type + edge)+ KD + curriculum | **train-deploy gap** |
| **V20** | dual-input type head + 一致性 KL + 显式计数 loss + edge refinement | **闭环一致性** + 全局/对象联动 |

每次迭代都在补一个**显式信号通路**,而非堆模型容量。

---

## 七、关键不变量(请不要轻易动)

| 不变量 | 来源 | 改动后果 |
|---|---|---|
| `img_size=128` | dataset + UNet 所有 stride | 形状错配,下游全崩 |
| `num_frames=10` | tubelet 切块假定整除 | num_temporal_patches 不整 |
| `temporal_patch_size=2` + `patch_size=16` | num_patches=320 硬编码 | pos_embed 大小不匹配 |
| `MAX_ATOMS=85` | 计数头 86 类、对象头 padding 长度 | 截断分子 / 形状错 |
| `COORD_SCALE=12.0` | dataset / postprocess / metrics 共用 | RDKit 键判定全错 |
| `_BOND_TOLERANCE=1.3`(postprocess)/ `1.2`(ring) | 物理化学常识 | 键判定灵敏度变化 |
| 10 元素列表顺序 | 类别 index 直接对应字符串列表 | 类别错位 |

---

## 八、相关文档

- 实现细节 — [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md)
- 配置参数 — [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)
- 流程框架图 — [`PIPELINE_AND_FRAMEWORK.md`](PIPELINE_AND_FRAMEWORK.md)
- 指标定义 — [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)
- FAQ 含设计动机 — [`FAQ_EXTENDED.md`](FAQ_EXTENDED.md)
