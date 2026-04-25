# 项目原理(Principles)

> 本文件介绍 AFM 3D 重建项目的**核心问题、关键洞察、设计哲学**。代码细节请见 [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md);指标定义请见 [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)。

---

## 一、问题定义

### 1.1 输入

10 层 AFM(原子力显微镜)图像栈 $X \in \mathbb{R}^{10 \times 128 \times 128}$,对应 10 个不同探针-表面距离的灰度切片。物理上,每一层是同一分子在某一固定 z 高度下的频率移动响应 $\Delta f(x, y, z_k)$。

### 1.2 输出

一个分子的三维原子结构 $M = (\mathbf{R}, \mathbf{Z}, E)$:

- $\mathbf{R} \in \mathbb{R}^{N \times 3}$ — N 个原子的 (x, y, z) 坐标
- $\mathbf{Z} \in \{1, 6, 7, 8, 9, 15, 16, 17, 35, 53\}^N$ — 每个原子的元素类型(H, C, N, O, F, P, S, Cl, Br, I)
- $E \subseteq \{(i, j) \mid 1 \le i < j \le N\}$ — 化学键(只考虑共价键,不区分单/双/三键)

### 1.3 任务挑战

| 挑战 | 物理来源 | 工程含义 |
|---|---|---|
| **观测不完备** | AFM 主要响应价电子云;氢原子信号弱 | 需要从间接证据推理弱信号原子 |
| **z 信息高度模糊** | 同一 (x, y) 不同 z 的原子可能产生相似 contrast | 必须用层间相关性而非单帧 |
| **类型混淆** | 邻近元素(C/N/O)的范德华半径接近,AFM 信号差异小 | 需要图结构上下文协助类型分类 |
| **训练-部署 gap** | 训练用 GT 中心可学到"理想路径";部署用峰检测中心 | 必须在训练阶段就模拟峰检测的不准确 |
| **小样本 / 长尾** | 杂原子(F/P/S/Cl/Br/I)出现频率远低于 C/H | 标准 CE 损失会把小类别压成 0 |

---

## 二、关键洞察(为什么这套设计能工作)

### 2.1 AFM 是"伪视频",层间相关性比单帧更可靠

- 单层 AFM 对环、平面分子做得不错,但对**非平面分子、双层分子**容易混淆。
- 10 层切片在 z 方向有 0.1–1 Å 的步长,**层间响应曲线**是 z-height 的强信号。
- 因此采用 **Video ViT**(时空联合自注意力)而非 2D ViT/CNN — 把每一帧当时间步,让注意力在 spatial 与 temporal 维度都建立长程依赖。

代码:[`src/models/video_vit.py`](../src/models/video_vit.py)。

### 2.2 三个解码头的物理对应

| 解码头 | 输出张量 | 对应物理量 | 监督信号 |
|---|---|---|---|
| Center Head | (B, 1, 128, 128) | 在 (x, y) 平面是否存在原子 | 高斯热图(σ=2 px) |
| Type Head | (B, 11, 128, 128) | 该位置是哪种元素(10 + 背景) | one-hot |
| Z Head | (B, 1, 128, 128) | 该位置原子的归一化 z 高度 | 每个原子位置的 z 真值 |

把 3D 重建解耦为 "**找原子在哪、识别什么、估计高度**" 三个 2D 子问题 — 这是 V15 的关键突破:**避免直接回归 3D 坐标**,因为坐标回归对 SE(3) 不敏感(没有等变结构)且需要昂贵的 Hungarian matching。

### 2.3 对象级头解决训练-部署不一致(V19 主创新)

> 单纯用三解码头出来的 dense 输出,部署时需要"先峰检测再采样" — 这一步训练阶段从未模拟,因此存在 **gt_object_score - peak_object_score = 0.18 ~ 0.34** 的稳定缝隙。

V19 引入 **object-level conditional heads**:

1. **Peak-center 解码** — 训练时同时跑两次前向:`gt_centers` 和 `peak_centers`(从预测中心图阈值化得到),共享主干。
2. **Curriculum** — 起初(α=0)主要监督 GT-center 头;`warmup_epochs` 后逐步把权重转移到 peak-center 头(α=1)。
3. **效果** — `peak_object_score`:V15 (0.48) → V19 full15 (**0.8016**),gap 缩小 67%。

代码:[`src/models/v19_center_type_head.py`](../src/models/v19_center_type_head.py)、[`src/models/v19_center_edge_head.py`](../src/models/v19_center_edge_head.py)、训练循环 [`src/train_v19_object_joint.py`](../src/train_v19_object_joint.py)。

### 2.4 类型上界蒸馏:解决长尾杂原子识别(F/P/S/Cl/Br/I)

> 杂原子出现频率 < 1%,常规 CE 训练完全压不出来:F1 ≈ 0.22。

**Type Upper Teacher** 流程:

1. 单独训练一个 [`train_v19_type_upper.py`](../src/train_v19_type_upper.py) 模型,**只接 GT-center 与局部 crop**(剔除中心定位噪声)— 这是类型分类的"上界"。
2. 主模型蒸馏 teacher 的软标签:`KL(student_logits / T || teacher_logits / T)`,T = 1.5。
3. focal CE,γ=1.5 强化稀有类。

效果:杂原子 F1 → **0.86**(V19 full15) / **0.74**(V20 medium10 缩减集)。

### 2.5 对象计数闭环(V20 新增)

> Dense baseline 不知道"图里到底有几个原子",输出常常少 / 多 20+ 个。计数 MAE = 19.86。

V20 引入 `object_count_head`:

1. **CE 头** — 把 1–80 视作 80 类做分类(`lambda_object_count=1.0`)。
2. **MAE 头** — 同时回归连续值(`lambda_object_count_mae=0.15`)做平滑。
3. **解码** — 推理时优先用预测 N 截断 top-N peak。

效果:计数 MAE Dense 19.86 → V20 **0.94**(↓ 21 倍)。

### 2.6 双输入类型头(V20 收尾)

V20 类型头同时消费 **GT 类型图特征** 和 **预测类型图特征**:

- `lambda_type_obj_pred=2.0`(主)+ `lambda_type_obj=0.25`(辅)
- `lambda_pred_type_consistency=0.50` 强制两路输出一致

意义:让"用预测中心采样"那一路真正参与梯度计算,而不是只在评估时偷偷启用。

---

## 三、设计哲学

### 3.1 把 3D 重建拆成 2D 监督的子任务

直接回归 3D 坐标 = 隐式要求模型同时学 SE(3) 等变性 + Hungarian matching + 物理约束。**很难收敛**(V1–V6 试过多次,详见 [`V1-V6_RETROSPECTIVE.md`](V1-V6_RETROSPECTIVE.md))。

V15 起改为 "**fixed-coordinate 监督**":在 128×128 像素网格上做 dense 预测,(x, y) 由网格位置确定,只回归 z。这样模型只需要学**像素级**的对齐,SE(3) 由数据增广(旋转)注入。

### 3.2 同时优化"中心-类型-边-计数"

不是单独优化 RMSD —— **RMSD 本身对类型错误不敏感**(把所有 C 标成 H 也能拿很好的 RMSD)。

我们用 **6 维评估**(详见 [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)):

1. Atom-level — 中心找对了吗?
2. Object-level — 类型分对了吗?
3. Edge — 键对了吗?
4. 3D 几何 — 重原子 RMSD、Z-MAE
5. Retrieval — 最相似分子排第几
6. Gap — strict vs robust 缝隙(诊断模型有没有"勉强对齐")

所有 7 个指标都进入综合分:`pred_object_score`(对象级 2D)与 `pred_object_3d_score`(对象级 3D)。

### 3.3 严格的 train/val/test split + Full-test 复评

- 训练日志 `val` 在 **val split**(随机抽 ~512 样本)
- 投稿用数字全部来自 **test split** 上的 `Full-test` 评估(`v20_eval_fulltest_object`)
- 两者差距 ~0.01,见 `fulltest_object_test.md`

### 3.4 不强求"完美",**优先稳定的提升轨迹**

V19 → V20 不是性能提升(V20 在缩减集上反而略低),而是**架构演进**:

- V19 = 主线投稿版(全样本、稳定、可复现)
- V20 = 闭环架构原型(预测对象闭环、计数头、双输入类型头)

V20 的产物是对**架构方向**的验证,不是对**数据规模**的追求。

---

## 四、与同类工作的差异

| 工作 | 路线 | 与本项目的对照 |
|---|---|---|
| AFM-Net (2D 类型分割) | 单帧 + dense 类型分割 | 不预测 z;无对象级 |
| EDAFM (Pérez 等) | 多帧 + 卷积 + GAN | 用反卷积重建表面;不直接出原子坐标 |
| ML4Chem 系列 | SchNet 类等变图网络 | 输入 = 已知原子坐标;不解 AFM |
| **本项目 V19/V20** | Video ViT + 三解码 + 对象级闭环 | 端到端从 AFM 栈到原子+键 |

---

## 五、相关阅读

- 实现细节 — [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md)
- 指标定义 — [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)
- 配置参数 — [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)
- 流程框架图 — [`PIPELINE_AND_FRAMEWORK.md`](PIPELINE_AND_FRAMEWORK.md)
- 历史复盘 — [`V1-V6_RETROSPECTIVE.md`](V1-V6_RETROSPECTIVE.md)
- 设计批判 — [`analysis/DESIGN_CRITIQUE.md`](analysis/DESIGN_CRITIQUE.md)
