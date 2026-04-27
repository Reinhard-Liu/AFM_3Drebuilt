# 扩展 FAQ

> 涵盖**专业技术问题、运行问题、结果解读问题**。简版 FAQ 见 [README § 十二](../README.md#十二常见问题答疑faq)。

---

## 一、设计与原理

### Q1.1 为什么 AFM 切片用 Video ViT 而不是普通 2D ViT?

10 张 AFM 是同一分子在不同 z 高度的扫描,**强空间 + 强时间相关**。Video ViT 的 tubelet `(2, 16, 16)` 在 spatial-temporal 一同切块,显式建模"原子在不同 z 上的连续变化"。如果用 2D ViT 把 10 张通道堆叠,attention 只能在空间内做,失去 z 相关建模能力,z_mae 会显著变差。

### Q1.2 为什么 UNet 三分支而不是单分支多通道?

三个任务的物理含义不同:
- **Center**:从噪声 AFM 中提取局部峰(类似 Hough)→ 需要 sigmoid + BCE/Dice
- **2D 结构**:像素分类 + 键密度 → 需要 Tanh(平滑)
- **Z**:连续回归 → 需要 Tanh + L1

单分支多通道意味着所有任务共享 decoder 容量,梯度互相干扰。三分支独立 skip 让每个任务有专属的多尺度融合,在共享 bottleneck 上才共享语义。

### Q1.3 为什么对象级条件头只用 192 维主干?

经实测 192 维 + 5×5 patch grid 已饱和。增加到 256/384 维边际收益 < 0.01。容量瓶颈不在 type head,而在**输入信号**(中心位置精度)。

### Q1.4 全局计数头为什么是 86 类(0-85)?

`MAX_ATOMS=85`(QUAM-AFM Lite 中最大分子约 60 原子,留 40% 余量)。86 类含 index=0 表示空分子(不会出现,但保留作健壮性占位)。

### Q1.5 为什么不直接用 DETR 式 set prediction?

DETR 的 Hungarian loss 在小 set(< 100)上收敛慢、对类不平衡敏感。本项目走"dense + 对象级条件头"的混合路线,中间状态都有显式监督,训练更稳。Hungarian **仅在评估** 用于 pred ↔ GT 配对,不进入 loss。

### Q1.6 KD 蒸馏温度为什么选 1.5?

经验值。T=1 几乎等于直接 CE,T=2 过度软化导致信号弱。1.5 在 type head 上验证收敛最快。配合 `T²=2.25` 补偿(KD 标准做法)使梯度幅度独立于温度。

### Q1.7 为什么 V19 → V20 不彻底重训?

Warm start + 增量改造的优点:
1. V19 已在 15 epoch 投入大量计算
2. V20 新增的 head 分支(`msg_mlp`、`refine_*`)**零初始化**,等价 V19,然后逐步学
3. 缩短 V20 训练到 10 epoch + curriculum=5,总计算量仅为重训的 1/3
4. 不冒"重训性能反而下降"的风险

### Q1.8 RDKit 弛豫的 0.3 Å 位移上限怎么定的?

物理参考:
- 单键 C-C 长度 ~1.54 Å,典型变形 ±0.05 Å
- 双键 / 芳环更刚 ±0.03 Å
- 0.3 Å 大约是"键长的 20%",超过即说明力场要做大幅修正,**这通常意味着模型预测有结构错误而非几何小偏**

实验上,关闭 cap 时约 5% 样本会被 RDKit"拉坏"(RMSD 增大)。0.3 Å cap 把这个比例降到 < 0.5%。

### Q1.9 为什么 enc1 不用 BatchNorm?

输入 AFM 已经做过 z-score 归一化(mean=0, std=1),enc1 加 BN 反而会去掉有用的尺度信息(尤其是 z 高差)。后续层(enc2-bottleneck)有非线性 + ReLU,BN 才有去内部协变量偏移的作用。

### Q1.10 三级分类的"粗 3 类"分组依据?

化学性质族:
- **族 0**:H, C(纯有机骨架,极轻)
- **族 1**:N, O, S, P(供 / 受电子异质,中等)
- **族 2**:F, Cl, Br, I(卤素,重原子,电负性大)

这与 AFM 信号的"亮度等级"近似对应:H/C 暗淡,卤素亮,N/O 中等。粗分类正是"先分亮度等级"。

---

## 二、训练运行

### Q2.1 V20 训练需要多大显存?

batch_size=8、img_size=128、base_ch=64:
- V19JointUNet 主干:~1.2 GB
- Type / Edge head:~0.5 GB
- AFM batch + features:~0.8 GB
- 优化器状态(AdamW):~2.5 GB
- **总计 ~5 GB**

显存紧张可调:`batch_size=4 → ~3 GB`,`batch_size=2 → ~1.7 GB`,但梯度噪声变大,可能需要更长 warmup。

### Q2.2 训练多久?

| 配置 | 单 epoch | 总时长 |
|---|---|---|
| V19 (full15_all,K-1 全集) | ~4-5 h | ~70 h |
| V19 (medium 子集) | ~1 h | ~15 h |
| V20 (medium10) | ~1 h | ~10 h |

GPU = RTX 3090 / 4090 单卡。多卡 DDP 未在主线启用(代码中 `num_workers=8` 已是 DataLoader 加速)。

### Q2.3 训练中途断了怎么办?

**自动恢复**:`supervise_*.sh` 检测到失败后等待 10s 自动重启。`run_*.sh` 内部读 `latest_*.pt` 即 resume(完整状态:model + optimizer + scheduler + history)。

**手动恢复**:
```bash
python -m src.train_v19_object_joint \
    --config configs/config_v20_object_joint_medium10.json \
    --resume_checkpoint experiments/<exp>/checkpoints/latest_v19_object_joint.pt
```

注意:**Warm start ≠ Resume**。Warm start 只加载 model 权重(strict=False),用于跨实验初始化;Resume 是严格继续训练。

### Q2.4 哪些 lambda 应当避免动?

| λ | 默认 | 原因 |
|---|---|---|
| `lambda_center=20.0` | 最重 | center 是所有下游的源头,弱化 → peak detection 失败 → 整链塌 |
| `lambda_teacher_type_distill=1.0` | 与硬标签同权 | 蒸馏信号是稳定的 KD 来源,弱化会失去上界引导 |
| `lambda_z_final=8.0` | 末段加重 | z 信号弱,需要在 curriculum 末段加大权重补偿 |

### Q2.5 OOM 怎么办?

按优先级:
1. `batch_size: 8 → 4 → 2`(线性减显存)
2. `num_workers: 8 → 4`(减 DataLoader 内存)
3. 关 `augment_rotation`(略损精度,减预处理内存)
4. **不要**改 `img_size=128`(下游 head 形状硬编码)
5. 启用 `torch.cuda.amp` 需自行加 GradScaler / autocast(主线代码不支持)

### Q2.6 训练卡在某个 epoch 不进?

`monitor_*.sh` 检测到 `STALL_SECONDS=1800`(V20)无 ckpt/log 更新即 kill。常见原因:
1. 数据加载死锁(`num_workers > 0` + Windows 进程问题)→ 试 `num_workers=0`
2. NaN loss → 检查 train.log,通常是 lambda_z 配错或某个新增 head 初始化未做
3. CUDA OOM 死锁(部分 GPU 在 OOM 后无法恢复)→ 重启进程

### Q2.7 可以多卡 DDP 吗?

主线代码暂未原生支持 DDP。手动改造方案:
1. `torch.nn.parallel.DistributedDataParallel(model)` 包裹三个 head
2. `Sampler=DistributedSampler` 替代默认
3. 注意 KD teacher 需 `find_unused_parameters=True`(部分 epoch 不参与梯度)

未广泛测试,生产建议单卡 + 长 epoch。

### Q2.8 不同机器复现不一致?

确定性切分(CID 排序)不引入 seed 偏差,但训练仍有非确定来源:
1. CUDA atomicAdd(无法关闭)
2. Conv2d cuDNN 算法选择(可设 `torch.backends.cudnn.deterministic=True` + `benchmark=False`,牺牲速度)
3. DataLoader workers 顺序

通常 final 指标差异 ≤ 0.5%。论文复现接受这个误差范围。

---

## 三、评估与指标

### Q3.1 为什么 `pred_object_score=0.7141` 看似不高?

综合分由 7 项加权,其中:
- `atom_count_score=0.97` 已饱和
- `atom_position_score=0.97` 接近饱和
- 但 `ring_integrity_score=0.46` 拉低均值(环识别本身困难)
- `atom_semantic_score=0.55` 也拉低(罕见类 F1=0)

**真实能力**约 = 80%(2D 主指标)/ 81%(3D)。综合分体现"全方面"的难度,单看 hetero_f1=0.74 / pair_dist@0.25=93% 等单项指标更直观。

### Q3.2 为什么对外宣传用 `peak_object_score=0.8338`?

V19 主指标。V20 引入了 pred 路径,但 V19 时 peak 是部署唯一选项,所以历史报告以 peak 为主。论文写"对象级综合分 = 0.83"通常指 peak。**V20 主指标已切到 pred,数字更保守。**

### Q3.3 `pred_object_3d_score > pred_object_score` 是不是 bug?

不是。RMSD 公式 `score = 1 - rmsd / 2.0` 在 dz 项很弱时反而把 rmsd_3d 拉得相对小(因为模型 z 接近 0 ~ GT z 也接近 0,二者差距小)。3D 综合分**不应** 单独看,应配 `z_mae` 对照。

### Q3.4 `edge_gap_robust = 0.28` 表示什么?

robust edge F1(0.91)与 strict edge F1(0.64)的差。物理含义:**模型已经知道哪些原子之间应该有键(拓扑),但中心位置略偏(2-3 px),strict 阈值下不计入**。这是 V20 后续优化的精准入口。

### Q3.5 检索 top1=0.74 是不是太低?

**不低**。closed-pool 候选 = 全 512 测试样本,其中很多分子骨架相似(同 scaffold 不同 sidechain)。top5=0.90 是更宽容的指标 — 给定一个 AFM,模型能在 5 个候选中找到正确的概率达 90%。

### Q3.6 z_corr 25% 是失败吗?

部分。这意味着只有 25% 样本的 z 预测与 GT 在分布上**强相关**(≥ 0.80)。但同时:
- nonplanarity_error = 0.072 Å(平整度判断准)
- z_mae = 0.0946(归一化空间)= 1.13 Å(实际)

模型懂"分子是不是平的",但不懂"具体哪个原子稍高/稍低"。物理上是 AFM 信号在 z 方向的天然弱(振幅 0.4 Å,跨度 0.5 Å),不是模型缺陷。

### Q3.7 macro_f1 与 hetero_f1 哪个更可信?

**hetero_f1**。macro_f1 包含 P / Br / I 等罕见类(QUAM-AFM Lite 中样本数 < 5),F1=0 拉低均值;hetero_f1 是二分类(H/C vs others),不受罕见类拖累。

### Q3.8 为什么 SUP-02(Graph)的 robust edge F1 比 V20 高?

GNN 局部图建模强,在放宽阈值下能捕获邻接关系。但:
- strict edge F1 V20 高(GNN 缺中心约束)
- z_mae V20 远好(GNN 没 dense z 监督)
- 杂原子 F1 V20 远好(GNN 没 patch grid)

综合分 V20 比 Graph 高 1.3×。SUP-02 不是失败,而是"局部强 + 全局弱"的另一种 baseline。

---

## 四、数据相关

### Q4.1 QUAM-AFM Lite vs 全量 QUAM-AFM 区别?

Lite 是 1,755 分子的子集,大小 ~3 GB,适合单机训练。全量 QUAM-AFM 含 ~6.86 万分子,~120 GB,需 SSD + 大内存。本项目主线在 Lite 上跑 K-1 参数组(振幅 0.4 Å × CO 0.4 N/m)。

### Q4.2 K-1 参数组什么意思?

QUAM-AFM 提供 8 组(振幅 × 弹性常数)合成图像。K-1 = (Amp=0.4 Å, k=0.4 N/m),是"低振幅 + 软探针"组合,信号最弱但最接近真实实验设置。

### Q4.3 augment_rotation 为什么只做 XY?

Z 方向(高度)在 AFM 物理上不可旋转(下方是衬底)。XY 旋转保证模型对方向无偏好(等变性)。

### Q4.4 数据集需要预处理吗?

不需要。`QUAMAFMDataset` 在线读 .npy / .h5,在线归一化(÷12.0)。预处理一次的话可写脚本把 numpy 缓存为 `.pt`,加速 IO ~2×。

### Q4.5 自定义数据怎么接入?

继承 `QUAMAFMDataset` 实现 `__getitem__` 返回:
```python
{
    "afm": torch.float32 (10, 128, 128),
    "coords": torch.float32 (n_atoms, 3),  # ÷12.0 归一化
    "atom_types": torch.int64 (n_atoms,),  # 0..9
    "edges": torch.int64 (n_edges, 2) 或 (n,n) 二值,
    "mask": torch.bool (MAX_ATOMS=85,),
    ...
}
```

注意 10 类元素顺序必须为 `[H, C, N, O, F, S, P, Cl, Br, I]`,与 dataset 共享。

---

## 五、部署与可视化

### Q5.1 生成的 `.mol` 文件能用 PyMOL 直接打开吗?

可以。`postprocess.coords_to_mol` 返回 `RDKit Mol` 对象,可:
```python
from rdkit.Chem import MolToMolBlock
mol_block = MolToMolBlock(mol)
with open("out.mol", "w") as f: f.write(mol_block)
```

后续 PyMOL / VMD / Avogadro 通用。

### Q5.2 想做实时推理(<100ms)?

V19/V20 主线推理时延约 200-300ms(单样本,单 GPU)。优化路径:
1. **TorchScript 导出**:模型 + 头独立 trace
2. **半精度推理**:`model.half()`(需检测 NaN)
3. **batch 推理**:同时处理 32 样本,平摊 ViT 主干开销

未提供官方部署脚本。

### Q5.3 可视化样本怎么生成?

EXP-01 自动输出 `samples/<idx>_best.png`,布局:
- 输入 AFM 中心切片(2D image)
- 预测中心 heatmap + GT center 叠加
- 类型 dense map(top 3 类)
- 解码后的分子结构图(2D RDKit draw)

如需 3D:`reports/samples_3d/<idx>.html`(需手动启用 `--export_3d`)。

---

## 六、与论文常见对比

### Q6.1 与 Hapala 2017 / 2022 的比较?

| 维度 | Hapala (probe particle) | 本项目 V20 |
|---|---|---|
| 输入 | AFM(物理仿真) | AFM(QUAM-AFM 合成) |
| 方法 | 物理模拟 + 优化 | 端到端深度学习 |
| 速度 | 分钟级单样本 | 200ms 单样本 |
| 数据集 | 十几个手工分子 | 1755 分子(可扩) |
| 类型 | 仅 C/H/O/N | 10 类全有机 |
| 键 | 后处理推断 | 学习的边头 |

Hapala 适合**精确物理重建**,本项目适合**大规模高吞吐**。

### Q6.2 与 Alldritt 2020(CNN AFM)的比较?

Alldritt 用 CNN + 像素级分类。本项目改进点:
1. Video ViT 替代 CNN(显式 z 建模)
2. 对象级条件头替代像素分类(原子级 RMSD 直接监督)
3. 显式键头替代距离阈值
4. 引入 RDKit 弛豫(物理后处理)

综合分提升 ~30%,杂原子 F1 提升 7 倍以上(在同等数据规模下)。

### Q6.3 与 Diffusion-based 方法的比较?

近期有用 diffusion 做 AFM 反演的工作(2024+)。本项目走非 diffusion 路线:
- **优势**:推理快(单步前向 vs diffusion 数百步)、训练简单(无 noise schedule 调参)
- **劣势**:无内置不确定性估计;对极端噪声 AFM 鲁棒性弱

适合不同场景,可作互补。

---

## 七、复现与论文写作

### Q7.1 复现需要的核心命令?

```bash
# 1. 环境
conda env create -f environment.yml
conda activate afm

# 2. 训练 V19 主线
bash scripts/launchers/run_v19_object_joint_full15_all.sh

# 3. 训练 V20 主线
bash scripts/launchers/run_v20_object_joint_medium10.sh

# 4. 评估(6 维)
bash scripts/run_all_eval_v20.sh    # 一键执行 EXP-01~04 + SUP-01/02

# 5. 索引
python -m src.tools.generate_v19_v20_experiment_summary
```

### Q7.2 复现误差范围?

预期 final pred_object_score = 0.7141 ± 0.005(不同 GPU / cuDNN 算法导致),hetero_f1 = 0.7434 ± 0.01。差距 > 0.02 应检查 config 与 seed。

### Q7.3 论文应当引用哪些数字?

| 论文位置 | 推荐数字 | 来源 |
|---|---|---|
| Abstract | 综合分 = 0.71 / 杂原子 F1 = 0.74 | EXP-01 |
| Method 表格 | V20 vs Dense vs Graph 9 行对比 | SUP-01/02 |
| Quantitative results | 93.2% pair_dist ≤ 0.25 Å | EXP-04 |
| 检索能力 | top1 = 74%,MRR = 0.81 | EXP-02 |
| 诊断 | edge_gap_robust = 0.28 | EXP-03 |
| 消融 | curriculum -0.15,edge head -0.14 | EXP-06/07 |

### Q7.4 何时使用 peak_* vs pred_* 数字?

- **强调"算法上限"**:gt_*(KD teacher 路径)
- **强调"V19 主线"**:peak_*(V19 时主指标)
- **强调"V20 闭环部署"**:pred_*(实际推理时的能力)

通常论文同时给 peak/gt/pred,分别评估。

---

## 八、相关文档

- 设计原理 — [`PRINCIPLES.md`](PRINCIPLES.md)
- 实现细节 — [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md)
- 配置参考 — [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)
- 指标定义 — [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)
- 结果解读 — [`RESULT_INTERPRETATION.md`](RESULT_INTERPRETATION.md)
- 排错 — [`RUNTIME_TROUBLESHOOTING.md`](RUNTIME_TROUBLESHOOTING.md)
