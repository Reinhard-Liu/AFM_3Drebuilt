# AFM 3D Rebuilt — 从原子力显微镜图像栈重建分子三维结构

> 基于 Video Vision Transformer + 对象级联合学习的 AFM → 3D 分子结构重建框架。
> 本仓库包含主线完整代码(V19 稳定版 / V20 前沿版)、训练评估报告与核心可视化成果。

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)]() [![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-orange.svg)]() [![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## 目录

- [1. 项目定位](#1-项目定位)
- [2. 项目结构](#2-项目结构)
- [3. 项目框架与技术选型](#3-项目框架与技术选型)
- [4. 整体流程](#4-整体流程)
- [5. 关键技术说明](#5-关键技术说明)
- [6. 核心模块介绍](#6-核心模块介绍)
- [7. 各模块运行指令](#7-各模块运行指令)
- [8. 项目成果](#8-项目成果)
- [9. 数据与可视化展示](#9-数据与可视化展示)
- [10. 常见问题答疑 (FAQ)](#10-常见问题答疑-faq)
- [11. 引用与致谢](#11-引用与致谢)

---

## 1. 项目定位

**解决的问题** — 给定一组 **10 层深度切片 × 128×128** 的 AFM (原子力显微镜) 图像栈,自动重建该分子的 **三维原子结构(坐标 + 元素类型 + 化学键)**。

**技术路线** — Video ViT 编码 AFM 堆栈 → 多头解码(中心图 / 类型图 / 边 / Z-height) → 对象级联合学习(V19) → 纯预测闭环增强(V20)。

**两个主线版本**
| 版本 | 配置文件 | 定位 | 状态 |
|---|---|---|---|
| **V19 Full15** | `config_v19_object_joint_full15_all.json` | 稳定主线(全样本 68,555 分子,15 epoch) | 已投稿就绪,`peak_object_score = 0.8016` |
| **V20 Medium10** | `config_v20_object_joint_medium10.json` | 前沿探索(缩减集 65k,10 epoch,4 项消融完整) | 已完成 EXP-01~04 + SUP-01~03 |

---

## 2. 项目结构

```
.
├── src/                    # 主源代码 (~30 kLOC Python)
│   ├── data/               # 数据集加载、AFM 堆栈读取、增强、环检测
│   ├── models/             # Video ViT / 三解码器 / 中心-类型头 / 中心-边头 / Diffusion (老版)
│   ├── utils/              # 评估指标、可视化、2D 绘图
│   ├── tools/              # 实验索引生成等工具
│   ├── train_v19_object_joint.py    # 【主训练入口】(V19/V20 通用)
│   ├── train.py                     # 老版扩散主训练入口
│   ├── v20_eval_*.py                # V20 评估脚本(full-test / 检索 / 缝隙 / 几何 / 基线)
│   ├── v19_visualize_test15.py      # V19 对象级 15 样本可视化
│   ├── visualize_val.py             # 老版扩散验证集可视化
│   └── quick_test.py                # 模块 smoke 测试
│
├── docs/                   # 项目文档(V1-V20 设计与分析)
│   ├── readme.md                    # 内部开发说明
│   ├── PROJECT_DESIGN_V15.md        # 项目架构定稿
│   ├── PROJECT_DESIGN_V16.md        # 环约束扩展
│   ├── V19_V20实验总索引与总结.md    # 【关键】V19V20 全量实验索引
│   ├── V1-V6_RETROSPECTIVE.md       # 早期版本复盘
│   ├── V19_*_plan.md / V20_*_plan.md # 各阶段实验方案
│   ├── analysis/                    # 15 份专题分析报告
│   └── guides/                      # 使用指南、RDKit 安装、命令对照
│
├── scripts/                # 辅助脚本
│   ├── test/               # 端到端链路验证 (Python)
│   ├── tools/              # 监控、Top5 查看、绘图工具 (Python)
│   └── shell/              # 可视化批处理、验证修改脚本
│
├── tests/                  # 单元测试
│
├── experiments/            # 训练/评估报告存档(仅保留 reports/ plots/ samples/ figures/)
│   ├── v19_object_joint_full15_all/       # V19 主线实验产物
│   │   ├── review/reports/review_summary.{md,json}
│   │   ├── review/plots/                  # 训练曲线、学习率调度
│   │   ├── review/samples/                # best/median/worst 诊断样本
│   │   ├── visualizations_object15/       # 15 张代表性样图
│   │   ├── visual_compar_object15/        # 15 张多候选对比图
│   │   └── checkpoints/{history_v19_object_joint.json, best_preview.png}
│   ├── v19_object_joint_full6h/           # V19 短训参考
│   ├── v20_object_joint_medium10*/        # V20 主线 + EXP-01~04 + SUP-01~03
│   ├── v20_dense_baseline_*/              # 2D Dense baseline
│   ├── v20_graph_baseline_*/              # Graph baseline
│   └── v6~v16/                            # V6-V16 历史迭代的 metrics.json 与 report
│
├── visualizations/         # V2~V5b 历史可视化存档
├── outputs/                # 示例推理输出(curves/demo/分子预测)
├── real_afm/               # 真实 AFM 验证样本(11 个分子)
├── dataverse_files/readme.txt    # K-1 数据集来源说明(数据本体需外部下载)
│
├── config_v19_object_joint_full15_all.json    # V19 主线配置
├── config_v20_object_joint_medium10.json      # V20 主线配置
├── config_v19_object_joint_full6h.json        # V19 短训配置
├── config_v19_object_joint_medium.json        # V19 中等规模配置
├── config_v20_dense_stage1_medium10.json      # Dense baseline 配置
├── config_v20_graph_baseline_medium10.json    # Graph baseline 配置
├── config_v17_*_eval.json                     # V17 历史评估配置
├── config.json                                # 老版扩散主配置
│
├── run_v19_object_joint_full15_all.sh         # V19 主线训练入口
├── run_v20_object_joint_medium10.sh           # V20 主线训练入口
├── run_v19_object_joint_full6h.sh             # V19 短训入口
├── run.sh                                     # 老版扩散入口
├── monitor_*.sh / supervise_*.sh / watch_*.sh # 训练监控 / 看门狗 / 日志跟踪
│
└── README.md / LICENSE / .gitignore
```

---

## 3. 项目框架与技术选型

### 3.1 总体架构

```
AFM 图像栈 (10 层 × 128×128)
         │
         ▼
  Video ViT 编码器          ← 时空联合自注意力(patch=16,depth=6)
         │
         ▼
  共享特征图 (B, base_ch=64, 128, 128)
         │
 ┌───────┼───────────────────────┐
 ▼       ▼                       ▼
中心图   原子类型/辅助图         Z-height 图
 │       │                       │
 ▼       ▼                       ▼
  对象级联合头 (V19 主创新)
 ├─ Peak-center 解码(预测原子中心,替代 GT-center)
 ├─ Peak 条件类型头 (lambda_type_obj_peak)
 ├─ Peak 条件边头   (lambda_edge_obj_peak)
 └─ 对象计数头     (V20 新增,lambda_object_count)
         │
         ▼
  后处理(V20: 轻量边细化 → 3D 精修)
         │
         ▼
  3D 分子结构(原子坐标 + 类型 + 键)
```

### 3.2 技术栈

| 层级 | 选型 | 理由 |
|---|---|---|
| 深度学习框架 | **PyTorch ≥ 1.10** + `torch.cuda.amp` | 混合精度、CosineAnnealing 调度、断点续训生态成熟 |
| 编码器骨干 | **Video ViT** (`src/models/video_vit.py`) | 10 层 AFM 天然是"伪视频",时空自注意力能抓层间相关性 |
| 解码器 | **UNet-3 头**(中心 / 类型 / Z)+ 对象级条件头 | 三解码器共享特征图,对象级头用 peak-center 条件替代 GT-center,解决 teacher-forcing 与部署不一致问题 |
| 蒸馏 | **Type Upper Teacher** 模型 + 温度 1.5 软标签 | 杂原子 F1 从 0.22 拉到 **0.86** |
| 后处理(可选) | **RDKit MMFF94 / UFF**(位移上限 0.3 Å) | 恢复化学合理键长;`RDKIT_AVAILABLE` 自动降级 |
| 数据 | **QUAM-AFM (K-1)** 68,555 分子,10 元素 H/C/N/O/F/S/P/Cl/Br/I | 覆盖 1-85 原子范围,最大化泛化 |
| 验证数据 | **真实 AFM**:EDAFM + Camphor(Zenodo) | 验证合成 → 真实迁移可行性 |
| 训练 | AdamW + CosineAnnealingLR + warm_start | `lr=8e-5~1.5e-4`, `wd=1e-4`, bs=8 |
| 分布式 | 当前 **单卡**(代码暂未接入 DDP) | V100/A100 40GB 级别足够 |

### 3.3 目录依赖图

```
src/data/dataset.py  ───┐
                         │
src/models/video_vit.py ─┼─► src/models/v19_joint_model.py ─► src/train_v19_object_joint.py
src/models/v19_*_head.py ┘                                       │
                                                                 ▼
                        src/utils/metrics.py ◄── src/v20_eval_*.py
                        src/utils/visualize.py ◄─ src/v19_visualize_test15.py
                        src/utils/mol2d.py ◄───── src/visualize_5mol.py
```

---

## 4. 整体流程

```
[1] 环境准备        conda + CUDA + PyTorch + (可选) RDKit
        ↓
[2] 数据准备        下载 QUAM-AFM 到 /path/to/K-1/  (或 dataverse_files/)
        ↓          下载真实 AFM 到 /path/to/real_afm_datasets/  (可选)
[3] 快速自检        python3 -m src.quick_test           (smoke test,<1 分钟)
        ↓
[4] 训练            bash run_v19_object_joint_full15_all.sh   (V19 主线)
        ↓           bash run_v20_object_joint_medium10.sh     (V20 主线)
[5] 监控 / 看门狗   watch_*.sh / monitor_*.sh / supervise_*.sh
        ↓
[6] Full-test 评估  python3 -m src.v20_eval_fulltest_object    --checkpoint ... --output_dir ...
        ↓           python3 -m src.v20_eval_retrieval_full     --checkpoint ... --output_dir ...
                    python3 -m src.v20_error_decompose         --checkpoint ... --output_dir ...
                    python3 -m src.v20_geom_diagnostics        --checkpoint ... --output_dir ...
[7] Baseline 对照   python3 -m src.v20_eval_dense_baseline     ...
        ↓           python3 -m src.v20_eval_graph_baseline     ...
[8] 真实 AFM 迁移   python3 -m src.v20_eval_real_afm_cases     ...
        ↓           python3 -m src.v20_visualize_real11        ...
[9] 可视化          python3 -m src.v19_visualize_test15        --checkpoint ... --output_root ...
        ↓
[10] 复盘 & 报告     python3 -m src.v19_object_joint_review     --checkpoint ... --history ... --output_dir ...
                    python3 -m src.tools.generate_v19_v20_experiment_summary
```

---

## 5. 关键技术说明

### 5.1 Video ViT 编码器 (`src/models/video_vit.py`)

- 输入:`(B, 10, 128, 128)` AFM 堆栈(10 层 ≈ 10 帧视频)
- Patchify:`patch=16`,得到 `(B, 10, 8, 8) = 640` tokens
- 6 层时空 Transformer:每层先做 spatial-attn,再做 temporal-attn,引用 `einops` 做 rearrange
- 输出共享特征图 `(B, 64, 128, 128)` 供三个解码器消费

### 5.2 对象级联合头 (V19 核心创新 `src/models/v19_center_type_head.py` / `v19_center_edge_head.py`)

**问题** — 传统做法 "先预测原子中心图 → 阈值提取中心 → 根据中心采样类型/键" 在训练时用 GT-center 采样,部署时用 peak-center 采样,**训练部署不一致**,导致 `peak_object_score` 远低于 `gt_object_score`。

**解法** — V19 引入 **peak-center 条件头**:训练时同时用 GT-center 与 peak-center 做前向,用 curriculum(`center_curriculum_alpha_start=0.0 → _final=1.0`)逐步切换监督源,最终让模型在 peak-center 上的表现对齐 GT-center。

关键超参:
```json
{
  "lambda_type_obj_peak_start": 0.25,   "lambda_type_obj_peak_final": 2.5,
  "lambda_edge_obj_peak_start": 0.25,   "lambda_edge_obj_peak_final": 2.5,
  "center_curriculum_alpha_start": 0.0, "center_curriculum_alpha_final": 1.0,
  "center_curriculum_warmup_epochs": 12
}
```

**效果** — `peak_object_score` 从 V15 的 0.48 提升到 V19 full15 的 **0.8016**(+67%)。

### 5.3 类型上界蒸馏 (Type Upper Teacher)

- 单独训练一个 `train_v19_type_upper.py` 模型作为类型分类 teacher(用 GT-center + 局部 crop)
- 主模型学习 teacher 的 **软标签**(`temperature=1.5`),`lambda_teacher_type_distill=1.0`
- 杂原子 F1:0.22 → **0.86**(V19),**0.74**(V20 medium10 缩减集)

### 5.4 对象计数闭环(V20 新增)

`lambda_object_count=1.0` (CE) + `lambda_object_count_mae=0.15`(MAE)让模型显式学会"这个图里有多少原子",原子计数 MAE 从 Dense baseline 的 **19.86** 降到 V20 的 **0.94**(减少 21 倍)。

### 5.5 RDKit 几何精修(可选)

`src/models/postprocess.py` 接入 MMFF94 / UFF 力场,对预测坐标做 ≤0.3 Å 位移的局部弛豫,修复不合理键长。无 RDKit 时自动跳过,不影响训练。

---

## 6. 核心模块介绍

### 6.1 数据 — `src/data/dataset.py` (476 行)

- 读取 K-1 的 XYZ 坐标 + 10 层 AFM 图像切片
- 首次加载扫描目录并生成 **pkl 缓存**(后续秒级启动)
- 过滤:`min_corrugation` 去除平躺分子,`require_ring` 可只保留含环分子
- 增强:3D 旋转(`augment_rotation=true` 时 tilt>30°)、噪声
- 返回:`(afm_stack, center_map, type_map, z_map, atoms_xyz, atoms_type, bonds)`

### 6.2 编码器 — `src/models/video_vit.py` (190 行)

ViViT 风格的时空 Transformer,详见 § 5.1。

### 6.3 主模型 — `src/models/v19_joint_model.py` (163 行)

组织 Video ViT + 共享特征图 + 3 个 UNet 头(中心 / 类型 / Z)+ 2 个对象级头(peak-type / peak-edge) + 对象计数头。前向接口约定:
```python
out = model(afm_stack)
# out.center_logits, out.type_logits, out.z_pred
# out.peak_type_logits, out.peak_edge_logits, out.object_count_logits
```

### 6.4 对象级头 — `src/models/v19_center_type_head.py` / `v19_center_edge_head.py`

核心创新(见 § 5.2)。训练时接收 `gt_centers` 与 `peak_centers` 两路输入,通过 curriculum 切换。

### 6.5 损失与评估 — `src/utils/metrics.py` (1370 行)

6 维评估体系:
1. **Atom-level**:`atom_center_score_r3`, `atom_xy_mae`, `atom_z_mae_r3`
2. **Object-level**:`pred_object_score`, `pred_object_type_acc`, `pred_object_hetero_f1`
3. **Edge**:`pred_object_edge_f1` (strict) / `_f1_robust`
4. **3D 几何**:`pred_object_heavy_rmsd`, `pred_object_z_mae`, `pred_object_nonplanarity_mae`
5. **检索**:Top-k 命中率,MRR
6. **Gap 分解**:strict vs robust 差距,匹配前后类型准确率差

### 6.6 训练主入口 — `src/train_v19_object_joint.py` (1604 行)

- CLI:**仅 `--config <json>` 和 `--resume_checkpoint <pt>`**(所有超参在 JSON 内)
- 支持 warm_start_checkpoint(V20 从 V19 best 热启动)
- 支持 supervised 中途 resume(断点续训)
- 自动调度:loss warmup + aux decay + curriculum alpha

### 6.7 评估套件 — `src/v20_eval_*.py`

| 脚本 | 对应实验 | 输出 |
|---|---|---|
| `v20_eval_fulltest_object.py` | EXP-01 对象级全评估 | `reports/fulltest_object_test.{md,json,csv}` + `samples/` |
| `v20_eval_retrieval_full.py` | EXP-02 闭集检索 | `reports/retrieval_fulltest_test.{md,json}` + `plots/` |
| `v20_error_decompose.py` | EXP-03 缝隙分解 | `reports/gap_decomposition_test.*` + `plots/` + `samples/` |
| `v20_geom_diagnostics.py` | EXP-04 几何诊断 | `reports/geom_diagnostics_test.*` + `plots/` |
| `v20_eval_dense_baseline.py` | SUP-01 2D Dense 对照 | `reports/dense_baseline_fulltest.md` |
| `v20_eval_graph_baseline.py` | SUP-02 Graph 对照 | `reports/graph_baseline_fulltest.md` |
| `v20_eval_real_afm_cases.py` | SUP-03 真实 AFM | `reports/sup03_real_afm_summary.*` + `figures/` |

### 6.8 可视化 — `src/v19_visualize_test15.py` / `visualize_val.py` / `visualize_5mol.py` / `v20_visualize_real11.py`

生成 `sample_XXXXX.png`(AFM / GT / pred 并排)和 `sample_XXXXX_5mol.png`(Top-5 候选分子对比)。

---

## 7. 各模块运行指令

> 所有命令在仓库根目录下执行。训练与评估都需要 GPU。

### 7.1 环境安装

```bash
# 建议 Python 3.12 + CUDA 11.8/12.x
conda create -n micro python=3.12 -y && conda activate micro
conda install -c conda-forge pytorch pytorch-cuda numpy scipy pillow tqdm matplotlib -y
conda install -c conda-forge rdkit -y          # 可选,用于几何精修
pip install einops
```

### 7.2 数据准备

将 **QUAM-AFM (K-1)** 数据集解压到 `/path/to/K-1/`,或在 config 里把 `"data_root"` 从 `"auto"` 改为绝对路径。详见 `dataverse_files/readme.txt`。

真实 AFM(可选,SUP-03 用):
- EDAFM → https://doi.org/10.5281/zenodo.10609676
- Camphor → https://doi.org/10.5281/zenodo.4710346
- 用 `src.v20_prepare_real_afm_edafm` / `src.v20_prepare_real_afm_camphor` 预处理成 10 层 × 128×128

### 7.3 快速自检

```bash
python3 -m src.quick_test
```

### 7.4 训练

```bash
# V19 主线:全样本 68,555 分子, 15 epoch (长训)
bash run_v19_object_joint_full15_all.sh

# V20 主线:缩减集 65k, 10 epoch (中等规模)
bash run_v20_object_joint_medium10.sh

# 老版扩散模型(ViT + conditional DDPM)
bash run.sh        # ⚠️ 见 FAQ Q1,建议直接 python3 -m src.train --config config.json
```

训练日志位于 `experiments/<exp_name>/logs/`,checkpoint 存于 `experiments/<exp_name>/checkpoints/`。

### 7.5 监控与看门狗

```bash
# 跟踪日志
bash watch_v19_object_joint_full15_all.sh

# 检查训练是否卡死(30 分钟无更新判定 stall)
bash monitor_v20_object_joint_medium10.sh

# 自动 resume 卡死的训练
bash supervise_v20_object_joint_medium10.sh
```

### 7.6 评估(V20 主线)

```bash
CKPT=experiments/v20_object_joint_medium10/checkpoints/best_v19_object_joint.pt

# EXP-01 对象级 Full-test (test split, 512 样本)
python3 -m src.v20_eval_fulltest_object \
    --checkpoint $CKPT \
    --output_dir experiments/v20_object_joint_medium10_exp01_fulltest \
    --split test --batch_size 8

# EXP-02 闭集检索
python3 -m src.v20_eval_retrieval_full \
    --checkpoint $CKPT \
    --output_dir experiments/v20_object_joint_medium10_exp02_retrieval_fulltest \
    --split test --batch_size 8

# EXP-03 缝隙分解诊断
python3 -m src.v20_error_decompose \
    --checkpoint $CKPT \
    --output_dir experiments/v20_object_joint_medium10_exp03_gap_decompose \
    --batch_size 8

# EXP-04 几何诊断
python3 -m src.v20_geom_diagnostics \
    --checkpoint $CKPT \
    --output_dir experiments/v20_object_joint_medium10_exp04_geom_diagnostics \
    --batch_size 8
```

### 7.7 Baseline 对照

```bash
# SUP-01 Dense (2D) baseline
python3 -m src.v20_eval_dense_baseline \
    --checkpoint <dense_stage1_best.pt> \
    --config_path config_v20_dense_stage1_medium10.json \
    --output_dir experiments/v20_dense_stage1_medium10_sup01_fulltest \
    --batch_size 16

# SUP-02 Graph baseline
python3 -m src.v20_eval_graph_baseline \
    --checkpoint <graph_baseline_best.pt> \
    --config_path config_v20_graph_baseline_medium10.json \
    --output_dir experiments/v20_graph_baseline_medium10_sup02_fulltest \
    --eval_val_size 512
```

### 7.8 真实 AFM 迁移(SUP-03)

```bash
python3 -m src.v20_eval_real_afm_cases \
    --checkpoint $CKPT \
    --edafm_root /path/to/edafm-data \
    --camphor_structure_root /path/to/camphor/structures \
    --output_dir experiments/v20_object_joint_medium10_sup03_real_afm

python3 -m src.v20_visualize_real11 \
    --checkpoint $CKPT \
    --summary_json experiments/v20_object_joint_medium10_sup03_real_afm/reports/sup03_real_afm_summary.json \
    --real_afm_roots "experiments/v20_object_joint_medium10_sup03_real_afm/edafm_cases,experiments/v20_object_joint_medium10_sup03_real_afm/camphor_cases" \
    --output_root experiments/v20_object_joint_medium10_sup03_visual11
```

### 7.9 可视化

```bash
# V19 对象级 15 样本
python3 -m src.v19_visualize_test15 \
    --checkpoint $CKPT \
    --output_root experiments/v20_object_joint_medium10_epoch10_visual15 \
    --num_samples 15 --peak_threshold 0.45 --min_distance_px 2 --batch_size 8

# 5 分子候选对比
python3 -m src.visualize_5mol \
    --checkpoint $CKPT --num_samples 15 --output_dir outputs/5mol_demo

# 老版扩散验证集可视化
python3 -m src.visualize_val \
    --checkpoint checkpoints/best_diffusion.pt \
    --num_samples 100 --output_dir outputs/val_diffusion
```

### 7.10 复盘报告

```bash
python3 -m src.v19_object_joint_review \
    --checkpoint $CKPT \
    --history experiments/v19_object_joint_full15_all/checkpoints/history_v19_object_joint.json \
    --output_dir experiments/v19_object_joint_full15_all/review \
    --batch_size 16 --report_title "V19 Full15 Final Review"

# 重新生成 V19V20 索引表
python3 -m src.tools.generate_v19_v20_experiment_summary
```

---

## 8. 项目成果

### 8.1 V19 Full15 主线最佳指标

| 指标 | 初始值 | 最优值 | 提升 |
|---|---|---|---|
| **pred_object_score** | 0.4849 | **0.8016** | +67% |
| pred_object_type_acc | 0.2538 | **0.8185** | +56 pp |
| pred_object_hetero_f1 | 0.2201 | **0.8649** | +65 pp |
| atom_center_score_r3 | 0.9983 | 0.9991 | ≈ 上限 |
| atom_xy_mae | 0.0174 | 0.0110 | ↓ 37% |
| atom_z_mae_r3 | 0.1003 | 0.0893 | ↓ 11% |

### 8.2 V20 Medium10 全评估(512 测试样本)

**对象级综合** — `pred_object_score 0.7141 ± 0.086`,`pred_object_3d_score 0.8112 ± 0.056`。

**类型识别** — Top-1 类型准确率 **69.42%**,杂原子 F1 **74.34%**。

**图结构**

| 指标 | 数值 |
|---|---|
| 严格边 F1 | 0.6358 |
| 稳健边 F1(放宽对齐) | **0.9138** |
| 缝隙 | 0.2780(EXP-03 分解显示 80.66% 样本存在高缝隙,主要来自对齐噪声) |

**几何精度** — 重原子 RMSD **0.066 Å**,Z-MAE **0.095 Å**,键长 MAE **0.187 Å**,93.16% 样本的两两距离误差 ≤ 0.25 Å。

### 8.3 闭集检索(EXP-02)

| 指标 | 数值 |
|---|---|
| Top-1 命中率 | **74.22%** |
| Top-3 命中率 | **86.33%** |
| Top-5 命中率 | **90.23%** |
| MRR | **0.8118** |
| 中位排名 | 1.0 |

大分子(≥35 原子)Top-1 达 **80.47%**,小分子(≤22 原子) **70.75%**。

### 8.4 对比 2D Dense Baseline(SUP-01)

| 指标 | V20(Object) | Dense(2D) | 提升倍数 |
|---|---|---|---|
| pred_object_score | 0.7141 | 0.2986 | **×2.39** |
| pred_object_type_acc | 0.6942 | 0.1221 | **×5.68** |
| pred_object_macro_f1 | 0.5345 | 0.0521 | **×10.25** |
| pred_object_hetero_f1 | 0.7434 | 0.2080 | **×3.57** |
| 原子计数 MAE | **0.94** | 19.86 | **÷21.1** |

### 8.5 真实 AFM 迁移(SUP-03)

在 **11 个真实 AFM 分子**(Camphor + EDAFM 系列)上完成端到端预测与可视化,验证合成→真实的工程化可行性。详见 `experiments/v20_object_joint_medium10_sup03_real_afm_expanded/reports/sup03_real_afm_summary.md`。

---

## 9. 数据与可视化展示

### 9.1 数据集

| 数据集 | 来源 | 规模 | 用途 |
|---|---|---|---|
| **QUAM-AFM (K-1)** | Dataverse (ver. 10-2022) | 68,555 分子 × 10 层 AFM | 主训练集 |
| **EDAFM** | Zenodo 10609676 | 6 个真实 AFM 系统 | SUP-03 零样本验证 |
| **Camphor Adsorbate** | Zenodo 4710346 | 5 个樟脑吸附系统 | SUP-03 零样本验证 |

**元素覆盖** — H, C, N, O, F, S, P, Cl, Br, I(10 种)
**原子规模** — 1–85 原子/分子
**AFM 参数** — K = 40 pN/nm, Amplitude = 40 pm
**切片** — 128×128 × 10 深度

### 9.2 可视化资产总览(本仓库已含 ~700 文件)

| 类别 | 位置 | 内容 |
|---|---|---|
| **V19 训练曲线** | `experiments/v19_object_joint_full15_all/review/plots/curves.png` | 15 epoch 训练 loss/metric 曲线 |
| **学习率调度** | `.../review/plots/schedule.png` | CosineAnnealingLR 曲线 |
| **V19 诊断样本 × 3** | `.../review/samples/{best,median,worst}_sample_*.png` | 最优/中位/最差样本的端到端重建 |
| **V19 主样本 × 15** | `.../visualizations_object15/sample_*.png` | 固定 15 样本的 AFM + GT + pred |
| **V19 对比样本 × 15** | `.../visual_compar_object15/sample_*_5mol.png` | Top-5 候选分子并排 |
| **V20 EXP-01 诊断 × 3** | `experiments/v20_object_joint_medium10_exp01_fulltest/samples/` | V20 最优/中位/最差 |
| **EXP-02 检索分层图** | `experiments/v20_object_joint_medium10_exp02_retrieval_fulltest/plots/` | Top-k 随分子大小分层 |
| **EXP-03 缝隙直方图** | `experiments/v20_object_joint_medium10_exp03_gap_decompose/plots/edge_gap_robust_hist.png` | 边 F1 严格 vs 稳健差距分布 |
| **EXP-03 高缝隙样本** | `.../samples/top_gap_sample_*.png` | 展示缝隙来源的 3 个极端样本 |
| **EXP-04 几何分布** | `experiments/v20_object_joint_medium10_exp04_geom_diagnostics/plots/` | 距离 / 键长 / 非平面度的均值分布 |
| **SUP-03 真实 AFM × 11** | `experiments/v20_object_joint_medium10_sup03_real_afm_expanded/figures/` | 11 个真实分子的 AFM + 预测 3D |
| **历史版本** | `visualizations/v2_* ~ v5b_*` | V2–V5b 模型演化的样本库 |
| **训练输出样例** | `outputs/curves/ demo/ molecules_*/ sample_analysis/ test_predictions/ val_resnet/` | 训练过程中的可视化产物 |

### 9.3 关键指标表 / 报告索引

所有实验的 `*.md` 总结和 `*.json` 结构化指标汇总在 `experiments/*/reports/`。总索引见 [`docs/V19_V20实验总索引与总结.md`](docs/V19_V20实验总索引与总结.md)。

---

## 10. 常见问题答疑 (FAQ)

### Q1. `run.sh` 跑完最后几步没产出 / 报错

A. `run.sh` 的 `[3/5]` 段的 `python3 -c "..."` 通过 heredoc 传入 `$SAVE_DIR` 时会变成空字符串(已知 bug),`[2/3] resnet3d` 段也被注释。建议直接:
```bash
python3 -m src.train --config config.json
python3 -m src.visualize_val --checkpoint checkpoints/best_diffusion.pt --output_dir outputs/val_diffusion
```
详见 `docs/guides/COMMAND_COMPARISON.md`。

### Q2. 训练卡住很久没更新日志 / 显存空转

A. 本仓库自带 `monitor_v20_object_joint_medium10.sh`,以 `STALL_SECONDS=1800`(30 分钟)判定 stall,基于 `checkpoints/latest_*.pt` + `history_*.json` + `logs/*.log` 的 mtime。若 monitor 报警,直接用配套的 `supervise_*.sh` 拉起,它会读 config.epochs 与 checkpoint 的 epoch 字段,自动 `--resume_checkpoint`。

### Q3. 数据集找不到 / `data_root` 报错

A. `config` 里的 `"data_root": "auto"` 会依次尝试 `/root/autodl-tmp/K-1/` 和 `/root/autodl-tmp/micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM`。两条路都不存在就会失败。请把 QUAM-AFM 解压到其中之一,或把 `data_root` 改为绝对路径。

### Q4. 首次训练加载数据很慢

A. `src/data/dataset.py` 首次扫描 XYZ + AFM 后会生成 **pkl 缓存**,后续启动 <5 秒。如果改了 `min_corrugation` / `require_ring` / `max_samples` 等过滤项,可能需要手动删除旧 pkl 以重建。

### Q5. RDKit 是必须的吗?

A. **不是**。`src/models/postprocess.py` 通过 `try: import rdkit` 判断 `RDKIT_AVAILABLE`,无 RDKit 时自动跳过 MMFF94/UFF 弛豫步骤,训练 / 推理流程不受影响,只是预测坐标不经过几何精修。安装方法:`conda install -c conda-forge rdkit`(`docs/guides/RDKIT_INSTALLATION.md` 给了完整验证清单,版本 2025.09.6 已测通)。

### Q6. 如何做快速 smoke 验证?

A. 三种方案,按耗时由短到长:
1. `python3 -m src.quick_test` — 模块级 smoke,<1 分钟
2. 套用 `config_*_smoke.json`(smoke 配置把 `max_samples` 调小、`epochs=1-2`)+ 主训练脚本,跑几分钟
3. `config_v19_object_joint_full6h.json` — 中规模 ~6 小时训练

### Q7. 训练指标和 EXP-01 Full-test 数字对不上?

A. Full-test 用 `val_size=512` 在 **test split** 评估;训练日志的 `val` 指标在 **val split**。二者通常差 0.01 级。查 `fulltest_object_test.md` 里有 "Full-test vs Validation 参考" 差值一栏。

### Q8. 断点续训何时自动触发?

A. `run_v*.sh` 启动时会检查 `$SAVE_DIR/latest_v19_object_joint.pt`,存在就 `--resume_checkpoint <它>`,否则从 `warm_start_checkpoint` 或零开始。`supervise_*.sh` 在 stall 判定后也走这条路。

### Q9. 想切到 3D-ResNet 基线?

A. 老版 `src.train` 支持。把 `config.json` 里的 `"model_type"` 改成 `"resnet3d"`,再 `python3 -m src.train --config config.json`,产物叫 `best_resnet3d.pt` / `history_resnet3d.json`。V19/V20 对象级主线**不**走这条路径。

### Q10. 训练脚本为什么只有 `--config` 和 `--resume_checkpoint`?

A. 这是设计选择 — 所有超参(`epochs`、`lr`、`lambda_*`、`warm_start_checkpoint` 等)都放 JSON,便于实验对照和 reproduce。CLI 定义在 `src/train_v19_object_joint.py` 第 1587–1589 行。类似的 `train_v19_joint.py` / `train_v19_stage1.py` / `train_v19_type_upper.py` 同样只有 `--config`。

### Q11. 如何启动多卡 DDP?

A. **当前代码暂不支持**。所有训练以 `nohup python3 -u -m ...` 单进程启动,无 `torchrun` / `torch.distributed.launch` 调用。如需 DDP,需手动改 `src/train_v19_object_joint.py`,包装 `DistributedDataParallel` + `DistributedSampler`。社区 PR 欢迎。

### Q12. OOM(显存不够)怎么调?

A. 按影响从大到小:
1. 降 `batch_size`(当前 8)
2. 降 `num_workers`(当前 8)
3. 降 `max_samples` 做 smoke
4. 关 `augment_rotation`(减少 dataloader 内存)
5. 评估脚本 `v20_eval_*.py` 的 `--batch_size` 调到 4 或 2
6. `img_size=128` 是 pipeline 硬编码,改动需同步下游,不推荐

### Q13. 为什么没有 `requirements.txt` / `pyproject.toml`?

A. 历史原因 — 项目从 Jupyter 实验逐步长成了完整训练 pipeline,依赖积累零散。推荐依赖见 § 7.1。欢迎贡献 `pyproject.toml` PR。

---

## 11. 引用与致谢

### 数据集

- **QUAM-AFM**:Rubén Pérez et al., Universidad Autónoma de Madrid。数据见 Dataverse。详细 `readme.txt` 含完整引用,见 `dataverse_files/readme.txt`。
- **EDAFM**:Zenodo DOI [`10.5281/zenodo.10609676`](https://doi.org/10.5281/zenodo.10609676)
- **Camphor Adsorbate**:Zenodo DOI [`10.5281/zenodo.4710346`](https://doi.org/10.5281/zenodo.4710346)

### License

MIT License — 详见 [LICENSE](LICENSE)。

---

> 仓库不含训练好的模型权重、数据集本体或训练 checkpoint(见 `.gitignore`)。需要在自己的机器上从 QUAM-AFM 重新训练。V19 Full15 15 epoch 约 36 小时(单 A100),V20 Medium10 10 epoch 约 10 小时(单 A100)。
