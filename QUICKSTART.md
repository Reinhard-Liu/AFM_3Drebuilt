# 快速开始(Quick Start)

5 分钟内跑通最短路径。完整说明见 [README.md § 九](README.md#九各模块运行指令)。

---

## 1. 环境

```bash
conda create -n micro python=3.12 -y && conda activate micro
conda install -c conda-forge pytorch pytorch-cuda numpy scipy pillow tqdm matplotlib -y
pip install einops
# 可选:conda install -c conda-forge rdkit -y
```

## 2. 数据

下载 QUAM-AFM (K-1) 解压到任一路径,将 [`configs/config_v19_object_joint_full6h.json`](configs/config_v19_object_joint_full6h.json) 中 `"data_root"` 改为该绝对路径。

## 3. 模块自检(< 1 分钟)

```bash
python3 -m src.quick_test
```

## 4. 中规模训练(单 A100 约 6 小时)

```bash
bash scripts/launchers/run_v19_object_joint_full6h.sh
```

跟踪日志:

```bash
bash scripts/launchers/watch_v19_object_joint_full6h.sh
```

## 5. 评估 + 可视化

```bash
CKPT=experiments/v19_object_joint_full6h/checkpoints/best_v19_object_joint.pt

python3 -m src.v19_visualize_test15 \
    --checkpoint $CKPT \
    --output_root experiments/v19_object_joint_full6h_visual15 \
    --num_samples 15 --batch_size 8

python3 -m src.v20_eval_fulltest_object \
    --checkpoint $CKPT \
    --output_dir experiments/v19_object_joint_full6h_eval \
    --split test --batch_size 8
```

完成后:

- 样图:`experiments/v19_object_joint_full6h_visual15/visualizations_object15/sample_*.png`
- 指标 markdown:`experiments/v19_object_joint_full6h_eval/reports/fulltest_object_test.md`

---

## 进阶

| 想要做的 | 看哪里 |
|---|---|
| V19 主线长训(15 epoch,~36h) | [README § 9.4](README.md#94-训练) |
| V20 主线 + 完整 EXP-01~04 | [README § 9.6](README.md#96-评估v20-主线) |
| 真实 AFM 零样本迁移 | [README § 9.8](README.md#98-真实-afm-迁移sup-03) |
| 训练原理 / 关键超参解释 | [README § 七](README.md#七关键技术说明) |
| 各实验报告索引 | [docs/V19_V20实验总索引与总结.md](docs/V19_V20实验总索引与总结.md) |
| 遇到错误 | [README § 十二 FAQ](README.md#十二常见问题答疑faq) |
