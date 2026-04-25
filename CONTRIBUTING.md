# 贡献指南(Contributing)

欢迎贡献。本仓库是一个研究型项目,目前处于论文投稿就绪阶段,贡献以下方向最受欢迎:

## 优先方向

1. **`pyproject.toml` / `requirements.txt`** — 当前依赖列表分散在 README,缺乏 `pip install -e .` 入口。
2. **多卡 DDP 支持** — `src/train_v19_object_joint.py` 当前仅单进程,需包装 `DistributedDataParallel` + `DistributedSampler`(见 README FAQ Q11)。
3. **更小的 smoke 配置** — 帮助新读者在 5 分钟内跑通最小训练循环。
4. **Bug 修复** — 尤其是 `run.sh` 内 heredoc 传 `$SAVE_DIR` 的已知问题(见 README FAQ Q1)。
5. **可视化脚本增强** — 例如交互式 3D 查看器。

## 工作流

1. Fork 本仓库
2. 创建分支:`git checkout -b feature/<short-name>`
3. 提交:本仓库使用 `Co-Authored-By` 风格的 commit message,但你的贡献按你的习惯写就好
4. 推送 + 提 PR,在 PR 描述里说明:
   - 解决什么问题
   - 修改了哪些文件
   - 如何验证(命令 + 预期输出)

## 代码风格

- Python:遵循源码现有风格(无强制 Black / isort)
- Shell:`set -euo pipefail` 起手,新增 launcher 沿用 `scripts/launchers/run_v*.sh` 自动定位 `ROOT` 的写法
- 路径:相对仓库根写,不使用 `/root/autodl-tmp/...` 等机器特定绝对路径

## 测试

- 修改 `src/data/`、`src/models/` 后,跑 `python3 -m src.quick_test`(<1 分钟)
- 修改训练逻辑后,用 `configs/config_v19_object_joint_full6h.json`(~6h)做端到端验证

## 报告 Bug / 提议方案

GitHub Issues。请尽量给:

- 错误堆栈
- 复现命令(完整 + 数据准备步骤)
- 环境(Python、PyTorch、CUDA、OS)

## 不接受的修改

- 提交模型权重 / 数据集到仓库(走 Zenodo / Hugging Face)
- 在源码引入硬编码个人路径
