# 运行排错(Runtime Troubleshooting)

> 涵盖**安装、数据、训练、评估**全流程常见错误。简版 FAQ 见 [README § 十二](../README.md#十二常见问题答疑faq)。

---

## A. 安装阶段

### A1. `conda install` 卡在 "Solving environment"

```bash
conda config --set solver libmamba    # 用快速 solver
conda install -c conda-forge pytorch pytorch-cuda numpy scipy pillow tqdm matplotlib -y
```

### A2. PyTorch 与 CUDA 版本不匹配

检查:

```bash
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
nvidia-smi   # 看 driver 支持的最高 CUDA
```

错配 → 卸载重装:

```bash
conda uninstall pytorch pytorch-cuda
conda install -c pytorch -c nvidia pytorch pytorch-cuda=11.8 -y
```

### A3. `ImportError: einops`

```bash
pip install einops
```

### A4. RDKit 安装失败 / Python 版本不兼容

详见 [`guides/RDKIT_INSTALLATION.md`](guides/RDKIT_INSTALLATION.md)。

最稳:**用 conda-forge,不要 pip**;Python 必须 ≥ 3.10。

```bash
conda install -c conda-forge rdkit -y
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO'))"
```

无 RDKit 也能跑,只是后处理 MMFF/UFF 跳过。

---

## B. 数据准备阶段

### B1. `FileNotFoundError: data_root not found`

`config.json` 里 `"data_root": "auto"` 自动尝试 2 条路径:
1. `/root/autodl-tmp/K-1/`
2. `/root/autodl-tmp/micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM`

都不存在则报错。**改成绝对路径**:

```jsonc
"data_root": "/Users/me/datasets/K-1"
```

### B2. K-1 文件解压后目录结构不对

期望:

```
K-1/
├── 1/                 # 1 原子分子目录
│   ├── *.xyz          # 真实坐标 + 元素
│   └── *.png 或 *.npy # 10 层 AFM 切片
├── 2/
...
```

不符合 → 检查解压来源。Dataverse 下载有时是嵌套压缩,需多解一层。

### B3. 首次扫描数据非常慢

`src/data/dataset.py` 首次启动会 walk 所有子目录建索引 → pkl 缓存。视样本数:K-1 全量 ~5 分钟。

加速:

```bash
ls /path/to/K-1/ | wc -l   # 确认目录数
df -h /path/to/K-1/        # 确认 IO 不在网盘
```

### B4. 改了 `min_corrugation` / `require_ring` 后旧 pkl 没更新

```bash
rm experiments/<exp>/checkpoints/*_index.pkl
```

让首次启动重扫。

---

## C. 训练阶段

### C1. CUDA OOM(显存爆)

按影响优先级:

1. `batch_size: 8` → 4 或 2
2. `num_workers: 8` → 4
3. `mixed_precision: true`(确认开启)
4. `max_samples: -1` → 1000(smoke 用)
5. **不要**把 `img_size` 从 128 改小 — 下游硬编码,会炸

### C2. `RuntimeError: CUDA error: device-side assert triggered`

通常是 type 标签越界(超出 0–10):

```bash
# 重现
python -c "
from src.data.dataset import K1Dataset
ds = K1Dataset('/path/to/K-1', split='train')
sample = ds[0]
print('atom_types:', sample['atom_types'])
print('max:', sample['atom_types'].max())
"
```

最大值应 ≤ 10。否则 `dataset.py` 的 `ELEMENT_TO_IDX` 不全。

### C3. NaN loss

可能原因:

- AMP 数值不稳:暂时关 `mixed_precision: false`,定位是哪一项 loss 爆
- `lr` 太大:从 1.5e-4 降到 5e-5
- 某个 lambda 突然 × 100:检查 curriculum 调度

**调试**:

```python
# 在 train loop 临时加
torch.autograd.set_detect_anomaly(True)
```

### C4. 训练卡住,GPU 显存空转(0% util)

DataLoader 死锁(常见于 `num_workers > 0`)。

```bash
# 看是否是 worker 问题
ps -ef | grep python | head
# 把 num_workers 临时改 0
```

仓库自带 stall 监控:

```bash
bash scripts/launchers/monitor_v20_object_joint_medium10.sh
# STALL_SECONDS=1800 默认,30 分钟无更新告警
```

配套自动 resume:

```bash
bash scripts/launchers/supervise_v20_object_joint_medium10.sh
```

### C5. Resume 失败

```
Error loading state_dict: Missing key(s)
```

可能原因:
- 模型架构改了(如新增 head)
- ckpt 是 V19 版本但 config 是 V20

**解** — 用 `warm_start_checkpoint` 而非 `--resume_checkpoint`:

```jsonc
"warm_start_checkpoint": "old_ckpt.pt",
"warm_start_strict": false   // 允许部分 missing
```

### C6. 训练日志没有写出来 / 为空

检查 `experiments/<exp>/logs/train.log`:

- 文件存在但是 0 字节 → `nohup` flush 还没到。等 30 秒
- 文件不存在 → 启动器 `cd` 错了。看 `bash -x scripts/launchers/run_*.sh` 调试

### C7. `best_v19_object_joint.pt` 没生成

只有当 `val_score` 创新高才存 best。前几个 epoch 可能没存。等 ≥ 3 epoch 再看。

---

## D. 评估阶段

### D1. `v20_eval_fulltest_object` 报错 `KeyError`

模型版本与评估脚本不一致。V19 ckpt 用 V19 评估脚本:

```bash
# V19 主线
python3 -m src.v19_eval_fulltest_object --checkpoint <v19.pt> ...

# V20 主线
python3 -m src.v20_eval_fulltest_object --checkpoint <v20.pt> ...
```

### D2. RDKit 后处理报错

```
[WARNING] MMFF setup failed for sample N
```

可能原因:
- 预测的键不构成有效分子(例如孤立原子 + 缺键)
- 自动降级到 UFF;UFF 也失败则跳过该样本

**不影响整体指标**,只是该样本不做几何弛豫。

### D3. 检索 Top-K 全是 0

embedding cache 没建立。第一次跑 `v20_eval_retrieval_full` 时会扫所有 K-1 样本生成 embedding(~10 分钟)。等够时间。

### D4. Plot 生成失败

```
ImportError: matplotlib backend
```

无显示器环境:

```bash
export MPLBACKEND=Agg
```

或在脚本头加:

```python
import matplotlib
matplotlib.use('Agg')
```

---

## E. 可视化阶段

### E1. `visualize_5mol.py` 找不到分子图

依赖 RDKit 渲染 SMILES。无 RDKit 时 fallback 到点云投影。安装 RDKit 解决。

### E2. 中文字体缺失(matplotlib 警告)

```
findfont: Generic family 'sans-serif' not found
```

不影响数据,只是图上中文显示成方块。装中文字体:

```bash
# Linux
sudo apt install fonts-noto-cjk
# Mac 默认含 PingFang
```

或在 plot 脚本里指定:

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
```

---

## F. 部署 / 推理阶段

### F1. 推理时 batch_size 与训练不一致 → 性能下降?

Batch normalization 不在主模型里,LayerNorm 对 batch_size 不敏感。**不会下降**。

### F2. 单张 AFM 推理需要多少内存?

batch_size = 1,A100 显存占用 ~6 GB。CPU 推理理论可行(~30s / 样本)。

### F3. 想做端到端 demo(给 AFM 出 SMILES)

```python
import torch
from src.models.v19_joint_model import V19JointModel
from src.models.postprocess import predict_to_mol

model = V19JointModel(config)
model.load_state_dict(torch.load('best.pt')['model'])
model.eval()

with torch.no_grad():
    out = model(afm_stack[None])  # add batch dim
    mol = predict_to_mol(out, postprocess=True)
    print(Chem.MolToSmiles(mol))
```

详细 demo 脚本欢迎贡献。

---

## G. Git / 工程

### G1. `git status` 显示 `experiments/` 一堆未跟踪文件

`.gitignore` 已排除 `*.pt`、`*.npy`、大体积 cache。报告类 `*.md`、`*.json` 是有意保留(供论文复现)。

### G2. 仓库 clone 后 `bash scripts/launchers/run_*.sh` 报路径错

启动器自动 `cd` 到仓库根。检查:

```bash
echo $0           # 在 launcher 里应能 resolve 到仓库根
realpath ./       # 仓库根含 src/ configs/ scripts/ 三个目录
```

### G3. push 大文件被拒

`*.pt` 已在 `.gitignore`。如果误提交:

```bash
git rm --cached path/to/large.pt
git commit -m "Remove accidentally committed checkpoint"
```

---

## H. 还是不行?

提 [GitHub Issue](https://github.com/Reinhard-Liu/AFM_3Drebuilt/issues) 模板:

```
## 环境
- OS:
- Python:
- PyTorch:
- CUDA:
- GPU:

## 复现命令
$ ...

## 错误
[完整 stack trace]

## 已尝试
1. ...
```

参考 [README § 十二](../README.md#十二常见问题答疑faq) 与 [`FAQ_EXTENDED.md`](FAQ_EXTENDED.md) 同步检查。
