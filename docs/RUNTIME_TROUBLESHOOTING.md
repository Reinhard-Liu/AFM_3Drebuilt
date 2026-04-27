# 运行排错(Runtime Troubleshooting)

> 涵盖**安装、数据、训练、评估**全流程常见错误。简版 FAQ 见 [README § 十二](../README.md#十二常见问题答疑faq)。

---

## 一、安装环境问题

### E1.1 `ImportError: No module named 'torch'`

```bash
conda env create -f environment.yml
conda activate afm
```

确认 environment.yml 中包含 PyTorch ≥ 2.0。如果 conda solver 卡住,试 `mamba env create -f environment.yml`。

### E1.2 `RuntimeError: CUDA error: no kernel image is available for execution on the device`

PyTorch 与 CUDA 版本不匹配。检查:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
nvidia-smi
```

通常 CUDA 11.8 / 12.1 + PyTorch 2.1+ 即可。如显示 `False`,重装对应 CUDA wheel。

### E1.3 `ImportError: cannot import name 'AllChem' from 'rdkit'`

RDKit 未正确安装。试:
```bash
conda install -c conda-forge rdkit=2023.9
# 或
pip install rdkit
```

测试:`python -c "from rdkit.Chem import AllChem; print('ok')"`。

### E1.4 `ModuleNotFoundError: No module named 'src'`

未在项目根目录运行,或未用 `python -m`。正确:
```bash
cd /path/to/AFM_micro
python -m src.train_v19_object_joint --config configs/...json
```

错误(直接运行脚本):
```bash
python src/train_v19_object_joint.py   # 会失败
```

### E1.5 Windows / Linux 路径分隔符错误

Config 文件中如出现 `\` 路径,Linux 解析失败。
- Linux:用 `/`
- Windows:Python 内部用 `/` 通常也兼容
- 推荐用 `Path` 操作或 `os.path.join`

---

## 二、数据相关错误

### E2.1 `FileNotFoundError: K-1 dataset root not found`

`data_root="auto"` 自动尝试两个路径:
- `/root/autodl-tmp/K-1/`
- `/root/autodl-tmp/micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM`

均不存在则报错。解决:
```jsonc
"data_root": "/your/abs/path/to/QUAM-AFM-Lite"
```

确认目录下含 `K-1/` 子目录与对应的分子文件。

### E2.2 `ValueError: dataset is empty after filtering`

通常 `min_corrugation` 或 `require_ring` 过严。试:
```jsonc
"min_corrugation": 0.0,
"require_ring": false
```

### E2.3 `KeyError: 'param_key' (K-1 not found)`

数据集只下载了部分参数组。确认:
```bash
ls /path/to/K-1/   # 应有大量分子目录
```

如只下载了某一参数,改 config 的 `param_key` 为实际可用的名称。

### E2.4 数据加载死锁(Windows + num_workers > 0)

Windows 多进程数据加载有时死锁。试:
```jsonc
"num_workers": 0
```

或在 Linux / WSL2 中训练。

### E2.5 `MemoryError` 数据集缓存爆内存

Dataset 双层缓存(samples + ring)在 max_samples 大时占用 RAM 很多。试:
1. 减小 `max_samples`(V20 = 65536 已是上限)
2. 降低 `num_workers`(每个 worker 独立缓存)
3. 检查代码中是否启用了 `lazy_load`(若有)

### E2.6 augment_rotation 后样本异常

旋转后边界外像素填充值不当,可能引入伪影。检查 `augmentation` 实现使用 `mode="reflect"` 或 `mode="constant", cval=0`(已是默认)。

---

## 三、训练初始化错误

### E3.1 `RuntimeError: Error(s) in loading state_dict for V19JointUNet: Missing key(s)`

Warm start ckpt 与当前模型不匹配。常见原因:
- ckpt 来自旧版本(头部分支不一致)
- `base_ch` 不同

解决:
1. 确认 warm_start ckpt 是 V19 主线的 best.pt
2. 确认当前 config 的 `base_ch=64`
3. 用 `strict=False` 加载(代码已是),只看缺失的 key 是不是新增 head 分支

### E3.2 `KeyError: 'optimizer_state_dict'` resume 时

`latest.pt` 损坏或来自 V18 之前的版本。检查:
```python
ckpt = torch.load("latest.pt")
print(ckpt.keys())   # 应含 model, optimizer, scheduler, history, epoch
```

如缺失,无法 resume,只能 warm start 重新走 curriculum。

### E3.3 Teacher 加载失败

`teacher_type_checkpoint` 找不到 → 蒸馏不启用,但训练继续。
- 检查 `train.log` 头部是否打印 `[teacher] not found, distill disabled`
- 如要启用:先训 `run_v19_type_upper_debug.sh` 生成 teacher

### E3.4 Optimizer 第一步 NaN

通常是某个新增 head 初始化未做(导致 Linear 输出爆炸)。检查:
- V20 边精化分支应**全部零初始化**(`v19_center_edge_head.py:42-76`)
- 自定义 head 用 `nn.init.normal_(weight, std=0.02)`(类似 cls_token)

调试:
```python
for name, p in model.named_parameters():
    if torch.isnan(p).any() or torch.isinf(p).any():
        print(name, p.abs().max())
```

---

## 四、训练运行时错误

### E4.1 NaN loss

诊断顺序:
1. **Lambda 配错**:`lambda_z_final=80` 等过大值 → 梯度爆炸
2. **学习率过高**:V20 用 `8e-5`(V19 的 53%),不要轻易调高
3. **batch 内含异常样本**:某分子 mask 全 0 → 除零
4. **focal-CE p_t = 0**:加 `eps=1e-8` 防止 `(1-p_t)^γ` 不稳定

定位:
```python
# 训练循环中临时启用 anomaly detection
torch.autograd.set_detect_anomaly(True)
```

### E4.2 OOM(显存不足)

按优先级:
1. `batch_size: 8 → 4 → 2`
2. `num_workers: 8 → 4 → 0`
3. 关 `augment_rotation`
4. **不要**改 `img_size`(下游头形状硬编码)
5. AMP(需自行加 GradScaler / autocast,主线代码不支持)

确认无内存泄漏:
```bash
nvidia-smi -l 1   # 显存应稳定,不持续上升
```

### E4.3 训练卡顿(无 ckpt 更新)

`monitor_*.sh` 检测到 `STALL_SECONDS` 无更新即 kill。常见原因:
1. **数据加载死锁**(num_workers + Windows)→ 见 E2.4
2. **NaN loss 但未 raise**(梯度被 grad_clip 截零)→ 检查 `train.log` 是否有 `loss=nan`
3. **CUDA OOM 死锁**(部分 GPU 在 OOM 后无法恢复)→ 重启进程
4. **磁盘满**(ckpt 写不进去)→ 检查 `df -h`

### E4.4 Loss 不下降 / 抖动剧烈

1. **Curriculum 太激进**:V20 `loss_warmup_epochs=5` 已是最低,再短会震荡 → 增加到 8 或 10
2. **Warm start 不匹配**:V20 加载非 V19 ckpt 时,中心头分布外 → 改用纯 random init
3. **Teacher 错误**:用 V18 时代的 teacher 给 V19 学 → 类型分布不匹配,KD 反向有害

### E4.5 GPU 占用低(< 50%)

1. **数据加载是瓶颈**:`num_workers` 增加,`pin_memory=True` 已开
2. **Batch 太小**:`batch_size: 8 → 16`(若显存允许)
3. **Conv 算法慢**:启用 cuDNN benchmark
   ```python
   torch.backends.cudnn.benchmark = True
   ```

### E4.6 训练速度倒退

resume 后比原始训练慢:
1. **lr scheduler 走偏**:resume 时 `T_max` 可能被重置 → 检查 `optimizer.param_groups[0]['lr']`
2. **某些 head 没启用 cudnn benchmark**:首 epoch 慢,后续应稳定
3. **磁盘 IO 退化**:tmpfs / 数据集挪到非 SSD

---

## 五、评估错误

### E5.1 `RuntimeError: Hungarian matching failed (no valid pairs)`

Pred 中 mask 全 0(空分子)。检查:
- `pred_object_count_mae` 是否 = 0(异常情况)
- 计数头预测可能在 0 类(空分子)上崩,临时跳过该样本

修复:在 metrics.py 的 `compute_object_metrics` 中加:
```python
if mask.sum() == 0 or gt_mask.sum() == 0:
    return default_metrics()  # 返回 0 / NaN
```

### E5.2 `MemoryError: scipy.optimize.linear_sum_assignment` 大 batch

Hungarian 对 n_pred × n_gt 距离矩阵求解,n > 100 内存指数增长。

解决:本项目 `MAX_ATOMS=85`,正常情况无问题。如自定义 dataset 有 > 100 原子分子,改用 `scipy.sparse.csgraph.min_weight_full_bipartite_matching`。

### E5.3 RDKit 弛豫失败率高

`postprocess.py` 的 MMFF94 → UFF fallback 链路:
- MMFF94 setup 失败 → 自动 UFF
- UFF 失败 → 跳过该样本

如失败率 > 10%,通常是模型预测含**孤立原子**(度=0)或**异常价态**。检查:
- 计数 MAE 是否过大(预测多余原子)
- 边头是否输出全 0(孤立)

### E5.4 评估指标比训练时低

训练 val 指标 vs 测试 test 指标差距大:
1. **Val / Test 切分不同样本**:正常(随机性)
2. **Test set 含特殊样本**(大分子 / 罕见类多)→ 检查 EXP-01 分层结果
3. **未用 best ckpt**:确认 `--checkpoint best.pt` 而非 `latest.pt`

### E5.5 检索 top1 = 0

Embedding 全部相似 → cls_feat 退化。检查:
1. 模型是否处于 train 模式(应 `.eval()`)
2. cls_feat 是否被零或常数填充
3. cosine 计算是否正确(query × pool / |query| / |pool|)

### E5.6 EXP-04 几何指标全 0

通常 `xy_match_radius_px=3.0` 配置下没有原子被匹配。检查:
- pred_object_center_score 是否极低(< 0.5)→ peak detection 失败
- 像素 → Å 转换是否正确(`COORD_SCALE=12.0`)

---

## 六、报告与可视化错误

### E6.1 `samples/<idx>_best.png` 缺失或全黑

`v20_eval_fulltest_object.py` 的可视化分支可能因 matplotlib backend 失败。
- Linux 服务器无 X11:`export MPLBACKEND=Agg`
- Windows:确认 PIL / matplotlib 安装完整

### E6.2 报告 markdown 数字异常

`fulltest_object_test.md` 显示 `nan` 或 `inf`:
- 评估时遇到全空 batch(所有样本异常)
- metric 函数除零

检查 `*.json` 原始字段,确认是否 `null` 而非 `nan`。

### E6.3 history.json 损坏

训练中断时 history 写入未 flush:
```python
import json
try:
    with open("history.json") as f: data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Corrupted at line {e.lineno}")
    # 手动截断到最后一个完整 epoch
```

### E6.4 复盘脚本崩溃

`v19_object_joint_review.py` 期待 history 含特定字段。如某 epoch 缺失 `val_metrics`(早 V18 版本):
```bash
python -m src.v19_object_joint_review --skip_incomplete
```

或手动补齐缺失字段为 `null`。

---

## 七、Launcher / 监控脚本错误

### E7.1 `bash: scripts/launchers/run_*.sh: Permission denied`

```bash
chmod +x scripts/launchers/*.sh
```

### E7.2 `monitor_*.sh` 一直触发 kill

`STALL_SECONDS` 太短。V20 = 1800s(30 分钟),如训练真的需要 > 30 分钟才出 ckpt(如全集 epoch),改:
```bash
STALL_SECONDS=3600   # 1 小时
```

### E7.3 `supervise_*.sh` 死循环重启

训练每次都崩(可能 OOM、NaN、或 config 错)。检查最近的 `train.log`:
```bash
tail -200 experiments/<exp>/checkpoints/train.log
```

定位错误后修复,**不要**让 supervise 一直重启 — 浪费 GPU。

### E7.4 `pgrep -f` 在 Windows 不可用

`watch_*.sh` 依赖 Linux 工具。Windows 替代:
```bash
# Git Bash 或 WSL 中运行
ps aux | grep "src.train_v19_object_joint"
```

或改用 PowerShell 实现监控。

---

## 八、常见性能问题

### P8.1 单 epoch 慢于预期(> 10 h on full set)

按降序检查:
1. `num_workers=8` 是否真的并发(`htop` 看 worker 进程)
2. cuDNN benchmark:训练前加 `torch.backends.cudnn.benchmark=True`
3. 数据集是否在 SSD(HDD 慢 5-10×)
4. 多个训练同时跑共享 GPU(`nvidia-smi`)

### P8.2 评估慢(EXP-01 > 30 分钟)

batch_size 默认评估 = 8。试 `--batch_size 32`(评估无 grad,可大 batch)。

### P8.3 RDKit 弛豫慢

每样本 100ms ~ 1s,512 样本约 1-5 分钟。优化:
- 关弛豫:`--no_rdkit_relax`
- 并行:RDKit 不天然多线程,可在评估脚本里加 `multiprocessing.Pool`

---

## 九、Git / 版本管理问题

### E9.1 git 大文件(ckpt > 100 MB)

确认 `.gitignore` 含:
```
experiments/**/checkpoints/*.pt
experiments/**/reports/samples/*.png
*.npy
*.h5
```

误提交大文件后:
```bash
git filter-repo --invert-paths --path experiments/<exp>/checkpoints/best.pt
```

### E9.2 git LFS

ckpt 需共享时用 git LFS:
```bash
git lfs track "*.pt"
git add .gitattributes
git add experiments/<exp>/checkpoints/best.pt
git commit -m "add ckpt via LFS"
```

注意 LFS 配额(GitHub 免费 1 GB)。

---

## 十、求助流程

提交 issue 时附:
1. **Python / PyTorch / CUDA 版本**:`python -c "import torch; print(torch.__version__, torch.version.cuda)"`
2. **完整报错栈**(`tail -200 train.log`)
3. **使用的 config 文件名**
4. **是否 warm start / resume**
5. **GPU 型号 + 显存**:`nvidia-smi`
6. **可复现的最小命令**

---

## 十一、相关文档

- 设计原理 — [`PRINCIPLES.md`](PRINCIPLES.md)
- 实现细节 — [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md)
- 配置参考 — [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)
- 流程框架 — [`PIPELINE_AND_FRAMEWORK.md`](PIPELINE_AND_FRAMEWORK.md)
- 指标定义 — [`METRICS_GLOSSARY.md`](METRICS_GLOSSARY.md)
- 结果解读 — [`RESULT_INTERPRETATION.md`](RESULT_INTERPRETATION.md)
- FAQ — [`FAQ_EXTENDED.md`](FAQ_EXTENDED.md)
