# 如何查看预测分子的Top-5相似分子CID

## 快速使用

### 方法1：使用提供的Python脚本（推荐）

```bash
# 查看样本0的Top-5相似分子
python 查看Top5相似分子.py --sample_id 0

# 查看前20个样本
python 查看Top5相似分子.py --num_samples 20

# 保存结果到文件
python 查看Top5相似分子.py --save top5_results.txt
```

### 方法2：直接读取JSON文件

```python
import json

# 读取预测结果
with open('checkpoints/predictions_diffusion.json', 'r') as f:
    predictions = json.load(f)

# 查看第0个样本的Top-5 CID索引
sample_0 = predictions['predictions'][0]
print("Top-5 CID索引:", sample_0['retrieval_cid_indices'])
print("相似度分数:", sample_0['retrieval_scores'])
```

输出示例：
```
Top-5 CID索引: [81837, 94356, 98999, 83750, 30238]
相似度分数: [0.2426, 0.2425, 0.2357, 0.2344, 0.2331]
```

**注意**：这里的数字是**内部索引**，不是真实的PubChem CID！

---

## 数据结构说明

### predictions_diffusion.json 文件结构

```json
{
  "num_samples": 100,
  "fields": [
    "coords (3D atomic coordinates)",
    "atom_types (predicted atom types)",
    "n_atoms_pred (predicted number of atoms)",
    "retrieval_cid_indices (Top-5 candidate molecule CID indices)"
  ],
  "predictions": [
    {
      "sample_id": 0,
      "coords": [[x1, y1, z1], [x2, y2, z2], ...],
      "atom_types": [0, 1, 0, 2, ...],
      "n_atoms_pred": 36,
      "retrieval_cid_indices": [81837, 94356, 98999, 83750, 30238],
      "retrieval_scores": [0.2426, 0.2425, 0.2357, 0.2344, 0.2331]
    },
    ...
  ]
}
```

### 字段说明

| 字段 | 含义 | 数据类型 |
|------|------|---------|
| `sample_id` | 测试样本编号 | int |
| `coords` | 预测的3D原子坐标 | list of [x, y, z] |
| `atom_types` | 预测的原子类型索引 | list of int |
| `n_atoms_pred` | 预测的原子数量 | int |
| `retrieval_cid_indices` | Top-5相似分子在训练集中的**内部索引** | list of 5 ints |
| `retrieval_scores` | Top-5相似分子的余弦相似度分数 | list of 5 floats |

---

## 从内部索引转换为真实CID

训练集中有 **211,505** 个唯一的CID，模型为每个CID分配了一个内部索引（0-211504）。

### 转换方法

```python
import torch
import sys
sys.path.insert(0, '/root/autodl-tmp/micro')
from src.data.dataset import QUAMAFMDataset

# 加载训练集获取映射
checkpoint = torch.load('checkpoints/best_diffusion.pt', map_location='cpu')
config = checkpoint['config']

train_ds = QUAMAFMDataset(
    data_root=config['data_root'],
    param_key=config['param_key'],
    img_size=config['img_size'],
    min_corrugation=config['min_corrugation'],
    split='train',
)

# 构建反向映射
idx_to_cid = {idx: cid for cid, idx in train_ds.cid_to_idx.items()}

# 转换索引为真实CID
internal_index = 81837
real_cid = idx_to_cid[internal_index]
print(f"内部索引 {internal_index} 对应的真实CID: {real_cid}")
# 输出: 内部索引 81837 对应的真实CID: 136960631
```

---

## 实际案例

### 测试样本 #0 的Top-5相似分子

| 排名 | PubChem CID | 内部索引 | 相似度分数 | PubChem链接 |
|------|-------------|---------|-----------|------------|
| 1 | 136960631 | 81837 | 0.2426 | https://pubchem.ncbi.nlm.nih.gov/compound/136960631 |
| 2 | 21768352 | 94356 | 0.2425 | https://pubchem.ncbi.nlm.nih.gov/compound/21768352 |
| 3 | 22419 | 98999 | 0.2357 | https://pubchem.ncbi.nlm.nih.gov/compound/22419 |
| 4 | 137179372 | 83750 | 0.2344 | https://pubchem.ncbi.nlm.nih.gov/compound/137179372 |
| 5 | 129827823 | 30238 | 0.2331 | https://pubchem.ncbi.nlm.nih.gov/compound/129827823 |

**说明**：
- 相似度分数范围：[-1, 1]，值越大越相似
- 分数计算：余弦相似度（cosine similarity）
- 样本#0预测了36个原子（真实为21个，预测偏多）

---

## 在数据集中查找相似分子的文件

假设你想查看CID=136960631的分子：

### 1. 查找XYZ坐标文件

```bash
find /root/autodl-tmp/micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM/K-1 \
     -name "CID_136960631*" -type d
```

典型路径：
```
/root/autodl-tmp/.../K-1/CID_136960631_conf_1_K-1/
```

### 2. 查看分子坐标

```bash
cat /path/to/CID_136960631_conf_1_K-1/*.xyz
```

### 3. 查看AFM图像

```bash
ls /path/to/CID_136960631_conf_1_K-1/*_df_*.jpg

# 示例输出：
# CID_136960631_conf_1_K-1_df_000.jpg  (第0层)
# CID_136960631_conf_1_K-1_df_001.jpg  (第1层)
# ...
# CID_136960631_conf_1_K-1_df_009.jpg  (第9层)
```

---

## 相似度分数的含义

### 分数范围
- **0.0 - 0.1**: 几乎不相似
- **0.1 - 0.3**: 低度相似（当前大部分结果在这个范围）
- **0.3 - 0.5**: 中度相似
- **0.5 - 0.7**: 高度相似
- **0.7 - 1.0**: 非常相似

### 为什么当前相似度分数较低？

从样本#0的结果看，最高相似度仅0.2426，说明：

1. **原子数量预测偏差大**
   - 预测：36个原子
   - 真实：21个原子
   - 误差导致检索的分子也可能是原子数偏多的分子

2. **特征空间分散**
   - 211,505个CID的分子库很大
   - 验证集分子是全新的（不在训练集中）
   - 检索头需要在高维空间中找到最相似的分子

3. **检索任务的训练不足**
   - 从之前分析知道，验证集的 `retrieval_loss = 0.0`
   - 模型主要在训练集分子上学习检索
   - 对全新分子的泛化能力有限

---

## 批量导出所有样本的Top-5 CID

```bash
# 导出所有100个测试样本的Top-5相似分子
python 查看Top5相似分子.py --num_samples 100 --save all_top5_cids.txt

# 查看导出的文件
cat all_top5_cids.txt
```

---

## 编程接口

如果需要在自己的代码中使用，可以参考以下示例：

```python
import json
import torch
from src.data.dataset import QUAMAFMDataset

def get_top5_cids(sample_id, predictions_file, checkpoint_file):
    """获取指定样本的Top-5真实CID"""

    # 加载配置
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    config = checkpoint['config']

    # 加载训练集获取映射
    train_ds = QUAMAFMDataset(
        data_root=config['data_root'],
        param_key=config['param_key'],
        img_size=config['img_size'],
        min_corrugation=config['min_corrugation'],
        split='train',
    )

    # 构建反向映射
    idx_to_cid = {idx: cid for cid, idx in train_ds.cid_to_idx.items()}

    # 读取预测结果
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    sample = predictions['predictions'][sample_id]
    cid_indices = sample['retrieval_cid_indices']
    scores = sample['retrieval_scores']

    # 转换为真实CID
    top5_cids = [idx_to_cid[idx] for idx in cid_indices]

    return list(zip(top5_cids, scores))

# 使用示例
result = get_top5_cids(
    sample_id=0,
    predictions_file='checkpoints/predictions_diffusion.json',
    checkpoint_file='checkpoints/best_diffusion.pt'
)

for rank, (cid, score) in enumerate(result, 1):
    print(f"Rank {rank}: CID={cid}, Score={score:.4f}")
```

---

## 常见问题

### Q1: 为什么有些CID很大（如136960631），有些很小（如22419）？

**A**: PubChem CID是按分子添加到数据库的顺序递增分配的，不反映分子的大小或复杂度。

### Q2: 如何判断检索结果是否准确？

**A**: 可以：
1. 查看相似度分数（越高越好）
2. 比较预测分子和检索到的分子的原子数量
3. 在PubChem上查看分子结构图，目视对比

### Q3: 可以用这些CID做什么？

**A**:
- 在PubChem查看分子的详细化学信息
- 在数据集中找到对应的XYZ文件和AFM图像
- 作为分子识别或分类的参考
- 用于分子生成的起始点或模板

### Q4: 所有100个测试样本都有Top-5 CID吗？

**A**: 是的。每个样本都会输出5个最相似的分子CID（从211,505个训练集分子中检索）。
