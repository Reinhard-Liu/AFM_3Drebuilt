# 版本演化总览：V1 → V20

> 一页纸版本史。详细的失败分析见 [LEGACY_METHODS_V1_V16.md](LEGACY_METHODS_V1_V16.md)，V19/V20 设计原理见 [PRINCIPLES.md](PRINCIPLES.md)。

---

## 一、五个时代速览

| 时代 | 版本 | 核心范式 | 关键里程碑 | 结论 |
|------|------|---------|----------|------|
| Ⅰ. 扩散反演 | V1-V5b | DDIM 在 3D 坐标上反演 | V2 RMSD 0.255 | 信息论上限 type≈68% |
| Ⅱ. 编码器迭代 | V6-V10 | ViT/Swin/CrossAttn 强化 | V6 试 7 改动崩 | 改架构无救 |
| Ⅲ. 检索头 | V11-V14 | GNN/化合价/EDM 等变 | V14 RMSD 0.166 | N/O 类型崩 |
| Ⅳ. 架构转折 | V15-V16c | 去 SE(3)；CID 检索 | V15 首次"看上去像分子" | 检索头放大错误 |
| Ⅴ. 语义注入 | V17-V18 | Bridge/z 头/两阶段 eval | V18 视觉通过率 0.0000 | 监督颗粒度问题 |
| **Ⅵ. 对象级监督** | **V19-V20** | **object-conditioned head** | **V19 peak 0.802 / V20 pred 0.714** | **实用化达成** |

---

## 二、关键指标历史曲线

| 版本 | epochs | 主要指标 | 视觉通过率 |
|------|-------|---------|----------|
| V1 | 60 | RMSD 1.830 | ~0% |
| V2 | 60 | RMSD **0.255**, Type 43.6% | ~5% |
| V3 | 60 | RMSD 1.038（Focal Loss 灾难） | <1% |
| V5b | 50 | RMSD 0.269, Type **48.5%** | ~5% |
| V6 | 70 | RMSD 0.519（7 改动崩） | <1% |
| V7-V10 | 各 50-70 | Composite ≈ 0.49 持平 | ~5% |
| V11 | 3 (崩) | val_loss 12.139 | — |
| V14 | 50 | RMSD **0.166** / N=3.6% / O=0.2% | ~5% |
| V15 | 50 | val_loss 5.49（首次像分子） | ~5% |
| V16/V16b | 各 50 | val_loss 11.718（采样器 bug） | ~0% |
| V16c | 50 | val_loss 12.995 (修 bug 后退化) | <1% |
| V17 Bridge-A/B | 各 30 | val_loss 17.21 / 21.04 | <1% |
| V18 (5 ckpt) | 各 30-50 | visual_pass_rate **0.0000** | 0% |
| **V19_full15** | **15** | **peak_object_score 0.802** | **~50%** |
| **V20 EXP-01** | **10** | **pred_object_score 0.714** | **~45%** |

> 注：视觉通过率是人工抽 50-100 张样本后判定"骨架可识别 + 类型大致对"的比例，仅供量级参考。

---

## 三、五个根因 → 五个解决方案

V1-V18 全部失败可以归纳为 5 条根因，V19/V20 一一对应：

| 根因（V1-V18） | V19/V20 解决方案 |
|---------------|-----------------|
| 训练-部署 gap：head 训 GT 推理见 noisy | Curriculum: GT-center → peak → pred |
| 评估指标偏离用户需求 | object_score / peak_object_score / 视觉 review |
| 单监督单元（per-atom） | 监督颗粒度上抬到 object 级 |
| 化学先验作 hard constraint | 改为 logit bias 形式的 soft prior |
| 缺"对象级"中间表示 | CenterConditionedTypeHead/EdgeHead |

---

## 四、推荐阅读顺序

1. [README.md](../README.md) §"版本演化"章节 — 看可视化对比
2. [LEGACY_METHODS_V1_V16.md](LEGACY_METHODS_V1_V16.md) — 详细失败分析
3. [PRINCIPLES.md](PRINCIPLES.md) — V19/V20 设计原理
4. [V19_V20实验总索引与总结.md](V19_V20实验总索引与总结.md) — V19/V20 全实验数据
5. [V1-V6_RETROSPECTIVE.md](V1-V6_RETROSPECTIVE.md) — 早期扩散反演的数学复盘

---

> 文档版本：v1.0  最后更新：2026-04-28
