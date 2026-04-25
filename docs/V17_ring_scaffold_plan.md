# V17 结构层重构方案与设计审查

## 一、结论先行

当前计划中的 `ring scaffold` 方向 **总体正确，但当前定义过窄、结构表达过弱、条件注入方式过轻**，不足以成为真正可用的显式结构层。

V17 不应继续沿用“单环 slot + center/type + residual addition + post-hoc correction”这条实现路径，而应升级为：

**Ring-System Scaffold Graph + Attachment/Bond Layer + Denoiser Cross-Attention Conditioning**

一句话概括：
- 不是“更强一点的 ring head”
- 而是“把结构层从单环检测升级成可条件化的骨架图表示”

## 二、为什么要做显式结构层

### 2.1 当前 V16d.1 已经证明了什么

V16c / V16d.1 的改进主要解决了：
- 采样器错误
- bond guidance 死逻辑
- bond 常量不一致
- 约束训练过弱
- 杂原子类别不平衡的一部分问题

这些改动确实提升了：
- Bond Validity
- Count Accuracy
- 一部分 RMSD

但没有解决最核心的问题：
- 预测结构整体仍与 GT 相差大
- 3D 拓扑没有真正学会
- 环系结构没有被显式组织起来
- 杂原子仍大量错误
- bottom atoms 仍弱

这说明继续沿着 `V16d.x` 做 loss / guidance 微调，不足以跨越当前瓶颈。

### 2.2 文献与开源实现给出的共同结论

以下工作都支持同一个判断：

1. **JT-VAE**：分子生成在 motif / junction-tree 层建模，整体拓扑更稳定。
2. **HGraph2Graph**：显式 motif + attachment 的层次结构是有效的。
3. **MoLeR**：scaffold-conditioned generation 比纯 atom-by-atom 更适合复杂骨架。
4. **DecompDiff**：3D 生成中引入 scaffold / arm 分解是合理的。
5. **MolDiff / MiDi / JODO**：仅做原子坐标生成不够，2D 关系与 3D 几何应联合建模。

因此，对当前项目最关键的不是“把环预测得更准一点”，而是：

**让结构层成为主生成器的真实条件，而不是后验修补项。**

## 三、对当前 `ring scaffold` 方案的判断

## 3.1 方向正确的部分

当前计划里正确的部分包括：
- 用 RDKit 从现有结构自动派生 ring/scaffold 标签，而不是等待人工标注
- 让模型显式学习环相关中层结构，而不是只看 atom coordinates
- 先做 GT scaffold 条件实验，再决定是否推进 predicted scaffold generator
- 不直接做 full V17，而先做 bridge / ablation

这些判断都保留。

## 3.2 当前方案存在的核心设计问题

### 问题 1：对象粒度定义错了

当前方案以“单个 5/6 元环”为基本对象，但真实决定分子拓扑的通常不是单环，而是：
- fused ring system
- bridged / spiro ring relation
- scaffold core
- scaffold 到 sidechain 的 attachment graph

也就是说，你现在定义的是 `ring scaffold`，但真正应该定义的是：

**ring-system scaffold graph**

### 问题 2：`ring_site_index` 标签不稳定

当前 `ring_site_index` 基于 RDKit 返回的环原子顺序，这会带来两个问题：
- 环的起点不固定
- 顺时针 / 逆时针方向不固定

于是同一个环可以有多个等价 site 编号，导致：
- `atom_site_accuracy` 被人为压低
- 模型实际上被迫学习一个不稳定标签

这也是当前 `atom_site_accuracy` 很低的重要原因之一。

### 问题 3：fusion 定义不严谨

当前实现里只要两个环共享任意原子就记作 fusion，这会把：
- fused
- spiro
- 某些桥联情况

混在一起。

而在 RDKit 语义里，真正的 fused rings 更接近“共享 bond / 邻接关系”，spiro 不应直接视作 fused。

### 问题 4：多环归属被简化掉了

当前 `atom_to_ring_ids` 虽然存了多 membership，但 head/loss 实际只吃第一个 ring id。这样会丢掉：
- fused atom
- bridge atom
- junction atom

这些对拓扑最关键的信号。

### 问题 5：条件注入太弱

当前 `ScaffoldConditioner` 把所有 ring 信息压成一个全局向量，再 residual 到 `c_global`。这会丢掉：
- per-ring identity
- per-site identity
- ring-to-ring relation
- attachment structure

这类细粒度结构信息不可能靠一个 pooled vector 稳定表达。

### 问题 6：没有把 bond / attachment 纳入结构层

当前方案几乎只在做“环表示”，但真正的结构层至少还要包含：
- coarse bond graph
- attachment edges
- scaffold-to-sidechain relations
- sidechain entry sites

否则还是会遇到 atom-bond inconsistency：
- 环位置可能更对
- 但键关系和连接拓扑仍然不像真分子

### 问题 7：几何表示不充分

当前 per-ring 几何只有：
- center
- normal
- size/type

这对平面芳香环勉强可用，但对：
- cyclohexane
- 非平面杂环
- fused ring system
- 扭曲构象

远远不够。

### 问题 8：post-hoc correction 不是结构层

之前的 oracle / soft correction 证明了：
- 如果把环原子放到更合理的位置，RMSD / Bond 会改善

但那证明的是“结构信息有价值”，不是“post-hoc correction 是最终正确方案”。

真正的结构层应当在 denoising 过程中参与，而不是生成后再补。

## 四、V17 的新目标：从 `ring scaffold` 升级到 `ring-system scaffold graph`

## 4.1 新的结构层对象定义

V17 应把结构层拆成三层：

### 第一层：Ring-System Tokens
每个 token 表示一个 ring system，而不是单个 ring。

建议字段：
- `system_objectness`
- `system_type`
- `n_rings_in_system`
- `system_aromaticity`
- `system_center`
- `system_pose`
- `system_extent`
- `system_confidence`

### 第二层：Site / Attachment Layer
对每个 ring system 提供稳定的 attachment 位点表示。

建议字段：
- `canonical_site_index`
- `site_anchor_coord`
- `site_has_attachment`
- `site_attachment_type`
- `site_heteroatom_type`
- `site_role`（shared / fused / external / hetero / empty）

### 第三层：Coarse Graph Layer
显式表示 ring system 之间以及 scaffold 与 sidechain 之间的关系。

建议字段：
- `system_relation_edges`
- `relation_type`：`fused / spiro / bridged / linked / none`
- `scaffold_to_sidechain_edges`
- `entry_site_indices`
- `coarse_bond_type`

## 4.2 不是“单环”，而是“ring-system + graph”

V17 的核心对象应该是：
- fused ring system
- attachment sites
- scaffold relation graph

而不是单独的 benzene / pyridine slot。

## 五、数据标注方案

## 5.1 标签来源

优先基于：
- QUAM-AFM 当前已有的分子结构
- RDKit `RingInfo`
- RDKit 的 aromaticity / bond 信息
- ScaffoldGraph 的 scaffold / ring-system 分析逻辑

不需要等待人工标注 AFM ring 数据。

## 5.2 新标签定义

### 分子级
- `n_ring_systems`
- `n_scaffold_nodes`
- `n_attachment_edges`

### Ring-system 级
- `ring_system_id`
- `ring_system_atom_indices`
- `ring_system_center`
- `ring_system_pose`
- `ring_system_type`
- `ring_system_aromaticity`
- `ring_system_size_signature`

### 原子级
- `atom_to_ring_system_ids`（multi-label）
- `atom_is_scaffold`
- `atom_role_in_scaffold`
- `atom_canonical_site_index`
- `atom_is_attachment_anchor`

### 关系级
- `ring_system_relation_edges`
- `ring_system_relation_type`
- `ring_system_shared_atoms`
- `attachment_edges`
- `attachment_site_index`

### 化学级
- `site_heteroatom_type`
- `site_expected_bond_environment`
- `site_substituent_degree`

## 5.3 Site 编号必须重新定义

不要直接用 RDKit ring atom 顺序。

建议两种可选方案：

### 方案 A：Canonical anchor + deterministic traversal
- 先为每个 ring system 选一个 canonical anchor atom
- 再固定方向遍历
- 得到稳定 site 编号

### 方案 B：Permutation-invariant site loss
- 不固定 site 编号的绝对值
- 对所有等价循环平移 / 镜像对齐后取最小损失

V17 第一版建议优先做 **方案 B**，更稳，也更符合 ring 的对称性。

## 5.4 Relation 类型必须细分

不要再把共享任意原子都归为 fusion。

至少区分：
- `fused`
- `spiro`
- `bridged`
- `linked`
- `independent`

这一步非常关键，否则关系图会被错误标签污染。

## 六、模型架构改进方案

## 6.1 新头：`RingSystemScaffoldHead`

替代现在的 `RingScaffoldHead`。

建议输入：
- `c_global`
- `c_patches`
- 可选 atom queries / patch queries

建议输出：
- `system_objectness`
- `system_type`
- `system_center`
- `system_pose`
- `atom_to_ring_system`（multi-label 或 matching）
- `site_occupancy`
- `relation_edges`
- `relation_type`

## 6.2 条件注入方式：用 token-level cross-attention，不用 pooled residual

当前 `residual addition to c_global` 不足以支撑结构层。

V17 建议做：
- scaffold tokens 作为 denoiser 的额外 memory
- denoiser 每层或部分层读取 scaffold tokens
- 必要时对 atom slots 使用 ring-aware attention mask

这样结构层才能在每一步 denoising 中持续发挥作用。

## 6.3 生成流程：Scaffold-first + Joint Generation

流程建议：
1. AFM encoder 产生 patch features
2. `RingSystemScaffoldHead` 预测 scaffold graph
3. denoiser 用 scaffold tokens 作为条件，联合生成全部原子
4. 只在低噪声后期，对高置信 scaffold / site 做软结构约束

注意：
- 不要做硬 ring-first
- 不要把结构完全定死后再补其他原子
- 不要继续依赖 post-hoc snap 作为主机制

## 6.4 结构层要和 bond 层一起做

V17 不能只预测 scaffold，还要给出至少一个 coarse bond / attachment 层。

推荐最小做法：
- scaffold 内部 coarse adjacency
- scaffold-to-sidechain attachment edges
- site-level bond expectation

否则结构层仍然过弱。

## 七、训练与实验路线

## 7.1 V17-Bridge（必须先做）

在进入 predicted scaffold generator 前，先做一个过渡版本：

### Bridge-A：GT scaffold cross-attention
- 用 GT scaffold graph tokens 直接条件化 denoiser
- 不做 post-hoc correction
- 不做 predicted scaffold

目标：验证结构层 token conditioning 本身是否有效。

### Bridge-B：GT scaffold cross-attention + low-noise soft constraint
- 只在低噪声阶段启用软结构约束
- 检查比 pure conditioning 多带来多少增益

如果 Bridge-A 无明显收益，则说明当前结构层定义仍有问题。
如果 Bridge-A 有收益，而 predicted scaffold 还弱，则说明方向对，但 scaffold head 还不够强。

### 当前实证结论（2026-04-10）

- 未训练的 Bridge-A/Bridge-B 结果不能直接当结构层证据；随机初始化的 bridge 模块会污染判断。
- 经过 `bridge_only` 小规模训练后，`GT scaffold tokens + low-noise soft constraint` 明显优于纯 guided baseline；说明结构层方向是对的，但需要比单纯 residual/token 注入更强的条件路径。
- 默认较强的 Bridge-B 软约束会带来明显 trade-off：`RMSD / Bottom / Ring` 提升，但 `Bond` 会下滑，说明“把原子拉向 scaffold”本身还不够 bond-aware。
- 已完成的软约束扫描表明，较温和的设置更稳：
  - `t < 120, pos=0.06, plane=0.03`：Composite 相比 baseline `+0.0207`
  - `t < 120, pos=0.08, plane=0.00`：Composite 相比 baseline `+0.0264`
  - `t < 150, pos=0.08, plane=0.04`：Composite 相比 baseline `+0.0342`，是当前最优调试点
- 关闭 plane projection 后，Bond 仍会下降，说明当前 Bond 回落不只是平面投影造成的，ring-system 位置牵引本身也会改写局部键几何。
- 当前阶段的正确结论不是“Bridge-B 已经完成”，而是：
  - `Bridge-A` 单独不够
  - `Bridge-B` 明显有价值
  - 下一步应优先修 Bridge-B 的 Bond drop，而不是立刻进入 predicted scaffold generator
- 一个关键工程事实：当前 soft scaffold constraint 只在采样/评估时生效，不进入训练 loss。因此后续的 Bridge-B debug，本质上是“训练 bridge 表征，再用 soft constraint 做 inference-time 选择与验证”，而不是直接把该约束反传进模型。

## 7.2 Predicted scaffold 进入条件

只有同时满足以下条件，才建议进入真正的 predicted scaffold generator：
- `atom_to_ring_system` 明显优于随机
- site 任务不再接近随机
- relation type 有可用精度
- GT scaffold conditioning 收益稳定，不是采样噪声
- 可视化上 scaffold 真的更像 GT

## 7.3 结构层与 V16d.1 的关系

V16d.1 仍有价值，但它的定位应限定为：
- 改善局部键长
- 改善杂原子类别不平衡
- 缓解过强 guidance / constraints

不要再期待 V16d.x 解决：
- 整体拓扑
- ring system 形成
- scaffold relation graph

这些必须交给 V17 结构层。

## 八、方案漏洞与设计缺陷审查

以下是对新方案本身的反向审查。

## 8.1 风险：结构层过重，预测误差会级联放大

### 问题
如果 predicted scaffold 错得太早、太强，会把整个生成过程带偏。

### 缓解
- 先做 GT scaffold bridge
- predicted scaffold 只做 soft conditioning
- 用 confidence gating 控制结构层影响强度

## 8.2 风险：site 定义仍可能不稳定

### 问题
即便做 canonicalization，对称杂环和 fused systems 仍可能存在多解。

### 缓解
- 优先做 permutation-invariant site loss
- 不把 site accuracy 当作唯一 gate
- 更多关注 attachment correctness 和 visual topology

## 8.3 风险：环系不足以表示整分子

### 问题
有些分子的重要结构不在 ring，而在：
- 长 sidechain
- 非环杂原子骨架
- bottom atoms

### 缓解
- V17 结构层必须显式带 attachment / sidechain edge
- 不要把 ring system 当作唯一结构对象

## 8.4 风险：sim-to-real gap

### 问题
QUAM-AFM / PPAFM 都是模拟数据，实验 AFM 上结构层可能退化。

### 缓解
- 结构层仍应先在模拟数据上完成
- 后续用少量真实 AFM 做 calibration / validation
- 评估不能只看数值，也要看可视化拓扑

## 8.5 风险：relation labels 的化学语义不干净

### 问题
如果 fused/spiro/bridged/linked 的定义不严谨，结构层会被错误监督污染。

### 缓解
- 标签逻辑优先与 RDKit 语义对齐
- 单独做 scaffold label audit 脚本
- 在进入训练前先抽样核查几百个分子

## 8.6 风险：只改善 RMSD，不改善化学合理性

### 问题
如果结构层只是把原子推向 scaffold center，RMSD 可能降，但 bond / type 不一定更好。
这已经在当前 Bridge-B 实验里出现过：`RMSD / Bottom / Ring` 可提升，但 `Bond` 仍可能回落约 `0.05`。

### 缓解
- 结构层训练和评估必须同时看：
  - Bond
  - Type
  - Count
  - Bottom
  - visual topology
  - attachment correctness
- 不允许只凭 RMSD 判定成功

## 九、当前进展（截至 2026-04-12）

这一部分不是讲设计设想，而是把 4 月 8 日到现在已经做过的版本和结论整理清楚。

### 9.1 第一阶段：V16c，先把基础问题修好

这一版的目标，不是追求高分，而是先修掉之前最致命的工程问题。

这一阶段主要做了：
- 修采样流程中的错误
- 修化学键约束里不生效的逻辑
- 统一训练和评估时使用的化学键判断标准
- 重新训练并找出稳定的最佳轮次

这一版已经完成的改进：
- 预测结构不再像早期那样直接坍成一团
- 模型开始能生成“像分子”的结果，而不是完全失真
- 化学键合理性、原子数量、原子类别都有明显起色

这一版还存在的问题：
- 从整体上看，预测出的分子形状还是和真实结构差得较远
- 环和环之间的组织关系没有真正学会
- 杂原子预测仍然不稳定
- 底部原子的恢复仍然偏弱

这一阶段的结论是：
**V16c 解决了“能不能正常生成”的问题，但没有解决“能不能学会真实结构”的问题。**

### 9.2 第二阶段：V16d.1，做短线提效

这一版主要是在 V16c 的基础上做局部加强，看看不改大结构、只改训练权重和约束方式，能把结果再推多远。

这一阶段主要做了：
- 给少见原子更高的训练权重
- 强化对化学键长度的关注
- 把后期约束调得更温和，避免过度拉扯
- 调整采样时约束生效的时机和强度

这一版已经完成的改进：
- 化学键合理性进一步上升
- 原子数量预测提升明显
- 局部结构比 V16c 更规整

这一版还存在的问题：
- 它更像是“修局部细节”，不是“重建整体结构”
- 整体分子轮廓、环的组织方式、侧链和骨架的关系，仍然没有根本解决

这一阶段的结论是：
**继续停留在 V16d 这条线上，只能做局部优化，不能跨过当前最大的结构瓶颈。**

### 9.3 第三阶段：V17-lite，验证“显式结构层”值不值得做

这一版开始正式验证一个更大的方向：模型是不是需要先学会一种中间结构，再去生成完整分子。

这一阶段主要做了：
- 从已有分子结构里自动提取环相关标签
- 把这些标签接入数据读取流程
- 让模型尝试学习“哪些原子属于环、环的大致类别和位置”
- 做小规模对照实验，判断这些中间信息是否真的有帮助

这一版已经完成的改进：
- 证明了“只看原子坐标”是不够的，显式结构信息确实有价值
- 也证明了“只识别单个环”这件事本身太弱，不足以成为主结构层

这一版还存在的问题：
- 环内位置编号不稳定
- 多个环之间的连接关系表达得太弱
- 只学环，不学环和侧链的连接，无法支撑整个分子重建

这一阶段的结论是：
**方向是对的，但对象定义太窄。真正要学的不是单个环，而是“环系骨架和它连出去的结构”。**

### 9.4 第四阶段：V17 桥接版，先把真实骨架信息接进生成器

这一阶段的想法是：先不要急着让模型自己预测骨架，而是先把真实骨架信息喂给模型，验证“如果生成器知道骨架，会不会明显变好”。

这一阶段主要做了：
- 把“环系骨架”整理成可输入的中间表示
- 把这类中间表示接进生成器
- 做小规模桥接实验，观察它能否真正影响生成结果

这一版已经完成的改进：
- 证明“骨架信息进入生成器”这条路是有效的
- 不是只有后处理才有用，在生成过程中使用结构信息也有价值

但也暴露了一个关键问题：
- 如果只是把原子往骨架的位置上拉，分子的整体形状会更像一些
- 可是局部化学键可能会被拉坏，出现“形状更像了，但键更不合理”的副作用

这一阶段的结论是：
**显式结构层值得做，但不能只做几何拉回，必须同时照顾局部化学连接。**

### 9.5 第五阶段：Bridge-B，给结构层补上“局部连接”约束

在上一阶段发现“只拉位置会伤化学键”之后，这一版开始专门修这个问题。

这一阶段主要做了两步：

第一步：
- 不只是看骨架的大致位置
- 还加入骨架内部、骨架附近原子之间的局部距离修正

第二步：
- 不再只盯着环本身
- 把环和侧链之间的连接也纳入结构修正

这一版已经完成的改进：
- 解决了早期“结构更像了，但化学键变差”的问题
- 证明只看环是不够的，必须把“环系骨架 + 连接出去的部分”一起考虑
- 这一版已经成为目前最稳的结构层基础版本之一

这一版还存在的问题：
- 原子数量预测仍然是独立短板
- 虽然骨架更稳、化学键更合理，但生成器对这些结构信息的利用还不够充分

这一阶段的结论是：
**真正有效的结构层，不是“单个环”，而是“环系骨架 + 连接点 + 侧链关系”。**

### 9.6 第六阶段：让模型不只读结构，还要学会自己说出结构

这一阶段开始尝试更进一步：不仅把骨架信息喂给生成器，还要让模型自己学会预测这些结构关系，再把预测出的关系重新喂回去。

这一阶段主要做了：
- 让模型学习哪些原子属于骨架，哪些属于连出去的部分
- 让模型学习“连接关系”的中间输出
- 再把这些中间输出重新送回生成器，形成闭环

这一版已经完成的改进：
- 证明这条闭环路线是可以跑通的
- 说明模型不只是能“读懂现成骨架”，也开始能“说出一部分骨架关系”

这一版还存在的问题：
- 这些中间结果虽然有用，但还不够稳定
- 原子数量预测还是明显拖后腿
- 这一版整体上还没有超过当前最强的“真实骨架桥接版”

这一阶段的结论是：
**模型已经开始能学结构关系，但还没有强到可以完全替代真实骨架信息。**

### 9.7 第七阶段：数量补偿与结构补齐

在发现“原子数量预测拖后腿”之后，这一版专门去补这个短板。

这一阶段主要做了：
- 利用骨架信息，帮助模型更好地估计应该有多少个原子
- 继续测试生成后的小幅修正，看看能不能顺带再抬高化学键合理性

这一版已经完成的改进：
- 原子数量预测出现了目前为止最大的一次提升
- 整体综合指标达到了当前阶段最高水平

但这里要特别说明：
- 这一版之所以能提升这么多，关键原因之一是用了“真实骨架信息”来辅助数量判断
- 所以它证明的是“这条架构方向是对的，而且真实骨架信息非常有价值”
- 还不能直接说明“模型已经会自己预测骨架并达到同样效果”

这一阶段的结论是：
**我们已经证明：如果生成器拿到足够好的骨架信息，结果会明显变好。下一步真正要解决的，是把这种能力从“依赖真实骨架”推进到“依赖模型自己预测的骨架”。**

### 9.8 到目前为止，已经确认下来的事情

已经基本可以确认的结论有 5 条：

1. 继续只在旧架构上调损失、调权重、调采样，不可能解决核心问题。
2. 显式结构层是值得做的，而且已经被多轮实验验证。
3. 结构层不能只表示“单个环”，必须表示“环系骨架和连接关系”。
4. 结构层不能只管大致位置，还必须同时管局部连接是否合理。
5. 当前最接近正确方向的路线，是“环系骨架 + 连接点 + 侧链关系 + 在生成过程中持续参与条件控制”。

### 9.9 到目前为止，还没有解决的问题

目前最主要的未解决问题有 4 个：

1. 模型自己预测出来的骨架信息，还没有稳定到可以完全替代真实骨架。
2. 原子数量预测虽然被补起来了，但目前最强结果仍然借助了真实骨架信息。
3. 杂原子预测和底部原子恢复仍然不是理想水平。
4. 环和环之间更细的关系，例如共享、桥接、融合等，还没有形成足够稳固的中间表示。

### 9.10 下一步应该做什么

下一步不应该再回到旧路线做零碎调参，而应该沿着已经验证过的主线继续推进。

建议顺序如下：

1. 先把“真实骨架辅助的数量补偿”替换成“模型自己预测的数量补偿”。
2. 继续强化模型对“环系骨架、连接点、侧链关系”的预测能力。
3. 让模型预测出的结构关系真正稳定地进入生成过程，而不是只作为辅助读出结果。
4. 在这条主线上继续验证杂原子恢复、底部原子恢复和整体形状是否同步改善。

一句话总结当前进展：

**从 4 月 8 日到现在，项目已经从“修基础错误、避免结构坍塌”，走到了“确认显式结构层方向正确，并找到环系骨架加连接关系这条主线”。现在最重要的，不是再做旧式调参，而是把这条结构层主线从依赖真实骨架，推进到依赖模型自己预测的骨架。**

## 十、最终建议

V17 不应继续沿用“ring scaffold”作为狭义单环表示，而应升级为：

**Ring-System Scaffold Graph + Attachment/Bond Layer + Cross-Attention Conditioning**

执行顺序建议：
1. 先冻结 V16d.1，停止继续做纯参数微调主线
2. 重做结构层标签：ring system / relation / attachment / stable site
3. 先做 GT scaffold bridge，并以 Bridge-B 作为当前主验证线
4. Bridge-B 先把 soft constraint 调到温和区间，再专门修 `Bond drop`
5. 只有 GT bridge 的收益在可视化和 Bond 上都稳定后，再做 predicted scaffold generator
6. predicted scaffold generator 第一版也必须是 soft conditioning，不是 hard snap

### 当前新增实证：Bridge-B 需要显式覆盖 non-scaffold / sidechain edge

在 `bond-aware Bridge-B` 的基础上，只用 ring-system 内局部边长约束仍然不够。后续 `sidechain_edge_scale`
扫描表明，把 `scaffold -> non-scaffold` 的局部边也纳入 soft correction 后，`Bond / Bottom / Composite`
可以继续同步抬升，而不再只改善 scaffold 自身几何。

200-sample, DDIM-30, `t<150`, `pos=0.08`, `plane=0.04`, `edge=0.12` 的扫描结果：

- baseline: `RMSD 0.3374`, `Bond(gt) 0.5298`, `Bottom 0.0187`, `Type 0.4411`, `Ring 0.8442`, `Comp 0.5322`
- Bridge-B + `sidechain_edge_scale=0.15`: `RMSD 0.3370`, `Bond(gt) 0.5639`, `Bottom 0.1703`, `Type 0.4508`, `Ring 0.8873`, `Comp 0.5751`

这说明当前结构层的正确信号不是“只盯住 ring”，而是：

- ring-system 几何
- attachment / sidechain 局部键长
- scaffold 与非 scaffold 的连接关系

换言之，V17 的显式结构层应继续沿 `ring-system scaffold + attachment/sidechain edge` 扩展，而不是退回狭义
`ring-only scaffold`。

## 十一、主要参考来源

- JT-VAE: https://proceedings.mlr.press/v80/jin18a.html
- HGraph2Graph: https://proceedings.mlr.press/v119/jin20a.html
- MoLeR: https://www.microsoft.com/en-us/research/publication/learning-to-extend-molecular-scaffolds-with-structural-motifs/
- Microsoft molecule-generation repo: https://github.com/microsoft/molecule-generation
- DecompDiff: https://proceedings.mlr.press/v202/guan23a.html
- MolDiff: https://proceedings.mlr.press/v202/peng23b.html
- MiDi repo: https://github.com/cvignac/MiDi
- JODO repo: https://github.com/GRAPH-0/JODO
- RDKit RingInfo docs: https://www.rdkit.org/docs/cppapi/classRDKit_1_1RingInfo.html
- ScaffoldGraph repo: https://github.com/UCLCheminformatics/ScaffoldGraph
- ScaffoldGraph paper: https://academic.oup.com/bioinformatics/article/36/12/3930/5814205
- QUAM-AFM: https://pmc.ncbi.nlm.nih.gov/articles/PMC9942089/
- SPMTH-60: https://pmc.ncbi.nlm.nih.gov/articles/PMC8306777/
- PPAFM: https://github.com/Probe-Particle/ppafm
