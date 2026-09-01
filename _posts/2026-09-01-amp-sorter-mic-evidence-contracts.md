---
layout: post
title: "从 AMP 判别到 MIC 排序：模型之外的数据契约与证据边界"
date: 2026-09-01
lang: zh
categories:
  - AI for Science
  - Protein Language Model
tags:
  - Protein Language Model
  - Antimicrobial Peptide
  - LoRA
  - SupCon
  - ArcFace
  - GATv2
description: "从 AMPsorter 的分类证据进入 MIC 活性排序，复盘 Soft Prompt、LoRA、SupCon、ArcFace 与 GATv2 的真实贡献，并通过 condition-aware、sequence-disjoint 的正式对照说明：更复杂的表征并不自动带来更好的 held-out performance。"
published: true
math: true
---

> 从 LoRA、SupCon、ArcFace 到 GATv2：为什么更复杂的表征不一定带来更好的任务结果

这是 AMP Discovery 系列第二篇。

上一篇沿着 Generate → Filter → Rank 的地图深入了 AMPgenerator：语言模型可以产生一条看起来合理的 peptide sequence，却不能因此把它升级成 AMP。生成之后，系统还要回答两个监督目标完全不同的问题：它是否更接近 AMP 判别空间，以及如果值得继续保留，模型会怎样估计它在特定条件下的活性强弱。

**Figure 1 — Intended Generate → Filter → Rank architecture**

> generation-stage sequence  
> ↓  
> **Filter — AMPsorter**  
> ↓ AMP-like computational classification candidate  
> **Rank — MIC predictor**  
> ↓  
> activity-ranked computational candidate

图里的箭头是一份模块职责图，不是已经核验的自动流水线。当前没有统一 candidate schema、对象级 handoff 或一个可追踪同一条序列穿过三个模块的 orchestrator。更重要的是，AMP probability 不等于 MIC，predicted low MIC 也不等于 measured MIC。本文讨论的是 Filter 与 Rank 的计算证据，不是湿实验活性。

这次复盘不按组件名清点复杂度，也不能只看架构图中的模块名；重点是检查它们是否进入真实 tensor path、是否经过公平对照，以及结果能否绑定到同一份数据、split、checkpoint 和 prediction artifact。

全文因此围绕一个问题展开：**组件进入 checkpoint，究竟有没有转化为任务级收益？** Sorter 部分从一组相对正面的独立分类结果开始，再逐步检查 objective、threshold 和 lineage；MIC 部分则从一次失败审计出发，先修 contract，再用正式 SEQ vs SEQ_GRAPH 对照回答 Graph 是否真的带来 held-out benefit。

把数据与工件放到一起后，叙事不再是新增模块逐级提高可信度：分类结果存在但 attribution 不完整，历史 MIC 执行链并不可信，repair 让对照第一次可比较，最终结果仍然是 mixed。

## 1. 为什么 Generator 后面还需要一个 AMPsorter

Generator 与 Sorter 都可以使用 Protein Language Model，但这不表示它们在做同一件事。自回归 Generator 学习的是序列上下文：在已有前缀之后，下一个氨基酸怎样分布。Sorter 接收一条完整序列，学习一个离散判断。把两者写成最简形式，差别已经很清楚：

$$
\text{Generation: }P(x_t\mid x_{<t}),\qquad
\text{Classification: }P(\mathrm{AMP}\mid x)
$$

语言模型觉得一条 sequence plausible，只表示它与训练分布中的上下文模式相容；它可能复现了常见长度、局部 motif 或残基组合，却没有接受“AMP vs non-AMP”的直接监督。反过来，分类器也不负责解释下一个 AA 应该是什么，它需要把完整序列压缩成与二分类边界有关的表示。

这就是为什么 Generate、Filter、Rank 是三个不同的问题。Generator 扩展搜索空间，Sorter 用二分类监督删去一部分不符合目标的序列，MIC predictor 再提供连续排序信号。三个输出都可以叫 score，但生成概率、AMP-like probability 和 predicted `log2_mic` 没有可互换的语义。

一个简单的例子是：某条富含常见阳离子残基的短肽可能在生成模型下非常“顺”，Sorter 也可能给出较高 AMP-like probability；但 MIC 回归仍需要知道条件，因为同一 peptide 面向不同 Gram category 时可以对应不同 target。前两层的高分不能替第三层补齐缺失的实验条件。

所以 Sorter 的价值不是为 Generator 盖章，而是增加一种不同来源的计算证据。它把“序列分布上合理”推进到“在当前二分类数据与模型下更接近 AMP 类”，但仍不能把这个判断写成 experimentally active AMP。

AMPsorter 是学习式 Filter，以训练标签形成 decision surface，也继承标签、split 与 calibration 的限制；规则过滤和模型高分都只是计算证据。模块化允许分别排查生成、Sorter threshold 与 MIC ranking，但不能把生成概率当 AMP 分数，或把分类概率当活性分数。

## 2. AMPsorter：从 Protein LM 表征到二分类

审计后可以确认的 AMPsorter 主路径是：AA-level `BertTokenizer` 把 peptide 拆成单氨基酸 token；输入前加入 10 个 shared Soft Prompt embeddings；本地 GPT-2-style protein LM 提供序列表征；LoRA 注入 Transformer 线性映射；最后进入普通 classification 或 ArcFace path。

**Figure 2 — Verified AMPsorter path**

> AA sequence  
> → AA-level tokenizer  
> → 10 shared Soft Prompt embeddings  
> → local GPT-2-style protein LM + LoRA  
> → classifier / ArcFace path  
> → class-1 probability

这里刻意使用 **local GPT-2-style protein LM**。checkpoint 的 config、tensor shape 和本地加载路径可以验证，但当前仓库没有恢复足以证明某个公开模型发行版和预训练语料的 provenance。安全写法应停留在架构与本地 checkpoint，而不是从目录名推断外部身份。

分类数据包含 18,422 条训练/CV sequence 和 4,606 条 independent test sequence，长度都在 8–32 AA，全部由 canonical AA20 组成。训练集和独测集 exact sequence overlap 为 0。这是一个有价值的检查，但必须马上接上边界：exact overlap=0 不等于 homology-isolated split。仓库没有可核验的 clustering log、cluster membership 或 source isolation artifact，因此不能缩写成“无数据泄漏”。

Sorter 输出的是 softmax class-1 probability。它适合作为 AMP-like 排序或固定 threshold 下的判别信号，却不是 calibrated biological probability。尤其当 deployment threshold 没有被冻结时，AUC 可以相对稳定，具体 cutoff 下的 precision/recall trade-off 却可能变化很大。

这个区别决定了后面如何读结果：AUC 评价排序能力，Accuracy、F1 和 MCC 还依赖 threshold；模型结构真实存在是一类证据，结构带来多少性能变化则是另一类证据。两者不能因为出现在同一个 checkpoint 里就自动合并。

tokenizer 也是可执行 contract：当前独测全为 AA20，可沿已核验的字符级 token path 进入 backbone，但这不证明任意非 canonical 输入都安全。对短 peptide，少数残基差异仍可能保留近同源关系；没有 cluster artifact 时，只能报告 exact overlap=0，不能扩写为近同源泛化证明。

## 3. Soft Prompt 与 LoRA：参数高效适配究竟改了哪里

本项目的 Soft Prompt 是 10×768 的共享连续向量。它不是 10 个可读文本 token，而是直接插入 embedding space 的可训练前缀。所有样本共享同一组向量，它们通过上下文影响后续 AA hidden states，为 frozen backbone 提供任务级适配接口。

LoRA 则作用在 Transformer 内的 `c_attn`、`c_proj` 和 `c_fc` 线性模块，设置为 rank 16、alpha 32、dropout 0.1。它不直接替换 base weight，而是学习低秩增量；对应 LoRA tensors 在历史分类 checkpoint 中真实存在。这说明 LoRA 不是架构图里的装饰，也不是定义后从未调用的 class。

两者的差别可以理解为：Soft Prompt 从输入上下文侧影响表示，LoRA 从 Transformer 线性映射侧改变可学习变换。它们共同把需要更新的参数范围控制在完整 backbone 的一小部分，使本地训练更可行，也让参数职责更容易检查。

但“参数高效”只描述更新方式，不是性能结论。现有 Sorter 没有一组冻结了数据、split、初始化和训练流程的 non-LoRA baseline，所以只能说 LoRA 被实现、进入 checkpoint，并参与了当前分类路径；不能写 LoRA 提升了多少 AUC，更不能把整个模型效果归因于它。

这是一条很实用的证据分界：**组件进入 checkpoint，不等于组件带来增益。** 前者可以通过 module、trainable parameter 和 state-dict key 验证；后者必须依赖受控对照。如果没有对照，最诚实的答案不是猜一个贡献比例，而是保留“机制已使用，增益未隔离”。

参数量比例不是本文的主角。较少的 trainable parameters 可能降低训练成本，却不会自动选择正确的 decision boundary。state dict 中存在 LoRA tensor 证明实现真实性；若要比较 LoRA 与 full fine-tuning、frozen head 或其他 adapter，仍需冻结数据、初始化、训练顺序与评价。Article 04 不用现有主结果替这组缺失对照作答。

## 4. CE、SupCon 与 ArcFace：三个 Objective 在约束什么

Sorter 部分真正有意思的地方，不是把三个 objective 各讲一遍教程，而是看它们改变了哪一个优化对象，以及变化有没有落到最终分类指标。

| Objective | 主要约束对象 | 本项目中的直观作用 | 当前证据边界 |
|---|---|---|---|
| CE | classifier decision | 让样本被分到正确类别 | 有真实分类路径 |
| SupCon | representation geometry | 同类表示靠近、异类表示分开 | objective 存在；稳定增益未证实 |
| ArcFace | normalized angular boundary | 在角度空间加入 margin | 历史目录有约 +0.01 AUC association；非严格因果 |

CE 直接关心分类是否正确。SupCon 不直接规定 decision threshold，而是约束 pooled representation 的几何关系。当前 Sorter 的 SupCon 实现没有上一篇 Generator historical path 中把 anchor 自己计为 positive 的问题；从实现机制看，它确实改变了表示 objective。

问题出在结果解释。四组历史目录做 paired-directory 比较时，LoRA → LoRA+SupCon 的 mean validation AUC delta 是 `-0.000534`；LoRA+ArcFace → combined 时，加入 SupCon 的 mean delta 是 `+0.001800`。一个略降，一个略升，而且差异都很小。目录又没有完整冻结 source commit、初始化、batch order 和 optimizer state。

所以 SupCon 的正式结论不是“增强了判别性”，而是：**SupCon objective exists, but there is no causal evidence of stable classification improvement.** 它改变了训练约束，却没有在现存 artifact 中表现为稳定、可归因的任务收益。

ArcFace 的证据比 SupCon 稍强。LoRA → LoRA+ArcFace 的 five-fold mean AUC association 为 `+0.009768`；LoRA+SupCon → combined 为 `+0.012103`，两个比较都是 5/5 folds 同方向。它说明 ArcFace-enabled historical directories 与更高 validation AUC 存在一致关联。

但这里仍然不能写“ArcFace caused +1% AUC”。首先，历史 run manifest 不完整；其次，ArcFace branch 同时改变 pooling 和 head，而不只是增加 angular margin；再次，没有完整初始化 lineage 证明目录之间只有这一个变量。最合适的等级是 **medium-strength artifact association**：值得写入讨论，但不能升级成严格因果结果。

这个例子把 representation learning 最常见的误区说得很具体：一种 objective 在数学上更强调类内紧凑或类间间隔，不代表 downstream metric 必然提高。Representation objective 的变化，最终仍要回到相同 task、相同 split 和相同评价定义上接受检验。

三种 objective 还作用在不同层面。CE 直接通过分类 logits 受到标签监督；SupCon 需要先选择一个 pooled representation，再定义 batch 内哪些样本互为 positive；ArcFace 则对 normalized feature 与 class weight 施加 angular margin。只要 pooling、sampler 或 head 随开关一起变化，实验变量就不再只是一个 loss term。这也是为什么阅读目录名不够，必须检查真实 forward path。

SupCon 的小幅正负 delta 尤其提醒我，不应把“embedding 看起来更有结构”当作分类提升的替代指标。即使可视化聚类更紧，也可能没有改变最需要区分的 hard cases；反过来，AUC 的小变化也未必能稳定复现在另一个初始化或 split。当前 artifact 最多说明 objective 与 ArcFace context 存在交互迹象，不足以给出稳定收益结论。

ArcFace 的同方向 association 值得保留，是因为它比单个最好 fold 更有信息：两个目录对照都在 5/5 folds 中同向。但 evidence level 仍取决于 lineage。若无法证明初始化、数据顺序和其他路径完全一致，“方向一致”可以提高关注优先级，却不能完成因果识别。这样的写法看起来不如一句“ArcFace 提升一个百分点”有冲击力，却更能经受复查。

## 5. Filter 的真实结果：AUC 0.907 之后还要问什么

当前最安全的 independent anchor 是 combined fold 5，因为 evaluator 的 checkpoint path 与这份独测 artifact 可以明确绑定。它覆盖 4,606 条 sequence，并使用固定 threshold 0.5 报告分类结果。

| Evaluation | N | AUC | Accuracy @ 0.5 | F1 @ 0.5 | MCC @ 0.5 |
|---|---:|---:|---:|---:|---:|
| Combined fold-5 independent artifact | 4,606 | 0.907233 | 0.825228 | 0.843172 | 0.653739 |

这组结果足以说明 Sorter 在当前独立文件上有真实的 AMP-vs-non-AMP 排序与判别能力。为了避免不必要的小数噪声，正文可以概括为 AUC 0.907、Accuracy 0.825、F1 0.843、MCC 0.654，但 threshold 0.5 必须跟着 Accuracy/F1/MCC 一起出现。

secondary number 是 combined historical five-fold validation mean AUC 0.919114。它必须叫 validation mean，并注明 historical run manifest incomplete；不能把 0.919 与 0.907 写成两个独立测试结果，更不能挑某个 single fold 或 test-label best threshold 的更漂亮数字替代主 anchor。

AUC 之后还要问三个问题。第一，split 是否做了 homology isolation？没有证据。第二，threshold 是否稳定？历史 ArcFace folds 的 best-MCC threshold 跨 fold 波动较大，说明 ranking metric 比 deployment operating point 更稳定。第三，run lineage 是否完整？独测 fold-5 binding 清楚，但四目录训练的完整配置、初始化和运行上下文没有全部冻结。

test-label threshold scan 可以用于理解模型输出分布，却不能作为无偏独测 threshold。看到测试标签后再选 cutoff，相当于让 test 参与了决策。因此文章只使用预先固定的 0.5 指标，并把 scan 留在 analysis boundary。

最终，Filter 部分可以正面地写“存在可核验的分类证据”，同时保留三条限制：exact-only split、threshold instability、historical lineage incomplete。限制并不会抹掉 AUC 0.907；它们只是规定这个数字能回答什么、不能回答什么。

为什么选择 fold 5，而不是目录标记的 ensemble？因为“更复杂的聚合”不一定等于“更清楚的 provenance”。fold-5 evaluator 明确指向对应 checkpoint，prediction、metric 与 threshold 可以形成较短的证据链；ensemble artifact 虽然存在，却没有保存完整 checkpoint list 与 aggregation manifest。作为技术博客的主结果，可追踪性比再挤出一点指标更重要。

Accuracy、F1 与 MCC 同时给出，也能避免只用 AUC 掩盖 operating point。Accuracy 容易受类别比例影响，F1 聚焦 precision/recall 的折中，MCC 则综合四个 confusion-matrix cell；它们在 threshold 0.5 下共同描述一个具体决策点。本文不把任何单项当作“模型全面优秀”的证明，而是把它们与 AUC、threshold caveat 放在同一行阅读。

如果后续真的要部署 Filter，下一步不会是继续引用 test-label scan，而应在独立 calibration/validation 数据上冻结 threshold，再对未参与选择的数据报告性能。当前文章的任务不是完成部署，而是如实界定：Sorter 已经有比 Generator 更直接的 AMP-like classification evidence，但还没有完成 homology-level generalization 与 operating-point closure。

## 6. 从 Filter 到 Rank：为什么 MIC 比二分类更难

Sorter 的标签只有 AMP 与 non-AMP，而 MIC regression 要学习一个连续目标。更关键的是，这个目标不是 sequence 的无条件属性。更合理的建模形式是：

$$
\hat y=f(x,\mathrm{gram},\mathrm{mods}[,\mathrm{graph}]),
\qquad y=\mathrm{log2\_mic}
$$

当前 cleaned artifact 有 2,387 rows、1,109 peptide IDs 和 1,046 unique sequences。每一行对应 peptide 与粗粒度 Gram category 的聚合 target，而不是一个可追溯到具体 organism、strain 和 assay 的原始测量。877 个 peptide IDs 出现在多个 Gram category 中，其中 823 个 ID 的 target 数值并不完全相同。

这使 historical MIC path 暴露出最根本的问题：`gram_class` 没有进入模型。sequence、graph 和四个 peptide-level modification flags 相同时，模型看到的是同一个 input；数据却可能要求它输出多个不同 target。**同一个输入对应多个 target 时，先修 condition，而不是先加 GNN。**

这里不宜简单称为 label noise。不同 Gram category 下的活性差异可能包含真实生物条件差异，也可能混合 assay、strain、unit、censor 与上游聚合差异；当前数据没有足够 lineage 把它们拆开。更准确的说法是 missing condition / target ambiguity：监督问题缺少一个已存在于表格、却没有进入模型的条件变量。

这也说明 AMP probability 为什么不能替代 MIC。Sorter 只问 whole sequence 更接近哪个二分类标签；MIC predictor 则要在 condition 下估计连续 target。即使 AMP-like probability 很高，没有 Gram condition 的回归仍可能不可辨识。Rank 的难点首先是 data contract，而不是回归 head 多深。

四个 modification flags 也要保持正确口径：amidated、cyclic、lipo 与 D-AA 是 peptide-level binary features，并没有标出具体 residue site。把它们拼入回归 head 可以让模型区分这些全局状态，却不能宣称模型建立了 modification-site structure。条件变量是否存在、以什么粒度存在，必须与公开描述一致。

target 的名字同样只允许写到 `log2_mic`。数据表保存的是转换后的数值，但 raw unit、censor 规则、具体 strain 与 assay lineage 没有恢复。于是 RMSE 和 MAE 只能解释为该 target space 中的误差，不能补上一个仓库没有证明的实验单位。对连续回归来说，这些 metadata 不只是附录信息，它们决定不同数值能否被视为同一测量尺度。

historical split 按 peptide ID 互斥仍不足以解决 exact sequence identity，因为不同 ID 可能对应相同 AA sequence。repair 转向 sequence-group split，正是把模型真实看到的输入作为分组单位。它没有完成 homology isolation，却至少消除了完全相同 sequence 跨 train/test 的直接冲突。

## 7. 漂亮的 Multimodal 架构图为什么不够

historical MIC design 在概念上很完整：GPT-2 sequence representation、ESM-2 teacher/contact、GATv2 residue graph、modification flags 与 distillation objective 汇入一个 MIC predictor。只看模块名，很容易把它讲成一个成熟的 multimodal model。问题是，设计层面的合理假设在真实 tensor、data 和 artifact path 中逐步失效。

**Figure 3 — Design intention vs executable evidence**

> **Design**：sequence + ESM relation + GATv2 + mods + distillation → MIC  
> **Observed path**：  
> same peptide + different Gram → different targets，but Gram not input  
> `H_T` intended `[L,640]` → stored `[640]` global vector  
> per-sample attention intended → first batch sample reused  
> masked sequence mean intended → PAD included  
> train loop intended → no `backward()` / optimizer update

最先失效的是 condition。第二个问题是训练真实性：historical loop 计算了多个 loss component，却没有完成 total loss、`zero_grad()`、`backward()` 和 `optim.step()`，所以 current executable path 不会更新参数。一个 checkpoint 文件存在，并不能反向证明当前代码真的训练了它。

第三个问题来自 teacher artifact。设计中的 residue teacher tensor 应为 `[L,640]`，但 1,109 个 `H_T` 工件全部退化成 `[640]`，并与 global `h_T` 相同；需要二维 residue tensor 的 node distillation 路径因此不成立。第四个问题发生在 attention export：程序选择了 batch 第一个 sample 的 attention，再循环保存给同 batch 其他 sequence。同 batch、同长度的 2,521 个可比 pair 全部出现相同 relation matrix，说明 per-sample identity 被破坏。

sequence tower 还执行了未加 mask 的 mean pooling，把 PAD hidden states 一起平均。同一输入只因 batch 内最长序列不同就可能产生不同表示；旧 prediction 中也观察到了与 padding contamination 一致的 spread。最后，旧 predictions、current checkpoint 和 current source 属于不同 lineage，无法组成一个完整可重放结果。

这些不是为了罗列“以前哪里写得不好”，而是在说明一个更普遍的原则：**Architecture diagram ≠ executable evidence.** 模块 class 存在、公式写在注释里、输出目录里有文件，都不能替代对 tensor shape、sample identity、optimizer step 和 artifact hash 的检查。

真实 graph 也需要降回准确口径。它是 **ESM-2 attention-derived residue relation graph**：node 是 residue，AA ID 进入 learned embedding，relation 决定 top-k topology。它不是 PDB、预测三维坐标、atomic graph 或 modification-site graph。真实 fusion 是 late concatenation，不是 Cross-Attention。这个边界必须明确。

attention relation 的问题很能说明 sample identity 为什么重要。矩阵 shape 可以是正确的 `[L,L]`，文件数量也可以与 peptide 数量对上，但如果 batch index 取错，每个文件只是把第一条样本的关系复制到其他样本。仅检查“文件存在”和“shape 合法”会全部通过；必须构造同 batch 不同 sequence 的对照，才能发现内容身份已经丢失。

对称性也不是一个可以靠命名推断的属性。historical path 先做对称化，再按行分别归一化，每一行除以不同最大值会重新破坏矩阵对称。repair 改成一次 global-max normalization，是为了让 `(A+Aᵀ)/2` 的关系在缩放后仍保持对称。这里修复的是图构造 contract，不是证明这种 relation 最符合真实结构。

artifact lineage 的失配则发生在更高层：prediction 的时间和来源不能与 current checkpoint 对齐，当前 source 又无法生成一次真实更新。即使旧 prediction 能重算出指标，它也只能叫 historical prediction artifact。指标公式正确，不代表它可以归属于手边这个 checkpoint；provenance 是评价的一部分。

## 8. Minimal Repair：先修 Contract，再谈 GATv2

repair 的目的不是保证结果更好，而是让“Graph 是否有用”终于成为一个可以公平回答的问题。新的路径先把 Gram condition 显式映射到 3×16 embedding；按 exact canonical sequence 做 seed-42 的 grouped split，train/validation/test exact overlap 均为 0；再修复本地 GPT-2-style checkpoint 的 WTE key compatibility，并逐元素验证 loaded WTE 与 source tensor 一致。

sequence pooling 改为 attention-mask-weighted mean。sequence 与 graph 都只看 first 50 residues，避免一边看到 50、一边看到更长节点范围。ESM exporter 在 boundary 把 canonical source sequence 转为 ESM alphabet 需要的 uppercase，但 sequence hash、split identity 和 GPT-2 input contract保持不变；每个 sample 使用自己的 attention，所有 layers/heads 求均值后只截 residue positions，再先对称化、后做 global-max normalization。

graph construction 保留最小 historical idea：sequential backbone 加 relation top-k 8，degree cap 24；relation 只决定 topology，GATv2 不消费一个名义存在却实际不用的 edge weight。正式 graph tower 是 3-layer GATv2、4 heads、256-d representation、dropout 0.3 和 global mean pooling。

**Figure 4 — Repaired formal pair**

> **SEQ**  
> sequence `[768]` + Gram embedding `[16]` + mods `[4]`  
> → late concat MLP → scalar `log2_mic`  
>  
> **SEQ_GRAPH**  
> same sequence + same Gram + same mods + GATv2 graph `[256]`  
> → late concat MLP → scalar `log2_mic`

distillation 在 primary pair 中关闭。原因不是它在理论上没有价值，而是 historical teacher artifact 已损坏，而且把 distillation 与 graph 同时加入会混淆 Graph contribution。ESM-2 在 repaired comparison 中只负责 offline relation extraction。

repair 还冻结了训练和评价 contract：真实 backward/optimizer loop，run-specific manifest，validation RMSE 选择 best checkpoint，test 只在 best checkpoint 上评价一次。这样做不会自动降低 RMSE，却能保证结果属于哪份数据、哪个 split、哪个模型，以及参数是否真的更新。

固定 split 最终得到 1,930/223/234 条 train/validation/test rows，对应 838/104/104 个 unique sequences，三组 exact intersection 都是 0。超过 context 的 36 rows、19 个 unique sequences 统一裁到 first 50，而不是在 sequence 和 graph branch 采用不同长度。截断会丢失尾部信息，但至少不会制造 modality scope mismatch。

legacy WTE key 被显式迁移，并验证 loaded WTE 与 source checkpoint tensor 一致，避免模型表面启动、token embedding 却因 key mismatch 被随机初始化。

每个 formal run 都保存 config、manifest、training metrics、best checkpoint、validation/test predictions 与 metrics；manifest 再绑定 dataset、split、checkpoint 和文件 hashes。失败则写 failure artifact 并停止，不会回退到旧 teacher、旧 graph 或旧 prediction。repair 的重点因此不是多写一份日志，而是让结果对象形成闭合 lineage。

## 9. 正式对照：更复杂的 Graph 到底有没有赢

正式 pair ID 为 `formal_pair_20260901T094935Z`。两组共享 dataset SHA、split membership SHA、seed 42、base checkpoint 与 WTE、common initialization、Gram condition、mods、data order、optimizer/training settings、precision、early stopping 与 model-selection rule。唯一 architecture difference 是 SEQ_GRAPH 多了 graph tower，以及由此必然变宽的第一层 fusion input；不能声称所有 tensor shape 或参数都完全相同。

两组都执行了 1,769 次 backward 和 1,769 次 optimizer step。SEQ 有 31 个 trainable tensors 的前后 hash 发生变化，SEQ_GRAPH 有 52 个；两组 frozen sequence aggregate hash 都保持不变。两者都在 epoch 19 取得最低 validation RMSE，并各自只执行一次 final test evaluation。这次比较不是“代码里写了训练”，而是 manifest 记录了真实更新。

**Formal SEQ vs SEQ_GRAPH — all metrics in `log2_mic` target space**

| Split | Variant | RMSE | MAE | Pearson | Spearman | R² |
|---|---|---:|---:|---:|---:|---:|
| Validation | SEQ | 2.0383888069 | 1.4794696930 | 0.6939801081 | 0.6362402920 | 0.4805794148 |
| Validation | SEQ_GRAPH | 2.0341112702 | 1.4602401207 | 0.6949710646 | 0.6542653931 | 0.4827571243 |
| Held-out test | SEQ | 2.7672196781 | 1.8926761687 | 0.4940062932 | 0.5026692351 | 0.1773632861 |
| Held-out test | SEQ_GRAPH | 2.7911734374 | 1.8900832457 | 0.4924677764 | 0.5024563699 | 0.1630597389 |

validation 上，Graph 版本的五项指标都略好。如果在这里停下，很容易得出“GATv2 有效”的结论。但 held-out test 改变了方向：SEQ_GRAPH 只有 MAE 略低；RMSE 更高，Pearson、Spearman 和 R² 都略低。automatic verdict 因而是 **B — Mixed**。

分组结果也没有提供一个隐藏的全面胜利。fungi、Gram-negative 与 Gram-positive 三个 test subset 中，SEQ_GRAPH RMSE 都略高；fungi 与 Gram-negative 的 Spearman 有极小改善，Gram-negative 与 Gram-positive 的 MAE 也略低。这说明 Graph 不是每一个指标都更差，但没有 demonstrated overall benefit。

文章最安全的解释是：**Graph branch 在 validation 与部分 subgroup ranking 上产生轻微变化，但没有转化为稳定的 held-out test improvement。** 这是 single-seed result，不做显著性、稳定性或普遍优劣措辞。

validation 与 test 的方向差异正是保留 one-shot test contract 的理由。validation 用于选择 epoch，因此我们本来就更容易在它上面看到适合当前选择规则的变化；test 的职责是只在决策完成后提供一次 held-out check。如果看到 test 结果后再换 patience、改 graph top-k 或重新选 checkpoint，这份 test 就会被纳入调参过程。

两组 artifact 的完整性也经过 manifest hash 检查：config、training metrics、best checkpoint、validation prediction、test prediction 和 metrics 六类文件在两个 run 中都匹配声明值。这不等于重新评价模型，而是确认正文读取的数字仍属于被冻结的 formal pair。数字、模型和输入对象能够互相绑定，才让 mixed verdict 有可审计含义。

需要注意的是，MAE 的极小改善不能单独盖过其他整体指标，也不能把 RMSE 的上升写成 Graph 全面退步。两种误差对大残差的敏感度不同，Pearson、Spearman 与 R²又观察不同关系。automatic verdict 综合总体与 subgroup，而不是从中挑最有利的一项，正是为了避免事后选择故事。

## 10. 为什么 Graph 没有稳定赢，并不等于 Graph 没价值

看到 held-out result 后，最危险的做法是立即编造一个确定原因。当前证据只能列出 possible explanations / limitations，不能证明其中任何一项导致了结果。第一，ESM attention graph 最终仍由 sequence model 产生，它不是一个独立的实验结构 modality；Graph tower 可能重新组织 sequence-derived relation，却不保证增加新的信息来源。

第二，formal dataset 只有 1,046 个 unique sequences。第三，raw MIC unit、censor handling、assay/strain lineage 和上游 aggregation 尚未恢复。第四，split 只隔离 exact sequence，没有 homology isolation。第五，实验只有 seed 42。这些条件共同限定结果适用范围，但不能被倒过来写成“Graph 没赢就是因为数据太少”。

负结果仍然有价值。它说明 repair 后的 graph path 可以完整训练、产生不同表示并改变部分 validation/subgroup metrics；同时，更复杂的 representation 没有在 held-out test 上形成一致优势。若未来要做多 seed、homology split 或更强的独立结构信息，应另立实验，而不是在看到本次结果后修改当前 contract。

当前结论只适用于这一 cleaned artifact、first-50 contract、split、seed 与冻结超参数；换数据、独立结构来源或多 seed 都是新实验。SEQ baseline 已包含 sequence representation、Gram embedding 与 modification flags，Graph 需要证明新增 topology 在此基础上提供额外信息，而正式结果没有显示稳定 test improvement。

回到整个 AMP Discovery 系统，这次 Filter → Rank 复盘留下五条更稳定的经验：

1. Generate、Filter、Rank 对应不同监督目标，任何一个模块都不能替另一个模块提供证据。
2. 参数高效组件进入 state dict，只证明机制被使用，不证明它带来了 performance gain。
3. Representation objective 的变化必须回到 task metric，不能用几何直觉替代实验。
4. MIC regression 首先要求 condition、split、pooling 和 lineage 正确，然后才值得比较复杂模型。
5. More complex representation ≠ better held-out performance.

Sorter 给出了真实但有边界的分类证据；MIC audit 证明漂亮架构可能在 data/tensor/artifact path 中失效；minimal repair 又让一次负面的正式对照变得可信。这比把每个模块都写成“提升性能”更接近项目真实进展。

最后仍要回到 candidate boundary。AMP-like prediction 不等于 experimentally active AMP，predicted `log2_mic` 不等于 measured MIC，computational candidate 也不等于 validated drug candidate。当前没有核验新的 synthesis、wet-lab MIC、toxicity、hemolysis、animal 或 clinical evidence。

这也是 Article 04 最终想保留的结论：架构复杂度可以改变表示，objective 可以改变优化轨迹，Graph 可以改变 validation 与 subgroup behavior；但只有 held-out、manifest-bound、claim-bounded 的任务结果，才决定我们能否说“它真的更好”。
