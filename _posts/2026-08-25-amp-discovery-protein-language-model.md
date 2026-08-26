---
layout: post
title: "从 Protein Language Model 到抗菌肽计算候选：一个生成—判别—活性预测系统的设计"
date: 2026-08-25 20:00:00 +0800
categories:
  - AI for Science
  - Protein Language Model
tags:
  - Protein Language Model
  - Antimicrobial Peptide
  - Soft Prompt
  - SupCon
  - AI for Science
description: "从 Generate → Filter → Rank 的完整 AMP Discovery 架构出发，深入 Protein-LM 生成模块，复盘 condition identity、Soft Prompt、SupCon、生成评测与 artifact provenance，并解释为什么更低的 Loss 并不自动意味着更好的生成结果。"
published: true
lang: zh
math: true
---

> 先看完整 Generate → Filter → Rank 架构，再深入 AMPgenerator：为什么 Loss 下降还远远不够

如果一个 Protein Language Model 能续写出 `KLLK...`、`RRWK...` 这样的 peptide sequence，它们就能被叫作抗菌肽吗？当然不能。语言模型首先学到的是序列分布：哪些氨基酸更可能相邻、一个前缀后面通常出现什么、什么样的局部模式更像训练数据。它并不会因为输出看起来“像肽”，就自动完成 AMP 判别、活性估计和湿实验验证。

这也是我重新梳理这个项目后最重要的认识：真正的问题从来不只是“生成一条序列”，而是如何把生成、判断与活性排序拆成边界清楚的计算链路。本篇先建立整个 AMP Discovery 系统的地图，再深入入口模块 AMPgenerator；AMPsorter 与 MIC predictor 的算法和实验留到下一篇。

这次复盘不从设计图倒推代码，而是从真实 tensor、checkpoint 与 artifact 反向确认系统究竟做了什么，并把设计与执行之间的偏差变成可检查的工程 contract。

## 1. 整个 AMP Discovery 系统为什么拆成 Generate → Filter → Rank

这个项目的系统设计可以压缩成三个动词：**Generate → Filter → Rank**。三个模块处理的不是同一个问题，也不能互相替代。

**Figure 1 — Whole AMP Discovery System**

> Raw biological data（上游 provenance 尚未完整恢复）  
> ↓  
> **AMPgenerator — Generate**  
> ↓ generated peptide sequences  
> **AMPsorter — Filter**  
> ↓ AMP-like / filtered sequences  
> **MIC predictor — Rank**  
> ↓ activity-ranked computational candidates

Generator 近似学习序列分布，关心“下一个氨基酸是什么”；条件版本还会接收一个条件信号。Sorter 接收完整序列，回答它是否更接近项目所关注的 AMP 判别空间。MIC predictor 则输出连续的计算估计，用于比较通过筛选的序列在模型下的活性强弱。用更直观的话说：Generator 扩大搜索空间，Sorter 缩小不符合目标的空间，MIC predictor 再为剩余结果提供排序信号。

为什么不让一个大模型一次完成全部任务？因为三层监督信号不同。Generator 面对的是序列上下文，学习“在这个前缀之后什么更像训练分布”；Sorter 面对的是离散判别标签，学习完整序列落在哪一侧；MIC predictor 面对的是连续目标，希望相近或不同的预测值能形成可用排序。把三个输出都叫“分数”，并不会让它们具有相同语义。生成概率高的序列可能只是在语言模型看来常见，分类概率高的序列也未必拥有较低的实验 MIC。

从系统设计上，希望 Generator、Sorter 与 MIC predictor 分别缩小序列生成、AMP 判别和活性排序中的不同不确定性。三层结果需要各自的训练数据与评价 contract；拆开以后，问题也能被定位到具体模块，而不是被一个最终分数掩盖。

举个简单的系统例子：一条序列可以在语言模型下拥有较高生成概率，却被 Sorter 判断为不接近 AMP；另一条序列可以通过判别阈值，但 MIC predictor 仍可能把它排在较弱位置。这不是三个模型彼此矛盾，而是它们分别观察序列分布、离散类别与连续排序信号。只有把接口拆开，才能知道每一步究竟增加了哪一种计算证据。

三个模块输出的语义不能互换：生成概率、分类概率与连续活性估计回答的是三个不同问题。完整的 candidate 与实验边界留到第 9 节收束。

图 1 表达的是 system design / intended computational funnel，不是已经完成并验证的 end-to-end runtime。当前材料足以确认三模块职责，但没有核验一条序列从 Generator 到 Sorter、再到 MIC 的完整对象级传递；本文也不会用下游模块替 Generator 的结果背书。

## 2. AMPgenerator：把蛋白质序列当成一种自回归语言

AMPgenerator 的核心直觉并不复杂：把氨基酸序列当成一种语言。自然语言模型做 next-token prediction，这里做 next-amino-acid prediction。给定序列前缀，模型为下一个氨基酸输出 logits，再由采样策略选择一个 token，如此循环直到 EOS 或达到生成上限。

例如，模型已经看到一段以带正电残基为主的前缀时，并不是从生化规则表里查询“下一位应该是什么”，而是根据训练过程中学到的上下文统计给 25 个词表位置分配分数。采样会把这组分数变成一个具体 token，新 token 又成为下一步上下文的一部分。生成因此是一个逐步展开的过程：早期一次选择会改变后面所有条件分布，错误或偏好也会沿序列累积。这一点后来会解释，为什么 validation loss 与实际自由采样不能被当成同一件事。

$$
P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t})
$$

repaired path 使用单氨基酸 tokenizer，词表大小为 25：前 5 个是基础 special tokens，后 20 个对应 canonical amino acids。EOS 直接使用 `[SEP]`，ID 为 3。它不是 Byte-Level BPE，也没有 BPE merge。字符级 token contract 的好处是明确：只要生成 ID 落在氨基酸区间，就能无歧义地还原为一个 AA 字符。

这里 tokenizer 并不是无关紧要的输入工具，它同时规定了模型能说什么、EOS 怎样表示、checkpoint 的 embedding 有多少行。如果 tokenizer 声称有 57 个 token，而模型 WTE 只有 25 行，那么“词表里存在某个 function token”并不意味着模型能合法读取它。repaired path 有意把语言词表收缩为 checkpoint 真正支持的前 25 项，把 function 移出离散 token 空间。这让 AA ID、model vocab、EOS 和 WTE shape 形成一份可以用断言验证的闭合 contract。

它背后是一个 12-layer、hidden size 768、8-head 的 GPT-2-style protein LM checkpoint。这个架构事实可以从 config 和 checkpoint 验证，但 checkpoint 的原始 pretraining provenance 无法从当前仓库完整恢复。因此更准确的称呼是 **Protein-GPT-style backbone**，而不是把它写死成某个公开 ProteinGPT2，更不能声称本项目已经用某个不可追溯的 60 万序列语料完成了预训练。

旧 checkpoint 的 WTE 曾因 historical wrapper key 不兼容而被标准 loader 重新初始化。repaired compatibility loader 修复映射，并验证恢复后的 25×768 WTE 与 checkpoint tensor 相同；它没有改变架构，只是让 checkpoint 被忠实恢复。

base context 为 50；扣除 11-prefix 与 EOS 后，训练最大 AA 长度为 38，超长序列按该 contract 截断。

## 3. 从无条件生成到条件生成：模型多了什么

如果希望同一个生成器接收功能标签，目标分布就从无条件序列建模变成：

$$
P(x\mid c)=\prod_{t=1}^{T}P(x_t\mid x_{<t},c)
$$

关键不在公式多了一个字母，而在 condition 是否真的进入 tensor graph。repaired generator 把输入组织成四段：10 个 shared soft prompt embeddings、1 个 explicit function embedding、真实 AA embeddings，以及末尾 EOS。17 个 stage2 label 先通过固定 taxonomy 映射为 `func_id`，再进入一个 17×768 的 `nn.Embedding`；function name 不再经过 tokenizer。

**Figure 2 — Repaired AMPgenerator**

> 17 labels → `func_id` → **1 function embedding**  
> **10 shared soft embeddings** ─┐  
> **AA token embeddings** ───────┼→ Frozen GPT-2-style backbone → next-AA logits  
> **`[SEP]` EOS** ───────────────┘

soft prompt 不是一段隐藏文本，也不是 10 个可读的特殊 token。它是直接插入 embedding space 的连续向量，形状为 10×768，在所有 function 之间共享。Function Embedding 同样直接进入 embedding space，但会随 `func_id` 改变，承担显式条件选择器的角色。

“shared”在这里很重要。10 个 soft vectors 为所有样本提供同一组可训练前缀，可以理解为让 frozen backbone 适应 peptide generation 任务的公共接口；它们本身不能告诉模型当前请求的是哪个 function。真正区分 17 个标签的是后面的单个 function embedding。两者分工后，任务适配与条件身份不再混在一串 tokenizer 可能错误解析的文本里。

输入顺序同样体现这种分工：公共 soft prompt 先提供任务级上下文，function embedding 再给出样本级条件，AA embeddings 承载已知序列前缀，EOS 定义合法结束。四部分都进入同一个 frozen backbone，但只有前两组 embedding 更新；function selector 因而不再依赖可读字符串是否被正确分词。

参数账本也因此很清楚：soft prompt 有 7,680 个参数，function embedding 有 13,056 个参数，总计 20,736 个可训练参数；约 85.1M 个 GPT-2 base 参数全部冻结。可训练部分约占 repaired model 的 0.024%。Parameter-efficient 只描述更新范围，不保证生成质量。

## 4. 第一个 Bad Case：字符串里有 Condition，不代表模型真的看到了 Condition

旧设计在字符串层看起来非常合理：输入里有 `<TASK=AMP_GEN>`，也有 `<FUNC=antigram_positive>` 或 `<FUNC=antigram_negative>`。日志里能看到 condition name，生成 CSV 里也能保存 condition 列。问题是，模型不读取人类看到的字符串，它读取 token IDs 和 embeddings。

审计实际 tokenizer path 后，custom prompt strings 被 `BertTokenizer` 拆成重复的 `[UNK]`。生成端又对 prompt 做了截断，function 部分整体消失。结果是 positive 与 negative 的实际 prompt IDs 完全相同。于是出现了一个非常典型、也很容易漏过的工程 bug：配置文件有 condition，Python 字符串有 condition，输出表格也有 condition，但进入模型的 tensor 没有 condition identity。

训练时 task/function 退化为同构的 `[UNK]` pattern；生成时截断又让 function 整段消失，train/inference 没有共享一个可靠的条件接口。

这解释了为什么单看“程序能跑”远远不够。训练 loss 仍然可以下降，checkpoint 仍然可以保存，sampling 仍然会产生不同序列；但这些现象都不能证明条件信号参与了计算。这里最值得保留的一条检查链是：**string-level condition ≠ token-level condition ≠ embedding-level condition ≠ logit-level condition**。字符串里有 condition，不代表 tensor graph 里有 condition。

这种 bug 容易发生，不是因为 prompt 思路本身荒谬，而是因为人类调试时天然偏向可读对象：我们打印字符串、查看 CSV 列、检查配置项是否存在，却很少把两个条件经过完整 preprocessing 后的 ID tensor 放在一起比较。对 conditional model 来说，最小单测应该反过来设计：固定所有其他输入，只切换 condition，逐层确认 ID 或 embedding 不同，并最终确认 logits 不同。任何一层相同，都需要解释，而不是等训练指标来替它辩护。

repair 没有继续修补脆弱的文本 prompt，而是采用 `func_id → explicit embedding → prefix`。正式训练后，我用相同 AA prefix，只切换 positive/negative condition，再比较 condition embeddings 和首步 logits。CE best checkpoint 的 logits 最大绝对差约为 0.03145；CE+SupCon 为 0.10187，两组都通过“不相同”检查。

这些数字的正确解释只有一个：explicit function signal 到达了 embeddings 和 logits，**operational conditioning verified**。它们不衡量条件控制强弱，0.10187 也不能被解释成比 0.03145 有更好的生物功能控制。Operational conditioning ≠ biological functional controllability；当前没有 independent biological function evaluator。

## 5. Generator 到底训练什么：CE、soft prompt 和 SupCon

训练输入依次是 10 个 soft positions、1 个 function position、AA sequence 和 EOS。CE labels 对前 11 个 prefix positions 设为 `-100`，PAD 同样 mask；只有 AA 与 EOS 接受监督。soft prompt 和 function embedding 没有自己的文字 target，但它们会改变后续位置的 hidden states 和 next-AA prediction，所以 CE 梯度仍能沿计算图回传到这两组参数。

正式 run 的首个 optimizer step 专门做了梯度归属检查：frozen base 没有 nonzero gradient，base WTE 更新前后不变；soft prompt 与 function embedding 的 gradient norm 均大于零。这个检查比“trainable params 只有 0.024%”更重要，因为参数列表写对了，不代表 loss 真的连接到了它们。

CE 在这里做的是 teacher-forced next-token learning。训练时，模型预测下一位时总能看到数据中的真实历史前缀，而不是自己上一步采样出的 token。因此一个较低的 CE 表明模型在 validation sequence 的真实上下文上分配了更高概率，却没有直接测试它在自由生成中偏离训练轨迹后能否恢复、能否及时输出 EOS。这个差异不是额外缺陷，而是自回归训练与采样之间必须通过生成实验补上的评价空白。

SupCon 处理的是 representation geometry。repaired implementation 从最后一层 hidden states 中只选择真实 AA positions，排除 soft prompt、function、EOS 和 PAD；然后做 masked mean pooling 与 L2 normalization。同 function label、且不是 anchor 自身的样本才算 positive。没有其他 positive 的 anchor 会被跳过；整个 batch 都没有 valid positive 时返回稳定的 0。

只池化 AA positions 可以排除 prefix 与 padding；去掉 self diagonal，则保证 positive 真的是 batch 中另一个同标签样本。

$$
L_{total}=L_{CE}+\lambda L_{SupCon}
$$

旧 SupCon 曾把 self 算作 positive，并混入 prompt/padding；repair 后才采用上述标准实现。正式 CE+SupCon 组使用 `λ=0.1`，它是否改善生成必须由固定采样回答。

需要强调的是，CE 与 SupCon 并不是两套互相独立的模型。它们共同作用于同一组 soft prompt 和 function embedding；区别只在于梯度中是否多了一项表示空间约束。于是最公平的问题不是“哪个 run 的日志更漂亮”，而是从同一初始化开始，只加入这一项以后，optimization 与固定采样结果分别怎样变化。这正是下一节配对实验的设计目的。

## 6. 一次严格配对实验：CE vs CE+SupCon

为了隔离 SupCon objective，本次只做了一次 single-seed paired experiment，不做超参搜索。CE 与 CE+SupCon 使用同一个 base checkpoint、同一份 data、相同 stratified split、相同初始 soft/function tensors、相同训练顺序、sampling config 与 generation seed；配置唯一差异是 `lambda_supcon`：前者为 0，后者为 0.1。对应 hashes 在 comparison manifest 中全部一致。

公平性不是“配置看起来相似”，而是 initial tensors、split membership、training order 与关键输入 hashes 均通过 comparison assertion。

这样的配对设计让问题保持单一：当两个 run 的优化曲线或生成统计不同，已知实验变量只有 SupCon 权重，而不是数据划分、初始化或采样随机性同时变化。它仍然只是一次 single-seed 短实验，不能给出普遍优劣结论，但足以检验这次 repaired implementation 中 objective 改动是否转化为可观察的生成差异。

stage2 数据包含 18,707 条 sequence、17 个 function labels，类别明显不均衡。seed 42 的 fixed stratified split 得到 16,837 条 train 与 1,870 条 validation，train/validation exact sequence overlap 为 0。但 exact overlap=0 不等于 homology-isolated split：上游 APD3、DRAMP、DBAASP 等来源 provenance 尚未恢复，multi-label 到 single-label 的历史映射规则也未恢复；本次没有 CD-HIT 或 near-homology isolation 证据。

这一数据 caveat 直接限制了实验能回答的问题。它适合比较同一 frozen dataset 上两个 objective 的短期行为，却不适合声称模型对未见同源家族具有严格泛化能力。17 个 labels 也是历史 single-label table 的现状，并不表示天然 peptide 只具有一种功能。实验保持数据不变，是为了完成 objective 对照；保持不变不等于上游标签建模问题已经解决。

训练设置固定为 2 epochs、batch size 8、learning rate 5e-4。CE run 按最低 validation CE 选 best；CE+SupCon 按最低 validation total 选 best，同时单独报告其 validation CE，避免只用包含 SupCon 的 objective 证明自己。

| Metric | CE | CE+SupCon |
|---|---:|---:|
| Best Val CE | 3.761 | 3.489 |
| Best Val Total | 3.761 | 3.684 |

如果故事停在这里，很容易写出“SupCon 更好”。但两组 objective 不完全相同，实验又只有一个 seed；validation 数字下降只能说明这次优化轨迹发生了变化。Best checkpoint 在生成前还进行了重新加载，source 与 loaded trainable tensor hash 相同。真正的问题是：它采样出来的行为有没有一起变好？

## 7. 为什么 Loss 下降还远远不够

每个 best checkpoint 都执行两个 condition、每个 condition 100 次，共 200 attempts。采样固定为 top-k 10、top-p 1.0、temperature 1.0、max new tokens 34。这里 `top_p=1.0` 实际没有执行 nucleus pruning；它只是让所有经 top-k 保留的概率质量继续参与采样。每条 attempt 的 seed 都由 generation seed 4242 与 attempt ID 确定。

生成评价从 raw attempt 开始，而不是从过滤后的成功 CSV 开始。每次解码都必须记录是否遇到 EOS、原始长度、字符是否合法、长度是否合约、最终是否 retained；失败也要保留 rejection reason。这样分母始终是 200，而不是在失败样本消失后，只对少数幸存者报告比例。对生成系统而言，“为什么没有留下来”本身就是模型行为的一部分。

| Metric | CE | CE+SupCon |
|---|---:|---:|
| Attempts | 200 | 200 |
| AA20 valid | 200/200 | 200/200 |
| EOS terminated | 5/200 | 1/200 |
| Length valid | 5/200 | 1/200 |
| Retained | 5/200 | 1/200 |
| Retained unique | 5/5 | 1/1 |
| Exact train match | 0 | 0 |
| Cross-condition exact overlap | 0 | 0 |

AA20-valid 100% 只说明输出 token 被约束在 20 个 canonical amino acids 中，不能改写成“generation validity 100%”。真正的 bottleneck 是 EOS termination：绝大多数 sequence 没有在 generation window 内合法结束，因而 length-valid 与 retained 同步下降。CE 最终保留 5/200，即 2.5%；CE+SupCon 只保留 1/200，即 0.5%。

这个结果也帮助区分两种“正确”。字符正确来自硬词表边界，只要 sampling 不越过 25-token model vocab，AA20 check 很容易通过；终止正确则要求模型学会在合适上下文给 EOS 足够概率，是分布学习的一部分。前者 200/200 不能抵消后者 5/200 或 1/200。若只展示 decoded sequence 而不展示 stop reason，就会把达到 max token window 的截断片段伪装成正常完成的 peptide。

同样，retained sequences 的 uniqueness 是 5/5 和 1/1，但分母太小，强调“100% unique”会遮住更重要的失败率。两组 exact train match 都是 0，也只能说明没有逐字符复制训练序列；exact non-match ≠ homology novelty，更不等于 biological novelty。

所以，本次 paired experiment 的安全结论是：SupCon changed the optimization trajectory，但**没有提供 SupCon 改善 generation quality 的证据**。更低的 validation CE 没有转化为更好的 EOS/retained behavior。Optimization success ≠ generation success，或者更直接地说：验证集 Loss 更低，不等于生成行为更好。

两张表必须成对阅读。训练表告诉我们 CE+SupCon 在这次 run 中到达了更低的 validation CE；生成表则告诉我们，这个变化没有带来更高的合法终止与保留比例。如果只展示第一张表，读者很容易把 optimization advantage 自动延伸为 generation advantage；如果只展示第二张表，又看不到 objective 的确改变了学习轨迹。完整结论恰好存在于两者之间：优化发生变化，但目标生成行为没有随之改善。

这也是保留负结果的意义：它让模型选择依据回到完整评价链，而不是由单个最有利的数字决定，也让后续改进拥有可比较的起点。

为什么会出现这种分离？validation CE 在真实前缀上逐位置平均，而自由采样让模型不断消费自己的输出；EOS 又只是许多候选 token 中的一个。SupCon 约束 pooled representation，也没有直接把“在 8–32 AA 之间终止”写成优化目标。这里不需要事后编造一个确定因果，只需承认测量对象不同：一个衡量 teacher-forced token prediction，一个观察完整 decoding process。正因为两者不同，generation-stage evaluation 才不可省略。

项目也在这里冻结。我没有因为结果不好增加 epoch 3、换 seed、改 λ 或调 top-k。负结果的价值就在于比较 contract 保持不动；如果看到结果再不断调配置，最后得到的就不再是这次严格配对实验。

## 8. 第二个 Bad Case：指标到底在评哪一批序列

**Figure 3 — Legacy QC Provenance Failure**

> final conditional output：454 rows ──╳──> legacy QC  
> stale `generated.csv`：269 rows ───────→ legacy QC  
> 两批 sequence overlap = 0

Generator 的另一个历史问题不在模型，而在 artifact provenance。legacy final conditional output 有 454 行，但旧 QC 实际读取另一份 269 行的 `generated.csv`，两批 sequence overlap 为 0。因此旧 QC 的 novelty、diversity 和 physicochemical 结果不能归属于那 454 条 conditional output。

这不是数量误差，而是评价对象整体换了；问题发生在任何统计公式之前。

这类错误之所以危险，是因为每个文件单独看都“存在”：生成 CSV 有内容，QC JSON 有数字，图也能画出来。只有追问“这个指标的输入究竟是哪一个 immutable artifact”，问题才会暴露。**Metric provenance is part of the evaluation contract.** 指标定义、分母和输入对象必须一起冻结。

换句话说，评价不是在生成结束后随便挑一个文件运行脚本，而是训练 run 的延续。checkpoint 决定分布，sampling config 和 seed 决定 attempts，过滤规则决定 retained set，evaluator 只能读取这条链上被声明的对象。任何一步换成同名旧文件，即使指标公式完全正确，结果也失去归属。Provenance 不是附属元数据，而是评价结论成立的前提。

repaired path 使用独立 run namespace 与 manifest，冻结 config、checkpoint、split、generation 和 evaluation，并强制 generation SHA-256 等于 evaluator input hash。`generation_attempts.jsonl` 保存成功与失败 attempts，因此失败样本不会在评价前消失。文件名不能证明 lineage，run ID 与内容 hash 才能。

这也说明为什么本文不发布旧的“80%+ novelty”、Self-BLEU、prompt consistency 或 physicochemical consistency。旧 novelty proxy 的 similarity threshold 与“多少候选是 novel”本来就是两个不同命题，且结果没有绑定 final conditional run。repaired formal run 只报告 exact train match=0，并立刻保留边界：它不是 80% novelty，不是 homology novelty，也不是生物学新颖性。

同理，任何 retained-only metric 都要同时给出 retained denominator。若 1 条幸存序列没有重复，uniqueness 可以等于 1，但它对 199 条失败 attempts 保持沉默。评价 contract 的任务不是挑一个最好看的比例，而是让 total attempts、过滤规则、成功样本和失败原因在同一个对象链上同时可见。

## 9. 一条序列什么时候才能叫 Candidate

回到开头的 Generate → Filter → Rank，AMPgenerator 的输出只能停在 generation-stage sequence：它说明模型可以在 tokenizer contract 下产生氨基酸序列，并把成功与失败 attempts 都交给可追溯评价。它并没有证明序列属于真实 AMP，也没有证明 positive/negative condition 对应的生物功能已经实现。

在系统设计里，下一步应由 AMPsorter 判断序列是否进入项目关注的 AMP-like 判别空间，再由 MIC predictor 提供计算活性排序。即使两步都给出理想分数，得到的仍然只是 computational candidate，而不是被实验验证的 peptide。当前项目没有新增 peptide synthesis、MIC wet-lab、toxicity assay、animal experiment，也没有一个已经核验的端到端脚本把三模块结果合成为最终候选榜单。

“Candidate”这个词因此需要限定语。generation-stage retained sequence 只通过了字符、终止和长度 contract；AMP-like candidate 还需要判别模型的输出与清晰阈值；activity-ranked computational candidate 则进一步带有 MIC 模型的相对排序。三种对象不能因为都存成 CSV 就混为一谈。只有合成并经过实验测量以后，才可能讨论真实活性；而实验活性、毒性、安全性与药物开发又是更后面的不同问题。

因此，系统图里的箭头目前首先是一份责任划分：它告诉我们下一阶段需要补充什么证据，而不是自动把上一阶段输出升级成下一阶段结论。Generator 的失败样本、Sorter 的判别结果与 MIC 的排序都应保留各自身份，最后才可能组成可追踪的 computational candidate record。

这也是为什么本篇不提前展开 Sorter 的 LoRA、ArcFace，也不展开 MIC 分支的 ESM-2、图网络或回归指标。那些技术回答的是后续模块如何建模，无法倒推 Generator 的序列已经是 AMP。下一篇可以分别检查分类与回归 artifact，但仍必须沿用相同原则：模型输出叫什么，取决于它真实接受的监督与验证，而不是系统愿景给它起了什么名字。

这次审计和 repair 最后沉淀出五条比“模型名”更重要的经验：

1. 先定义 discovery funnel，明确 Generate、Filter、Rank 分别减少哪一种不确定性。
2. Tokenizer 与 condition 必须检查 tensor identity；不能只检查 prompt 字符串。
3. Parameter-efficient tuning 的关键不是 trainable params 少，而是梯度到底流向哪里。
4. Lower loss 不会自动带来 better generation，checkpoint 必须经过固定 sampling contract 检查。
5. Generation metrics 必须绑定同一个 immutable run；computational sequence 也必须与 biological validation 保持边界。

所以 Article 03 最终不是一个“我们取得了优秀 AMP generation results”的故事。更准确的复盘是：我构建并重新审计了一条 Protein-LM peptide generation path，修正 condition identity、checkpoint loading、SupCon 与 artifact provenance，然后在一次严格配对实验中发现，更低的 validation objective 没有转化为更好的 generation behavior。

下一篇再进入 AMPsorter 与 MIC predictor：生成之后，如何做判别与活性排序，以及那些计算结果各自能支持什么、又不能支持什么。无论后续模型多复杂，边界都不应变化：**Generated sequence ≠ experimentally validated AMP.**

最终，Generate → Filter → Rank 不是一条把不确定性自动变成事实的流水线，而是一组逐层提出更具体计算问题的工具。每一层都需要自己的数据、对照、artifact 和失败样本；任何一层的结果都不能越过尚未完成的验证。对这个项目而言，能够清楚说出“现在只到这里”，不是削弱故事，而是让后续每一步真正有可检验的起点。
