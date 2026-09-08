---
layout: post
title: "从轨迹监督到 Rollout 优化：Medical Deep Search Agent 的 Agentic GRPO"
date: 2026-09-08
lang: zh
published: true
math: true
categories:
  - AI Agents
  - LLM Post-Training
tags:
  - Agentic RL
  - GRPO
  - Controlled Failure
  - Fatal-aware GRPO
  - Credit Assignment
description: "从 Agentic SFT 进入 rollout optimization，拆解 Medical Deep Search Agent 中 G4 GRPO、policy-token credit、Controlled Failure、Fatal-aware masking 与严格 E2F/E3 因果评估的真实训练与证据边界。"
---

> 从 Vanilla GRPO、Controlled Failure 到 Fatal-aware Credit Assignment

上一篇写到 Agentic SFT 时，我把重点放在一个很底层的边界上：Observation 是环境给出的上下文，Action 才是模型应当学习的输出。E1C 为 policy 学习 canonical Tool Call 提供了 supervised initialization，使后续 rollout 能进入严格 parser 和工具环境。但 masked cross-entropy 仍然是在回答：“示范者处于这个 State 时，会做什么？”

它没有回答另一个更接近策略优化的问题：同一个 prompt 采样出几条不同的工具链，哪一条 outcome 更值得强化？如果一次搜索迅速获得了可引用证据，另一次反复调用后仍然失败，固定示范只能告诉模型目标动作是什么，却不能直接比较模型自己走出来的路径。

因此，本文的起点不是把 SFT 推翻，而是把训练对象从 demonstrated action 扩展到 sampled trajectory：先让模型在环境中行动，再根据完整结果计算 reward，最后把相对信号送回真正属于 policy 的 token。这条链路最终也暴露了一个更难的问题——当长轨迹在中途进入连续失败，整条 rollout 的同一个 reward，是否应该同样惩罚此前仍有价值的动作？

## 1. SFT 已经会调用工具，为什么还要做 Agentic RL？

Agentic SFT 本质上仍是 behavior cloning。把当前可见状态记为 `s_t`，工具动作或 Finish 记为 `a_t`，它提高的是示范动作的条件概率 `πθ(a_t | s_t)`。这为可执行动作协议提供 supervised initialization；但它不会比较多个 sampled rollout、直接优化完整 trajectory outcome，或自动获得 demonstration 之外的失败恢复经验。

Agentic RL 处理的是下一层问题：让当前 policy 对同一问题采样多条轨迹，在共同环境里得到结果，再比较哪些动作序列相对更值得增加概率。

**图 1｜从轨迹监督到 rollout optimization**

```text
Agentic SFT
State -> demonstrated Action -> masked CE
             |
             v
可执行的 Tool Call / Finish grammar
             |
             v
Agentic GRPO
同一 Prompt -> 4 条 sampled trajectories -> outcome comparison -> policy update
```

这也是为什么本项目把正式 group size 冻结为 `G=4`。每组不是四个离线答案，而是同一 prompt 下四次真实 policy rollout：模型生成动作，环境执行工具，Observation 改变下一步 State，直到 Finish、policy error 或预算终止。RL 优化的是这条交互链产生的相对结果，而不是重新做一次答案级 SFT。

示范也许只展示“先查药品标签，再引用证据结束”，模型采样时却可能先查错 source、失败后恢复，或在没有证据时 Finish。这些状态未必出现在 SFT corpus，却会在 runtime 中真实出现；示范路径上的交叉熵无法比较这些自生成结果。

Grouped rollout 提供的不是绝对真理，而是局部比较。四条轨迹共享 prompt 和评分契约，reward 把它们放进同一个相对坐标系；GRPO 再提高高于组均值的动作概率，压低低于组均值的动作概率。这样可以绕开单独训练 critic 的成本，也把比较限制在更可控的同题样本内。不过，若四条轨迹拿到相同 reward，组内就没有方向；如果 reward 只衡量格式，策略学到的也只会是格式。这两点后来分别变成 E2C signal sparsity 和 Reward v2.1 边界。

因此，Agentic SFT 先初始化动作协议，Agentic RL 再研究动作连成轨迹后哪些相对结果值得强化。前者不是 held-out causal capability proof，后者也不是自动成立的能力提升。

但“能做 rollout update”不等于“能力已经提升”。后文的 E2C 能证明正式训练闭环完成，却不能单独提供 causal capability evidence。这个区分贯穿全文：工程闭环、机制暴露和行为结论必须分别取证。

## 2. 把 GRPO 接进真实 Agent Runtime

教科书式的一行 loss 无法表达 Agent runtime 的麻烦：四条 trajectory 可能包含不同数量的决策、工具调用和 Observation，也可能在不同位置终止。本项目实现的 GRPO contract 先约束数据闭环，再允许 optimizer 工作。

**图 2｜本项目的 G4 GRPO runtime loop**

```text
Prompt + State
  -> G=4 fresh trajectories
  -> Tool Environment
  -> trajectory outcomes
  -> Reward v2.1
  -> group-relative advantages
  -> frozen old logprobs
  -> current new logprobs
  -> PPO-style clipped update
```

一个训练 group 只有在四个 slot 都得到 valid policy outcome、finite reward 和 exact capture 时才 eligible。任一成员无效，整组拒绝；系统不会悄悄降成 G3，也不会给 invalid trajectory 填一个 reward 0。后两种做法都会改变组均值和方差，让本不存在的比较信号进入梯度。

“完整”不只是文件里有四行：trajectory、capture、工具执行、outcome 与 reward 必须对齐。Infrastructure-invalid 没有可用 reward；policy 格式错误则是可计零分的模型行为。混淆两者会让基础设施故障伪装成负样本。

Complete-group 是 estimator 的语义，不只是数据清洗规则。少掉一条后，剩余三条 reward 的均值和方差都会变化；补一个零又会人为创造低分样本。Fail closed 虽然减少可更新 group，却保证每个 advantage 确实来自预先冻结的 G4 比较。

第一条核心公式只描述组内相对位置：

\[
\begin{aligned}
\mu &= \frac{1}{G}\sum_i r_i, \\
\sigma^2 &= \frac{1}{G}\sum_i(r_i-\mu)^2, \\
A_i &= \frac{r_i-\mu}{\sqrt{\sigma^2}+\varepsilon}, \qquad G=4.
\end{aligned}
\]

这里使用 population variance，也就是分母为 4。若 `variance <= 1e-12`，四个 advantage 精确设为 0，整个 optimizer batch 跳过更新，不加入人为 jitter。这让“没有组内学习信号”本身成为可报告结果，而不是用噪声制造一次看似成功的 step。

Zero-advantage group 是合法但不更新的数据；缺 capture、身份漂移或非有限 reward 则不可训练。Old-policy 的严格 update boundary 留到第 4 节展开。

该实现没有 value critic；正式 E2C、E2F、E3 的 `KL beta=0`，也没有常驻 reference model。它采用的是 group-relative trajectory advantage、token-level clipped ratio 和 trajectory-balanced aggregation。这个描述比“标准 GRPO 就是这样”更准确，因为每项都是当前工程契约，而不是对所有 GRPO 实现的概括。

## 3. Reward 是 trajectory-level，但哪些 Token 真正被更新？

一条 rollout 最终只有一个 scalar reward，但它经过的是多轮 `Action -> Observation`。如果把整段对话当成无差别的 token 序列，搜索结果、工具错误甚至状态快照都会收到策略梯度，模型就会再次被训练去“制造环境”。所以 RL 必须延续上一篇的 token ownership。

**图 3｜RL token ownership**

| Segment | 参与 attention | 接受 policy update |
|---|---:|---:|
| System prompt | 是 | 否 |
| User question | 是 | 否 |
| Policy state | 是 | 否 |
| Tool Observation / error | 是 | 否 |
| Assistant Tool Action | 是 | 是 |
| Assistant Finish / final response | 是 | 是 |

System、Question、State 和历史 Observation 全部位于 teacher-forced prompt prefix，mask 为 0；实际生成的 Tool Action 与 Finish IDs 使用 policy mask。换句话说，Observation 会改变下一次决策的条件，却不承担 policy gradient。Trajectory advantage 只广播到 policy-generated tokens。

设想一个工具返回 retryable timeout。下一轮 prompt 会包含这条失败 observation、剩余预算和已经尝试过的 query；它们当然需要参与 attention，因为模型必须据此选择重试、改写或结束。但这些 token 的作者是 Environment，不是 Policy。若让它们进入 loss，模型得到的训练任务会变成“预测 timeout 文本本身”，甚至可能学习复述一个它无权宣布的工具结果。Ownership mask 把“读懂世界”和“生成动作”分开了。

第二条公式体现了更新粒度：

\[
\begin{aligned}
\rho_{i,t} &= \exp\!\left(\log\pi_{\mathrm{new}}(y_{i,t})-\log\pi_{\mathrm{old}}(y_{i,t})\right), \\
o_{i,t} &= \min\!\left(\rho_{i,t}A_i,\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\right), \\
L_i &= -\frac{\sum_t M_{i,t}o_{i,t}}{\sum_t M_{i,t}}, \qquad
L=\frac{1}{N}\sum_i L_i.
\end{aligned}
\]

实现里有两次平均。先在每条 trajectory 内，对跨 turns 的有效 policy tokens 求均值；再让不同 trajectories 等权。它没有把整个 batch 的所有 token 做 global mean，否则较长轨迹仅仅因为生成得多，就会在 update 中占更高权重。Advantage 是 trajectory-level，ratio 与 clipping 是 token-level，这两层语义不能混为一谈。

这个选择对 Agent 尤其重要。一次快速成功可能只有一个 Tool Action 和一个 Finish，失败轨迹却可能包含四五轮重试；global-token mean 会天然放大后者，不论它的 advantage 大小是否更有信息。Trajectory-balanced aggregation 先把每条轨迹压回一个等权样本，再做组内比较，让“生成长度”不再暗中充当额外权重。

它也意味着 scalar reward 并没有被粗暴地贴到所有输入 token 上。每个有效 policy token共享同一条 trajectory advantage，却拥有各自的 old/new ratio；clipping 在 token 级限制更新幅度。后面的 Fatal-aware 机制正是在这个 mask 层继续做文章：组级 reward 和 advantage 保持不变，只改变其中一部分 policy tokens 是否有资格承接 credit。

## 4. Old Logprob 为什么必须绑定真实 Rollout

Agent rollout 中最容易被低估的工程问题，是“训练的 token”和“当时生成的 token”是否真是同一批。Decoded text 并不是充分证据：decode 可能跳过 special token，再次 tokenize 也未必恢复原序列；temperature、top-p 参与采样，但 warped sampling score 也不是目标模型的 raw policy logprob。

本项目在 `processor -> model.generate()` 的边界直接保存 actual generated token IDs，并把 prompt、adapter、tokenizer/processor、formatter 与 runtime identity 绑定到 policy turn。重点不是字段清单，而是让每次决策都能回到唯一的生成现场。

Old logprob 在任何 optimizer step 之前，由同一个 frozen adapter snapshot 对 `prompt IDs + generated IDs` 做 teacher-forced causal recomputation。第一个生成 token 对应 prompt 最后位置的 logits，后续 token 对应前一个 causal position。全部 old values 完成物化后 detach；当前模型再对相同 IDs fresh forward，得到 new logprob。

多轮 Agent 使这种绑定比普通单轮文本更敏感。Decision 0 的 prompt 只有初始问题和状态；Decision 1 已经包含第一次工具 Observation；Decision 2 又多了新的 evidence 或 error。即使三次生成文本碰巧相同，它们的条件分布也不同。因此 capture 的基本单位是 policy turn，而不是整条最终字符串。每个 turn 都要固定自己的 prompt identity、decision index 和 generated IDs。

Processor identity 也不能省略。即使 formal RL 是 text-first，processor config、attention mask 与可重建材料仍须固定，避免 update 阶段重新运行一套已经漂移的预处理逻辑。

这个 update boundary 特意不提供“算完一组就顺手更新”的入口。否则后半批所谓 old policy 已经混入前半批更新，不再是同一个 snapshot。Capture hash 则把 generated IDs、policy mask 和 processor inputs 等身份一起锁住，错位或篡改直接 fail closed。

所以 capture 不只是审计日志，也是 objective 的数值输入。它把“当时模型生成了什么”与“更新时要在哪些位置重算概率”合并到同一个不可变对象；只要 token 数、mask 长度或 adapter identity 不一致，ratio 就没有合法定义。

从资源角度看，这个设计也适合本地单卡：不需要同时常驻第二份 reference model。旧策略 logprob 在参数更新前统一重算，随后 detach 并可移到 CPU；GPU 再负责当前模型的 fresh forward、backward 与 optimizer。它牺牲了一些重算时间，换来更小的显存压力和清楚的旧/新策略边界。

仓库没有证据支持把 `generated_text_sha256` 包装成历史事故，因此本文只把它当成 provenance guard。

## 5. E2C：一张 RTX 4060 上的 Formal Vanilla GRPO

E2C 是这套闭环第一次进入 formal Vanilla GRPO training。它从 E1C final adapter `bee3dc...` 开始，在单张 RTX 4060 Laptop GPU 上处理 24 个 TRAIN prompts；每个 prompt 采样 G4，共 96 条 trajectories。训练继续使用 Qwen3-VL-2B 的 4-bit NF4 base、BF16 compute 和 LoRA-only trainable surface，microbatch 为 1，并用 trajectory streaming 控制单次可微计算图。

“LoRA-only continuation”在这里有两层含义。Base model 保持量化冻结，E1C adapter 以可训练状态加载，真正更新的是同一组 196 个 LoRA target modules、约 871 万参数；输出仍是一个新的 adapter，而不是重写完整 2B 权重。Rollout collection 与 update 串行进行，old logprob 物化后可以释放不再需要的图和张量。这个组合把一次真实的多轮 GRPO transaction 压进了消费级单卡，而不是把关键步骤替换成 CPU mock。

Microbatch 1 也不意味着把 G4 缩成四次彼此无关的训练。四条轨迹仍共享同一组 reward normalization 和 advantage；streaming 只是逐条构建并释放计算图，再按 trajectory 等权累积 loss。它改变显存占用方式，不改变 formal group 的统计单位。

**图 4｜E2C 的实际学习信号**

```text
24 formal G4 groups / 96 trajectories
├─ 8 nondegenerate groups -> 16 optimizer steps
└─ 16 exact-zero-advantage groups -> skip optimizer
                                  └─ groups 19–23 连续为零
```

正式 runtime 为 11,016 秒，约 3 小时 04 分；最终 adapter 是 `2049b9...`。这些数字证明 E2C 不是 scaffold，也不是一组 smoke，而是具有冻结 config、完整 rollout、optimizer transaction 和 output checkpoint 的 formal training closure。

每个 nondegenerate group 执行两个 optimizer epochs，因此 8 个有信号 groups 对应 16 actual steps；maximum 48 只是预算上限。E2C 没有另行声明可借用的 SFT sequence budget，计划上限也不能冒充实际训练规模。

它同时给出了一个不那么好看的事实：16/24 groups 没有组内 reward variance，真正发生更新的只有 8 组、16 steps。我们可以把它称为 within-group learning signal sparse，却不能据此判断 policy 已经 convergence，也不能反过来断言 collapse。现有 artifact 只列出行为趋同、reward saturation、reward resolution 不足等候选解释，没有识别唯一原因。

这正是 formal closure 与 capability evidence 的区别。E2C 证明后端、capture、advantage、logprob、backward 和 checkpoint 链路真实运行；它没有形成同父、单 treatment 的行为对照，因此不能单独支持“Vanilla GRPO 提升了 Agent”。

我更愿意把 E2C 称为“正式工程与数学闭环”。它回答了这套 objective 是否能在真实 Agent trajectory、真实工具状态和真实 LoRA optimizer 上运行，也暴露了 reward resolution 的实际瓶颈。它没有回答 checkpoint 是否在用户关心的行为上优于 parent；那个问题需要后面同父、单 treatment、held-out evaluation 的设计。

## 6. Reward v2.1：优化了什么，又没有优化什么？

GRPO 能否学到差异，首先受 reward resolution 限制。Reward v1 的一次 24-slot probe 出现 6/6 degenerate groups：四条轨迹虽然在行为上可能不同，标量结果却无法拉开组内位置。项目随后把 reward 升级到 v2.1，并在正式训练前增加 signal preflight，而不是直接把 optimizer 跑起来。

Reward v2.1 的外层很克制：format gate 通过后，`0.45 × evidence quality + 0.45 × process quality + 0.10 × tool efficiency`，最后限制到 `[0,1]`。Evidence 关注 source family、与隐藏 reference 的 graded proximity 以及 citation；process 关注合法 Finish、非重复调用和 productive-tool ratio；efficiency 在证据覆盖后惩罚多余调用。Parser、action 或 arguments 不合法时，format gate 把 total 置零；infrastructure-invalid 则不参与训练。

隐藏 reference 只服务于 judge/reward，不进入 policy prompt。Graded proximity 与 citation 检查帮助 G4 拉开差异，却仍是冻结的 evidence proxy，不是医学语义判断。

这里最重要的不是权重，而是两个显式的零：`medical correctness = 0`，`safety = 0`。没有 validated medical judge 接进 formal loop，因此 reward 不是医学正确性，更不是临床安全性。一个 JSON-valid Finish 可能仍然没有充分回答任务；一个 citation-valid 结果也不保证每个医学 claim 都被专家认可。

Proxy risk 也由此出现。模型可能过早 Finish，以较短路径取得 process 分；可能重复搜索，希望碰到更接近 reference 的证据；词汇接近也不等于语义充分。仓库确实观察到了 premature finish 和 repeated calls，但没有证据证明 policy 在有意识地“reward hacking”。更准确的说法是：v2.1 让组内信号可用，同时保留了 stopping calibration 和 evidence proxy 的盲点。

Reward breakdown 因而比 total 更重要。两个 trajectory 都拿到 0.6，可能一个证据覆盖较好但调用偏多，另一个正常 Finish 却 citation 较弱；只记录 total 会抹掉训练信号来自哪里。项目把 evidence、process、efficiency 分项持久化，也让 Phase 8 能把 parser、Finish、citation、productive tool 与重复调用拆开报告，而不是用一个高 reward 代替全部行为解释。

这对失败分析尤其有用：format gate 解释动作语法是否存活，process 分量显示是否完成或重复调用，evidence 分量说明检索结果是否接近冻结目标。它们仍不能替代医学判断，却能避免把所有退化都笼统归因于一个 total reward。

这也解释了为什么 recovery 和 Fatal 没有被塞进一个“安全奖励”。V2.1 没有独立的 recovery bonus 或 Fatal penalty；恢复是否发生，要从失败 exposure 之后是否出现 later successful tool result 来评估。把它事后强行解释成 reward 的医疗安全含义，会超出 objective 实际编码的内容。

## 7. 96 条轨迹零失败：为什么“没有坏事发生”反而是问题

E2C 完成后，我原本已经有了 96 条 natural rollouts。Fatal-aware 机制似乎可以直接接上去，但 exposure audit 给出的结果是：126 次工具调用全部成功，tool failure 为 0，Fatal 为 0。

**没有 failure exposure，就没有 recovery evidence。**这组零不能证明 runtime 很鲁棒，只能说明当前 deterministic local tools 和任务分布太温和：Fatal mask 没有可作用 token，recovery 也没有被真实触发。一个稀有事件机制如果从未见到事件，代码通过单元测试仍然不等于训练得到验证。

**图 5｜从自然环境转向 Controlled Failure**

```text
Natural E2C
96 trajectories -> 126 successful calls -> 0 failure -> 0 Fatal
                                             |
                                             v
                          Fatal-aware objective 无 exposure
                                             |
                                             v
Controlled-Failure V2
冻结 schedule -> 真实 ToolAction -> backend 前注入 retryable timeout
```

Controlled-Failure V2 因此被设计成 deterministic、frozen、arm-independent 的机制 stressor。Schedule 只读取 group、sample 和 logical tool-call ordinal；policy 必须先发出真实 ToolAction，runtime 才会在 backend 执行前短路，写入一个 policy-visible、`retryable=true` 的 timeout observation。Backend 没有真的调用，也不会用内部 retry 次数冒充多个 policy decision。

V2 冻结了 control、immediate-fatal-opportunity、prefix-fatal-opportunity 和 isolated-recoverable-failure 四类 profile。E2F/E3 共享的 recovery instruction 要求：最新 observation 为 retryable 且仍有预算时，不应只因一次失败立刻 Finish，而应重试或调整 query/source。

这是 RL rollout runtime 中 E2F/E3 共享的 instruction，不是 E1R Recovery SFT curriculum 的 supervision claim。E1R 正式监督只覆盖 exact same-tool/same-arguments retry 与 limitation/stop，没有 query refinement、argument repair、rerouting 或 cross-tool switching 的正式监督。

这个 instruction 并不强迫 policy 产生下一次 action。模型仍可能忽略规则、输出 malformed JSON 或结束；schedule 也不会替它补齐尚未走到的失败。Controlled runtime 只控制“到达某个调用 ordinal 时环境返回什么”，不控制模型能否活到那里。正因为如此，realized exposure 本身就是需要记录的中间变量。

另一个关键区分是 `opportunity != realized exposure`。相同 schedule 为两臂安排相同注入机会，却不保证完全相同的实际失败：某个 policy 可能在目标 ordinal 前已经 Finish，或者先发生 parser failure。未到达的 fault 只能记为 `scheduled_but_unreached`，不能当成已经接受 treatment。早期 full exposure audit 中 84 个 scheduled injections 只实现了 52 个，这个坏案例促使后续 artifact 分开记录 opportunity、exposure、recovery 与 Fatal。

Controlled Failure 让失败成为可重复变量，却没有把它变成真实世界。它不描述线上 timeout 的自然频率、相关性和持续时间，也不覆盖生产服务的全部故障类型。这里得到的是机制定向 stress evidence，不是 production failure distribution。

V2 还在正式双臂结果出现前冻结，修正的是早期 V1 暴露机会过晚、policy 难以到达的问题，而不是根据 E2F 或 E3 的 reward 调参。这个时间顺序很重要：如果看到某个 arm 表现不好后再移动 injection point，后面的差异就同时混入 treatment 和协议选择，严格比较将不再成立。

## 8. Fatal-aware Credit Assignment：保留前缀，切断失败后缀

本项目里的 Fatal 不是普通 exception 上的布尔 flag。Runtime 先重建 `policy decision -> ToolCallRecord -> final ToolObservation` 关系：连续三个不同 policy decision 对应的、policy-visible、可计数失败 observations，才确认一条 Fatal。成功 observation 会重置连续计数；backend 内部 attempts、infrastructure failure 和 parser error 都不能混进来。

“Policy-visible”是定义中的关键。一次 ToolAction 即使 backend 内部重试三遍，模型最终只看到一个失败 observation，它在 credit assignment 中仍只算一次；反过来，模型明确发出三次独立 ToolAction、每次都收到最终失败，才形成三连失败。这让 Fatal 贴合 policy 实际经历的状态变化，而不是底层 transport 日志的重试次数。

**图 6｜Fatal confirmation 与 mask start 是两个坐标**

```text
有效 policy prefix
  -> failure #1  [fatal_start / mask 从这里开始]
  -> failure #2
  -> failure #3  [Fatal confirmed]
  -> later ToolAction / Finish

时间方向 ---------------------------------------->
保留梯度：prefix | 屏蔽梯度：failure #1 及其全部 policy suffix
```

第三次失败是 detection time，第一次失败是 retrospective credit boundary。二者如果混为一谈，就会保留前两次已经属于确认失败链的 action。E3 使用的第三条核心公式非常简单：

\[
M'_{i,t}=
\begin{cases}
M_{i,t}, & \text{non-Fatal}, \\
M_{i,t}\,\mathbf{1}\!\left[\operatorname{decision}(t)<\operatorname{fatal\_start}_i\right], & \text{Fatal}.
\end{cases}
\]

完整 rollout 仍然先计算 Reward v2.1，再在完整 G4 上计算 vanilla advantage。E3 不修改 reward，不为 prefix 重算一个局部 reward，也不把负 advantage 自动改成零；它只改变这个 advantage 能流向哪些 policy tokens。Observation 原本就没有梯度，真正被切断的是第一次失败 ToolAction 及之后的 Tool Action、Finish 或 final response。

为什么从第一次失败开始，而不是第三次确认后才 mask？因为确认只能在第三个 observation 到达时发生，但回头看，这三个 action 已经构成同一条连续失败链。如果只屏蔽第三次之后的 token，前两次失败 action 仍会分享整条异常 trajectory 的 credit。`fatal_start` 与 `fatal_confirmation` 分开保存，让 detection 和 attribution 使用不同坐标，同时保持规则可重放。

这并不宣称第一次失败之前的 prefix 一定正确。它只是避免一个 negative trajectory-level advantage 无差别惩罚所有早期动作；prefix 是否真的有用，仍受原 reward、组内相对位置和数据覆盖限制。若一条轨迹在 fatal-start 之前没有可保留 token，它可以对 objective 贡献为零，但 group 身份和原始 advantage 仍保持完整，不能删除该样本后把 G4 改成 G3。

这也给出了全文的核心句：**Reward 属于整条 rollout 的结果，但 credit 不必无差别地分给整条 rollout 的每个 policy token；E3 保留 Fatal 前缀的学习信号，并切断已确认失败链起点及其后缀的梯度归因。**

Fatal mask 依然不是 runtime halt。环境不会在第三次失败后强制 STOP，目标中也没有正向的停止标签。它做的是 failure-gradient hygiene：避免整条异常后缀继续参与更新，而不是直接训练一个安全终止策略。

同样不能把 E3 与 one-sided clamp 混写。E3 的 effective advantage 始终等于 vanilla advantage；它只 mask token。Clamp 是后来 E4 候选机制，要求 Fatal、存在可保留前缀且 vanilla advantage 为负这几个条件同时满足。本项目没有足够 exposure 把它提升成正式 arm，因此，本文已经实际验证的算法核心到 Fatal mask 为止。

## 9. 唯一严格因果对：E2F vs E3

Checkpoint lineage 在这里必须画对：E2C 从 E1C `bee3dc...` 开始，是 contextual branch；E2F 和 E3 则分别从 E1R `fb4613...` fresh load。只有 E2F/E3 被正式 plan 冻结为 strict causal pair。

**图 7｜E2F 与 E3 的 treatment contract**

| Field | E2F | E3 |
|---|---|---|
| Parent | E1R `fb4613...` | 同一 E1R |
| Tasks / order / G4 seeds | 12 groups，共享冻结身份 | 相同 |
| Failure protocol / recovery prompt | Controlled-Failure V2 | 相同 |
| Runtime / toolset / corpus / reward | 冻结 | 相同 |
| Sampling / optimizer / max budget | 冻结 | 相同 |
| Policy mask | Vanilla mask | Fatal suffix mask |

两臂各自重新加载 parent、创建 optimizer、采样 fresh rollout，不共享更新后的状态。唯一预注册 treatment 是 policy mask：E2F 保留原始 mask，E3 在 confirmed Fatal trajectory 上屏蔽 fatal-start 及其后缀；两臂都使用原始 G4 advantage，也都没有 one-sided clamp。

共享条件不止是超参数表面相同。两臂使用同一组 12 个 TRAIN candidates、相同顺序与 G4 root seeds，同一个 failure schedule、recovery prompt、system context、toolset、corpus、Reward v2.1、sampling 和 policy-state representation。学习率、weight decay、clip epsilon、KL beta、每个有信号 group 的 epochs、microbatch 和最大 24-step budget 也一致。这些身份被 plan 和 receipt 绑定，而不是靠运行结束后人工对表。

独立 fresh start 则防止另一种污染：不能先训练 E2F，再把它的 optimizer 或更新后 checkpoint 交给 E3；也不能让两个 arm 共享已经生成的 rollout。二者面对相同的注入规则，但各自的 policy 会决定是否到达注入点、生成多长的轨迹和哪些 group 产生 variance。严格因果设计冻结的是 intervention 之前的条件，不是强迫 intervention 之后的行为完全相同。

训练结果可以随着 treatment 路径分化。E2F 完成 48 trajectories、11 个 nondegenerate groups、22 steps，观察到 7 条 Fatal，masked token 为 0；E3 同样完成 48 trajectories，但为 12 个 nondegenerate groups、24 steps、15 条 Fatal，并实际 mask 6,165 个 policy tokens。这些是 downstream outcomes，不是额外 hyperparameter differences；两臂也绝不是逐 token 生成了相同 trajectories。

这里还有一次值得保留的坏案例。E2F V001 在 group 4 遇到合法的 `step_budget_exhausted` terminal shape：policy 已生成末尾 ToolAction，但预算耗尽使环境没有执行 call/observation，旧 resolver 却要求严格 1:1 关系。已经提交的 4 groups、16 rollouts、6 steps 被保留为 immutable partial-invalid，而不是并入 final E2F。修复只允许这一个合法 unmatched action，且不把它计为 success、failure 或 Fatal；修复后，final E2F V002 与 E3 都从 E1R fresh restart。

Phase 8 先看 STANDARD TEST。E2F 与 E3 在 27 个 paired tasks 上都是 Finish 27/27、parser-valid 27/27，mean tool depth 相同；E3 diagnostic reward 略低，两臂 premature finish 都是 25/27。因此最强结论只是“没有观察到明显 structural regression”，不是 improvement、equivalence 或 non-inferiority。

这组三个“不支持”并非措辞洁癖。Improvement 需要方向明确的目标和证据；equivalence 需要等效区间；non-inferiority 需要预注册 margin。当前 STANDARD 只有冻结样本上的描述性配对结果。若把若干相同计数写成“两个模型无差异”，就把有限观测扩大成了统计结论，也会掩盖两臂同样很高的 premature-finish 问题。

再看 Controlled-Failure TEST，差异才出现在预注册机制所针对的条件里。

**图 8｜同一 strict pair 在两种评估条件下**

| TEST，N=27 | E2F | E3 |
|---|---:|---:|
| STANDARD Finish | 27/27 | 27/27 |
| STANDARD parser valid | 27/27 | 27/27 |
| STANDARD premature finish | 25/27 | 25/27 |
| Stress Finish | 8/27 | 27/27 |
| Stress parser/policy failure | 19/27 | 0/27 |
| Stress successful recovery | 1/27 | 13/27 |
| Stress Reward v2.1 | 0.2078 | 0.6001 |

最后一行只是一项 diagnostic proxy，不是 medical outcome。严格允许的归因是：在冻结的、text-first、注入式 failure protocol 下，由 Fatal-mask training 得到的 E3 checkpoint 相对 E2F 表现出更强的 failure-conditioned structural resilience。它不能被扩大成“E3 整体更好”。

Stress 表中的 recovery 也要看清 denominator。`1/27 -> 13/27` 是全部 paired TEST tasks 上的 successful-recovery event；按实际 exposure 条件化则是 E2F 1/11、E3 13/15。Exposure 数不同是 policy-dependent reachability 的结果，不能用同一个虚构分母重算。这里使用 all-task paired count 作为主比较，conditional rate 只作描述。

Phase 8 的 evaluation-time runtime 没有再应用训练 mask。E2F 与 E3 接受相同任务、seed 和 schedule，checkpoint 是唯一模型变量。因此结果描述的是两种训练 intervention 最终产生的 policy 行为，不是评估器在 E3 路径上额外屏蔽了输出。这个区分让“训练时 credit assignment”与“运行时行为测量”保持了干净边界。

## 10. 更强恢复，也更容易过度坚持：Agentic RL 的真实边界

Controlled-Failure TEST 的正结果必须和负面副作用放在同一张账上。E3 的 repeat-call 从 E2F 的 3/27 上升到 14/27；它走得更深，也更容易过度坚持。E3 观察到 9/27 Fatal，而且 9/9 在确认后继续行动。这个方向并不支持“Fatal masking 减少 Fatal”，更不支持“模型学会 Fatal STOP”。E2F 的 Fatal 为零，很大程度上是因为它经常在第三次失败前就 parser collapse，不能解释为更安全。

Post-Fatal continuation 也不能被简单贴成 unsafe。9 条继续轨迹里有 7 条后来取得成功工具结果，另外 2 条继续到 Finish。它说明恢复与停止之间存在尚未解决的 calibration：继续可能带来恢复，也可能形成重复调用、证据不足时过早 Finish，或没有必要的持久化行为。Fatal mask 没有训练这个取舍的答案。

从目标机制看，这个副作用并不意外。Masking 只移除某些 suffix token 的梯度，没有提供“第三次失败后应该生成什么”的正样本，也没有单独奖励停止。模型更能维持合法动作语法、活过注入失败，就更可能走到第三次失败并继续；因此 E3 的 Fatal 观察数高于 E2F，既不能直接解释为更危险，也不能宣称 masking 降低了 Fatal。它首先反映 survival、exposure depth 和 stopping behavior 的共同变化。

E4 本来希望在 E3 mask 之外加入 one-sided advantage clamp，避免负的 trajectory advantage 惩罚保留下来的前缀。但 Gate B 只观察到 2 个 eligible clamp-actionable cases，低于预注册 minimum 3。项目没有换 seed 追第三个样本，E4 因而没有 formal training、checkpoint 或 evaluation。这个 negative gate 证明的是研究流程守住了边界，不是 clamp 有效或无效。

最终 Phase 8 完成 14/14 jobs、392/392 evaluation items 和 1,003 captures。392 是五个 checkpoints 的 STANDARD validation/test 加上 E2F/E3 stress validation/test 的总 evaluation items；它不是每个 checkpoint 的样本量，也不包含 training rollouts。本文主结果的 denominator 始终是 paired TEST `N=27`。

其中 STANDARD 为 280 items，stress 为 112 items。Closure 将 frozen plan、14 个 run manifests、392 个 item manifests 与 aggregate 绑定，证明证据可恢复、无需重跑且结论未变；它不新增实验或统计含义。

到这里，能被证据支持的范围很清楚：项目完成了 formal GRPO engineering closure，冻结并执行了 Controlled-Failure V2，建立了同父 E2F/E3 的单 treatment 比较，并在该协议下观察到更强的 failure-conditioned structural resilience。它不能证明一般 Agent 能力、医学正确性、clinical safety、production outage robustness、确定性的 Fatal STOP，也不能证明 E4 clamp effectiveness。所有正式 RL 和 Phase 8 证据仍是 text-first，而不是 multimodal GRPO 结论。

这套边界反过来也说明了这次工程真正完成了什么。我们不再只凭一条训练 loss 讲故事，而是把 checkpoint parent、rollout token、工具 observation、fault opportunity、实际 exposure、credit mask 和 held-out outcome 串成了可追溯链。正面结果与负面结果使用同一套 artifact：恢复次数上升是真实结果，重复调用增加、停止校准缺失和 E4 gate 未通过也同样是真实结果。

**Failure resilience 不等于 clinical safety，也不等于 production robustness。**

SFT 解决动作语言，RL 解决行为选择；Controlled Failure 让失败成为可观察变量，Fatal-aware optimization 则进一步限制异常轨迹中的 credit 分配。对 Agent 来说，后训练真正优化的不是一段答案，而是在环境反馈持续改变状态时，模型接下来如何行动。
