---
layout: post
title: "从答案监督到轨迹监督：Medical Deep Search Agent 的 Agentic SFT"
date: 2026-09-07
lang: zh
published: true
math: true
categories:
  - AI Agents
  - LLM Post-Training
tags:
  - Agentic SFT
  - Recovery SFT
  - QLoRA
  - Tool Calling
  - Medical AI
description: "从普通答案监督进入 Agent trajectory supervision，拆解 Medical Deep Search Agent 中 Tool Call、Observation、Loss Mask、QLoRA 与 Recovery Curriculum 的真实训练契约，以及为什么标准 SFT 仍停留在 Behavior Cloning。"
---

> 用环境 Observation 作上下文，只监督 Tool Call、Recovery Action 与 Final Response

我第一次把 Base 模型放进 Medical Deep Search 的严格运行时，看到的并不是普通的“医学问题回答得不够好”。

在那组历史 27-task artifact 里，Base 模型 27 次都把输入中的 `policy_state_context` 包在 fenced JSON 里回显。严格 parser 无法把这些文本解析成 canonical policy action，于是 27/27 parser-invalid，工具调用总数为 0，也没有一次完成最终响应。

这件事后来成了我理解 Agent 后训练的起点：模型会生成文本，不等于模型知道什么属于 Agent Action。如果样本只给出问题和答案，它可能学会答案的语言风格，却没有被明确训练过是否搜索、怎样构造参数、如何读取 Observation，以及何时结束。

Agentic SFT 真正改变的不是“多塞一些工具示例”，而是先划清 token ownership：哪些文本由 Environment 提供，哪些动作由 Policy 负责。即使它们最终都进入同一组 `input_ids`，训练责任也不能混在一起。

这个边界在日志里尤其容易被忽略。System prompt、用户问题、状态快照、检索结果和 assistant 输出经过 tokenizer 后都只是整数序列；Observation 又恰好借用 `user` role 传输。若只按 role 或整段文本计算 loss，模型就会被要求复述一个它本不应制造的世界。真正需要追问的是：这段 token 是环境事实，还是策略选择？前者应当被阅读，后者才应当被模仿。

因此，本文关注的不是怎样注册一个工具，而是怎样把可执行交互转换成含义清楚的监督样本。这个看似底层的选择，决定了后面的 loss 究竟在优化动作，还是在奖励模型伪造工具输出。

> **Observation 是模型决策的条件，但不是模型需要模仿的动作。**

## 1. Medical Deep Search Post-Training 到底训练什么

Medical Deep Search 的目标不是把基础模型训练成“自动诊断系统”，而是训练一个证据搜索策略：模型在明确的工具协议里生成动作，读取环境返回的证据，再给出受来源约束的总结。系统执行的是一串依赖前一步 Observation 的决策，不是一次性生成一段答案。

完整工程也不是一条从左到右的直线。真实 checkpoint lineage 在 E1C 之后发生了分叉。

**图 1｜真实 Post-Training branching map**

> Base：`Qwen3-VL-2B-Instruct`  
> └─ E1C：Agentic SFT，得到 `bee3dc...`  
> 　 ├─ E2C：Vanilla RL，从 E1C 开始  
> 　 └─ E1R：Recovery SFT，得到 `fb4613...`  
> 　 　 └─ E2F / E3：后续 RL 分支，从 E1R 开始

本文只展开 E1C Agentic SFT 与 E1R Recovery SFT；后续 RL 只作为接口出现。E1C epoch 2 是 final Agentic SFT adapter，一边直接交给 E2C，另一边先进入 E1R；E1R 才是 Recovery SFT adapter，也是 E2F/E3 的起点。

这张地图的意义是防止错误归因。E1C 的训练目标是让模型对齐到可执行的动作协议，E1R 增加窄范围失败后的示范覆盖，RL 才处理 sampled rollout 的结果信号。若把它们画成单线，很容易把不同初始化和目的下的数字误写成连续性能提升。

这里的“可执行动作空间”也需要降到工程语义上理解：模型输出必须能被严格 parser 接受，工具调用必须满足 schema，最终响应还要引用已经进入状态的 evidence ID。它不意味着模型因此拥有临床诊断资格，也不意味着一次格式正确的调用必然搜到高质量证据。E1C 先解决的是策略接口的冷启动，而不是整个医学任务。

Checkpoint 身份之所以在地图里保留，是因为后续比较都依赖父节点。`bee3dc...` 表示 E1C epoch 2 的 final adapter；`fb4613...` 表示在其上继续训练得到的 E1R。只要这两个身份混淆，读者就无法判断某个变化来自 Agentic SFT、Recovery Curriculum，还是后续 rollout optimization。

这种分层也保持了 Policy 与 Tool Environment 的解耦。SFT 不需要把检索器的内部实现写进模型权重，只需让策略遵守调用协议，并把返回结果当作新的 State。工具可以升级、语料可以替换，动作契约和训练责任仍然能够独立审计。

## 2. 普通 SFT 与 Agentic SFT 的根本区别

普通问答 SFT 的典型训练单元是 `Question → Answer`。Agent 的困难却发生在最终答案之前：它必须判断当前信息是否足够、是否调用工具、怎样构造参数，以及如何根据返回继续或停止。

**图 2｜两种监督对象的差别**

| 普通 QA SFT | Agentic SFT |
|---|---|
| Question | Question + 当前 State |
| 直接生成 Answer | 生成 Tool Action |
| 没有外部 Observation | 环境执行并返回 Observation |
| 一次性模仿最终文本 | 根据新 State 继续 Action 或 Final |

Agentic SFT 的样本因此是一段交互轨迹：`State → Action → Observation → Action → ... → Final`。State 不只包含原始问题，也包含工具预算、已获得的 evidence ID、当前可见证据，以及前一步工具是否成功。相同的问题，在不同 Observation 之后，正确动作可能完全不同。

从学习目标看，这仍然是 behavior cloning：策略模仿示范者在某个可见状态下采取的下一步。相比只模仿最终答案，是否调用工具、source-family 选择、JSON 语法、参数内容、动作顺序、引用与停止时机，都成为了明确的 token-level 监督对象。

可以把单步策略写成 `πθ(a_t | s_t)`：`s_t` 是当前可见状态，`a_t` 是工具动作或最终响应。E1C 提高示范动作在已覆盖状态下的条件概率，并不直接优化整条轨迹的结果指标。

例如用户询问某种药物的主要安全警告。初始 State 没有证据，示范 Action 查询 drug labeling；环境返回标签证据后，新 State 已经包含 evidence ID，下一步可能查询 literature，也可能结束。删除 Observation，第二个动作就失去状态条件；监督 Observation，又会让模型模仿本应由检索器产生的内容。

这与“把最终答案写得更完整”有本质差别。普通 QA 样本可以把检索、判断和表达折叠在一个 Answer 里，训练时看不见中间决策是否存在。Agent trajectory 则把每个可执行节点暴露出来：初始状态下先做什么，证据回来后再做什么，以及什么时候已有信息足以支持 Final Response。模型学习的对象从单个答案扩展成了条件化动作序列。

State 也不是装饰性的对话背景。它携带剩余预算、已经使用过的调用和当前证据集合，让同一个自然语言问题可以对应不同 Action。如果 evidence 已经存在，继续发出相同查询可能是浪费；如果 Observation 明确失败且允许重试，立即结束又可能过早。Agentic SFT 的价值，正是让这些差异进入动作的条件分布。

当然，behavior cloning 仍受示范覆盖限制。它只说明“在数据出现过的状态附近，目标动作应具有更高概率”，并不会探索数据之外的替代路径。这个上限将在最后一节统一收口，而不是把 SFT 描述成已经完成自主策略学习。

## 3. 一条真实 Agent trajectory 长什么样

轨迹最终要进入 Qwen chat template，逻辑事件必须映射到真实 role。正式 SFT rendering 没有独立的 `role=tool`，传输层只使用 `system`、`user` 和 `assistant`。

**图 3｜E1C 轨迹的真实角色与内容**

| 顺序 | Transport role | 逻辑内容 | 策略动作 |
|---:|---|---|---:|
| 1 | `system` | 系统约束、JSON grammar、工具 schema | 否 |
| 2 | `user` | 医学证据搜索问题 | 否 |
| 3 | `user` | `policy_state_context` | 否 |
| 4 | `assistant` | `tool_call` JSON | 是 |
| 5 | `user` | `tool_observation` JSON | 否 |
| 6 | `user` | 更新后的 policy state | 否 |
| 7 | `assistant` | 可选的第二次 Tool Call | 是 |
| 8 | `user` | 可选的第二次 Observation | 否 |
| 9 | `assistant` | `final_response` JSON | 是 |

Tool Call 是模型真正要预测的 canonical JSON，包括动作类型、工具名与结构化 arguments。生成结束后，严格 parser 才验证语法并把它转成执行请求；parser 不是替模型生成动作的模板。

Observation 同样被规范化为 JSON，可以携带执行结果、摘要、错误类型、是否可重试、evidence ID 与 provenance。它使用 `user` transport role，却在逻辑所有权上属于 Environment Output，只作为下一步决策的上下文。

正式 E1C 没有公开自然语言 reasoning/rationale 事件。这里的轨迹监督不是训练一段可见思维链，而是监督结构化 Action 与 Final Response。训练使用 `Qwen3VLSFTFormatter`，推理使用独立 inference formatter；二者共享 Qwen processor、原生 chat template 和兼容的 canonical grammar，准确说法是“协议兼容”，而非同一个 formatter 对象贯穿全程。

Canonical JSON 让动作可以被确定执行，也暴露了严格边界：多余的 Markdown fence、缺失引号或截断对象，都可能把合理意图变成 parser-invalid 文本。E1C 监督的不只是“应该搜索”，还包括让搜索意图落进 runtime 可以消费的动作协议。

这条 serialization contract 还连接了训练与执行。训练数据里的 Tool Call 并不是为了方便人阅读而写的伪代码，而是与推理期 parser 约定一致的字符串；工具执行后的 Observation 再以同样稳定的结构回到下一轮上下文。只要字段命名、枚举值或消息边界在两端漂移，训练 loss 即使下降，也可能只是在拟合一套 runtime 不认识的方言。

Final Response 也属于 Policy Action。它必须在已经取得的 evidence 范围内组织结论、引用和限制，不能把未出现的检索结果偷偷写回状态。结构化对象本身不能保证内容正确，但能让评测器检查引用是否来自已获得证据，也让下游系统区分结论、置信度与限制。这是可追踪性的最低契约，不是医学正确性的替代品。

从 ownership 看，transport role 只是消息如何穿过 chat template，logical role 才决定谁对内容负责。`user` 可以承载真实用户问题，也可以承载环境生成的 Observation；两者都不是模型应复制的标签。相反，assistant 产生的 Tool Call 与 Final Response 是策略选择，所以进入监督集合。

这也解释了为什么保存 trajectory 时不能只留最终对话文本。工具调用参数、Observation、evidence provenance、步骤顺序和最终响应必须共同保留，才能重建模型在每个动作前实际看到的 State。否则，一条看似正确的 Final Response 无法证明中间证据真的经过环境进入上下文。

## 4. Observation 只做上下文，Action 才吃 Loss

这是全文最核心的实现决定。如果把渲染后的整段对话直接计算 causal LM loss，模型会同时被要求“预测”搜索结果、错误消息和工具延迟。可这些内容来自外部环境，不是策略动作。训练模型模仿 Observation，会混淆 Environment 与 Policy 的责任。

因此，项目按 token ownership 建立 mask，而不是只看表面 chat role。

**图 4｜Loss ownership mask**

| Token 区域 | 模型可见 | 进入 CE | `labels` |
|---|---:|---:|---:|
| System 与工具 schema | 是 | 否 | `-100` |
| User question / Policy state | 是 | 否 | `-100` |
| Assistant Tool Call JSON | 是 | 是 | token ID |
| Tool name 与 arguments | 是 | 是 | token ID |
| Tool Observation JSON | 是 | 否 | `-100` |
| Assistant Final Response | 是 | 是 | token ID |
| Padding 与边界 special token | 是 | 否 | `-100` |

监督目标是一条 masked causal cross-entropy：

$$
\mathcal{L}_{\mathrm{SFT}}=-\frac{1}{|M|}\sum_{t\in M}\log \pi_\theta(y_t\mid y_{<t})
$$

`M` 是 policy-owned payload：Tool Call、可选公开 rationale、Final Response 或 Clarification，以及策略消息的 closing delimiter；assistant opening framing 仍被 mask。Collator 同时将右侧 padding 的 attention mask 置 0，并维持 padding label 为 `-100`。

正式 E1C sequence audit 记录了 728,456 个 context tokens 和 129,682 个 supervised policy tokens，且没有 truncation。只在 `M` 上归一化 loss，也避免大量 Observation token 在数值上淹没真正稀少的动作 token。

“Context only”不等于“不进入计算图”。Question、State 与 Observation 没有自己的 CE label，却参与 attention，改变每个 supervised token 的条件分布。失败 Observation 中的 `retryable=true` 不需要由模型复述，但会改变下一条 Action 的概率。

实现上仍需精确定位 assistant payload 的 token 边界，避免把 JSON 引号、括号或 delimiter 分给错误所有者。E1R 的正式 alignment audit 对 596 个事件边界完成了 text-exact 与 token-exact 检查；本文不再展开逐消息 renderer 的工程细节。

这里最容易出现的错误，是把“assistant-only loss”当作足够精确的实现描述。Chat template 可能自动加入 opening framing、结束 delimiter 和其他 special token，并非 assistant 消息范围里的每个 token 都有相同所有权。本项目监督 policy payload 与必要 closing delimiter，同时屏蔽 opening framing；这个差异虽小，却会直接改变模型被要求预测的协议边界。

Mask 也让指标解释更稳定。Context token 数约为 supervised token 的五倍多；如果所有位置共同参与平均 CE，曲线主要反映的可能是模型复述问题和检索文本的能力。现在的 loss 只对策略拥有的 token 求平均，因此一次下降更接近“示范动作变得更可能”，而不是“上下文语言建模变得更熟练”。

但这仍不是因果证明。Attention 会把 Observation 的内容传给后续 Tool Call，所以模型可以利用成功摘要、错误类型与已有 evidence ID；mask 只规定哪些位置提供直接标签，不保证模型一定以我们期望的方式使用上下文。是否真的形成稳定策略，仍要由 rollout 测试回答。

换一种说法，mask 管的是监督责任，不是信息流。环境文本可以影响梯度，因为它参与后续 token 的条件计算，但它没有独立的 target token loss。这个区分避免了两个常见误解：Observation 不是被模型“忽略”，Action supervision 也不是只监督整条 assistant 消息的粗粒度开关。

## 5. 303 条轨迹到底覆盖了什么

正式 E1C 数据版本是 `medical_sft_mvp_v006`。它包含 303 条轨迹记录，其中 279 条进入训练，24 条进入 validation，没有 test split。问题文本共有 193 个唯一字符串，因此不能写成“303 个唯一问题”。

**图 5｜E1C 数据与训练轮廓**

| 项目 | 数值 |
|---|---:|
| Trajectories | 303 |
| Train / Validation / Test | 279 / 24 / 0 |
| Unique question strings | 193 |
| Tool calls | 463 |
| 1-call / 2-call trajectories | 143 / 160 |
| Visuals / vision tokens | 0 / 0 |

303 条记录都是多轮 agent trajectory，而不是直接从 User 跳到 Final 的 QA。143 条包含一次调用，160 条包含两次，总计 463 次。任务覆盖 document evidence、medication safety 和 multi-source synthesis。

但 multi-source 不是 multi-tool。正式数据只有一个物理工具：`medical_knowledge_search`。`drug_labeling` 与 `medical_literature` 是 arguments 中的 source family。58 条轨迹查询 drug labeling 后结束，85 条查询 medical literature 后结束，160 条按两种顺序组合 source family。模型学习的是 call-vs-finish、source-family 选择、查询参数与动作顺序，并没有多个物理工具之间通用 routing 的证据。

真实 processor 渲染后，每条记录长度为 1,783–3,765 tokens，中位数 3,459；监督 token 为 333–540。大量长度来自问题、状态和 Observation，它们不吃 loss，却构成下一步 Action 的条件。

另一个必须公开的边界是：模型与 processor 虽具备视觉能力，正式 E1C 的 visuals 和 vision tokens 都是 0，E1R 也是 0。本文描述的是 VLM-capable runtime 上的 text-first Agentic SFT，不是多模态训练成果。

303 与 193 的差异来自记录身份和问题表面不是同一个统计口径。Trajectory record 还绑定 source candidate、动作顺序与 provenance；同一个问题字符串可以形成不同的、可追踪的训练事件。因此，303 可以描述 trajectory records 的规模，却不能被改写成 303 个语义上独立的问题。这个限定既不否定数据版本的唯一身份，也避免把重复问题模板包装成更大的任务覆盖。

同样，463 是示范中的调用总数，不是搜索能力种类。真正值得读的是路径结构：一部分样本取得单一 source family 后结束，另一部分在两个 evidence family 间继续一步，再生成 Final Response。数据工厂有意把工具调用变成必要中间动作，而不是让模型绕过环境直接完成问答。

这批数据的价值首先是协议与轨迹正确性。Train/validation 的隔离可以支持 teacher-forced 训练检查，但 0 test split 意味着它本身不承担最终行为评测。数据规模、任务覆盖和评测证据必须分别陈述，不能因为 trajectory schema 完整就顺带宣称泛化能力已经得到验证。

从 curriculum 角度看，一调用与两调用样本共同提供了停止时机的对照：取得第一类证据后，有些状态应当结束，有些状态仍需要第二类来源。虽然这种覆盖仍很窄，它至少让“继续搜索”和“形成 Final Response”都成为显式示范动作，而不是隐藏在最终答案生成过程中。

## 6. 2B + QLoRA：让后训练真的能在本地运行

Agent trajectory 比普通短问答长得多。即使只有 303 条记录，真实序列中位数也接近 3.5K tokens。要在一张 RTX 4060 Laptop GPU 上完成训练，模型尺寸、量化方式与可训练参数面必须一起设计。

E1C 使用 `Qwen3-VL-2B-Instruct`。模块审计给出的 unique base parameter count 是 2,127,532,032，可简写为 2.128B。Base 以 bitsandbytes NF4 4-bit 加载，启用 double quantization，计算 dtype 为 BF16；训练在冻结的 4-bit Base 上挂载 LoRA。

LoRA 配置为 `r=8`、`alpha=16`、`dropout=0`，目标是 196 个 language attention / MLP projection modules。最终可训练参数为 8,716,288，约占 unique base parameters 的 0.41%。视觉编码器、projector、embedding 与 LM head 均不在 target 中；正式数据没有 vision tokens，也就没有为视觉更新提供监督基础。

训练使用 microbatch 1、gradient accumulation 4、有效 batch 4，并开启 gradient checkpointing，不做 packing，也不允许 truncation。正式 E1C 在 RTX 4060 Laptop GPU 上运行 2 epochs、140 optimizer steps，约 4 小时 32 分，得到 SHA 以 `bee3dc...` 开头的 adapter。

这套 local-first 路线的重点不是展示“消费级 GPU 也能跑”，而是用预检压缩风险：先审计 LoRA target，再用真实 processor 检查序列长度，最后执行训练。模型 revision、dataset fingerprint、target-module hash、config 与 adapter SHA 因而能逐层核对。

Artifact 中存在语义未完全解释的 CUDA peak allocated/reserved 计数，甚至大于报告的物理显存容量，所以本文不发布 peak VRAM，也不从它反推硬件效率。可以确认的事实只是上述配置在该 GPU 上完成并留下可校验 checkpoint。

QLoRA 在这里解决的是可运行性与迭代成本，而不是凭空提高策略上限。4-bit Base 降低冻结权重的驻留成本，LoRA 将更新限制在语言 attention 与 MLP projection，gradient accumulation 则用时间换取有效 batch。三者共同让长轨迹训练适配本地硬件，但每个选择也缩小了可调整参数面。

为什么不更新 vision encoder 或 projector？最直接的答案不是“视觉模块不重要”，而是当前正式训练序列里没有视觉 token。若在没有视觉监督的情况下把视觉模块加入 target，只会扩大训练面，却无法从这批数据建立更新它们的因果依据。等真正的 image-grounded trajectory 进入数据，视觉 target 是否需要调整应当重新审计。

本地执行还有一个不显眼但重要的收益：失败与成功都能留下完整 lineage。序列预检、正式训练和最终 checkpoint 都可以绑定到具体 processor、数据 fingerprint 与配置，而不是只剩一张无法复现的 loss 截图。对这个阶段而言，可核验比追求更大的 batch 更重要。

## 7. Loss 下降，等于 Agent 真正变强了吗

E1C 的 teacher-forced validation NLL 从 epoch 1 的 0.007488 降到 epoch 2 的 0.005205。两个 adapter SHA 不同，optimizer step 从 70 增至 140，训练日志保持 finite。这证明训练确实发生，epoch 2 在同一批 24 条 validation trajectory 的 supervised tokens 上拟合得更好。

但 teacher-forced fit 不等于 rollout success，更不等于 medical correctness。验证时正确历史前缀已经给出；真实 rollout 一旦第一步偏离示范，后续 State 就可能进入训练数据未覆盖的区域。NLL 也不会判断检索证据是否充分或最终回答是否安全。

E1C 还有一组 10 条选定 TRAIN records 的 greedy grammar sanity，epoch 1 和 epoch 2 都达到 10/10 canonical action、正确 source filter 与 `top_k`。这是一项有用的训练集诊断，只能说明 checkpoint 能在这些样本上生成目标 grammar，不是 held-out 行为评测。

项目历史上曾用 27 tasks 对比 Base 与较早的 E1 adapter，观察到结构化工具行为变化；但那个 adapter 不是最终 E1C，评测也不覆盖 medical semantics。它只能作为早期背景，不能迁移成最终 E1C 的效果数字。Phase 8 中的 E1C 也是统一 runtime 下的 contextual baseline，而非预注册的 Base-vs-E1C 因果实验。

因此，本阶段最可靠的结论是：训练目标、loss、grammar sanity 与 checkpoint lineage 已闭环；最终 E1C 相对 Base 的严格 held-out 能力增益仍未被现有 artifact 单独证明。E1C 的训练证据强，因果能力证据更弱，两者不能用一条下降的 loss 曲线替代。

如果要真正回答“E1C 是否让 Agent 变强”，至少需要冻结同一批 held-out tasks、工具语料、parser、生成参数与 Base revision，只改变是否加载 E1C adapter。JSON validity、参数正确性、证据充分性与最终回答质量也应分开报告，避免一个总分掩盖失败发生在哪一层。现有 artifact 不满足这组完整条件，所以本文不补算，也不外推旧数字。

这份克制对技术复盘很重要。训练完成说明工程链路跑通；validation NLL 下降说明 teacher-forced 目标拟合改善；TRAIN grammar sanity 说明部分示范可以被规范生成。三者都是事实，却没有任何一个单独等价于开放式 rollout 稳定、医学内容正确或最终用户获益。

把证据分层之后，E1C 的价值反而更清楚：它交付了一个身份确定、协议可执行、可继续训练的 adapter，也暴露出正式因果评测仍是空白。后续阶段不需要用夸大的结论维护叙事，只需要在这个明确起点上补齐行为证据。

## 8. 从成功轨迹到 Recovery Curriculum

E1C 的 303 条示范都是 success-only demonstrations，正式 Observation 全部成功，没有 failure recovery variant。模型能模仿“搜索成功后下一步做什么”，却没有被示范过下游调用突然失败时怎样行动。

E1R 在 E1C 最终 adapter 上增加一组窄而可验证的 Recovery Curriculum。生成器从 schema-valid 的 E1C TRAIN trajectory 出发，保留成功 prefix 与 evidence provenance，再在选定位置插入受控失败 Observation。

**图 6｜Recovery Curriculum 的构造**

| E1C success-only | E1R controlled-failure variant |
|---|---|
| 真实成功 prefix | 同一个成功 prefix |
| 下游工具成功 | 插入 retryable / non-retryable failure |
| 继续并 Final | 监督 exact retry 或 limitation/stop |

正式 `medical_sft_failure_recovery_v005` 共 144 条 train-only records：24 条 normal replay，120 条 controlled failure。失败类型为 `timeout`、`rate_limited`、`backend_unavailable` 和 `capability_unavailable`；目标动作中 72 条走 retry，48 条走 stop。

“Retry”的含义必须保持很窄：重复相同 tool name 和相同 arguments。正式数据没有训练 query refinement、argument repair、rerouting 或 cross-tool switching，也只有 `medical_knowledge_search` 一个物理工具。它教的是在明确可重试条件下继续同一次调用，或在不可恢复和连续失败边界上诚实停止，不是完整的自主故障处理。

数据长度也触发了一次真实修正：4096-token 候选有 27/144 超长，因而被拒绝；最终 v005 使用 delta-once evidence state，将上限提高到 4608，144/144 全部适配且无 truncation。Controlled synthetic failure 让失败位置、错误类型与目标动作可审计，但也牺牲了部分真实 rollout failure 的分布复杂性。

120 条 failure records 内部也不是同一种动作：一部分在一次或两次 retryable failure 后继续 exact retry，合计 72 条；另一部分在 non-retryable failure 或连续失败边界上生成 limitation/stop，合计 48 条。24 条 normal replay 则保留正常路径，避免整个小 curriculum 只教模型对错误状态作答。

保留真实成功 prefix 是这个设计的关键。恢复动作并非从空白错误消息开始，而是在已经取得部分 evidence 的状态里发生。这样，模型看到的不只是“工具失败”，还看得到失败前已经积累了什么、预算还剩多少，以及是否存在可以安全总结的证据。Controlled failure 只改变选定的环境 Observation，目标动作的变化因此更容易解释。

Delta-once 也不是随意截短文本：evidence ID 持续留在 State 中，详细 evidence content 对每个 ID 只传一次。它减少重复上下文，又保留后续 Action 引用已有证据的能力。最终 4608-token 版本通过全量 sequence audit，说明压缩没有靠静默 truncation 掩盖超长样本。

代价仍然存在。真实故障可能需要改变 query、切换来源、降低 `top_k` 或请求澄清，而 v005 只覆盖 exact retry 与 stop。把这一窄动作集合写清楚，读者才能区分“训练过恢复相关样本”和“已经掌握通用恢复策略”。

## 9. Recovery SFT 能证明什么

E1R 的 checkpoint lineage 很清楚。Runner 先校验 E1C epoch 2 adapter 的 SHA `bee3dc...`，再把它作为可训练 PEFT adapter 加载。训练沿用相同的 4-bit Base、LoRA target、formatter ownership 与 masked CE，只替换为上述 144 条 curriculum，并将最大序列长度调整为 4608。

正式运行 1 epoch、36 optimizer steps，约 1 小时 40 分，得到新的 `fb4613...` adapter。输入输出 SHA 不同，receipt、source lineage、config 和 checkpoint timestamp 能互相闭合。这证明 Recovery SFT 的数据、训练和 parent-child checkpoint 真实存在。

但 E1R 没有 validation/test split，`best_epoch=1` 只是唯一 epoch。Gate A 使用 12 条正常 TRAIN candidates、每条采样 4 次，只能说明小型 normal probe 没观察到明显结构退化；其中没有失败注入。Phase 8 STANDARD track 同样没有 retryable failure exposure，也不能证明 recovery improvement。

**E1R 能证明与不能证明的边界**

| 能证明 | 不能证明 |
|---|---|
| 144 条 curriculum 的构成 | Recovery success 提高 |
| exact retry/stop 是监督目标 | 学会 refinement / rerouting |
| E1C→E1R checkpoint lineage | Reward 或 task success 提高 |
| 36 steps 与新 adapter | 医学回答质量提高 |

训练覆盖和行为证据不是同一件事。要证明效果，需要未来在相同 runtime 下进行 failure-exposed、held-out 的 E1C-vs-E1R comparison；当前最稳妥的结论止于“模型接受过这些窄范围动作示范”。

E1R 的训练配置还体现了“继续训练”而非从 Base 重启：parent adapter 作为可训练 PEFT 权重加载，learning rate 调整为 `5e-5`，formatter ownership 与 mask 规则保持不变。这样得到的新 SHA 能证明参数确实更新，却不能告诉我们更新方向是否在未见失败上泛化。

Gate A 与 Phase 8 的作用因此主要是排除部分灾难性问题。前者没有发现 normal probe 上明显的 parser、finish 或工具深度退化；后者提供统一 runtime 的 contextual baseline。但没有 failure exposure，就没有 recovery metric 的判别条件。即便某个总 reward 相同或不同，也不能单独归因于 Recovery Curriculum。

这也是表格里“能证明/不能证明”的分界：数据计数、监督动作、训练步数和 checkpoint 关系是直接 artifact 事实；能力提升则需要对照、暴露与明确指标。未来实验应让 E1C 与 E1R 面对相同的 held-out failure cases，并同时检查正确 retry、预算内终止、证据取得与最终响应，而非只数调用次数。

## 10. 从 Behavior Cloning 到 Agentic RL

到这里，两段 SFT 的职责已经足够清楚。Agentic SFT 模仿 demonstrated successful-policy actions：生成 canonical Tool Call，根据 Observation 继续，并用 evidence ID 组织 Final Response。Recovery SFT 模仿窄范围的 exact retry 与 limitation/stop actions。

两者都使用 standard masked CE。它们提高示范动作的条件概率，却不会比较同一问题的多个 sampled rollouts，也不会根据整条轨迹的结果直接更新策略。

**图 7｜SFT 到 RL 的接口，而不是结果预告**

| 阶段 | 学习信号 | 本文边界 |
|---|---|---|
| Agentic SFT | 示范 action token 的 CE | 模仿成功轨迹中的策略动作 |
| Recovery SFT | retry/stop action 的 CE | 增加窄范围失败示范覆盖 |
| Agentic RL | rollout-level signal | 下一篇展开，本文不讨论算法与结果 |

真实 handoff 仍然分支：E2C 从 E1C `bee3dc...` 开始，E2F/E3 从 E1R `fb4613...` 开始。不能画成 `E1C → E1R → all RL`，也不能把不同分支的结果倒推成 E1R 的独立贡献。

> Environment 决定模型看见什么，Policy 决定模型生成什么；Observation 是条件，Action 才是责任。

当这条边界落实为真实 role、canonical serialization、token mask、数据 manifest 与 checkpoint lineage，Agentic SFT 才不只是普通 SFT 的新名字，而成为后续 rollout optimization 的可执行起点。Article 05 到这里停止：它交付动作语言、成功示范与窄范围失败示范；下一篇再讨论如何用模型自己采样出的行为更新策略。
