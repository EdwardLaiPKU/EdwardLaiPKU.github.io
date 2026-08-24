---
layout: post
title: "从 Single-Agent RAG 的局限到 Task-Oriented Multi-Agent：医疗 Agent 系统的架构演进"
date: 2026-08-24
categories:
  - Agent
  - Medical AI
tags:
  - Multi-Agent
  - RAG
  - TaskPlan
  - Tool Calling
  - Agent Architecture
description: "一个能运行的 Multi-Agent 系统，为什么还需要继续重构？本文复盘医疗 Agent 从 free-form subtasks、全局 Skill 暴露，演进到 Typed TaskPlan、capability-scoped execution 与明确 runtime ownership 的过程，并讨论 SharedContext、Memory、RAG 和 Safety 在系统中的边界。"
published: true
math: false
---

> 从自由子任务与全局 Skills，到类型化任务、能力边界与可控编排

## 开场：一条医疗问题里其实有三种任务

> 我最近餐后血糖偏高、偶尔头晕，严重吗、应该何时就医？日常饮食和运动如何调整？请同时给出最新糖尿病临床指南依据。

对用户来说，这是一条问题；对 Agent runtime 来说，它至少包含三种不同工作：`risk_triage` 判断风险与就医时机，`lifestyle_guidance` 组织饮食和运动建议，`clinical_guideline` 查找最新指南依据。三项工作的证据要求、可用能力和输出约束并不相同。

如果把整条问题交给一个 LLM + RAG，它可能回答得出来。但系统如何确认风险被优先处理、指南真的经过检索？如果答案有问题，失败发生在路由、工具调用、检索还是最终综合？当这些问题无法被稳定回答时，瓶颈就不只是模型能力，而是系统缺少可检查的执行结构。

这也是我回看这套医疗 Agent 架构时最深的感受：Multi-Agent 真正难的不是多创建几个 Agent，而是把任务语义、能力边界和运行责任变成可以执行、验证和失败的契约。后面的演进，基本都围绕这三件事展开。

这三个问题也对应三种不同的失控：任务语义不清时，系统不知道该拆什么；能力边界不清时，系统不知道谁能做什么；运行责任不清时，即使各个组件都返回了结果，也没人能说明哪一层应该对最终答案负责。把它们分开，才有可能逐层定位错误，而不是只盯着最后一段文本猜测模型哪里出了问题。

## 1. 从 Single-Agent + RAG 这个概念基线说起

Single-Agent + RAG 在这里是分析系统边界的 conceptual baseline，不是仓库可以证明的历史第一版。它仍然是一个很有价值的基线：链路简单、故障面小，对普通知识问答通常已经够用；RAG 还能补充模型参数之外的领域知识，并让回答有机会建立在检索材料上。

**Figure 1：Conceptual Single-Agent RAG**

```text
User -> Single Agent -> RAG -> Answer
                  \-> optional tools
```

问题出现在复合任务上。一个 Agent 同时承担意图识别、风险优先级、工具选择、检索、建议生成和最终措辞，等于把规划与执行压进同一个概率过程。模型不一定答不出来，但系统会越来越难控制、评估和定位错误。

这也是我后来区分“知识不足”和“控制不足”的原因。RAG 能缓解前者，却不会自动定义任务之间的优先级、某个角色可以调用什么，或者最终由谁对答案闭环。继续增加文档或上下文，修复不了控制边界。

对于单意图问题，这种差别并不明显：一次检索、一次生成就能完成工作，额外编排反而增加延迟和故障面。到了“风险判断 + 行为建议 + 最新证据”这类组合问题，系统需要的不只是更多知识，而是把不同责任拆开，并能确认其中任何一步失败后会发生什么。复合任务的问题不是模型一定答不出来，而是一次看似成功的生成掩盖了太多无法单独验证的决定。

## 2. 真正可追溯的 V1 已经是 Multi-Agent

仓库中最早可追溯的 V1 已经有 Coordinator、LeadAgent、Consultation、Diagnostic、Research、SharedContext、九个 Skills 和 Dense RAG。它不是等待被“升级成多智能体”的幼稚 baseline，而是一套已经能拆解请求、指派角色和综合结果的中心化 Multi-Agent 系统。本文中的 V1/V2 指整个 Medical Agent 架构阶段，与后文 PreRiskGate 自身的 V1/V2 版本号不是同一个维度。

**Figure 2：Earliest Traceable V1**

```text
User -> Coordinator -> LeadAgent
                      -> subtasks[{description, assigned_agent}]
       -> one Worker, or SharedContext + several Workers
       -> Consultation / Diagnostic / Research
       -> globally exposed 9 Skills
       -> Dense RAG (BGE + Milvus)
       -> Lead synthesis -> User
```

V1 已经形成第一层抽象：Agent 是决策层，Skill 是能力层，能力目录包括七个领域能力和两个记忆访问能力。Worker 内部通过 ReAct-style function/tool-calling loop 执行：`LLM -> tool call -> observation -> LLM`。AgentLoop 解决的是“一个 Worker 怎么执行”，还没有回答“一个复杂请求该怎样被整个系统组织”。

这一层已经让领域能力可以被多个角色复用，tool observation 也能回到同一轮决策；但上层任务语言与能力暴露仍然偏自由。

当时三个专业角色分别处理一般咨询、诊断相关问题和研究问题。当前规范角色是 Consultation、Triage、Research，`DiagnosticAgent` 只保留为兼容别名。Diagnostic → Triage 不只是 rename，而是把责任从“诊断”收缩到风险识别与分流。简单计划只进入一个 Worker；复杂计划才把两到三个任务放进 SharedContext 并行执行，再由 Lead 综合。

所以真实演进不是 Single-Agent → Multi-Agent，而是从一个能运行但边界偏软的 Multi-Agent V1，走向任务语义、能力权限与 runtime ownership 更明确的 V2。


## 3. 一个已经能跑的 V1，问题到底在哪里

回看 V1，我认为最关键的不是它缺了哪些组件，而是三个已有抽象没有形成足够硬的契约。

### Free-form task semantics

旧路由已经会输出 `description` 和 `assigned_agent`，所以不能说它“没有任务拆分”。问题在于 task semantics 藏在自然语言 description 中。系统知道这段文字交给谁，却不稳定地知道它究竟属于风险分诊、生活方式建议还是临床指南；任务数量、类型与角色兼容性主要依赖 prompt 约定。

### Global capability exposure

Skill 与 Agent 的实现解耦本身没有问题，问题是所有 Worker 都能看到全部九个 Skills。即使 prompt 告诉 Consultation 不要做深度研究、告诉 Research 不要给生活方式建议，相应工具 schema 仍处于模型的动作空间里。角色分工更像行为期待，而不是权限边界。

### Loose runtime ownership

旧版已经有 Coordinator，也已有 Lead 的规划和综合，但 planning、dispatch、memory、synthesis、safety、persistence、return 之间缺少足够明确、可执行的责任 contract。系统能跑通，故障时却可能出现每层都参与了一部分、没有一层拥有请求闭环的情况。

三个问题彼此加强：任务没有稳定类型，能力就难以按任务收敛；能力没有权限边界，专业 Worker 容易退化为角色提示；生命周期没有明确 owner，context、安全与持久化便可能在不同执行路径上漂移。V2 的重点不是增加更多抽象，而是把这三个软边界变硬。

这里最容易产生的误判，是把“已经有 Coordinator、已经有多个 Agent、已经有工具”当作架构问题已经解决。组件存在只说明系统拥有这些名词，不说明它们之间形成了可执行的约束。V1 的价值恰恰在于它证明了流程可以工作；V2 要解决的，则是流程在边界输入、失败路径和后续重构中还能否保持同样的含义。

## 4. Free-form Subtasks → Typed TaskPlan

TaskPlan 的作用不是让 JSON 更整齐，而是把路由输出变成执行契约。当前每个任务都有稳定的 `task_type`，取值限定为七种：`general_consultation`、`lifestyle_guidance`、`risk_triage`、`symptom_analysis`、`clinical_research`、`clinical_guideline`、`disease_classification`。

**Code 1：TaskPlan 的核心 schema**

```python
@dataclass(frozen=True)
class TaskPlanTask:
    task_id: str
    task_type: str
    agent: str
    instruction: str

@dataclass(frozen=True)
class TaskPlan:
    complexity: str
    tasks: tuple[TaskPlanTask, ...]
```

**Table 1：Free-form Subtasks vs Typed TaskPlan**

| 维度 | Free-form Subtasks | Typed TaskPlan |
|---|---|---|
| 任务表达 | `description` | 有限 `task_type` + `instruction` |
| 语义边界 | 藏在自然语言中 | 封闭 taxonomy |
| 约束来源 | prompt convention | schema contract |
| Agent 指派 | 自由指派 | task–Worker compatibility validation |
| 执行分支 | 列表长度间接决定 | 显式 `complexity` 与基数共同验证 |
| 数量规则 | 输出后再解释 | simple 一个；complex 两到三个；上限三个 |

类型化之后，系统可以在执行前检查任务 ID、类型、数量、Agent 配对，以及每个 Worker 是否至多只接收一项任务。通过验证的计划再用一层很薄的 adapter 转成运行态 SubTask，例如 `task_type -> type`、`agent -> assigned_agent`，同时带上 session 与 request context。TaskPlan 负责“应该做什么”，SubTask 负责“怎样进入执行”。

`complexity` 也不再只是对列表长度的事后猜测。simple 计划必须恰好有一个任务，complex 计划必须有两到三个任务，验证通过后 Coordinator 才据此选择单 Worker 或 Swarm 路径。这样，规划结果不只是告诉系统“调用哪个 Agent”，还同时声明了执行模式和可以被检查的任务边界。

这一步对我最大的价值不是“JSON 更规范”，而是让错误更早暴露。过去，一个看起来合理的自由文本子任务可能一路进入执行层；现在，未知类型、多余字段、非法 Worker 配对或错误任务数量会在规划与执行之间直接失败，并进入受控回退。

这带来的是结构可控性，而不是语义正确性的证明。一个合法的 `clinical_guideline` 任务仍可能来自错误理解；模型也可能把本应独立的研究任务合进一般咨询。Schema 能减少结构错误，但不能证明自然语言理解正确。换句话说，schema validity ≠ semantic correctness，这条边界会在坏案例中再次出现。


## 5. Capability Reuse ≠ Capability Exposure

能力层应该复用，暴露面应该收敛。三个 Worker 可以共享知识检索与记忆访问的实现，但并不意味着它们应该看到同一套工具。

**Table 2：Current Capability Matrix**

| Capability | Consultation | Triage | Research |
|---|:---:|:---:|:---:|
| `search_knowledge` | ✓ | ✓ | ✓ |
| `assess_risk` | ✓ | ✓ |  |
| `analyze_symptoms` |  | ✓ |  |
| `recommend_lifestyle` | ✓ |  |  |
| `disease_code` |  | ✓ | ✓ |
| `clinical_guideline` |  |  | ✓ |
| `deep_research` |  |  | ✓ |
| `search_history` | ✓ | ✓ | ✓ |
| `search_similar_cases` | ✓ | ✓ | ✓ |

**Code 2：ResearchAgent capability allowlist**

```yaml
research_agent:
  allowed_tools:
    - clinical_guideline
    - deep_research
    - disease_code
    - search_knowledge
    - search_history
    - search_similar_cases
```

真正的 hard boundary 在 Agent 构造和工具暴露阶段：每个 Worker 得到按策略过滤后的 SkillRegistry，模型看到的 schema 也来自这个 registry。于是系统可以写出 deterministic invariant，例如 `ConsultationAgent cannot access deep_research`，而不只是期待 Consultation “最好不要调用”它。Validator 仍会做 warning-level second check，但授权边界不是 validator，而是 registry construction / exposure。

这项改变有三个直接收益。第一，缩小模型的工具动作空间，减少无关 schema 的竞争；第二，让专业 Worker 的边界真实存在，不再只靠角色描述；第三，让权限成为可以直接测试的 invariant。不要把“模型应该克制”当作权限系统：prompt 解释职责，filtered registry 定义权限。

能力复用与能力暴露分开以后，Agent 的专业性才不只是 system prompt 中的一段人设。Consultation 可以复用通用知识检索，却不能越过边界调用 `deep_research`；Research 可以访问指南与研究能力，却看不到生活方式建议工具。实现仍然共享，决策空间却按责任被裁剪，这比继续在 prompt 里堆叠“不要做什么”更可靠。

更重要的是，这个边界可以从正反两面测试：既检查一个 Worker 应该看到什么，也直接检查它不应该执行什么。当新的 Skill 被加入共享发现机制时，默认结果不应是每个 Agent 自动获得它，而应由策略明确决定暴露对象。这样，能力目录的增长不会悄悄扩大每个 Worker 的权限。


## 6. Coordinator vs Lead：谁拥有 Request Lifecycle

V2 没有新造一个 Coordinator；它重新明确了已有角色的责任：Lead 是 planner / synthesizer，Coordinator 是 request runtime owner。

**Table 3：Coordinator vs Lead**

| Responsibility | Coordinator | Lead |
|---|:---:|:---:|
| Context / memory loading | ✓ |  |
| PreRisk | ✓ |  |
| TaskPlan generation |  | ✓ |
| Single / Swarm dispatch | ✓ |  |
| Worker dispatch | ✓ |  |
| Swarm synthesis |  | ✓ |
| Final Host safety | ✓ |  |
| Persistence / return | ✓ |  |

Lead 负责把请求变成经过解析、验证或回退的 TaskPlan，并在多个 Worker 返回后综合贡献。Coordinator 决定何时读取 context、运行前置风险判断、选择单 Worker 或 Swarm 路径、分发任务、执行 Host 级最终安全、持久化并返回。

这也划清了两个经常混在一起的决策。Lead 可以决定一个复合请求包含哪些任务，以及多份贡献如何组织成一份回答；它不负责选择 Worker 内部的具体工具。后者由每个 Worker 的 AgentLoop 在自己的 scoped registry 中决定。计划、专业执行、综合与 Host 交付因此形成连续但不重叠的责任链。

这个区分让我不再用“谁调用了谁”判断所有权。runtime ownership 也不要求所有底层 write 都位于同一个类；关键是只有一个 Host 对请求从进入到交付的生命周期负责。Lead 可以影响答案，但不拥有请求闭环。

显式 owner 的意义在失败路径上更明显：普通路由与 PreRisk override、单 Worker 与 Swarm、正常综合与 fallback 最终都要回到同一个 Host 出口。只有这样，安全复核、持久化文本和实际返回给用户的文本才不会各自沿着不同路径结束。


## 7. 不要把所有东西都 Agent 化

明确了 Worker 与 runtime owner 以后，SharedContext、Memory、RAG 的位置也更容易理解。它们都是跨 Agent 基础设施，没有必要被包装成与三个 Worker 平级的新角色。

SharedContext 是 request-scoped execution blackboard，保存 SubTasks、status、events 与 contributions，让 Coordinator 和 Lead 知道一次 Swarm 执行正在发生什么。它不是聊天记录，也不承担跨 session 的用户记忆。

Memory 处理对话连续性：Redis 是默认 short-term backend，并可明确降级到 in-memory；Mem0 是可选、按用户隔离的 long-term memory。第一篇不需要深入后端细节，重要的是它们与 SharedContext 使用不同的生命周期和身份边界。

RAG 则位于 `Agent -> Skill -> Retrieval infrastructure` 这条链路的下方。Worker 调用知识或指南 Skill，Skill 再使用 BGE/Milvus Dense、BM25 与 RRF 组成的检索基础设施。RAG 为不同角色复用知识能力，却不是另一个负责决策的 Agent。

简化来看：SharedContext 记录“这次并行执行正在发生什么”，Memory 记录“用户此前发生过什么”，RAG 回答“外部知识库中有什么相关证据”。把三者分开，才能为 context propagation 指定正确的 owner 和测试边界。

这层区分也防止了“context”一词掩盖不同问题。Worker 需要知道本次任务和同一请求的背景，不代表它应该直接读取其他 Worker 的全部过程；系统需要恢复对话历史，也不代表这些消息应该写进 Swarm 的任务状态；检索返回了证据，也不代表证据天然变成长期记忆。不同状态只有经过明确的传播路径，才会进入下一层。


## 8. Medical Safety 为什么必须跨整个生命周期

医疗安全不是某个 SafetyAgent 的任务，也不能只靠最终 prompt 里的一句免责声明。当前链路有三个不同层次：PreRiskGate 改变路由，Agent-level `assess_risk` 支持 Worker 执行，FinalSafetyBoundary 约束用户可见输出。

Coordinator 在读取 short-term history 后、进入普通 router 前运行 PreRiskGate V1。若结果为 `ESCALATE`，系统不会直接拼接一段安全回答，而是创建一个确定性的单任务 `risk_triage -> Triage` TaskPlan；普通的复杂路由暂时让位于风险处理。

`assess_risk` 则是 Consultation 和 Triage 范围内的 Skill。Host-level routing safety 和 Worker capability 不是一回事：前者决定请求先走哪条路径，后者帮助专业 Worker 完成已分配任务。

最后仍需要统一 FinalSafetyBoundary。Worker 的结果安全，并不意味着 Lead synthesis 后的答案天然安全；多份贡献重新组织后，风险提示的顺序和措辞都可能变化。因此单 Worker、fallback 与 Swarm synthesis 的用户可见出口都进入最终边界，Coordinator 在持久化和返回前再做幂等复核。Safety 在这里是 request lifecycle 的属性，而不是一个孤立模块。

这种分层没有把安全判断交给某一个“更聪明的 Agent”。相反，它把不同性质的约束放到最合适的位置：紧急风险尽早影响路由，任务内部的判断交给受限 Worker，最终交付再由统一边界兜住。三层之间可以互补，却不能把其中一层的存在当作跳过另一层的理由。


## 9. 一次复杂请求完整走一遍

回到开头的血糖问题。在一条用于说明架构语义的示例路径中，Lead 可以生成包含 `risk_triage`、`lifestyle_guidance`、`clinical_guideline` 的复杂 TaskPlan。Coordinator 验证计划后把任务转成 SubTasks，复制同一 session/context，再交给 capability-scoped Workers 并行执行。

Worker 不争夺最终回答权，只向 SharedContext 写入 contributions。Lead 读取贡献并综合：风险与就医条件需要靠前，生活建议不能冲淡风险提示，指南依据需要与回答建立关系。综合结果进入 FinalSafety，Coordinator 完成 Host 复核、持久化与返回。整条链路可以缩写为：`TaskPlan -> SubTasks -> scoped Workers -> contributions -> Lead synthesis -> FinalSafety -> Coordinator -> persistence`。

这条路径的价值不只在于并行。每一个中间产物都给出了更清楚的故障位置：TaskPlan 不合法是规划问题，能力不可见是权限问题，contribution 缺失是执行问题，综合后措辞越界则由最终安全出口处理。即使用户最终只看到一段回答，系统内部也不再只有一个不可分解的生成结果。

这是一条 architecture-semantic example，不是某次 stochastic router 的固定 trace。真实输入可能被规划成一个、两个或三个任务；如果命中 PreRisk 的升级规则，还会先进入确定性的单 Triage 计划。类型化路由让执行可验证，并没有把自然语言路由变成确定函数。


## 10. Bad Cases 与 Executable Contracts

路由是最直接的坏案例。边界型、多意图输入仍可能过度拆分、拆分不足或选错任务类型。Typed TaskPlan 可以拒绝非法结构和错误配对，却不能证明模型正确理解了“查指南”究竟是独立研究任务，还是咨询回答中的证据要求。schema validity ≠ semantic correctness。

PreRisk 暴露的是泛化问题。确定性规则路径清楚、可复现，但同义改写与上下文组合仍是弱点。一个更丰富的候选方案在开发集上很强，却没有通过 frozen holdout 的晋级条件，因此没有进入当前运行路径。dev 强不代表 frozen holdout 能晋级，尤其不能用前者覆盖安全组件在未见样本上的失败。

Hybrid Retrieval 的坏案例则提醒我，聚合改善不等于每条 query 都改善。Dense 与 BM25 可以互补，固定 RRF 仍可能在个别查询上退步；排序分数也不能被解释为临床置信度。这里的细节更适合在下一篇展开，但它说明基础设施同样需要保留反例。


测试的价值不是 pytest 数量，而是把 Prompt 和文档中的设计约定变成 executable contracts。第一篇最值得保留的只有四类 invariant：TaskPlan schema、capability visibility、context propagation、final safety exits。它们分别防止规划结构漂移、工具权限扩大、请求上下文丢失，以及某条用户出口绕过最终安全。

这四类契约正好对应全文的控制边界。它们不证明某次回答在医学上正确，也不消除模型路由的随机性；它们保证的是，换模型、改 prompt 或重构运行时之后，系统仍然以同一套任务语言、权限范围、上下文路径和最终出口工作。对 Agent 系统来说，这类“不允许悄悄变化”的约束往往比测试数量本身更重要。

> **Engineering Note**：本次写作审计中，377 项核心架构契约测试可复现；完整 frozen-artifact replay 仍暴露 Windows 换行和临时目录权限问题，因此本文不使用“全量测试全绿”作为结论。


## 11. 最终架构与 Lessons Learned

把三条主线合在一起，当前系统是一套中心化、固定 Worker 的 Task-Oriented Multi-Agent 架构：TaskPlan 定义工作语义，filtered registry 定义能力边界，Coordinator 拥有请求生命周期；SharedContext、Memory、RAG 与 Safety 围绕这三条边界提供基础设施。

**Figure 3：Current V2 Runtime**

```text
User
  -> SwarmCoordinator                         [runtime owner]
       -> ShortTermMemory                     [Redis default / memory fallback]
       -> PreRiskGate V1
            -> ESCALATE: TaskPlan[t1 risk_triage -> Triage]
            -> PASS/error: optional Mem0 -> LeadAgent router
       -> validated TaskPlan
            -> simple: assigned Worker
            -> complex: adapter -> SharedContext -> assigned Workers in parallel
                 Workers: Consultation | Triage | Research
                 each -> AgentLoop -> filtered SkillRegistry -> allowed Skills
                         Knowledge Skills -> Dense/Milvus + BM25 -> RRF
                         Memory Skills -> scoped memory facade
                 contributions -> LeadAgent synthesis
       -> FinalSafetyBoundary + Host idempotent recheck
       -> persistence/session summary -> user-facing result
```

### 一个刻意保留的边界：Deep Search

ResearchAgent 通过稳定的 `deep_research` adapter seam 接入当前仓库内的 inference workflow。Agentic SFT、GRPO、rollout 与 Fatal-Aware training 属于独立的 Medical Deep Search 项目，不是本文仓库当前具备的训练能力，也不是现行调用已经跨越的外部运行时。


### 五条 Lessons Learned

1. **Multi-Agent 的关键不是 Agent 数量，而是任务语义。** 先定义有限、可验证的任务，再分配专业 Worker。

   如果任务只是一段自由描述，多几个 Agent 只是多几个可能的执行者；只有任务类型进入 schema，编排层才真正拥有可讨论的业务语义。

2. **Capability reuse 和 exposure 是两个不同问题。** 实现可以共享，模型可见的动作空间应按职责收敛。

   共享代码降低重复，收敛暴露面降低误调用；把两者混在一起，往往会为了复用而意外扩大权限。

3. **Planner 和 runtime owner 不应该是同一个概念。** Lead 负责计划与综合，Coordinator 对请求闭环负责。

   能生成好计划的组件，不必同时承担记忆、安全、持久化和返回；责任拆开以后，失败归属反而更清楚。

4. **Shared execution state 与 conversational memory 要分层。** 它们服务不同生命周期，也需要不同的传播契约。

   同一次 Swarm 的任务状态不应被误当成用户历史，用户历史也不应自动成为每个 Worker 的共享黑板。

5. **Medical safety 是 request lifecycle 的属性。** 路由前、Worker 执行中和最终用户出口承担不同安全职责。

   安全不是在末尾补一句话，也不是创建一个角色就结束；它必须覆盖请求改变路径、能力执行和最终交付的关键边界。

本篇主要讨论 orchestration。下一篇我会继续拆解这个架构里三个容易“代码里看起来有，但运行时未必真正成立”的部分：Hybrid Retrieval、Memory 和 Safety，包括 Dense + BM25 + RRF 的实际 trade-off、session/context 的传播边界，以及医疗安全为什么不能只依赖 LLM 判断。
