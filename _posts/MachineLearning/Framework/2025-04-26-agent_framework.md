---

layout: post
title: agent框架
category: 架构
tags: MachineLearning
keywords: agent

---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$']], // 支持 $和$$ 作为行内公式分隔符
      displayMath: [['$$', '$$']], // 块级公式分隔符
    },
    svg: {
      fontCache: 'global'
    }
  };
</script>
<script async src="/public/js/mathjax/es5/tex-mml-chtml.js"></script>

* TOC
{:toc}

## 简介

[万字长文分析 10 种流行 Agent 框架设计思路，教你如何构建真正可靠的 Agent 系统？](https://mp.weixin.qq.com/s/8Hkq0HkVv4lYaNTB5O5jCA)OpenAI 采用更高层次、更具思想领导力的方式来定义 Agent：Agent 是能独立代表你完成任务的技术系统。这种模糊表述无助于真正理解 Agent 本质，在 Anthropic，架构上区分工作流和Agent：工作流是通过预定义代码路径编排大模型和工具的系统；Agent则是大模型动态自主控制流程和工具使用的系统，自主决定任务完成方式。Anthropic 将 Agent 定义为"...本质上只是**大模型在循环中基于环境反馈使用工具的系统**"。实际生产环境中，几乎所有的Agent 系统都是工作流和Agent的混合体。这正是我反感讨论"某物是否是 Agent"，而更倾向于讨论"**系统的 Agent 化程度**"的原因。

[Microsoft Agent Framework 与 Semantic Kernel 全维度深度拆解与实战指南](https://mp.weixin.qq.com/s/SMg-1KqEdM_7YqtNO6GKtA)在 AI 应用工程化的演化曲线上，行业其实经历了几个阶段：
1. Prompt 拼接期：直接调用 LLM，手写上下文，体验先行；
2. 函数/插件期：出现“让模型主动调用函数”能力（OpenAI function calling 等），开始把业务 API 暴露为结构化工具；
3. 规划与链式期：出现 Planner / Chain / Graph（如 SK Planner、LangChain Chains），对调用序列进行自动或半自动生成；
4. 多智能体期：从“单大脑 + 工具”转为“多个具有独立角色、记忆、工具、策略的主体”进行协同——本质是软件体系结构的再分层；
5. 运行时与治理期：需要托管、可观测、可靠恢复、跨边界（网络/组织/系统）交互与标准协议整合；面向 Agent 生命周期与交互协议的宿主体系。
6. 生态互操作期：通过 Model Context Protocol、RAG 数据网格、事件溯源等形成“智能操作系统”式平台。

## 理念之争：agent vs workflow

llm在持续计划（推理模型 ==> 自动推理等），目标都是为了提升单次LLM任务的准确率。但在实际业务中，很难通过一次LLM调用将所有问题解决。
### 业务上看

[LLM Agent是否能颠覆传统软件技术架构：反思代码与Prompt的本质差异](https://mp.weixin.qq.com/s/GgIupn6jHnpGZs9U0bBSxQ)一个虚拟案例：一个软件服务商，要开发一个自动的流程，对于其官方平台上用户对产品的评论进行分类和处理。如果用户表达了对产品问题的批评，则将问题映射到内部的问题分类树上的一个具体问题，并向用户发送一个致歉邮件，并提供产品优惠券。如果用户表达了对产品功能的赞赏，则将其赞赏的功能映射到营销团队的产品优势分类中，并向用户发送邮件邀请其成为内测用户。
1. 首先这个问题并不算难，以目前大部分LLM应用层开发人员的思维来说，构建一个小workflow对其进行实现就能做到一个很高的准确率，每个环节也很方便调教。在这个流程中，仍然是使用传统编程语言进行开发的流程，只是其中调用了包含LLM的函数，整个流程是在Python解释器（或其他语言）中执行的。下文称此方式为方案A。
2. 还有一种不同的设计思路：写一个巨大的Prompt，把上述所有逻辑都以Prompt形式编写，把LLM不能执行的操作都封装为tools，然后靠LLM根据该Master Prompt自主执行，自主决定调用什么函数。在这个设计中，几乎没有什么Python层的逻辑了（除了tools实现），所有逻辑都是在LLM的in-context中运行的。下文称此方式为方案B。

对于大部分开发者来说，方案A是理所当然，方案B是某种不稳定的玩具设计，而且没有什么明显的好处。在一般的架构设计思路下，都是建议把能够明确衡量的步骤单独拆分，这样可以进行验证，也往往可以直接输出该部分结果。只有说一个子任务很难用代码或者Prompt具体描述时，才考虑赋予LLM更多的自主权，给它更完整的Context和控制权，放手交给它来做。而后者的情况一般baseline都很低，有点像是研发团队怎么都弄不出更好了，所以交给AI试试吧。当然方案B也并不是没有优势，如果是一个站在一个运营岗的角度上，能够完全靠自己调优的巨大Prompt+tools，相对于每次变更需求都要去找产研开发和排期来说，有着更大吸引力。持有两种思路的人我都见过。而且在不同的领域中也有着不同的倾向。在文本模态，并且中间环节的结果可验证可处理的领域大多倾向于方案A，特别是随着系统“可靠性”要求的提升，越倾向于方案A。在非文本模态输出，或者是一些偏艺术/创意的领域，中间的文字结果也很难评估，这些会更倾向于方案B。

### 技术上看

构建Agent的本质其实是做三件事：
1. 如何利用好大模型？难点：哪些交给LLM做，哪些人工设定；人工设定太多通用性差、LLM控制太多稳定性差。**LLM的优势在重复性低、创造性高的区域**。比如：用户意图分析、工具选择、上下文参数分析等等。但是对于复杂任务、多任务编排、高频执行的固定场景等，没有优势。如何理解创造性和重复性（四象限）：固定输入+固定输出 意味着低创造性；执行频率高意味着重复性高。聚焦在研发过程的Agent的场景中，适合LLM来做并且稳定性相对较高的有：
    1. adapter(自然语言上下文<->程序运行上下文，如用户自然语言输入到程序执行，如从上下文中填充tool的参数) 
    2. router(在一组选择中，根据描述选择合适的几个，如从tool list中选择合适的进行执行)
    究其本质，还是利用了大模型最原始的能力：语义理解（Text Understanding）和文本生成（Text Generation）。
2. 如何组装好上下文？难点：哪些应该放到上下文；上下文太长，效率低、稳定性差；上下文太短，功能难以满足要求。在简单问题上，在注意力可涵盖的范围，堆叠上下文影响不大；复杂问题或者上下文过长的时候，lose in the middle以及注意力分配问题会很严重。聚焦在研发过程的Agent，除了需要对tool的调用、知识库的检索之外；还应该关注如何进行多轮对话、历史信息的维护（比如oncall场景，多轮对话排查），以及需要解决接口中无效信息繁多的问题（接口调用返回的json中，实际有效的信息较少）。
3. 如何让Agent和世界交互？不存在难点，工程编码问题。

[OpenAI“Agent万能论”遭打脸！LangChain创始人：Deep Search恰恰证明Workflows不可取代](https://mp.weixin.qq.com/s/vEaVEIM_hD2IE4ZVdotEgA)
1. LangChain创始人与OpenAI就Agent框架设计理念产生争议，认为不应将Agents和Workflows严格二分，大多数Agentic系统是两者结合；他指出Agent框架核心难点是确保LLM在每步都能获得恰当上下文，而非仅提供封装，框架应支持从结构化工作流到模型驱动的灵活转换；
2. 有些人将 Agent 视为完全自主的系统，能够长时间独立运行，灵活使用各种工具来完成复杂任务。 也有些人认为 Agent 是遵循预设规则、按照固定 Workflows 运作的系统。在 Anthropic，我们把所有这些变体都归类为 Agentic 系统，但在架构上，我们明确区分 Workflows 和 Agents：
    1. Workflows：依靠预先编写好的代码路径，协调 LLM 和工具完成任务；
    2. Agents：由 LLM 动态推理，自主决定任务流程与工具使用，拥有更大的决策自由度。
3. 让 Agents 稳定可靠地工作，依然是个巨大的挑战。LLM 本身能力存在局限性，且在上下文信息传递方面常出现错误或不完整的情况，后者在实践中更为普遍。导致 Agent 效果不佳的常见原因包括：System Message 不完整或过于简短、用户输入模糊不清、未向 LLM 提供正确的工具、工具描述不清晰、缺乏恰当的上下文信息以及工具返回的响应格式不正确。**构建可靠 Agents 的关键挑战在于确保大模型接收到正确的上下文信息**，而 Workflows 的优势在于它们能够将正确的上下文传递给给 LLMs ，可以精确地决定数据如何流动。
4. 许多 Agent 框架提供的 Agent 封装（如包含 prompt、model 和 tools 的类）虽然易于上手，但可能限制对 LLM 输入输出的控制，从而影响可靠性，像 Agents SDK（以及早期的 LangChain， CrewAI 等）这样的框架，既不是声明式的也不是命令式的，它们只是封装。它们提供一个 Agent 封装（一个 Python 类），这个类里面封装了很多用于运行 Agent 的内部逻辑。**它们算不上真正的编排框架，仅仅是一种封装**。 这些封装最终会让你非常非常难以理解或控制到底在每一步传递给 LLM 的具体内容是什么。这一点非常重要，拥有这种控制能力对于构建可靠的 Agents 至关重要。这就是 Agent 封装的危险之处。
5. 在实际应用中，Agentic 系统往往并非由单一 Agent 组成，而是由多个 Agent 协作完成。在多 Agent 系统中，通信机制至关重要。因为构建可靠 Agent 的核心，依然是确保 LLM 能接收到正确、充分的上下文信息。为了实现高效通信，常见的方法包括「Handoffs」（交接）等模式，像 Agents SDK 就提供了这种风格的封装。但有时候，这些 Agents 之间最好的通讯方式是 Workflows。而 Agent 框架则通过提供统一封装、记忆管理、人机协作、流式处理、可观测性和容错机制，大幅降低构建可靠 Agentic 系统的复杂度，但前提是开发者需理解其底层机制。
6. 大模型越来越厉害，那么是不是都会变成 Agents？虽然工具调用 Agents 的性能在提升，但“能够控制输入给 LLM 的内容依然会非常重要（垃圾进，垃圾出）”，**简单的 Agent 循环并不能覆盖所有应用需求**。
    1. OpenAI 的 Deep Research 项目是 Agent 的一个好例子，这同时也证明了针对特定任务训练的模型可以只用简单 Agent 循环。它的成功前提是：“你能针对你的特定任务训练一个 SOTA 模型”，而当前只有大型模型实验室能够做到这一点。对于大多数初创公司或企业用户来说，这并不现实。

总结来看，简单 Agents 在特定条件下有效，但仅限于数据和任务极为匹配的场景。对绝大多数应用而言，Workflows 仍然不可或缺，且生产环境中的 Agentic 系统将是 Workflows 和 Agents 的结合。**生产级框架必须同时支持两者**。

什么导致 Agent 有时表现不佳？大模型出错。为什么大模型会出错？两个原因：(a) 模型能力不足，(b) 传入模型的上下文错误或不完整。根据我们的经验，后者更常见。具体诱因包括：
1. 不完整或过于简略的系统消息
2. 模糊的用户输入
3. 缺乏恰当的工具
4. 工具描述质量差
5. 未传入正确的上下文
6. 工具响应格式不当

构建可靠 Agent 系统的核心难点在于确保大模型在每一步都能获得适当的上下文。这既包括严格控制输入大模型的内容，也包括运行必要的步骤来生成相关内容。在讨论 Agent 框架时，请牢记这一点。**任何让控制大模型输入内容变得更困难的框架，都是在帮倒忙**。

随着模型进步，所有系统都会变成 Agent 而非工作流吗？支持 Agent（相对工作流）的常见论点是：虽然当前效果不佳，但未来会改善，因此你只需要简单的工具调用型 Agent。我认为以下几点可能同时成立：
1. 这类工具调用型 Agent 的性能会提升
2. 控制大模型输入内容仍然至关重要（垃圾进，垃圾出）
3. 对某些应用，这种工具调用循环就足够
4. 对其他应用，工作流仍会更简单、经济、快速和优秀
5. 对大多数应用，生产级 Agent 系统将是工作流和 Agent 的组合

是否存在简单工具调用循环就足够的场景？我认为只有当使用针对特定用例进行充分训练/微调/强化学习的大模型时才会成立。这有两种实现路径：
1. 你的任务具有独特性。你收集大量数据并训练/微调/强化学习自己的模型。一个使用针对特定任务训练模型的简单工具调用型 Agent 案例是：OpenAI 的 Deep Research，这说明确实可行，并能产出卓越的 Agent。如果你能为特定任务训练顶尖模型——那么确实，你可能不需要支持任意工作流的框架，简单工具调用循环就足够。此时 Agent 会比工作流更受青睐。
2. 你的任务具有普遍性。大型模型实验室正在训练/微调/强化学习与你的任务相似的数据。编程是个有趣的例子，编码相对通用。

即使对于 Agent 明显优于任何工作流方案的应用，你仍会受益于框架提供的与底层工作流控制无关的功能：短期记忆存储、长期记忆存储、人在回路、人在环上、流式传输、容错性和调试/可观测性。**框架应提供的核心价值：可靠的编排层，让开发者能精确控制传入大模型的上下文，同时无缝处理持久化、容错和人机协作等生产环境问题**。


## 框架

### 现有框架对比

[Agentic AI：8个开源框架对比-2025更新](https://mp.weixin.qq.com/s/waol_6y7VH_SQZwNBJ9qQw) PS：有细节控制需求用langgraph，稍微抽象好一点可以尝试 Agno
Agentic AI 主要是围绕着大型语言模型（LLMs）构建系统，让它们能够拥有准确的知识、数据访问能力和行动能力。你可以把它看作是使用自然语言来自动化流程和任务。在自动化中使用自然语言处理并不是什么新鲜事 - 我们已经用 NLP 多年来提取和处理数据了。新的是我们现在能给语言模型的自由度，允许它们处理模糊性并动态做出决策。没错，大模型的优势就是处理模糊性和有一定的规划能力。但仅仅因为 LLMs 能理解语言，并不意味着它们就有代理性 - 甚至理解你想要自动化的任务。这就是为什么构建可靠系统需要大量的工程技术。**框架的核心，是帮你进行提示工程化和管理数据在大型语言模型（LLMs）之间的传输——但它们也提供了额外的抽象层，让你更容易上手**。

大多数框架都带有相同的核心构建模块：支持不同的模型、工具、内存和RAG。
1. 大多数开源框架或多或少都是模型不可知的。这意味着它们被构建为支持各种提供商。
2. 所有具有代理性的框架都支持工具化，因为工具对于构建能够采取行动的系统至关重要。它们还使得通过简单的抽象定义自己的自定义工具变得容易。如今，大多数框架都支持MCP，无论是官方的还是通过社区解决方案的。
3. 为了使代理能够在LLM调用之间保留短期记忆，所有框架都使用状态。状态帮助LLM记住在早期步骤或对话的部分中说过的内容。
4. 大多数框架还提供简单的选项来设置RAG，与不同的数据库结合，为代理提供知识。
5. 最后，几乎所有框架都支持异步调用、结构化输出、流式传输以及添加可观察性的能力。

有些框架缺少的东西
1. 有些框架有内置的多模态处理解决方案- 也就是文本、图像和声音。只要模型支持，你完全可以自己实现这一点。
3. 短期记忆（状态）总是包括在内的 - 没有它，你就无法构建一个使用工具的系统。然而，**长期记忆更难实现**，这也是框架之间的差异所在。有些提供内置解决方案，而其他的则需要你自己去连接其他解决方案。
3. 框架在处理多智能体能力方面也各不相同。多智能体系统允许你构建协作或分层的设置，通过监督者连接智能体团队。

框架在抽象程度、给予Agent的控制权以及你需要编写多少代码才能让事情运行起来方面各不相同。
1. 抽象程度。CrewAI 和在某种程度上的 Agno 都是为即插即用而设计的。 LangGraph也有相当的抽象程度，但它使用基于图的系统，你需要手动连接节点。这给了你更多的控制权，但也意味着你必须自己设置和管理每个连接，这带来了更陡峭的学习曲线。
2. 另一个区别点是框架假设Agent应该有多少自主权。有些是建立在这样的想法上：LLMs 应该足够聪明，能够自己弄清楚如何完成任务。其他的则倾向于严格控制 - 给代理一个任务，并一步一步指导它们。AutoGen 和 SmolAgents 属于第一种类型。其余的更倾向于控制。

一个 Agent 系统可能会因为其 Agent 循环向用户暴露的方式不同而呈现出不同的 “风格”
1. （LangGraph的）Agent 系统可以表示为节点和边。节点代表工作单元，边代表转移关系。节点和边本质上就是普通的 Python/TypeScript 代码——虽然图结构是声明式表达的，但图内部的逻辑仍是常规的命令式代码。边可以是固定或条件的。因此，虽然图结构是声明式的，但图的遍历路径可以完全动态。
    1. LangGraph 内置持久化层，支持容错、短期记忆和长期记忆。该持久化层还支持"人在回路"和"人在环上"模式，如中断、审批、恢复和时间旅行。
    2. LangGraph 内置支持流式传输：包括 token 流、节点更新流和任意事件流。
2. 大多数 Agent 框架都包含 Agent 抽象。**通常始于一个包含提示词、模型和工具的类，然后不断添加参数...最终你会面对一个控制多种行为的冗长参数列表，全部隐藏在类抽象之后。要了解运行机制或修改逻辑，必须深入源代码**。这些抽象最终会让你难以理解或控制在每一步传入大模型的具体内容。明确地说，Agent 抽象确实有其价值——能降低入门门槛。但我认为这些抽象还不足以（或许永远不足以）构建可靠的 Agent。我们认为最佳方式是将这些抽象视为类似 Keras 的存在——提供高级抽象来简化入门。但关键是确保它们构建在底层框架之上，避免过早触及天花板。这正是我们在 LangGraph 之上构建 Agent 抽象的原因。这既提供了简单的 Agent 入门方式，又能在需要时轻松切换到底层 LangGraph。

框架的通用价值在于提供有用的抽象，既降低入门门槛，又为工程师提供统一的构建方式，简化项目维护。
1. Agent 抽象
2. 短期记忆，长期记忆
3. 人机协作，获取用户反馈、审批工具调用或编辑工具参数。允许用户实时影响 Agent
4. 事后回溯，让用户在事后检查 Agent 运行轨迹
5. 流式传输
6. 调试/可观测性
7. 容错性，LangGraph 通过持久化工作流和可配置重试简化容错实现。
8. 优化，相比手动调整提示词，有时定义评估数据集并自动优化 Agent 更高效。

### 从能力来看框架

Agent 的四大能力，**每一环节的能力可大可小，每一环节的角色也可多可少**。PS：每谈论一个东西，这个东西就有它固有的脉络，按脉络走就都梳理清楚了。
1. 规划，Agent 的规划能力必须要有清晰、详细的指令才能稳定发挥，prompt 工程比想象中重要、执行起来也更琐碎。其次，从打分上看，目前 LLM 的数学、逻辑推理能力在 COT 的基础上也仅能勉强达到及格水平，所以**不要让 Agent 一次性做过于复杂的推理性规划工作**，而是把复杂任务强行人工拆解，而不是通过提示词让 LLM 自己拆解。PS：指令微调可以节约这部分的提示词，以及提高准确率。
3. 记忆
    1. 短期记忆，存储即时对话上下文、当前任务步骤、临时操作结果等动态信息。流程：Input（输入）→ Encode（编码）→ Store（存储）→ Erase（清除）
    2. 长期记忆，用户历史行为、企业知识库、任务经验等数据，支持个性化服务（偏好设置）和复杂推理。例如记录用户过去3个月的所有购物偏好，用于精准推荐。Receive（接收）→ Consolidate（整合）→ Store（存储）→ Retrieve（提取）。 有文章进行了更细的分类。 
      1. 情节记忆（Episodic Memory），比如记住是和谁互动（这个不见的“who”代表的是user，也可能是agent）、讨论了什么、何时发生等。
      2. 语义记忆（Semantic Memory），保存事实、概念、语言等一般知识，防止常识性幻觉
      3. 程序性记忆（Procedural Memory），保留已学会的操作和流程，可自动化执行，比如如何格式化文档、发邮件或跟随既有流程。
4. 行动。Action 能力强烈依赖于基座模型的 function calling 能力，即理解任务并依据 API 描述将自然语言的任务转换为 API request 的能力。由于执行准确率有限，要考虑到 function calling 失败的备选方案，比如：客服案例中，function calling 失败直接对接到人工；API 参数缺失，考虑根据 API 返回的错误信息，转换成用户语言继续追问用户。
2. 反省，反思能力依赖于它的记忆能力。PS：传统的软件系统没有反思能力，万一执行失败了咋办？报警喊人来修

如果希望最后能够达到通用 Agent 的目标，要做到通用，需要有如下的核心路径：
- 工具尽可能丰富，让 Agent 有充分的外部工具可以调用。比如给它一个编辑器的接口，它可以写出优雅的代码；给它一个浏览器的接口，它可以获取多元的网页信息。
- 流程尽可能简洁、可复用。如果我们做的 Agent 在完成任务时需要依赖复杂的工作流，那么一定是无法做到通用级别的。流程做的简单和通用，反而有助于解决通用任务，以不变应万变。简而言之，Less Structure, More Intelligence。

从最根本的角度来看，一套 Agent 能够工作的好，并不在于有多少数量的 Agent，而是是否有上述的规划、行动、反思的能力，且有很简单、自然的方式将这些能力串联。接下来的问题就是如何做好这个串联的过程。《思考，快与慢》里面将人脑分为了系统一和系统二，前者适合短期的、即时的、比较简单的工作，而后者适合需要逻辑推理或者大量思考的工作。这是一个很优秀的思考模型，我们可以按照这个思路来做通用 Agent：
1. 系统一即 Executor，专门用来根据指令找到工具，并且执行工具，这个场景不需要太多深度推理。
2. 系统二即 Awarer，用来做环境感知和下一步的决策。当然，这里的决策不仅仅包括最开始的任务拆解，也包括了后续每个小任务执行完之后的反思和下一个步骤具体行为的决策。PS：其它文章也看到，Agent的技术拆分方式：规划，执行。

举个例子，比如我现在要启动一个获取新能源汽车价格的任务：
- （任务拆解）首先系统二 Awarer 开始规划：
  - 第一步搜索网页、第二步查看网页详情、第三步整理 markdown 并输出。
- （选择并执行工具）然后系统一 Executor 拿到每一步指令后调用 Search API 获取搜索结果
- （反思并决策下一步动机）Awarer 看到搜索结果后，决策要进入某一个具体的网址查看详情，告诉 Executor 即将要前往拿到页面详情。
- （选择并执行工具）Executor 判断要调用浏览器工具，并进入某个具体的网页查看详情。
-  ...
- 结果：产出完善的价格分析报告

加上工程的各个环节之后，涉及的细节相较而言会更复杂。
1. 当用户输入消息之后，消息会被推入一个核心的管理对象 EventStream 当中。随后，Awarer、Executor、User 三者的消息互通全部通过 EventStream 完成
2. Awarer 最先拿到消息，然后对任务进行拆解，规划出诸多的 Plan Tasks。后续的所有操作，全部基于这些 Tasks 展开。Awarer 的输出结构如下：
  ```
  {
    "plan": [后续的任务规划],
    "status": [下一步的动作],
    "step": [下一步到达哪个步骤],
    "reflection": [基于已有环境的反思]
  }
  ```
3. 接下来，Executor 获取其 Awarer 的 status 信息，从丰富的工具库中选择工具进行执行。执行完的结果，推入 EventStream 中，被 Awarer 获取。
4. Aware 随后进行一次思考，同样返回上面的输出结构，但不同的是，在已有 plan 的情况下，Aware 不会重复输出 plan 的内容了，因此后续的输出也会更加简洁。
就这样，Awarer、Executor 达成了非常默契的持续交互。所有环境相关的信息，都放在 EventStream 中统一进行管理，Awarer、Executor 不需要直接通信，从 EventStream 推入或者拿取上下文即可。在 Aware 架构运行的流程中，无论任何时候，都允许用户的介入，且让用户获得整个 Agent 系统的立即反馈。用户可以将自己发的消息直接推入 EventStream，Agent 运行时系统会自动中断任何正在进行的请求和流程，然后要求 Awarer 根据用户的新输入重新规划下一步的行动，从而实现 Human in the loop 的效果。

### 规划能力的存在形态

假设已经有了一个Agent class，一般有以下成员
1. name
2. llm + instruction。llm 一般具备functioncall 能力
2. tools

规划能力如何做
1. 有一个专门的plan llm 成员，此时instruction 也需要为plan 特化。
2. 整个系统以multiagent 来支持，plan 是一个专门的agent。
3. plan 动作作为一个tool  [如何让 Agent 规划调用工具](https://mp.weixin.qq.com/s/CpdXBPTmRZOmTWutywgw3A)，此时agent配套的instruction
    ```
    <instruction>
    1. 你是一个 agent，请持续调用工具直至完美完成用户的任务，停止调用工具后，系统会自动交还控制权给用户。请只有在确定问题已解决后才终止调用工具。
    2. 请善加利用你的工具收集相关信息，绝对不要猜测或编造答案。
    3. 「思考和规划」是一个系统工具，在每次调用其他任务工具之前，你必须**首先调用思考和规划工具**：针对用户的任务详细思考和规划，并对之前工具调用的结果进行深入反思（如有），输出的顺序是thought, plan, action, thoughtNumber。
    - 「思考和规划」工具不会获取新信息或更改数据库，只会将你的想法保存到记忆中。
    - 思考完成之后不需要等待工具返回，你可以继续调用其他任务工具，你一次可以调用多个任务工具。
    - 任务工具调用完成之后，你可以停止输出，系统会把工具调用结果给你，你必须再次调用思考和规划工具，然后继续调用任务工具，如此循环，直到完成用户的任务。
    </instruction>
    ```
    PS： 这个方案的好处是，12中的plan 能力是非标的，进而导致使用它的agent 也是非标的，进而不方便替换。

### 标准化工作

[Agent 框架协议“三部曲”：MCP、A2A、AG-UI](https://mp.weixin.qq.com/s/WYQWJJ8w-29-j5FcQndGng) PS: 标准化工作是推进框架必不可少的。

## 多 Agent 系统

OpenAI 在报告中指出：对于复杂工作流，将提示词和工具分配给多个 Agent 可以提高性能和扩展性。当你的 Agent 无法遵循复杂指令或持续选择错误工具时，可能需要进一步拆分系统并引入更多独立 Agent。**多 Agent 系统的关键在于通信机制**。再次强调，构建 Agent 的难点在于为大模型提供正确上下文。Agent 间的通信方式至关重要。
1. 交接(handoffs)是一种方式——这是 Agents SDK 中我相当欣赏的一个 Agent 抽象。
2. 但有时最佳通信方式可能是工作流。将其中的大模型调用替换为 Agent。这种工作流与 Agent 的混合往往能提供最佳可靠性。


## 框架示例

[开发AI Agent到底用什么框架——LangGraph VS. LlamaIndex](https://mp.weixin.qq.com/s/fdVnkJOGkaXsxkMC1pSiCw)为什么LangGraph和LlamaIndex都基本上能做到用一个统一的底层编排系统来支持各种自主程度不同的Agentic System？（workflow和agent）不同程度的自主性，本质意味着什么？**这个本质在于系统编排的执行路径是在何时决策的**。静态编排的执行路径是完全提前确定好的，不具有太多自主性。自主性必然要求某种程度的动态编排特性才能支持。LangGraph和LlamaIndex的编排系统都有一个重要特性：能够在执行过程中动态地改变节点偏序关系。在LangGraph中，这是通过节点在superstep末尾发送动态消息做到的，而在LlamaIndex中则是通过每个step节点动态地发送事件做到的。换句话说，它们都具有动态编排的特性。对比一下，类似Dify那样，在程序执行之前就可视化地编排出整个Workflow的系统，对于自主系统的支持则是非常有限的。当深入到这个抽象层次上来看，LangGraph和LlamaIndex的编排系统，虽然它们暴露的编程接口、实现的完备程度、对于概念的抽象都完全不同，但最最底层的逻辑又是殊途同归的。

### langchain使用

在chain中，操作序列是硬编码的。智能体通过将LLM与动作列表结合，自动选择最佳动作序列，从而实现自动化决策和行动。Agent在LangChain框架中负责决策制定以及工具组的串联，可以根据用户的输入决定调用哪个工具。通过精心制定的提示，我们能够赋予代理特定的身份、专业知识、行为方式和目标。提示策略为 Agent 提供了预设模板，结合关键的指示、情境和参数来得到 Agent 所需的响应。具体的说，Agent就是将大模型进行封装来简化用户使用，根据用户的输入，理解用户的相应意图，通过action字段选用对应的Tool，并将action_input作为Tool的入参，来处理用户的请求。当我们不清楚用户意图的时候，由Agent来决定使用哪些工具实现用户的需求。

自定义tool 实现

```python
from langchain.tools import BaseTool

# 天气查询工具 ，无论查询什么都返回Sunny
class WeatherTool(BaseTool):
    name = "Weather"
    description = "useful for When you want to know about the weather"
    def _run(self, query: str) -> str:
        return "Sunny^_^"
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")

# 计算工具，暂且写死返回3
class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math."
    def _run(self, query: str) -> str:
        return "3"
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("BingSearchRun does not support async")

```

```python
# 这里使用OpenAI temperature=0，temperature越大表示灵活度越高，输出的格式可能越不满足我们规定的输出格式，因此此处设置为0
llm = OpenAI(temperature=0)
tools = [WeatherTool(), CalculatorTool()]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("Query the weather of this week,And How old will I be in ten years? This year I am 28")
```

```
# 执行结果
I need to use two different tools to answer this question
Action: Weather
Action Input: This week
Observation: Sunny^_^
Thought: I need to use a calculator to answer the second part of the question
Action: Calculator
Action Input: 28 + 10
Observation: 3
Thought: I now know the final answer
Final Answer: This week will be sunny and in ten years I will be 38.
```

LangChain Agent中，内部是一套问题模板(langchain-ai/langchain/libs/langchain/langchain/agents/chat/prompt.py)：

```
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""
```

**这个提示词就是 Agent 之所以能够趋动大模型，进行思考 - 行动 - 观察行动结果 - 再思考 - 再行动 - 再观察这个循环的核心秘密**。有了这样的提示词，模型就会不停地思考、行动，直到模型判断出问题已经解决，给出最终答案，跳出循环。

通过这个模板，加上我们的问题以及自定义的工具，会变成下面这个样子（# 后面是增加的注释）

```
Answer the following questions as best you can.  You have access to the following tools: #  尽可能的去回答以下问题，你可以使用以下的工具：

Calculator: Useful for when you need to answer questions about math.
 # 计算器：当你需要回答数学计算的时候可以用到
Weather: useful for When you want to know about the weather #  天气：当你想知道天气相关的问题时可以用到
Use the following format: # 请使用以下格式(回答)

Question: the input question you must answer #  你必须回答输入的问题
Thought: you should always think about what to do
 # 你应该一直保持思考，思考要怎么解决问题
Action: the action to take, should be one of [Calculator, Weather] #  你应该采取[计算器,天气]之一
Action Input: the input to the action #  动作的输入
Observation: the result of the action # 动作的结果
...  (this Thought/Action/Action Input/Observation can repeat N times) # 思考-行动-输入-输出 的循环可以重复N次
Thought: I now know the final answer # 最后，你应该知道最终结果了
Final Answer: the final answer to the original input question # 针对于原始问题，输出最终结果


Begin! # 开始
Question: Query the weather of this week,And How old will I be in ten years?  This year I am 28 #  问输入的问题
Thought:
```

我们首先告诉 LLM 它可以使用的工具，在此之后，定义了一个**示例格式**，它遵循 Question（来自用户）、Thought（思考）、Action（动作）、Action Input（动作输入）、Observation（观察结果）的流程 - 并重复这个流程直到达到 Final Answer（最终答案）。如果仅仅是这样，openai会完全补完你的回答，中间无法插入任何内容。因此LangChain使用OpenAI的stop参数，截断了AI当前对话。`"stop": ["\nObservation: ", "\n\tObservation: "]`。做了以上设定以后，OpenAI仅仅会给到Action和 Action Input两个内容就被stop停止。以下是OpenAI的响应内容：
```
I need to use the weather tool to answer the first part of the question, and the calculator to answer the second part.
Action: Weather
Action Input: This week
```
这里从Tools中找到name=Weather的工具，然后再将This Week传入方法。具体业务处理看详细情况。这里仅返回Sunny。
由于当前找到了Action和Action Input。 代表OpenAI认定当前任务链并没有结束。因此向tool请求后拼接结果：Observation: Sunny 并且让他再次思考Thought。开启第二轮思考：下面是再次请求的完整请求体:
```
Answer the following questions as best you can. You have access to the following tools:

Calculator: Useful for when you need to answer questions about math.
Weather: useful for When you want to know about the weather


Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Calculator, Weather]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: Query the weather of this week,And How old will I be in ten years? This year I am 28
Thought: I need to use the weather tool to answer the first part of the question, and the calculator to answer the second part.
Action: Weather
Action Input: This week
Observation: Sunny^_^
Thought:
```
同第一轮一样，OpenAI再次进行思考，并且返回Action 和 Action Input 后，再次被早停。
```
I need to calculate my age in ten years
Action: Calculator
Action Input: 28 + 10
```
由于计算器工具只会返回3，结果会拼接出一个错误的结果，构造成了一个新的请求体进行第三轮请求：
```
Answer the following questions as best you can. You have access to the following tools:

Calculator: Useful for when you need to answer questions about math.
Weather: useful for When you want to know about the weather


Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Calculator, Weather]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: Query the weather of this week,And How old will I be in ten years? This year I am 28
Thought: I need to use the weather tool to answer the first part of the question, and the calculator to answer the second part.
Action: Weather
Action Input: This week
Observation: Sunny^_^
Thought:I need to calculate my age in ten years
Action: Calculator
Action Input: 28 + 10
Observation: 38
Thought:
```
此时两个问题全都拿到了结果，根据开头的限定，OpenAi在完全拿到结果以后会返回I now know the final answer。并且根据完整上下文。把多个结果进行归纳总结：下面是完整的相应结果：
```
I now know the final answer
Final Answer: I will be 38 in ten years and the weather this week is sunny.
```
可以看到。ai严格的按照设定返回想要的内容，并且还以外的把28+10=3这个数学错误给改正了。通过 `verbose=True` 可以动态查看上述过程。 PS: 通过prompt 引导llm 进行文字接龙，通过解析文字接龙来进行tool 的调用。

根据输出再回头看agent的官方解释：An Agent is a wrapper around a model, which takes in user input and returns a response corresponding to an “action” to take and a corresponding “action input”. **本质上是通过和大模型的多轮对话交互来实现的**（对比常规聊天时的一问一答/单轮对话）， 不断重复“Action+ Input -> 结果 -> 下一个想法”，一直到找到最终答案。通过特定的提示词引导LLM模型以固定格式来回复，LLM模型回复完毕后，解析回复，这样就获得了要执行哪个tool，以及tool的参数。然后就可以去调tool了，调完把结果拼到prompt中，然后再让LLM模型根据调用结果去总结并回答用户的问题。


大多数 Agent 主要是在某种循环中运行 LLM。目前，我们使用的唯一方法是 AgentExecutor。我们为 AgentExecutor 添加了许多参数和功能，但它仍然只是运行循环的一种方式。langgraph是一个新的库，旨在创建语言 Agent 的图形表示。这将使用户能够创建更加定制化的循环行为。用户可以定义明确的规划步骤、反思步骤，或者轻松设置优先调用某个特定工具。

### agno

[agno](https://github.com/agno-agi/agno)What are Agents? Agents are AI programs that execute tasks autonomously. They solve problems by running tools, accessing knowledge and memory to improve responses. Unlike traditional programs that follow a predefined execution path, agents dynamically adapt their approach based on context, knowledge and tool results.

Instead of a rigid binary definition, let's think of Agents in terms of agency and autonomy.
1. Level 0: Agents with no tools (basic inference tasks).
2. Level 1: Agents with tools for autonomous task execution.
3. Level 2: Agents with knowledge, combining memory and reasoning.
4. Level 3: Teams of specialized agents collaborating on complex workflows.
PS: llm 根据kb、memory来执行tool（相对传统结构化编程 if else while 明确过程来说）

从框架设计思想上来说
1. 几个基本组件都做了抽象，比如Document/Embedder/AgentKnowledge(知识库)/AgentMemory/Model(模型)/Reranker/Toolkit/VectorDb。PS：一般业务开发时，langchain 或llamaindex 的类似抽象也很难完全满足需要，往往要独立的提出这些组件的抽象，此时就是一个很好的参考。 
2. 只有一个Agent class，持有了几乎所有可能需要的组件，Agent.run 用一个最复杂的流程 来兼容L0到L3。PS：这也导致我们很难对链路做个性化调整，但确实是很多开源框架的思路。具体细节上有很多值得参考的地方，比如流式吐字、多模态、推理模型等。

||手写Agent|使用框架|
|---|---|---|
|入口|Agent.run|Workflow.run|
|逻辑串联|代码手撸，父类定义几个抽象方法，做好编排，子类做个性化实现|使用langgraph或llamaindex workflow 来做逻辑串联|
|步骤定义|类的方法|workflow类的方法|
|状态共享|类的成员|workflow.context, workflow类成员|
|推理的上下文|memory|memory，之前笔者不太熟的时候，会用current_reasonngs 来记录|

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are a Thai cuisine expert!",
    instructions=[
        "Search your knowledge base for Thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results."
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipes",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)
agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
```

### langchain源码

Agent 可以看做在Chain的基础上，进一步整合Tool 的高级模块。与Chain 相比，Agent 具有两个新增的能力：思考链和工具箱。**能根据环境变化更新计划，使决策更加健壮**。思考链释放了大模型的规划和调度潜能，是Agent的关键创新，Agent 定义了工具的标准接口，以实现无缝集成。与Chain 直接调用模块/Runnable相比，它只关心tool 输入和输出，tool内部实现对Agent 透明，工具箱大大扩展了Agent的外部知识来源，使其离真正的通用智能更近一步。


LangChain关键组件
1. 代理（Agent）：这个类决定下一步执行什么操作。它由一个语言模型和一个提示（prompt）驱动。提示可能包含代理的性格（也就是给它分配角色，让它以特定方式进行响应）、任务的背景（用于给它提供更多任务类型的上下文）以及用于激发更好推理能力的提示策略（例如 ReAct）。LangChain 中包含很多种不同类型的代理。PS： **一般情况下，一个Agent 像Chain一样，都会对应一个prompt**。
2. 工具（Tools）：工具是代理调用的函数。这里有两个重要的考虑因素：一是让代理能访问到正确的工具，二是以最有帮助的方式描述这些工具。如果你没有给代理提供正确的工具，它将无法完成任务。如果你没有正确地描述工具，代理将不知道如何使用它们。LangChain 提供了一系列的工具，同时你也可以定义自己的工具，或者将函数转换为tool。
    ```python
    class BaseTool(RunnableSerializable[Union[str, Dict], Any]):
        name: str
        description: str
        def invoke(self, input: Union[str, Dict],config: Optional[RunnableConfig] = None,**kwargs: Any,) -> Any:
            ...
            return self.run(...)
    class Tool(BaseTool):
        description: str = ""
        func: Optional[Callable[..., str]]
        coroutine: Optional[Callable[..., Awaitable[str]]] = None
    ```
3. 代理执行器（AgentExecutor）：代理执行器是代理的运行环境，它调用代理并执行代理选择的操作。执行器也负责处理多种复杂情况，包括处理代理选择了不存在的工具的情况、处理工具出错的情况、处理代理产生的无法解析成工具调用的输出的情况，以及在代理决策和工具调用进行观察和日志记录。AgentExecuter负责迭代运行Agent，**直至满足设定的停止条件**，这使得Agent能够像生物一样循环处理信息和任务。

AgentExecutor由一个Agent和Tool的集合组成。AgentExecutor负责调用Agent，获取返回（callback）、action和action_input，并根据意图将action_input给到具体调用的Tool，获取Tool的输出，并将所有的信息传递回Agent，以便猜测出下一步需要执行的操作。`AgentExecutor.run也就是chain.run ==> AgentExecutor/chain.__call__ ==> AgentExecutor._call()` 和逻辑是 _call 方法，核心是 `output = agent.plan(); tool=xx(output); observation = tool.run(); 

```python
def initialize_agent(tools,llm,...)-> AgentExecutor:
    agent_obj = agent_cls.from_llm_and_tools(llm, tools, callback_manager=callback_manager, **agent_kwargs)
    AgentExecutor.from_agent_and_tools(agent=agent_obj, tools=tools,...)
    return cls(agent=agent, tools=tools, callback_manager=callback_manager, **kwargs)
# AgentExecutor 实际上是一个 Chain，可以通过 .run() 或者 _call() 来调用
class AgentExecutor(Chain):
    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]
    tools: Sequence[BaseTool]
    """Whether to return the agent's trajectory of intermediate steps at the end in addition to the final output."""
    max_iterations: Optional[int] = 15
    def _call(self,inputs: Dict[str, str],...) -> Dict[str, Any]:
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(name_to_tool_map,inputs,intermediate_steps,...)
            # 返回的数据是一个AgentFinish类型，表示COT认为不需要继续思考，当前结果就是最终结果，直接将结果返回给用户即可；
            if isinstance(next_step_output, AgentFinish):
                return self._return(next_step_output, intermediate_steps, run_manager=run_manager)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(tool_return, intermediate_steps, run_manager=run_manager)
            iterations += 1
            time_elapsed = time.time() - start_time
        return self._return(output, intermediate_steps, run_manager=run_manager)          
    def _take_next_step(...):
        # 调用LLM决定下一步的执行逻辑
        output = self.agent.plan(intermediate_steps,**inputs,...)
        if isinstance(output, AgentFinish): # 如果返回结果是AgentFinish就直接返回
            return output
        if isinstance(output, AgentAction): # 如果返回结果是AgentAction，就根据action调用配置的tool
            actions = [output]
        result = []
        for agent_action in actions:
            tool = name_to_tool_map[agent_action.tool]
            observation = tool.run(agent_action.tool_input,...)
            result.append((agent_action, observation))  # 调用LLM返回的AgentAction和调用tool返回的结果（Obversation）一起加入到结果中
        return result
```

![](/public/upload/machine/agent_executor_run.png)

Agent.plan() 可以看做两步：
1. 将各种异构的历史信息转换成 inputs，传入到 LLM 当中；
2. 根据 LLM 生成的反馈，采取决策。LLM 生成的回复是 string 格式，langchain 中ZeroShotAgent 通过字符串匹配的方式来识别 action。
因此，agent 能否正常运行，与 prompt 格式，以及 LLM 的 ICL 以及 alignment 能力有着很大的关系。
   1. LangChain主要是基于GPT系列框架进行设计，其适用的Prompt不代表其他大模型也能有相同表现，所以如果要自己更换不同的大模型(如：文心一言，通义千问...等)。则很有可能底层prompt都需要跟著微调。
   2. 在实际应用中，我们很常定期使用用户反馈的bad cases持续迭代模型，但是Prompt Engeering的工程是非常难进行的微调的，往往多跟少一句话对于效果影响巨大，因此这类型产品达到80分是很容易的，但是要持续迭代到90分甚至更高基本上是很难的。

```python
# 一个 Agent 单元负责执行一次任务
class Agent(...):
    llm_chain: LLMChain
    allowed_tools: Optional[List[str]] = None
    # agent 的执行功能在于 Agent.plan()
    def plan(self,intermediate_steps: List[Tuple[AgentAction, str]],callbacks: Callbacks = None,**kwargs: Any,) -> Union[AgentAction, AgentFinish]:
        #  # 将各种异构的历史信息转换成 inputs，传入到 LLM 当中
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        # 根据 LLM 生成的反馈，采取决策
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs) 
        # full_output 是纯文本，通过断点调试可以看到，真的就是靠正则表达式提取tool的名称
        # 最后的输出 AgentAction 中会包括：需要使用的 tool，使用该 tool 时候，对应的执行命令。
        return self.output_parser.parse(full_output)
```

有人希望通过一些开源的  LLM 来实现 ReAct Agent，但实际开发过程中会发现开源低参数（比如一些 6B、7B 的 LLM）的 LLM  对于提示词的理解会非常差，根本不会按照提示词模板的格式来输出（例如不按`Action: xx Action Input: xx`返回），这样就会导致我们的 Agent 无法正常工作，所以如果想要实现一个好的  Agent，还是需要使用好的 LLM，目前看来使用gpt-3.5模型是最低要求。



## 其它

[Windsurf团队关于Agent的认知，相当精彩](https://mp.weixin.qq.com/s/0HHW0bouQ3ZAr5kFiNld4A)“有价值的问题” 与 “技术是否已经足够可靠” 之间的交集，协作式 Agent（AI flows） 方法所需的稳健性门槛要远低于完全自主的 Agent 方法。 Agent 系统所面临的普遍挑战：**虽然 Agent 系统代表着未来的发展方向，但今天的 LLM 可能还不具备足够的能力，在没有任何人类参与或纠正的情况下，从头到尾完成这些复杂任务**。现实促使人们开始采取一种新的 Agent 系统方法，这种方法基于一个共识：在人类与 Agent 之间，应该存在**一种合理的任务分配平衡**。
1. 需要有清晰的方式让人类在流程执行过程中观察它的行为，这样一旦流程偏离预期，人类可以尽早进行纠正。可以被要求批准 AI 的某些操作（例如执行终端命令）。这些流程必须运行在与人类进行实际工作的相同环境中。换句话说，在这个现实中，让人类观察 Agent 在做什么很重要，而让代 Agent 观察人类在做什么也同样重要。PS：与之对应的是 Agent在没有人类参与的情况下在后台运行

我们花这么大篇幅来区分 “自主 Agent” 和 “协作式 Agent”，是因为这两种方式在构建 Agent 系统时，其实是截然不同的路径。它们涉及到完全不同程度的人机协作、不同程度的信任需求、不同的交互方式等等。



