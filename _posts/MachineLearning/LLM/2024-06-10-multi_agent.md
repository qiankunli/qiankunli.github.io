---

layout: post
title: Multi-Agent探索
category: 技术
tags: MachineLearning
keywords: langchain langgraph lcel

---

* TOC
{:toc}


##  Single-Agent  面临的困境 

[Multi-Agent ，知多少？](https://mp.weixin.qq.com/s/Z970vS3mGA20YOoqIQMOdw)
1.  Single-Agent 系统的知识获取和认知范畴高度依赖于其训练数据集和模型算法，这使得它难以全面把握多元异构的信息要素，以及复杂环境中瞬息万变的细微变化。 Single-Agent 很容易产生知识盲区和认知偏差，从而在面临新的情景时无法作出前瞻性的正确决策，导致决策失误。
2. 即便是当下最先进的 Single-Agent ，其可用的算力资源和计算能力在物理层面仍有明确的上限，无法做到无限扩展。一旦面临极其错综复杂、计算量密集的任务， Single-Agent 无疑会遭遇算力瓶颈，无法高效完成处理，性能将大打折扣。
3. Single-Agent 系统本质上是一种集中式的架构模式，这决定了它存在着极高的故障风险。一旦核心代理发生故障或遭受攻击，整个系统将完全瘫痪，难以继续运转，缺乏有效的容错和备份机制，无法确保关键任务的连续性和可靠性。
4. 复杂环境下的决策往往需要各种异构智能算法模型的协同配合，而封闭的 Single-Agent 系统难以灵活整合不同AI范式，无法充分挖掘多元异质智能的协同潜能，解决复杂问题的能力相对有限。
5. Single-Agent 系统通常是封闭式的，**新的功能、知识很难被快速注入和更新，整体的可扩展性和可升级性较差**，无法高效适应不断变化的复杂业务需求，存在架构上的先天缺陷。

在 Multi-Agent 系统架构中，由众多独立自治的智能体代理组成，它们拥有各自独特的领域知识、功能算法和工具资源，可以通过灵活的交互协作，共同完成错综复杂的决策任务。与单一代理系统将所有职责高度集中在一个代理身上不同， Multi-Agent 系统则实现了职责和工作的模块化分工，允许各个代理按照自身的特长和专长，承担不同的子任务角色,进行高度专业化的分工协作。     此外， Multi-Agent 系统具有天然的开放性和可扩展性。当系统面临任务需求的不断扩展和功能的持续迭代时，通过引入新的专门代理就可以无缝扩展和升级整体能力，而无需对现有架构进行大规模的重构改造。这与单一代理系统由于其封闭集中式设计,每次功能扩展都需要对整体架构做根本性的修改形成鲜明对比。 

以rag系统为例
1. 简单的rewrite ==> retrieve ==> generate
2. [rag的尽头是agent](https://mp.weixin.qq.com/s/iZjfHEe2TXCJYPAGQ6beUQ) `rewrite ==> retrieve ==> generate` 可以解决的问题终归有限， 这里涉及到很多花活，比如拆分子问题、联网、ircot等，需要agent 根据当前的已知信息，判断下一步 ==> 行动 ==> 根据观察判断下一步
3. [rag的尽头是multi-agent](https://mp.weixin.qq.com/s/uSHGFKpPzdrJjDL3BZVDWw)  单个agent 可以解决的问题也终归有限，用户的“知识”不只在文档里，也在数据库表里，也是知识图谱里，都编排在一个agent里，对路由器/plan 组件的要求很高，很多时候即便人也无法判断，这时候就要“三个臭皮匠，顶一个诸葛亮”了。

从架构的角度，与 single agent 相比，multi-agent 架构更易于维护扩展。即使是基于 single agent 的接口，使用 multi-agent 的实施架构也可能使系统更加模块化，开发人员更容易添加或删除功能组件。**目前的技术条件下，无法构建出一个满足所有功能的 single agent**，但可以将不同的 Agent 和 LLM 进行组合，构建出一个满足使用要求的 multi-agent。

## 多Agent框架

智能体的发展：从单任务到多代理协同与人代理交互。

AutoGen、ChatDev、CrewAI [CrewAI：一个集众家所长的MutiAgent框架](https://mp.weixin.qq.com/s/BmXVkCz7Atw0iVZRRYg-3Q)
PS： 你要是上万个tool的话，llm 上下文塞不下，此时让一个llm 针对一个问题决策使用哪一个tool 就很难（ToolLLaMa已经支持16k+了），此时很自然的就需要多层次的Agent，低层次的Agent 更专业聚焦一些。

### AutoGen

### AutoGPT

Andrej Karpathy 在 2017 年提出的 Software 2.0：基于神经网络的软件设计。真的很有前瞻性了。这进一步引出了当前正在迅速发展的 Agent Ecosystem。AutoGPT ，BabyAGI 和 HuggingGPT 这些项目形象生动地为我们展示了 LLM 的潜力除了在生成内容、故事、论文等方面，它还具有强大的通用问题解决能力。如果说 ChatGPT 本身的突破体现在人们意识到**语言可以成为一种服务**，成为人和机器之间最自然的沟通接口，这一轮新发展的关键在于人们意识到语言（不一定是自然语言，也包括命令、代码、错误信息）也是模型和自身、模型和模型以及模型和外部世界之间最自然的接口，让 AI agent 在思考和表达之外增加了调度、结果反馈和自我修正这些新的功能模块。于是**在人类用自然语言给 AI 定义任务目标（只有这一步在实质上需要人类参与）之后可以形成一个自动运行的循环**：
1. agent 通过语言思考，分解目标为子任务
2. agent 检讨自己的计划
3. agent 通过代码语言把子任务分配给别的模型，或者分配给第三方服务，或者分配给自己来执行
4. agent 观察执行的结果，根据结果思考下一步计划，回到循环开始

原生的 ChatGPT 让人和 AI 交流成为可能，相当于数学归纳法里 n=0 那一步。而新的 agent ecosystem 实现的是 AI 和自己或者其他 AI 或者外部世界交流，相当于数学归纳法里从 n 推出 n+1 那一步，于是新的维度被展开了。比如将机器人强大的机械控制能力和目前 GPT-4 的推理与多模态能力结合，也许科幻小说中的机器人将在不久成为现实。

与Chains依赖人脑思考并固化推理过程的做法不同，AutoGPT是一个基于GPT-4语言模型的、实验性的开源应用程序，**可以根据用户给定的目标，自动生成所需的提示**，并执行多步骤的项目，无需人类的干预和指导（自己给自己提示）。AutoGPT的本质是一个自主的AI代理，可以利用互联网、记忆、文件等资源，来实现各种类型和领域的任务。这意味着它可以扫描互联网或执行用户计算机能够执行的任何命令，然后将其返回给GPT-4，以判断它是否正确以及接下来要做什么。下面举一个简单的例子，来说明AutoGPT的运行流程。假设我们想让AutoGPT帮我们写一篇关于太空的文章，我们可以给它这样的一个目标：“写一篇关于太空的文章”。然后AutoGPT会开始运行，它会这样做：

1. AutoGPT会先在PINECONE里面查找有没有已经写好的关于太空的文章，如果有，它就会直接把文章展示给我们，如果没有，它就会继续下一步。
2. AutoGPT会用GPT-4来生成一个提示，比如说：“太空是什么？”，然后用GPT-4来回答这个提示，比如说：“太空是指地球大气层之外的空间，它包含了许多星球，卫星，彗星，小行星等天体。”
3. AutoGPT会把生成的提示和回答都存储在PINECONE里面，并且用它们来作为文章的第一段。
4. AutoGPT会继续用GPT-4来生成新的提示，比如说：“太空有什么特点？”，然后用GPT-4来回答这个提示，比如说：“太空有很多特点，比如说，太空没有空气，没有重力，没有声音，温度变化很大等等。”
5. AutoGPT会把生成的提示和回答都存储在PINECONE里面，并且用它们来作为文章的第二段。
6. AutoGPT会重复这个过程，直到它觉得文章已经足够长或者足够完整了，或者达到了一定的字数限制或者时间限制。
7. AutoGPT会把最终生成的文章展示给我们，并且询问我们是否满意。如果我们满意，它就会结束运行；如果我们不满意，它就会根据我们的反馈来修改或者补充文章。

```python
agent = SimpleAgent.from_workspace(agent_workspace, client_logger)
print("agent is loaded")
plan = await agent.build_initial_plan()
print(parse_agent_plan(plan))
while True:
    current_task, next_ability = await agent.determine_next_ability(plan)
    print(parse_next_ability(current_task, next_ability)
    user_input = click.prompt("Should the agent proceed with this ability?", default = "y")
    ability_result = await agent.execute_next_ability(user_input)
    print(parse_ability_result(ability_result))
```

[探索AI时代的应用工程化架构演进，一人公司时代还有多远？](https://mp.weixin.qq.com/s/xgdMbYv__YNKFJ2n7yMDBQ)在冯诺依曼架构或者哈佛架构设备的实际开发中，我们会去关心如何使用相应协议去寻址去读写总线操作不同设备，如UART、I2C、SPI总线协议，这都是我们要学习掌握的，但我们基本不会关心CPU中的CU、ALU等单元。计算机架构这样求同存异的继续发展下去，将这些单元与高速存储及总线等作为抽象概念去进一步封装。而AI应用也是类似的，Agent会将相关的规划反思改进能力不断的作为自身的核心能力封装。因此，对于未来的AI应用极有可能不是在传统计算机上运行的程序，而是标准化的需求，在以规划能力专精的Agent 大模型作为CPU的AI计算机虚拟实例上直接运行的，而我们今天所谈论的应用架构，也会沉积到底层转变为AI计算机的核心架构。最终AI计算机将图灵完备，通过AI的自举将迭代产物从工程领域提升到工业领域。

![](/public/upload/machine/llm_agent.jpg)

## 应用

### 用于RAG 

https://github.com/padas-lab-de/PersonaRAG 未读