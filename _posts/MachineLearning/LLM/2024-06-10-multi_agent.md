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

OpenAI 是以 AGI 为愿景的公司，现在的 Agent 在一定程度上可以看作 AGI 的代偿 —— 除工具使用外，规划和记忆本是 LLM 所应覆盖的范畴。如果 LLM 自身能力增强，一切还可能重新洗牌。不过在那之前，Retrieve 和 Task Decomposition 应该会长期把持 LLM 显学的位置。Agent 是角色和目标的承载，LLMs、Plans、Memory 和 Tools 服务于角色扮演和目标实现。那么，自然的，服务于相同或相关目标时，多个 Agent 之间可以共享 thread context，但需要保持自身权限的独立，即 Multi-Agent。

[Multi-Agent ，知多少？](https://mp.weixin.qq.com/s/Z970vS3mGA20YOoqIQMOdw)
1.  Single-Agent 系统的知识获取和认知范畴高度依赖于其训练数据集和模型算法，这使得它难以全面把握多元异构的信息要素，以及复杂环境中瞬息万变的细微变化。 Single-Agent 很容易产生知识盲区和认知偏差，从而在面临新的情景时无法作出前瞻性的正确决策，导致决策失误。
2. 即便是当下最先进的 Single-Agent ，其可用的算力资源和计算能力在物理层面仍有明确的上限，无法做到无限扩展。一旦面临极其错综复杂、计算量密集的任务， Single-Agent 无疑会遭遇算力瓶颈，无法高效完成处理，性能将大打折扣。Agent一般都通过XoT (CoT、ToT、GoT) + React等方法来规划和思考，上下文会不断的加长，迟早会突破窗口限制。因此拆分Agent的功能避免超过上下文窗口限制是一个很有效的方法。 而且，Prompt是Agent工作的中很关键的因素，单一的Agent如果维护的大量的上下文，难免"脑子"会乱。如果只执行特定的任务，掌握特定技能，当然理论上表现会更好。
3. Single-Agent 系统本质上是一种集中式的架构模式，这决定了它存在着极高的故障风险。一旦核心代理发生故障或遭受攻击，整个系统将完全瘫痪，难以继续运转，缺乏有效的容错和备份机制，无法确保关键任务的连续性和可靠性。
4. 复杂环境下的决策往往需要各种异构智能算法模型的协同配合，而封闭的 Single-Agent 系统难以灵活整合不同AI范式，无法充分挖掘多元异质智能的协同潜能，解决复杂问题的能力相对有限。
5. Single-Agent 系统通常是封闭式的，**新的功能、知识很难被快速注入和更新，整体的可扩展性和可升级性较差**，无法高效适应不断变化的复杂业务需求，存在架构上的先天缺陷。

工作流模式适用于 Bot 技能流程相对固定的场景，在该模式下，Bot 用户的所有对话均会触发固定的工作流处理。**在 AI 比你懂自己之前，场景分流方法将长期有效**。多 Agent 模式通过以下方式来简化复杂的任务场景。
1. 您可以为不同的 Agent 配置独立的提示词，将复杂任务分解为一组简单任务，而不是在一个 Bot 的提示词中设置处理任务所需的所有判断条件和使用限制。
2. 多 Agent 模式允许您为每个 Agent 节点配置独立的插件和工作流。这不仅降低了单个 Agent 的复杂性，还提高了测试 Bot 时 bug 修复的效率和准确性，您只需要修改发生错误的 Agent 配置即可。

在 Multi-Agent 系统架构中，由众多独立自治的智能体代理组成，它们拥有各自独特的领域知识、功能算法和工具资源，可以通过灵活的交互协作，共同完成错综复杂的决策任务。与单一代理系统将所有职责高度集中在一个代理身上不同， Multi-Agent 系统则实现了职责和工作的模块化分工，允许各个代理按照自身的特长和专长，承担不同的子任务角色,进行高度专业化的分工协作。     此外， Multi-Agent 系统具有天然的开放性和可扩展性。当系统面临任务需求的不断扩展和功能的持续迭代时，通过引入新的专门代理就可以无缝扩展和升级整体能力，而无需对现有架构进行大规模的重构改造。这与单一代理系统由于其封闭集中式设计,每次功能扩展都需要对整体架构做根本性的修改形成鲜明对比。 

以rag系统为例
1. 简单的rewrite ==> retrieve ==> generate
2. [rag的尽头是agent](https://mp.weixin.qq.com/s/iZjfHEe2TXCJYPAGQ6beUQ) `rewrite ==> retrieve ==> generate` 可以解决的问题终归有限， 这里涉及到很多花活，比如拆分子问题、联网、ircot等，需要agent 根据当前的已知信息，判断下一步 ==> 行动 ==> 根据观察判断下一步
3. [rag的尽头是multi-agent](https://mp.weixin.qq.com/s/uSHGFKpPzdrJjDL3BZVDWw)  单个agent 可以解决的问题也终归有限，用户的“知识”不只在文档里，也在数据库表里，也是知识图谱里，都编排在一个agent里，对路由器/plan 组件的要求很高，很多时候即便人也无法判断，这时候就要“三个臭皮匠，顶一个诸葛亮”了。
如果LLM能力足够，一个agent 选择tools 就能解决所有问题。**实质上还是LLM 能力有限**（无法较好的拆解问题，制定plan，也无法较好的判断问题结束），针对某一链路、某一场景进行特化，复查场景采取多角色协作方式。

从架构的角度，与 single agent 相比，multi-agent 架构更易于维护扩展。即使是基于 single agent 的接口，使用 multi-agent 的实施架构也可能使系统更加模块化，开发人员更容易添加或删除功能组件。**目前的技术条件下，无法构建出一个满足所有功能的 single agent**，但可以将不同的 Agent 和 LLM 进行组合，构建出一个满足使用要求的 multi-agent。

## 多Agent框架

智能体的发展：从单任务到多代理协同与人代理交互。多智能体应用让不同的Agent之间相互交流沟通来解决问题。

AutoGen、ChatDev、CrewAI [CrewAI：一个集众家所长的MutiAgent框架](https://mp.weixin.qq.com/s/BmXVkCz7Atw0iVZRRYg-3Q)
PS： 你要是上万个tool的话，llm 上下文塞不下，此时让一个llm 针对一个问题决策使用哪一个tool 就很难（ToolLLaMa已经支持16k+了），此时很自然的就需要多层次的Agent，低层次的Agent 更专业聚焦一些。

## AutoGen

AutoGen 代理是可定制的、可对话的，并且无缝地允许人类参与。in AutoGen
1. 收发消息、生成回复：an agent is an entity that can send messages, receive messages and generate a reply using models, tools, human inputs or a mixture of them. This abstraction not only allows agents to model real-world and abstract entities, such as people and algorithms, but it also simplifies implementation of complex workflows as collaboration among agents.
2. 可扩展、可组合：Further, AutoGen is extensible and composable: you can extend a simple agent with customizable components and create workflows that can combine these agents and power a more sophisticated agent, resulting in implementations that are modular and easy to maintain.

一个agent 具备干活儿所需的所有必要资源：An example of such agents is the built-in ConversableAgent which supports the following components:
1. A list of LLMs
2. A code executor
3. A function and tool executor
4. A component for keeping human-in-the-loop
You can switch each component on or off and customize it to suit the need of your application. For advanced users, you can add additional components to the agent by using registered_reply.

### 核心概念

AutoGen先声明了一个Agent的协议(protocol)，规定了作为一个Agent基本属性和行为：
1. name属性：每个Agent必须有一个属性。
2. description属性：每个Agent必须有个自我介绍，描述自己的能干啥和一些行为模式。
3. send方法：发送消息给另一个Agent。
4. receive方法：接收来自另一个代理的消息Agent。
5. generate_reply方法：基于接收到的消息生成回复，也可以同步或异步执行。

```python
class Agent:
    def __init__(self, name: str,):
        self._name = name
    def send(self, message: Union[Dict, str], recipient: "Agent", request_reply: Optional[bool] = None):
        """(Abstract method) Send a message to another agent."""
    def receive(self, message: Union[Dict, str], sender: "Agent", request_reply: Optional[bool] = None):
        """(Abstract method) Receive a message from another agent."""
    def generate_reply(self,messages: Optional[List[Dict]] = None,sender: Optional["Agent"] = None,**kwargs,) -> Union[str, Dict, None]:
        """(Abstract method) Generate a reply based on the received messages."""
class ConversableAgent(Agent):
    def __init__( self,name: str,system_message,...):
        self._oai_messages = ... #  Dict[Agent, List[Dict]] 存储了Agent 发来的消息（也包含自己的发出的消息？）
        
    def register_reply(self,trigger,reply_func,...):
        """Register a reply function.The reply function will be called when the trigger matches the sender."""
    def initiate_chat( self, recipient: "ConversableAgent", clear_history,...)
        """Initiate a chat with the recipient agent."""
    def register_for_llm(self,*,name,description, api_style: Literal["function", "tool"] = "tool",) -> Callable[[F], F]:
       """Decorator factory for registering a function to be used by an agent."""
       # 数据进入到 self.llm_config["functions"] 或 self.llm_config["tools"]
    def register_for_execution( self, name: Optional[str] = None, ) -> Callable[[F], F]:
       """Decorator factory for registering a function to be executed by an agent."""
       # 数据进入到  self._function_map
```
ConversableAgent核心是receive，当ConversableAgent收到一个消息之后，它会调用generate_reply去生成回复，然后调用send把消息回复给指定的接收方。同时ConversableAgent还负责实现对话消息的记录，一般都记录在ChatResult里，还可以生成摘要等。

一个开启llm，关闭code executor、第三方工具、不需要人输入的agent
```python
import os
from autogen import ConversableAgent
agent = ConversableAgent(
    "chatbot",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY")}]},
    code_execution_config=False,  # Turn off code execution, by default it is off.
    function_map=None,  # No registered functions, by default it is None.
    human_input_mode="NEVER",  # Never ask for human input.
)
reply = agent.generate_reply(messages=[{"content": "Tell me a joke.", "role": "user"}])
print(reply)
```

AutoGen在ConversableAgent的基础上实现了AssistantAgent，UserProxyAgent和GroupChatAgent三个比较常用的Agent类，这三个类在ConversableAgent的基础上添加了特定的system_message和description。其实就是添加了一些基本的prompt而已。

一个绑定了角色的Agent.  assign different roles to two agents by setting their system_message.
```python
cathy = ConversableAgent(
    "cathy",
    system_message="Your name is Cathy and you are a part of a duo of comedians.",
    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.9, "api_key": os.environ.get("OPENAI_API_KEY")}]},
    human_input_mode="NEVER",  # Never ask for human input.
)
joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": os.environ.get("OPENAI_API_KEY")}]},
    human_input_mode="NEVER",  # Never ask for human input.
)
# 让两人说一段喜剧show
result = joe.initiate_chat(cathy, message="Cathy, tell me a joke.", max_turns=2)
```

决定Agent的行为模式就有几点关键：
1. 提示词，system_message，就是这个Agent的基本提示词
2. 能调用的工具，取决于给Agent提供了哪些工具，在AssistantAgent默认没有配置任何工具，但是提示它可以生成python代码，交给UserProxyAgent来执行获得返回，相当于变相的拥有了工具
3. 回复逻辑，如果你希望在过程中掺入一点私货而不是完全交给LLM来决定，那么可以在generate_reply的方法中加入一点其他逻辑来控制Agent返回回复的逻辑，例如在生成回复前去查询知识库，参考知识库的内容来生成回复。

### 群聊

除了两个agent 互相对话之外，支持多个agent群聊（Group Chat），An important question in group chat is: What agent should be next to speak? To support different scenarios, we provide different ways to organize agents in a group chat: We support several strategies to select the next agent: round_robin, random, manual (human selection), and auto (Default, using an LLM to decide). Group Chat由GroupChatManager来进行的，GroupChatManager也是一个ConversableAgent的子类。它首选选择一个发言的Agent，然后发言Agent会返回它的响应，然后GroupChatManager会把收到的响应广播给聊天室内的其他Agent，然后再选择下一个发言者。直到对话终止的条件被满足。GroupChatManager选择下一个发言者的方法有：
1. 自动选择: "auto"下一个发言者由LLM来自动选择；
2. 手动选择："manual"由用户输入来决定下一个发言者；
3. 随机选择："random"随机选择下一个发言者；
4. 顺序发言："round_robin" ：根据Agent顺序轮流发言
5. 自定义的函数：通过调用函数来决定下一个发言者；

如果聊天室里Agent比较多，还可以通过设置GroupChat类中设置allowed_or_disallowed_speaker_transitions参数来规定当特定的Agent发言时，下一个候选的Agent都有哪些。例如可以规定AgentB发言时，只有AgentC才可以响应。通过这些发言控制的方法，可以将Agent组成各种不同的拓扑。例如层级化、扁平化。甚至可以通过传入一个图形状来控制发言的顺序。

AutoGen允许在一个群聊中，调用另外一个Agent群聊来执行对话(嵌套对话Nested Chats)。这样做可以把一个群聊封装成单一的Agent，从而实现更加复杂的工作流。

[AutoGen多代理对话项目示例和工作流程分析](https://developer.aliyun.com/article/1394332) 未细读。

## Qwen-Agent（未细读）

## OpenAI-Swarm Multi-Agent

[OpenAI终于open了，Swarm开源来袭~](https://mp.weixin.qq.com/s/PUsQHrDfgiwuhTiolag0tA) 未细读

[OpenAI-Swarm Multi-Agent 框架源码解读](https://mp.weixin.qq.com/s/h9uo509jUDL3uRjaIuq1uA) 未细读。

[初识 OpenAI 的 Swarm：轻量级、多智能体系统的探索利器](https://mp.weixin.qq.com/s/XMMD_19g1CzDUzfeSm2qPQ) 未细读。

## autogen-magentic-one

https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one

## AutoGPT

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

## 未来

[REAPER——一种基于推理的规划器，专为处理复杂查询的高效检索而设计](https://zhuanlan.zhihu.com/p/712032784)