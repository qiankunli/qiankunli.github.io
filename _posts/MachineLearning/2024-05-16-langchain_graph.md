---

layout: post
title: LLM工作流编排
category: 技术
tags: MachineLearning
keywords: langchain langgraph lcel

---

* TOC
{:toc}

## 不再是简单的顺序调用

从顺序式为主的简单架构走向复杂的WorkFlow，推理阶段的RAG Flow分成四种主要的基础模式：顺序、条件、分支与循环。PS： 一个llm 业务有各种基本概念，prompt/llm/memory，整个工作流产出一个流式输出，处理链路上包含多个step，且step有复杂的关系（顺序、条件、分支与循环）。一个llm 业务开发的核心就是个性化各种原子能力 以及组合各种原子能力。

以一个RAG Agent 的工作流程为例
1. 根据问题，路由器决定是从向量存储中检索上下文还是进行网页搜索。
2. 如果路由器决定将问题定向到向量存储以进行检索，则从向量存储中检索匹配的文档；否则，使用 tavily-api 进行网页搜索。
3. 文档评分器然后将文档评分为相关或不相关。
4. 如果检索到的上下文被评为相关，则使用幻觉评分器检查是否存在幻觉。如果评分器决定响应缺乏幻觉，则将响应呈现给用户。
5. 如果上下文被评为不相关，则进行网页搜索以检索内容。
6. 检索后，文档评分器对从网页搜索生成的内容进行评分。如果发现相关，则使用 LLM 进行综合，然后呈现响应。

## LCEL 

[langchain入门3-LCEL核心源码速通](https://juejin.cn/post/7328204968636252198)LCEL实际上是langchain定义的一种DSL，可以方便的将一系列的节点按声明的顺序连接起来，实现固定流程的workflow编排。LCEL语法的核心思想是：一切皆为对象，一切皆为链。这意味着，LCEL语法中的每一个对象都实现了一个统一的接口：Runnable，它定义了一系列的调用方法（invoke, batch, stream, ainvoke, …）。这样，你可以用同样的方式调用不同类型的对象，无论它们是模型、函数、数据、配置、条件、逻辑等等。而且，你可以将多个对象链接起来，形成一个链式结构，这个结构本身也是一个对象，也可以被调用。这样，你可以将复杂的功能分解成简单的组件，然后用LCEL语法将它们组合起来，形成一个完整的应用。

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_BASE"] = "http://xx:8000/v1"
os.environ["OPENAI_API_KEY"] = "EMPTY"
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
output_parser = StrOutputParser()
model = ChatOpenAI(model="moonshot-v1-8k")
chain = prompt | model | output_parser
answer = chain.invoke({"topic": "ice cream"})
print(answer)
# 调用笑话对象，传入一个主题字符串，得到一个笑话字符串的流
chain.stream("dog")
```

`chain = prompt | model | output_parser`。我们可以看到这段代码中使用了运算符`|`，熟悉python的同学肯定知道，这里使用了python的magic method，也就是说它一定具有`__or__`函数。`prompt | model`就相当于`prompt.__or__(model)`。实际上，prompt实现了`__ror__`，这个magic method支持从右往左的or运算，`dict|prompt`,相当于`prompt.__ror__(dict)` 。

![](/public/upload/machine/langchain_lcel.jpg)


|Component|	Input Type|	Output Type|
|---|---|---|
|Prompt|	Dictionary|	PromptValue|
|ChatModel|	Single string, list of chat messages or a PromptValue|	ChatMessage|
|LLM|	Single string, list of chat messages or a PromptValue|	String|
|OutputParser|	The output of an LLM or ChatModel|	Depends on the parser|
|Retriever|	Single string|	List of Documents|
|Tool|	Single string or dictionary, depending on the tool|	Depends on the tool|

我们使用的所有LCEL相关的组件都继承自RunnableSerializable，RunnableSequence 顾名思义就按顺序执行的Runnable，分为两部分Runnable和Serializable。其中Serializable是继承自Pydantic的BaseModel。（py+pedantic=Pydantic，是非常流行的参数验证框架）Serializable提供了，将Runnable序列化的能力。而Runnable，则是LCEL组件最重要的一个抽象类，它有几个重要的抽象方法。

```
invoke/ainvoke: 单个输入转为输出。
batch/abatch:批量转换。
stream/astream: 单个流式处理。
astream_log:从输入流流式获取结果与中间步骤。
```

同时Runnbale也实现了两个重要的magic method ，就是前面说的用于支持管道操作符|的 `__or__` 与`__ror__`。Runnable之间编排以后，会生成一个RunnableSequence。如果我们运行最终编排好的Chain，例如chain.invoke({"topic": "ice cream"})，实际上就是执行了RunnableSequence的invoke。那我们先来看看invoke函数。

```python
# config对象，可以设置一些并发数、标签等等配置，默认情况下为空。
def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:  
    from langchain_core.beta.runnables.context import config_with_context  
  
    # 根据上下文补充config
    config = config_with_context(ensure_config(config), self.steps)  
    # 创建回调管理器，用于支持运行中产生的各种回调
    callback_manager = get_callback_manager_for_config(config)  
    # 创建运行管理器，用于处理异常重试，结束等情况
    run_manager = callback_manager.on_chain_start(  
        dumpd(self), input, name=config.get("run_name") or self.get_name()  
    )  
	# ！！关键内容！！
    # 调用整个链
    try:  
	    # 顺序执行step，每一步的输出，将作为下一步的输入
        for i, step in enumerate(self.steps):  
            input = step.invoke(  
                input,  
                # 为下一个step更新config 
                patch_config(  
                    config, callbacks=run_manager.get_child(f"seq:step:{i+1}")  
                ),  
            )  
    # finish the root run  
    except BaseException as e:  
        run_manager.on_chain_error(e)  
        raise  
    else:  
        run_manager.on_chain_end(input)  
        return cast(Output, input)
```

LCEL提供了多种优势，例如一流的流支持、异步支持、优化的并行执行、支持重试和回退、访问中间结果、输入和输出模式以及无缝 LangSmith 跟踪集成。但因为语法上的问题，要实现 loop 和 condition 的情况就比较困难。于是LangChain社区推出了一个新的项目——LangGraph，期望基于LangChain构建支持循环和跨多链的计算图结构，以描述更复杂的，甚至具备自动化属性的AI工程应用逻辑，比如智能体应用。

## LangGraph

[彻底搞懂LangGraph：构建强大的Multi-Agent多智能体应用的LangChain新利器 ](https://mp.weixin.qq.com/s/MzLz4lJF0WMsWrThiOWPog)相对于Chain.invoke()直接运行，Agent_executor的作用就是为了能够实现多次循环ReAct的动作，以最终完成任务。为什么需要将循环引入运行时呢？考虑一个增强的RAG应用：我们可以对语义检索出来的关联文档（上下文）进行评估：如果评估的文档质量很差，可以对检索的问题进行重写（Rewrite，比如把输入的问题结合对话历史用更精确的方式来表达），并把重写结果重新交给检索器，检索出新的关联文档，这样有助于获得更精确的结果。这里把Rewrite的问题重新交给检索器，就是一个典型的“循环”动作。而在目前LangChain的简单链中是无法支持的。其他一些典型的依赖“循环”的场景包括：代码生成时的自我纠正：当借助LLM自动生成软件代码时，根据代码执行的结果进行自我反省，并要求LLM重新生成代码；Web访问自动导航：每当进入下一界面时，需要借助多模态模型来决定下一步的动作（点击、滚动、输入等），直至完成导航。

那么，如果我们需要在循环中调用LLM能力，就需要借助于AgentExecutor。其调用的过程主要就是两个步骤：
1. 通过大模型来决定采取什么行动，使用什么工具，或者向用户输出响应（如运行结束时）；
2. 执行1步骤中的行动，比如调用某个工具，并把结果继续交给大模型来决定，即返回步骤1；
这里的AgentExecute存在的问题是：过于黑盒，所有的决策过程隐藏在AgentExecutor背后，缺乏更精细的控制能力，在构建复杂Agent的时候受限。这些精细化的控制要求比如：
1. 某个Agent要求首先强制调用某个Tool
2. 在 Agent运行过程中增加人机交互步骤
3. 能够灵活更换Prompt或者背后的LLM
4. 多Agent（Multi-Agent）智能体构建的需求，即多个Agent协作完成任务的场景支持。
所以，让我们简单**总结LangGraph诞生的动力**：LangChain简单的链（Chain）不具备“循环”能力；而AgentExecutor调度的Agent运行又过于“黑盒”。因此需要一个具备更精细控制能力的框架来支持更复杂场景的LLM应用。

LangGraph的实现方式是把之前基于AgentExecutor的黑盒调用过程用一种新的形式来构建：状态图（StateGraph）。把基于LLM的任务（比如RAG、代码生成等）细节用Graph进行精确的定义（定义图的节点与边），最后基于这个图来编译生成应用；在任务运行过程中，维持一个中央状态对象(state)，会根据节点的跳转不断更新，状态包含的属性可自行定义。

![](/public/upload/machine/lang_graph_agent.jpg)

## 示例

一个最基础的ReAct范式的Agent应用对应的Graph如下：

![](/public/upload/machine/lang_graph_example.jpg)

简单的实现代码如下（省略了部分细节）：

```python
# 定义一个Graph，传入state定义（参考上图state属性）
workflow = StateGraph(AgentState)

# 两个节点
#节点1: 推理节点，调用LLM决定action，省略了runreason细节
workflow.add_node("reason", run_reason)

#节点2: 行动节点，调用tools执行action，省略executetools细节
workflow.add_node("action", execute_tools)
#入口节点：总是从推理节点开始
workflow.set_entry_point("reason")
#条件边：根据推理节点的结果决定下一步
workflow.add_conditional_edges(
    "reason",
    should_continue, #条件判断函数（自定义，根据状态中的推理结果判断）
    {
        "continue": "action", #如果条件函数返回continue，进action节点
        "end": END, #如果条件函数返回end，进END节点
    },
)
#普通边：action结束后，总是返回reason
workflow.add_edge("action", "reason")
#编译成app
app = workflow.compile()
#可以调用app了，并使用流式输出
inputs = {"input": "you task description", "chat_history": []}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")
```

### 原理

1. StateGraph，它将该对象传递给每个节点。然后，节点会以键值对的形式，返回对状态属性的操作。这些操作可以是在状态上设置特定属性（例如，覆盖现有值）或者添加到现有属性。
2. 在创建了StateGraph之后，我们需要向其中添加Nodes（节点）。添加节点是通过`graph.add_node(name, value)`语法来完成的。其中，`name`参数是一个字符串，用于在添加边时引用这个节点。`value`参数应该可以是函数或runnable 接口，它们将在节点被调用时执行。它们可以接受一个字典作为输入，这个字典的格式应该与State对象相同，在执行完毕之后也会输出一个字典，字典中的键是State对象中要更新的属性。说白了，Nodes（节点）的责任是“执行”，在执行完毕之后会更新StateGraph的状态。
3. 节点通过边相互连接，形成了一个有向无环图（DAG），边有几种类型：
    1. Normal Edges：即确定的状态转移，这些边表示一个节点总是要在另一个节点之后被调用。
    2. Conditional Edges：输入是一个节点，输出是一个mapping，连接到所有可能的输出节点，同时附带一个判断函数，根据输入判断流转到哪一个输出节点上。



