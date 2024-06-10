---

layout: post
title: LLM工作流编排
category: 技术
tags: MachineLearning
keywords: langchain langgraph lcel

---

* TOC
{:toc}

## 从顺序式为主的简单架构走向复杂的WorkFlow

编程语言大类上可以分为命令式编程和声明式编程，前者深入细节，各种 if else、各种 while/for，程序员掌控每个像素；后者把任务「描述」清楚，重点在业务流程翻译成所用的语言上，具体怎么实现甩给别人（大部分是系统自带）。由于这一波 LLMs 强大的理解、生成能力，**关注细节的命令式编程似乎不再需要**，而偏重流程或者说业务逻辑编排的 pipeline 能力的声明式编程，成了主流「编程」方式。

推理阶段的RAG Flow分成四种主要的基础模式：顺序、条件、分支与循环。PS： 一个llm 业务有各种基本概念，prompt/llm/memory，整个工作流产出一个流式输出，处理链路上包含多个step，且step有复杂的关系（顺序、条件、分支与循环）。一个llm 业务开发的核心就是个性化各种原子能力 以及组合各种原子能力。

以一个RAG Agent 的工作流程为例
1. 根据问题，路由器决定是从向量存储中检索上下文还是进行网页搜索。
2. 如果路由器决定将问题定向到向量存储以进行检索，则从向量存储中检索匹配的文档；否则，使用 tavily-api 进行网页搜索。
3. 文档评分器然后将文档评分为相关或不相关。
4. 如果检索到的上下文被评为相关，则使用幻觉评分器检查是否存在幻觉。如果评分器决定响应缺乏幻觉，则将响应呈现给用户。
5. 如果上下文被评为不相关，则进行网页搜索以检索内容。
6. 检索后，文档评分器对从网页搜索生成的内容进行评分。如果发现相关，则使用 LLM 进行综合，然后呈现响应。

## LCEL 

在 LangChain 里只要实现了Runnable接口，并且有invoke方法，都可以成为链。实现了Runnable接口的类，可以拿上一个链的输出作为自己的输入。

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

### 基石Runnable

我们使用的所有LCEL相关的组件都继承自RunnableSerializable，RunnableSequence 顾名思义就按顺序执行的Runnable，分为两部分Runnable和Serializable。其中Serializable是继承自Pydantic的BaseModel。（py+pedantic=Pydantic，是非常流行的参数验证框架）Serializable提供了，将Runnable序列化的能力。而Runnable，则是LCEL组件最重要的一个抽象类，它有几个重要的抽象方法。

```python
class Runnable(Generic[Input, Output], ABC):
    @abstractmethod
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
```

Runnable所有接口都接收可选的配置参数，可用于配置执行、添加标签和元数据，以进行跟踪和调试。
1. invoke/ainvoke: 单个输入转为输出。
2. batch/abatch:批量转换。
3. stream/astream: 单个流式处理。
4. astream_log:从输入流流式获取结果与中间步骤。
有时我们希望 使用常量参数调用Runnable 调用链中的Runnable对象，这些常量参数不是序列中前一个Runnable 对象输出的一部分，也不是用户输入的一部分，我们可以使用Runnable.bind 方法来传递这些参数。


同时Runnbale也实现了两个重要的magic method ，就是前面说的用于支持管道操作符|的 `__or__` 与`__ror__`。Runnable之间编排以后，会生成一个RunnableSequence。

```python
class Runnable(Generic[Input, Output], ABC):
    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Callable[[Iterator[Any]], Iterator[Other]],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other], Any]],
        ],
    ) -> RunnableSerializable[Input, Other]:
        """Compose this runnable with another object to create a RunnableSequence."""
        return RunnableSequence(self, coerce_to_runnable(other))

    def __ror__(
        self,
        other: Union[
            Runnable[Other, Any],
            Callable[[Other], Any],
            Callable[[Iterator[Other]], Iterator[Any]],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any], Any]],
        ],
    ) -> RunnableSerializable[Other, Output]:
        """Compose this runnable with another object to create a RunnableSequence."""
        return RunnableSequence(coerce_to_runnable(other), self)
```

**Runnable 对象表示一个可调用的函数或操作单元，RunnableSequence 可以看成由lcel 构建的调用链的实际载体**。如果我们运行最终编排好的Chain，例如chain.invoke({"topic": "ice cream"})，实际上就是执行了RunnableSequence的invoke。那我们先来看看invoke函数。

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
Runnable 还有很多增强/装饰方法，对underlying runnable 增加一些配置、逻辑得到一个新的Runnable，以openai为例，**底层实质扩充了openai.ChatCompletion.create 前后逻辑或调用参数**。
```python
class Runnable(Generic[Input, Output], ABC):
    def assign(self,**kwargs)-> RunnableSerializable[Any, Any]:
        return self | RunnableAssign(RunnableParallel(kwargs))
    # Bind kwargs to pass to the underlying runnable when running it.
    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        return RunnableBinding(bound=self, kwargs=kwargs, config={})
    # Bind config to pass to the underlying runnable when running it.
    def with_config(self,config: Optional[RunnableConfig] = None,**kwargs: Any,) -> Runnable[Input, Output]:
        return RunnableBinding(...)
    # Bind a retry policy to the underlying runnable.
    def with_retry(self,retry_if_exception_type,...) -> Runnable[Input, Output]:
        return RunnableRetry(...)
    # Bind a fallback policy to the underlying runnable.
    def with_fallbacks(self,fallbacks,...)-> RunnableWithFallbacksT[Input, Output]:
        return RunnableWithFallbacks(self,fallbacks,...)
```

### 一些实践

```python
def add_one(x: int) -> int:
    return x + 1
def mul_two(x: int) -> int:
    return x * 2
runnable_1 = RunnableLambda(add_one) # RunnableLambda 可以把一个Callable类转成Runnable类（python所有可调用对象都是Callable 类型），从而可以将你自定义的函数集成到chain中
runnable_2 = RunnableLambda(mul_two)
sequence = runnable_1 | runnable_2
sequence.invoke(1)

def mul_three(x: int) -> int:
    return x * 3
sequence = runnable_1 | {  # Runnable对象的列表或字典/this dict is coerced to a RunnableParallel
    "mul_two": runnable_2,
    "mul_three": runnable_3,
}
sequence.invoke(1) # 会输出一个dict {'mul_two':4, 'mul_three':6}

branch = RunnableBranch(
    (lambda x: isinstance(x, str), lambda x: x.upper()),
    (lambda x: isinstance(x, int), lambda x: x + 1),
    (lambda x: isinstance(x, float), lambda x: x * 2),
    lambda x: "goodbye",
)
branch.invoke("hello") # "HELLO"
branch.invoke(None) # "goodbye"
```

RunnableParallel 的使用可以有以下三种形式，三种形式等价：
```
{"context": retriever, "question": RunnablePassthrough()}
RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
RunnableParallel(context=retriever, question=RunnablePassthrough())
```
在使用LCEL构建链时，原始用户输入可能不仅要传给第一个组件，还要传给后续组件，这时可以用RunnablePassthrough。RunnablePassthrough可以透传用户输入。
```python
# 用户输入的问题，不止组件1的检索器要用，组件2也要用它来构建提示词，因此组件1使用RunnablePassthrough方法把原始输入透传给下一步。
chain = (
    # 由于组件2 prompt的输入要求是字典类型，所以组件1把检索器和用户问题写成字典格式，并用组件2的变量作为键。
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

两个Runnable 对象之间（很多时候是第一个）需要一些数据处理、转换、透传的逻辑
1. 在构建复杂的RunnableSequence时，我们可能需要将很多信息从上游传递到下游，此时可以用到RunnablePassthrough。此外，还可以使用RunnablePassthrough.assign 方法在透传上游数据的同时添加一些新的的数据。 
2. RunnableMap，底层是RunnableParallel，通常以一个dict 结构出现，value 是一个Runnable对象，lcel 会并行的调用 value部分的Runnable对象 并将其返回值填充dict，之后将填充后的dict 传递给RunnableSequence 的下一个Runnable对象。


目前Memory模块还是Beta版本，创建带Memory功能的Chain，并不能使用统一的LCEL语法。但是，LangChain提供了工具类RunnableWithMessageHistory，支持了为Chain追加History的能力，从某种程度上缓解了上述问题。不过需要指定Lambda函数get_session_history以区分不同的会话，并需要在调用时通过config参数指定具体的会话ID。

```python
llm = xx
prompt =  xx
chain = prompt | llm | output_parser
history = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="question",
    history_messages_key="chat_history",
)
```

LCEL提供了多种优势，例如一流的流支持、异步支持、优化的并行执行、支持重试和回退、访问中间结果、输入和输出模式以及无缝 LangSmith 跟踪集成。但因为语法上的问题，要实现 loop 和 condition 的情况就比较困难。于是LangChain社区推出了一个新的项目——LangGraph，期望基于LangChain构建支持循环和跨多链的计算图结构，以描述更复杂的，甚至具备自动化属性的AI工程应用逻辑，比如智能体应用。

## LangGraph

[彻底搞懂LangGraph：构建强大的Multi-Agent多智能体应用的LangChain新利器 ](https://mp.weixin.qq.com/s/MzLz4lJF0WMsWrThiOWPog)相对于Chain.invoke()直接运行，Agent_executor的作用就是为了能够实现多次循环ReAct的动作，以最终完成任务。为什么需要将循环引入运行时呢？考虑一个增强的RAG应用：我们可以对语义检索出来的关联文档（上下文）进行评估：如果评估的文档质量很差，可以对检索的问题进行重写（Rewrite，比如把输入的问题结合对话历史用更精确的方式来表达），并把重写结果重新交给检索器，检索出新的关联文档，这样有助于获得更精确的结果。这里把Rewrite的问题重新交给检索器，就是一个典型的“循环”动作。而在目前LangChain的简单链中是无法支持的。其他一些典型的依赖“循环”的场景包括：代码生成时的自我纠正：当借助LLM自动生成软件代码时，根据代码执行的结果进行自我反省，并要求LLM重新生成代码；Web访问自动导航：每当进入下一界面时，需要借助多模态模型来决定下一步的动作（点击、滚动、输入等），直至完成导航。

那么，如果我们需要在循环中调用LLM能力，就需要借助于AgentExecutor，将Agent 置于一个循环执行环境（PS：所谓自主运行就是靠循环）。其调用的过程主要就是两个步骤：
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

langgraph正在成为构建Agent的推荐方式。

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

LangGraph 三个核心要素
1. StateGraph，LangGraph 在图的基础上增添了全局状态变量，是一组键值对的组合，可以被整个图中的各个节点访问与更新，从而实现有效的跨节点共享及透明的状态维护。它将该对象传递给每个节点。然后，节点会以键值对的形式，返回对状态属性的操作。这些操作可以是在状态上设置特定属性（例如，覆盖现有值）或者添加到现有属性。
2. 在创建了StateGraph之后，我们需要向其中添加Nodes（节点）。添加节点是通过`graph.add_node(name, value)`语法来完成的。其中，`name`参数是一个字符串，用于在添加边时引用这个节点。`value`参数应该可以是函数或runnable 接口，它们将在节点被调用时执行。其输入应为状态图的全局状态变量，在执行完毕之后也会输出一组键值对，字典中的键是State对象中要更新的属性。说白了，Nodes（节点）的责任是“执行”，在执行完毕之后会更新StateGraph的状态。
3. 节点通过边相互连接，形成了一个有向无环图（DAG），边有几种类型：
    1. Normal Edges：即确定的状态转移，这些边表示一个节点总是要在另一个节点之后被调用。
    2. Conditional Edges：输入是一个节点，输出是一个mapping，连接到所有可能的输出节点，同时附带一个判断函数，根据全局状态变量的当前值判断流转到哪一个输出节点上，以充分发挥大语言模型的思考能力。

当我们使用这三个核心要素构建图之后，通过图对象的compile方法可以将图转换为一个 Runnable对象（Runnable也有Runnable.get_graph 转为Graph对象），之后就能使用与lcel完全相同的接口调用图对象。


