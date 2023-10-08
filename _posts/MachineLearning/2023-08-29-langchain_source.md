---

layout: post
title: LangChain源码学习
category: 技术
tags: MachineLearning
keywords: langchain

---

* TOC
{:toc}

## 前言

LLM 不管是 GPT 还是 BERT，有且只有一个核心功能，就是预测你给定的语句的下一个词最有可能是什么（靠Prompt激发），除此之外的工作，比如解析 PDF、比如对话式搜索、甚至拿过来一个大任务分解、创建子任务，最终完成，都需要有一整套的工具来把核心功能包装，便于开发人员搭积木，这个工具就是 LangChain。

LangChain底层就是Prompt、大模型API、以及三方应用API调用三个个核心模块。对于LangChain底层不同的功能，都是需要依赖不同的prompt进行控制。**基于自然语言对任务的描述进行模型控制，对于任务类型没有任何限制，只有说不出来，没有做不到的事情**。

PS：看LangChain的感受就是：遇事不决问LLM。这跟常规的工程项目 严丝合缝的逻辑 + ifelse控制流非常不一样。 比如外挂知识库，LLM 不只用于最后一步 对topk 匹配的chunk 做一下润色给出anwser，前期的文档切分、存储、history的存储、选用，用户query的意图识别、转换都可能用到LLM。

## LLM模型层

一次最基本的LLM调用需要的prompt、调用的LLM API设置、输出文本的结构化解析（output_parsers 在 prompt 中插入了需要返回的格式说明）等。从 BaseLanguageModel 可以看到**模型层抽象接口方法predict 输入和输出是str**，也就是 TEXT IN TEXT OUT。

```python
# BaseLanguageModel 是一个抽象基类，是所有语言模型的基类
class BaseLanguageModel(...):
    # 基于用户输入生成prompt
    @abstractmethod
    def generate_prompt(self,prompts: List[PromptValue],stop: Optional[List[str]] = None,...) -> LLMResult:
    @abstractmethod
    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any ) -> str:
    @abstractmethod
    def predict_messages(self, messages: List[BaseMessage],) -> BaseMessage:
# BaseLLM 增加了缓存选项, 回调选项, 持有各种参数
class BaseLLM(BaseLanguageModel[str], ABC):
    1. 覆盖了 __call__ 实现  ==> generate ==> _generate_helper ==> prompt 处理 +  _generate 留给子类实现。 
    2. predict ==> __call__
# LLM类期望它的子类可以更加简单，将大模型的调用方法完全封装，不需要用户实现完整的_generate方法，只需要对外提供一个非常简单的call方法就可以操作LLMs
class LLM(BaseLLM):
    1. _generate ==> _call 留给子类实现。 输入文本格式提示，返回文本格式的答案
```

LangChain 本质上就是对各种大模型提供的 API 的套壳，是为了方便我们使用这些 API，搭建起来的一些框架、模块和接口。因此，要了解 LangChain 的底层逻辑，需要了解大模型的 API 的基本设计思路。重点有两类模型：Chat Model 和 Text Model（当然，OpenAI 还提供 Image、Audio 和其它类型的模型），Chat 模型和 Text 模型的调用是完全一样的，**只是输入（input/prompt）和输出（response）的数据格式有所不同**。
1. Chat Model，聊天模型，用于产生人类和 AI 之间的对话，有两个专属于 Chat 模型的概念，一个是消息，一个是角色。每个消息都有一个 role（可以是 system、user 或 assistant）和 content（消息的内容）。系统消息设定了对话的背景（比如你是一个很棒的智能助手），然后用户消息提出了具体请求。
    1. system：系统消息主要用于设定对话的背景或上下文。这可以帮助模型理解它在对话中的角色和任务。例如，你可以通过系统消息来设定一个场景，让模型知道它是在扮演一个医生、律师或者一个知识丰富的 AI 助手。系统消息通常在对话开始时给出。
    2. user：用户消息是从用户或人类角色发出的。它们通常包含了用户想要模型回答或完成的请求。用户消息可以是一个问题、一段话，或者任何其他用户希望模型响应的内容。
    3. assistant：助手消息是模型的回复。例如，在你使用 API 发送多轮对话中新的对话请求时，可以通过助手消息提供先前对话的上下文。然而，请注意在对话的最后一条消息应始终为用户消息，因为模型总是要回应最后这条用户消息。
2. Text Model，文本模型

Chat Model响应

```json
{
 'id': 'chatcmpl-2nZI6v1cW9E3Jg4w2Xtoql0M3XHfH',
 'object': 'chat.completion',
 'created': 1677649420,
 'model': 'gpt-4',
 'usage': {'prompt_tokens': 56, 'completion_tokens': 31, 'total_tokens': 87},
 'choices': [
   {
    'message': {
      'role': 'assistant',
      'content': '你的花店可以叫做"花香四溢"。'
     },
    'finish_reason': 'stop',
    'index': 0
   }
  ]
}
```

## Prompt

[Langchain 中的提示工程](https://cookbook.langchain.com.cn/docs/langchain-prompt-templates/)我们只要让机器将下一个单词预测的足够准确就能完成许多复杂的任务！并且是**自己写大部分让大模型补小部分**。下面是一个典型的提示结构。并非所有的提示都使用这些组件，但是一个好的提示通常会使用两个或更多组件。让我们更加准确地定义它们。
1. 指令 ：告诉模型该怎么做，如何使用外部信息（如果提供），如何处理查询并构建 Out。
2. 外部信息 或 上下文 ：充当模型的附加知识来源。这些可以手动插入到提示中，通过矢量数据库 （Vector Database） 检索（检索增强）获得，或通过其他方式（API、计算等）引入。
3. 用户 In 或 查询 ：通常（但不总是）是由人类用户（即提示者）In 到系统中的查询。
4. Out 指示器 ：标记要生成的文本的 开头。如果生成 Python 代码，我们可以使用 import 来指示模型必须开始编写 Python 代码（因为大多数 Python 脚本以 import 开头）。

![](/public/upload/machine/prompt_structure.jpg)

我们不太可能硬编码上下文和用户问题。我们会通过一个 模板 PromptTemplate 简化使用动态 In 构建提示的过程。我们本可以轻松地用 f-strings（如 f"insert some custom text '{custom_text}' etc"）替换。然而，使用Langchain 的 PromptTemplate 对象，我们可以规范化这个过程，添加多个参数，并**以面向对象的方式构建提示**。

![](/public/upload/machine/llm_chain.jpg)

few-shot learning 适用于将这些示例在提示中提供给模型，通过示例来强化我们在提示中传递的指令，我们可以使用 Langchain 的 FewShotPromptTemplate 规范化这个过程，**比如根据查询长度来可变地包含不同数量的示例**，因为我们的提示和补全 (completion) Out 的最大长度是有限的，这个限制通过 最大上下文窗口 maximum context window 进行衡量，上下文窗口 (ontext window) = In 标记 (input_tokens) + Out 标记 (output tokens)。如果我们传递一个较短或较长的查询，我们应该会看到所包含的示例数量会有所变化。

```
prompt = "" " The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples: 

User: How are you?
AI: I can't complain but sometimes I still do.

User: What time is it?
AI: It's time to get a watch.

User: What is the meaning of life?
AI: "" "
```

## Chain模块

LangChain是语言链的涵义，那么Chain就是其中的链结构，属于组合各个层的中间结构，可以称之为胶水层，将各个模块（models, document retrievers, other chains）粘连在一起，实现相应的功能，也是用于程序的调用入口。

Chain模块有一个基类Chain，是所有chain对象的基本入口，与用户程序的交互、用户的输入、其他模块的输入、内存的接入、回调能力。chain通过传入String值，控制接受的输入和给到的输出格式。Chain的子类基本都是担任某项专业任务的具体实现类，比如LLMChain，这就是专门为大语言模型准备的Chain实现类（一般是配合其他的chain一起使用）。PS： 注意，这些是Chain 的事情，模型层不做这些
1. 针对每一种chain都有对应的load方法，load方法的命名很有规律，就是在chain的名称前面加上`_load`前缀
2. **从 Chain可以看到核心方法run/_call输入输出是dict**，有dict 自然有key，所以每个 Chain 里都包含了两个很重要的属性：input_keys 和 output_keys。 
    1. input 这些keys 都会被用于format chain 对应的prompt template。最终prompt template 要求有哪些variables，用户输入、memory 就需要提供哪些
        1. 用户会输入一些key，对应用户在prompt template 中的variables，若是只插入了一个str，则视为key=input 或 key=query 或 key=question
        2. memory 会提供key=history 或 chat_history
        3. vector 会提供key=input_documents 或 key= context     
    2. 除了大模型获取的text输出，其它组件也会附着一些key
        1. llm 输出对应key=result/key=response
        2. memory会获取特定的key 保存下来作为hitory
        3. vector会带上附属的 key=source_documents
PS： 不准确的说，**各种chain的核心是预定了很多prompt template的构建方法**。

链有很多种调用方式。
1. 直接调用，当我们像函数一样调用一个对象时，它实际上会调用该对象内部实现的 __call__ 方法。
2. 通过 run 方法，也等价于直接调用 _call_ 函数。
3. predict 方法类似于 run，只是输入键被指定为关键字参数而不是 Python 字典。
4. apply 方法允许我们针对输入列表运行链，一次处理多个输入。

    ```python
    input_list = [
        {"flower": "玫瑰",'season': "夏季"},
        {"flower": "百合",'season': "春季"},
        {"flower": "郁金香",'season': "秋季"}
    ]
    result = llm_chain.apply(input_list)
    print(result)
    ```
5. generate 方法类似于 apply，只不过它返回一个 LLMResult 对象，而不是字符串。LLMResult 通常包含模型生成文本过程中的一些相关信息，例如令牌数量、模型名称等。

```python
class Chain(...,ABC):
    memory: Optional[BaseMemory] = None
    callbacks: Callbacks = Field(default=None, exclude=True)
    @abstractmethod
    def _call(self,inputs: Dict[str, Any],run_manager: Optional[CallbackManagerForChainRun] = None,) -> Dict[str, Any]:
    1. 覆盖了 __call__ 实现 ==> 输入处理 + _call + 输出处理  
```

![](/public/upload/machine/langchain_chain_run.png)

```python
from langchain import PromptTemplate, OpenAI, LLMChain
 
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate.from_template("将下面的句子翻译成英文：{sentence}")
llm_chain = LLMChain(
    llm = llm, 
    prompt = prompt
)
result = llm_chain("今天的天气真不错")
print(result['text'])
```

![](/public/upload/machine/chain_call.jpg)

```python
class Chain(Serializable, Runnable[Dict[str, Any], Dict[str, Any]], ABC):
    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
    # # Chain 的本质其实就是根据一个 Dict 输入，得到一个 Dict 输出
    def __call__(self,inputs: Union[Dict[str, Any], Any],...) -> Dict[str, Any]:
        inputs = self.prep_inputs(inputs)
        outputs = (self._call(inputs, run_manager=run_manager) if new_arg_supported else self._call(inputs))
class LLMChain(Chain):
    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables
    def _call(self,inputs: Dict[str, Any],run_manager: Optional[CallbackManagerForChainRun] = None,) -> Dict[str, str]:
        response = self.generate([inputs], run_manager=run_manager)
            prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
                for inputs in input_list:
                    selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
                    prompt = self.prompt.format_prompt(**selected_inputs)
                    prompts.append(prompt)
            return self.llm.generate_prompt(prompts,stop)  # 
        return self.create_outputs(response)[0]
```
继承 Chain 的子类主要有两种类型：
1. 通用工具 Chain: 控制 Chain 的调用顺序， 是否调用，他们可以用来合并构造其他的 Chain 。比如MultiPromptChain、EmbeddingRouterChain、LLMRouterChain(使用 LLM 来确定动态选择下一个链)。
2. 专门用途 Chain: 和通用 Chain 比较来说，他们承担了具体的某项任务，可以和通用的 Chain 组合起来使用，也可以直接使用。有些 Chain 类可能用于处理文本数据，有些可能用于处理图像数据，有些可能用于处理音频数据等。
|`__call__逻辑`||||
|---|---|---|---|
|Chain|prep_inputs<br>inputs = inputs + memory external_context|_call|prep_outputs <br> memory.save_context|
|LLMChain||generate=prep_prompts+generate_prompt|
|||docs = _get_docs(question)<br> answer = combine_documents_chain.run(question,docs)|
|AgentExecutor||while._should_continue <br> agent.plan + tool.run|
|ConversationalRetrievalChain||chat_history_str = get_chat_history <br>new_question = question_generator.run(new_question,chat_history_str) <br> docs = _get_docs(new_question,inputs) <br> answer = combine_docs_chain.run(new_question,docs)<br>||

PS：用一个最复杂的场景比如 ConversationalRetrievalChain 打上断点，观察各个变量值的变化，有助于了解Chain的运行逻辑。 

### RetrievalQA

```python
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",retriever=self.vector_db.as_retriever())
answer = qa(query)                                          
```

其实质是 通过retriever 获取相关文档，并通过BaseCombineDocumentsChain 来获取答案。

```
RetrievalQA.from_chain_type 
    ==> load_qa_chain ==> _load_stuff_chain ==> StuffDocumentsChain(xx)
    ==> RetrievalQA(chain, retriever)
```

```python
# langchain/chains/retrieval_qa/base.py
@classmethod
def from_chain_type(cls,llm: BaseLanguageModel,chain_type: str = "stuff",chain_type_kwargs: Optional[dict] = None,**kwargs: Any,) -> BaseRetrievalQA:
    """Load chain from chain type."""
    _chain_type_kwargs = chain_type_kwargs or {}
    combine_documents_chain = load_qa_chain(llm, chain_type=chain_type, **_chain_type_kwargs)
    return cls(combine_documents_chain=combine_documents_chain, **kwargs)
# langchain/chains/question_answering/__init__.py
def load_qa_chain(llm: BaseLanguageModel,chain_type: str = "stuff",verbose: Optional[bool] = None,callback_manager: Optional[BaseCallbackManager] = None,**kwargs: Any,) -> BaseCombineDocumentsChain:
    loader_mapping: Mapping[str, LoadingCallable] = {
        "stuff": _load_stuff_chain,
        "map_reduce": _load_map_reduce_chain,
        "refine": _load_refine_chain,
        "map_rerank": _load_map_rerank_chain,
    }
    return loader_mapping[chain_type](
        llm, verbose=verbose, callback_manager=callback_manager, **kwargs
    )
def _load_stuff_chain(...)-> StuffDocumentsChain:
    _prompt = prompt or stuff_prompt.PROMPT_SELECTOR.get_prompt(llm)
    llm_chain = LLMChain(llm=llm,prompt=_prompt,verbose=verbose,callback_manager=callback_manager,callbacks=callbacks,)
    return StuffDocumentsChain(llm_chain=llm_chain,...) 
```

## Agent

Agent在LangChain框架中负责决策制定以及工具组的串联，可以根据用户的输入决定调用哪个工具。通过精心制定的提示，我们能够赋予代理特定的身份、专业知识、行为方式和目标。提示策略为 Agent 提供了预设模板，结合关键的指示、情境和参数来得到 Agent 所需的响应。具体的说，Agent就是将大模型进行封装来简化用户使用，根据用户的输入，理解用户的相应意图，通过action字段选用对应的Tool，并将action_input作为Tool的入参，来处理用户的请求。当我们不清楚用户意图的时候，由Agent来决定使用哪些工具实现用户的需求。

大佬：这一波Agent热潮爆发，其实是LLM热情的余波，大家太希望挖掘LLM潜力，为此希望LLM担任各方面的判断。但实际上有一些简单模块是不需要LLM的，不经济也不高效。例如我们要抽取每轮对话的情绪，可以用LLM，其实也可以用情绪识别模型。例如我们希望将长对话压缩后作为事件记忆存储，可以用LLM，也可以用传统摘要模型，一切只看是否取得ROI的最佳平衡，而不全然指望LLM。

### 使用

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
T
hought: I now know the final answer # 最后，你应该知道最终结果了
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
Observation: 3
Thought:
```
此时两个问题全都拿到了结果，根据开头的限定，OpenAi在完全拿到结果以后会返回I now know the final answer。并且根据完整上下文。把多个结果进行归纳总结：下面是完整的相应结果：
```
I now know the final answer
Final Answer: I will be 38 in ten years and the weather this week is sunny.
```
可以看到。ai严格的按照设定返回想要的内容，并且还以外的把28+10=3这个数学错误给改正了。通过 `verbose=True` 可以动态查看上述过程。 PS: 通过prompt 引导llm 进行文字接龙，通过解析文字接龙来进行tool 的调用。

根据输出再回头看agent的官方解释：An Agent is a wrapper around a model, which takes in user input and returns a response corresponding to an “action” to take and a corresponding “action input”.

### stop token

[How to Get Better Outputs from Your Large Language Model](https://developer.nvidia.com/blog/how-to-get-better-outputs-from-your-large-language-model/)It is especially useful to design a stopping template in a **few-shot** setting so the model can learn to stop appropriately upon completing an intended task. Figure shows separating examples with the string “===” and passing that as the stop word.

![](/public/upload/machine/llm_stop.jpg)


### 认知框架Cognitive Architecture

**AgentType 对应一个Agent class，对应一个prompt**（又是prompt 起了关键作用），AgentType 有以下几种选择
1. zero-shot ReAct，完全依靠对所用到的tools 的说明书来理解和使用tools，理论上支持无限多个。
2. Structured tool chat，跟第一个不同的地方在于接收一个结构化的dict 作为参数且能记住上下文。
3. OpenAI functions，OpenAI 在大模型层面针对 API的调用做了训练，相当于帮大家做了SFT，可以想象效果必然好。
4. conversational，类似于第一、二类型，针对对话场景做了优化，比如聊天记录、聊天轮次等meta-data
5. self-ask，通过自问自答的方式把大问题拆解成小问题之后再组成最终的单子。

ReAct是 Shunyu Yao 等人在 ICLR 2023 会议论文《ReAct: Synergizing Reasoning and Acting in Language Models》中提出的，一个关键启发在于：大语言模型可以通过生成推理痕迹和任务特定行动来实现更大的协同作用。具体来说，就是引导模型生成一个任务解决轨迹：观察环境 - 进行思考 - 采取行动，也就是观察 - 思考 - 行动。那么，再进一步进行简化，就变成了推理 - 行动。ReAct 框架会提示 LLMs 为任务生成推理轨迹和操作，这使得代理能系统地执行动态推理来创建、维护和调整操作计划，同时还支持与外部环境（例如 Google 搜索、Wikipedia）的交互，以将额外信息合并到推理中。PS：使用LLM来做ifelse


### 源码

LangChain关键组件
1. 代理（Agent）：这个类决定下一步执行什么操作。它由一个语言模型和一个提示（prompt）驱动。提示可能包含代理的性格（也就是给它分配角色，让它以特定方式进行响应）、任务的背景（用于给它提供更多任务类型的上下文）以及用于激发更好推理能力的提示策略（例如 ReAct）。LangChain 中包含很多种不同类型的代理。
2. 工具（Tools）：工具是代理调用的函数。这里有两个重要的考虑因素：一是让代理能访问到正确的工具，二是以最有帮助的方式描述这些工具。如果你没有给代理提供正确的工具，它将无法完成任务。如果你没有正确地描述工具，代理将不知道如何使用它们。LangChain 提供了一系列的工具，同时你也可以定义自己的工具。
3. 代理执行器（AgentExecutor）：代理执行器是代理的运行环境，它调用代理并执行代理选择的操作。执行器也负责处理多种复杂情况，包括处理代理选择了不存在的工具的情况、处理工具出错的情况、处理代理产生的无法解析成工具调用的输出的情况，以及在代理决策和工具调用进行观察和日志记录。

AgentExecutor由一个Agent和Tool的集合组成。AgentExecutor负责调用Agent，获取返回（callback）、action和action_input，并根据意图将action_input给到具体调用的Tool，获取Tool的输出，并将所有的信息传递回Agent，以便猜测出下一步需要执行的操作。`AgentExecutor.run 实质是chain.run ==> AgentExecutor.__call__ 实质是chain.__call__() ==> AgentExecutor._call()`

```python
def initialize_agent(tools,llm,...)-> AgentExecutor:
    agent_obj = agent_cls.from_llm_and_tools(llm, tools, callback_manager=callback_manager, **agent_kwargs)
    AgentExecutor.from_agent_and_tools(agent=agent_obj, tools=tools,...)
    return cls(agent=agent, tools=tools, callback_manager=callback_manager, **kwargs)
# AgentExecutor 实际上是一个 Chain，可以通过 .run() 或者 _call() 来调用
class AgentExecutor(Chain):
    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]
    tools: Sequence[BaseTool]
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

有人希望通过一些开源的  LLM 来实现 ReAct Agent，但实际开发过程中会发现开源低参数（比如一些 6B、7B 的 LLM）的 LLM  对于提示词的理解会非常差，根本不会按照提示词模板的格式来输出，这样就会导致我们的 Agent 无法正常工作，所以如果想要实现一个好的  Agent，还是需要使用好的 LLM，目前看来使用gpt-3.5模型是最低要求。


## Memory

记忆 ( memory )允许大型语言模型（LLM）记住与用户的先前交互。默认情况下，LLM 是 无状态 stateless 的，这意味着每个传入的查询都独立处理，不考虑其他交互。对于无状态代理 (Agents) 来说，唯一存在的是当前输入，没有其他内容。有许多应用场景，记住先前的交互非常重要，比如聊天机器人。在 LangChain 中，有几种方法可以实现对话记忆，它们都是构建在 ConversationChain 之上的。


ConversationChain 的提示模板 `print(conversation.prompt.template)`：
```
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation:
{history}
Human: {input}
AI:
```

ConversationSummaryMemory 的提示模版

```
Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:
```

使用这种方法，我们可以总结每个新的交互，并将其附加到所有过去交互的 summary 中。

### 使用了memory的chain的流程

使用了memory的chain的流程如下，整个执行的流程其实除来入口调用predict及标红部分，其他执行的步骤和流程一样。标红部分就是memory的使用部分。主要包括load_memory_variables和save_context。PS：跟BaseMemory的接口方法也都对上了。

![](/public/upload/machine/chain_memory.jpg)

### 底层实现

```python
class BaseMemory(Serializable, ABC):
    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """The string keys this memory class will add to chain inputs."""
    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain."""
    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this chain run to memory."""
class BaseChatMemory(BaseMemory, ABC):
    """Abstract base class for chat memory."""
    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory) # 这就是所谓的存在内存里了
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = False
# 消息在内存中的形态
class BaseChatMessageHistory(ABC):
    messages: List[BaseMessage] 
    def add_user_message(self, message: str) -> None:   # 带有Chat字样的类是为聊天设计的，主要保存聊天的历史信息，实现了add_xx_message方法
    def add_ai_message(self, message: str) -> None:
# 消息格式
class BaseMessage(Serializable):
    content: str
    additional_kwargs: dict = Field(default_factory=dict)
```