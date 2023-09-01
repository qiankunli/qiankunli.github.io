---

layout: post
title: langchain源码学习
category: 技术
tags: MachineLearning
keywords: langchain

---

* TOC
{:toc}

## 前言（未完成）

LangChain底层就是Prompt，大模型API，以及三方应用API调用三个个核心模块。对于LangChain底层不同的功能，都是需要依赖不同的prompt进行控制。**基于自然语言对任务的描述进行模型控制，对于任务类型没有任何限制，只有说不出来，没有做不到的事情**。

## 从langchain中学到的

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

我们在 Prompt 中定义了一个占位符 `{sentence}`，但是在调用 Chain 的时候并没有明确指定该占位符的值，LangChain 是怎么知道要将我们的输入替换掉这个占位符的呢？实际上，每个 Chain 里都包含了两个很重要的属性：input_keys 和 output_keys，用于表示这个 Chain 的输入和输出。所以 LLMChain 的入参就是 Prompt 的 input_variables。

![](/public/upload/machine/chain_call.jpg)

LLMChain(xx) ==> Chain.__call__ ==> LLMChain._call ==> LLMChain.prompt.format_prompt + LLMChain.llm.generate_prompt

```python
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
```

![](/public/upload/machine/langchain_chat.jpg)

## agent


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
# 这里使用OpenAI temperature=0需要限定为0
llm = OpenAI(temperature=0)
tools = [WeatherTool(), CalculatorTool()]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
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

通过这个模板，加上我们的问题以及自定义的工具，会变成下面这个样子（# 后面是增加的注释）：

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
通过这个模板向openai告知了一系列的规范，包括目前现有哪些工具集，你需要思考回答什么问题，你需要用到哪些工具，你对工具需要输入什么内容等。如果仅仅是这样，openai会完全补完你的回答，中间无法插入任何内容。因此LangChain使用OpenAI的stop参数，截断了AI当前对话。`"stop": ["\nObservation: ", "\n\tObservation: "]`。做了以上设定以后，OpenAI仅仅会给到Action和 Action Input两个内容就被stop停止。以下是OpenAI的响应内容：
```
I need to use the weather tool to answer the first part of the question, and the calculator to answer the second part.
Action: Weather
Action Input: This week
```
这里从Tools中找到name=Weather的工具，然后再将This Week传入方法。具体业务处理看详细情况。这里仅返回Sunny。
由于当前找到了Action和Action Input。 代表OpenAI认定当前任务链并没有结束。因此像请求体后拼接结果：Observation: Sunny 并且让他再次思考Thought。开启第二轮思考：下面是再次请求的完整请求体:
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

### 原理

```python
# 一个 Agent 单元负责执行一次任务
class Agent(BaseSingleActionAgent):
    llm_chain: LLMChain
    allowed_tools: Optional[List[str]] = None
    # agent 的执行功能在于 Agent.plan()
    def plan(self,intermediate_steps: List[Tuple[AgentAction, str]],callbacks: Callbacks = None,**kwargs: Any,) -> Union[AgentAction, AgentFinish]:
        #  # 将各种异构的历史信息转换成 inputs，传入到 LLM 当中
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        # 根据 LLM 生成的反馈，采取决策
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
        # 最后的输出 AgentAction 中会包括：需要使用的 tool，使用该 tool 时候，对应的执行命令。
        return self.output_parser.parse(full_output)
# AgentExecutor 实际上是一个 Chain，可以通过 .run() 或者 _call() 来调用
class AgentExecutor(Chain):
```

Agent.plan() 可以看做两步：
1. 将各种异构的历史信息转换成 inputs，传入到 LLM 当中；
2. 根据 LLM 生成的反馈，采取决策。LLM 生成的回复是 string 格式，langchain 中ZeroShotAgent 通过字符串匹配的方式来识别 action。
因此，agent 能否正常运行，与 prompt 格式，以及 LLM 的 ICL 以及 alignment 能力有着很大的关系。
    1. LangChain主要是基于GPT系列框架进行设计，其适用的Prompt不代表其他大模型也能有相同表现，所以如果要自己更换不同的大模型(如：文心一言，通义千问...等)。则很有可能底层prompt都需要跟著微调。
    2. 在实际应用中，我们很常定期使用用户反馈的bad cases持续迭代模型，但是Prompt Engeering的工程是非常难进行的微调的，往往多跟少一句话对于效果影响巨大，因此这类型产品达到80分是很容易的，但是要持续迭代到90分甚至更高基本上是很难的。

```
agent = initialize_agent(xx)
    AgentExecutor.from_agent_and_tools
agent/AgentExecutor.run 
    Chain.run
        Chain.__call__
            AgentExecutor._call
                 while self._should_continue(iterations, time_elapsed):
                     next_step_output = self._take_next_step(...,inputs,...)
                         output = self.agent.plan(...)
                            full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
```