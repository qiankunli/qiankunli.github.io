---

layout: post
title: Agent Functon Calling
category: 技术
tags: MachineLearning
keywords: llm agent

---

* TOC
{:toc}


## 简介


智能体的基本概念是在没有人工定义工作流（Workflow）的情况下，利用外部工具或功能，选择要执行的一系列操作。从技术角度来看，智能体通过大模型理解用户意图并生成结构化描述，进而执行相关操作。市场上现在出现了众多种类的智能体应用，其中大致可以分为两种主要的方式：以ReACT行动链为主的较为复杂的智能体结构，和以Function Calling（函数调用）模型为主的轻量级智能体结构。

OpenAI于23年6月份的更新的gpt-4-0613 and gpt-3.5-turbo-0613版本中为模型添加了Function Calling功能，通过给模型提供一组预定义的函数（Function list）以及用户提问（Query），让大模型自主选择要调用的函数，并向该函数提供所需的输入参数。随后我们就可以在自己的环境中基于模型生成的参数调用该函数，并将结果返回给大模型。然而Function Calling这种类型的智能体结构对模型有较高的要求，LLM模型必须进行针对性微调，以便根据用户提示检测何时需要调用函数，并使用符合函数签名的JSON进行响应；PS：否则就得用partialjson 来处理llm输出的不完整json。

[ChatGLM3 的工具调用（FunctionCalling）实现原理](https://zhuanlan.zhihu.com/p/664233831)使用Function Call功能时，你需要定义（并不是真的写程序去定义一个函数，而仅仅是用文字来描述一个函数）一些function（需要指定函数名，函数用途的描述，参数名，参数描述），传给LLM，当用户输入一个问题时，LLM通过文本分析是否需要调用某一个function，如果需要调用，那么LLM返回一个json，json包括需要调用的function名，需要输入到function的参数名，以及参数值。本质上是LLM（按特定意图）帮我们在做个文本结构化，Function calling 允许开发者更可靠的从模型中获得结构化数据，无需用户输入复杂的Prompt。实现上，把json格式的函数描述直接转换成了字符串，然后和其他输入一并送入LLM，**归根到底function call能力就是在prompt上边做了手脚**，微调时候有function call格式的数据。

## 与ReAct对比

1. 对模型的要求。Function Calling类的智能体对模型的要求相对较高。模型的训练数据必须包含function call相关的内容，以确保模型能够理解和生成结构化的输出。这类模型通常还需要具备更好的结构化输出稳定性，以及关键词和信息提取的能力。这意味着，模型需要较大的参数量，经过精细的调整和优化，才能满足Function Calling的需求。这种方式的优点在于，模型可以直接生成符合特定格式的数据，从而提高了解析和处理的效率。相比之下，ReACT框架对模型的要求则相对较低。ReACT不需要模型本身支持function calling格式的输出。在计划的生成过程中，它可以支持自然语言的规划文本，并在后续步骤中解析这些自然语言的输入。其优势在于，模型不需要进行复杂的结构化输出，只需生成自然语言即可。这使得模型的训练和优化过程更为简单，同时也降低了模型的出错率。
2. 对提示词的要求。Function Calling类的智能体结构通过微调模型来支持用户输入选择函数和结构化输入，这个过程其实这提高了输出稳定性，并简化了提示工程的复杂程度。相比之下，ReACT方式需要对模型进行更加细致的指导，让通用模型拥有输出规划、函数所需参数的能力，虽然这缓解了对模型本身输出能力的依赖，却增加了对提示工程的依赖，需要针对模型的特性来设计对应的提示模板，生成规划（函数的选择）和函数所需的API，并可能需要提供样例，消耗的上下文Token也相对更多一些。尽管Function Calling对模型的要求更高，但通过提示模板，普通的模型也可以具备简单的Function Calling的能力。通过在prompt中说明希望的输出格式和字段描述，大模型可以直接输出符合要求的内容。
3.  对推理的要求。在智能体结构的设计中，ReACT和Function Calling在推理要求上存在显著差异。Function Calling强调的是单/多步骤的JSON输出，而ReACT则允许LLM输出一个自然语言规划，这为智能体提供了思考的空间并能在后续的执行过程中动态地修正规划（Reflection）。Function Calling通过微调的模型，使其更擅长返回结构化输出。这种方法可以产生确定性的结果，同时降低错误率。然而，由于缺乏思维链，整个Function Calling的过程重度依赖于模型自身的推理性能，引发了大模型推理的重要问题 -- 缺乏解释性，整个推理过程相较于ReACT方式更加黑盒。相比之下，ReACT的CoT部分允许智能体在执行任务时进行更深入的推理和思考。这种方法使得智能体能够更准确地理解任务，并在执行过程中进行调整。在实际应用中，Function Calling和ReACT都可能需要执行多轮的拆解、推理和调用才能得到结果。Function Calling隐藏了Thought过程，而ReACT则更加开放和透明。这使得ReACT更适合需要高度定制和灵活性的应用场景，而Function Calling则更适合需要快速和确定性输出的场景。

## openai like api

以查询天气为例，当我们在openai的请求里添加了funtions相关的字段，他会增加一个判断是否需要调用function的环节。

```
curl --location 'http://xx:8000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer xx' \
--data '{
    "model": "xx",
    "messages": [
        {
            "role": "user",
            "content": "10月27日北京天气怎么样"
        }
    ],
    "functions": [
        {
            "name": "get_current_weather",
            "description": "获取今天的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "获取天气情况的城市或者国家，比如北京、东京、新加坡"
                    },
                    "time": {
                        "type": "string",
                        "description": "时间信息"
                    }
                },
                "required": [
                    "location",
                    "time"
                ]
            }
        }
    ],
    "stream": true
}'
```

以下以向ChatGPT输入“10月27日北京天气怎么样”为例：

1. 请求里没有functions字段得到的结果如下，他会告诉你一大段答案(应该是假的)，就是走Chatgpt正常的回答。
    ```
    根据天气预报，10月27日北京的天气预计为晴到多云，气温较低。最高气温约为16摄氏度，最低气温约为4摄氏度。需要注意保暖措施，适时添衣物。
    ```
2. 请求里如果有functions字段，返回了一个json，并帮我们从输入文本里抽取了get_current_weather所需要的location和time的函数值
    ```json
    {
        "id": "xx",
        "object": "chat.completion.chunk",
        "created": 1718021863,
        "model": "xx",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "function_call": {
                        "arguments": "{\"location\": \"北京\", \"time\": \"10 月 27 日\"}",
                        "name": "get_current_weather"
                    }
                },
                "finish_reason": "stop"
            }
        ]
    }
    ```
## OpenAI 原生调用

```python
 from openai import OpenAI
    client = OpenAI(base_url="xx",api_key="")
    completion = client.chat.completions.create(
        model="doubao-pro-4k",
        messages=[
            {"role": "user", "content": "How much does pizza salami cost?"}
        ],
        functions=[{
            "name": "get_pizza_info",
            "description": "Get name and price of a pizza of the restaurant",
            "parameters": {
                "properties": {
                    "pizza_name": {
                        "type": "string",
                        "description": "The name of the pizza, e.g. Salami"
                    }
                },
                "type": "object"
            }
        }],
        temperature=0.3,
    )
    print(completion.choices[0].message)
    # ChatCompletionMessage(content='', role='assistant', function_call=FunctionCall(arguments='{"pizza_name": "Salami"}', name='get_pizza_info'), tool_calls=None)
```

## 结合langchain调用

```python
from langchain import LLMMathChain
from langchain.agents import AgentType
from langchain_core.tools import tool

@tool
def get_pizza_info(query: str) -> str:
    """
    Get name and price of a pizza of the restaurant
    """
    pizza_info = {
        "name": query,
        "price": "10.99",
    }
    return json.dumps(pizza_info)

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate, SystemMessagePromptTemplate

llm = ChatOpenAI(model=LLM_MODEL)
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template('You are a helpful assistant'),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    HumanMessagePromptTemplate.from_template('{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])
tools = [get_pizza_info]
agent = create_openai_tools_agent(llm, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(agent_executor.invoke({'input': 'What is the capital of france?'})['output'])
print(agent_executor.invoke({'input': 'How much does pizza salami cost?'})['output'])
```


## AgentTuning

[如何生成Function Calling微调数据？](https://mp.weixin.qq.com/s/E1Y_WgvTrCLuQP1bb7c1Cw)

finetune 就是让LLM “更懂”特定的instruction。 [AGENTTUNING：为LLM启用广义的代理能力](https://zhuanlan.zhihu.com/p/664357514) 有一点粗糙。工具调用能力的获得离不开模型微调，不同于通过Prompt 诱导llm 按照特定格式 响应tool的名字，通过特定的训练样本（可以练习）强化llm 返回tool的名字（和tool调用参数）。**tool 信息入了LLM，意味着平时调用的 Prompt 可以少写点字，提高了执行效率**。ChatGLM3的训练工具调用的样本数据是如何构造的？

<pre>
<|system|>
Answer the following questions as best as you can. You have access to the following tools:
[
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    }
]
<|user|>
今天北京的天气怎么样？
<|assistant|>
好的，让我们来查看今天的天气
<|assistant|>get_current_weather
```python
tool_call(location="beijing", unit="celsius")
```
<|observation|>
{"temperature": 22}
<|assistant|>
根据查询结果，今天北京的气温为 22 摄氏度。
</pre>


