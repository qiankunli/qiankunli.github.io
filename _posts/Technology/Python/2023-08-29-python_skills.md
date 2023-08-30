---

layout: post
title: Python实践
category: 技术
tags: Python
keywords: Python skills

---

* TOC
{:toc}

## 前言 


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