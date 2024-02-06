---

layout: post
title: 大模型推理服务框架
category: 架构
tags: MachineLearning
keywords: llm inference

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


* TOC
{:toc}

## 简介

1. vLLM是一个开源的大模型推理加速框架，通过PagedAttention高效地管理attention中缓存的张量，实现了比HuggingFace Transformers高14-24倍的吞吐量，就像在操作系统中管理CPU虚拟内存一样
2. NVIDIA FasterTransformer (FT) 是一个用于实现基于Transformer的神经网络推理的加速引擎。它包含Transformer块的高度优化版本的实现，其中包含编码器和解码器部分。使用此模块，您可以运行编码器-解码器架构模型（如：T5）、仅编码器架构模型（如：BERT）和仅解码器架构模型（如：GPT）的推理。FT框架是用C++/CUDA编写的，依赖于高度优化的 cuBLAS、cuBLASLt 和 cuSPARSELt 库，这使您可以在 GPU 上进行快速的 Transformer 推理。与 NVIDIA TensorRT 等其他编译器相比，FT 的最大特点是它支持以分布式方式进行 Transformer 大模型推理。在底层，节点间或节点内通信依赖于 MPI 、 NVIDIA NCCL、Gloo等。因此，使用FasterTransformer，您可以在多个 GPU 上以张量并行运行大型Transformer，以减少计算延迟。同时，TP 和 PP 可以结合在一起，在多 GPU 节点环境中运行具有数十亿、数万亿个参数的大型 Transformer 模型。
3. DeepSpeed-MII 是 DeepSpeed 的一个新的开源 Python 库，旨在使模型不仅低延迟和低成本推理，而且还易于访问。

使用大模型时，我们在huggingface或modelscope 看到的代码类似下面，很明显不能直接向用户提供服务。 
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:...
```

一般有几个需求
1. 统一api，这样切换模型时上游应用无感，最好是 OpenAI-compatible，其api 被主要上游框架（比如langchain）兼容
    1. 支持流式输出和普通输出
2. 支持多实例，进而支持灰度发布等
3. 支持通用的加速库比如vllm等

设计思路
1. 因为要支持不同的llm 库或加速库，比如Transformer、vllm等，且不同的llm在一些细节上有差异，因此推理侧必须有一个统一的LLM 抽象，在Fastchat里是XXModelWorker，在xinference 里是XXLLM
2. 将python llm 库 api化，一个api 要有一个api handler 函数，一般抽象为一个对象 作为api handler的载体，这个对象持有上面的XxLLM 执行chat/generate 方法，有时候还要支持/封装分布式、异步等细节。在Fastchat里是ModelWorker，在xinference 里是WorkerActor
3. 不同的llm 还有很多差别的（比如加载 load_model、运行chat/generate、模型配置转换），也有很多共性，所以模型设计的分层抽象很重要，Fastchat 的思路是 提供了一个ModelAdapter（主要差异化了加载） 和一个 generate_stream_gate 函数成员（差异化text生成），inference的思路是一个模型（比如chatglm、llama等）一个XXLLM
  1. 这里的模型配置转换说的是，比如一个chat message 包含role 和content 两个部分，role=system/user/assistant 各家各有差异，但因为对外提供的接口一般是openai 风格，所以有一个转换的过程。
4. 除了文本生成模型，还经常需要部署embedding模型、rerank模型、图生图、文生图等（入参出参与LLM 肯定不一样了），Fastchat 的方式是 让ModelWorker支持除了generate_stream_xx 外的get_embeddings、get_rerank方法，inference的思路除了LLM之外还定义了 EmbeddingModel、RerankModel等。

## 简单封装

[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 在github 有一个仓库，一般包含
1. 模型介绍 README.md
2. 模型的对外接口 api.py/cli_demo.py/web_demo.py。 自己使用 fastapi 基于python库直接对外提供RESTful APIs.

以api.py 为例
```python
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch

app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,prompt, history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {"response": response,"history": history,"status": 200,"time": time}
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
```

## FastChat

[一文入门最热的LLM应用开发框架LangChain](https://mp.weixin.qq.com/s/bYzNNL3F0998Do2Jl0PQtw)

FastChat功能覆盖训练，推理，评估的全过程。设计目标非常明确，就是在性能、功能及风格上全面对标OpenAI ChatGPT，以成为ChatGPT的开源平替。在生态集成上，由于它完全兼容OpenAI的风格，基于ChatGPT的langchain应用，可以无缝地使用FastChat替代。 推理侧类似工具Xinference/OpenLLM/RayLLM

[FastChat](https://github.com/lm-sys/FastChat)是一个用于训练、服务和评估基于聊天机器人的大型语言模型的开放平台。The core features include:
1. The training and evaluation code for state-of-the-art models (e.g., Vicuna).
2. A distributed multi-model serving system with web UI and OpenAI-compatible RESTful APIs.


```sh
# 命令行方式与llm 交互
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3
# webui方式与llm交互，此时需启动3个组件 web servers ==> controller ==> model workers
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.gradio_web_server
# 提供OpenAI-compatible RESTful APIs  openai_api_server ==> controller ==> model workers
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

![](/public/upload/machine/langchain_chatchat_call.jpg)

### FastChat源码分析

```
fastchat
    /fastchat
        /model
            /model_adapter.py        # BaseModelAdapter  ChatGLMAdapter
            /model_chatglm.py        # generate_stream_chatglm
            /model_exllama.py
            /model_registry.py
        /protocol
            /api_protocol.py
            /openai_api_protocol.py
        /serve
            /model_worker.py         # app = FastAPI()     ModelWorker
            /controller.py.          # app = FastAPI().    Controller
            /openai_api_server.py    # app = fastapi.FastAPI()
        /train
```

使用ModelWorker 加载model 提供http 接口 

```python
app = FastAPI()
@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    output = worker.generate_gate(params)
    release_worker_semaphore()
    return JSONResponse(output)
if __name__ == "__main__":
    ...
    worker = ModelWorker(...,args.model_path,)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
```
ModelWorker实现
```python
BaseModelWorker
     init_heart_beat
         # 将modelWorker id注册到controller，并保持心跳。均通过http接口

# 加载模型，调用模型（底层都是调用流式接口）
ModelWorker
     def __init__():
          self.model, self.tokenizer = load_model(model_path, device=device,...)
            # load_model 对应一个专门的 ModelAdapter 抽象，用来适配模型的加载
            adapter = get_model_adapter(model_path)
            model, tokenizer = adapter.load_model(model_path, kwargs)
     generate_stream_gate(self, params) 
     generate_gate(self, params)    # 根据参数返回输出，调用generate_stream_gate
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())
```
api => ModelWorker.generate_gate ==> ModelWorker.generate_stream_gate ==> ModelWorker.model.stream_generate
```python
generate_stream_gate
    get_generate_stream_function(model: torch.nn.Module, model_path: str)
       # 根据模型不同选择对应的函数 
       generate_stream_chatglm
            prompt = params["prompt"]
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            for total_ids in model.stream_generate(**inputs, **gen_kwargs):
                  response = tokenizer.decode(output_ids)
                  response = process_response(response)
```


### FastChat, How to support a new model?

1. FastChat uses the Conversation class to handle prompt templates and BaseModelAdapter class to handle model loading.
2. Implement a conversation template for the new model at `fastchat/conversation.py`. You can follow existing examples and use register_conv_template to add a new one. Please also add a link to the official reference code if possible. PS： 毕竟fastcaht 服务chat 场景嘛，对话请求传入的时候 一般是 `prompt = "\n###user:天为什么这么蓝？\n###"`，要把这个还原为 `history = [{"role": "user", "content": "天为什么这么蓝？"}]`，不同的模型 对role的称呼不同。
3. Implement a model adapter for the new model at `fastchat/model/model_adapter.py`. You can follow existing examples and use register_model_adapter to add a new one. PS：不同的模型加载时有一些特定的参数，比如 chatglm 的trust_remote_code 参数，`model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **from_pretrained_kwargs)`
4. ModelWorker 主要逻辑是执行 `generate_stream(model,tokenizer,params)` ，很常规的 `input_ids = tokenizer(prompt); output_ids = model(input_ids,xx)`。 如果模型的generate 逻辑有一些特别的处理，则需要自定义generate_stream_xx，并加入get_generate_stream_function 逻辑（根据模型名等 路由到不同的generate_stream_xx）
5. (Optional) add the model name to the "Supported models" section above and add more information in `fastchat/model/model_registry.py.`

如何理解FastChat 都干了什么？本质是对下面的 原始的大模型推理代码进行抽象（模型加载、模型推理=tokenizer+model）和封装，对外提供rest api。

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:...
```

## vllm 源码分析

[vLLM代码及逻辑介绍](https://zhuanlan.zhihu.com/p/675322419)

[如何让vLLM适配一个新模型](https://zhuanlan.zhihu.com/p/680636375)




