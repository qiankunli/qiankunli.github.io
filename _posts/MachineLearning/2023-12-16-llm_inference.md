---

layout: post
title: 大模型推理
category: 架构
tags: MachineLearning
keywords: large model

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## 简介

* TOC
{:toc}


## 思路

[大模型推理加速技术概要](https://mp.weixin.qq.com/s/kr5-QFhPXrUb7omTvJ-rDw)目前大模型推理加速技术栈大体可以分成三层（从低到高）：
1. 线性代数计算库，cuBLAS、Eigen、Intel MKL、ARM Compute Library等，其中定义了矩阵乘法、矩阵和向量乘法等数十个标准函数。线性代数层的加速主要依赖以下优化：
    1. GPU多核计算能力：通过调用CUDA、OpenCL等API，来利用GPU的并行能力。
    2. CPU SIMD和多核 ：单指令多数据SIMD在x86上有SSEx和AVX等指令，在ARM上有NEON和SVE，都广泛被使用，也有的库通过OpenMP再叠加多核能力。
    3. Tiling分块：矩阵乘法GEMM作为机器学习关键操作，可以通过Tiling的方法，大幅减少对于内存带宽的需求，提高速度。
    4. Autotuning自动调优：通过参数空间搜索，可以在多个分块办法和操作核之间自动优选适合本机的优化方案。
2. 模型推理引擎，TensorRT、TensorFlowServing、TVM等。 和线性代数层的优化不同，执行引擎能够看到整个神经网络的架构，也能够同时处理多个来自客户端的请求，所以可以使用涉及多个算子、整个模型，以及多个请求间的优化来提高执行效率。执行引擎一般有这些办法将模型推理进一步加速：
    1. Operator Fusion 算子融合：因为内存带宽往往是一大瓶颈，所以简单将多个相邻的算子找准机会合并起来计算，就可以减少对数据的扫描而大幅提升性能，所以Fusion是算子间优化的重要步骤，可以手工进行，也可以由执行引擎自动进行。
    2. Quantization 量化：随着GPU对数据结构支持的多元化，当前推理的基线数据类型已经是FP16，比几年前的FP32提高了不少速度。即便如此，将模型量化为INT8进行推理，依然可以提高较多速度，而在手机平台上，量化推理能进一步降低能耗。
    3. Distribution 分布式：使用多卡推理，以及通信加速，来提升能推理的模型规模和速度。
    4. Batching 批量化：将多个请求合并处理，是提高性能的另外一个关键办法，这个能大幅提高性能的原因主要有两个：1. 合并请求可以增大代数运算的矩阵规模，而下层代数库处理越大的矩阵规模，相对性能越高。2. 合并请求可以减少对静态的模型参数矩阵的扫描次数，减少内存带宽消耗。
3. 大模型调度引擎，vLLM、TensorRT-LLM（原FasterTransformer）、llama.cpp等。大模型调度引擎是2022年开始新出现的一层抽象。为什么有了执行引擎还需要大模型调度引擎？主要是因为大家希望进一步优化推理性能，而大模型架构相对固定（Transformer架构及变形），通过专门针对大模型而不是更通用的神经网络进行推理优化，就可以利用大模型架构的特点和算法特性，来进一步提高性能。
    1. KV Cache：这是fairseq等系统很早就开始有的基础方法，就是将transformer attention计算中的Key和Value张量集合缓存下来，避免每输出一个token都重复计算。
    2. Iteration-level scheduling 迭代层调度：这是2022年Orca引入的方法（参考文献1），推理引擎默认都是按请求批量化，而LLM推理需要多次迭代进行自回归计算，所以按“迭代”为单位进行批量化，可以提高并行度和性能。
    3. PagedAttention 分页注意力: 这是今年vLLM引入的方法（参考文献2），背后洞察是上面提到的KV cache占用大量GPU内存，一个13B模型每个输出token对应的KV张量，需要800KB，而最长输出长度2048个token的话，一个请求就需要1.6GB显存。因此vLLM引入类似操作系统中的分页机制，大幅减少了KV cache的碎片化，提高性能。
    4. GPTQ量化。有一批研究专注于寻找更优的量化方法，llama.cpp支持近期发表的GPTQ（参考文献3），默认将模型量化到4比特，大幅提升性能且准确率下降很小。
    5. Fused kernels等各类手工优化：很多时候，手打优化都是少不了的办法，llama.cpp短时间积累大量用户，就是因为项目作者不怕麻烦，快速积累了大量手工小优化，集腋成裘，形成领先的综合性能。

![](/public/upload/machine/vllm_arch.jpg)

1. vLLM是一个开源的大模型推理加速框架，通过PagedAttention高效地管理attention中缓存的张量，实现了比HuggingFace Transformers高14-24倍的吞吐量，就像在操作系统中管理CPU虚拟内存一样
2. NVIDIA FasterTransformer (FT) 是一个用于实现基于Transformer的神经网络推理的加速引擎。它包含Transformer块的高度优化版本的实现，其中包含编码器和解码器部分。使用此模块，您可以运行编码器-解码器架构模型（如：T5）、仅编码器架构模型（如：BERT）和仅解码器架构模型（如：GPT）的推理。FT框架是用C++/CUDA编写的，依赖于高度优化的 cuBLAS、cuBLASLt 和 cuSPARSELt 库，这使您可以在 GPU 上进行快速的 Transformer 推理。与 NVIDIA TensorRT 等其他编译器相比，FT 的最大特点是它支持以分布式方式进行 Transformer 大模型推理。在底层，节点间或节点内通信依赖于 MPI 、 NVIDIA NCCL、Gloo等。因此，使用FasterTransformer，您可以在多个 GPU 上以张量并行运行大型Transformer，以减少计算延迟。同时，TP 和 PP 可以结合在一起，在多 GPU 节点环境中运行具有数十亿、数万亿个参数的大型 Transformer 模型。
3. DeepSpeed-MII 是 DeepSpeed 的一个新的开源 Python 库，旨在使模型不仅低延迟和低成本推理，而且还易于访问。

当前的生成式大模型的推理可以分为两个阶段：Context 阶段和 Generation 阶段。Context 阶段是批量计算输入的 Prompt，属于计算密集型。Generation 阶段是逐字生成下一个 Token，属于访存密集型，虽然每一轮 Generation 的计算量小于 Context 阶段，但是访存量相当。大模型推理主要面临三个挑战：输入输出变长、计算规模大、显存占用大，针对这些挑战当前有多种优化手段进行优化：
1. 服务层面，打破之前的 Batch 只能同时返回结果的限制，允许部分请求结束后插入新的请求。
2. 计算方面，也有一些算子融合，KV Cache 这样的无损加速方案，也有模型量化加速方案，比如 Smooth Quant 量化方案将激活和权重的分布进行平衡来降低模型精度损失。
3. 显存方面，Generation 计算的访存密集型可以通过 Flash Attention 优化访存，也可以通过 Paged Attention 方法优化推理过程显存占用从而支持更大的吞吐。Paged Attention基本构建了一个类似于CPU内存管理的内存管理系统，以减少内存碎片并充分利用内存吞吐量
    1. 对于较短的文本输入 (词元数小于 1024)，推理的内存需求很大程度上取决于模型权重的大小。


### 影响因素

[语言大模型推理性能工程：最佳实践](https://mp.weixin.qq.com/s/mniKrBWkDE1tWWb2wQBDCA)
我们应该如何准确衡量模型的推理速度呢？首个词元生成时间（Time To First Token，简称TTFT）；单个输出词元的生成时间；时延：模型为用户生成完整响应所需的总时间；吞吐量：推理服务器在所有用户和请求中每秒可生成的输出词元数。
以下通用技术可用于优化语言大模型的推理：
1. 算子融合：将相邻的不同算子合并在一起通常可以获得更短的时延。
2. 量化：对激活值和权重进行压缩，以使用更少的比特数。一般来说，所有量化技术的工作原理如下: $Y=X*W$ 变成 $Y=X* dequantize(W); quantize(W)$，当输入向量走过模型计算图时，所有权重矩阵都会依次执行反量化和重量化操作。因此，使用权重量化时，推理时间通常 不会 减少，反而会增加。
3. 压缩：稀疏性或蒸馏。
4. 并行化：在多个设备间进行张量并行，或者针对较大的模型进行流水线并行。
除上述方法以外，还有许多针对Transformer的重要优化技术，如KV（键-值）缓存。[Transformer推理性能优化技术很重要的一个就是K V cache，能否通俗分析，可以结合代码? - 看图学的回答 - 知乎](https://www.zhihu.com/question/596900067/answer/3257946543)

![](/public/upload/machine/kv_cache.jpg)

提问【天王盖地虎，】的QKV实际上重复计算了很多遍。由于GPT是单向注意力，每层的提问的KV只根据上一层的提问的KV（或提问的嵌入向量）计算，不跟据回答中任何字符的KV计算，完全可以把它们缓存起来避免重复计算。至于为什么不缓存Q，因为推理场景下我们只取最后一个词，那么每层输出HS[-1]就可以了。HS[-1]根据全部的V和注意力矩阵的最后一行A[-1]计算，而A[-1]根据Q[-1]和全部的K计算，Q[-1]只根据输入最后一个字符X[-1]计算。所以我们通过传入KVCache保证K和V是完整的，输入字符只传入最后一个，也就是上一次GPT生成出来的字符，就可以了。

在LLM中，计算主要由矩阵乘法计算主导；这些维度较小的计算在大多数硬件上通常受内存带宽的限制。在以自回归方式生成词元时，激活矩阵的维度之一（由批大小和序列中的词元数定义）在小型批大小上较小。因此，速度由我们将模型参数从GPU内存加载到本地缓存/寄存器中的速度决定，而不是由计算加载数据的速度决定。相比峰值计算性能，推理硬件中可用和可实现的内存带宽能够更好地预测词元的生成速度。

对于服务成本来说，推理硬件的利用率非常重要。由于GPU的价格十分高昂，因此我们需要尽可能地让它们完成更多工作。共享推理服务通过将多个用户的工作负载组合在一起，填补各自的差距，并将重叠的请求进行批处理，以降低成本。对于LLaMA2-70B等大型模型，只有在较大的批大小下才能实现更好的性价比。拥有能够以较大批大小运行的推理服务系统对于成本效率至关重要。然而，较大的批大小意味着较大的KV缓存，这反过来又增加了部署模型所需的GPU数量。我们需要在这两者之间博弈，进行取舍，共享服务运营商需要权衡成本，优化系统。

### 在线推理

[揭秘大语言模型实践：分布式推理的工程化落地才是关键！](https://mp.weixin.qq.com/s/QeDmD-XlvkkJ7LMNJEynHg)与以往的模型不同，单张 GPU 卡的显存可能不足以支撑大语言模型。因此，需要使用模型并行技术，将大语言模型进行切分后，在多张 GPU 卡上进行推理。我们使用 DeepSpeed Inference 来部署大语言模型分布式推理服务。DeepSpeed Inference 是 Microsoft 提供的分布式推理解决方案，能够很好的支持 transformer 类型的大语言模型。。DeepSpeed Inference 提供了模型并行能力，在多 GPU 上对大模型并行推理。通过张量并行技术同时利用多个 GPU，提高推理性能。DeepSpeed 还提供了优化过的推理定制内核来提高 GPU 资源利用率，降低推理延迟。

有了大模型分布式推理方案，然而想要在 Kubernetes 集群中高效部署大模型推理服务，还存在很多工程化挑战，比如大规模的 GPU 等异构资源如何高效地管理运维和自动调度？如何快速部署推理服务，服务上线后如何保证资源能够应对波动的访问量？以及没有适合的工具进行推理服务时延、吞吐、GPU 利用率、显存占用等关键指标监控，没有合理的模型切分方案，模型版本管理等。

[大模型的好伙伴，浅析推理加速引擎FasterTransformer](https://mp.weixin.qq.com/s/Gkf_zIYWs4u7AJrJLDVq_Q) 未细读
FasterTransformer 是真对于 Transofrmer 类型模型（也包括 encoder-only、decoder-only）的推理加速方案，其提供了 Kernel Fuse、Memory reuse、kv cache、量化等多种优化方案，同时也提供了 Tensor Parallel 和 Pipeline Parallel 两种分布式推理方案。

### 分布式推理

1. 在提升模型显存使用效率方面，Flash Attention 和 Paged Attention 是两种常用的方法。在输入序列中，模型会根据每个词的重要性来分配显存。对于重要性较高的词，模型会分配更多的显存空间来存储其信息；而对于重要性较低的词，模型则会分配较少的显存空间。
2. 量化。量化过程主要涉及两个方面：参数环节的小型化和降低数据类型。通过这一步骤，我们能够使得模型加载的参数更小，从原本的 FP32 降低到 FP16，从而提高推理性能。在量化过程中，我们还会采用混合精度量化技术。这种技术能够在保证模型准确性的前提下，将异常值保留精度，并在混合精度分块矩阵最后再加回去。
3. 模型稀疏化。模型稀疏化是一种重要的优化方法。它的主要目的是减少模型参数的数量，从而降低模型的复杂度，提高模型的泛化能力和计算效率。模型稀疏化的主要方法有剪枝、量化、低秩近似等。剪枝是一种直接删除模型中部分参数的方法，它可以有效地减少模型的规模，但需要注意不能过度剪枝，以免影响模型的性能。低秩近似则是通过将模型转换为低秩矩阵，来减少模型的参数数量。

## 模型服务框架

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

### 简单封装

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

### FastChat

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

设计思路
1. 因为要支持不同的llm 库或加速库，比如Transformer、vllm等，且不同的llm在一些细节上有差异，因此推理侧必须有一个统一的LLM 抽象，在Fastchat里是XXModelWorker，在xinference 里是XXLLM
2. 将python llm 库 api化，一个api 要有一个api handler 函数，一般抽象为一个对象 作为api handler的载体，这个对象持有上面的XxLLM 执行chat/generate 方法，有时候还要支持/封装分布式、异步等细节。在Fastchat里是ModelWorker，在xinference 里是WorkerActor
3. 不同的llm 还有很多差别的（比如加载 load_model、运行chat/generate、模型配置转换），也有很多共性，所以模型设计的分层抽象很重要，Fastchat 的思路是 提供了一个ModelAdapter（主要差异化了加载） 和一个 generate_stream_gate 函数成员（差异化text生成），inference的思路是一个模型（比如chatglm、llama等）一个XXLLM
  1. 这里的模型配置转换说的是，比如一个chat message 包含role 和content 两个部分，role=system/user/assistant 各家各有差异，但因为对外提供的接口一般是openai 风格，所以有一个转换的过程。
4. 除了文本生成模型，还经常需要部署embedding模型、rerank模型、图生图、文生图等（入参出参与LLM 肯定不一样了），Fastchat 的方式是 让ModelWorker支持除了generate_stream_xx 外的get_embeddings、get_rerank方法，inference的思路除了LLM之外还定义了 EmbeddingModel、RerankModel等。

### FastChat源码分析

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

## 一些材料

推理加速
1. 模型优化技术
2. 模型压缩技术
3. 硬件加速
4. GPU加速
5. 模型并行化和分布式计算技术


[迈向100倍加速：全栈Transformer推理优化](https://mp.weixin.qq.com/s/1QlZ_d4BrAcD9YE9BEdyYg)本文回顾了从GPU架构到MLsys方法，从模型架构到解码算法的全栈Transformer推理优化方法。可以看出，大部分性能提升都来自于一个原则的利用：Transformer推理受内存限制，因此我们可以释放额外的计算能力/flops。其次，优化要么来自于优化内存访问，比如Flash Attention和Paged Attention，要么来自于释放计算能力，比如Medusa和前向解码。我们相信MLSys和建模仍有许多改进空间。在即将到来的2024年，随着模型变得更大、上下文变得更长以及随着更多开源MoE（混合专家模型）、更高内存带宽和更大内存容量的硬件，以及具有更大DRAM和专用计算引擎的移动设备的亮相，将出现更强大且人人可操作、可访问的AI。一个新时代即将到来。

GPU编程基础：在执行model.generate(prompt)时，我们进行以下操作： 

1. 内存访问:
    1. 从高带宽内存（HBM）加载模型权重到L2缓存，然后传输到SM（流处理器单元）
2. 计算:
    1. 在SM中执行矩阵乘法，SM请求张量核心执行计算
3. A100:
    1. 108个SM，DRAM容量为80G，40M L2缓存
    2. bf16张量核心：每秒312万亿浮点运算（TFLOPS）
    3. DRAM内存带宽为2039GB/秒 = 2.039T/秒
4. 如果模型很大，我们将其分割到多个GPU上，比如两个由NVLink连接的GPU
    1. NVLink 300GB/秒 = 0.3T/秒
    2. 我们大致观察了速度层次结构。尽管不能直接比较，但它们的数量级差异是我们需要优化的主要方面：
    3. 312T（SM计算） > 2.03T（DRAM内存访问） > 0.3T=300G（NVLink跨设备通信） > 60G（PCIe跨设备通信）

这意味着，如果我们希望速度更快，我们应该尽力：

1. 充分利用SM
2. 减少单个GPU的内存访问（因为它比计算慢得多），
3. 减少GPU之间的通信（因为它甚至比内存访问还要慢）。

调用model.generate(prompt)时有两个步骤：
1. 预填充：
    1. 为提示计算键值（kv）缓存。
    2. 这一步骤受计算限制，因为我们并行计算了一系列词元。
2. 解码：
    1. 自回归采样下一个词元。
    2. 这一步骤受内存限制，因为我们仅计算一个词元，未充分利用SM。







