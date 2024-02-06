---

layout: post
title: 大模型推理
category: 架构
tags: MachineLearning
keywords: llm inference

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
    2. Iteration-level scheduling 迭代层调度：这是2022年Orca引入的方法，推理引擎默认都是按请求批量化，而LLM推理需要多次迭代进行自回归计算，所以按“迭代”为单位进行批量化，可以提高并行度和性能。
    3. PagedAttention 分页注意力: 这是今年vLLM引入的方法（参考文献2），背后洞察是上面提到的KV cache占用大量GPU内存，一个13B模型每个输出token对应的KV张量，需要800KB，而最长输出长度2048个token的话，一个请求就需要1.6GB显存。因此vLLM引入类似操作系统中的分页机制，大幅减少了KV cache的碎片化，提高性能。
    4. 低比特量化。传统的量化方法分为Quantization-Aware Training (QAT) 和 Post-Training Quantization (PTQ)。PTQ主要是对模型权重值和激活值进行INT8/INT4量化，QAT一般效果是更好的，但是它需要重训模型所以成本会大一些，相关的研究成果相较PTQ也少一些，在fintune阶段会用的比较多一些，例如 QLoRA。GPTQ量化。有一批研究专注于寻找更优的量化方法，llama.cpp支持近期发表的GPTQ（参考文献3），默认将模型量化到4比特，大幅提升性能且准确率下降很小。
    5. Fused kernels等各类手工优化：很多时候，手打优化都是少不了的办法，llama.cpp短时间积累大量用户，就是因为项目作者不怕麻烦，快速积累了大量手工小优化，集腋成裘，形成领先的综合性能。

![](/public/upload/machine/vllm_arch.jpg)

## 影响因素

[语言大模型推理加速指南](https://mp.weixin.qq.com/s/B3TD2p_5HKoYkzzupLoUxQ)

### 算法优化

每个周期的generate函数都必须处理逐渐增多的词元，因为每个周期我们都要在上下文中添加新词元。这意味着要从一个由10个词元组成的提示生成100个词元，你需要在模型上运行的不止109个词元，而是10 + 11 + 12 + 13 + ... + 109 = 5950个词元！（初始提示可以并行处理，这也是为什么在推理API中提示词元通常更便宜的原因之一。）同时也意味着，随着生成的进行，模型会越来越慢，因为每个连续的词元生成都有一个越来越长的前缀。

### 系统工程优化

1. 低比特量化，通过是否需要重新训练分为QAT和PTQ，围绕QAT和PTQ也衍生了很多经典算法
2. 并行计算，大多数模型并行都是应用在分布式训练场景，不过像PaLM inference继承了TP的思想是应用在大规模Transformer推理场景的。
3. 内存管理，随着KV Cache优化变成一个事实标准，对存储的使用需求越来越大。这里主要提三个系统优化，vLLM使用的pagedAttention优化、SpecInfer提出的tree attention优化和LightLLM采用的更精细的标记级内存管理机制。但这些优化在提高吞吐量的同时会牺牲一部分请求响应延时。
4. 请求调度，早期的LLM 推理框架只支持request-level的调度，比如像fastertransformer这种，ORCA率先提出了iteration-level的调度系统，结合selective-batching输出了更高效的调度能力。后来continuous batching逐步应用到vLLM、TensorRT-LLM(flight batching)和RayLLM上，也成为了事实标准。
5. 内核优化，即内核融合、定制attention、采样优化（采样算法的选择会极大地影响LLM的生成质量）、变长序列优化和自动编译优化。

当前的生成式大模型的推理可以分为两个阶段：Context 阶段和 Generation 阶段。Context 阶段是批量计算输入的 Prompt，属于计算密集型。Generation 阶段是逐字生成下一个 Token，属于访存密集型，虽然每一轮 Generation 的计算量小于 Context 阶段，但是访存量相当。大模型推理主要面临三个挑战：输入输出变长、计算规模大、显存占用大，针对这些挑战当前有多种优化手段进行优化：
1. 服务层面，打破之前的 Batch 只能同时返回结果的限制，允许部分请求结束后插入新的请求。
2. 计算方面，也有一些算子融合，KV Cache 这样的无损加速方案，也有模型量化加速方案，比如 Smooth Quant 量化方案将激活和权重的分布进行平衡来降低模型精度损失。
3. 显存方面，Generation 计算的访存密集型可以通过 Flash Attention 优化访存，也可以通过 Paged Attention 方法优化推理过程显存占用从而支持更大的吞吐。Paged Attention基本构建了一个类似于CPU内存管理的内存管理系统，以减少内存碎片并充分利用内存吞吐量
    1. 对于较短的文本输入 (词元数小于 1024)，推理的内存需求很大程度上取决于模型权重的大小。

[语言大模型推理性能工程：最佳实践](https://mp.weixin.qq.com/s/mniKrBWkDE1tWWb2wQBDCA)
我们应该如何准确衡量模型的推理速度呢？首个词元生成时间（Time To First Token，简称TTFT）；单个输出词元的生成时间；时延：模型为用户生成完整响应所需的总时间；吞吐量：推理服务器在所有用户和请求中每秒可生成的输出词元数。
以下通用技术可用于优化语言大模型的推理：
1. 算子融合：将相邻的不同算子合并在一起通常可以获得更短的时延（避免了反复从HBM中读写数据）。
2. 量化。浮点数有不同的大小，这对性能很重要。对于常规软件（例如 JavaScript 数字和 Python 浮点数），我们大多使用64位（双精度）IEEE 754 浮点数。然而，大多数机器学习一直以来都使用的是32位（单精度）IEEE 754。一个70 亿参数的模型在fp64下需要占用56GB，而在fp32下只需28GB。在训练和推理期间，大量的时间都用来将数据从内存搬运到缓存和寄存器，所以搬运的数据越少越好。截至2023年，GPU不支持小于fp16的数据格式，除了int8（8 位整数）。
    1. 对激活值和权重进行压缩，以使用更少的比特数。一般来说，所有量化技术的工作原理如下: $Y=X*W$ 变成 $Y=X* dequantize(W); quantize(W)$，当输入向量走过模型计算图时，所有权重矩阵都会依次执行反量化和重量化操作。因此，使用权重量化时，推理时间通常 不会 减少，反而会增加。
3. 压缩：稀疏性或蒸馏。
4. 并行化：在多个设备间进行张量并行，或者针对较大的模型进行流水线并行。

在LLM中，计算主要由矩阵乘法计算主导；这些维度较小的计算在大多数硬件上通常受内存带宽的限制。在以自回归方式生成词元时，激活矩阵的维度之一（由批大小和序列中的词元数定义）在小型批大小上较小。因此，速度由我们将模型参数从GPU内存加载到本地缓存/寄存器中的速度决定，而不是由计算加载数据的速度决定。相比峰值计算性能，推理硬件中可用和可实现的内存带宽能够更好地预测词元的生成速度。

对于服务成本来说，推理硬件的利用率非常重要。由于GPU的价格十分高昂，因此我们需要尽可能地让它们完成更多工作。共享推理服务通过将多个用户的工作负载组合在一起，填补各自的差距，并将重叠的请求进行批处理，以降低成本。对于LLaMA2-70B等大型模型，只有在较大的批大小下才能实现更好的性价比。拥有能够以较大批大小运行的推理服务系统对于成本效率至关重要。然而，较大的批大小意味着较大的KV缓存，这反过来又增加了部署模型所需的GPU数量。我们需要在这两者之间博弈，进行取舍，共享服务运营商需要权衡成本，优化系统。

KV Cache/PagedAttention/FlashAttention 本质是基于GPU计算和内存架构，对注意力计算过程的优化

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

### KV Cache

有许多针对Transformer的重要优化技术，如KV（键-值）缓存，每个Transformer层有一个KV缓存。但是有一个问题就是KV Cache非常的大，比如说拿LLaMA-13B举例子，假设每个token在一层的大小有20KB，LLaMA-13B有40层，这样这个token大小就会达到800KB，而一个sequence一般来说会有几千的token，也就是说一个sequence就会达到几个G。[Transformers KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249) 动图实在太贴切了。

```
K = X * W_k
V = X * W_v
Q = X * W_q
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V

# 初始化缓存
K_cache = []
V_cache = []
for x in X:  # 对每一个新的输入x
    # 计算新的K和V
    K_new = x * W_k
    V_new = x * W_v
    # 将新的K和V添加到缓存中
    K_cache.append(K_new)
    V_cache.append(V_new)
    # 使用缓存中的K和V计算Attention
    output = Attention(Q, K_cache, V_cache)
```

Memory waste in KV Cache
1. 内部碎片：推理过程具有非常大的动态性，输出的长度不能预先知道，传统的serving system为了保险起见，就会预留非常大的空间，比如模型支持的最大输出2048个token，它就会预留这么大的空间，那么如果我产生的输出仅有10个token，剩下的2038的slots就会作为内部碎片被浪费。
2. 外部碎片：因为每个request长度不等，就像os中的malloc，长度不等，不停的malloc就会产生外部碎片。

[如何解决LLM大语言模型的并发问题？](https://www.zhihu.com/question/613263140/answer/3271554389)vLLM：Efficient memory management for LLM inference 受到操作系统中的分页和虚拟内存的启发，将KV Block当做页，将Request当做进程，允许在非连续的内存空间中存储连续的KV。PagedAttention机制：传统要求将keys和values存到连续的内存空间，因为我们知道传统大家都是用TensorFlow、pytorch之类的，它是一个个tensor，所以很自然的就假设给它们一段连续的内存空间，但是对于LLM来说，这个假设就不是一个好的假设，因此PagedAttention允许在非连续内存空间中存储连续的keys和values，vLLM维护一个Block table，存放逻辑空间到物理空间的映射。现在有一个Prompt：Alan Turing is a computer scientist，当产生一个新的token时，会查看Block table中的Physical block no.,然后找到对应物理内存的地方存储进去，并更新Block table中的Filled slots内容。当产生“renowned”的时候，是新开了一个Block，所以也要更新Block table，新开一个物理内存（每个kv block中有固定的token数目）。

![](/public/upload/machine/paged_attention.gif)

PS：Transformer （和Attention） layer 已经支持了缓存机制 (use_cache=true)，kvcache 在代码上如何体现可以理解。pageattention 是不是可以理解为：pageattention 初始化了cache，只要把这个cache 引用传给 Transformer （和Attention） forward 函数参数，Transformer 就可以用这个cache 到计算过程中了？


### Flash Attention

[图解大模型计算加速系列：Flash Attention V1，从硬件到计算逻辑](https://mp.weixin.qq.com/s/J2i2MDv4us_GMwCyku0tnw)
1. Fast（with IO-Awareness），计算快。它发现：计算慢的卡点不在运算能力，而是在读写速度上。所以它通过**降低对显存（HBM）的访问次数**来加快整体运算速度（通过分块计算（tiling）和核函数融合（kernel fusion）来降低对显存的访问），这种方法又被称为IO-Awareness。
2. Memory Efficicent，节省显存。在标准attention场景中，forward时我们会计算并保存N*N大小的注意力矩阵；在backward时我们又会读取它做梯度计算，这就给硬件造成了的存储压力。在Flash Attention中，则巧妙避开了这点，使得存储压力降至。在后文中我们会详细看这个trick。
3. Exact Attention，精准注意力。

我们知道显存的带宽相比SRAM要小的多，读一次数据是很费时的，但是SRAM存储又太小，装不下太多数据。所以我们就以SRAM的存储为上限，尽量保证每次加载数据都把SRAM给打满，能合并的计算我们尽量合并在一起，节省数据读取时间。举例来说，我现在要做计算A和计算B。在老方法里，我做完A后得到一个中间结果，写回显存，然后再从显存中把这个结果加载到SRAM，做计算B。但是现在我发现SRAM完全有能力存下我的中间结果，那我就可以把A和B放在一起做了，这样就能节省很多读取时间，我们管这样的操作叫kernel融合。kernel包含对线程结构（grid-block-thread）的定义，以及结构中具体计算逻辑的定义。flash attention将矩阵乘法、mask、softmax、dropout操作合并成一个kernel，做到了只读一次和只写回一次，节省了数据读取时间。

### 调度优化

Batching就是将一段时间内到达的用户请求合并到一起，提交到GPU中执行，从而提高系统的吞吐量。然而，**与传统的 DNN Model 在推理时只要正向执行一遍不同，基于 Transformer 的 Generative Model 在推理时是迭代式的（Iterative），每个请求都需要迭代式执行多次，每次生成部分结果（一个 Token），且每个请求的迭代次数可能是不同的（例如迭代直到模型生成一个 End-Of-Sequence Token）**。因此将现有的 Batching 方式应用在 Generative Model 时，可能导致有的请求已经迭代结束了，但是还需要和同Batch中没有迭代结束的请求继续一起执行。这个问题的核心在于，传统的 Batching 技术是以 Request 为粒度的，将多个 Request 绑定在一起提交给执行引擎，多个 Request 同时开始同时结束。因此需要一个新的 Batching 的方式，这也是本项工作核心的 Insight：使用更细粒度的，Iteration-level Batching，在每个 Iteration 中将不同的 Request 合并到一起。

![](/public/upload/machine/iteration_level_batching.jpg)

为了进行批次生成，我们改为一次向模型传递多个序列，在同一前向传递中为每个序列生成一个补全（completion），这需要在左侧或右侧使用填充词元对序列进行填充，使它们达到相同的长度。填充词元（可以是任何词元，我这里使用 `[end]`）在注意力掩码中被屏蔽，以确保它们不会影响生成。

但在上面的例子中，请注意 “Mark is quick. He moves quickly.” 在其他序列之前完成，但由于整个批次尚未完成，我们被迫继续为其生成词元（“Random”）。这并不影响准确度，我们只需简单地将生成的序列截断到 `[end]` 词元即可，但这样很浪费资源，因为GPU资源正在用于生成我们即将丢弃的词元。连续批处理通过将新序列插入批次来解决这一问题，插入位置是 `[end]` 词元之后。在 `[end]` 词元之后生成随机词元的代替方案是，在批次的相应行中插入新序列，并使用注意力掩码机制来防止该序列受到上一序列中词元的影响。（实际上，先前的序列充当了额外的填充内容。）


## 一些材料

[大模型推理加速技术的学习路线是什么? ](https://www.zhihu.com/question/591646269/answer/3333428921)

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


函数计算推出 GPU 闲置计费功能，在保障性能的前提下，可以帮助您大幅降低 GPU 的成本开销。以往部署大型语言模型（LLM）可能需要昂贵的 GPU 支持，尤其在需要大量计算资源时。但请求处理并不是每时每刻都处于活跃状态，势必存在流量的潮汐现象，后端的计算资源会出现空载导致成本的浪费。借助函数计算 GPU 闲置计费功能，用户的开销将会根据实际计算负载动态调整。

### 在线推理

[揭秘大语言模型实践：分布式推理的工程化落地才是关键！](https://mp.weixin.qq.com/s/QeDmD-XlvkkJ7LMNJEynHg)与以往的模型不同，单张 GPU 卡的显存可能不足以支撑大语言模型。因此，需要使用模型并行技术，将大语言模型进行切分后，在多张 GPU 卡上进行推理。我们使用 DeepSpeed Inference 来部署大语言模型分布式推理服务。DeepSpeed Inference 是 Microsoft 提供的分布式推理解决方案，能够很好的支持 transformer 类型的大语言模型。。DeepSpeed Inference 提供了模型并行能力，在多 GPU 上对大模型并行推理。通过张量并行技术同时利用多个 GPU，提高推理性能。DeepSpeed 还提供了优化过的推理定制内核来提高 GPU 资源利用率，降低推理延迟。

有了大模型分布式推理方案，然而想要在 Kubernetes 集群中高效部署大模型推理服务，还存在很多工程化挑战，比如大规模的 GPU 等异构资源如何高效地管理运维和自动调度？如何快速部署推理服务，服务上线后如何保证资源能够应对波动的访问量？以及没有适合的工具进行推理服务时延、吞吐、GPU 利用率、显存占用等关键指标监控，没有合理的模型切分方案，模型版本管理等。

[大模型的好伙伴，浅析推理加速引擎FasterTransformer](https://mp.weixin.qq.com/s/Gkf_zIYWs4u7AJrJLDVq_Q) 未细读
FasterTransformer 是真对于 Transofrmer 类型模型（也包括 encoder-only、decoder-only）的推理加速方案，其提供了 Kernel Fuse、Memory reuse、kv cache、量化等多种优化方案，同时也提供了 Tensor Parallel 和 Pipeline Parallel 两种分布式推理方案。

### 分布式推理

1. 在提升模型显存使用效率方面，Flash Attention 和 Paged Attention 是两种常用的方法。在输入序列中，模型会根据每个词的重要性来分配显存。对于重要性较高的词，模型会分配更多的显存空间来存储其信息；而对于重要性较低的词，模型则会分配较少的显存空间。
2. 量化。量化过程主要涉及两个方面：参数环节的小型化和降低数据类型。通过这一步骤，我们能够使得模型加载的参数更小，从原本的 FP32 降低到 FP16，从而提高推理性能。在量化过程中，我们还会采用混合精度量化技术。这种技术能够在保证模型准确性的前提下，将异常值保留精度，并在混合精度分块矩阵最后再加回去。
3. 模型稀疏化。模型稀疏化是一种重要的优化方法。它的主要目的是减少模型参数的数量，从而降低模型的复杂度，提高模型的泛化能力和计算效率。模型稀疏化的主要方法有剪枝、量化、低秩近似等。剪枝是一种直接删除模型中部分参数的方法，它可以有效地减少模型的规模，但需要注意不能过度剪枝，以免影响模型的性能。低秩近似则是通过将模型转换为低秩矩阵，来减少模型的参数数量。



