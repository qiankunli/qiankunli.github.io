---

layout: post
title: 大模型推理系统：引擎、调度与分布式架构
category: 架构
tags: MachineLearning
keywords: llm vLLM

---

* TOC
{:toc}

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$']], // 支持 $和$$ 作为行内公式分隔符
      displayMath: [['$$', '$$']], // 块级公式分隔符
    },
    svg: {
      fontCache: 'global'
    }
  };
</script>
<script async src="/public/js/mathjax/es5/tex-mml-chtml.js"></script>

## 简介

[万字详解大模型推理加速核心原理：分形规律与资源计算公式](https://mp.weixin.qq.com/s/ytNCNlWJzXCRx7DvTxfp4A) 未细读

[AI Infra 和传统 Infra 断代了吗？](https://zhuanlan.zhihu.com/p/1916694074750132757)从表面看起来，AI Infra 确实和传统 Infra 很不一样
1. 传统 Infra 处理的是 web request，数据存储，和分布式服务协调，而 AI Infra（特别是大模型）更多围绕的是 GPU 推理，KV Cache 管理，以及大模型训练框架等全新领域。
2. 请求形态也不一样：web request通常是毫秒级的request，stateless，而 LLM 推理一个 session 往往持续数秒甚至更久(随着context window 和模型大小增加），还要动态维护 token-level 的上下文状态。
3. tech stack 看起来也不同：传统用的是 Kubernetes + Docker，现在大家在用 GPU, vLLM, DeepSpeed, FlashAttention, Triton, NCCL 这些仅仅从名字上听起来就很高大上的架构。
本质其实没变，仍然是系统设计和资源调度的问题。回到工程本身，其实我们仍然在面对和传统 Infra 极其类似的问题：
1. 如何调度资源（从CPU/内存 变成了 GPU 显存）
2. 如何处理高并发请求（从http resource request ，变成了 prompt request）
拿 vLLM 举个例子：它像是给 LLM 写了一个操作系统，用来调度“页面”（KV Cache），管理“进程”（requests), 本质上是引用了OS 的内存管理principles用来管理 kv cache。

Infra 的“三大难题”：Scaling, Sharding, and Copying所有系统的底层挑战，基本都绕不开这三个关键词：
1. Scaling（扩展）：系统如何支持更大的规模和更高的并发？
    1. 在传统 Infra 中，这意味着如何横向扩展服务器，部署更多容器，使用负载均衡 (load balancing) 来分散请求
    2. 在 AI Infra 中，这些问题转化为如何通过 数据并行， 模型并行，流水线并行 来分布和执行 GPU workload，以支持超大模型的训练以及 large number of inference requests
2. Sharding（切片）：系统如何切分状态和计算，以实现并行处理？
    1. 在数据库系统中，这是将数据按照主键或范围切分到不同的分区，以支持高吞吐访问
    2. 在 AI Infra 中，sharding 变成了对模型参数，KV Cache，activation，gradients，以及optimizer states的split，比如tensor parallelism和KV paging等，是实现分布式推理和训练的前提
3. Copying（复制）：系统如何高效同步数据或状态？
    1. 传统系统中，复制体现在数据库副本同步或者缓存预热，以及Kafka Replication
    2. 在 AI Infra 中，复制的代价更加显著，比如data parallelism 怎么copy model to different GPUs（所以会有ZeRO optimization 来shard 参数，gradient 等等），**通常需要依赖高性能通信机制（比如 RDMA和NCCL）**
这些挑战的本质没有变：仍然是如何高效并且低成本地协调跨不同机器的资源。但在 AI Infra 中，由于gpu显存limited，large context window，以及模型参数量大，它们变得更加脆弱和重要，也更需要更好的工程策略去解决这些问题。

真正有经验的 Infra 工程师，不仅仅是能搭件一个working的系统，而是有能力去从头到尾追踪每一个延迟点，把系统之间的关联和可能存在的bottleneck拆解成一系列可量化的问题，并在上线后持续做 cost/performance profiling。这正是 AI Infra （或者传统 Infra）对工程基本功要求极高的原因。

## 技术栈

[大模型推理加速技术概要](https://mp.weixin.qq.com/s/kr5-QFhPXrUb7omTvJ-rDw)目前大模型推理加速技术栈大体可以分成三层（从低到高）：
1. 线性代数计算库，cuBLAS、Eigen、Intel MKL、ARM Compute Library等，其中定义了矩阵乘法、矩阵和向量乘法等数十个标准函数。线性代数层的加速主要依赖以下优化：
    1. GPU多核计算能力：通过调用CUDA、OpenCL等API，来利用GPU的并行能力。
    2. CPU SIMD和多核 ：单指令多数据SIMD在x86上有SSEx和AVX等指令，在ARM上有NEON和SVE，都广泛被使用，也有的库通过OpenMP再叠加多核能力。
    3. Tiling分块：矩阵乘法GEMM作为机器学习关键操作（矩阵乘还经常用作张量形状的变换），Tiling（平铺）是一种优化技术，它涉及将大的矩阵分解成更小的块或“瓦片”（tiles），**这些小块的大小通常与CPU或GPU的缓存大小相匹配**，以便可以完全加载到缓存中
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

## 模型文件格式

一些常见的大模型权重存储格式：
1. PyTorch格式，pth, .pt, .bin：这是PyTorch框架的原生格式，广泛使用但安全性相对较低。
2. Safetensors格式，.safetensors：由Hugging Face开发，安全性更高，加载速度更快，支持内存映射，是目前推荐的格式。
3. GGUF格式，.gguf：由Georgi Gerganov定义的一种大模型文件格式，旨在快速加载和保存模型，适用于CPU推理。
4. ONNX格式，.onnx：开放神经网络交换格式，旨在实现不同深度学习框架之间的互操作性。
5. TensorFlow格式，.h5, .pb, .ckpt：TensorFlow使用的格式，包括HDF5格式、Protocol Buffers格式和Checkpoint文件。
6. 量化格式，.gguf (llama.cpp使用), .ggml (旧版GGML), model-q4_0.bin (4-bit量化), model-q8_0.bin (8-bit量化)：量化格式通过降低权重和激活值的精度来减小模型大小和内存占用，但可能略微降低精度。
7. 分片格式， pytorch_model-00001-of-00003.bin：用于存储超大型模型，将权重文件分割成多个部分。

## 推理引擎

[当我们谈论 AI 推理的 KV Cache，我们在说什么？](https://mp.weixin.qq.com/s/N44kMpWtgmfqLB4S5MUjaA)回过头来看，vLLM 最大的贡献是什么呢？我认为更多地在于它定义了一个推理框架应该具备的基础功能模块，譬如围绕着虚拟内存建设的高效 KV Cache 管理模块，譬如考虑 KV Cache 分布情形来做相应请求路由的灵活的 Scheduler 调度模块，譬如对各类解码算法的支持，parallel sampling，beam search，等等，譬如一个多级存储架构（从 GPU VRAM，到 CPU DRAM，再到 NVMe Storage，甚至是 Storage Backends）以支持 KV Cache 根据其热度在不同层之间的高效流动，......更为重要的是，vLLM 展示了 KV Cache 的确可以是 LLM Inference 的性能胜负手，此处的优化空间巨大。此番意义，不亚于茫茫的淘金时代，当众人在苦苦找寻的时候，忽然有人喊了一声：“我这里发现金子了！”，于是乎，众人皆蜂拥而至 ......

### 通用流程的抽象 

`前处理 → DNN推理 → 后处理`，无论是分类（classification）、检测（detection）、分割（segmentation）还是姿态估计（pose estimation）等任务，这一流程都是适用的。差异主要体现在前处理和后处理的具体实现上。
引擎创建： `builder → network → config → parser → serialize → save file`。 network 估计指的model 计算图解析和加载
引擎推理：`load file → deserialize → engine → context → enqueue`。file 估计指的是图片文件

**为实现代码的可复用性，我们可以采用面向对象的编程思想**，将通用的流程和操作封装在基类中，不同的任务通过继承和重写基类的方法，实现各自的特定逻辑。

```c++
class InferenceEngine {
public:
    virtual void buildEngine() = 0;
    virtual void loadEngine(const std::string& engineFile) = 0;
    virtual void preprocess(const cv::Mat& image) = 0;
    virtual void infer() = 0;
    virtual void postprocess() = 0;
    virtual ~InferenceEngine() {}
};
```

### 请求调度

1. 静态请求级调度，传入的请求在到达时被分组到批次中并一同处理。批次中的所有请求都会并行处理，**新请求需要等到当前批次的所有请求完成后才能被处理**。这种方法虽然简单，但会导致低效，尤其是当单个请求具有不同的输入和输出长度时。较短序列的请求会因为批次内最长运行请求的完成而受到显著延迟。
2. 迭代级调度。将任务分解为更小的单位，称为“ iterations”，而非对整个请求进行调度。在自回归模型中，迭代通常被定义为生成一个单独的 token。迭代级调度能显著提升计算效率，因为请求的 token 长度通常不同。提前完成的请求可以让新的请求加入批次，而不必等到整个批次完成。这种方式减少了硬件资源的空闲时间，并提高了整体吞吐量，尤其是在请求之间的 token 数量不同的工作负载中。
    1. Packed Batching是高效执行已调度请求的另一个关键组件，尽管它本身并不是一种调度技术。迭代级调度通常需要在同一迭代中处理预填充阶段和解码阶段。这两个阶段在输入 token 大小方面差异显著：预填充阶段一次处理多个 token，而解码阶段每次只处理一个 token，即前一迭代的输出 token。当预填充阶段的请求和解码阶段的请求被分组到同一批次中时，这种 token 长度的不一致会导致大量填充。即使所有批次请求都处于预填充阶段，由于输入 token 长度不同，也会产生填充。这种填充会降低计算效率，因为填充的额外计算实际上是无效的。
3. Continuous Batching 或 In-flight Batching，通过集成迭代级批处理和Packed Batching，我们得到了 vLLM 和 TensorRT-LLM 调度器的核心：Continuous Batching（也称为“In-flight Batching”）。这种方法旨在最大限度地减少队列等待时间并减少填充开销，从而提高硬件利用率和服务性能。
    1. Continuous batching 还有个别名，叫做：batching with iteration-level scheduling，这里的 iteration 就是指一次 decode 计算。也就是说**在每次 decode 的迭代过程中，做 batch 的调度调整**。但调度本身不是无代价的，它可能涉及到接收和处理新的输入请求，重新组织输入数据的形状，甚至各种状态的重新初始化，这些都需要消耗 CPU 时间。这也就意味着在这段时间里，GPU 是闲着的，GPU 没有得到充分利用。所以在实现时，程序并不会真的每个 iteration 都做 scheduling，目前看到有两种做法：间隔调度：比如每 16 次 decode 计算后，检查一下是否有新的输入，以及是否有空闲的槽位，然后对 batch 做一次调度调整。排队比例调度。比如当前 batch 中有 10 个请求的 decode 正在进行，而排队中有 12 个请求，超过了排队比例 1.2，那么就启动一次调度调整。
4. vllm在continuous batching基础上引入了两项独特的改进：不使用混合批处理（no mixed batching）以及优先处理 prefill 请求（prefill prioritization）。PS：分成共存派（每个iteration中既有prefill又有decode）和独立派（每个iteration中只有prefill或decode）
    1. 不使用混合批处理，这意味着 prefill 请求只会与其他 prefill 请求一起进行批处理，而 decode 请求只会与其他 decode 请求一起处理。这种设计简化了计算路径，因为每个批次仅处理相同阶段的请求。由于没有混合批处理，调度器必须采用另一种策略：优先处理 prefill 请求。
    2. 为什么需要优先处理 prefill 请求：假设当前批次中的某个请求完成了，而请求池中还有新的请求等待加入批次。由于不支持混合批处理，新的请求无法直接加入当前批次，因为它需要先完成 prefill 阶段，才能进入 decode 阶段。因此，新的请求无法与当前正在处理的 decode 请求一起被批处理。这种限制破坏了连续批处理的概念。为了解决这一问题，当前批次的 decode 请求需要暂时延后处理，先处理 prefill 请求，以确保连续批处理的流程不被中断。因此，为了在后续的 decode 迭代中确保有足够的 decode 请求可以处理，必须优先调度 prefill 请求。但是会设置一个阈值防止 decode 一直等待。也就是优先确保 TTFT，然后设置一个阈值来保证 TPOT 不会太差。很明显，这只是一个折中的方法。
    3. 导致在decode中会被插入prefill从而导致decode卡顿以及decode阶段MFU低下这两个问题。

以下参数决定了请求如何被分组到批次中：

1. 最大批次大小（max batch size）
2. 最大 token 数量（max number of tokens）
3. KV 缓存（KV Cache）的大小。如果没有足够的剩余 KV 缓存存储请求的上下文，该请求将无法被调度。

管理 KV 缓存大小并非确定性的——它会随着每个生成的 token 增长，最终可能扩展到最大输出 token 长度。因此，**管理剩余 KV 缓存涉及一定程度的估算**。与其他估算挑战类似，我们可以采用悲观或乐观的方式分配 KV 缓存。这两种策略分别称为预分配（preallocation）和按需分配（on-demand allocation）。
1. 在预分配中，一旦请求被调度，其 KV 缓存的内存空间会基于输入 token 数量和最大生成 token 数量之和进行保留。这种方法确保了在解码阶段不会出现内存不足的情况，因为所需的最大 KV 缓存内存已经提前分配
2. 按需分配随着 token 的生成动态分配 KV 缓存内存，而不是预先为最大值保留内存。但它引入了 KV 缓存耗尽（preemption）的风险。在这种情况下，批次中的某些请求必须被中断，其 KV 缓存需要被清除以释放内存，从而避免死锁。清除可以通过两种方式实现：将 KV 缓存交换到主存储器（host memory）或完全丢弃缓存。与交换相比，丢弃更常被优先选择，因为主存储器的读写操作会引入显著的开销。而丢弃仅需要一次预填充迭代，将先前生成的 token 与原始输入 token 连接即可，因此在大多数情况下是一种更高效的选项。

可调优能力LLM 推理服务往往用于生产环境，而生产环境面临的情况是复杂多样的。
1. 对于做阅读理解的应用来说，Prompt 可能会非常长，但生成的内容可能会非常短，开发人员可能会更追求吞吐；
2. 对于聊天应用来说，Prompt 可能较短，生成的内容也不会太长，开发人员可能会更追求延迟；
3. 对于创作类应用来说，Prompt 可能很短，生成的内容会更长，开发人员可能会更追求首 Token 延迟。
对 Continuous Batching 实现来说，就要求它调度策略尽量清晰，并且参数可调。所以更灵活的实现，未来可能会更受欢迎。

[一些已成为LLM 推理引擎中事实标准的方法](https://zhuanlan.zhihu.com/p/685706549) 建议细读。

### 调度优化/动态批处理

[大模型推理服务调度优化技术-Continuous batching](https://mp.weixin.qq.com/s/Se4lzaTLNZF29BXLRjw0xw)
1. 单处理，也就是单个提示（Prompt）传过来直接送入到LLM进行推理。因为每次只能处理一条数据，对GPU资源的利用率较低。
2. 静态批处理（static batching），将请求凑成固定Batch再执行，不同的request组成batch后，要等最长的一个request执行完毕，才能整体退出，批处理的大小在推理完成之前保持不变。因此，GPU 未得到充分利用。
3. 动态批处理（Dynamic batching），动态批处理是指允许将一个或多个推理请求组合成单个批次（必须动态创建）以最大化吞吐量的功能。
    1. 静态批处理是基于固定的请求个数来触发的，**比如每4个请求一批进行处理**；动态批处理是在静态批处理之上，增加一个时间窗口的维度，比如也是4个请求，但同时还有一个时间窗口100ms的约束，那么在100ms以内，如果积累了4个请求，那么就会触发后置处理，或者是在100ms的窗口时间内，没有达到4个请求，那么也会触发后置处理，动态相比静态处理来说，对用户侧会更友好，但是从资源利用率上来说没有静态批处理好，使用哪种方式可以结合场景进行权衡选择。
4. 连续批处理（Continuous Batching），在Token粒度动态插入/移除请求。无论是动态批处理还是静态批处理，通常在相同形状的输入和输出请求的场景，提高GPU的利用率。但对于自回归大模型推理场景而言，都不太适用（同一批次中的数据输入和输出长度都不一样）。Continuing Batching（有的地方也叫做 Inflight batching 或者 Iteration batching）指请求在到达时一起批量处理，但它不是等待批次中所有序列都完成，而是当一个输入提示生成结束之后，**就会在其位置将新的输入Prompt插入进来**，从而比静态批处理具备更高的 GPU 利用率。由于**每次迭代的批处理大小是动态的**，因此，有些地方也叫动态Batching。PS：当一个request执行完毕之后，可以继续插入新的request，从request视角是立即开始立即结束。

提升模型服务吞吐最重要的手段是 Batching 策略，Batching主要包含以下三个步骤：

1. 模型服务调度层将多个不同的请求组成一个 batch 的模型输入；
2. 将batch 化的模型输入放入推理后端进行推理，得到一个 batch 结果；
3. 再将推理的 batch 结果拆分，并封装成不同的 Response 返回到对应的请求中。

一般来说，合并越多的请求作为单次推理的输入，服务吞吐越高。所以从请求 Batching 的角度去提升模型服务吞吐的本质是提升单次推理的最大合并请求数，即 batch size。对于模型服务来说，单次推理最大合并请求数主要受显存制约。合并越多的请求，batch size 越大，KVCache 的显存占用则越大。KVCache 的显存占用上限可简单的通过显卡的最大显存减去模型权重显存计算得到。

Batching就是将一段时间内到达的用户请求合并到一起，提交到GPU中执行，从而提高系统的吞吐量。然而，**与传统的 DNN Model 在推理时只要正向执行一遍不同，基于 Transformer 的 Generative Model 在推理时是迭代式的（Iterative），每个请求都需要迭代式执行多次，每次生成部分结果（一个 Token），且每个请求的迭代次数可能是不同的（例如迭代直到模型生成一个 End-Of-Sequence Token）**。因此将现有的 Batching 方式应用在 Generative Model 时，可能导致有的请求已经迭代结束了，但是还需要和同Batch中没有迭代结束的请求继续一起执行。这个问题的核心在于，传统的 Batching 技术是以 Request 为粒度的（Request-Level），将多个 Request 绑定在一起提交给执行引擎，多个 Request 同时开始同时结束。因此需要一个新的 Batching 的方式，这也是本项工作核心的 Insight：使用更细粒度的，Iteration-level Batching，在每个 Iteration 中将不同的 Request 合并到一起。对于新到达的请求，有机会在当前的迭代执行后进行处理，从而减少等待时间。**通过迭代级调度，调度器可以完全控制每次迭代处理的请求数量和哪些请求**。PS： batch的粒度不同。

![](/public/upload/machine/iteration_level_batching.jpg)

为了进行批次生成，我们改为一次向模型传递多个序列，在同一前向传递中为每个序列生成一个补全（completion），这需要在左侧或右侧使用填充词元对序列进行填充，使它们达到相同的长度。填充词元（可以是任何词元，我这里使用 `[end]`）在注意力掩码中被屏蔽，以确保它们不会影响生成。

但在上面的例子中，请注意 “Mark is quick. He moves quickly.” 在其他序列之前完成，但由于整个批次尚未完成，我们被迫继续为其生成词元（“Random”）。这并不影响准确度，我们只需简单地将生成的序列截断到 `[end]` 词元即可，但这样很浪费资源，因为GPU资源正在用于生成我们即将丢弃的词元。连续批处理通过将新序列插入批次来解决这一问题，插入位置是 `[end]` 词元之后。在 `[end]` 词元之后生成随机词元的代替方案是，在批次的相应行中插入新序列，并使用注意力掩码机制来防止该序列受到上一序列中词元的影响。（实际上，先前的序列充当了额外的填充内容。）

[vLLM（二）架构概览](https://zhuanlan.zhihu.com/p/681716326)vllm Scheduler 使用 iterative-level 策略对请求进行调度（选择要被处理的请求），被调度的请求在生成一个 token 后会被重新调度。得益于 itertive-level 策略，vLLM 能够在每一轮新的迭代时选择不固定数量的请求进行处理（即 batch size 每次都不一定相同），因此它能够尽可能多地处理请求。

请求的处理通常分为两个阶段，第一个阶段对 prompt 进行处理（也被称为填充阶段，后文使用填充阶段表示这一个阶段），生成 prompt KV cache 的同时生成第一个 token，第二个阶段是生成阶段，不断预测下一个 token。目前对 iterative-level 的实现有两种方式，一种是区分填充阶段和生成阶段，另一种是不区分这两个阶段。vLLM 采用的 iterative-level 策略是区分两个阶段的（https://github.com/vllm-project/vllm/pull/658），即同一批被调度的请求要么都处于填充阶段，要么都处于生成阶段，Scheduler 中有 3 个队列，waiting（接受到的新请求会先放入 waiting 队列）、running（被调度的请求）和 swapped 队列（swapped 队列用于存放被抢占的请求，即当请求处于生成阶段时，但由于空间的不足，需暂时将 running 队列中优先级低的请求移到 swapped 队列）。在调度时，Scheduler 会按照先到先处理（first come first served）的原则从 waiting 队列中选择请求放入 running 队列，此外，Scheduler 的另一个核心组件是 BlockSpaceManager，它主要负责块表的维护。

![](/public/upload/machine/vllm_overview.jpg)

假设 vLLM 接收到 3 个请求（记为 s0, s1, s2）并放入 waiting 队列中，它们的 prompt 分别为 "Hello, my name is"、"The future of AI is" 和 "The life is"。接下来开始 vLLM 的调度和处理。
1. vLLM 的第一轮处理，假设 vLLM 在这一轮只能调度两个请求进行处理，那么根据先到先处理的原则，会从 waiting 队列中选择 s0 ("Hello, my name is") 和 s1 ("The future of AI is") 放入到 running 队列。对于 s0，Worker 生成的 token 为 Dustin，对于 s1，Worker 生成的 token 为 bright。同时，Worker 会将计算过程产生的 KV 值存储在 KV cache 中
    ![](/public/upload/machine/vllm_scheduler_1.jpg)
2. vLLM 的第二轮处理，由于 waiting 队列中还有一个请求 s2（The life is)，因此，vLLM 在第二轮只会处理这一个请求，因为前面提到，vLLM 只会处理要么都是填充阶段的请求，要么都是生成阶段的请求。
    ![](/public/upload/machine/vllm_scheduler_2.jpg)
3. vLLM 的第三轮处理，waiting 队列中没有要处理的新请求，所以会从 running 队列中选择此轮要处理的请求（这些请求均处于生成阶段）。但由于没有多余的空间，vLLM 只会选择 s0 和 s1 进行处理。经过多轮调度和推理，最终完成 3 个请求的处理，以上就是 vLLM 的工作流。

[让LLM推理加速的batching是什么技术（in-flight batching）](https://zhuanlan.zhihu.com/p/679723881)

[借着triton inference server聊一下各种batching方法](https://mp.weixin.qq.com/s/R2PPbHcOgJVAM3nPVOKdFw) 未读

### 模型文件加载

如果使用c++ 来写推理或训练引擎的话，就没有python调用c这个复杂的事儿了。对于一个推理框架，大概可以理解为，
1. 专用的推理框架入口是onnx/pnnx等模型文件，只需要graph、节点/等概念，不需要pytorch 中类似layer概念（那是为了编程上抽象复用的）。 
2. 先基于onnx/pnnx等模型文件，自己提一套抽象/对象比如RuntimeGraph+RuntimeGraph+Operator等（为此有一个全局的算子注册机制），将模型权重、参数加载进来 构成计算图对象/内存表示，Operator 分为有参数算子和无参数算子，weight也就是tensor会赋值给有参数 Operator.weight。
3. RuntimeGraph.run 按拓扑排序执行，执行到某个节点RuntimeNode时，RuntimeNode为算子准备入参、拿到出参（也就是tensor），可能跨节点通信，Operator为 cuda 函数准备入参（cuda 函数的入参、出参也就是tensor，必须事先准备好 指针形式传给cuda函数）。概念上从大到小是Graph ==> node ==> Operator ==> cuda 函数。
4. tensor/显存的申请、释放都是上层组件负责（cuda 函数内不管，cuda 函数是无状态的），会有一个DeviceAllocator（分别对应cpu和gpu）组件负责内存和显存的分配和释放、内存和显存之间的copy等接口（比如tensor.to_cuda。再复杂一点先提前申请一个大的，内部再复用一下），对DeviceAllocator封装后提供tensor对象（tensor持有DeviceAllocator 引用，初始化时调用DeviceAllocator.allocate，析构时调用DeviceAllocator.release）。只是给算子函数传入input/weight/output 指针，算子也分为cpu和gpu实现。

### 资源管理的抽象

对于资源的申请和释放，例如内存的分配和释放，我们也可以进行封装，使得这些操作对使用者透明。这不仅提高了代码的可复用性，也减少了内存泄漏的风险。

```c++
class MemoryManager {
public:
    MemoryManager(size_t size) {
        cudaMalloc(&devicePtr_, size);
    }
    ~MemoryManager() {
        cudaFree(devicePtr_);
    }
    void* getDevicePtr() const { return devicePtr_; }
private:
    void* devicePtr_;
};
```

我们希望我们的代码比较好的可读性，就意味着我们在设计的时候尽量通过接口来暴露或者隐蔽一些功能。比如说，我们可以使用worker作为接口进行推理。在main中，我们只需要做到`创建一个worker -> woker读取图片 -> worker做推理`就好了。同时，worker也只暴露这些接口。在worker内部，我们可以让worker根据main函数传入的参数，启动多种不同的task（分类、检测、分割）。

```c++
class Worker {
public:
    Worker(const std::string& taskType, const std::string& modelPath);
    void loadImage(const std::string& imagePath);
    void infer();
    void displayResult();
private:
    std::shared_ptr<InferenceEngine> engine_;
    cv::Mat image_;
};
```
在主程序中，我们只需要与 Worker 类交互：
```c++
int main() {
    Worker worker("classification", "model.engine");
    worker.loadImage("image.jpg");
    worker.infer();
    worker.displayResult();
    return 0;
}
```

为框架设计插件机制，允许用户自定义前处理、后处理等步骤。插件可以在运行时加载，方便功能的扩展。

```c++
class Plugin {
public:
    virtual void execute() = 0;
    virtual ~Plugin() {}
};

class CustomPreprocessor : public Plugin {
    void execute() override {
        // 自定义前处理逻辑
    }
};
```


### 在线推理框架

[揭秘大语言模型实践：分布式推理的工程化落地才是关键！](https://mp.weixin.qq.com/s/QeDmD-XlvkkJ7LMNJEynHg)与以往的模型不同，单张 GPU 卡的显存可能不足以支撑大语言模型。因此，需要使用模型并行技术，将大语言模型进行切分后，在多张 GPU 卡上进行推理。我们使用 DeepSpeed Inference 来部署大语言模型分布式推理服务。DeepSpeed Inference 是 Microsoft 提供的分布式推理解决方案，能够很好的支持 transformer 类型的大语言模型。。DeepSpeed Inference 提供了模型并行能力，在多 GPU 上对大模型并行推理。通过张量并行技术同时利用多个 GPU，提高推理性能。DeepSpeed 还提供了优化过的推理定制内核来提高 GPU 资源利用率，降低推理延迟。

有了大模型分布式推理方案，然而想要在 Kubernetes 集群中高效部署大模型推理服务，还存在很多工程化挑战，比如大规模的 GPU 等异构资源如何高效地管理运维和自动调度？如何快速部署推理服务，服务上线后如何保证资源能够应对波动的访问量？以及没有适合的工具进行推理服务时延、吞吐、GPU 利用率、显存占用等关键指标监控，没有合理的模型切分方案，模型版本管理等。

[大模型的好伙伴，浅析推理加速引擎FasterTransformer](https://mp.weixin.qq.com/s/Gkf_zIYWs4u7AJrJLDVq_Q) 未细读
FasterTransformer 是真对于 Transofrmer 类型模型（也包括 encoder-only、decoder-only）的推理加速方案，其提供了 Kernel Fuse、Memory reuse、kv cache、量化等多种优化方案，同时也提供了 Tensor Parallel 和 Pipeline Parallel 两种分布式推理方案。

[​揭秘NVIDIA大模型推理框架：TensorRT-LLM](https://mp.weixin.qq.com/s/xv3gBjmejoxJEpvFoeUXOg)

[大模型推理优化实践：KV cache复用与投机采样](https://mp.weixin.qq.com/s/W9iVW7niyi_HvEWxOcnwuA)RTP-LLM 是阿里巴巴大模型预测团队开发的大模型推理加速引擎，该引擎与当前广泛使用的多种主流模型兼容，并通过采用高性能的 CUDA 算子来实现了如 PagedAttention 和 Continuous Batching 等多项优化措施。RTP-LLM 还支持包括多模态、LoRA、P-Tuning、以及 WeightOnly 动态量化等先进功能。

[高性能 LLM 推理框架的设计与实现](https://mp.weixin.qq.com/s/4o86rMuburB8jcbU0aYC7g)PPL.LLM，商汤，开源。 

![](/public/upload/machine/single_machine_infer.png)

## 分布式推理

[LightLLM中DeepSeek V3/R1 Two MicroBatch Overlap 实现解析](https://mp.weixin.qq.com/s/V7LmiDRcBiSC0Dfl5jGm3w) 未细读（在代码上体现如何overlap）。在DeepSeek-V3/R1推理系统中，多机多卡的专家并行会引入比较大的通信开销，所以DeepSeek使用了双 batch 重叠来掩盖通信开销，提高整体吞吐。

![](/public/upload/machine/micro_batch_overlap.png)

### 并行与分布式部署

1. 在提升模型显存使用效率方面，Flash Attention 和 Paged Attention 是两种常用的方法。在输入序列中，模型会根据每个词的重要性来分配显存。对于重要性较高的词，模型会分配更多的显存空间来存储其信息；而对于重要性较低的词，模型则会分配较少的显存空间。
2. 量化。从感知上来讲模型的参数量越大，其中的信息冗余程度也就越高，低精度量化在传统的小模型推理中已经是一个常见的优化手段了，对于更大参数量的语言模型更是如此。量化过程主要涉及两个方面：参数环节的小型化和降低数据类型。通过这一步骤，我们能够使得模型加载的参数更小，从原本的 FP32 降低到 FP16，从而提高推理性能。在量化过程中，我们还会采用混合精度量化技术。这种技术能够在保证模型准确性的前提下，将异常值保留精度，并在混合精度分块矩阵最后再加回去。
    1. BF16拥有与FP32相同的8位指数部分，因而能够表示与FP32几乎一样广泛的数值范围，这对于避免上溢和下溢非常重要。尽管BF16在尾数精度上不如HF16，但在深度学习应用中，这种较宽的数值范围通常比尾数的额外几位精度更为重要。这是因为深度学习模型通常对权重的尾数精度不是非常敏感，而更依赖于能够处理范围广泛的梯度和权重值。
    2. 量化对于文本生成特别有效，因为我们关心的是选择 最可能的下一个词元的分布 ，而不真正关心下一个词元的确切 logit 值。所以，只要下一个词元 logit 大小顺序保持相同， argmax 或 topk 操作的结果就会相同。
    3. 常用量化方法：GPTQ、AWQ和GGUF
3. 模型稀疏化。模型稀疏化是一种重要的优化方法。它的主要目的是减少模型参数的数量，从而降低模型的复杂度，提高模型的泛化能力和计算效率。模型稀疏化的主要方法有剪枝、量化、低秩近似等。剪枝是一种直接删除模型中部分参数的方法，它可以有效地减少模型的规模，但需要注意不能过度剪枝，以免影响模型的性能。低秩近似则是通过将模型转换为低秩矩阵，来减少模型的参数数量。
4. 并行。**在Attention层中采用TP、SP，也可以开始CP；FFN层如果是dense结构用TP+SP，如果是sparse结构(MoE)常用EP；DP是所有层都适用**。ZeRO策略(参数分片/shard)、PP层间的流水线并行相对而言当前的使用频率较低，在一些特定场景中可考虑开启。[分布式推理并行策略](https://mp.weixin.qq.com/s/KlDLR1SqSJFdSGJmpyd0zQ)
    ![](/public/upload/machine/infer_parallelism.png)
    4. 推理引擎都是做成多卡TP而不是PP，主要是因为从服务器视角看PP的吞吐上限更高，但是从单个请求视角看TP的延迟会更低，在线服务往往不需要那么高的吞吐，延迟更加重要。后来vLLM还增加了流水线并行（Pipeline Parallelism）的支持，从vLLM版本 0.5.1 开始支持跨多节点的流水线并行，对于那些跨多个节点的超大模型和低带宽连接，流水线并行是一种更优的选择。

随着 DeepSeek V3/R1 与 Kimi K2  等 MoE 架构的模型的横空出世，更大参数量与上下文的模型以及更复杂的使用场景使得单机的 GPU 部署方式无法再适用，因为节点内卡间的通信以及跨节点的通信（根据网络拓扑的不同）在不同的并行方式下都会引入难以忽略的延迟，对多个关键指标都会造成显著的降级，从而影响推理服务的质量。

![](/public/upload/machine/distribute_infer.png)

## 以 KV Cache 为核心的分布式架构

传统的推理服务架构采用同构部署模式，即所有的GPU节点既负责Prefill也负责Decode。随着长上下文场景的增加，这种模式暴露出了严重的性能缺陷。在同构集群中，当一个长Context的Prefill请求（例如处理一本小说的输入）被调度到某张GPU上时，该GPU会被长时间占用（可能长达数秒）。此时，该GPU上原本正在进行的Decode任务会被强制挂起，导致正在等待生成下一个Token的用户感受到明显的卡顿，即Token间延迟（Inter-Token Latency, ITL）激增。这种现象被称为“**队头阻塞**”（Head-of-Line Blocking）或“干扰效应”（Interference Effect）。PD分离架构将GPU资源划分为两个独立的池，Prefill 侧计算 Prompt 并产出 KV；Decode 侧接收 KV 后持续生成 token，从架构上隔离长 Prefill 对短 Decode 的干扰。

[LLM PD 分离背后的架构问题](https://zhuanlan.zhihu.com/p/27836625742) 未细读。

Mooncake 采用了以 KVCache 为中心的分离式推理架构，主要由三个核心部分组成：

1. Prefill 池：这个部分负责集中管理所有的预填充阶段的计算任务。
2. Decoding 池：这个部分集中处理所有解码阶段的任务。
3. KVCache 池：这个部分负责存储所有中间过程中应用到的 KVCache，并决定何时使用这些缓存，何时释放它们。

prefill-decode 分离（PD 分离）架构主要是考虑到了 LLM 的 prefill 和 decode 的两个阶段的特性不同，prefill 阶段是 compute bound（对于上下文的embeding和自注意力计算，模型会并行处理输入提示中的所有 Token，一次性计算出整个输入序列的 Attention 状态），decode 阶段是 memory bound（需要结合上下文以及当前token之前生成的token对应的KV值进行计算，过程是串行的），prefill 阶段的能力我们用 TTFT 首 token 时延来衡量，decode 的能力我们用 TPOT 生成每个 token 的时间来衡量。
1. 但是在同一张卡上做 prefill 和 decode 会出现问题，在机器的算力等条件固定的情况下，你增加 bsz，prefill 阶段机器到算力瓶颈了，反而影响 TTFT，你减小 bsz，decode 阶段又是访存瓶颈的，decode 阶段可以比 prefill 阶段承载更大的 bsz。那么问题来了，到底要不要增大 bsz？
2. 有了 PD 分离之后，我们可以把 prefill 阶段放在 H800 这样的算力高的机器，decode 阶段放在 H20 这样算力低的机器但是访存能力不会差太多的机器（毕竟显卡更新换代过程中算力增长是遥遥领先访存能力增长的），这样我们的如何 bsz 如何均衡的问题似乎可以得到解决，不同机器只负责一个阶段，**bsz 也只需要根据你这个阶段的特性来设置就好了**。decode 阶段可以比 prefill 阶段承载更大的 bsz。
    1. TTFT 和 TPOT 可以各自独立优化（两个阶段的画像差异很大）。P和D的并行策略可以不一样，比如，P实例处理请求数一般较少，DP设置小；而decode需将并发打上去，DP数量设置大； P实例的MoE层可使用TP并行，D实例则一般使用EP并行。不要让”擅长大矩阵计算的 Prefill”和”擅长高带宽 KV Cache 访问的 Decode”去争同一批 GPU。
    2. 两种资源可以分别扩容。真实业务的流量画像往往是不均衡的。比如一个 RAG 场景：prompt 动辄几万 token，但输出只有两三百字。反过来，如果你的业务是”短 prompt + 长输出”（比如 agent、长文写作），那就反着扩 Decode。
3. 但是 PD 分离有个很重要的问题，
    1. KV Cache 必须搬家。从 Prefill GPU 搬到 Decode GPU，增加了通信和网络传输的成本(KV cache从prefill节点到decode节点)，如果是卡间分离那么会增加通信的成本，如果是不同机器上进行分离那么就会增加网络传输 KV Cache 的成本 `KV Cache size  ×  request rate  =  所需 KV 传输带宽`。如果 KV Cache 大、请求率又高，网络会直接成为整个推理系统的瓶颈，而不是算力。PD 分离不是无条件提速，而是拿 TTFT 换 TPOT、拿网络开销换 GPU 利用率和可运维性。**如果你的业务对首字延迟极度敏感（比如短 prompt 的对话补全），PD 分离甚至可能是负收益**。
    2. KV Cache 到底该放在哪？KV Cache 不是必须一直待在 GPU HBM 里，于是自然演化出分层存储
    3. 调度器从配角变成主角，而且是拓扑感知（topology-aware）的。不分离的时候，调度逻辑很朴素：Request → 一个 GPU worker。分离之后：
        ```
        Request
        ↓
        Scheduler
        ↓
        选择 Prefill GPU
        ↓
        生成 KV
        ↓
        选择 Decode GPU
        ↓
        传输 KV
        ↓
        Decode
        ```
DeepSeek V3采用的也是PD 分离部署方式（prefill-32张卡，decode-320张卡），由于集群分开增加的中间状态数据的传输，在NVLink这类传输技术的加持下，有开销但是可控。prefill和decode两个集群的GPU数量需要按照实际场景调整，大部分在1:2到1:4之间，具体看实际场景中TTFT和TPOT的表现和要求。

Context Caching 

![](/public/upload/machine/context_caching.jpg)

[Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://zhuanlan.zhihu.com/p/706109023)Mooncake的核心是其以KVCache为中心的调度器，将预填充/prefill服务器与解码/decoding服务器分开，因为LLM服务的这两个阶段具有非常不同的计算特性，Prefill是计算密集，受限算力带宽用不满，Decode是访存密集性，受限带宽算力用不满。所以用同一种硬件部署两阶段往往顾此失彼，不是最有性价比。拆分Prefill/Decode之后，LLM推理系统就更像一个分布式内存系统+流处理系统，其中KVCache随着请求从预填充移动到解码服务器而转移，将KVCache和计算分离开，它将GPU集群的CPU、DRAM、SSD和RDMA资源分组组成Distributed KVCache Pool，KVCache也是分块以Paged方式管理，KVCache Blocks如何在Pool中调度，请求和复用KVCache乃精髓。这就是传统计算机系统研究者最擅长的领域，sys三板斧，batch， cache，调度都可以招呼上，比如
1. Prefill阶段可以利用request间存在共同前缀的机会，尽可能复用KVCache。PS：比如agent 循环调用 多次请求的prompt prefix是一样的。
2. Decode可以进一步拆成Attention和非Attention算子分离调度。

[Mooncake](https://github.com/kvcache-ai/Mooncake)


[AI 推理场景的痛点和解决方案](https://mp.weixin.qq.com/s/SeUJxNK10fhR6YsWSJRYwg) 未细读。

[大模型推理框架RTP-LLM P-D分离之道：从思考到实战](https://mp.weixin.qq.com/s/4FVw5paNSUCeQEUp9hoJ5Q) 未读

[浅谈基于 Kubernetes 的 LLM 分布式推理框架架构：概览](https://mp.weixin.qq.com/s/5Q2Rjg6YKs7V9kOL41eACQ)
1. KV Cache Offloading。KV Cache Offloading 指的是将 GPU 显存中的 KV Cache 卸载到 CPU 内存或是外部存储的过程，当 LLM 需要访问被卸载的 KV Cache 时，它会按需将这些 Block 重新加载回 GPU 显存中。AIBrix 提供了多层 KV Cache 的缓存框架，默认情况下会使用 DRAM 中的 L1 Cache，在需要共享 KV Cache 的场景下则可以使用 L2 Cache，即分布式的外部存储。Dynamo 与 llm-d 等框架同样也支持使用 LMCache 等框架将不常用的 KV Cache 卸载到 CPU 内存与外部存储中。
    ![](/public/upload/machine/kvcache_offloading.png)
2. KV Cache Sharing。随着 KV Cache Offloading 的引入，如何在不同的 LLM 推理实例共享 KV Cache 也是个被广泛研究的问题。
    1. Centralized：即通过一个中心化的 KV Cache 的池来管理不同实例的 KV Cache，其优点在于能够最大化地共享 KV Cache，可以更好地利用 Prefix Caching 的能力，但是在高并发的场景之下其可能会成为单点的性能瓶颈。
    2. Peer-to-Peer：即直接通过 P2P 通信机制在不同实例间传输 KV Cache，避免了中心化的存储，且具有更好的容错与动态扩缩容的能力的支持。
3. 从分布式推理到分离式推理。目前基于基础设施层的分离式推理（Disaggregated Inference）也是被广泛讨论的话题，即模块化地拆分推理的各个模块形成分离式的部署，彼此之间通过协议传输数据。
    1. 基于Attention的LLM模型通常由注意力模块和前馈网络模块组成，解码阶段中，注意力模块的参数量少但需要大量KV交互，是访存密集型的，而前馈网络模块参数量大，是计算密集型的，因此Step-3模型论文提出了AFD（Attention - FFN Disaggregation）架构，即将注意力模块和前馈网络模块部署在不同的设备上，在PD分离的基础上进一步优化资源的利用率和推理服务效率。
    2. MoE模型由若干专家模型组成，主流的大模型拥有几千甚至更多的专家数量，例如DeepSeek V3系列有约1.4万个专家模型，如果所有的专家模型全部部署在一张GPU卡，或一台服务器上是远远无法承载的。由于MoE模型的稀疏激活特性，在多机部署时GPU之间的通信（all-to-all）仅需按需传输，仅把对应的token传给激活的专家模块。仅用NCCL在这种通信模型效率很低，导致带宽资源大量浪费，原因是NCCL更擅长全量而密集的通信。由DeepSeek开发的DeepEP是专门为混合专家模型（MoE）+专家并行（EP）设计的通信库。
4. 由目前 AIBrix 与 llm-d 等框架的设计上来看，将负载均衡与 KV Cache 管理等功能由推理引擎的层面上升到集群编排的层面去解决是目前的主流趋势，编排侧能够更好地与现有的集群资源管理系统（如 Kubernetes）的生态系统集成，避免一些重复性的工作，推理引擎也可以关注在内部的推理过程优化，只需要暴露一些特定的抽象与接口供编排侧对接。

vLLM 与 SGLang鏖战正酣时，小强 LMCache 盯上了具体的 KV Cache 这一块，专注于缓存层的抽象与优化，与两强的关系都不赖。LMCache 自身以一种分布式的方式，有效地管理起计算侧的 GPU 缓存。向下，LMCache 充分地利用各种存储后端（Mooncake，Redis，InfiniStore，......），适时地将 KV Cache offloading，从而提供更具水平扩展性的 KV Cache Layer。LMCache 归纳总结的 KV Cache 两个主要场景
1. Context Caching，这里缓存的 KV Cache 来自公共的 System Prompt，以及专业知识库相关的 RAG Context 等等，通常这部分 KV Cache 比较大，甚至需要 offloading 至 CPU Memory，乃至 Local Disk；
2. P/D Disaggregation Caching，这个架构的一个强依赖就是 Prefill 引擎与 Decode 引擎之间的高效通信，分享 Prefix KV Caches。
旨在统一 KV Cache Layer 的 LMCache 祭出了何等大杀器？无他，惟如下三点：
1. 模块化的 KV Cache Connector，提供标准化的访问接口，将 KV Cache Layer 从快速演进的 LLM Inference 框架中脱离出来。
2. 有关 KV Cache 的高效管理，包括高效地 storing KV Cache，包括高效地 loading KV Cache。一大优化是 pipelining 技术，涉及到 GPU compute 与 data loading/storing 之间的 pipelining，以及 KV Cache 的 storing 与 loading 之间的 pipelining；另一大优化是 batching 技术；以 chunk 为单位（远大于具体操作的 page 页大小）store/load KV Caches，以充分地利用存储设备与 GPU 内存之间的网络带宽；在不同存储层之间移动 KV Cache 数据时候，应用 zero-copy 零拷贝技术，减少内存拷贝；
3. 面向 KV Cache 的管控平面，，用户可以针对 KV Caches 做诸如 pinning, lookup, cleanup, movement, and compression 等操作。譬如 2025 年早些时候，某金融公司即在生产场景下给 LMCache 提出需求，希望能够 pin 住 KV Cache 中经常需要访问的金融类文档信息；
LMCache是一个两层的架构，中心化的 Cache Controller，与散落在每个节点上的 LMCache Worker 共同组成。

## 控制面

vLLM通过PagedAttention等技术，极大地提升了单机或单节点的LLM推理吞吐量。它完美地解决了“数据平面”的效率问题。但当你尝试将vLLM部署到成百上千张GPU的生产环境，服务海量模型和用户时，新的挑战出现了：
1. 「路由（Routing）」：如何将请求智能地分发到最合适的模型副本？简单的轮询（Round-Robin）远远不够。
2. 「自动伸缩（Autoscaling）」：如何根据真实负载动态调整GPU资源？传统的QPS指标在LLM场景下几乎失效。
3. 「容错与管理」：如何处理硬件故障，如何高效地管理成百上千个LoRA适配器？
这些问题，vLLM本身并不直接解决。它们属于“「控制平面」”的范畴。

函数计算推出 GPU 闲置计费功能，在保障性能的前提下，可以帮助您大幅降低 GPU 的成本开销。以往部署大型语言模型（LLM）可能需要昂贵的 GPU 支持，尤其在需要大量计算资源时。但请求处理并不是每时每刻都处于活跃状态，势必存在流量的潮汐现象，后端的计算资源会出现空载导致成本的浪费。借助函数计算 GPU 闲置计费功能，用户的开销将会根据实际计算负载动态调整。

### 路由与负载均衡

[Higress LLM 服务负载均衡的新实践](https://mp.weixin.qq.com/s/TIv1BlU8vHeGA2HBGqjAaA)在面对 LLM 服务时，这些传统方法（常见的负载均衡算法有轮询、随机、最小请求数、一致性哈希等）往往暴露出以下几个关键缺陷：
1. 忽略任务复杂度差异：LLM 推理请求的复杂度差异极大。例如，一个长文本生成任务可能需要数十倍于短文本分类任务的计算资源。而传统负载均衡器无法感知这种差异，容易导致某些节点过载，而其他节点空闲，造成资源浪费和响应延迟。
2. 缺乏对 GPU 资源水位的感知：在 LLM 推理服务中，计算瓶颈主要集中在 GPU 上，传统负载均衡器往往无法感知到这一细粒度的资源消耗情况，导致某些 GPU 节点因显存不足而拒绝请求或响应缓慢，而其他节点却处于空闲状态。
3. 缺乏对 KV Cache 的复用能力：在并发请求处理中，如果多个请求具有相似的前缀，则它们的 KV Cache 可能存在重叠部分，可以通过共享或压缩的方式减少显存占用并提升生成速度。传统负载均衡策略并未考虑请求之间的语义相似性或 KV Cache 的可复用性，难以将具有潜在复用价值的请求分配到同一 GPU 实例上，从而错失优化机会。

LLM 推理的核心是 KV Cache，其存储了模型在 Prefill 阶段计算出的 Key 和 Value。由于不同请求的 Prompt 可能存在重叠的前缀，这些共享的前缀信息可以被多个请求复用，从而显著提升推理效率。因此，在负载均衡时需要考虑如何高效地利用这些缓存（KVCache aware），而**不是简单地将请求随机分配到不同的实例上**。
1. Prefix-Aware。如果一个请求的前缀（Prompt）已经在某个实例的KV缓存中，就将请求路由到该实例，实现缓存复用，极大加速Prefill。这与CDN的回源策略、数据库的查询缓存有异曲同工之妙。这种策略需要负载均衡器维护每个实例的缓存状态信息，并根据请求的前缀特征进行路由。
2. Profile-based SLO-aware。在异构GPU环境中，路由器会根据每个实例的性能画像（Profile）和服务等级目标（SLO），进行负载感知和队列管理，确保高优先级的请求被快速处理。
2. Fairness。LLM 实例的公平调度也是负载均衡的一个重要方面，尤其是在多租户环境中，公平性确保了所有用户都能获得相对一致的服务质量，而不会因为某些实例过载而导致其他用户的请求延迟。AIBrix 基于 Sheng et al. 实现了 Virtual Token Counter (VTC) 的 Fair Queuing 调度策略。VTC 为每个客户端维护一个虚拟 Token 计数器，通过跟踪每个客户端已接受的服务量来实现公平调度，优先服务计数器值最小的客户端。

### 自动扩缩容

目前 Kubernetes 的生态系统中被广泛使用的自动扩缩容工具主要有以下三个：

1. HPA（Horizontal Pod Autoscaler）：Kubernetes 的水平扩缩器
2. KPA（Knative Pod Autoscaler）：Knative 的水平扩缩器
3. KEDA（Kubernetes Event-driven Autoscaling）：事件驱动的服务扩缩器，支持服务的从零到一部署

针对 LLM 推理服务的自动扩缩容，其关键在于如何决定触发自动扩缩容的指标。AIBrix 的演讲者在 vLLM Meetup Beijing 分享了与传统的微服务的自动扩缩容不同，LLM 推理请求的 QPS 可能与产生的延迟并不是正相关的，且 SM Active 等 GPU 指标也不一定能及时反映出指标的变化。因此，如何基于 LLM 推理的特性来进行可靠的自动扩缩容仍然是需要探索的问题。长期来看，笔者认为 LLM 推理服务的自动扩缩容也有可能会由 Reactive 逐渐发展到 Proactive 甚至是 Predictive 的形态（如时间序列分析与基于强化学习的预测）。
