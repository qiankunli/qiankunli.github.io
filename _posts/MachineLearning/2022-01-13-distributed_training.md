---

layout: post
title: 分布式训练的一些问题
category: 架构
tags: MachineLearning
keywords:  distributed training

---

## 简介

* TOC
{:toc}

分布式训练的本质是分布式计算。

[关于深度学习框架的一些自己见解](https://zhuanlan.zhihu.com/p/375634204)我觉得做深度学习框架其实有两个派别的人，一派是从分布式系统的人来做的，另外一派是从做算法的人来做的。不同的人的背景不同，所以做这个事情的角度也会不同，从而产生不同门派。tensorflow属于系统派，而pytorch属于算法派。
1. TensorFlow从设计之初就在考虑超大的模型分布式训练的场景，设想这个框架需要能够从规模上很好支持分布式，能够很好的扩展到任意大的深度模型的框架。很自然会把模型的开发过程分成构图和执行两个阶段。构图的时候只是生成一个逻辑执行计划，然后通过显式方式的提交（或者execute），系统再根据用户指定的或者系统智能的决定的placement进行分图，并在这些分图中添加合适的Send-Recv的Op从而构成一个分布式的执行计划。但是这样的设计理念也会带来一些困恼，我们在模型训练时候有时候有些类似控制图的部分，在这种设计理念下，我们必须要把这些控制流图的代码也op化，然后把这些op也整体串联在Tensor的Flow执行图中。但是这种方式会使得一些习惯单机开发的研究人员觉得比较晦涩。
2. 框架的另外一派是算法派，特别是感知类模型（图像，语音，语言类）训练，因为这类训练一般都是同步训练，且是数据并行，这样我们就可以利用MPI的AllReduce的通讯源语来进行梯度的汇集计算。因为面向是数据并行的场景，这样话在神经网络部分其实都是**单机程序**，从而可以利用任何python的语法糖去构建任何的动态的训练控制逻辑（大家也把这种称作动态图）。对于算法研究人员来讲，这种方式写代码比较随性也方便debug，所以在研究界pytorch得到大量的关注和使用。

[利用多 GPU 加速深度学习模型训练](https://mp.weixin.qq.com/s/wiqOHIVfL2gKnRUhY62EBA) 从GPU 说到 horovod，很不错的文章

## 基本理念

分布式TensorFlow入门教程 https://zhuanlan.zhihu.com/p/35083779

[炼丹师的工程修养之四： TensorFlow的分布式训练和K8S](https://zhuanlan.zhihu.com/p/56699786)无论是TensorFlow还是其他的几种机器学习框架，分布式训练的基本原理是相同的。大致可以从以下五个不同的角度来分类。

1. 并行模式， 对于机器学习的训练任务，原来的“大”问题主要表现在两个方面。
  1. 一是模型太大，我们需要把模型“拆”成多个小模型分布到不同的Worker机器上；
  2. 二是数据太大，我们需要把数据“拆”成多个更小的数据分布到不同Worker上。
2. 架构模式，通过模型并行或数据并行解决了“大问题”的可行性，接下来考虑“正确性”。以数据并行为例，当集群中的每台机器只看到1/N的数据的时候，我们需要一种机制在多个机器之间同步信息（梯度），来保证分布式训练的效果与非分布式是一致的（N * 1/N == N）。相对成熟的做法主要有基于参数服务器（ParameterServer）和基于规约（Reduce）两种模式。Tensorflow 既有 PS 模式又有对等模式，PyTorch 以支持对等模式为主，而 MXNET 以支持 KVStore 和 PS-Lite 的 PS 模式为主。 [ 快手八卦 --- 机器学习分布式训练新思路(1)](https://mp.weixin.qq.com/s/CYJzTP-wU3pmyR1lPCSsvg) 
3. 同步范式（参数更新方式）， 在梯度同步时还要考虑“木桶”效应，即集群中的某些Worker比其他的更慢的时候，导致计算快的Worker需要等待慢的Worker，整个集群的速度上限受限于最慢机器的速度。因此梯度的更新一般有**同步(Sync)、异步(Async)和混合**三种范式。
  1. 同步模式中，在每一次迭代过程中，所有工作节点都需要进行通信，并且下一步迭代必须等待当前迭代的通信完成才能开始。**确保所有的设备都是采用相同的模型参数来训练**，需要各个设备的计算能力要均衡，而且要求集群的通信也要均衡。
  2. 反之，异步式分布算法 则不需要等待时间：当某个节点完成计算后就可直接传递本地梯度，进行模型更新。
4. 物理架构，这里主要指基于GPU的部署架构，基本上分为两种：单机多卡和多机多卡
5. 通信技术，要讨论分布式条件下多进程、多Worker间如何通信，分为以分为 Point-to-point communication 和 Collective communication 两类，Collective communication常见的技术有MPI，NCCL，GRPC，RDMA等

![](/public/upload/machine/distribute_tensorflow.png)

## 并行模式

1. 模型并行，从不同视角看
  1. 网络拆分为层：深度学习模型一般包含很多层，如果要采用模型并行策略，一般需要将不同的层运行在不同的设备上，但是实际上层与层之间的运行是存在约束的：前向运算时，后面的层需要等待前面层的输出作为输入，而在反向传播时，前面的层又要受限于后面层的计算结果。所以除非模型本身很大，一般不会采用模型并行，因为模型层与层之间存在串行逻辑。但是如果模型本身存在一些可以并行的单元，那么也是可以利用模型并行来提升训练速度，比如GoogLeNet的Inception模块。
  2. 计算图拆分为子图： 就是把计算图分成多个最小依赖子图，然后放置到不同的机器上，同时在上游子图和下游子图之间自动插入数据发送和数据接收节点，并做好网络拓扑结构的配置，各个机器上的子图通过进程间通信实现互联。
2. 数据并行（主要方案）：因为训练费时的一个重要原因是训练数据量很大。数据并行就是在很多设备上放置相同的模型，并且各个设备采用不同的训练样本对模型训练。训练深度学习模型常采用的是batch SGD方法，采用数据并行，可以每个设备都训练不同的batch，然后收集这些梯度用于模型参数更新。所有worker共享ps 上的模型参数，并按照相同拓扑结构的数据流图进行计算。
3. 流水并行，本质是解决模型并行（模型较后部分的计算必须等前面计算完成）后 效率低的问题 [所有人都能用的超大规模模型训练工具](https://mp.weixin.qq.com/s/3VU_9lednIkuD4dj_4NTZA)

由于模型分割开的各个部分之间有相互依赖关系，因此计算效率不高，所以在模型大小不算太大的情况下（只要一个GPU卡放的下）一般不使用模型并行。


## 通信技术

训练框架面临的是 单机CPU与GPU 之间、单机多GPU之间、多机CPU 之间、多机GPU 之间的通信问题，有各种优化方案，但要对上层提供统一通信接口，并进一步结合机器学习的特点提供 collective Communication 接口。

### 协议层

[海思专家如何看待RDMA技术？](https://mp.weixin.qq.com/s/UqSydz8hJCFcX5CF30gXSw) 比较好的一篇文章

DMA全称为Direct Memory Access，即直接内存访问。是一种外设**绕开CPU**独立直接访问内存的机制。
![](/public/upload/machine/dma.png)

CPU的最主要工作是计算，而不是进行数据复制。可以看到总线上又挂了一个DMA控制器，它是专门用来读写内存的设备。有了它以后，当我们的网卡想要从内存中拷贝数据时，除了一些必要的控制命令外，整个数据复制过程都是由DMA控制器完成的。过程跟CPU复制是一样的，只不过这次是把内存中的数据通过总线复制到DMA控制器内部的寄存器中，再复制到I/O设备的存储空间中。CPU除了关注一下这个过程的开始和结束以外，其他时间可以去做其他事情。**DMA控制器一般是和I/O设备在一起的**，也就是说一块网卡中既有负责数据收发的模块，也有DMA模块。

![](/public/upload/machine/rdma.png)

rdma 即remote dma。同样是把本端内存中的一段数据，复制到对端内存中，在使用了RDMA技术时，两端的CPU几乎不用参与数据传输过程（只参与控制面）。本端的网卡直接从内存的用户空间DMA拷贝数据到内部存储空间，然后硬件进行各层报文的组装后，通过物理链路发送到对端网卡。对端的RDMA网卡收到数据后，剥离各层报文头和校验码，通过DMA将数据直接拷贝到用户空间内存中。PS：RDMA 网卡一般很贵。 

RDMA本身指的是一种技术，具体协议层面，包含Infiniband（IB），RDMA over Converged Ethernet（RoCE）和internet Wide Area RDMA Protocol（iWARP）。三种协议都符合RDMA标准，使用相同的上层接口，在不同层次上有一些差别。上述几种协议都需要专门的硬件（网卡）支持。
1. Infiniband规定了一整套完整的链路层到传输层（非传统OSI七层模型的传输层，而是位于其之上）规范，但是其无法兼容现有以太网，除了需要支持IB的网卡之外，企业如果想部署的话还要重新购买配套的交换设备。
2. RoCE从英文全称就可以看出它是基于以太网链路层的协议，v1版本网络层仍然使用了IB规范，而v2使用了UDP+IP作为网络层，使得数据包也可以被路由。
3. iWARP协议是IETF基于TCP提出的

||tcp/ip|rdma|
|---|---|---|
|硬件|以太网卡|RDMA 网卡|
|驱动||rdma-core|
|接口|socket|libibverbs|

[系列解读SMC-R：透明无感提升云上TCP应用网络性能](https://mp.weixin.qq.com/s/Zz0qTbG9ZbRT53LHPJ_koQ)Shared Memory Communication over RDMA (SMC-R) 是一种基于 RDMA 技术、兼容 socket 接口的内核网络协议，由 IBM 提出并在 2017 年贡献至 Linux 内核。SMC-R 能够帮助 TCP 网络应用程序透明使用 RDMA，获得高带宽、低时延的网络通信服务。

### GPU 卡间通信

[深度学习分布式训练框架 horovod (3) --- Horovodrun背后做了什么](https://mp.weixin.qq.com/s/SkByud8mz4rjulJNec6jig)
Collective communication包含多个sender和多个receiver（相对于P2P 模式只有一个sender和一个receiver），一般的通信原语包括 broadcast，All-to-one (gather),all-gather，One-to-all (scatter)，reduce，all-reduce，reduce-scatter，all-to-all等。集合通信库的主要特征是：大体上会遵照 MPI 提供的接口规定，实现了包括点对点通信（SEND,RECV等），集合通信（ REDUCE，BROADCAST，ALLREDUCE等）等相关接口，然后根据自己硬件或者是系统的需要，在底层实现上进行了相应的改动，保证接口的稳定和性能。

![](/public/upload/machine/gxpu_communication.png)

[谈分布式机器学习系统中的网络相关问题](https://zhuanlan.zhihu.com/p/61731822)

#### NCCL 

The NVIDIA Collective Communication Library (NCCL) implements multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs and Networking. NCCL provides routines such as all-gather, all-reduce, broadcast, reduce, reduce-scatter as well as point-to-point send and receive that are optimized to achieve high bandwidth and low latency over PCIe and NVLink high-speed interconnects within a node and over NVIDIA Mellanox Network across nodes. [Point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)One-to-all (scatter) ,All-to-one (gather) , All-to-all 都可以基于 ncclSend 和 ncclRecv 来实现。

```c
// nccl/src/nccl.h.in
ncclResult_t  ncclGroupStart();
ncclResult_t  ncclGroupEnd();
// peer to peer
ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,ncclComm_t comm, cudaStream_t stream);
// Collective Communication 
ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff,size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,cudaStream_t stream);
...
// 初始化
ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
struct ncclComm {
  struct ncclChannel channels[MAXCHANNELS];
  ... 
  // Bitmasks for ncclTransportP2pSetup
  int connect;
  uint32_t* connectSend;
  uint32_t* connectRecv;

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index
  int64_t busId;   // my PCI bus ID in int format

  int node;
  int nNodes;
  int localRanks;

  // Intra-process sync
  int intraRank;
  int intraRanks;
  int* intraBarrier;
  int intraPhase;
  ....
};
```


NCCL 最初只支持单机多 GPU 通信，从 NCCL2 开始支持多机多 GPU 通信。

#### Gloo

Gloo是facebook开源的用于机器学习任务中的集合通信库. It comes with a number of collective algorithms useful for machine learning applications. These include a barrier, broadcast, and allreduce. 

Gloo 为CPU和GPU提供了集合通信程序的优化实现。但如果是在使用NVIDIA-硬件的情况下，主流的选择是NVIDIA自家的NCCL。

#### MPI 与 NCCL/GLOO

[利用多 GPU 加速深度学习模型训练](https://mp.weixin.qq.com/s/wiqOHIVfL2gKnRUhY62EBA)多机软件设计一般采用 MPI（Message Passing Interface）实现数据交互。MPI 是一种消息传递库接口描述标准，规定了点对点消息传递、协作通信、组和通讯员概念、进程拓扑、环境管理等各项内容，支持 C 和 Fortran 语言。**NCCL 出现得更晚一些，参考并兼容了 MPI 已有 API**。**NCCL 更多考虑了 GPU 的特性**，例如任意两块 GPU 之间的通信开销是有区别的，跨 QPI 情况与同一 PCIe Switch 情况，以及有 NVLink/ 无 NVLink 情况就有明显差异，但 MPI 认为两种情况下 GPU 与 GPU 都是等同的，甚至 **MPI 认为跨机器的 GPU 也是等同的**，这对于多 GPU 通信效率会有影响。MPI 可以和 NCCL 结合，实现**层次化的**并行通信机制，即同一台机器上的不同 GPU 之间采用 NCCL 通信，而不同机器上的 GPU 之间采用 MPI 辅助通信。[NCCL and MPI](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html)


## 优化手段

![](/public/upload/machine/distributed_trainning_optimize.png)
关于深度的分布式训练，主要工作主从技术栈上呈现从浅层到深层的一个过程。
1. 前三类的优化基本上是处于框架层，需要平台为用户提供基础的框架支持。比如说在计算图的并行化策略方面，我们通过GSPMD和GPipe提供了包括数据并行、模型并行和流水线并行的通用化的并行化策略的抽象层。此外，我们通过DeepSpeed来做支持，用ZeRO (Zero Redundancy Optimizer)来优化Optimizer的显存使用，以及我们可以用低精度的压缩来加速参数的同步
2. 集合通信层的一些优化，这类优化对于用户跟上层的框架完全透明，不需要改动上层的代码就能够真正落地。拿网络的协议站做一个类比的话，NCCL基本上跟IP协议一样，是整个协议栈的narrow waist的位置。[Fast Socket：NCCL的高性能网络栈](https://mp.weixin.qq.com/s/P50A3bGJfoekGcIxImv16A) 提到了对NCCL 本身的优化，比较底层。

## 数据并行ps/allreduce

[深度学习分布式训练框架——基础知识](https://mp.weixin.qq.com/s/djGvx3fNJfKCXmjwTfJ-CA)
1. 中心化分布式，存在一个中心节点，它的作用是汇总并分发其他计算节点的计算结果，更进一步，中心节点可以采用同步更新策略（Synchronous updating），也可以采用异步更新策略（Asynchronous updating）。一般情况下，参数服务器数目远少于工作机器，导致参数服务器端极易成为网络瓶颈。
2. 去中心化分布式


embedding 场景下架构模式选择： 参数服务器适合的是高纬稀疏模型训练，它利用的是维度稀疏的特点，每次 pull or push 只更新有效的值。但是深度学习模型是典型的dense场景，embedding做的就是把稀疏变成稠密。所以这种 pull or push 的不太适合。而网络通信上更优化的 all-reduce 适合中等规模的深度学习。又比如由于推荐搜索领域模型的 Embedding 层规模庞大以及训练数据样本长度不固定等原因，导致容易出现显存不足和卡间同步时间耗费等问题，所以 all-reduce 架构很少被用于搜索推荐领域。

