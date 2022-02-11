---

layout: post
title: 分布式训练
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
3. 同步范式， 在梯度同步时还要考虑“木桶”效应，即集群中的某些Worker比其他的更慢的时候，导致计算快的Worker需要等待慢的Worker，整个集群的速度上限受限于最慢机器的速度。因此梯度的更新一般有**同步(Sync)、异步(Async)和混合**三种范式。
  1. 同步模式中，在每一次迭代过程中，所有工作节点都需要进行通信，并且下一步迭代必须等待当前迭代的通信完成才能开始。
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


## 数据并行架构模式

[深度学习分布式训练框架——基础知识](https://mp.weixin.qq.com/s/djGvx3fNJfKCXmjwTfJ-CA)

参数服务器适合的是高纬稀疏模型训练，它利用的是维度稀疏的特点，每次 pull or push 只更新有效的值。但是深度学习模型是典型的dense场景，embedding做的就是把稀疏变成稠密。所以这种 pull or push 的不太适合。而网络通信上更优化的 all-reduce 适合中等规模的深度学习。又比如由于推荐搜索领域模型的 Embedding 层规模庞大以及训练数据样本长度不固定等原因，导致容易出现显存不足和卡间同步时间耗费等问题，所以 all-reduce 架构很少被用于搜索推荐领域。

### 单机模式

[Paddle计算图拆分与优化](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/performance/graph.html)
深度学习网络中有两个基本元素：Operator（比如加减/FC/Embedding查表） 和 Variable（分为稠密参数和稀疏参数）
单机的计算图执行流程：

1. 计算图拿到参数的值(Var)之后，会首先执行前向OP(FF OP)，OP可能会执行在不同的设备上，如CPU/GPU/Kunlun 或其他AI芯片，我们用XPU代指。
2. 前向OP计算完成后，得到loss，会继续计算反向OP(BP OP)，得到各个参数的梯度(Var_Grad)
3. 指定SGD/Adam等优化器，利用参数的梯度（Var_Grad）更新原始参数（Var）
4. 重复以上流程，迭代参数的更新，实现深度学习网络的训练

![](/public/upload/machine/paddle_host.png)

1. Worker(Trainer)在计算得到参数的梯度(Var_Grad)后，会通过RPC发送给PServer
2. PServer监听通信端口，将收到的不同参数分别通过不同的Oprimizer OP完成更新
3. Worker在下一个迭代时，请求PServer上最新的参数
4. 重复以上流程，迭代参数的更新，实现分布式参数服务器的训练


### ps-worker

[Paddle计算图拆分与优化](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/performance/graph.html)参数服务器模式下，所有参数的全局真值都被分片保存在各个Pserver上，**PServer执行Optimizer OP**（另一种选择是 ps 只是汇总梯度，weight还是worker 更新的），执行参数的更新。以PS-CPU异步模式的计算图介绍PServer的计算图

![](/public/upload/machine/paddle_ps.png)

进程内通信主要实现内存数据的跨设备复制（本地CPU和GPU内存之间、多个GPU内存之间）， 进程间通信主要实现数据流和控制流基于网络协议的传输。Tensorflow 的运行时框架会根据应用代码中的集群和设备配置，在数据流图构建时自动插入通信操作，并在数据流图执行过程中动态调度这些操作。

![](/public/upload/machine/ps_worker_communication.png)

1. 对于逻辑上具有数据传输语义，但物理上通信的源和目标位于同一设备、同一地址空间的进程内通信，只传输数据指针，不传输数据本身。
2. 对于cpu 和gpu 内存间的设备通信，本质是跨设备内存的数据复制，尽量使用底层设备接口实现（比如DMA机制），而不是用面向应用层的API
3. 涉及GPU的内存传输一般需要经过CPU 内存中转，除非使用支持RDMA 协议的高性能网络。
4. 尽量做到 计算与通信过程重叠，很多通信函数采用异步模式及多线程并发执行

### all-reduce

[Ring AllReduce简介](https://mp.weixin.qq.com/s/K8l7H2zCUr9sGzYehizFMA) 各种配图都比较详细了。

### api 代码示例
ps 架构的参数放在 ps 节点上，**worker和ps直接通过 rpc 进行push/pull**。《用python实现深度学习框架》给出的ps/worker 模型的grpc 通信api定义
```
service ParameterService{
  # ps 接收从各个worker 发送过来的请求，拿到各个节点的梯度，并根据节点名称累加到相应的缓存中
  rpc Push(ParameterPushReq) returns (ParameterPushResq){}
  rpc Push(ParameterPushReq) returns (ParameterPushResq){}
  rpc VariableWeightsInit(VariableWeightsReqResp) returns(VariableWeightsInit){}
}
# ps server 实现
class ParameterService(xxx):
  def Push(self,push_req...):
    node_with_gradients,acc_no = self._deserialize_push_req(push_req)
    if self.sync:
      self._push_sync(node_with_gradients,acc_no)
    else:
      self._push_async(node_with_gradients,acc_no)
  def _push_sync(self,node_with_gradients,acc_no):
    if self.cond.accquire():
      # 等待上一轮所有worker 等下拉完成
      while self.cur_pull_num != self.worker_num:
        self.cond.wait()
      # 记录上传次数
      self.cur_push_num +=1
      # 把梯度更新到缓存
      self._update_gradients_cache(node_with_gradients)
      # 累计梯度数量
      self.acc_no += acc_no
      if self.cur_push_num >= self.worker_num:
        self.cur_pull_num = 0
        self.cond.notify_all()
      self.cond.release()
    else
      self.cond.wait()
# main 实现
def train(worker_index):
  ...
if __name__ == '__main__':
  # 解析参数
  if role == 'ps':
    server = ps.ParameterServiceServer(...)
    server.serve()
  else:
    worker_index = args.worker_index
    train(worker_index)
```

Ring AllReduce 分为Split/ScatterReudce/AllGather 三个步骤(《用python实现深度学习框架》配图解释的非常好)，对于每个worker 来说，它既是右邻居的client，要把自身的梯度发送出去，又是左邻居的server，接收来自左邻居的梯度。AllReduce 的gprc 定义
```
service RingAllReduceService{
  rpc Receive(RingAllReduceReq) returns (RingAllReduceResp){}
  rpc VariableWeightsInit(VariableWeightsReqResp) returns(VariableWeightsInit){}
}
message RingAllReduceReq{
  enum Stage{
    INIT = 0;
    Scatter = 1;
    Gather = 2;
  }
  Stage stage = 1;
  NodeGradients node_gradients = 2;
}
```

VariableWeightsInit 在单机训练的场景下，各变量节点的值是随机初始化的，但是分布式训练场景下，如果多个worker节点也各自随机初始化自己的变量节点，则会导致模型参数在多个worker 节点上不一致。其实从理论上说，随机甚至还是好事，不过从编程来说，还得加上这个保证。


## 通信技术

训练框架面临的是 单机CPU与GPU 之间、单机多GPU之间、多机CPU 之间、多机GPU 之间的通信问题，有各种优化方案，但要对上层提供统一通信接口，并进一步结合机器学习的特点提供 collective Communication 接口。

### 协议层

[海思专家如何看待RDMA技术？](https://mp.weixin.qq.com/s/UqSydz8hJCFcX5CF30gXSw) 比较好的一篇文章

DMA全称为Direct Memory Access，即直接内存访问。意思是外设对内存的读写过程可以不用CPU参与而直接进行。
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

### GPU 卡间通信

[深度学习分布式训练框架 horovod (3) --- Horovodrun背后做了什么](https://mp.weixin.qq.com/s/SkByud8mz4rjulJNec6jig)
Collective communication包含多个sender和多个receiver，一般的通信原语包括 broadcast，gather,all-gather，scatter，reduce，all-reduce，reduce-scatter，all-to-all等。

![](/public/upload/machine/gpu_communication.png)

#### NCCL 

The NVIDIA Collective Communication Library (NCCL) implements multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs and Networking. NCCL provides routines such as all-gather, all-reduce, broadcast, reduce, reduce-scatter as well as point-to-point send and receive that are optimized to achieve high bandwidth and low latency over PCIe and NVLink high-speed interconnects within a node and over NVIDIA Mellanox Network across nodes.
```c
// nccl/src/nccl.h.in
ncclResult_t  ncclGroupStart();
ncclResult_t  ncclGroupEnd();
// peer to peer
ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream);
// Collective Communication 
ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff,
  size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
  cudaStream_t stream);
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
Gloo is a collective communications library. It comes with a number of collective algorithms useful for machine learning applications. These include a barrier, broadcast, and allreduce.

Gloo 为CPU和GPU提供了集合通信程序的优化实现。它特别适用于GPU，因为它可以执行通信而无需使用GPUDirect 将数据传输到CPU的内存。它还能够使用 NCCL 执行快速的节点内通信，并实现其自己的节点间例程算。你不需要考虑内存数据的拷贝，只需要实现逻辑就可以。

Gloo 支持集体通信（collective Communication），并对其进行了优化。由于 GPU 之间可以直接进行数据交换，而无需经过 CPU 和内存，因此，在 GPU 上使用 gloo后端速度更快。

#### MPI 与 NCCL/GLOO

[利用多 GPU 加速深度学习模型训练](https://mp.weixin.qq.com/s/wiqOHIVfL2gKnRUhY62EBA)多机软件设计一般采用 MPI（Message Passing Interface）实现数据交互。MPI 是一种消息传递库接口描述标准，规定了点对点消息传递、协作通信、组和通讯员概念、进程拓扑、环境管理等各项内容，支持 C 和 Fortran 语言。MPI 同样提供了前面提到的各种通信原语如 Reduce、Scatter、Broadcast 等，对应的 API 与 NCCL 十分相似。**事实上，NCCL 出现得更晚一些，参考并兼容了 MPI 已有 API**。NCCL 更多考虑了 GPU 的特性，例如**任意两块 GPU 之间的通信开销是有区别的**，跨 QPI 情况与同一 PCIe Switch 情况，以及有 NVLink/ 无 NVLink 情况就有明显差异，但 MPI 认为两种情况下 GPU 与 GPU 都是等同的，甚至 **MPI 认为跨机器的 GPU 也是等同的**，这对于多 GPU 通信效率会有影响。MPI 可以和 NCCL 结合，实现**层次化的**并行通信机制，即同一台机器上的不同 GPU 之间采用 NCCL 通信，而不同机器上的 GPU 之间采用 MPI 辅助通信。

集合通信库的主要特征是：大体上会遵照 MPI 提供的接口规定，实现了包括点对点通信（SEND,RECV等），集合通信（ REDUCE，BROADCAST，ALLREDUCE等）等相关接口，然后根据自己硬件或者是系统的需要，在底层实现上进行了相应的改动，保证接口的稳定和性能。

Horovod 在单机的多个 GPU 之上采用 NCCL 来通信，在多机（CPU或者GPU）之间通过 Ring AllReduce 算法进行通信。

## 分布式代码实现

分布式机制如何与框架融合？
1. 如何实现一个大统一的分布式通信框架？实现allreduce, allgather等collective operations通信工作。如果tensor在显存中，那么它会使用NCCL库执行。而如果是在内存中，则会使用MPI或者Gloo执行。
2. Horovod是一个库，怎么嵌入到各种深度学习框架之中？比如怎么嵌入到Tensorflow，PyTorch，MXNet，Keras？
3. 如何将梯度的同步通信完全抽象为框架无关的架构？

### ps

《用python实现深度学习框架》
```python
class Trainer(object):
  def train_and_eval(self,train_x,train_y,test_x=None,test_y=None):
    # 初始化权重变量
    self._vaiable_weights_init()
    # 开始模型训练
    ##  第一层循环，迭代epoches轮
    for self.epoch in range(self.epoches):
      ## 模型训练
      self.train(train_x,train_y)
      ## 模型评估
      if self.eval_on_train:
        ...
  def train(self,train_x,train_y):
    # 遍历训练数据集
    for i in range(len(list(train_x.values())[0])):
      self.one_step(self._get_input_values(train_x,i),train_y[i])
      if (i+1) % self.batch_size == 0:
        self._optimizer_update()
  # 执行一次前向传播和一次反向传播
  def one_step(self, data_x,data_y,is_eval=False):
    ## 根据输入节点的名称，从输入数据dict中找到对应数据
    for i in range(len(self.inputs)):
      input_value = data_x.get(self.inputs[i].name)
      self.inputs[i].set_value(np.mat(input_value).T)
    ## 标签赋值给标签节点
    self.input_y.set_value(np.mat(data_y).T)
    ## 只有在训练阶段才执行优化器
    if not is_eval:
      self.optimizer.one_step()
    @abc.abstractmethod
    def _variable_weights_init(self):
      raise NotImplementedError()
    @abc.abstractmethod
    def _optimizer_update(self):
      raise NotImplementedError()
```

《用python实现深度学习框架》在分布式场景下，以ps 模式为例，只要定义一个 DistributedTrainer 实现_variable_weights_init 和 _optimizer_update 方法即可实现一个针对PS 架构的训练器。PS：allreduce 机制也大致类似，可以查看书中示例
```python
class DistTrainerParameterServer(Trainer):
  def __init__(self,*args,**kargs):
    Trainer.__init__(self,*args,**kargs)
    cluster_conf = kargs['cluster_conf']
    ps_host = cluster_conf['ps'][0]
    self.ps_client = ps.ParameterServiceClient(ps_host)
  def _variable_weights_init(self):
    var_weight_dict = dict()
    for node in default_graph.nodes:
      if isinstance(node,Variable) and node.trainable:
        var_weight_dict[node.name] = node.value
    # 把自己的weight 初始值发给ps
    duplicated_var_weight_dict = self.ps_client.variable_weights_init(var_weights_dict)
    # 使用ps返回的初始值，重新初始化本地参数
    for var_name,weights in duplicated_var_weight_dict.items():
      update_node_value_in_graph(var_name,weights)
    print('[INIT] worker variable weights initialized')
  def _optimizer_update(self):
    # 把当前梯度上传到ps 上，此操作可能会block，直到所有节点都上传完成
    acc_gradient = self.optimizer.acc_gradient
    self.ps_client.push_gradients(acc_gradient,self.optimizer.acc_no)
    # 从ps那里把所有节点的平均梯度都拉回来，此操作可能会block，直到所有节点都下拉完成
    node_gradient_dict = self.ps_client.pull_gradients()
    # 使用得到的平均梯度，利用优化器的优化算法，更新本地变量
    self.optimizer.update(node_gradient_dict)
```

### ps tensorflow实现

在TensorFlow 的实现中，Tensor对象及内部 TensorBuffer 对象本身始终保存在 CPU 内存上，TensorBuffer 内部指针指向的缓冲区有可能位于CPU 或GPU 内存，Tensor 的跨设备复制主要指 内部缓冲区的复制过程
1. CPU 内存间的复制无需使用特殊函数，对于TensorBuffer 指向的缓冲区不做完整复制，只需复制指针并增加引用计数。
2. CPU 与GPU 内存之间的数据复制 函数 需要不同GPU 设备的DeviceContext 子类重写。

Tensorflow使用gRPC 实现进程间通信，分布式模式中的每个进程具有一个 GrpcServer 实例（也包含客户端对象），包含GrpcMasterService 和 GrpcWorkerService 两个服务

||GrpcMasterService主要用于会话相关操作|GrpcWorkerService 主要用于数据流图相关操作|
|---|---|---|
|控制接口|CreateSession,ExtendSession,...|GetStatus,RunGraph|
|数据通信接口||RecvTensor|

1. 服务端均以异步模式实现，master 服务的客户端调用多为同步模式，worker 服务的客户端调用多为异步模式
2. 对于同一进程内的 grpc 调用，均有可能直接访问本地对象
3. RecvTensor 有一定特殊性，Tensor 的大小可能很大，自定义了消息传输与解析格式（没有用grpc 自带的），客户端调用的生命周期管理也有所不同
3. 为进一步提高进程间数据通信性能，在保留gRPC 通信的同时，利用TensorFlow 的扩展机制，支持RDMA 特性的InfiniBand 协议栈，上层使用与grpc 接口一致，只需开启相关配置即可。

### horovod/allreduce 

[深度学习分布式训练框架 horovod (3) --- Horovodrun背后做了什么](https://mp.weixin.qq.com/s/SkByud8mz4rjulJNec6jig)

单机多 GPU 训练 `horovodrun -np 2 -H localhost:4 --gloo python /horovod/examples/tensorflow2/tensorflow2_mnist.py`
，`-np` 指的是进程的数量，`localhost:4`表示localhost节点上4个GPU。会启动4个进程执行 `python tensorflow2_mnist.py`（底层使用ssh进行命令分发），使用的是allreduce 模型，rank/local_rank/world_size，Rendezvous 这些概念也都有，数据也要分片。

多机多 GPU 分布式训练，这里使用 4 个服务器，每个服务器使用 4 块 GPU：`horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py`

horovodrun ==> run_commandline ==> _run ==> _run_static ==> _launch_job ==> run_controller ==> gloo_run ==> launch_gloo

1. 建立 RendezvousServer，这个会被底层 Gloo C++ 环境使用到；
    1. Horovod 在进行容错 AllReduce 训练时，除了启动 worker 进程外，还会启动一个 driver 进程。这个 driver 进程用于帮助 worker 调用 gloo 构造 AllReduce 通信环。
    2. Driver 进程需要给 Gloo 创建一个带有 KVStore 的 RendezvousServer，其中 KVStore 用于存储通信域内每个节点的 host 和 其在逻辑通信环分配的序号 rank 等信息。
    3. 这个 RendezvousServer 运行在 Horovod 的 driver 进程里。driver 进程拿到所有 worker 进程节点的地址和 GPU 卡数信息后，会将其写入RendezvousServer 的 KVStore 中，然后 worker 就可以调用 gloo 来访问 RendezvousServer 构造通信环。
2. host_alloc_plan = get_host_assignments 来根据host进行分配slot，就是horovod的哪个rank应该在哪个host上的哪个slot之上运行；
3. get_run_command 获取到可执行命令；
4. slot_info_to_command_fn 来得到在slot之上可执行的 slot command；
5. 依据 slot_info_to_command_fn 构建 args_list，这个 list 之中，每一个arg就是一个 slot command；
6. 多线程执行，在每一个 exec_command 之上执行每一个 arg（slot command）；

worker 负责训练和模型迭代。

1. 每个 worker 节点会向 RendezvousServer 发起请求来得到自己的邻居节点信息，从而构造通信环。
2. 在这个通信环之中，每个 worker 节点有一个左邻居和一个右邻居，在通信过程中，每个 worker 只会向它的右邻居发送数据，只会从左邻居接受数据。

```python
def launch_gloo(command, exec_command, settings, nics, env, server_ip):
    # Make the output directory if it does not exist
    if settings.output_filename:
        _mkdir_p(settings.output_filename)
    # start global rendezvous server and get port that it is listening on
    # 建立 RendezvousServer，这个会被底层 Gloo C++ 环境使用到
    rendezvous = RendezvousServer(settings.verbose)
    # allocate processes into slots
    # 来根据host进行分配slot，就是horovod的哪个rank应该在哪个host上的哪个slot之上运行
    hosts = parse_hosts(settings.hosts)
    host_alloc_plan = get_host_assignments(hosts, settings.num_proc)
    # start global rendezvous server and get port that it is listening on
    global_rendezv_port = rendezvous.start()
    rendezvous.init(host_alloc_plan)
    # 获取到可执行命令
    run_command = get_run_command(command, server_ip, nics, global_rendezv_port)
    # 得到在slot之上可执行的 slot command
    slot_info_to_command = _slot_info_to_command_fn(run_command, env)
    event = register_shutdown_event()
    # 依据 slot_info_to_command_fn 构建 args_list，这个 list 之中，每一个arg就是一个 slot command
    args_list = [[slot_info_to_command(slot_info), slot_info, [event]]
                 for slot_info in host_alloc_plan]
    # If an error occurs in one thread, entire process will be terminated.
    # Otherwise, threads will keep running.
    # 多线程执行，在每一个 exec_command 之上执行每一个 arg（slot command）
    res = threads.execute_function_multithreaded(exec_command,args_list,block_until_all_done=True)
    for name, value in sorted(res.items(), key=lambda item: item[1][1]):
        exit_code, timestamp = value
```

[深度学习分布式训练框架 horovod (7) --- DistributedOptimizer](https://mp.weixin.qq.com/s/0doWry-c1mEya18_7w9Jyw)Horovod 要求开发者使用Horovod自己定义的 hvd.DistributedOptimizer 代替 TensorFlow 官方的 optimizer，从而可以在优化模型阶段得到梯度。hvd.DistributedOptimizer继承keras Optimizer，然后hvd.DistributedOptimizer在其重载的get_gradients中把获取到的梯度传给`hvd.allreduce(gradients, …)`，从而实现整个horovod集群的梯度集体归并。具体计算梯度的逻辑是：
1. TF 调用 hvd.DistributedOptimizer 的 compute_gradients 方法：
  1. hvd.DistributedOptimizer 首先会利用 TF 官方 optimizer.compute_gradients 计算出本地梯度；
  2. 然后利用 AllReduce 来得到各个进程平均后的梯度；
  3. compute_gradients 返回一个(梯度，权值)对的列表。由apply_gradients使用；
2. TF 调用 hvd.DistributedOptimizer 的 apply_gradients 方法：
  1. 调用 TF 官方 optimizer.apply_gradients 对传入的参数进行处理，返回一个更新权值的op。TF 可以用这个返回值进行后续处理；
对于 TF2.x，每行代码顺序执行，不需要构建图，所以 Horovod 梯度更新部分的实现并不是基于计算图的实现

## 其它

### python 使用c

很多机器学习框架都会采用如下套路：shell脚本（可选），python端 和 C++端。

1. Shell脚本是启动运行的入口，负责解析参数，确认并且调用训练程序；
2. Python是用户的接口，引入了C++库，封装了API，负责运行时和底层C++交互；
3. C++实现底层训练逻辑；

[深度学习分布式训练框架 horovod (2) --- 从使用者角度切入](https://mp.weixin.qq.com/s/so6rsNt161F4TeR-LvU6hQ)

引入库的作用是获取到 C++ 的函数，并且用 python 封装一下，这样就可以在 python 世界使用 C++代码了。比如下文，python 的 _allreduce 函数就会把功能转发给 C++，由 MPI_LIB.horovod_allreduce 完成。

```python
def _allreduce(tensor, name=None, op=Sum, prescale_factor=1.0, postscale_factor=1.0,
               ignore_name_scope=False):
    if name is None and not _executing_eagerly():
        name = 'HorovodAllreduce_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_allreduce(tensor, name=name, reduce_op=op,
                                     prescale_factor=prescale_factor,
                                     postscale_factor=postscale_factor,
                                     ignore_name_scope=ignore_name_scope)
## 初始化时执行
def _load_library(name):
    """Loads a .so file containing the specified operators.
    """
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    return library

# Check possible symbol not found error from tensorflow version mismatch
try:
    MPI_LIB = _load_library('mpi_lib' + get_ext_suffix())
except Exception as e:
    check_installed_version('tensorflow', tf.__version__, e)
    raise e
else:
    check_installed_version('tensorflow', tf.__version__)
```