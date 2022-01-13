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


## 架构模式

### 区别

[深度学习分布式训练框架——基础知识](https://mp.weixin.qq.com/s/djGvx3fNJfKCXmjwTfJ-CA)在中等规模模型情况下，all-reduce 更适合。当规模巨大时候则应该使用参数服务器。

参数服务器 适合的是高纬稀疏模型训练，它利用的是维度稀疏的特点，每次 pull or push 只更新有效的值。但是深度学习模型是典型的dense场景，embedding做的就是把稀疏变成稠密。所以这种 pull or push 的不太适合。而 网络通信上更优化的 all-reduce 适合中等规模的深度学习。

又比如由于推荐搜索领域模型的 Embedding 层规模庞大以及训练数据样本长度不固定等原因，导致容易出现显存不足和卡间同步时间耗费等问题，所以 all-reduce 架构很少被用于搜索推荐领域。

### 代码示例

《用python实现深度学习框架》给出的ps/worker 模型的grpc 通信api定义
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
[深度学习分布式训练框架 horovod (3) --- Horovodrun背后做了什么](https://mp.weixin.qq.com/s/SkByud8mz4rjulJNec6jig)
Collective communication包含多个sender和多个receiver，一般的通信原语包括 broadcast，gather,all-gather，scatter，reduce，all-reduce，reduce-scatter，all-to-all等。具体实现
1. The NVIDIA Collective Communication Library (NCCL) implements multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs and Networking. NCCL provides routines such as all-gather, all-reduce, broadcast, reduce, reduce-scatter as well as point-to-point send and receive that are optimized to achieve high bandwidth and low latency over PCIe and NVLink high-speed interconnects within a node and over NVIDIA Mellanox Network across nodes.
2. Gloo is a collective communications library. It comes with a number of collective algorithms useful for machine learning applications. These include a barrier, broadcast, and allreduce.

集合通信库的主要特征是：大体上会遵照 MPI 提供的接口规定，实现了包括点对点通信（SEND,RECV等），集合通信（ REDUCE，BROADCAST，ALLREDUCE等）等相关接口，然后根据自己硬件或者是系统的需要，在底层实现上进行了相应的改动，保证接口的稳定和性能。

Gloo 为CPU和GPU提供了集合通信程序的优化实现。它特别适用于GPU，因为它可以执行通信而无需使用GPUDirect 将数据传输到CPU的内存。它还能够使用 NCCL 执行快速的节点内通信，并实现其自己的节点间例程算。你不需要考虑内存数据的拷贝，只需要实现逻辑就可以。

Gloo 支持集体通信（collective Communication），并对其进行了优化。由于 GPU 之间可以直接进行数据交换，而无需经过 CPU 和内存，因此，在 GPU 上使用 gloo后端速度更快。

## python 使用c

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