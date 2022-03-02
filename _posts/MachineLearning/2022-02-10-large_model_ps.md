---

layout: post
title: 大模型训练和ps
category: 架构
tags: MachineLearning
keywords: feature engineering

---

## 简介

* TOC
{:toc}

大模型：id 化导致模型变大，模型变大需要更多的数据才能收敛。

只要单卡放的下，走数据并行，ps 或allreduce 都行，allreduce 通信成本小一些。若规模变大
1.  稀疏模型，稀疏参数特殊处理
    1.  使用ps，加上一些稀疏tensor 的优化，且将 embedding  存储和更新 负担转嫁到 ps
    2.  稠密参数allreduce，想办法解决 稀疏tensor 的存储、通信成本。 比如 [HybridBackend](https://github.com/alibaba/HybridBackend)架构中参数放在 worker 上：稠密参数 replication 存放，每个 worker 都有副本，梯度更新时进行 allreduce；稀疏参数 partition 存放，每个 worker 只有部分分片，梯度更新时进行 alltoall。allreduce 和 alltoall 都会使用 nccl 进行同步通信，效率较高。hb 进行 alltoall 时，通信的是稀疏梯度，而不是完整的参数，通信量上和 ps 是一致的，但是通信效率会更高。
2.  稠密模型，单卡无论如何也放不下了，就只能采取模型并行 及附属的一些优化方案

[基于tensorflow做扩展支持大模型的做法](https://zhuanlan.zhihu.com/p/396804900)

## 什么是大模型

[TensorFlow在推荐系统中的分布式训练优化实践](https://mp.weixin.qq.com/s/LjdHBEyQhJq3ptMj8XVT-w)随着美团业务的发展，推荐系统模型的规模和复杂度也在快速增长，具体表现如下：
1. 训练数据：训练样本从到百亿增长到千亿，增长了近10倍。
2. 稀疏参数：个数从几百到几千，也增长了近10倍；总参数量（也就是tf.Variable）从几亿增长到百亿，增长了10~20倍。
3. 模型复杂度：越来越复杂，模型单步计算时间增长10倍以上。
对于大流量业务，一次训练实验，从几个小时增长到了几天，而此场景**一次实验保持在1天之内是基本的需求**。

[深度学习分布式训练的现状及未来](https://zhuanlan.zhihu.com/p/466002243)大模型主要分为两类：
1. 搜索、推荐、广告类任务，它的特点是海量样本及大规模稀疏参数（sparse embeddings），适合使用 CPU/GPU 参数服务器模式（PS）；参数服务器模式从第一代 Alex Smola 在 2010 年提出的 LDA（文本挖掘领域的隐狄利克雷分配模型），到第二代 Jeff Dean 提出的 DistBelief，接着到第三代李沐提出的相对成熟的现代 Parameter Server 架构，再到后来的百花齐放：Uber 的 Horvod，阿里的 XDL、PAI，Meta 的 DLRM，字节的 BytePs、美团基于 Tensorlow 做的各种适配等等。参数服务器的功能日趋完善，性能也越来越强，有纯 CPU、纯 GPU，也有异构模式。
2. CV、NLP 任务，它的特点是常规样本数据及大规模稠密参数，它适合用纯 GPU 集合通信模式（Collective）。基于纯 GPU 的集合通信模式的分布式训练框架，伴随着 Nvidia 的技术迭代，特别是 GPU 通信技术（GPU Direct RDMA）的进步，性能也变得愈来愈强。

[广告推荐中大规模分布式模型](https://zhuanlan.zhihu.com/p/161972813) 为啥一两百类的特征，我们却总是听说大规模特征？举个例子，用户 userid 这一维特征，比如系统中用户有1亿个，那么每个 id 实际上也可以当做一个独立的特征对待。这样一算，特征规模就上去了。这里就要重新理解 embedding 的概念了。对于模型而言，id 查了embedding表后得到向量，输入进来进行计算，是对数据进行抽特征。如果类比到图像分类，抽取 rgb 特征来分类 （一个值变成 3个255）

[参数量卷到一百万亿！华人团队开源史上最大的推荐训练系统Persia](https://mp.weixin.qq.com/s/N1C-GVVxs2Hm6-xVWNvnzg)
一般来说，推荐系统模型首先需要将不同的ID特征（如用户ID和session ID）映射到一个固定长度的低维向量，而系统中的用户ID、交叉特征数量都特别多，就需要更大规模的模型来捕获特征和映射。但更大规模的embedding layer也需要更大的内存来载入，不得不说大模型太费钱了！

![](/public/upload/machine/recsys_big_model.png)

有了embedding后，剩下的工作就简单了，设计后续layer来适配不同的任务。通常只占据整个模型的0.1%，无需大内存，主要是一些计算密集型的工作。

在实现上
1. 推理服务在运行时 也会访问ps （distributed inference），根据 ID feature 查询对应的 embedding 向量。当然，有的框架直接将 ps 组件的功能内嵌到各个worker 上了。
2. 针对 大模型 包含 embedding layer的场景，input 层和 embedding 层之间不是全连接的，而是一个 embedding_lookup 的Op
3. 常规的dense 模型，input是一个一维向量。 针对多个id feature，为了 确保与模型的input 层对齐，input 实际变成了一个 `map<string,tensor>`，key 为id feature 名字，value 为id feature 值对应的 tensor。


## 大了难在哪

[搞定大模型训练](https://mp.weixin.qq.com/s/xAnfeR4hhR6bFiRMe8rS4A)

我们的模型可能会很大，或者数据量会很大。仅仅用一块GPU卡可能连模型都放不下，或者batch size只能设置的很小，但是我们知道有些情况下大的batch size往往会提供更好的效果。

1. 假设我们只有一个GPU，我们的模型一次只能输入batch size为8的数据，那么我们怎么样实现batch size为32的更新呢？那就需要时间换空间了，即我们训练32/8=4步才去更新模型，也就是所谓的梯度累积。
2. Gradient-Checkpointing，   那么如果你的GPU连batch size为1都跑不了怎么办？我们在训练深度学习模型的时候，需要先做前向传播，然后将中间得到的激活值存储在内存中，然后反向传播的时候再根据loss和激活值计算梯度。也就是说内存消耗其实跟模型的层数线性相关。那么怎么减少这个内存消耗呢？最简单的想法就是我不存这些中间信息，计算梯度的时候，到了某一层我重新计算它的激活值，这个方法虽然可以让内存消耗是个常量，但是运行时间会是`O(n^2)`，这是没法接受的。那么就有一个折中的办法，我不存全部的中间数据，只存部分，那么我们在计算梯度的时候不需要从头计算了，只需要从最近的checkpoint点计算就好。
3. 我们训练模型一般都是用单精度(FP32)的参数，但是其实我们还使用半精度(FP16)。半精度可以降低内存消耗，从而训练更大的模型或者使用更大的batch size；同时运算时间受内存和算术带宽的限制，在有些gpu(Tensor cores)上可以为半精度提供更大的算术带宽，从而提高训练效率，减少inference用时。

[参数量卷到一百万亿！华人团队开源史上最大的推荐训练系统Persia](https://mp.weixin.qq.com/s/N1C-GVVxs2Hm6-xVWNvnzg)模型训练时消耗的显存，往往是参数量的几倍还多。

[知乎高赞回答——为什么说大模型训练很难？](https://mp.weixin.qq.com/s/r4e_vrj4yqV7PAzHtcisCQ)

1. 算子拆分  单个矩阵乘法可以分到两个device上计算 `Y = WX = [W1,W2]X = [W1X,W2X]`。我们在工程上要做的就是：将切分到两个device上，将复制到两个device上，然后两个device分别做矩阵乘法即可。有的时候，切分会带来额外的通信，比如矩阵乘法切到了reduction维度上，为了保持语义正确，就必须再紧跟一个AllReduce通信。 这里复杂之处在于，你不能无脑地将所有算子全部做拆分，因为拆分可能会引入一些额外通信，降低总体吞吐。所以你得做些分析，决定哪些算子被拆分，现在大部分框架都不支持这种全自动化策略，要么是半自动或纯手工，要么是针对某种模型把它的拆分方案写死。所以只能造轮子解决这个事
2. 流水并行  不切算子，而是将不同的Layer切分到不同的Device上，就可以形成Pipeline方案，GPipe就是这样一种方案，提出了将一个batch拆分成若干个micro-batch，依次推入到Pipeline系统中，即可消除Bubble time。和算子拆分类似，全自动化方案工作量不小，比如Pipeline怎么切，才能让通信量小，计算还能均匀，这需要一定的算法和工程量


[浅谈工业界分布式训练（一）](https://mp.weixin.qq.com/s/hErbnqv49xTqjJANtL-G0Q) 除了上述的数据量级大，不同场景下分布式训练的痛点
对CV和NLP场景
1. CV和NLP场景模型复杂，单机性能要求高，比如卷积的计算时间在CPU上和 GPU上相差十倍到几十倍。 ==> 业界主要使用高性能的GPU进行计算，并采用All-reduce的通信拓扑进行参数的同步更新。
2. 模型大（DenseNet部分），比如NLP领域，GPT-3这样的模型高达1750亿参数，显存占用高达2.8 TB，单机内存无法容纳。而Bert-Large虽然只有3.4亿参数规模，但由于各种内存占用，在16G V100上，训练也仅能使用batch Size=8。 ==> 当面对GPT-3这种DenseNet部分大的模型，Allreduce 单卡内存无法容纳，我们需要采用模型并行(model parallelism)的方式将计算图划分到不同的设备上构建有向无环图(DAG)进行分布式训练，其中Gpipe, Megatron, Oneflow和Whale都提出模型并行的相关解决方案。相比于数据并行每个worker只有一部分数据，模型并行下每个node使用所有数据.
    1. Intra-layer parallelism(Tensor Parallelism) 。主要是将一层Layer中的矩阵计算分别拆分到不同的机器上进行运算，比如简单的Y_1=W_1 X_1这一次矩阵乘法中，我们将模型参数W_1或者输入数据X_1，按某个维度分别拆分到不同设备上计算，比如1D Megatron。
    2. Inter-layer parallelism（Pipeline Parallelism）。而Inter-Layer Parallism会将模型的layers拆分到不同的机器上，则一次forward就需要跨过不同的机器串行地进行运算，而流行并行通过将batch size切分为更小的mirco batch，减少数据依赖，从而将整个计算过程异步起来，最大化资源利用率。举例来说，在一个简单的三层MLP中（的Y_i = W_i X_i, i=1,2,3）会存在三次矩阵乘法 W_i X_i，流水线并行会把W_i X_i分别分配到三台机器上进行运算。

推广搜场景
1. 模型小，词表大。模型中的DenseNet部分，不像BERT是模型巨大词表小，往往一台机器的内存就可以容纳，但是其特征量级可能高达成百上千亿，造成Sparsenet部分或者Embedding lookup table高达TB级别，使得单机无法容纳。
2. 一个batch的embedding lookup量级大，造成查询耗时大。由于特征数量多，一个batch可能包含几十万个ID类特征，tf原生的embedding lookup查询耗时大，造成训练和inference性能低。尤其在线上inference的时候，无法在给定RT内完成服务响应。
3. 数据具有大规模稀疏的特点。不同于CV和NLP场景，数据是稠密的图像和文本，搜广推的数据非常稀疏的，第一这来源于很多数据无法对所有用户和场景有效采集到，第二是因为建模使用的特征量级大造成的高维稀疏性。这会影响了数据的存储格式和计算效率。

针对上述的问题，各个大厂的训练框架进行很多相关优化，目前总结下来，核心的两点，一个在于分布式通信拓扑的设计，还有一个在于Embedding Lookup的性能优化。


[第一视角：深度学习框架这几年](https://mp.weixin.qq.com/s/MEy_aGOUeWPDcQnI9-M5Bg) 作者深度参与了tf 等几个框架，对很多事情有独到的了解


## ps server 优化

第一手的材料 就是李沐大神的 论文[Scaling Distributed Machine Learning with the Parameter Server](https://web.eecs.umich.edu/~mosharaf/Readings/Parameter-Server.pdf) **巨大的模型其实就是巨大的参数**。以及一个大神的简单 c实现 [Superjomn/SwiftSnails](https://github.com/Superjomn/SwiftSnails)

分布式常用的2种模式有ParameterServer 和 AllReduce/RingAllReduce。随着开源框架的火热迭代，再者 GPU 显存也越来越大，AllReduce 模式研究的更多一点。毕竟大多数研究的还是 dense 模型，就算上百层的网络，参数也大不到哪去。所以很多训练都是数据并行，每个节点可以存储完整的模型参数。但是像 CTR 这类用到大规模离散特征的，本质也是一个大规模离散模型，一般称为 sparse 模型。几百 G 的模型很常见，一个节点也不太可能放的下。这时候 parameter Server 模型更加适用一些。

两点设计使ps能够克服Master/Slave架构应用于大规模分布式训练时的困难：
1. 所有参数不再存储于单一的master节点，而是由一群ps server节点负责存储、读写；
2. 得益于推荐系统的特征是超级特征的特点，一个batch的训练数据所包含的feature数目是有限的，因此，我们没必要训练每个batch时都将整个模型（i.e., 上亿的embedding）在server/worker之间传来传去，而只传递当前batch所涵盖的有限几个参数就可以了。

### 异步

每个worker 互相不干扰，各自 pull 参数，然后计算梯度后，再通过 push 将梯度值回传给 server。server 再汇总所有 worker 的梯度结果后一起更新最终参数。这里有2个问题
1. 一个是有同步更新还是异步更新，虽然各有利弊，但一般都是采取异步。
2. 另一个问题是pull-计算-push 这个操作太频繁，通信有压力，拖慢计算。所以可以采取时效性不那么高的方法，就是不必每次都 pull 和 push，比如可以隔3次 pull 一下，隔5次 push 一下。经过多轮训练后，模型的参数就训练完了。

[机器学习参数服务器ps-lite (1) ----- PostOffice](https://mp.weixin.qq.com/s/4scg6j0ae8IxyGHEOAXHcg)
1. 参数服务器是机器学习领域的分布式内存数据库，其作用是存储模型和更新模型。
2. 在参数服务器之前，大部分分布式机器学习算法是通过定期同步来实现的，比如集合通信的all-reduce，或者 map-reduce类系统的reduce步骤。当async sgd出现之后，就有人提出了参数服务器。

[重温经典之ps-lite源码解析(1)：基础](https://zhuanlan.zhihu.com/p/467650462)在纯异步的ASP模式中，每台worker在发送完自己的梯度后，不用等其他worker，就可以开始训练下一个batch的数据。由于无须同步，ASP的性能优势比较明显。但是，的确存在可能性，一个非常慢的worker基于老参数计算出过时梯度，传到server端会覆盖一些参数的最新进展。但是在实际场景下，由于推荐系统的特征空间是超级稀疏的，因此两个worker同时读写同一feature造成冲突的可能性还是较低的，因此纯异步ASP模式的应用还是比较普遍的。

[浅谈工业界分布式训练（一）](https://mp.weixin.qq.com/s/hErbnqv49xTqjJANtL-G0Q)同步更新虽然可以保证Consistency，但由于各节点计算能力不均衡无法保证性能，而异步更新或者半同步更新并没有理论上的收敛性证明，Hogwild!算法证明了异步梯度下降在凸优化问题上按概率收敛，而深度学习问题一般面对的是非凸问题，所以无论异步和半同步算法都无收敛性的理论保障。所以**只是根据经验，大部分算法在异步更新是可以收敛**，求得最优或者次优解（其实现在无论是学术界和工业界，深度学习只要效果好就行）。当然目前比较好的方式针对针对SparseNet部分( 低IO pressure, 但高memory consumption)，DenseNet部分 (高IO pressure，但低memory consumption)的特点，对sparsenet进行异步更新（因为Embedding Lookuptable的更新是稀疏的，冲突概率低），DenseNet采用同步更新/AllReduce的方式尽量逼近同步训练的效果。

hogwild 一般特指 parameter server 最常见的用法：完全无锁的异步训练。hogwild 这个术语本身在学术界用的更多，工程上用的比较少了。


### 分布式

ps server 并不是只有一个master 来分发所有参数，而是存在一个 Server group，即多个 server 机器，每个机器只负责存一部分参数就行。这样就避免唯一 master 不停广播的通信代价问题。前面说了，server 存的是`<key,value>`，每个 server 上这个 key 的范围就是通多一致性哈希来分配的。这样设计有个好处，就是加入一个新节点，或者丢失删除一个节点后，参数可以通过环形只在相邻的节点上调整即可，避免节点频繁大规模的参数变动。

![](/public/upload/machine/multi_ps.png)

[ElasticDL Parameter Server Design](https://aiqianji.com/frankiegu/elasticdl/src/d727d3d8ee4cf8254f18a5f9a001b5471587864c/docs/designs/parameter_server.md)

1. 可以存的更多。models could be large and overrun the memory space of a single PS instance. In such case, we need to partition the model and store different partitions in different PS instances. 
2. 分担通信负担。distributes the model parameter communication from workers among PS instances. 

优化点：每个kv都是都是很小的值，如果对每个key都发送一次请求，那么服务器会不堪重负。为了解决这个问题，可以考虑利用机器学习算法中参数的数学特点（即参数一般为矩阵或者向量），将很多参数打包到一起进行更新。

### 存储两类数据

[广告推荐中大规模分布式模型](https://zhuanlan.zhihu.com/p/161972813) ps server 上存储的参数格式是`<key, value>`对，支持set/get/update 以及自定义函数。每个 worker 读入一个 batch 数据后，会向 server 执行 pull 操作，获取当前计算需要的参数的最新的值。比如稀疏参数的获取，发现样本中，location 这一维特征有北京，上海，南京3个sign，那么则将这3个当做 key 去请求 server 获得当前的最新 embedding 向量。计算前向和梯度，也是需要 dense 模型参数的，所以也会 pull DNN 网络的参数。

Each PS node has a dictionary-based data structure to store its partition of model parameters.We consider two kinds of model parameters:

1. non-embedding parameters，一般不大，不分区，存在一个ps 上，tf.Variable name 作为key，tf.Variable 作为value。可以使用 `hash(p_name) % N` 选择存储的ps 
2. **embedding tables**，Each embedding layer has an embedding table which maps a discrete ID i to an embedding vector vᵢ. Embedding tables could be huge, especially in some recommending and ranking models. Thus, we partition each embedding table and store every partition in an unique PS pod. For an embedding vector vᵢ, we select the (i mod N)-th parameter server to store it. **To store an embedding vector**, We use its corresponding embedding layer name and discrete ID as the key, and a 1-D numpy.ndarry as the value.  PS：例如一个形状为 [m, n, l, k]  的 tensor 可以按切片的数量保存为 m 个形状为 [n, l, k] 的 KV 数据，key 为 tensor_name 和 m 维度序号组合的唯一命名。 [针对大规模 Embedding 参数的定制处理](https://mp.weixin.qq.com/s/JGbELXp0aLn9n7JE1wQXvA)


初始化：We use lazy initialization for model parameters in PS. PS does not have the model definition. Thus **workers are responsible for initializing parameters** and push the initialized parameters to corresponding PS pods. Each PS pod has a parameter initialization status, which is False after the PS pod launch. 
1. When a worker tries to get non-embedding parameters from the PS pod through a RPC call pull_variable, the PS pod tells the worker that the parameter initialization status is False in response. If the worker has already initialized non-embedding parameters, it sends non-embedding parameter values to the PS pod by a gRPC call push_model. When the PS pod receives non-embedding parameters in its first RPC service for push_model, it initializes non-embedding parameters and sets the parameter initialization status as True.PS: 这也是为什么 ps 挂了没事，worker 会push
2. For an embedding vector, the corresponding PS pod will initialize it in the first pull_embedding_vector service that contains this embedding vector. The PS pod needs the embedding vector size and the initialization method for the initialization. The embedding vector size and the initialization method are in the model definition and workers can send them in push_model to PS pods together with non-embedding parameter values.

参数更新：
1. A worker computes gradients in each training iteration, which contain gradients for non-embedding parameters and some embedding vectors if applicable. The worker partitions these gradients using their corresponding parameter names or discrete IDs for embedding vectors. Then the worker sends gradient partitions to their corresponding PS pods by RPC calls push_gradient.When a PS pod receives gradients in push_gradient, it uses a TensorFlow optimizer to apply gradients to non-embedding parameters.
1. We have already implemented an OptimizeWrapper to sparsely update embedding vectors. **OptimizeWrapper uses corresponding embedding vectors to form a temporary variable**, applies gradients to this temporary variable, and writes results back to these embedding vectors. The PS pod can use this OptimizeWrapper directly to update embedding vectors. 

故障恢复：The model may contain one or more embedding layers with embedding tables as their parameters. If so, a minibatch of training data in a worker contains some embedding IDs, which correspond to a subset of embedding tables. The worker pulls all non-embedding parameters and only a subset of embedding tables from PS pods in the training. Thus, the PS pod can recover non-embedding parameters from workers but not embedding tables.
1. For non-embedding parameters, the PS pod can recover them from workers in the same way as the parameter initialization by setting its parameter initialization status as False.
1. For embedding tables, PS creates replicas to support fault-tolerance. For each PS pod PSᵢ, it can store M replicas of its embedding table partitions in M PS pods indexed from (i+1) mod N to (i+M) mod N. The relaunched PS pod can recover embedding tables from one of its replicas. PS: 一个ps 存了两份 replica，还周期性的同步呢。

Live replication of parameters between servers supports hot failover. Failover and selfrepair in turn support dynamic scaling by treating machine removal or addition as failure or repair respectively. PS：多副本 ==> 容错 ==> 弹性。每个参数会在PS集群中有三个副本，存储在不同的节点上来实现冗余。其中一个节点会被选为primary，来提供针对某个参数的服务。当这个节点失效时，另外两个副本中会被挑选出一个作为新的primary，来继续此参数的服务。因而，参数服务器也是需要调度的。

### 一种ps实现

[ElasticDL Parameter Server Design](https://aiqianji.com/frankiegu/elasticdl/src/d727d3d8ee4cf8254f18a5f9a001b5471587864c/docs/designs/parameter_server.md)

Message Definition
```
message Tensor {
    enum DataType {
        BOOL = 0;
        INT16 = 1;
        INT32 = 2;
        INT64 = 3;
        FP16 = 4;
        FP32 = 5;
        FP64 = 6;
    }
    string name = 1;
    DataType data_type = 2;
    repeated int64 dim = 3;
    bytes content = 4;
    repeated int64 indices = 5;
}
message EmbeddingTableInfo{
    string name = 1;
    repeated int64 dim = 2;
    string initializer = 3;
}
message Model {
    int64 version = 1;
    # repeated 则表示数组
    repeated Tensor variables = 2;
    repeated EmbeddingTableInfo embedding_table_info = 3;
}
message PullVariableRequest{
    int64 version = 1;
}
message PullVariableResponse{
    bool model_init_status = 1;
    Model model = 2;
}
message PushGradientRequest{
    int32 model_version = 1;
    repeated Tensor gradients = 2;
}
message PushGradientResponse{
    bool accepted = 1;
    int32 model_version = 2;
}
message PullEmbeddingVectorRequest{
    string name = 1;
    repeated int64 ids = 2;
}
message SynchronizeEmbeddingRequest {
    int32 replica_index = 1;
}
message SynchronizeEmbeddingResponse {
    repeated Tensor embedding_vectors = 1;
}
```
RPC Definition
```
service PServer{
    # pull trainable tensorflow variables created by Keras layers
    rpc pull_variable(PullVariableRequest) returns (PullVariableResponse);

    # pull embedding vectors in ElasticDL embedding layers
    # Do we need to create a new message `PullEmbeddingVectorRequest` rather than use `Tensor`?
    rpc pull_embedding_vector(PullEmbeddingVectorRequest) returns (Tensor);

    # push trainable tensorflow variables and meta info for ElasticDL embedding layers
    rpc push_model(Model) returns (google.protobuf.Empty);

    rpc push_gradient(PushGradientRequest) returns (PushGradientResponse);

    # PS to recover embedding vectors after relaunch
    rpc get_replica(SynchronizeEmbeddingRequest) returns (SynchronizeEmbeddingResponse);

    # PS replica synchronization
    rpc synchronize_embedding(SynchronizeEmbeddingRequest) returns (SynchronizeEmbeddingResponse);
}
```

Data Structure

```python
class Tensor(object):
    def __init__(self, name=None, value=None, indices=None):
        self.name = name
        self.value = value
        self.indices = indices
    @classmethod
    def from_tensor_pb(cls, tensor_pb):
        """Create an ElasticDL Tensor object from tensor protocol buffer.
        Return the created tensor object.
        """
        pass
    def to_tensor_pb(self):
        pass
    def to_tf_tensor(self):
        pass
    def to_ndarray(self):
        pass

def serialize_to_pb(tensor, pb):
    """Serialize ElasticDL Tensor to tensor protocol buffer."""
    pass
def deserialize_from_pb(pb, tensor):
    """Deserialize tensor protocol buffer to ElasticDL Tensor."""
    pass
def tensor_pb_to_ndarray(tensor):
    """Deserialize tensor protocol buffer and return a numpy ndarray."""
    pass
def tensor_pb_to_tf_tensor(tensor):
    """Deserialize tensor protocol buffer and return a TensorFlow tensor."""
    pass
# In `Parameters`, interfaces `set_*_param` have two arguments, `value` and `name` (or `layer_name`).If `value` is a ElasticDL `Tensor` instance, `name` can be None.Otherwise `value` is a numpy ndarray, and `name` must be specified.
class Parameters(object):
    def __init__(self):
        # Parameter initialization status
        self.parameter_init_status = False
        # Non-embedding parameter dict, maps parameter name to tf.Variable instance
        self.non_embedding_params = {}
        # Embedding table dict, maps embedding layer name to `EmbeddingTable` instance
        self.embedding_params = {}

    @property
    def non_embedding_params(self):
        return self._non_embedding_params
    def set_embedding_param(self, value, layer_name=None):
        pass
    def get_embedding_param(self, layer_name, ids):
        return self._embedding_params.get(layer_name).get(ids)
    def set_non_embedding_param(self, value, name=None):
        pass
    def init_non_embedding_param(self, value, name=None):
        pass
    def set_meta_info(self, layer_name, dim, initializer):
        pass

class EmbeddingTable(object):
    def __init__(self, dim, initializer):
        # Embedding vector dict, maps ID to 1-D numpy.ndarray
        self._embedding_vectors = {}
        # the dimension of embedding vectors
        self._dim = dim
        # the initializer name for initializing embedding vectors
        self._initializer = initializer

    def get(self, ids):
        values = []
        for id in ids:
            if id not self._embedding_vectors:
                val = initialize_embedding_vector(self._dim, self._initializer)
            else:
                val = self._embedding_vectors.get(id)
            values.append(val)
        return np.concatenate(values).reshape(len(ids), -1)

    def set(self, ids, values):
        pass
```

Here is the pseudocode for a worker to pull variable from the PS. If the non-embedding variables are not initialized, the PS will tell the worker to initialize them and report to the PS.
```python
class PServer(elasticdl_pb2_grpc.PServerServicer):
    ...
    def pull_variable(self, request):
        res = PullModelResponse()
        if self._need_initialize_model:
            res.model_init_status = True
            return res
        res.model_init_status = False
        res.model = self._get_model() # get model in this PS instance
        return res

    def push_model(self, request):
        model = request.model
        ... # initialize model in this PS instance

class Worker(object):
    ...
    def pull_variable(self):
        # for-loop should be implemented in multithread
        for ps_index in range(self._ps_node_num):
            req = PullModelRequest() # create request code keeps the same with current code
            res = self._stub[ps_index].pull_variable() # pull variable from PS
            if res.model_init_status:
                # worker initializes its model here if needed
                model = serialize_model_to_pb()
                self._stub[ps_index].push_model(model) # get model in this worker
            req = PullModelRequest() # create request code keeps the same with current code
            res = self._stub[ps_index].pull_variable() # pull variable from PS
            if res.model_init_status:
                raise Error or try a pre-defined constant times
```

## embedding/稀疏场景优化

[一文梳理推荐系统中Embedding应用实践](https://mp.weixin.qq.com/s/9vnCX4IuHsA3hUi6t0Y0KQ)**在自然语言中，非端到端很常见**，因为学到一个好的的词向量表示，就能很好地挖掘出词之间的潜在关系，那么在其他语料训练集和自然语言任务中，也能很好地表征这些词的内在联系，预训练的方式得到的Embedding并不会对最终的任务和模型造成太大影响，但却能够「提高效率节省时间」，这也是预训练的一大好处。但是**在推荐场景下，根据不同目标构造出的序列不同，那么训练得到的Embedding挖掘出的关联信息也不同**。所以，「在推荐中要想用预训练的方式，必须保证Embedding的预训练和最终任务目标处于同一意义空间」，否则就会造成预训练得到Embedding的意义和最终目标完全不一致。比如做召回阶段的深度模型的目标是衡量两个商品之间的相似性，但是CTR做的是预测用户点击商品的概率，初始化一个不相关的 Embedding 会给模型带来更大的负担，更慢地收敛。



[TensorNet——基于TensorFlow的大规模稀疏特征模型分布式训练框架](https://mp.weixin.qq.com/s/v8HqeR7UYFs4Ex5p5tFyVQ)对于最简单的wide&deep模型，如果在一个广告系统中有3亿用户，那么就需要定义一个维度为3亿的embedding矩阵，在训练模型时需要在这个3亿维的矩阵上做embedding_lookup得到当前batch内的用户的embedding信息，近而在embedding之上做更加复杂的操作。

![](/public/upload/machine/ps_embedding.png)

TensorNet使用一个较小的，可以容纳特征在一个batch内所有数据的embedding矩阵代替TensorFlow默认实现中需要定义的较大的embedding矩阵。

TensorNet异步训练架构
1. TensorNet将sparse参数和与dense参数分别使用不同的parameter server管理。
2. TensorNet不设单独的parameter server节点。在每个worker中都会维护一个sparse paramter server和dense parameter server。这省去了开发人员管理ps节点和worker节点的不少麻烦。
3. TensorNet将模型的所有dense参数合并后使用分布式数组切分到不同的机器上，每次pull和push参数的时候只有一次网络请求。相较于TensorFlow对每个tensor都有一次网络请求的方法极大的减少了网络请求的次数从而提升了模型训练的速度。

TensorNet同步训练架构
1. TensorNet使用单独的sparse parameter server节点保存所有sparse参数。通过parameter server可以解决TensorFlow支持的sparse特征维度不能太大的问题。
2. TensorNet对sparse参数做了特殊的定制化的同步。TensorNet在训练时只同步当前训练的batch所关注的稀疏特征，相较于TensorFlow会将所有参数都同步的模式通信数据减少到了原来的万分之一，乃至十万分之一。

[TensorFlow在推荐系统中的分布式训练优化实践](https://mp.weixin.qq.com/s/LjdHBEyQhJq3ptMj8XVT-w)在原生的TensorFlow中构建Embedding模块，用户需要首先创建一个足够装得下所有稀疏参数的Variable，然后在这个Variable上进行Embedding的学习。

```python
user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
```
然而，使用Variable来进行Embedding训练存在很多弊端：
1. Variable的大小必须提前设定好，对于百亿千亿的场景，该设定会带来巨大的空间浪费；
2. 训练速度慢，无法针对稀疏模型进行定制优化。
我们首先解决了有无的问题，使用HashTable来替代Variable，将稀疏特征ID作为Key，Embedding向量作为Value。相比原生使用Variable进行Embedding的方式，具备以下的优势：
1. HashTable的大小可以在训练过程中自动伸缩，避免了开辟冗余的存储空间，同时用户无需关注申请大小，从而降低了使用成本。
2. 针对HashTable方案实施了一系列定制优化，训练速度相比Variable有了很大的提高，可以进行千亿规模模型的训练，扩展性较好。

核心流程大致可以分为以下几步：
1. 稀疏特征ID（通常我们会提前完成统一编码的工作）进入Embedding模块，借助TensorFlow搭建的Send-Recv机制，这些稀疏特征ID被拉取到PS端，PS端上的Lookup等算子会实际从底层HashTable中查询并组装Embedding向量。
2. 上述Embedding向量被Worker拉回进行后续训练，并通过反向传播计算出这部分参数的梯度，这些梯度进一步被位于PS端的优化器拉回。
3. PS端的优化器首先调用Find算子，从HashTable获取到梯度对应的原始稀疏参数向量和相应的优化器参数，最终通过优化算法，完成对Embedding向量和优化器参数的更新计算，再通过Insert算子插入HashTable中。

[HybridBackend](https://github.com/alibaba/HybridBackend)架构中参数放在 worker 上：
1. 稠密参数 replication 存放，每个 worker 都有副本，梯度更新时进行 allreduce；
2. 稀疏参数 partition 存放，每个 worker 只有部分分片，梯度更新时进行 alltoall。

allreduce 和 alltoall 都会使用 nccl 进行同步通信，效率较高。hb 进行 alltoall 时，通信的是稀疏梯度，而不是完整的参数，通信量上和 ps 是一致的，但是通信效率会更高。



## 其它

[第一视角：深度学习框架这几年](https://mp.weixin.qq.com/s/MEy_aGOUeWPDcQnI9-M5Bg) 
1. 推荐场景在电商，视频，资讯等众多头部互联网公司的火爆导致推荐系统对AI硬件的消耗在互联网公司超过了传统NLP，CV，语音等应用的总和。
2. 无量一开始采用的是基于参数服务器的架构。对tf 的改造有两个方向
    1. 把TensorFlow作为一个本地执行的lib，在外围开发，TensorFlow被复用来提供Python API以及完成基础算子的执行。而参数服务器，分布式通信等方面则是自己开发，没有复用TensorFlow。
    2. 基于TensorFlow底层进行改造，研发难度会比较大，而且很可能与社区版TensorFlow走向不同的方向，进而导致TensorFlow版本难以升级。