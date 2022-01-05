---

layout: post
title: tensorflow学习
category: 架构
tags: MachineLearning
keywords:  tensorflow

---

## 简介

* TOC
{:toc}

TensorFlow，这是个很形象的比喻，意思是 张量(Tensor)在神经网络中流动(Flow)。

1. 在数学中，张量是一种几何实体(对应的有一个概念叫矢量)，广义上可以表示任何形式的数据。在NumPy等数学计算库或TensorFlow等深度学习库中，我们通常使用**多维数组**来描述张量，所以不能叫做矩阵，矩阵只是二维的数组，张量所指的维度是没有限制的。[线性代数/矩阵的几何意义](https://mp.weixin.qq.com/s/bi1gOmUK_GU_1cfiWQPF6Q) 未读完 
2. 张量这一概念的核心在于，它是一个**数据容器**。它包含的数据几乎总是数值数据，因此它是数字的容器。
3. 在物理实现时（TensorFlow）是一个句柄，它存储张量的元信息以及指向张量数据的内存缓冲区指针。
4. 张量是执行操作时的输入输出数据，用户通过执行操作来创建或计算张量，张量的形状不一定在编译时确定，可以在运行时通过形状推断计算出。  

TensorFlow 使用库模式（不是框架模式），工作形态是由用户编写主程序代码，调用python或其它语言函数库提供的接口实现计算逻辑。用户部署和使用TensorFlow 时，不需要启动专门的守护进程，也不需要调用特殊启动工具，只需要像编写普通本地应用程序那样即可上手。

![](/public/upload/machine/tensorflow_overview.png)


## 单机TensorFlow

[TensorFlow on Kubernetes的架构与实践](https://mp.weixin.qq.com/s/xsrRZVnPp-ogj59ZCGqqsQ)

![](/public/upload/machine/single_tensorflow.jpeg)

### 概念

Tensorflow底层最核心的概念是张量，计算图以及自动微分。

1. Tensor
2. Variable，特殊的张量， 维护特定节点的状态。**tf.Variable 方法是操作**，返回时是变量。与Tensor 不同在于
  1. 普通Tensor 的生命周期通常随依赖的计算完成而结束，内存也随即释放。
  2. 变量常驻内存， 在每一步训练时不断更新其值，以实现模型参数的更新。

[动态图与静态图的浅显理解](https://mp.weixin.qq.com/s/7IyaIij9sE7tm7wmL0ti3g) PS： 笔者个人的学习路径是先 pytorch 后tensorflow

```python
x = tf.constant("hello")
y = tf.constant("world")
z = tf.strings.join([x,y],separator=" ")
print(z)
# output
# Tensor("StringJoin:0", shape=(), dtype=string)
```
可以看到，在tensorflow1.0 静态图场景下，z 输出为空。`z = tf.strings.join([x,y],separator=" ")` **没有真正运行**（我们发明一个叫tensorflow的deep learning dsl，并且提供python api，让用户在python中通过元编程编写tensorflow代码），只有运行`session.run(z)` z 才会真正有值。在TensorFlow1.0时代，采用的是静态计算图，需要先使用TensorFlow的各种算子创建计算图，然后再开启一个会话Session，显式执行计算图。而在TensorFlow2.0时代，采用的是动态计算图，即每使用一个算子后，该算子会被动态加入到隐含的默认计算图中立即执行得到结果，不需要使用Session了，像原始的Python语法一样自然。

从数据流图中取数据、给数据流图加载数据等 一定要通过session.run 的方式执行，没有在session 中执行之前，整个数据流图只是一个壳。 **python 是一门语言，而tensorflow 的python api 并不是python** ，而是一种特定语言，也因此tensorflow python中不要使用逻辑控制，“python代码”都会send 到一个执行引擎来跑。

数据流图上节点的执行顺序的实现参考了拓扑排序的设计思想
1. 以节点名称作为关键字，入度作为值，创建一张散列表，并将此数据流图上的所有节点放入散列表中。
2. 为此数据流图创建一个可执行节点队列，将散列表中入度为0的节点加入到该队列，并从散列表中删除这些节点
3. 依次执行该队列中的每一个节点，执行成功后将此节点输出指向的节点的入度值减1，更新散列表中对应节点的入度值
4. 重复2和3，知道可执行节点队列变为空。


### 会话

Session 提供求解张量和执行操作的运行环境，它是发送计算任务的客户端，所有计算任务都由它分发到其连接的执行引擎（进程内引擎）完成。

```python
# 创建会话 target =会话连接的执行引擎（默认是进程内那个），graph= 会话加载的数据流图，config= 会话启动时的配置项
sess = tf.session(target=...,grah=...,config=...)
# 估算张量或执行操作。 Tensor.eval 和 Operation.run 底层都是 sess.run
sess.run(...)
# 关闭会话
sess.close()
```

```python
import tensorflow as tf
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
sess = tf.Session()
print(sess.run(c))
```

### demo

```python
import tensorflow as tf
X = tf.placeholder(...)
Y_ = tf.placeholder(...)
w = tf.Variable(...)
b = tf.Variable(...)
Y = tf.matmul(X,w) + b
# 使用交叉熵作为损失值
loss = tf.reduce_mean(...)
# 创建梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 定义单步训练操作
train_op = optimizer.minimize(loss,...)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for step in xrange(max_train_steps):
      sess.run(train_op,...)
```

当我们调用 `sess.run(train_op)` 语句执行训练操作时，程序内部首先提取单步训练操作依赖的所有前置操作，这些操作的节点共同组成一幅子图。然后程序将子图中的计算节点、存储节点和数据节点按照各自的执行设备分类（可以在创建节点时指定执行该节点的设备），相同设备上的节点组成了一幅局部图。每个设备上的局部图在实际执行时，根据节点间的依赖关系将各个节点有序的加载到设备上执行。

![](/public/upload/machine/tensorflow_graph.png)

## 分布式

分布式TensorFlow入门教程 https://zhuanlan.zhihu.com/p/35083779

[炼丹师的工程修养之四： TensorFlow的分布式训练和K8S](https://zhuanlan.zhihu.com/p/56699786)无论是TensorFlow还是其他的几种机器学习框架，分布式训练的基本原理是相同的。大致可以从以下五个不同的角度来分类。

1. 并行模式， 对于机器学习的训练任务，原来的“大”问题主要表现在两个方面。一是模型太大，我们需要把模型“拆”成多个小模型分布到不同的Worker机器上；二是数据太大，我们需要把数据“拆”成多个更小的数据分布到不同Worker上。
    1. 模型并行：深度学习模型一般包含很多层，如果要采用模型并行策略，一般需要将不同的层运行在不同的设备上，但是实际上层与层之间的运行是存在约束的：前向运算时，后面的层需要等待前面层的输出作为输入，而在反向传播时，前面的层又要受限于后面层的计算结果。所以除非模型本身很大，一般不会采用模型并行，因为模型层与层之间存在串行逻辑。但是如果模型本身存在一些可以并行的单元，那么也是可以利用模型并行来提升训练速度，比如GoogLeNet的Inception模块。
    2. 数据并行（主要方案）：因为训练费时的一个重要原因是训练数据量很大。数据并行就是在很多设备上放置相同的模型，并且各个设备采用不同的训练样本对模型训练。训练深度学习模型常采用的是batch SGD方法，采用数据并行，可以每个设备都训练不同的batch，然后收集这些梯度用于模型参数更新。所有worker共享ps 上的模型参数，并按照相同拓扑结构的数据流图进行计算。
2. 架构模式，通过模型并行或数据并行解决了“大问题”的可行性，接下来考虑“正确性”。以数据并行为例，当集群中的每台机器只看到1/N的数据的时候，我们需要一种机制在多个机器之间同步信息（梯度），来保证分布式训练的效果与非分布式是一致的（N * 1/N == N）。相对成熟的做法主要有基于参数服务器（ParameterServer）和基于规约（Reduce）两种模式。Tensorflow 既有 PS 模式又有对等模式，PyTorch 以支持对等模式为主，而 MXNET 以支持 KVStore 和 PS-Lite 的 PS 模式为主。
3. 同步范式， 在梯度同步时还要考虑“木桶”效应，即集群中的某些Worker比其他的更慢的时候，导致计算快的Worker需要等待慢的Worker，整个集群的速度上限受限于最慢机器的速度。因此梯度的更新一般有**同步(Sync)、异步(Async)和混合**三种范式。
4. 物理架构，这里主要指基于GPU的部署架构，基本上分为两种：单机多卡和多机多卡
5. 通信技术，要讨论分布式条件下多进程、多Worker间如何通信，常见的技术有MPI，NCCL，GRPC，RDMA等


![](/public/upload/machine/distribute_tensorflow.png)


PS：训练是可以拆的，既然可以拆，一个机器多个卡，跟跑在多个机器上就区别不大了，如同把cluster里面的设备当成本机设备一样使用。

```py
// with tf.device 语句来指定某一行代码作用域对应的目标设备
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...
```

ps worker模型，  一般来说，基于TensorFlow 库写一个机器学习任务（read dataset，定义层， 确定loss）执行即可。

TensorFlow 没有提供一次性启动整个集群的解决方案，所以用户需要在每台机器上逐个手动启动一个集群的所有ps 和worker 任务。为了能够以同一行代码启动不同的任务，我们需要将所有worker任务的主机名和端口、 所有ps任务的主机名和端口、当前任务的作业名称以及任务编号这4个集群配置项参数化。通过输入不同的命令行参数组合，用户就可以使用同一份代码启动每一个任务。

```sh
// 在在参数服务器上执行
python mnist_dist_test_k8s.py --ps_hosts=10.244.2.140:2222 --worker_hosts=10.244.1.134:2222,10.244.2.141:2222 --job_name="ps" --task_index=0
// 在第一个worker节点上执行
python mnist_dist_test_k8s.py --ps_hosts=10.244.2.140:2222 --worker_hosts=10.244.1.134:2222,10.244.2.141:2222 --job_name="worker" --task_index=0
// 在第二个worker节点上执行
python mnist_dist_test_k8s.py --ps_hosts=10.244.2.140:2222 --worker_hosts=10.244.1.134:2222,10.244.2.141:2222 --job_name="worker" --task_index=1
```


