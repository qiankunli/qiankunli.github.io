---

layout: post
title: tensorflow
category: 架构
tags: MachineLearning
keywords:  tensorflow

---

## 简介

* TOC
{:toc}

TensorFlow 使用库模式（不是框架模式），工作形态是由用户编写主程序代码，调用python或其它语言函数库提供的接口实现计算逻辑。用户部署和使用TensorFlow 时，不需要启动专门的守护进程，也不需要调用特殊启动工具

## 单机TensorFlow

[TensorFlow on Kubernetes的架构与实践](https://mp.weixin.qq.com/s/xsrRZVnPp-ogj59ZCGqqsQ)

![](/public/upload/machine/single_tensorflow.jpeg)

## 分布式

分布式TensorFlow入门教程 https://zhuanlan.zhihu.com/p/35083779

[炼丹师的工程修养之四： TensorFlow的分布式训练和K8S](https://zhuanlan.zhihu.com/p/56699786)无论是TensorFlow还是其他的几种机器学习框架，分布式训练的基本原理是相同的。大致可以从以下五个不同的角度来分类。

1. 并行模式， 对于机器学习的训练任务，原来的“大”问题主要表现在两个方面。一是模型太大，我们需要把模型“拆”成多个小模型分布到不同的Worker机器上；二是数据太大，我们需要把数据“拆”成多个更小的数据分布到不同Worker上。
    1. 模型并行：深度学习模型一般包含很多层，如果要采用模型并行策略，一般需要将不同的层运行在不同的设备上，但是实际上层与层之间的运行是存在约束的：前向运算时，后面的层需要等待前面层的输出作为输入，而在反向传播时，前面的层又要受限于后面层的计算结果。所以除非模型本身很大，一般不会采用模型并行，因为模型层与层之间存在串行逻辑。但是如果模型本身存在一些可以并行的单元，那么也是可以利用模型并行来提升训练速度，比如GoogLeNet的Inception模块。
    2. 数据并行（主要方案）：因为训练费时的一个重要原因是训练数据量很大。数据并行就是在很多设备上放置相同的模型，并且各个设备采用不同的训练样本对模型训练。训练深度学习模型常采用的是batch SGD方法，采用数据并行，可以每个设备都训练不同的batch，然后收集这些梯度用于模型参数更新。所有worker共享ps 上的模型参数，并按照相同拓扑结构的数据流图进行计算。
2. 架构模式，通过模型并行或数据并行解决了“大问题”的可行性，接下来考虑“正确性”。以数据并行为例，当集群中的每台机器只看到1/N的数据的时候，我们需要一种机制在多个机器之间同步信息（梯度），来保证分布式训练的效果与非分布式是一致的（N * 1/N == N）。相对成熟的做法主要有基于参数服务器（ParameterServer）和基于规约（Reduce）两种模式。
3. 同步范式， 在梯度同步时还要考虑“木桶”效应，即集群中的某些Worker比其他的更慢的时候，导致计算快的Worker需要等待慢的Worker，整个集群的速度上限受限于最慢机器的速度。因此梯度的更新一般有同步(Sync)、异步(Async)和混合三种范式。
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

TensorFlow 没有提供一次性启动整个集群的解决方案，所以用户需要在每台机器上逐个手动启动一个集群的所有ps 和worker 任务。

```sh
// 在在参数服务器上执行
python mnist_dist_test_k8s.py --ps_hosts=10.244.2.140:2222 --worker_hosts=10.244.1.134:2222,10.244.2.141:2222 --job_name="ps" --task_index=0
// 在第一个worker节点上执行
python mnist_dist_test_k8s.py --ps_hosts=10.244.2.140:2222 --worker_hosts=10.244.1.134:2222,10.244.2.141:2222 --job_name="worker" --task_index=0
// 在第二个worker节点上执行
python mnist_dist_test_k8s.py --ps_hosts=10.244.2.140:2222 --worker_hosts=10.244.1.134:2222,10.244.2.141:2222 --job_name="worker" --task_index=1
```