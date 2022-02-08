---

layout: post
title: tensorflow原理
category: 架构
tags: MachineLearning
keywords: tensorflow

---

## 简介

![](/public/upload/machine/tensorflow_graph.png)

神经网络在 视觉上 是一层一层的，表达式上是张量计算，执行上是数据流图。

## 数据流图的执行

《深入理解Tensorflow》数据流图计算粗略的分为 应用程序逻辑、 会话生命周期和算法核函数执行 这3个层次
1. 在应用程序逻辑中，用户使用Python 等应用层API 及高层抽象编写算法 模型，无需关心图切分、进程间通信等底层实现逻辑。算法涉及的计算逻辑和输入数据**绑定到图抽象中**，计算迭代控制语义体现在会话运行前后（即session.run）的控制代码上。
2. 在会话生命周期层次，单机会话与分布式会话具有不同的设计。 
  1. 单机会话采用相对简单的会话层次和图封装结构，它将图切分、优化之后，把操作节点和张量数据提交给底层执行器
  2. 分布式会话分为 client、master和worker 三层组件，它们对计算任务进行分解和分发，并通过添加通信操作 来确保计算逻辑的完整性。
3. 在算法核函数执行层次， 执行器抽象将会话传入的核函数加载到各个计算设备上有序执行。为充分利用多核硬件的并发计算能力，这一层次提供线程池调度机制；为实现众多并发操作的异步执行和分布式协同， 这一层次引入了通信会合点机制。

![](/public/upload/machine/run_graph.png)

应用层数据流图 表示为Python API 中的tensoflow.Graph 类，通信时表示为 基于Protocol Buffers 文件定义的GraphDef ，运行时的数据流图 表示为C++ 代码中的Graph 类及其成员类型。

### 数据流图的创建
1. 全图构造
2. 子图提取
3. 图切分，将一幅子图按照其 操作节点放置的设备，切分为若干局部数据流图的过程，切分生成的每幅局部图仅在一个设备上运行，通信操作节点（SendOp,RecvOp）被插入局部图，以确保执行子图的逻辑语义同切分之前一致。
4. 图优化

![](/public/upload/machine/create_graph.jpeg)

### 单机会话运行 和 分布式会话运行

读入数据流图的待执行子图以及必要的输入张量，依据图中定义的依赖关系，将每个节点对应的操作核函数有序的加载到各个计算设备上并发执行，并将计算结果作为后续节点的输入。会话的生命周期 最终完成子图上定义的所有计算语义，将输出结果以张量形式返回给创建会话的应用程序。ps：跟一个dag 流程编排软件的执行逻辑差不多。 

![](/public/upload/machine/execute_graph.png)

分布式会话运行，ps-worker相比单机来说 除了图按进程切分为局部图 和分到到worker 之外，worker 每次执行完子图之后会执行一个回调，在回调中进行grpc 通信（张量传输等），针对grpc 通信效率低的问题 又引入RDMA 等机制。

### 操作节点执行
操作节点执行 过程本质是 节点对应的核函数的执行过程。会话运行时，ExecutorImpl::Initialize 会对数据流图上每个操作节点 调用create_kernel 函数，这时创建的 核函数对象 是对应 操作在特定设备上的特化版本。

## 梯度计算

Tensorflow的底层结构是由张量组成的计算图。计算图就是底层的编程系统，每一个计算都是图中的一个节点，计算之间的依赖关系则用节点之间的边来表示。计算图构成了前向/反向传播的结构基础。给定一个计算图, TensorFlow 使用自动微分 (反向传播) 来进行梯度运算。tf.train.Optimizer允许我们通过minimize()函数自动进行权值更新，此时`tf.train.Optimizer.minimize()`做了两件事：

1. 计算梯度。即调用`compute_gradients (loss, var_list …)` 计算loss对指定val_list的梯度，返回元组列表 `list(zip(grads, var_list))`。
2. 用计算得到的梯度来更新对应权重。即调用 `apply_gradients(grads_and_vars, global_step=global_step, name=None)` 将 `compute_gradients (loss, var_list …)` 的返回值作为输入对权重变量进行更新；
将minimize()分成两个步骤的原因是：可以在某种情况下对梯度进行修正，防止梯度消失或者梯度爆炸。

## 自定义算子

对于 TensorFlow，可以自定义 Operation，即如果现有的库没有涵盖你想要的操作, 你可以自己定制一个。为了使定制的 Op 能够兼容原有的库，你必须做以下工作:

1. 在一个 C++ 文件中注册新 Op. Op 的注册与实现是相互独立的. 在其注册时描述了 Op 该如何执行. 例如, 注册 Op 时定义了 Op 的名字, 并指定了它的输入和输出.
2. 使用 C++ 实现 Op. 每一个实现称之为一个 "kernel", 可以存在多个 kernel, 以适配不同的架构 (CPU, GPU 等)或不同的输入/输出类型.
3. 创建一个 Python 包装器（wrapper）. 这个包装器是创建 Op 的公开 API. 当注册 Op 时, 会自动生成一个默认 默认的包装器. 既可以直接使用默认包装器, 也可以添加一个新的包装器.
4. (可选) 写一个函数计算 Op 的梯度.
5. (可选) 写一个函数, 描述 Op 的输入和输出 shape. 该函数能够允许从 Op 推断 shape.
6. 测试 Op, 通常使用 Pyhton。如果你定义了梯度，你可以使用Python的GradientChecker来测试它。

示例参考 [TensorFlow 增加自定义运算符](https://mp.weixin.qq.com/s/G7BAWaPL5Lh3_q5EplNJBQ) c++ 部分编译完成后得到一个so 文件
```python
import tensorflow as tf
zero_out_op = tf.load_op_library('zero_out.so')
with tf.Session():
  print(zero_out_op.zero_out([1,2,3,4,5])).eval()
```