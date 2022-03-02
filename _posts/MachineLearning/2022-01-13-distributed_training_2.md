---

layout: post
title: 分布式训练——数据并行
category: 架构
tags: MachineLearning
keywords:  distributed training

---

## 简介

* TOC
{:toc}

[深度学习分布式训练框架——基础知识](https://mp.weixin.qq.com/s/djGvx3fNJfKCXmjwTfJ-CA)
1. 中心化分布式，存在一个中心节点，它的作用是汇总并分发其他计算节点的计算结果，更进一步，中心节点可以采用同步更新策略（Synchronous updating），也可以采用异步更新策略（Asynchronous updating）
2. 去中心化分布式

参数服务器适合的是高纬稀疏模型训练，它利用的是维度稀疏的特点，每次 pull or push 只更新有效的值。但是深度学习模型是典型的dense场景，embedding做的就是把稀疏变成稠密。所以这种 pull or push 的不太适合。而网络通信上更优化的 all-reduce 适合中等规模的深度学习。又比如由于推荐搜索领域模型的 Embedding 层规模庞大以及训练数据样本长度不固定等原因，导致容易出现显存不足和卡间同步时间耗费等问题，所以 all-reduce 架构很少被用于搜索推荐领域。

## 单机模式

[Paddle计算图拆分与优化](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/performance/graph.html)
深度学习网络中有两个基本元素：Operator（比如加减/FC/Embedding查表） 和 Variable（分为稠密参数和稀疏参数）
单机的计算图执行流程：

1. 计算图拿到参数的值(Var)之后，会首先执行前向OP(FF OP)，OP可能会执行在不同的设备上，如CPU/GPU/Kunlun 或其他AI芯片，我们用XPU代指。
2. 前向OP计算完成后，得到loss，会继续计算反向OP(BP OP)，得到各个参数的梯度(Var_Grad)
3. 指定SGD/Adam等优化器，利用参数的梯度（Var_Grad）更新原始参数（Var）
4. 重复以上流程，迭代参数的更新，实现深度学习网络的训练

![](/public/upload/machine/paddle_host.png)

1. Worker(Trainer)在计算得到参数的梯度(Var_Grad)后，会通过RPC发送给PServer
2. PServer监听通信端口，将收到的不同参数分别通过不同的Oprimizer OP完成更新。 PS： **参数更新放在ps的问题是** 添加新的优化算法必须修改 PS 的实现
3. Worker在下一个迭代时，请求PServer上最新的参数
4. 重复以上流程，迭代参数的更新，实现分布式参数服务器的训练


## ps-worker

[Paddle计算图拆分与优化](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/performance/graph.html)参数服务器模式下，所有参数的全局真值都被分片保存在各个Pserver上，**PServer执行Optimizer OP**（另一种选择是 ps 只是汇总梯度，weight还是worker 更新的），执行参数的更新。以PS-CPU异步模式的计算图介绍PServer的计算图

![](/public/upload/machine/paddle_ps.png)

进程内通信主要实现内存数据的跨设备复制（本地CPU和GPU内存之间、多个GPU内存之间）， 进程间通信主要实现数据流和控制流基于网络协议的传输。Tensorflow 的运行时框架会根据应用代码中的集群和设备配置，在数据流图构建时自动插入通信操作，并在数据流图执行过程中动态调度这些操作。

![](/public/upload/machine/ps_worker_communication.png)

1. 对于逻辑上具有数据传输语义，但物理上通信的源和目标位于同一设备、同一地址空间的进程内通信，只传输数据指针，不传输数据本身。
2. 对于cpu 和gpu 内存间的设备通信，本质是跨设备内存的数据复制，尽量使用底层设备接口实现（比如DMA机制），而不是用面向应用层的API
3. 涉及GPU的内存传输一般需要经过CPU 内存中转，除非使用支持RDMA 协议的高性能网络。
4. 尽量做到 计算与通信过程重叠，很多通信函数采用异步模式及多线程并发执行

## all-reduce

[Ring AllReduce简介](https://mp.weixin.qq.com/s/K8l7H2zCUr9sGzYehizFMA) 各种配图都比较详细了。

## 《用python实现深度学习框架》

### ps

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

### AllReduce
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

![](/public/upload/machine/tf_ps.png)

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

