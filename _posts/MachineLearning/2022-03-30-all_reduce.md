---

layout: post
title: 数据并行——allreduce
category: 架构
tags: MachineLearning
keywords: allreduce

---

## 简介

* TOC
{:toc}

## 基本原理

[Ring AllReduce简介](https://mp.weixin.qq.com/s/K8l7H2zCUr9sGzYehizFMA) 各种配图都比较详细了。

## 《用python实现深度学习框架》 api示例

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


## horovod/allreduce 

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

