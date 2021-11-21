---

layout: post
title: pytorch分布式训练
category: 架构
tags: MachineLearning
keywords:  pytorch distributed elastic 弹性

---

## 简介

* TOC
{:toc}

## 整体思路

![](/public/upload/machine/data_parallel_train.png)

[DDP系列第一篇：入门教程](https://zhuanlan.zhihu.com/p/178402798)多机多卡 
1. 分布式训练的几个选择：模型并行 vs 数据并行，PS 模型 vs Ring-Allreduce ，pytorch 单机多卡（DP）用PS 模型，多机多卡（DDP）用Ring-Allreduce
2. 假设 有两台机子，每台8张显卡，那就是2x8=16个进程，并行数是16。DDP 推荐每个进程一张卡，在16张显卡，16的并行数下，DDP会同时启动16个进程。rank表示当前进程的序号（0~16），用于进程间通讯。local_rank 表示每台机子上的进程的序号（0~7），**也用来指定机器上的gpu 序号**。用起来 就是一行代码，`model = DDP(model, device_ids=[local_rank], output_device=local_rank)`，**后续的模型关于前向传播、后向传播的用法，和单机单卡完全一致**。[DDP系列第二篇：实现原理与源代码解析](https://zhuanlan.zhihu.com/p/187610959)
3. 每个进程跑的是同一份代码，进程从外界（比如环境变量）获取 rank/master_addr/master_port 等参数，rank = 0 的进程为 master 进程

## 单机单卡

[Pytorch分布式训练](https://mp.weixin.qq.com/s/G-uLl3HXzFJOW03nA7etig)

单机单卡训练步骤：定义网络，定义dataloader，定义loss和optimizer，开训

```python
BATCH_SIZE = 256
EPOCHS = 5
if __name__ == "__main__":
    # 1. define network
    device = "cuda"
    net = torchvision.models.resnet18(num_classes=10)
    net = net.to(device=device)  
    # 2. define dataloader
    trainset = torchvision.datasets.CIFAR10(...)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )
    print("            =======  Training  ======= \n")
    # 4. start to train
    net.train()
    for ep in range(1, EPOCHS + 1):                 # 控制在全部数据上训练的次数
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)      # 获得输出结果和样本值的损失函数
            optimizer.zero_grad()
            loss.backward()                         # 根据loss 进行反向梯度传播
            optimizer.step()                        # 用计算的梯度去做优化
```


## 单机多卡

如果是单机多卡，定义网络时加入 `net = nn.DataParallel(net)`

![](/public/upload/machine/data_parallel.png)

[ PyTorch 分布式(2) ----- DataParallel(上)](https://mp.weixin.qq.com/s/cGfKl6yydc5Xmd_Ok3mnkA) 缺点很多。

## 分布式训练

[ PyTorch 分布式(5) ------ DistributedDataParallel 总述&如何使用](https://mp.weixin.qq.com/s/WdLpHfWLRvDLLxeanFduxA)

![](/public/upload/machine/data_distributed_parallel.png)

GPU之间只传递梯度
1. 加载模型阶段。每个GPU都拥有模型的一个副本，所以不需要拷贝模型。rank为0的进程会将网络初始化参数broadcast到其它每个进程中，确保每个进程中的模型都拥有一样的初始化值。
2. 加载数据阶段。**DDP 不需要广播数据，而是使用多进程并行加载数据**。在 host 之上，每个worker进程都会把自己负责的数据从硬盘加载到 page-locked memory。DistributedSampler 保证每个进程加载到的数据是彼此不重叠的。
3. 前向传播阶段。在每个GPU之上运行前向传播，计算输出。每个GPU都执行同样的训练，所以不需要有主 GPU。
4. 计算损失。在每个GPU之上计算损失。
5. 反向传播阶段。运行后向传播来计算梯度，在计算梯度同时也对梯度执行all-reduce操作。每一层的梯度不依赖于前一层，所以梯度的All-Reduce和后向过程同时计算，以进一步缓解网络瓶颈。在后向传播的最后，每个节点都得到了平均梯度，这样模型参数保持同步。[关于AllReduce](https://zhuanlan.zhihu.com/p/100012827)
6. 更新模型参数阶段。因为每个GPU都从完全相同的模型开始训练，并且梯度被all-reduced，因此每个GPU在反向传播结束时最终得到平均梯度的相同副本，所有GPU上的权重更新都相同，也就**不需要模型同步了**。注意，在每次迭代中，模型中的Buffers 需要从rank为0的进程广播到进程组的其它进程上。

### 模板代码

```python
import torch.distributed as dist
import torch.utils.data.distributed

parser = argparse.ArgumentParser(description='PyTorch distributed training on cifar-10')
parser.add_argument('--rank', default=0,help='rank of current process')
parser.add_argument('--word_size', default=2,help="word size")
parser.add_argument('--init_method', default='tcp://127.0.0.1:23456',help="init-method")
args = parser.parse_args()
...
dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.word_size)
...
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
## 数据并行需要进行数据切片
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=args.world_size,rank=rank)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
...
net = Net() 
net = Net().to(device) # device 代表到某个 gpu 或cpu 上
## 使用DistributedDataParallel 修饰模型
net = torch.nn.parallel.DistributedDataParallel(net)
```

[PyTorch分布式训练简明教程](https://mp.weixin.qq.com/s/0aSBHvscloEnPMRLyNjQsg)
1. 用dist.init_process_group初始化分布式环境
    1. 一般来说，nccl 用于 GPU 分布式训练，gloo 用于 CPU 进行分布式训练。
    2. 这个调用是阻塞的，必须等待所有进程来同步，如果任何一个进程出错，就会失败。
1. 数据侧，我们nn.utils.data.DistributedSampler来给各个进程切分数据，只需要在dataloader中使用这个sampler就好
    1. 使用 DDP 时，不再是从主 GPU 分发数据到其他 GPU 上，而是各 GPU 从自己的硬盘上读取属于自己的那份数据。
    1. 训练循环过程的每个epoch开始时调用train_sampler.set_epoch(epoch)，（主要是为了保证每个epoch的划分是不同的）其它的训练代码都保持不变。
1. 模型侧，我们只需要用DistributedDataParallel包装一下原来的model

pytorch 中的任何 net 都 是 `torch.nn.Module` 的子类，DistributedDataParallel 也是 `torch.nn.Module` 子类，任何 `torch.nn.Module` 子类 都可以覆盖 `__init__` 和  `forward`方法 ，DistributedDataParallel 可以从  net 拿到模型数据（以及 在哪个gpu 卡上运行） ，也可以 从指定或默认的 process_group 获取信息。最后在`__init__`  和 forward 中加入 同步梯度的逻辑，**完美的将 同步梯度的逻辑 隐藏了起来**。

### 实际例子

```python
BATCH_SIZE = 256
EPOCHS = 5
if __name__ == "__main__":
    # 0. set up distributed device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank) # 把模型放置到特定的GPU上
    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    # 1. define network
    net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    net = net.to(device)
    # DistributedDataParallel
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    # 2. define dataloader
    trainset = torchvision.datasets.CIFAR10(...)
    # DistributedSampler
    # we test single Machine with 2 GPUs so the [batch size] for each process is 256 / 2 = 128
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )
    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.01 * 2,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )
    if rank == 0:
        print("            =======  Training  ======= \n")
    # 4. start to train
    net.train()     # 标记为训练模式，推理时使用 net.eval()
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0
        # set sampler
        train_loader.sampler.set_epoch(ep)
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()       # 将梯度归零
            loss.backward()             # 反向传播计算得到每个参数的梯度值
            optimizer.step()            # 通过梯度下降执行一步参数更新，optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是tensor.backward()方法产生的。
            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()
            # 输出loss 和 正确率
    if rank == 0:
        print("\n            =======  Training Finished  ======= \n")
```


每条命令表示一个进程。若已开启的进程未达到 word_size 的数量，则所有进程会一直等待

### 进程间通信

如何发现对方（怎么知道大家是一伙儿的？），发现了对方之后如何定身份？谁是master 谁是worker？ 定了身份之后怎么协作？涉及到几个概念

worker 发现彼此
1. init_method，开始的前提：如何联系到其它机器上的进程。
    1. Shared file-system initialization：init_method='file:///mnt/nfs/sharedfile'
    1. dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',ank=args.rank, world_size=4)
    2. dist.init_process_group(backend, init_method='env://',ank=args.rank, world_size=4) 这个进程会自动读取环境变量 MASTER_ADDR/MASTER_PORT/WORLD_SIZE/RANK
4. store ，**elastic 机制之后支持的**，主要是给 elastic agent 使用的。

worker 交流梯度
1. processgroup。 进程组就是给每一个训练的 process 建立一个Communication thread。主线程（Computation thread）在前台进行训练，这个Communication thread 在后台做通信（交流梯度）。[PyTorch 分布式(7) ----- DistributedDataParallel 之进程组](https://mp.weixin.qq.com/s/XPzLC7nuXDkQKtmZgTFalQ)
    ![](/public/upload/machine/pytorch_process_group.png)
2. backend，如何协作？后端这个概念是一个**逻辑上**的概念。本质上后端是一种IPC通信机制。对于用户来说，就是采用那种方式来进行集合通信，从代码上看，就是走什么流程（一系列流程），以及后端使用 ProcessGroupMPI 还是  ProcessGroupGloo …..。各主机之间需要进行通信。因此，需要指定通信的协议架构等。torch.distributed 对其进行了封装。提供了分布式通信原语

    ![](/public/upload/machine/pytorch_distributed_backend.jpeg)

## 弹性分布式训练 

关键就是
1. 启动的时候，参与训练的 节点数不确定了。
2. 训练过程中，可能会有新的节点 或节点挂掉，要能容错，继续训练。

### 如何启动 train_script 

train_script 启动方式有以下几种
1. 直接 python 启动 需要以下参数（来自run.py 代码注释），比较重要的是local_rank,world_size,master_host,master_port, backend
    1. ``LOCAL_RANK`` -  The local rank.
    2. ``RANK`` -  The global rank.
    3. ``GROUP_RANK`` - The rank of the worker group. A number between 0 and ``max_nnodes``. When
    running a single worker group per node, this is the rank of the node.
    4. ``ROLE_RANK`` -  The rank of the worker across all the workers that have the same role. The role
    of the worker is specified in the ``WorkerSpec``.
    5. ``LOCAL_WORLD_SIZE`` - The local world size (e.g. number of workers running locally); equals to
    ``--nproc_per_node`` specified on ``torchrun``.
    6. ``WORLD_SIZE`` - The world size (total number of workers in the job).
    7. ``ROLE_WORLD_SIZE`` - The total number of workers that was launched with the same role specified
    in ``WorkerSpec``.
    8. ``MASTER_ADDR`` - The FQDN of the host that is running worker with rank 0; used to initialize
    the Torch Distributed backend.
    9. ``MASTER_PORT`` - The port on the ``MASTER_ADDR`` that can be used to host the C10d TCP store.
    10. ``TORCHELASTIC_RESTART_COUNT`` - The number of worker group restarts so far.
    11. ``TORCHELASTIC_MAX_RESTARTS`` - The configured maximum number of restarts.
    12. ``TORCHELASTIC_RUN_ID`` - Equal to the rendezvous ``run_id`` (e.g. unique job id).
2. 由 launch.py 启动，大部分参数直接 传给 train_script.py，只是根据 nnodes 和 nproc_per_node 算了一下 local_rank 和 world_size 传给train_script
3. 由 run.py 启动，**参数 都是给 rendezvous 用的**， 由rendezvous 协商出 train_script 需要的参数，由 run.py spawn worker 进程时传给worker/传给train_script。

后两种 仅仅是为启动 train_script 方便，该给 train_script 传的参数还是要传。PS：train_script 是算法写的，在train_script 内使用的库 具备分布式、弹性 能力之前，能力扩充 主要通过 加壳子的方式来解决。

```python
# 老的
python -m torch.distributed.launch 
    --nnodes=2
    --nproc_per_node=xx
    --master_addr=xx 
    --master_port=xx
# 新的
python -m torchelastic.distributed.run 
    --nnodes=1:4
    --nproc_per_node=xx
    --rdzv_id=JOB_ID
    --rdzv_backend=etcd
    --rdzv_endpoint=ETCD_HOST:ETCD_PORT
    TRAINING_SCRIPT.py (... train script args ...)
```

## train_script 的守护者elastic agent

以下内容来自  `python3.9/site-packages/torch/distributed/run.py` 注释

`python -m torch.distributed.run train_script.py ` 对于每一个node 有两个角色
1. run.py 负责启动 elastic agent
2. elastic agent 负责启动 train_script.py， 并给train_script.py 传递必要的参数（环境变量或参数形式，由脚本的获取方式决定）。

![](/public/upload/machine/torchelastic_diagram.jpeg)

run.py 启动命令，`python -m  torch.distributed.run [args] training_script [training_script_args]`，包含以下参数

关于workGroup 的布局
1. `--nnodes`
2. `--nproc_per_node`
关于集会机制
3. `--rdzv_backend`
4. `--rdzv_endpoint`
5. `--rdzv_id`
6. `--rdzv_conf`
7. `--standalone`   Standalone 模式是分布式模式的一种特例，它主要针对单机多 Worker 的方式提供了一些便利的设置，指定Standalone后 不再需要设置一些多余的参数如 rdzv_backend 和 rdzv_endpoint 等。
User-code launch related arguments.
8. `--max_restarts`
9. `--monitor_interval`     监听 worker 状态的间隔
10. `--start_method`        可选值 `["spawn", "fork", "forkserver"]`，默认值spawn
11. `--role`
以下为兼容 老版 launch.py 提供的参数

12. `--node_rank`
13. `--master_addr`
14. `--master_port`

### worker 如何发现？——rendezvous/集会机制

[Elastic Introduction](https://github.com/pytorch/elastic/blob/master/design/torchelastic/0.2.0/design_doc.md) 
[Elastic Agent 的设计：如何管理多个 worker 进程](https://mp.weixin.qq.com/s/hlOYLKSHFDZWN21AsUn6bg) 不同的 elastic agent 

Elastic Agent  通过 rendezvous 进行 worker 之间的相互发现，以便在不同的节点间确定 RANK。 需要一个类似配置中心的东西 etcd 或自带的c10d，对应有一个 Store 抽象（有EtcdStore 和 TcpStore），在此基础上封装了  RendezvousHandler 来管理 rendezvous（在etcd 上对应一个 带version 的path） 。PS： RendezvousHandler 和 Store 的协作很迷，在Store 基础上封装了 barrier 协调步调的函数，**代码上看 etcd 系列的会清晰一下**。
    ```python
    class RendezvousHandler(ABC):
        def get_backend(self) -> str
        def next_rendezvous(self,) -> Tuple[Store, rank, world_size]
        def is_closed(self) -> bool
        def set_closed(self)
        def num_nodes_waiting(self) -> int
        def get_run_id(self) -> str
        def shutdown(self) -> bool
    class Store(__pybind11_builtins.pybind11_object):
        def add(self, arg0, arg1)
        def compare_set(self, arg0, arg1, arg2)
        def delete_key(self, arg0)
        def get(self, arg0)
        def num_keys(self)
        def set(self, arg0, arg1)
        def set_timeout(self, arg0)
        def wait(self, *args, **kwargs)
    ```
当一个worker 挂了之后，其它所有的worker 也都会挂掉。<font color="red">存在疑问<font>： 一个worker 不管是自己运行失败、还是被外力杀死，其它worker 怎么知道的？
1. 可能是 rendezvous 缺了一个worker ，其它worker 交换不了梯度了，也就报错了，进而被各自的elastic agent 监听到。 
2. 可能是 elastic agent 通知了 注册中心 ，其它elastic agent 监听到了这个信息，进而陆续关停自己的worker。
    1. 被外力干死，此时 elastic agent 注册了 `signal.signal(signal.SIGTERM, _terminate_process_handler)`  之后 引起 `rdzv_handler.shutdown()`（可能变更了rendezvous state，etcd 和 c10d 处理方式不同）。
    2. worker 运行失败，elastic agent 周期性 _monitor_workers 发现 worker 失败，通知了注册中心/store，继而被其它node 的elastic agent 感知到。
但第二种猜测没有找到代码支持，并且etcd 和c10d 的逻辑也非常不一样。

When a worker process fails, the corresponding elastic agent managing it kills all the workers on that node, establishes rendezvous with the other agents and restarts workers with the new rendezvous information. However, when an agent exits with a non-zero error code, it is up to a higher-level orchestrator such as Kubernetes to restart the agent (which in turn will restart all the workers it is responsible for). The same recovery mechanism holds for node-level failures. An orchestrator such as Kubernetes will schedule a job such that a minimum replicas of the elastic agent are running and each agent will in turn orchestrate the user's training script. 当一个worker 挂掉时，Elastic agent 将干掉节点上的worker(属于同一个local work group) 并通知其它agent。**当Elastic agent 挂掉时， 则需要 更高层级比如k8s 来重启Elastic agent**。

![](/public/upload/machine/torchelastic_agent_diagram.jpeg)

新的processgroup 被k8s 拉起来之后，关键问题就是两个
1. worker 发现，有几个？ rank=0 是谁？ rendezvous 的技能。
2. checkpoint 加载，防止之前 训练的浪费了。

”注册中心“可以是c10d/etcd，使用不同的“注册中心”有不同的问题
1. c10d 运行在rank0 节点， 因此使用c10d时，非rank0 节点挂掉ok，rank0 节点挂掉会导致训练任务失败
2. 使用etcd时，非rank0 节点挂掉ok，rank0 节点挂掉后 其它节点会作为rank0节点，可能会有问题：有些框架喜欢在rank0 做一些特殊工作

## elastic agent 源码

elastic agent 的可扩展性非常好，在 1.9.0 版本中，一共有三个 Agent，分别是 ElasticAgent、SimpleElasticAgent 和 LocalElasticAgent。
其中 ElasticAgent 是一个 Abstract Class，SimpleElasticAgent 对其中的某些函数进行了实现（半成品），而 LocalElasticAgent 则实现了管理单机上所有 worker 进程的 elastic agent。

`python3.9/site-packages/torch/distributed/run.py` ==> ` main()` ==> `run(args)` ==>  `elastic_launch(config=config,entrypoint=cmd,)(*cmd_args)` ==> `__call__(*args)` ==> launch_agent ==> agent.run ==> SimpleElasticAgent._invoke_run

```python
# /python3.9/site-packages/torch/distributed/launcher/api.py
class elastic_launch:
    def __init__(self,config: LaunchConfig,entrypoint: Union[Callable, str, None],):
        self._config = config
        self._entrypoint = entrypoint
    def __call__(self, *args):
        return launch_agent(self._config, self._entrypoint, list(args))
def launch_agent(config: LaunchConfig,entrypoint: Union[Callable, str, None],args: List[Any],) -> Dict[int, Any]:
    ## 构造WorkerSpec  包括worker 启动参数等
    entrypoint_name = _get_entrypoint_name(entrypoint, args)
    rdzv_parameters = RendezvousParameters(
        backend=config.rdzv_backend,
        endpoint=config.rdzv_endpoint,
        min_nodes=..,max_nodes=..,..)
    rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_parameters)
    master_addr, master_port = _get_addr_and_port(rdzv_parameters)
    try:
        spec = WorkerSpec(
            local_world_size=config.nproc_per_node,
            entrypoint=entrypoint,...,
            master_addr=master_addr,master_port=master_port,)
        agent = LocalElasticAgent(spec=spec, start_method=config.start_method, log_dir=config.log_dir)
        result = agent.run()  # 启动agent
    except Exception:
        ...
    finally:
        rdzv_handler.shutdown() # 以 EtcdRendezvousHandler 为例，EtcdRendezvousHandler.shutdown 会在 etcd 上记录 本次rendezvous closed。
```

elastic agent 周期性 _monitor_workers ，判断worker SUCCEEDED/FAILED/HEALTHY，如果发现失败的 worker ，主动stop worker

```python
# python3.9/site-packages/torch/distributed/elastic/agent/server/api.py
class SimpleElasticAgent(ElasticAgent):
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        try:
            return self._invoke_run(role)
        except SignalException as e:
            ...
        finally:
            self._shutdown()
    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        spec = self._worker_group.spec
        self._initialize_workers(self._worker_group)    ## 启动worker
        while True:
            time.sleep(monitor_interval)    ## 每隔  monitor_interval 查看下 worker 的状态
            run_result = self._monitor_workers(self._worker_group)  #对workers 的运行状态进行监控。并且根据不同的状态进行不同的处理
            state = run_result.state
            self._worker_group.state = state
            if   state == WorkerState.SUCCEEDED: exit...
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}: _restart_workers/_stop_workers...
            elif state == WorkerState.HEALTHY: ...
            else: raise Exception(f"[{role}] Worker group in {state.name} state")
```
_initialize_workers 执行了大部分初始化的工作，其中包括为每个 worker 分配 RANK 等。

```python
class SimpleElasticAgent(ElasticAgent):
    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
    role = worker_group.spec.role
    self._rendezvous(worker_group)                  # 启动前先集会获取下数据
    log.info(f"[{role}] Starting worker group")
    worker_ids = self._start_workers(worker_group)  # 启动worker
    for local_rank, w_id in worker_ids.items():
        worker = worker_group.workers[local_rank]
        worker.id = w_id
    worker_group.state = WorkerState.HEALTHY
```
LocalElasticAgent 实现了 `_start_workers`，关键就是为 worker 准备环境 变量和参数
```python
# python3.9/site-packages/torch/distributed/elastic/agent/server/local_elastic_agent.py
class LocalElasticAgent(SimpleElasticAgent):
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        spec = worker_group.spec
        store = worker_group.store
        master_addr, master_port = super()._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts
        args: Dict[int, Tuple] = {}
        envs: Dict[int, Dict[str, str]] = {}
        ## 为worker 进程准备 环境变量 和启动参数
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),"RANK": str(worker.global_rank),"GROUP_RANK": str(worker_group.group_rank),
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),"WORLD_SIZE": str(worker.world_size),
                "MASTER_ADDR": master_addr,"MASTER_PORT": str(master_port),
                ...
            }
            envs[local_rank] = worker_env
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)
        self._pcontext = start_processes(
            entrypoint=spec.entrypoint,
            args=args,envs=envs,log_dir=attempt_log_dir,
            start_method=self._start_method,...)
        return self._pcontext.pids()
```
start_processes ==> PContext.start ==> `SubprocessContext._start` ==> `SubprocessHandler.__init__` ==> `SubprocessHandler._popen` ==> subprocess.Popen

## checkpoint

一般来说，用户可以使用torch.save和torch.load作为checkpoints，以便从检查点恢复训练。checkpoint 可以保存在共享文件系统上，由rank=0 负责保存，其它rank 进程只负责加载，此时要确保在保存完成之前所有进程都不会开始加载 （涉及到使用  `dist.barrier()`）。

有几个关键点
1. checkpoint 保存了什么。 取决于 load checkpoint 之后 是 恢复训练（还是老样本，从中断处开始） 还是增量训练（样本变了）
    1. model 数据
    2. 优化器数据，学习率等可能因为 错误率而变化
    3. epoch，当前训练到了第几轮
2. 如何封装checkpoint 
    1. 定一个class state , `save_checkpoint(state: State, filename: str)`  `load_checkpoint(checkpoint_file: str, ...) -> State`
    2. 直接 对保存的数据 save_checkpoint 和 load_checkpoint

[pytorch：模型的保存与加载](https://zhuanlan.zhihu.com/p/269143428) 保存时，有保存参数和保存模型的分别。

在支持弹性训练之前，save/load checkpoint 是一个可选工作，**支持弹性之后，save/load checkpoint 就是必须的了**。The application writer is responsible for loading and restarting from an existing checkpoint file is available. PyTorch Elastic Trainer  does not mandate how checkpoints are managed. 开发者要对  checkpoing 的保存、加载、加载时机负责。run.py 不关心checkpoint 的管理（大部分只有rank=0 才进行checkpoint 处理）。

## 其它

### 概念

来自run.py 代码注释

1. ``Node`` - A physical instance or a container; maps to the unit that the job manager works with.
2. ``Worker`` - A worker in the context of distributed training.
3. ``WorkerGroup`` - The set of workers that execute the same function (e.g. trainers).
4. ``LocalWorkerGroup`` - A subset of the workers in the worker group running on the same node.
5. ``RANK`` - The rank of the worker within a worker group.
6. ``WORLD_SIZE`` - The total number of workers in a worker group.
7. ``LOCAL_RANK`` - The rank of the worker within a local worker group.
8. ``LOCAL_WORLD_SIZE`` - The size of the local worker group.
9. ``rdzv_id`` - A user-defined id that uniquely identifies the worker group for a job. This id is
   used by each node to join as a member of a particular worker group.
9. ``rdzv_backend`` - The backend of the rendezvous (e.g. ``c10d``). This is typically a strongly
   consistent key-value store.
10. ``rdzv_endpoint`` - The rendezvous backend endpoint; usually in form ``<host>:<port>``.
A ``Node`` runs ``LOCAL_WORLD_SIZE`` workers which comprise a ``LocalWorkerGroup``. The union of
all ``LocalWorkerGroups`` in the nodes in the job comprise the ``WorkerGroup``.