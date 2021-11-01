---

layout: post
title: pytorch 弹性分布式训练
category: 架构
tags: MachineLearning
keywords:  pytorch distributed elastic 弹性

---

## 简介

* TOC
{:toc}

## 分布式训练

多gpu 训练

![](/public/upload/machine/multi_gpu.png)

[分布式训练](https://time.geekbang.org/opencourse/videodetail/100077201-407557)

![](/public/upload/machine/distribute_gpu.png)

1. 黄色表示 参数，绿色表示梯度
2. 分布式**同步**数据并行 是多GPU 数据并行在多机器上的扩展。异步并行是每个机器做自己的批量更新，不用同步。
3. 网络通信是瓶颈。从分布式文件系统上读数据；参数同步。
4. 大batch 可能会导致收敛变慢

[DDP系列第一篇：入门教程](https://zhuanlan.zhihu.com/p/178402798)多机多卡 
1. 分布式训练的几个选择：模型并行 vs 数据并行，PS 模型 vs Ring-Allreduce ，pytorch 单机多卡（DP）用PS 模型，多机多卡（DDP）用Ring-Allreduce
2. 假设 有两台机子，每台8张显卡，那就是2x8=16个进程，并行数是16。DDP 推荐每个进程一张卡，在16张显卡，16的并行数下，DDP会同时启动16个进程。rank表示当前进程的序号（0~16），用于进程间通讯。local_rank 表示每台机子上的进程的序号（0~7），**也用来指定机器上的gpu 序号**。用起来 就是一行代码，`model = DDP(model, device_ids=[local_rank], output_device=local_rank)`，**后续的模型关于前向传播、后向传播的用法，和单机单卡完全一致**。[DDP系列第二篇：实现原理与源代码解析](https://zhuanlan.zhihu.com/p/187610959)
3. 每个进程跑的是同一份代码，进程从外界（比如环境变量）获取 rank/master_addr/master_port 等参数，rank = 0 的进程为 master 进程

## 实现原理 run.py

以下内容来自  `python3.9/site-packages/torch/distributed/run.py` 注释

`python -m torch.distributed.run train_script.py ` 对于每一个node 有两个角色
1. run.py 负责启动 elastic agent
2. elastic agent 负责启动 train_script.py， 并给train_script.py 传递必要的参数（环境变量或参数形式，由脚本的获取方式决定）。

### 概念

![](/public/upload/machine/torchelastic_diagram.jpeg)

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

### run.py 启动参数

`python -m  torch.distributed.run [args] training_script [training_script_args]`

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

### train_script 启动参数或环境变量

如果单独启动 `python train_script.py` 需要传以下变量， 使用run.py 启动train_script之后，这些信息会由 run.py spawn worker 进程时传给worker。The following environment variables are made available to you in your script:

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

### rendezvous/集会机制

[Elastic Introduction](https://github.com/pytorch/elastic/blob/master/design/torchelastic/0.2.0/design_doc.md) 
[Elastic Agent 的设计：如何管理多个 worker 进程](https://mp.weixin.qq.com/s/hlOYLKSHFDZWN21AsUn6bg)rendezvous核心工作是： **如何在不同的节点间确定 RANK**

**Membership Changes:**
1. Node departure (scale-down): The agent is notified of the departure, all existing workers are
   stopped, a new ``WorkerGroup`` is formed, and all workers are started with a new ``RANK`` and
   ``WORLD_SIZE``.
2. Node arrival (scale-up): The new node is admitted to the job, all existing workers are stopped,
   a new ``WorkerGroup`` is formed, and all workers are started with a new ``RANK`` and
   ``WORLD_SIZE``.

The application writer is responsible for loading and restarting from an existing checkpoint file is available. PyTorch Elastic Trainer  does not mandate how checkpoints are managed. 开发者要对  checkpoing 的保存、加载、加载时机负责。run.py 不关心checkpoint 的管理。

When a worker process fails, the corresponding elastic agent managing it kills all the workers on that node, establishes rendezvous with the other agents and restarts workers with the new rendezvous information. However, when an agent exits with a non-zero error code, it is up to a higher-level orchestrator such as Kubernetes to restart the agent (which in turn will restart all the workers it is responsible for). The same recovery mechanism holds for node-level failures. An orchestrator such as Kubernetes will schedule a job such that a minimum replicas of the elastic agent are running and each agent will in turn orchestrate the user's training script. 当一个worker 挂掉时，Elastic agent 将干掉节点上的worker(属于同一个local work group) 并通知其它agent。当Elastic agent 挂掉时， 则需要 更高层级比如k8s 来重启Elastic agent 。

![](/public/upload/machine/torchelastic_agent_diagram.jpeg)

### train_script 改造

train_script 不需要 依赖第三方库，仅由run.py 启动即可。 
```python
python -m torchelastic.distributed.run --nproc_per_node=NUM_GPUS_ON_NODE
            --nnodes=1:4
            --rdzv_id=JOB_ID
            --rdzv_backend=etcd
            --rdzv_endpoint=ETCD_HOST:ETCD_PORT
            TRAINING_SCRIPT.py (... train script args ...)
```

```python
def main():
     args = parse_args(sys.argv[1:])
     state = load_checkpoint(args.checkpoint_path)
     initialize(state)
     # torch.distributed.run ensure that this will work
     # by exporting all the env vars needed to initialize the process group
     torch.distributed.init_process_group(backend=args.backend)
     for i in range(state.epoch, state.total_num_epochs)
          for batch in iter(state.dataset)
              train(batch, state.model)
          state.epoch += 1
          save_checkpoint(state)
```
主要有两点

1. 内部要注意 initialize the process group. All the parameters for initializing the group (the world size, the numerical rank, the master address and port) are passed in as environment variables by the parent elastic agent.
2. 保存和加载checkpoint，这也是pytorch Elastic 设计文档中提到要优化的地方。

## elastic agent 源码

```python
# /python3.9/site-packages/torch/distributed/launcher/api.py
class elastic_launch:
    def __init__(self,config: LaunchConfig,entrypoint: Union[Callable, str, None],):
        self._config = config
        self._entrypoint = entrypoint
    def __call__(self, *args):
        return launch_agent(self._config, self._entrypoint, list(args))
def launch_agent(config: LaunchConfig,entrypoint: Union[Callable, str, None],args: List[Any],) -> Dict[int, Any]:
    entrypoint_name = _get_entrypoint_name(entrypoint, args)
    rdzv_parameters = RendezvousParameters(
        backend=config.rdzv_backend,
        endpoint=config.rdzv_endpoint,
        run_id=config.run_id,
        min_nodes=config.min_nodes,
        max_nodes=config.max_nodes,
        **config.rdzv_configs,
    )
    rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_parameters)
    master_addr, master_port = _get_addr_and_port(rdzv_parameters)
    try:
        spec = WorkerSpec(
            role=config.role,
            local_world_size=config.nproc_per_node,
            entrypoint=entrypoint,
            args=tuple(args),
            rdzv_handler=rdzv_handler,
            max_restarts=config.max_restarts,
            monitor_interval=config.monitor_interval,
            redirects=config.redirects,
            tee=config.tee,
            master_addr=master_addr,
            master_port=master_port,
        )
        agent = LocalElasticAgent(spec=spec, start_method=config.start_method, log_dir=config.log_dir)
        result = agent.run()
    except Exception:
        ...
    finally:
        rdzv_handler.shutdown()
```
elastic agent 的可扩展性非常好，在 1.9.0 版本中，一共有三个 Agent，分别是 ElasticAgent、SimpleElasticAgent 和 LocalElasticAgent。
其中 ElasticAgent 是一个 Abstract Class，SimpleElasticAgent 对其中的某些函数进行了实现（半成品），而 LocalElasticAgent 则实现了管理单机上所有 worker 进程的 elastic agent。
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
        self._initialize_workers(self._worker_group)
        while True:
            time.sleep(monitor_interval)    ## 每隔  monitor_interval 查看下 worker 的状态
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state
            if state == WorkerState.SUCCEEDED:
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                if self._remaining_restarts > 0:
                    self._remaining_restarts -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    self._exit_barrier()
                    return run_result
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                group_rank = self._worker_group.group_rank
                if num_nodes_waiting > 0:
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f"[{role}] Worker group in {state.name} state")
    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        role = worker_group.spec.role
        self._rendezvous(worker_group)  # 启动前先集会获取下数据
        log.info(f"[{role}] Starting worker group")
        worker_ids = self._start_workers(worker_group)
        for local_rank, w_id in worker_ids.items():
            worker = worker_group.workers[local_rank]
            worker.id = w_id
        worker_group.state = WorkerState.HEALTHY
# python3.9/site-packages/torch/distributed/elastic/agent/server/local_elastic_agent.py
class LocalElasticAgent(SimpleElasticAgent):
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        spec = worker_group.spec
        store = worker_group.store
        master_addr, master_port = super()._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts
        args: Dict[int, Tuple] = {}
        envs: Dict[int, Dict[str, str]] = {}
        ## 为woker 进程准备 环境变量 和启动参数
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                ...
            }
            envs[local_rank] = worker_env
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)
        self._pcontext = start_processes(name=spec.role,entrypoint=spec.entrypoint,
            args=args,envs=envs,log_dir=attempt_log_dir,start_method=self._start_method,...)
        return self._pcontext.pids()
```

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
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

             # 输出loss 和 正确率
    print("\n            =======  Training Finished  ======= \n")
```
如果是单机多卡，定义网络时加入 `net = nn.DataParallel(net)`

## 分布式训练


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
## 只读取属于 该rank worker 的样本
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=args.world_size,rank=rank)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
...
net = Net() 
net = Net().to(device) # device 代表到某个 gpu 或cpu 上
## 使用DistributedDataParallel 修饰模型
net = torch.nn.parallel.DistributedDataParallel(net)
```

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

## 弹性分布式训练 （未完成）

###  checkpoint 的保存与加载

1. 定义一个 State class 保存到文件上。如果文件存在共享存储，所有worker 都可以访问到，则所有worker 直接加载最新的checkpoint 即可。如果每个worker 将checkpoint 保存在本地，则有一个worker 间彼此通信 确定最新checkpoint 的过程。PS： 一个worker 挂掉了，重启启动，加载本地的checkpoint，这个checkpoint 大概率不是最新的了。
2. State class 包含哪些内容由 开发 自己决定。

```python
class State:
    def __init__(self, arch, model, optimizer):
        self.epoch = -1
        self.best_acc1 = 0
        self.arch = arch
        self.model = model
        self.optimizer = optimizer
    def capture_snapshot(self):
        return {
            "epoch": self.epoch,
            "best_acc1": self.best_acc1,
            "arch": self.arch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
    def apply_snapshot(self, obj):
        self.epoch = obj["epoch"]
        self.best_acc1 = obj["best_acc1"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])
    def save(self, f):
        torch.save(self.capture_snapshot(), f)
    def load(self, f):
        snapshot = torch.load(f)
        self.apply_snapshot(snapshot)
def load_checkpoint(checkpoint_file: str,arch: str,model: DistributedDataParallel,optimizer,) -> State:
    state = State(arch, model, optimizer)
    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file)
        print(f"=> loaded checkpoint file: {checkpoint_file}")
    # logic below is unnecessary when the checkpoint is visible on all nodes! create a temporary cpu pg to broadcast most up-to-date checkpoint
    ...
    print(f"=> done restoring from previous checkpoint")
    return state
def save_checkpoint(state: State, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
```

checkpoint 的保存 pytorch-lightning 库 [SAVING AND LOADING WEIGHTS](https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html ) 提供了一些参考，但没有 与worker group 其它worker 同步查找最新 checkpoint 的功能。