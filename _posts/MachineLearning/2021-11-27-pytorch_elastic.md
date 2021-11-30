---

layout: post
title: pytorch弹性分布式训练
category: 架构
tags: MachineLearning
keywords:  pytorch distributed elastic 弹性

---

## 简介

* TOC
{:toc}

## 弹性分布式训练 



关键就是
1. pytorch 层面，启动的时候，参与训练的 节点数不确定了，rank=0也不确定了。要能支持 互相发现worker（依赖注册中心） 以及协商各自的rank。 训练job 的每个worker 本质是相互协作的（尤其用了allreduce 来交换梯度），这个跟一般微服务多实例 仅仅是分担一下流量区别很大。扩缩容意味着 所有worker 都要重启。
2. 资源调度层面，为pytorch 启动pod 时，要提供注册中心地址信息。
3. 代码层面 要能随时 save 和load checkpoint，训练过程中，可能会有新的节点 或节点挂掉，要能容错，继续训练。

## train_script 的守护者elastic agent

`python -m torch.distributed.run train_script.py ` 对于每一个node 有两个角色
1. run.py 负责启动 elastic agent
2. elastic agent 负责启动 train_script.py， 并给train_script.py 传递必要的参数（环境变量或参数形式，由脚本的获取方式决定）。

elastic agent 是一个独立的进程，负责管理其下的 workers。它起到了类似进程管理系统 supervisor 的作用，会在启动的时候确保每个 worker 的启动参数（比如 WORLD_SIZE 和 RANK）正确，worker 的失效也是由 elastic agent 负责捕获处理。

![](/public/upload/machine/torchelastic_diagram.jpeg)

### worker 如何发现？—— Store

[Elastic Introduction](https://github.com/pytorch/elastic/blob/master/design/torchelastic/0.2.0/design_doc.md) 
[Elastic Agent 的设计：如何管理多个 worker 进程](https://mp.weixin.qq.com/s/hlOYLKSHFDZWN21AsUn6bg) 

Elastic Agent  通过 Store 进行 worker 之间的相互发现，以便在不同的节点间确定 RANK。 需要一个类似配置中心的东西 etcd 或自带的c10d，对应有一个 Store 抽象（有EtcdStore 和 TcpStore）

```python
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

”注册中心“可以是c10d/etcd，使用不同的“注册中心”有不同的问题
1. c10d 运行在rank0 节点， 因此使用c10d时，非rank0 节点挂掉ok，rank0 节点挂掉会导致训练任务失败
2. 使用etcd时，非rank0 节点挂掉ok，rank0 节点挂掉后 其它节点会作为rank0节点，可能会有问题：有些框架喜欢在rank0 做一些特殊工作

### rendezvous/集会机制

pytorch 封装了  RendezvousHandler 来管理 rendezvous（在etcd 上对应一个 带version 的path） ，核心功能
1. next_rendezvous， 如何在不同的节点间确定 RANK
2. 监听 worker 失效，标记 本轮 rendezvous 失败
    1. worker 运行失败。elastic agent 周期性 _monitor_workers 发现 worker 失败，SimpleElasticAgent.run 从死循环退出，执行`rdzv_handler.shutdown()` 通知了注册中心/store，对于etcdStore 来说 `/rdzv/active_version = {"status" = "closed"}`
    2. 被外力干死。elastic agent 注册了 `signal.signal(signal.SIGTERM, _terminate_process_handler)`  之后 引起 `rdzv_handler.shutdown()`

```python
# /pytorch/torch/distributed/elastic/rendezvous/api.py
# rdzv backend: etcd/etcd-v2/c10d/static
class RendezvousHandler(ABC):
    def get_backend(self) -> str
    def next_rendezvous(self,) -> Tuple[Store, rank, world_size]  # 建立 rendezvous 时 agent 获取 rank 和world_size
    def is_closed(self) -> bool
    def set_closed(self)
    def num_nodes_waiting(self) -> int
    def get_run_id(self) -> str
    def shutdown(self) -> bool  # 监听worker 失效后关闭 本轮rendezvous

```
pytoch 针对 RendezvousHandler 维护了一个 RendezvousHandlerRegistry，各种类型的  backend 注册了 RendezvousHandler 实现
```python
# /pytorch/torch/distributed/elastic/rendezvous/api.py
class RendezvousHandlerRegistry:
    _registry: Dict[str, RendezvousHandlerCreator]
    def register(self, backend: str, creator: RendezvousHandlerCreator) -> None:
    def create_handler(self, params: RendezvousParameters) -> RendezvousHandler:
rendezvous_handler_registry = RendezvousHandlerRegistry()
```

launch_agent ==> `rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_parameters)` 首先 根据backend 确定类型，再根据 rdzv_parameters 对 rendezvous_handler 初始化。

实际运行发现 rdzv_endpoint 中指定的port 由 `python -m torch.distributed.run train_script.py` 进程监听，也就是 **c10d 运行在 elastic agent 上**。PS： **代码上看 etcd 系列的会清晰一下**。

当一个worker 挂了之后，其它所有的worker 也都会挂掉。<font color="red">存在疑问<font>： 其它 节点的 elastic agent /worker 怎么知道的？
1. 可能是 rendezvous 缺了一个worker ，其它worker 交换不了梯度了，也就报错了，进而被各自的elastic agent 监听到。 
2. 可能是 elastic agent 通知了 注册中心 
    1. 其它node 的 elastic agent 监听到了这个信息，进而陆续关停自己的worker。
    2. 其它node 的 worker 监听到了这个信息，自己退出
理论上是第二种，但第二种没有找到 代码依据，第三种不知道是否成立。 

## 资源调度层

When a worker process fails, the corresponding elastic agent managing it kills all the workers on that node, establishes rendezvous with the other agents and restarts workers with the new rendezvous information. However, when an agent exits with a non-zero error code, it is up to a higher-level orchestrator such as Kubernetes to restart the agent (which in turn will restart all the workers it is responsible for). The same recovery mechanism holds for node-level failures. An orchestrator such as Kubernetes will schedule a job such that a minimum replicas of the elastic agent are running and each agent will in turn orchestrate the user's training script. 当一个worker 挂掉时，Elastic agent 将干掉节点上的worker(属于同一个local work group) 并通知其它agent。**当Elastic agent 挂掉时， 则需要 更高层级比如k8s 来重启Elastic agent**。

![](/public/upload/machine/torchelastic_agent_diagram.jpeg)

新的processgroup 被k8s 拉起来之后，关键问题就是两个
1. worker 发现，有几个？ rank=0 是谁？ rendezvous 的技能。
2. checkpoint 加载，防止之前 训练的浪费了。

## 代码层——save/load checkpoint

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
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}: 
                ...
                self._stop_workers(self._worker_group) ==> LocalElasticAgent._shutdown ==> 给子进程发送SIGTERM
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

## 其它

### run.py 启动命令

`python -m  torch.distributed.run [args] training_script [training_script_args]`，包含以下参数

关于workGroup 的布局
1. `--nnodes`
2. `--nproc_per_node`
关于集会机制
3. `--rdzv_backend`  - The backend of the rendezvous (e.g. ``c10d``). This is typically a strongly
   consistent key-value store.
4. `--rdzv_endpoint` - The rendezvous backend endpoint; usually in form ``<host>:<port>``.
5. `--rdzv_id`  - A user-defined id that uniquely identifies the worker group for a job. This id is
   used by each node to join as a member of a particular worker group.
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

### 如何启动 train_script 

train_script 启动方式有以下几种
1. 直接 python 启动 比较重要的参数是local_rank,world_size,master_host,master_port, backend
    1. WORLD_SIZE,  数据分几份
    2. RANK,         自己是第几份，自己是不是master（rank=0 是master）
    2. NPROC_PER_NODE,  `RANK / NPROC_PER_NODE` 可以推算 多卡里用哪一张卡
    4. MASTER_ADDR:MASTER_PORT,     如何发现其它成员，组建process_group时需要
2. 由 launch.py 启动，大部分参数直接 传给 train_script.py，launch根据 nnodes 和 nproc_per_node 算了一下 local_rank 和 world_size 传给train_script。 `python -m torch.distributed.launch --nnodes=2 --node_rank=xx --nproc_per_node=xx --master_addr=xx  --master_port=xx TRAINING_SCRIPT.py (... train script args ...)`。 **这个方法已经淘汰**，并且在不同的时期 launch.py 的实现有变化。
3. 由 run.py 启动，**参数 都是给 rendezvous 用的**， 由rendezvous 协商出 train_script 需要的参数，由 run.py spawn worker 进程时传给worker/传给train_script。 `python -m torchelastic.distributed.run --nnodes=1:4 --nproc_per_node=xx --rdzv_id=JOB_ID --rdzv_backend=etcd --rdzv_endpoint=ETCD_HOST:ETCD_PORT TRAINING_SCRIPT.py (... train script args ...)`

launch 和 run 仅仅是为启动 train_script 方便，该给 train_script 传的参数还是要传，不传参数用环境变量也行，传参或传环境变量需要的参数是一致的，可以到launch.py 或 run.py 源码下查看其启动需要哪些参数。PS：train_script 是算法写的，在train_script 内使用的库 具备分布式、弹性 能力之前，能力扩充 主要通过 加壳子的方式来解决。

python直接  启动 train_script 需要以下参数（来自run.py 代码注释）
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



