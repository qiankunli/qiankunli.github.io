---

layout: post
title: tensorflow原理
category: 架构
tags: MachineLearning
keywords: tensorflow

---

## 简介

本文内容主要来自 《深入理解Tensorflow》 和 《Tensorflow内核剖析》

![](/public/upload/machine/tensorflow_graph.png)

神经网络在 视觉上 是一层一层的，表达式上是张量计算，执行上是数据流图。

||||
|---|---|---|
|视图层|可视化|TensorBoard|
|工作流层|数据集准备、存储、加载|Keras|
|计算图层|图构造/操作/优化/执行/前向计算/后向传播|TensorFlow Core|
|数值计算层|Kernel实现/矩阵乘法/卷积计算|Eigen/cuBLAS/cuDNN|
|网络层|通信|grpc/RDMA|
|设备层|硬件|CPU/GPU|

## 数据流图的整体执行

TensorFlow 使用数据流图表达计算过程和共享状态，使用节点表示抽象计算，使用边 表示数据流。如下图，展示了 MNIST 手写识别应用的数据流图。在该模型 中，前向子图使用了 2 层全连接网络，分别为 ReLU 层和 Softmax 层。随后，使用 SGD 的 优化算法，构建了与前向子图对应的反向子图，用于计算训练参数的梯度。最后，根据参数 更新法则，**构造训练参数的更新子图**，完成训练参数的迭代更新。

![](/public/upload/machine/tf_mnist.png)

《深入理解Tensorflow》数据流图计算粗略的分为 应用程序逻辑、 会话生命周期和算法核函数执行 这3个层次
1. 在应用程序逻辑中，用户使用Python 等应用层API 及高层抽象编写算法 模型，无需关心图切分、进程间通信等底层实现逻辑。算法涉及的计算逻辑和输入数据**绑定到图抽象中**，计算迭代控制语义体现在会话运行前后（即session.run）的控制代码上。
2. 在会话生命周期层次，单机会话与分布式会话具有不同的设计。 
  1. 单机会话采用相对简单的会话层次和图封装结构，它将图切分、优化之后，把操作节点和张量数据提交给底层执行器
  2. 分布式会话分为 client、master和worker 三层组件，它们对计算任务进行分解和分发，并通过添加通信操作 来确保计算逻辑的完整性。
3. 在算法核函数执行层次， 执行器抽象将会话传入的核函数加载到各个计算设备上有序执行。为充分利用多核硬件的并发计算能力，这一层次提供线程池调度机制；为实现众多并发操作的异步执行和分布式协同， 这一层次引入了通信会合点机制。

### client-master-worker

![](/public/upload/machine/run_graph.png)

应用层数据流图 表示为Python API 中的tensoflow.Graph 类，通信时表示为 基于Protocol Buffers 文件定义的GraphDef (以 Session 为桥梁，建立 Client 与 Master 之间的通道，并将 Protobuf 格式的 GraphDef 序列 化后传递给 Master)，运行时的数据流图 表示为C++ 代码中的Graph 类及其成员类型。

在分布式的运行时环境中，Client 执行 Session.run 时，传递整个计算图给后端的 Master。此时，计算图是完整的，常称为 Full Graph。随后，Master 根据 Session.run 传 递给它的 fetches, feeds 参数列表，反向遍历 Full Graph，并按照依赖关系，对其实施剪 枝，最终计算得到最小的依赖子图，常称为 Client Graph。
接着，Master 负责将 Client Graph 按照任务的名称分裂 (SplitByTask) 为多个 Graph Partition;其中，每个 Worker 对应一个 Graph Partition。随后，Master 将 Graph Partition 分别注册到相应的 Worker 上，以便在不同的 Worker 上并发执行这些 Graph Partition。最 后，Master 将通知所有 Work 启动相应 Graph Partition 的执行过程。
其中，Work 之间可能存在数据依赖关系，Master 并不参与两者之间的数据交换，它们 两两之间互相通信，独立地完成交换数据，直至完成所有计算。

对于每一个任务，TensorFlow 都将启动一个 Worker 实例。Worker 主要负责如下 3 个 方面的职责:
1. 处理来自 Master 的请求;
2. 对注册的 Graph Partition 按照本地计算设备集实施二次分裂 (SplitByDevice)，并通知各个计算设备并发执行各个 Graph Partition;
3. 按照**拓扑排序算法**在某个计算设备上执行本地子图，并调度 OP 的 Kernel 实现; 
4. 协同任务之间的数据通信。Worker 要负责将 OP 运算的结果发送到其他的 Worker 上去，或者接受来自 其他 Worker 发送给它的运算结果，以便实现 Worker 之间的数据交互。TensorFlow 特化实 现了源设备和目标设备间的 Send/Recv。
  1. 本地 CPU 与 GPU 之间，使用 cudaMemcpyAsync 实现异步拷贝;
  2. 本地 GPU 之间，使用端到端的 DMA 操作，避免主机端 CPU 的拷贝。
  3. 对于任务间的通信，TensorFlow 支持多种通信协议。1. gRPC over TCP;2. RDMA over Converged Ethernet。并支持 cuNCCL 库，用于改善多 GPU 间的通信。

Kernel 是 OP 在某种硬件设备的特定实现，它负责执行 OP 的具体运算。目前， TensorFlow 系统中包含 200 多个标准的 OP，包括数值计算，多维数组操作，控制流，状 态管理等。

## 数据流图的创建

### 《深入理解Tensorflow》

1. 全图构造
2. 子图提取
3. 图切分，将一幅子图按照其 操作节点放置的设备，切分为若干局部数据流图的过程，切分生成的每幅局部图仅在一个设备上运行，通信操作节点（SendOp,RecvOp）被插入局部图，以确保执行子图的逻辑语义同切分之前一致。
4. 图优化

![](/public/upload/machine/create_graph.jpeg)

### 《Tensorflow内核剖析》

假如存在一个简单的分布式环境:1 PS + 1 Worker

![](/public/upload/machine/tf_client_master_worker.png)

图构造：Client 构建了一个简单计算图;首先，将 w 与 x 进行矩阵相 乘，再与截距 b 按位相加，最后更新至 s 中。

![](/public/upload/machine/tf_create_graph.png)

图执行：Client 创建一个 Session 实例，建立与 Master 之间的 通道;接着，Client 通过调用 Session.run 将计算图传递给 Master。Master 会实施一系列优化技术，例如公共表达式消除，常量折叠等。最后，Master 负责任务之间的协同，执
 行优化后的计算图。

![](/public/upload/machine/tf_run_graph.png)

图分裂：存在一种合理的图划分算法。Master 将模型参数相关的 OP 划 分为一组，并放置在 ps0 任务上;其他 OP 划分为另外一组，放置在 worker0 任务上执行。

![](/public/upload/machine/tf_split_graph.png)

子图注册：在图分裂过程中，如果计算图的边跨越节点或设备，Master 将 该**边实施分裂**，在两个节点或设备之间插入 Send 和 Recv 节点，实现数据的传递。其中，Send 和 Recv 节点也是 OP，只不过它们是两个特殊的 OP，由内部运行时管理和控制，对用户不可见;并且，它们仅用于数据的通信，并没有任何数据计算的逻辑。最后，Master 通过调用 RegisterGraph 接口，将子图注册给相应的 Worker 上，并由相 应的 Worker 负责执行运算。
 
![](/public/upload/machine/tf_register_child_graph.png)

子图运算：Master 通过调用 RunGraph 接口，通知所有 Worker 执行子图 运算。其中，Worker 之间可以通过调用 RecvTensor 接口，完成数据的交换。

![](/public/upload/machine/tf_worker_run_graph.png)


## 会话管理

### 会话生命周期与图控制

《Tensorflow内核剖析》
1. 创建会话：Client 首次执行 tf.Session.run 时，会将整个图序列化后，通过 gRPC 发送CreateSessionRequest 消息，将图传递给 Master。随后，Master 创建一个 MasterSession 实例，并用全局唯一的 handle 标识，最终通过
CreateSessionResponse 返回给 Client。 
2. 迭代运行：随后，Client 会启动迭代执行的过程，并称每次迭代为一次 Step。此时，Client 发送 RunStepRequest 消息给 Master，消息携带 handle 标识，用于 Master 索引相应的 MasterSession 实例。
3. 注册子图：Master 收到 RunStepRequest 消息后，将执行图剪枝，分裂，优化等操作。最终按照任 务 (Task)，将图划分为多个子图片段 (Graph Partition)。随后，Master 向各个 Worker 发送 RegisterGraphRequest 消息，将子图片段依次注册到各个 Worker 节点上。当 Worker 收到 RegisterGraphRequest 消息后，再次实施分裂操作，最终按照设备 (Device)，将图划分为多个子图片段 (Graph Partition)。当 Worker 完成子图注册后，通过返回 RegisterGraphReponse 消息，并携带 graph_handle 标识。这是因为 Worker 可以并发注册并运行多个子图，每个子图使用 graph_handle 唯一 标识。
4. 运行子图：
Master 完成子图注册后，将广播所有 Worker 并发执行所有子图。这个过程是通过 Master 发送 RunGraphRequest 消息给 Worker 完成的。其中，消息中携带 (session_handle, graph_handle, step_id) 三元组的标识信息，用于 Worker 索引相应的子图。
Worker 收到消息 RunGraphRequest 消息后，Worker 根据 graph_handle 索引相应的子 图。最终，Worker 启动本地所有计算设备并发执行所有子图。其中，每个子图放置在单独 的 Executor 中执行，Executor 将按照拓扑排序算法完成子图片段的计算。上述算法可以形 式化地描述为如下代码。
  ```python
     def run_partitions(rendezvous, executors_and_partitions, inputs, outputs):
       rendezvous.send(inputs)
       for (executor, partition) in executors_and_partitions:
         executor.run(partition)
       rendezvous.recv(outputs)
  ```
5. 关闭会话：当计算完成后，Client 向 Master 发送 CloseSessionReq 消息。Master 收到消息后，开 始释放 MasterSession 所持有的所有资源。

![](/public/upload/machine/tf_run_graph_seq.png)

PS：tf 运行时 很像一个c++ 写的grpc server 程序。

### 单机会话运行 

读入数据流图的待执行子图以及必要的输入张量，依据图中定义的依赖关系，将每个节点对应的操作核函数有序的加载到各个计算设备上并发执行，并将计算结果作为后续节点的输入。会话的生命周期 最终完成子图上定义的所有计算语义，将输出结果以张量形式返回给创建会话的应用程序。ps：跟一个dag 流程编排软件的执行逻辑差不多。 

![](/public/upload/machine/execute_graph.png)

在本地模式下，Client, Master, Worker 部署在同一台机器同 一进程内，并由 DirectSession 同时扮演这三个角色。

Tensorflow 的关键路径为 run_step，用python 简化描述一下

```python
def run_step(devices, full_graph, inputs, outputs):
  client_graph = prune(full_graph, inputs, outputs)
  executors_and_partitions = split(client_graph, devices)
  run_partitions(executors_and_partitions, inputs, outputs)
def run_partitions(executors_and_partitions, inputs, outputs):
  frame = FunctionCallFrame()
  frame.set_args(inputs)
  do_run_partitions(executors_and_partitions)
  frame.get_ret_vals(outputs)
def do_run_partitions(executors_and_partitions):
  barrier = ExecutorBarrier(executors_and_partitions.size())
  for (executor, partition) in executors_and_partitions:
    executor.run(partition, barrier)
  barrier.wait()
```

在每个计算设备上，启动一个 Executor 执行分配给它的 PartitionGraph（即executor.run）。当某 一个计算设备执行完所分配的 PartitionGraph 之后，ExecutorBarrier 的计数器加 1，直至 所有设备完成 PartitionGraph 列表的执行，barrier.wait() 阻塞操作退出。


### 分布式会话运行

```
tf.train.ClusterSpec({
  "worker": [
    "worker0:2222",  # /job:worker/task:0
    "worker1:2222",  # /job:worker/task:1
    "worker2:2222"   # /job:worker/task:2
  ],
  "ps": [
    "ps0:2222",      # /job:ps/task:0
    "ps1:2222"       # /job:ps/task:0
]})
```

一般地，在分布式运行时中，Task (比如 `/job:worker/task:0`) 运行在独立的进程中，并在其上运行一个 tf.train.Server 实例。Server 表示 Task 的服务进程，它对外提供 MasterService 和 WorkerService 服务(grpc)。也 就是说，Server 可以同时扮演 Master 和 Worker 两种角色。

```
service MasterService {
  rpc CreateSession(CreateSessionRequest)
      returns (CreateSessionResponse);
  rpc ExtendSession(ExtendSessionRequest)
      returns (ExtendSessionResponse);
  rpc PartialRunSetup(PartialRunSetupRequest)
      returns (PartialRunSetupResponse);
  rpc RunStep(RunStepRequest)
      returns (RunStepResponse);
  rpc CloseSession(CloseSessionRequest)
      returns (CloseSessionResponse);
  rpc ListDevices(ListDevicesRequest)
      returns (ListDevicesResponse);
  rpc Reset(ResetRequest)
      returns (ResetResponse);
service WorkerService {
  rpc GetStatus(GetStatusRequest)
      returns (GetStatusResponse);
  rpc CreateWorkerSession(CreateWorkerSessionRequest)
      returns (CreateWorkerSessionResponse);
  rpc RegisterGraph(RegisterGraphRequest)
      returns (RegisterGraphResponse);
  rpc DeregisterGraph(DeregisterGraphRequest)
      returns (DeregisterGraphResponse);
  rpc RunGraph(RunGraphRequest)
      returns (RunGraphResponse);
  rpc CleanupGraph(CleanupGraphRequest)
      returns (CleanupGraphResponse);
  rpc CleanupAll(CleanupAllRequest)
      returns (CleanupAllResponse);
  rpc RecvTensor(RecvTensorRequest)
      returns (RecvTensorResponse) 
  rpc Logging(LoggingRequest)
      returns (LoggingResponse);
  rpc Tracing(TracingRequest)
      returns (TracingResponse);
```

在分布式模式中，Client 负责计算图的构造，然后通过调用 Session.run，启动计算图
的执行过程。
Master 进程收到计算图执行的消息后，启动计算图的剪枝，分裂，优化等操作;最终
将子图分发注册到各个 Worker 进程上，然后触发各个 Worker 进程并发执行子图。
Worker 进程收到子图注册的消息后，根据本地计算设备资源，再将计算子图实施二 次分裂，将子图分配在各个计算设备上，最后启动各个计算设备并发地执行子图;如果 Worker 之间存在数据交换，可以通过进程间通信完成交互。其中，在分布式运行时，图分裂经历了两级分裂过程。
1. 一级分裂:由 MasterSession 完成，按照 SplitByWorker 或 SplitByTask 完成图 分裂过程;
2. 二级分裂:由 WorkerSession 完成，按照 SplitByDevice 完成图分裂过程。

![](/public/upload/machine/tf_distributed.png)

```python
def run_step(devices, full_graph, inputs, outputs):
  executors_and_partitions = split(full_graph, devices)
  run_partitions(executors_and_partitions, inputs, outputs)
def run_partitions(executors_and_partitions, inputs, outputs):
  remote_rendezvous = RpcRemoteRendezvous()
  send_inputs(remote_rendezvous, inputs)
  do_run_partitions(executors_and_partitions)
  recv_outputs(remote_rendezvous, outputs)
def send_inputs(remote_rendezvous, inputs):
      for (key, tensor) in inputs:
        remote_rendezvous.send(key, tensor)
def do_run_partitions(executors_and_partitions):
  barrier = ExecutorBarrier(executors_and_partitions.size())
  for (executor, partition) in executors_and_partitions:
    executor.run(partition, barrier.on_done())
  barrier.wait()
def recv_outputs(remote_rendezvous, outputs):
  for (key, tensor) in outputs:
    remote_rendezvous.recv(key, tensor)
```

在分布式模式中，可能存在多个 Client 同时接入一个 Master， Master 为其每个接入的 Client 创建一个 MasterSession 实例。Worker 也可能同时为多个 Master 提供计算服务，Worker 为其每个请求计算的 Master 创建一个 WorkerSession 实例。 为了区分不同的 Client 的计算服务，使用不同的 session_handle 区分。

### 汇合点机制 / 设备间通信

在具体实现上，Tensorflow实现了Recv-Driven的数据交换模式，如上图所示，位于DeviceA和DeviceB的两张计算图会异步并发的执行，位于DeviceB的Recv执行时会发起一条RPC请求发往DeviceA，DeviceA收到请求后，会将请求路由到Rendezvous中，如果在当中发现所需要的数据已经生产好，并被Send算子注册了进来，那么就地获取数据，返回给DeviceB；如果此时数据还没有生产好，则将来自于DeviceB的Recv请求注册在Rendezvous中，等待后续DeviceA生产好后，由Send算子发送过来，找到注册的Recv，触发回调，返回数据给DeviceB。

跨设备的 PartitionGraph 之间可能存在数据依赖关系，它们之间通过插入 Send/Recv 节点完成交互。事实上，在本地模式中，Send/Recv 通过 Rendezvous 完成数据交换的。Send 将数据放在 Rendezvous 上，而 Recv 则根据标识从 Rendezvous 取走。其中，Send 不阻塞， 而 Recv 是阻塞的。也可以使用基于 FunctionCallFrame 函数调用替代之，使用 Arg/RetVal 分别替代 Send/Recv 节点，从而实现了函 数调用交换数据的方式。

ndOp/RecvOp 通过 Rendezvous 交换数据的;它实现了消息发送/接受，与具体消息传 递相解耦。例如，在单进程内，SendOp/RecvOp 基于 IntraProcessRendezvous 传递数据的; 而在多进程环境中，SendOp/RecvOp 则可以基于 GrpcRendezvous 传递数据。

```c++
// sendOp ==> Rendezvous.Send
struct SendOp : OpKernel {
  void Compute(OpKernelContext* ctx) override {
    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->input_alloc_attr(0);
    ctx->rendezvous()->Send(
      CreateParsedkey(ctx), args, ctx->input(0),
      ctx->is_input_dead());
  }
}
// recvOp ==> Rendezvous.RecvAsync 
struct RecvOp : AsyncOpKernel {
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->output_alloc_attr(0);
    ctx->rendezvous()->RecvAsync(
      CreateParsedKey(ctx), args, CreateDoneCallback(ctx));
  }
}
```


## 操作节点执行

在某个设备上，PartitionGraph 的起始节点为 Arg 节点，结束节点为 RetVal 节点。整 个过程可以看成函数调用过程，Arg 用于传递函数参数，RetVal 用于返回函数值。
更确切地说，Arg 完成 PartitionGraph 的输入，RetVal 完成 PartitionGraph 的输出。 对于 Arg 节点，其调用时序为:set_arg -> get_arg。其中，前者由 DirectSession 在启动 Executor 列表之前，通过调用 FunctionCallFrame.SetArgs(feeds)，传递输入参数列表的 值;后者由 Arg 的 Kernel 实现调用。

每个 Executor 将执行 PartitionGraph 的**拓扑排序**算法，将入度为 0 的 OP 追加到 ready_queue 之中，并将其关联的 OP 的入度减 1。调度器调度 ready_queue 之中 OP ，并 将其放入 ThreadPool 中执行对应的 Kernel 实现。
在所有 Partition 开始并发执行之前，需要外部将其输入传递给相应的 Arg 节点;当 所有 Partition 完成计算后，外部再从 RetVal 节点中取走数据。其中，Arg/RetVal 节点之 间的数据时通过 FunctionCallFrame 完成交互的。


```c++
// tensorflow/tensorflow/core/common_runtime/direct_session.cc
Status DirectSession::Run(const RunOptions& run_options,...){
  ...
  for (const auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }
}
// tensorflow/tensorflow/core/common_runtime/executor.cc
void ExecutorState::RunAsync(Executor::DoneCallback done) {
  const Graph* graph = impl_->graph_;
  TaggedNodeSeq ready;
  // Initialize the ready queue.
  for (const Node* n : impl_->root_nodes_) {
    DCHECK_EQ(n->in_edges().size(), 0);
    ready.push_back(TaggedNode{n, root_frame_, 0, false});
  }
  if (ready.empty()) {
    done(Status::OK());
  } else {
    num_outstanding_ops_ = ready.size();
    root_frame_->iterations[0]->outstanding_ops = ready.size();
    done_cb_ = std::move(done);
    // Schedule to run all the ready ops in thread pool.
    ScheduleReady(ready, nullptr);
  }
}
void ExecutorState::Process(TaggedNode tagged_node, int64 scheduled_usec) {
  // Parameters passed to OpKernel::Compute.
  TensorValueVec inputs;
  OpKernelContext::Params params;
  params.step_id = step_id_;
  Device* device = impl_->params_.device;
  params.device = device;
  params.inputs = &inputs;
  // Prepares inputs.
  // Set up compute params.
  OpKernel* op_kernel = item.kernel;
  params.op_kernel = op_kernel;
  params.frame_iter = FrameAndIter(input_frame->frame_id, input_iter);
  params.is_input_dead = is_input_dead;
  params.output_attr_array = item.output_attrs();
  if (item.kernel_is_async) {
    // Asynchronous computes.
    AsyncOpKernel* async = item.kernel->AsAsync();
    ...
    device->ComputeAsync(async, &state->ctx, done);
  } else {
    // Synchronous computes.
    OpKernelContext ctx(&params, item.num_outputs);
    if (stats) nodestats::SetOpStart(stats);
    device->Compute(CHECK_NOTNULL(op_kernel), &ctx);
    if (stats) nodestats::SetOpEnd(stats);
    s = ProcessOutputs(item, &ctx, &outputs, stats);
  }
  // This thread of computation is done if completed = true.
  if (completed) Finish();
}
```

操作节点执行 过程本质是 节点对应的核函数的执行过程。会话运行时，ExecutorImpl::Initialize 会对数据流图上每个操作节点 调用create_kernel 函数，这时创建的 核函数对象 是对应 操作在特定设备上的特化版本。

## BP阶段/梯度计算

Tensorflow的底层结构是由张量组成的计算图。计算图就是底层的编程系统，每一个计算都是图中的一个节点，计算之间的依赖关系则用节点之间的边来表示。计算图构成了前向/反向传播的结构基础。给定一个计算图, TensorFlow 使用自动微分 (反向传播) 来进行梯度运算。tf.train.Optimizer允许我们通过minimize()函数自动进行权值更新，此时`tf.train.Optimizer.minimize()`做了两件事：

1. 计算梯度。即调用`compute_gradients (loss, var_list …)` 计算loss对指定val_list的梯度，返回元组列表 `list(zip(grads, var_list))`。
2. 用计算得到的梯度来更新对应权重。即调用 `apply_gradients(grads_and_vars, global_step=global_step, name=None)` 将 `compute_gradients (loss, var_list …)` 的返回值作为输入对权重变量进行更新；
将minimize()分成两个步骤的原因是：可以在某种情况下对梯度进行修正，防止梯度消失或者梯度爆炸。

《Tensorflow内核剖析》

```python
class Optimizer(object):
  def minimize(self, loss, var_list=None, global_step=None):
    """Add operations to minimize loss by updating var_list.
    """
    grads_and_vars = self.compute_gradients(loss, var_list=var_list)
    return self.apply_gradients(grads_and_vars,
      global_step=global_step)
```

compute_gradients 将根据 loss 的值，求解 var_list=[v1, v2, ..., vn] 的梯度，最终 返回的结果为:[(grad_v1, v1), (grad_v2, v2), ..., (grad_vn, vn)]。其中，compute_gradients 将调用 gradients 方法，构造反向传播的子图，可以形式化地描述为

``` python
def compute_gradients(loss, grad=I):
  vrg = build_virtual_reversed_graph(loss)
  for op in vrg.topological_sort():
    # 对每个正 向子图中的 OP 寻找其「梯度函数」
    grad_fn = ops.get_gradient_function(op)
    # 调用该梯度函数，该梯度函数将构造该 OP 对 应的反向的局部子图。
    grad = grad_fn(op, grad)
def apply_gradients(grads_and_vars, learning_rate):
    for (grad, var) in grads_and_vars:
      # 对于每个 (grad_vi, vi)，构造一个更 新 vi 的子图
      apply_gradient_descent(learning_rate, grad, var)
```
综上述，正向的一个 OP 对应反向的一个局部子图，并由该 OP 的梯度函数负责构造。 一般地，梯度函数满足如下原型:

```python
@ops.RegisterGradient("op_name")
def op_grad_func(op, grad)
```
对于一个梯度函数，第一个参数 op 表示正向计算的 OP，根据它可以获取正向计算时 OP 的输入和输出;第二个参数 grad，是反向子图中上游节点传递过来的梯度，它是一个 已经计算好的梯度值 (初始梯度值全为 1)。一般地，正向子图中的一个 OP，对应反向子图中的一个局部子图。因为，正向 OP 的 梯度函数实现，可能需要多个 OP 才能完成相应的梯度计算。例如，Square 的 OP，对应梯 度函数构造了包含两个 2 个乘法 OP。

```
@ops.RegisterGradient("Square")
    def SquareGrad(op, grad):
      x = op.inputs[0]
      with ops.control_dependencies([grad.op]):
        x = math_ops.conj(x)
        return grad * (2.0 * x)
```

![](/public/upload/machine/tf_bp_grad.png)

一个简单的总结，当调用 Optimizer.minimize 方法时，使用 compute_gradients 方法，实现反向计算图的构造;使用 apply_gradients 方法，实现参数更新的子图构造。参数更新子图以 grads_and_vars 为输入，执行梯度下降的更新算法;最后，通 过 train_op 完成 global_step 值加 1，至此一轮 Step 执行完成。

## 自定义算子

一个Op可以接收一个或者多个输入Tensor，然后产生零个或者多个输出Tensor，分别利用Input和Output定义。在注册一个Op之后，就需要继承OpKernel，实现他的计算过程Compute函数，在Compute函数中，我们可以通过访问OpKernelContext来获得输入和输出信息。当我们需要申请新的内存地址时，可以通过OpKernelContext去申请TempTensor或者PersistentTensor。一般Tensorflow的Op都采用Eigen来操作Tensor

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