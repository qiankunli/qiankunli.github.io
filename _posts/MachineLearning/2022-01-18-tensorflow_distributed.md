---

layout: post
title: tensorflow分布式训练
category: 架构
tags: MachineLearning
keywords:  tensorflow distributed

---

## 简介（未完成）

* TOC
{:toc}

TensorFlow的分布式框架是基于ps/worker 模式的，与单机代码差距比较大，相对来说，horovod 支持的allreduce 模式性能更好些，对单机代码的改动也小一些。通常情况下，同步训练通过全归约（all-reduce）实现，而异步训练通过参数服务器架构实现。

## 使用
### 使用Strategy

![](/public/upload/machine/tf_strategy_api.png)

[使用 TensorFlow 进行分布式训练](https://mp.weixin.qq.com/s/bHTI6KOxIeMYMIaa6ptcSA) tf.distribute.Strategy 是一个可在多个 GPU、多台机器或 TPU 上进行分布式训练的 TensorFlow API。使用此 API，您只需改动较少代码就能基于现有模型和训练代码来实现单机多卡，多机多卡等情况的分布式训练。

1. MirroredStrategy，单机多卡数据并行，使用 nccl all-reduce 在设备之间传递变量更新
2. MultiWorkerMirroredStrategy，多机多卡分布式版本，使用all-reduce
3. ParameterServerStrategy，ps/worker 异步模式

[基于Tensorflow高阶API构建大规模分布式深度学习模型系列: 开篇](https://zhuanlan.zhihu.com/p/38470806)

```python

import tensorflow as tf
import os
import json

NUM_WORKERS = 1
IP_ADDRS = ['localhost']
PORTS = [12345]

def model_fn(...):
  .....
def input_fn(...):
  .....
# 需要每个机器配置TF_CONFIG环境变量
os.environ['TF_CONFIG'] = json.dumps({
  'cluster': {
      'worker': ['%s:%d' % (IP_ADDRS[w], PORTS[w]) for w in range(NUM_WORKERS)],
      'ps': ['%s:%d' % (IP_ADDRS[w], PORTS[w]) for w in range(NUM_WORKERS)]
  },
  'task': {'type': 'worker', 'index': 0}
})
# Method for using ParamterServerStrategy
strategy = tf.distribute.experimental.ParameterServerStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)
classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir='/tmp/multiworker', config=config)
tf.estimator.train_and_evaluate(classifier,train_spec=tf.estimator.TrainSpec(input_fn=input_fn),eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)
```

## Python层——Strategy 原理（未完成）

分布式训练，从client-master-worker 设计来说是完全支持的，但是Strategy 在其中起到了什么样的角色还有待分析。 

[TensorFlow 分布式 DistributedStrategy 之基础篇](https://mp.weixin.qq.com/s/n-hfz8bicdW8cU46yobnPQ)从系统角度或者说从开发者的角度看，Strategy 是基于Python作用域或装饰器来实现的一套机制。它提供了一组命名的分布式策略，如ParameterServerStrategy、CollectiveStrategy来作为Python作用域，这些策略可以被用来捕获用户函数中的模型声明和训练逻辑，其将在用户代码开始时生效。在后端，分布式系统可以重写计算图，并根据选择的策略合并相应的语义。

```python
# 示例
strategy = tf.distribute.XXStrategy(...)
with strategy.scope():
  # with 作用域内的代码 都可以通过 tf.distribute.get_strategy() 获取 当前的Strategy 
#
# tensorflow/python/distribute/distribution_strategy_context.py
def _push_per_thread_mode(context):
  ops.get_default_graph()._distribution_strategy_stack.append(context) 
def _pop_per_thread_mode():
  ops.get_default_graph()._distribution_strategy_stack.pop(-1)
def get_strategy():
  return _get_per_thread_mode().strategy
def _get_per_thread_mode():
  try:
    return ops.get_default_graph()._distribution_strategy_stack[-1]  # pylint: disable=protected-access
  except (AttributeError, IndexError):
    return _get_default_replica_mode()
```
strategy.scope ==> StrategyExtendedV2._scope 返回一个 _CurrentDistributionContext 对象，作为一个上下文管理器，包含 `__enter__` 和 `__exit__` 方法（with退出时执行），关键就是  `__enter__` 执行了 `_push_per_thread_mode(self._context)` ==> = distribution_strategy_context._push_per_thread_mode，`__exit__` 执行了 _pop_per_thread_mode()。 实质是将 contenxt 信息保存在了thread local上，with 作用域内的代码 都可以通过 `tf.distribute.get_strategy()` 获取 当前的Strategy 。

```python
class StrategyBase(object):
  def run(self, fn, args=(), kwargs=None, options=None):      # Invokes `fn` on each replica, with the given arguments.
    ...
  def reduce(self, reduce_op, value, axis):   # Reduce `value` across replicas and return result on current device.
    ...
  def extended(self):     # Strategy 很多操作直接转给了 StrategyExtendV1/V2
    return self._extended
  def scope(self):
    return self._extended._scope(self) 
  def make_dataset_iterator(self, dataset):
  def make_input_fn_iterator(self,input_fn,replication_mode=InputReplicationMode.PER_WORKER):
  def experimental_distribute_dataset(self, dataset, options=None):
  def distribute_datasets_from_function(self, dataset_fn, options=None):
  def experimental_run(self, fn, input_iterator=None):
    with self.scope():
      args = (input_iterator.get_next(),) if input_iterator is not None else ()
    return self.run(fn, args=args)
```

StrategyBase 对模型数据准备（比如dataset）、训练以及reduce 等操作都提供了抽象函数，由各个Strategy 子类具体实现。在与上层训练代码结合的时候，有两种方式
1. 上层训练代码 一定是写在 `with strategy.scope()` 下的，因此上层训练代码 本身可以通过 `tf.distribute.get_strategy()` 获取当前Strategy，进行处理或直接使用 Strategy 方法 PS：猜的
2. 完全自己基于Strategy 手写训练代码，A custom training loop can be as simple as
  ```python
  with my_strategy.scope():
    @tf.function
    def distribute_train_epoch(dataset):
      def replica_fn(input):
        # process input and return result
        return result
      total_result = 0
      for x in dataset:
        per_replica_result = my_strategy.run(replica_fn, args=(x,))
        total_result += my_strategy.reduce(tf.distribute.ReduceOp.SUM,per_replica_result, axis=None)                                      
      return total_result
    dist_dataset = my_strategy.experimental_distribute_dataset(dataset)
    for _ in range(EPOCHS):
      train_result = distribute_train_epoch(dist_dataset)
  ```

## Core层——运行

### 基本概念

在Tensorflow分布式中，主要介绍几个概念

1. Cluster : 是所有job的集合
2. Job: 是任务的集合
3. Task：是具体的任务


首先我们需要定义一个由参与分布式计算的机器组成的集群
```py
tf.train.ClusterSpec({
    "worker": [
      "10.244.2.141:2222",
      "10.244.2.142:2222",      
    ],
    "ps": [
        "10.244.2.140:2222",
    ]
})
```

可以通过脚本或者借助调度框架来动态构建 ClusterSpec。 "ps" 及 "worker" 为 `job_name`


### 进程和主服务
[分布式tensorflow（一）](https://lynnapan.github.io/2017/09/04/distributed%20tensorflow/)

1. client 是访问服务的部分
2. master是用来维护参数或者数据初始化的
3. worker是用来执行具体任务的
4. chief，集群中一般有多个worker，需要指定其中一个worker为主节点（cheif，默认worker0），chief节点会执行一些额外的工作，比如模型导出之类的。

客户端（client）进程负责构建计算图（graph），创建 tensorflow::Session 实例。客户端一般由 Python 或 C++ 编写。当客户端调用 Session.run() 时将向主进程（master）发送请求，主进程会选择特定工作进程（worker）完成实际计算。客户端、主进程和工作进程可以位于同一台机器实现本地计算，也可以位于不同机器即分布式计算。主进程和工作进程的集合称为服务端(server)，一个客户端可以同多个服务端交互。服务端进程会创建 tf.train.Server 实例并持续运行。client 与 server 之间以 grpc 通信

在每一台机器上起一个tf.train.Server的服务，然后放在一个集群里，整个集群的server会通过网络通信。

![](/public/upload/machine/tensorflow_server.png)


## 部署

### tensorflow手工启动示例

TensorFlow 没有提供一次性启动整个集群的解决方案，所以用户需要在每台机器上逐个手动启动一个集群的所有ps 和worker 任务。为了能够以同一行代码启动不同的任务，我们需要将所有worker任务的主机名和端口、 所有ps任务的主机名和端口、当前任务的作业名称以及任务编号这4个集群配置项参数化。通过输入不同的命令行参数组合，用户就可以使用同一份代码启动每一个任务。

```sh
// 在在参数服务器上执行
python mnist_dist_test_k8s.py --ps_hosts=10.244.2.140:2222 --worker_hosts=10.244.1.134:2222,10.244.2.141:2222 --job_name="ps" --task_index=0
// 在第一个worker节点上执行
python mnist_dist_test_k8s.py --ps_hosts=10.244.2.140:2222 --worker_hosts=10.244.1.134:2222,10.244.2.141:2222 --job_name="worker" --task_index=0
// 在第二个worker节点上执行
python mnist_dist_test_k8s.py --ps_hosts=10.244.2.140:2222 --worker_hosts=10.244.1.134:2222,10.244.2.141:2222 --job_name="worker" --task_index=1
```

### 与k8s 整合

[Kubeflow实战系列: 利用TFJob运行分布式TensorFlow](https://mp.weixin.qq.com/s/PmAU0MrPkKh6YiWpFXTRFg) TFJob 的核心是构建ClusterSpec。

tf_operator的工作就是创建对应的5个Pod, 并且将环境变量TF_CONFIG传入到每个Pod中，TF_CONFIG包含三部分的内容，当前集群ClusterSpec， 该节点的角色类型，以及id。比如该Pod为worker0，它所收到的环境变量TF_CONFIG为:

```json
{
  "cluster":{
    "master":[],   // tfjob 加的一个概念，可以当做worker 
    "ps":[],
    "worker":[]
  },
  "task":{
    "type":"worker",
    "index":0
  },
  "environment":"cloud"
}
```

对于使用者来说，他只需要在这里代码中使用通过获取环境变量TF_CONFIG中的上下文。

```python
def main(unused_argv):
  # 从环境变量 TF_CONFIG 中读取json 格式数据 反序列化成python 对象
  tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
  task_config = tf_config.get('task', {})
  task_type = task_config.get('type')
  task_index = task_config.get('index')

  FLAGS.job_name = task_type
  FLAGS.task_index = task_index

  # 构建ClusterSpec
  cluster_config = tf_config.get('cluster', {})
  ps_hosts = cluster_config.get('ps')
  worker_hosts = cluster_config.get('worker')
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

  # 创建Server 对象
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()

  is_chief = (FLAGS.task_index == 0)
  # 创建 worker_device
  if FLAGS.num_gpus > 0:
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
  with tf.device(tf.train.replica_device_setter(worker_device=worker_device,ps_device="/job:ps/cpu:0",cluster=cluster)):
    ...
```

这里面可以看到 一个代码风格是 训练代码是模板化的，无论在哪来运行，要做的就是 赋值 FLAGS.job_name/FLAGS.task_index/ps_spec/worker_spec/worker_device

```python
def main(unused_argv):
  FLAGS.job_name = ...
  FLAGS.task_index = ...
  ps_spec = ...
  worker_spec = ...
  cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()
  is_chief = (FLAGS.task_index == 0)
  worker_device = ...
  with tf.device(tf.train.replica_device_setter(worker_device=worker_device,ps_device="/job:ps/cpu:0",cluster=cluster)):
    ...
```

## 部分源码分析

[[翻译] TensorFlow 分布式之论文篇 (1)](https://mp.weixin.qq.com/s/KHTFE3tinwwNEzDTnPJpXA)
[分布式tensorflow源码解读2：MonitoredTrainingSession](https://zhuanlan.zhihu.com/p/91608555)
1. 如果chief节点，负责session的初始化或者从已有checkpoint恢复session，并且创建一些用于保存checkpoint和summary的hooks。如果是非chief的worker节点，则需要依赖chief节点完成初始化或恢复session这些操作后才能设置属于自己的session。
2. MonitoredTrainingSession可根据不同的角色去创建不同种类的Session，其中chief节点是由ChiefSessionCreator类去创建session，而非chief的worker节点是由WorkerSessionCreator类创建，特殊之处就是创建时调用的是wait_for_session()，大致意思是需要等待chief节点的session创建完成之后才去创建属于自己节点的session。
3. 创建session都是属于SessionManager类的一个方法
  ```python
  class SessionManager(object):
    # prepare_session函数的作用就是如果有checkpoint存在，就从checkpoint恢复session，如果不存在checkpoint就从传入的`init_op`和 调用`init_fn`函数去创建session。
    def prepare_session(self,master,init_op=None,saver=None,checkpoint_dir=None,...,init_fn=None):
      sess, is_loaded_from_checkpoint = self._restore_checkpoint(...)
      if not is_loaded_from_checkpoint:
        if init_op is None and not init_fn and self._local_init_op is None:
          raise RuntimeError("Model is not initialized and no init_op or init_fn or local_init_op was given")
        if init_op is not None:
          sess.run(init_op, feed_dict=init_feed_dict)
        if init_fn:
          init_fn(sess)
      return sess
  ```
## horovod

因为 **TensorFlow的分布式框架是基于参数服务器的**，这种结构容易造成网络堵塞；并且开源版 TensorFlow 的跨机通信是通过 gRPC + Protocol Buffers 实现的，这种方案的问题是，首先 gRPC 本身的效率就比较差，其次使用 Protocol Buffers 序列化就意味着节点间的所有交互必须经过内存，无法使用 GPUDirect RDMA，限制了速度提升；

[深度学习分布式训练框架 horovod (18) --- kubeflow tf-operator](https://mp.weixin.qq.com/s/ecpGB1ZLfu7G3Q3xCnuaeg)
[深度学习分布式训练框架 horovod (19) --- kubeflow MPI-operator](https://mp.weixin.qq.com/s/83_5FKrGFy1oupMIkulJhg)

## 其它

### 容错性

1. 最好的情况就是非Chief的worker task出错了,因为这些task实际上是无状态的。那么当遇到了这种错误，当这样的一个worker task恢复的时候，它会重新与它的PS task中建立连接，并且重新开始之前崩溃过的进程。
2. 比较差的一种情况就是PS task失败了，那么就有一个麻烦，因为PS task是有状态的，所有的worker task需要依赖他们来发送他们的梯度并且取得新的参数值。所以这种情况下，他们的chief worker task负责监测这种错误，如果发生了这种错误，chief worker task就打断整个训练，并从上一个检查点恢复所有的PS tasks。
3. 最糟糕的情况就是chief worker task失败了，打断训练，并在当它恢复了时候从上一个好的检查点恢复。

Fault tolerance 的API

MonitoredTrainingSession会自动帮你初始化参数，并且当PS 任务失败会自动恢复。
```python
server = tf.train.Server(...)
is_cheif = FLAGS.task_index == 0
with tf.train.MonitoredTrainingSession(master=server.target,is_chief=is_chief) as sess:
  while not sess.should_stop():
    sess.run(train_op)
```



### 低阶api ps-worker代码/使用Server（知道就行了）

```python
#coding=utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data    # 数据的获取不是本章重点，这里直接导入

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("job_name", "worker", "ps or worker")
tf.app.flags.DEFINE_integer("task_id", 0, "Task ID of the worker/ps running the train")
tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222", "ps机")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "worker机，用逗号隔开")

# 全局变量
MODEL_DIR = "./distribute_model_ckpt/"
DATA_DIR = "./data/mnist/"
BATCH_SIZE = 32

# main函数
def main(self):
    # ==========  STEP1: 读取数据  ========== #
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True, source_url='http://yann.lecun.com/exdb/mnist/')   
    # ==========  STEP2: 声明集群  ========== #
    # 构建集群ClusterSpec和服务声明
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps":ps_hosts, "worker":worker_hosts})    # 构建集群名单
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)    # 声明服务
    n_workers = len(worker_hosts)    # worker机的数量
    # ==========  STEP3: ps机内容  ========== #
    # 分工，对于ps机器不需要执行训练过程，只需要管理变量。server.join()会一直停在这条语句上。
    if FLAGS.job_name == "ps":
      with tf.device("/cpu:0"):
        server.join()
    # ==========  STEP4: worker机内容  ========== #
    # 下面定义worker机需要进行的操作
    is_chief = (FLAGS.task_id == 0)    # 选取task_id=0的worker机作为chief
    # 通过replica_device_setter函数来指定每一个运算的设备。
    # replica_device_setter会自动将所有参数分配到参数服务器上，将计算分配到当前的worker机上。
    device_setter = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_id,cluster=cluster)
    # 这一台worker机器需要做的计算内容
    with tf.device(device_setter):
      # 输入数据
      x = tf.placeholder(name="x-input", shape=[None, 28*28], dtype=tf.float32)    # 输入样本像素为28*28
      y_ = tf.placeholder(name="y-input", shape=[None, 10], dtype=tf.float32)      # MNIST是十分类
      # 第一层（隐藏层）
      with tf.variable_scope("layer1"):
          weights = tf.get_variable(name="weights", shape=[28*28, 128], initializer=tf.glorot_normal_initializer())
          biases = tf.get_variable(name="biases", shape=[128], initializer=tf.glorot_normal_initializer())
          layer1 = tf.nn.relu(tf.matmul(x, weights) + biases, name="layer1")
      # 第二层（输出层）
      with tf.variable_scope("layer2"):
          weights = tf.get_variable(name="weights", shape=[128, 10], initializer=tf.glorot_normal_initializer())
          biases = tf.get_variable(name="biases", shape=[10], initializer=tf.glorot_normal_initializer())
          y = tf.add(tf.matmul(layer1, weights), biases, name="y")
      pred = tf.argmax(y, axis=1, name="pred")
      global_step = tf.contrib.framework.get_or_create_global_step()    # 必须手动声明global_step否则会报错
      # 损失和优化
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, axis=1))
      loss = tf.reduce_mean(cross_entropy)
      # **通过tf.train.SyncReplicasOptimizer函数实现函数同步更新**
      opt = tf.train.SyncReplicasOptimizer(
        tf.train.GradientDescentOptimizer(0.01),
        replicas_to_aggregate=n_workers,
        total_num_replicas=n_workers
      )
      sync_replicas_hook = opt.make_session_run_hook(is_chief)
      train_op = opt.minimize(loss, global_step=global_step)
      if is_chief:
        train_op = tf.no_op()
      hooks = [sync_replicas_hook, tf.train.StopAtStepHook(last_step=10000)]    # 把同步更新的hook加进来
      config = tf.ConfigProto(
        allow_soft_placement=True,    # 设置成True，那么当运行设备不满足要求时，会自动分配GPU或者CPU。
        log_device_placement=False,   # 设置为True时，会打印出TensorFlow使用了哪种操作
      )

      # ==========  STEP5: 打开会话  ========== #
      # 对于分布式训练，打开会话时不采用tf.Session()，而采用tf.train.MonitoredTrainingSession()
      # 详情参考：https://www.cnblogs.com/estragon/p/10034511.html
      with tf.train.MonitoredTrainingSession(master=server.target,is_chief=is_chief,checkpoint_dir=MODEL_DIR,hooks=hooks,save_checkpoint_secs=10,config=config) as sess:
        print("session started!")
        start_time = time.time()
        step = 0      
        while not sess.should_stop():
          xs, ys = mnist.train.next_batch(BATCH_SIZE)    # batch_size=32
          _, loss_value, global_step_value = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})
          if step > 0 and step % 100 == 0:
            duration = time.time() - start_time
            sec_per_batch = duration / global_step_value
            print("After %d training steps(%d global steps), loss on training batch is %g (%.3f sec/batch)" % (step, global_step_value, loss_value, sec_per_batch))
          step += 1

if __name__ == "__main__":
    tf.app.run()
```




