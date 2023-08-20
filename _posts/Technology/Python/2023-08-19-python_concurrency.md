---

layout: post
title: Python并发
category: 技术
tags: Python
keywords: Python

---

* TOC
{:toc}

Python有两大类并行方式：多线程与多进程。由于GIL的存在，这两种方式有着截然不同的特点：

1. 多线程可以直接共享数据，但至多只能用一个CPU核。Python有一个GIL来保证同一时间只能有一个线程来执行
2. 多进程可以用多个CPU核，但各进程的数据相互独立（可shared_memory等机制共享数据）

ps： 但使用体验上都很类似

## 线程

```python
# 方式一 用一个目标函数实例化一个Thread然后调用 start() 方法启动它。
from threading import Thread
import time

def sayhi(name):
    time.sleep(2)
    print('%s say hello' % name)
if __name__ == '__main__':
    t = Thread(target=sayhi, args=('egon',))
    t.start()
    print('主线程')

# 方式二，定义一个 Thread 类的子类，重写 run() 方法来实现逻辑
from threading import Thread
import time
class Sayhi(Thread):
    def __init__(self,name):
        super().__init__()
        self.name=name
    def run(self):
        time.sleep(2)
        print('%s say hello' % self.name)
if __name__ == '__main__':
    t = Sayhi('egon')
    t.start()
    print('主线程')
```

## 进程

multiprocess模块的完全模仿了threading模块的接口，二者在使用层面，有很大的相似性

```python
def function1(id):  # 这里是子进程。 和线程一样，也可以定义一个进程对象继承Process
    print(f'id {id}')

def run__process():  # 这里是主进程
    from multiprocessing import Process
    process = [Process(target=function1, args=(1,)),
               Process(target=function1, args=(2,)), ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 在进程结束之前一直等待，如果没有 join() ，主进程退出之后子进程会留在idle中，你必须手动杀死它们。

# run__process()  # 主线程不建议写在 if外部。由于这里的例子很简单，你强行这么做可能不会报错
if __name__ == '__main__':
    run__process()  # 正确做法：主线程只能写在 if内部


# 方式二 定义 Process 的子类
import multiprocessing
class MyProcess(multiprocessing.Process):
    def run(self):
        print ('called run method in process: %s' % self.name)
        return
if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = MyProcess()
        jobs.append(p)
        p.start()
        p.join()
```

多进程库提供了 Pool 类来实现简单的多进程任务。 

## 异步编程

Python3.2带来了 concurrent.futures 模块，这个模块具有线程池和进程池、管理并行编程任务、处理非确定性的执行流程、进程/线程同步等功能。current.Futures 模块提供了两种 Executor 的子类，各自独立操作一个线程池和一个进程池。
1. concurrent.futures.ThreadPoolExecutor(max_workers)
2. concurrent.futures.ProcessPoolExecutor(max_workers)。 使用了多核处理的模块，让我们可以不受GIL的限制，大大缩短执行时间。

```python
import concurrent.futures
import time
number_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def evaluate_item(x):
    result_item = count(x)      # 计算总和，这里只是为了消耗时间
    return result_item          # 打印输入和输出结果

def  count(number) :
    for i in range(0, 10000000):
        i=i+1
    return i * number

if __name__ == "__main__":
    # 线程池执行
    start_time_1 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(evaluate_item, item) for item in number_list]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
    print ("Thread pool execution in " + str(time.time() - start_time_1), "seconds")
    # 进程池
    start_time_2 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(evaluate_item, item) for item in number_list]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
    print ("Process pool execution in " + str(time.time() - start_time_2), "seconds")
```

## 协程（未完成）

## ray 并发库

Ray 是一个并行和分布式 Python 的开源库。从高层次上看，Ray 生态系统由三部分组成：核心 Ray 系统、用于机器学习的可扩展库（包括原生库和第三方库），以及用于在任何集群或云提供商上启动集群的工具。

[如何用 Python 实现分布式计算？](https://mp.weixin.qq.com/s/OwnUSDt96BxT8R4acqKi3g)Ray 是基于 Python 的分布式计算框架，采用动态图计算模型，提供简单、通用的 API 来创建分布式应用。使用起来很方便，你可以通过装饰器的方式，仅需修改极少的的代码，让原本运行在单机的 Python 代码轻松实现分布式计算，目前多用于机器学习。
1. 提供用于构建和运行分布式应用程序的简单原语。
2. 使用户能够并行化单机代码，代码更改很少甚至为零。
3. Ray Core 包括一个由应用程序、库和工具组成的大型生态系统，以支持复杂的应用程序。比如 Tune、RLlib、RaySGD、Serve、Datasets、Workflows。

```python
# 一个装饰器就搞定分布式计算
ray.init()  # 在本地启动 ray，如果想指定已有集群，在 init 方法中指定 RedisServer 即可
@ray.remote   # 声明了一个 remote function，是 Ray 的基本任务调度单元，它在定义后，会被立马序列化存储到 RedisServer 中，并且分配一个唯一的 ID，这样就能保证集群所有节点都能看到这个函数的定义；
def f(x):
    return x * x

futures = [f.remote(i) for i in range(4)]
print(ray.get(futures))  # [0, 1, 4, 9]    可以通过 ObjectID 获取 ObjectStore 内的对象并将之转换为 Python 对象，这个方法是阻塞的，会等到结果返回；
```
装饰器 `@ray.remote` 也可以装饰一个类：
```python
import ray
ray.init()

@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0
    def increment(self):
        self.n += 1
    def read(self):
        return self.n

counters = [Counter.remote() for i in range(4)]
tmp1 = [c.increment.remote() for c in counters]
tmp2 = [c.increment.remote() for c in counters]
tmp3 = [c.increment.remote() for c in counters]
futures = [c.read.remote() for c in counters]
print(ray.get(futures)) # [3, 3, 3, 3]
```


### 设计

[分布式计算框架Ray介绍](https://mp.weixin.qq.com/s/rY9cC9VGft7-bMEH0_xKWQ)当我们要构建一个涉及大规模数据处理或者复杂计算的应用，传统的方式是使用现成的大数据框架，例如 Apache Flink 和 Apache Spark。**这些系统提供的API通常基于某种特定的计算范式（例如DataStream、DataSet）**，要求用户基于这些特定的计算范式实现应用逻辑。对于传统的数据清洗、数据分析等应用，这种用法能够很好地适用。但是，随着分布式应用的逻辑越来越复杂（例如分布式机器学习应用），**许多应用的逻辑并不能直接套用现有的计算范式**。在这种情况下，**开发者如果想要细粒度地控制系统中的任务流，就需要自己从头编写一个分布式应用**。**但是现实中，开发一个分布式应用并不简单。除了应用本身的代码逻辑，我们还需要处理许多分布式系统中常见的难题**，例如：分布式组件通信、服务部署、服务发现、监控、异常恢复等。处理这些问题，通常需要开发者在分布式系统领域有较深的经验，否则很难保证系统的性能和健壮性。为了简化分布式编程，Ray提供了一套简单、通用的分布式编程API，屏蔽了分布式系统中的这些常见的难题，让开发者能够使用像开发单机程序一样简单的方式，开发分布式系统。Ray的API基于两个核心的概念：Task和Actor。PS：至少有一个状态中心，以及任务的分发与回收

Ray Task可以类比于单机程序中的函数。在Ray中，Task表示一个无状态的计算单元。Ray可以把一个任意的Python函数或Java静态方法转换成一个Task。在这个过程中，Ray会在一个远程进程中执行这个函数。并且这个过程是异步的，这意味着我们可以并行地执行多个Task。PS：和python线程、进程的异步执行很像。

```python
# `square`是一个普通的Python函数，`@ray.remote`装饰器表示我们可以把这个函数转化成Ray task.
@ray.remote
def square(x):
    return x * x
obj_refs = []
# `square.remote` 会异步地远程执行`square`函数。通过下面两行代码，我们并发地执行了5个Ray task。`square.remote`的返回值是一个`ObjectRef`对象，表示Task执行结果的引用。
for i in range(5):
    obj_refs.append(square.remote(i))
# 实际的task执行结果存放在Ray的分布式object store里，我们可以通过`ray.get`接口，同步地获取这些数据。
assert ray.get(obj_refs) == [0, 1, 4, 9, 16]
```

**Actor可以类比于单机程序中的类**。Ray使用Actor来表示一个有状态的计算单元。在Ray中，我们可以基于任意一个Python class或Java class创建Actor对象。这个Actor对象运行在一个远程的Python或者Java进程中。同时，我们可以通过ActorHandle远程调用这个Actor的任意一个方法（每次调用称为一个Actor Task），多个Actor Task在Actor进程中会顺序执行，并共享Actor的状态。

```python
# `Counter`是一个普通的Python类，`@ray.remote`装饰器表示我们可以把这个类转化成Ray actor.
@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0
    def increment(self):
        self.value += 1
    def get_value(self):
        return self.value
# `Counter.remote`会基于`Counter`类创建一个actor对象，这个actor对象会运行在一个远程的Python进程中。
counter = Counter.remote()
# `Counter.remote`的返回值是一个`ActorHandle`对象。通过`ActorHandle`，我们可以远程调用Actor的任意一个方法（actor task）。
[counter.increment.remote() for _ in range(5)]
# Actor task的返回值也是一个`ObjectRef`对象。同样地，我们通过`ray.get`获取实际的数据。
assert ray.get(counter.get_value.remote()) == 5
```

Obect Store是Ray架构中的一个关键组件，Task计算的中间结果会存放在分布式Object Store中。除此之外，我们也可以使用put接口，显式地把一个Python或Java对象存放到Object Store中。

在Ray中，我们可以把一个Task输出的ObjectRef传递给另一个Task（包括Actor task）。在这种情况下，Ray会等待第一个Task执行结束之后，再开始执行第二个Task。同时，我们也可以把一个ActorHandle传递给一个Task，从而实现在多个远程Worker中同时远程调用一个Actor。通过这些方式，我们可以动态地、灵活地构建各种各样复杂的分布式任务流。

```python
# 通过把一个task输出的`ObjectRef`传递给另一个task，我们定义了两个task的依赖关系。Ray会等待第一个task执行结束之后，再开始执行第二个task。
obj1 = square.remote(2)
obj2 = square.remote(obj1)
assert ray.get(obj2) == 16
```

除了Task和Actor两个基本概念，Ray还提供了一系列高级功能，来帮助开发者更容易地开发分布式应用。这些高级功能包括但不限于：设置Task和Actor所需的资源、Actor生命周期管理、Actor自动故障恢复、自定义调度策略、Python/Java跨语言编程。

### 离线推理Ray

[基于 Ray 的大规模离线推理](https://mp.weixin.qq.com/s/pS5RJCA5O_s6pPcib0JsuQ) Ray 项目是 UC Berkeley 的 RISElab 实验室在 2017 年前后发起的，定位是通用的分布式编程框架——Python-first。**理论上通过 Ray 引擎用户可以轻松地把任何 Python 应用做成分布式**，尤其是机器学习的相关应用，目前 Ray 主攻的一个方向就是机器学习。Ray 的架构分为三层
1. 最下面一层是各种云基础设施，也就是说 Ray 帮用户屏蔽了底层的基础设施，用户拉起一个 Ray Cluster之后就可以立即开始分布式的编程，不用考虑底层的云原生或各种各样的环境；
2. 中间层是 Ray Core 层。这一层是 Ray 提供的核心基础能力，主要是提供了 Low-level 的非常简洁的分布式编程 API。基于这套 API，用户可以非常容易地把现有的 Python 的程序分布式化。值得注意的是，这一层的 API 是 Low-level，没有绑定任何的计算范式，非常通用；
3. 最上层是 Ray 社区基于 Ray Core 层做的丰富的机器学习库，这一层的定位是做机器学习 Pipeline。比如，数据加工读取、模型训练、超参优化、推理，强化学习等，都可以直接使用这些库来完成整个的 Pipeline，这也是 Ray 社区目前主攻的一个方向。

![](/public/upload/machine/ray_arch.jpg)

上图展示的是 Ray Cluster 的基本架构，每一个大框就是一个节点。（这里的节点是一个虚拟的概念，可以是一个物理机，一个 VM 或一个 Linux 的 Docker。比如在 K8s 上，一个节点就是一个 Pod。）

1. Head 节点：是 Ray Cluster 的调度中心，比较核心的组件是 GCS，负责全局存储、调度、作业、状态等，Head节点也有可观测性 Dashboard。
2. Worker 节点：除了 Head 节点之外，其他都是 Worker 节点，承载具体的工作负载。
    1. Raylet：每个节点上面都有一个守护进程 Raylet，它是一个 Local Scheduler，负责 Task 的调度以及 Worker 的管理。
    2. Object Store  组件：每个节点上都有 Object Store 组件，负责节点之间 Object 传输。在整个 Cluster 中每个节点的 Object Store 组件组成一个全局的分布式内存。同时，在单个节点上，Object Store 在多进程之间通过共享内存的方式减少 copy。
3. Driver：当用户向 Ray Cluster 上提交一个 Job，或者用 Notebook 连接的时候，Ray挑选节点来运行 Driver 进行，执行用户代码。作业结束后 Driver 销毁。
4. Worker：是 Ray 中 Task 和 Actor 的载体。

Ray 的Low-level和  High-level API

![](/public/upload/machine/ray_api.jpg)

在部署 Ray 时，开源社区有完整的解决方案 Kuberay 项目。每个 Ray Cluster 由 Head 节点和 Worker 节点组成，每个节点是一份计算资源，可以是物理机、Docker 等等，在 K8s 上即为一个 Pod。启动 Ray Cluster 时，使用 Kuberay 的 Operator 来管理整个生命周期，包括创建和销毁 Cluster 等等。Kuberay 同时也支持自动扩展和水平扩展。Ray Cluster 在内部用于收集负载的 Metrics，并根据 Metrics 决定是否扩充更多的资源，如果需要则触发 Kuberay 拉起新的 Pod 或删除闲置的 Pod。

用户可以通过内部的平台使用 Ray，通过提交 Job 或使用 Notebook 进行交互式编程。平台通过 Kuberay 提供的 YAML 和 Restful API 这两种方式进行操作。