---

layout: post
title: Python并发
category: 技术
tags: Python
keywords: Python

---

* TOC
{:toc}

## 前言 

Python是一个解释性语言，边解释边执行，实现这种特性的标准实现叫作 CPython。它分两步来运行 Python 程序：
1. 首先解析源代码文本，并将其编译为字节码（bytecode）
2. 然后采用基于栈的解释器来运行字节码
3. 不断循环这个过程，直到程序结束或者被终止

灵活性有了，但是为了保证程序执行的稳定性，也付出了巨大的代价：引入了 全局解释器锁 GIL（global interpreter lock）。以保证同一时间只有一个字节码在运行，这样就不会因为没用事先编译，而引发资源争夺和状态混乱的问题了。PS：解释器中实施 GIL 与否是个设计决定，JVM 没有 GIL，但是也有锁，只是锁的粒度细小得多，一般不会整个线程给你锁上。当然，像GC、JIT触发之类的情况下还是会有一些锁的操作的。

Python有两大类并行方式：多线程与多进程。由于GIL的存在，这两种方式有着截然不同的特点：

1. 多线程可以直接共享数据，但至多只能用一个CPU核。Python有一个GIL来保证同一时间只能有一个线程来执行，但也保证了任何时间点对应的线程都在做事（所以还是比单线程好一些）。当然，GIL 在较新的python版本已经退出历史舞台了。
2. 多进程可以用多个CPU核，但各进程的数据相互独立（可shared_memory等机制共享数据）

fluent python：导致并发编程困难的因素很多，但最基本的因素：启动进程、线程十分容易，关键是如何跟踪进程或线程。调用一个函数，发出调用的代码开始阻塞，直到函数返回。因此你知道函数什么时候执行完毕，而且能轻松得到函数的返回值。如果函数抛出异常，则把函数放在try/except里，捕获错误。这些熟悉的概念在你启动线程/进程后都不可用了：判断程序的状态很难，调度程序随时可能会中断线程，因此一定不能忘记持有锁，保护程序的关键部分。无法轻松得知操作何时结束，若想获取结果或捕获错误，则需要设置某种通信信道，例如消息队列。如果不需要了，如退出呢？怎么样退出才能不中断作业，避免留下未处理完毕的数据和未释放的资源呢？解决这些问题通常也涉及到消息队列。

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

在python中，协调线程的信号机制，使用threading.Event类最简单，threading.Event有一个内部bool标志，一开始是False，调用Event.set() 可以将标志设为True。这个标志为False时，在一个线程中调用Event.wait() 该线程将被阻塞，直到另一个线程调用Event.set()，致使Event.wait() 返回True。PS： threading.Event 对应有multiprocessing.Event,asyncio.Event。PS：感觉类似于barrier


## 进程

multiprocess模块的完全模仿了threading模块的接口，二者在使用层面，有很大的相似性。创建multiprocessing.Process实例后，一个全新的Python解释器以子进程的形式在后台启动。


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

## 协程

协程是非抢占式的多任务子例程的概括，可以允许有多个入口点在例程中确定的位置来控制程序的暂停与恢复执行。例程是什么？编程语言定义的可被调用的代码段，为了完成某个特定功能而封装在一起的一系列指令。一般的编程语言都用称为函数或方法的代码结构来体现。所有对函数的调用都遵循一个流程：进入函数，从头开始，直到 return 结尾，返回父函数调用处。这种调用模型我们再熟悉不过了。**在 asyncio 中，每个线程不再只有一个堆栈**。相反，每个线程都有一个被称为事件循环（Event Loop）的对象。Event Loop 中包含一个称为任务（Task）的对象列表。每个 Task 维护一个堆栈，以及它自己的 Instruction Pointer。

async 修饰词声明异步函数，而调用异步函数，我们便可得到一个协程对象（coroutine object）。

```python
import asyncio
import time

async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)

async def main():
    print(f"started at {time.strftime('%X')}")
    await say_after(1, 'hello')
    await say_after(2, 'world')
    print(f"finished at {time.strftime('%X')}")

asyncio.run(main())
# 
# started at 14:07:43
# hello
# world
# finished at 14:07:46
```

按道理应该是间隔了2s，可是实际间隔3s。Python中的异步执行模式依赖于Event Loop，在等待的间隙中需要从Event Loop中找其它可以运行的程序，await关键字就是将coroutine转化成一个task加入到Event Loop中去，而执行第一个say_after的时候，第二个say_after并没有加入到Event Loop中去，所以在第一个say_after等待的时候无法去执行第二个say_after，最终导致的结果就是程序运行了3s，并没有达到异步的效果。有两种方式解决这个问题
1. 提前将两个say_after加入到eventloop中
    ```python
    async def main():
    task1 = asyncio.create_task(say_after(1, 'hello'))
    task2 = asyncio.create_task(say_after(2, 'world'))
    print(f"started at {time.strftime('%X')}")
    await task1
    await task2
    print(f"finished at {time.strftime('%X')}")
    ```
2. 使用gather方法
    ```python
    async def main():
    print(f"started at {time.strftime('%X')}")
    await asyncio.gather(
        say_after(1, 'hello'),
        say_after(2, 'world')
    )
    print(f"finished at {time.strftime('%X')}")
    ```

### 异步任务之间的数据交互

异步任务之间的数据交互是完全可能的，且在多任务异步程序中非常常见。
1. 使用共享变量，如果异步任务运行在同一线程中（通常情况下是这样），可以使用全局变量或闭包变量共享数据。
2. asyncio.Queue，支持在生产者和消费者之间安全地传递消息。PS：有些go channel的味道
3. 使用 Futures  和 Events，适合于任务间的事件通知和状态同步。Future是对协程的封装(提供了取消/回调等)，代表一个未来对象，执行结束后会把最终结果设置到Future对象上。

没有可以从外部终止线程的api，必须发送信号。`class asyncio.Event`可用于通知多个 asyncio 任务某个事件已发生。asyncio.Event管理一个内部标志，该标志可以使用set() 方法设置为true，并使用clear() 方法重置为false。 wait() 方法阻塞，直到标志设置为 true 。该标志最初设置为false。

### 协程实现原理

异步编程是以进程、线程、协程、函数/方法作为执行任务程序的基本单位，结合回调、事件循环、信号量等机制，以提高程序整体执行效率和并发能力的编程方式。**协程即可以挂起并移交控制权的函数**，协程拥有自己的寄存器上下文和栈（每个线程不再只有一个堆栈）。协程调度切换时，将寄存器上下文和栈保存到其他地方，在切回来时，恢复先前保存的寄存器上下文和栈。因此它在执行过程中可以中断，转而执行其他的协程，在适当的时候再回来继续执行。如果熟知了python生成器，还可以将协程理解为生成器+调度策略，生成器中的yield关键字，就可以让生成器函数发生中断，而调度策略，可以驱动着协程的执行和恢复。


```python
# define a generator
def generator():
	for i in range(10):
		yield i
# create the generator
gen = generator()
# step the generator
result = next(gen)
# 更常见的用法
for result in generator()
    print(result)
```
异步生成器是使用 yield 表达式的协程，实现了 anext() 方法，可以与 async for 表达式一起使用。
```python
# define an asynchronous generator
async def async_generator():
	for i in range(10)
        await asyncio.sleep(1)
		yield i
# create the iterator
it = async_generator()
# get an awaitable for one step of the generator
awaitable = anext(gen)
# execute the one step of the generator and get the result
result = await awaitable
# 更常见的用法
# traverse an asynchronous generator
async for result in async_generator():
	print(result)
```
使用asyncio 的python 代码只有一个执行流，除非显式启动额外的线程或进程，这意味着任何时间点上都只有一个协程在执行，若想实现并发，则要把控制权由一个协程传给另一个协程（协程通过await委托给另一个协程，类似于生成器通过yield from委托给另一个生成器），协程通过await显式让出控制权，把控制权还给调度程序，即`await coroutine`。
```python
class Awaitable(metaclass=ABCMeta):
    __slots__ = ()
    @abstractmethod
    def __await__(self):
        yield
class Coroutine(Awaitable):
    __slots__ = ()
    @abstractmethod
    def send(self, value):
        """Send a value into the coroutine.
        Return next yielded value or raise StopIteration.
        """
        raise StopIteration
    @abstractmethod
    def throw(self, typ, val=None, tb=None):
        """Raise an exception in the coroutine.
        Return next yielded value or raise StopIteration.
        """
        if val is None:
            if tb is None:
                raise typ
            val = typ()
        if tb is not None:
            val = val.with_traceback(tb)
        raise val
    def close(self):
        """Raise GeneratorExit inside coroutine.
        """
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            pass
        else:
            raise RuntimeError("coroutine ignored GeneratorExit")
```


生成器与协程的实现。模组、函数、类都是代码块，编译器会为每个代码块创建代码对象，代码对象描述了代码块的具体内容，包括代码对应的字节码、常数、变量名和其它相关信息。而函数则通过函数对象保存其代码对象、函数名、默认参数以及 `__doc__`属性等信息。

生成器函数也是函数，只是其代码对象带有 CO_GENERATOR 标记。用户调用生成器函数时，Python 会检查此标记，如果存在，则不执行函数，而是返回生成器对象。类似地，原生协程函数也是函数，只是代码对象带有 CO_COROUTINE 标记，看到此标记时，Python 会直接返回原生协程对象。执行函数时，Python 会创建函数帧/Frame对象，用于保存代码对象/CodeObject的执行状态，包括代码对象本身以及局部变量的值、全局变量与内置变量字典的引用、值栈、指令指针等等。
1. 生成器对象保存着生成器函数的函数帧，以及一些辅助数据，如生成器名称、运行标记等。关键的区别在于，运行普通函数时，每次创建一个新的函数帧，而运行生成器时，使用的是同一个函数帧，因而能保存其运行状态。
2. 协程本质上就是一个类型不同的生成器对象。区别在于 generator 类实现了 `__iter__()` 与 `__next__()` 方法，coroutine 类实现的是 `__await__()` 方法


## GIL

GIL，是最流行的 Python 解释器 CPython 中的一个技术术语。它的意思是全局解释器锁，每一个 Python 线程，在 CPython 解释器中执行时，都会锁住GIL，阻止别的线程执行，同样的，每一个线程执行完一段后，会释放 GIL，以允许别的线程开始利用资源（CPython 中还有另一个机制，叫做 check_interval，默认5ms，意思是 CPython 解释器会去轮询检查线程 GIL 的锁住情况。每隔一段时间，Python 解释器就会强制当前线程去释放 GIL，这样别的线程才能获取GIL有执行的机会）。CPython轮流执行 Python 线程。这样一来，用户看到的就是“伪并行”——Python 线程在交错执行，来模拟真正并行的线程。python标准库发起的系统调用均可以释放GIL，包括磁盘io、网络io、time.sleep等，因此GIL 对与网路编程的影响较小。

为什么 CPython 需要 GIL 呢？这其实和 CPython 的实现有关。CPython 使用引用计数来管理内存，所有 Python 脚本中创建的实例，都会有一个引用计数，来记录有多少个指针指向它。当引用计数只有 0 时，则会自动释放内存。如果有两个 Python 线程同时引用了 a，就会造成引用计数的 race condition，引用计数可能最终只增加 1，这样就会造成内存被污染。因为第一个线程结束时，会把引用计数减少 1，这时可能达到条件释放内存，当第二个线程再试图访问 a 时，就找不到有效的内存了。所以说，CPython 引进 GIL 其实主要就是这么两个原因：
1. 设计者为了规避类似于内存管理这样的复杂的竞争风险问题（race condition）；
2. 因为 CPython 大量使用 C 语言库，但大部分 C 语言库都不是原生线程安全的（线程安全会降低性能和增加复杂度）。

绕过 GIL 的大致思路有这么两种：
1. 绕过 CPython，使用 JPython（Java 实现的 Python 解释器）等别的实现；
2. 把关键性能代码，放到别的语言（一般是 C++）中实现。

GIL 的存在与 Python 支持多线程并不矛盾。前面我们讲过，GIL 是指同一时刻，程序只能有一个线程运行；而 Python 中的多线程，是指多个线程交替执行，造成一个“伪并行”的结果，但是具体到某一时刻，仍然只有 1 个线程在运行，并不是真正的多线程并行。

## ray 并发库

ray core 可以作为 multiprocess 模块的平替，ChatGPT 本身在各种 dirty-work 里面也大量应用了 Ray，请参考另一篇专门写ray的博客。

xoscar: Python actor framework for heterogeneous computing. PS：在python 里一个 对象跟并行体是一体的，Java里对象是对象，Thread 是Thread，所以要么XXThread 里塞业务逻辑，要么是对象里塞Thread/ExecutorPool，将这个对象的部分方法实现为异步的。但在python 里，只要 继承特定的class（会带上装饰器），则其所有方法均可以异步、远程调用。 

```python
# Define an actor
import xoscar as xo
# stateful actor, for stateless actor, inherit from xo.StatelessActor
class MyActor(xo.Actor):
    def __init__(self, *args, **kwargs):
        pass
    async def __post_create__(self):
        # called after created
        pass
    async def __pre_destroy__(self):
        # called before destroy
        pass
    def method_a(self, arg_1, arg_2, **kw_1):  # user-defined function
        pass
    async def method_b(self, arg_1, arg_2, **kw_1):  # user-defined async function
        pass
# Create an actor
import xoscar as xo
actor_ref = await xo.create_actor(
    MyActor, 1, 2, a=1, b=2,
    address='<ip>:<port>', uid='UniqueActorName')
# Invoke a method
await actor_ref.method_a.send(1, 2, a=1, b=2)
```