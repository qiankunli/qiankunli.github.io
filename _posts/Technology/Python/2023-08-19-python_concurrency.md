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

PS： 但使用体验上都很类似

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

## 协程

异步编程是以进程、线程、协程、函数/方法作为执行任务程序的基本单位，结合回调、事件循环、信号量等机制，以提高程序整体执行效率和并发能力的编程方式。**协程即可以挂起并移交控制权的函数**，协程拥有自己的寄存器上下文和栈（每个线程不再只有一个堆栈）。协程调度切换时，将寄存器上下文和栈保存到其他地方，在切回来时，恢复先前保存的寄存器上下文和栈。因此它在执行过程中可以中断，转而执行其他的协程，在适当的时候再回来继续执行。如果熟知了python生成器，还可以将协程理解为生成器+调度策略，生成器中的yield关键字，就可以让生成器函数发生中断，而调度策略，可以驱动着协程的执行和恢复。

```python
# 同步函数，同步函数本身是一个 Callable 对象，调用这个函数的时候，函数体内的代码被执行。
def function():
    return 1
# 生成器函数
def generator():
    yield 1
# 异步函数，也是一个 Callable 对象实例，调用这个函数的时候，函数体内的代码不会被执行。相反，Python 创建了一个 Coroutine 对象实例，并将其分配给返回值。
async def async_function():
    # 当 time.sleep() 被调用，整个程序都会暂停，什么都做不了。asyncio.sleep()是异步的，或者说是非阻塞的。
    await asyncio.sleep(3)
    return 1
# 异步生成器，调用方一般await for 生成器
async def async_generator():
    yield 1
print(async_function())
# <coroutine object async_function at 0x102ff67d8>
```

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

###  Task

调用异步函数并不能执行函数，**异步函数就不能由我们自己直接执行**（这也是函数与协程的区别），异步代码是以 Task 的形式去运行，被 Event Loop 管理和调度的。`result = async_function()` 协程对象 result 虽然生成了，但是还没有运行，要使代码块实际运行，需要使用 asyncio 提供的其他工具。
1. 最常见的是 await 关键字。当我们await一个Coroutine，这个异步函数就会被提交给asyncio底层，然后asyncio就会开始执行这个函数。
    1. 如果一个对象可以在 await 语句中使用，那么它就是 可等待 对象。许多 asyncio API 都被设计为接受可等待对象。可等待 对象有三种主要类型: 协程, Task  和 Future.
2. `result = asyncio.run(async_function())`。 这是程序入口处专用的。run里边就不能再run了。
3. `asyncio.create_task(async_function())`，创建一个 Task 对象实例，异步函数被包在了Task，Task 对象实例立即被运行。这一点与await不同，**如果我们有一个Coroutine，我们必须await它，才能把相应的异步函数提交给asyncio（然后才开始运行）**。当然了，虽然Task不await也能执行，但我们通常还是需要await各个Task。因为这可以确保它们执行完成并收集运行结果，不然我们就得用Future。
    1. 任务可以便捷和安全地取消。 当任务被取消时，asyncio.CancelledError 将在遇到机会时在任务中被引发。
    2. 任务组合并了一套用于等待分组中所有任务完成的方便可靠方式的任务创建 API。class asyncio.TaskGroup，持有一个任务分组的 异步上下文管理器。 可以使用 create_task() 将任务添加到分组中。 async with 语句将等待分组中的所有任务结束。 
。当该上下文管理器退出时所有任务都将被等待。当首次有任何属于分组的任务因 asyncio.CancelledError 以外的异常而失败时，分组中的剩余任务将被取消。一旦所有任务被完成，如果有任何任务因 asyncio.CancelledError 以外的异常而失败，这些异常会被组合在 ExceptionGroup 或 BaseExceptionGroup 中并将随后引发。PS： 有点像go中的waitgroup，启动多个任务，发生异常、等一个任务执行完、等待所有任务都完成、限制某个任务执行过程都提供了接口。

`result = await async_function()` 和普通的同步函数没有任何区别，主要原因是：这里其实没有将协程放到 Event Loop 中，这用到了asyncio.create_task()。

```python
async def main():
    task1 = asyncio.create_task(async_function())
    task2 = asyncio.create_task(async_function())
    # await task 等待 Task 执行结束
    print(await task1)
    print(await task2)
asyncio.run(main())
```
Task 对象有以下api，非常直观。
```
done() # 如果 Task 对象 已完成 则返回 True。
result() # 返回 Task 的结果。
exception() # 返回 Task 对象的异常。
add_done_callback(callback, *, context=None) # 添加一个回调，将在 Task 对象 完成 时被运行。
get_stack(*, limit=None) # 返回此 Task 对象的栈框架列表。
print_stack(*, limit=None, file=None) # 打印此 Task 对象的栈或回溯。
cancel(msg=None) # 请求取消 Task 对象。
cancelled() # 如果 Task 对象 被取消 则返回 True。
uncancel() # 递减对此任务的取消请求计数。返回剩余的取消请求数量。
```
### 事件循环

EventLoop是用于在单个线程中执行协程的环境。EventLoop是异步程序的核心。事件循环，顾名思义，就是一个循环。它管理一个任务列表（协同程序）并尝试在循环的每次迭代中按顺序推进每个任务，以及执行其他任务，如执行回调和处理 I/O。“asyncio”模块提供了访问事件循环并与之交互的功能，这不是典型应用程序开发所必需的，asyncio 模块提供了一个用于访问当前事件循环对象的低级 API，以及一套可用于与事件循环交互的方法。

Asyncio 和其他 Python 程序一样，是单线程的，它只有一个主线程，Future 是一个可以被等待的对象，Task 在 Future 的基础上加入了一个 coroutine。他们都是 asyncio 的核心，但是他们都需要一个 EventLoop 来运行。任务只有两个状态：一是预备状态；二是等待状态。event loop 会维护两个任务列表，分别对应这两种状态；并且选取预备状态的一个任务（具体选取哪个任务，和其等待的时间长短、占用的资源等等相关），使其运行，一直到这个任务把控制权交还给 event loop 为止。当任务把控制权交还给 event loop 时，event loop 会根据其是否完成，把任务放到预备或等待状态的列表，然后遍历等待状态列表的任务，查看他们是否完成。如果完成，则将其放到预备状态的列表；这样，当所有任务被重新放置在合适的列表后，新一轮的循环又开始了：event loop 继续从预备状态的列表中选取一个任务使其执行…如此周而复始，直到所有任务完成。

Python 3.4 加入了asyncio 库，使得Python有了支持异步IO的官方库。这个库，底层是事件循环（EventLoop），上层是协程和任务。每个线程都有一个被称为事件循环（Event Loop）的对象（可以理解成 while True 循环），Event Loop 中包含一个称为任务（Task）的对象列表。每个 Task 维护一个堆栈，以及它自己的 Instruction Pointer。在任意时刻，Event Loop 只能有一个 Task 实际执行，毕竟 CPU 在某一时刻只能做一件事，当 Task 遇到需要等待的事情，比如 IO bound 应用需要等待数据到达。此时，Task 中的代码不再等待，而是让出控制权。Event Loop 暂停正在运行的 Task。未来的某个时刻，当这个 Task 所等待的事情已经成熟，Event Loop 将再次唤醒这个 Task。Task 让出控制权后，Event Loop 唤醒某个休眠的 Task，并将这个新唤醒的 Task 设置为当前执行的 Task。线程会遍历各个任务，在前面任务IO等待的过程中，进行后面任务的CPU计算，**使得 CPU 闲置的时间更少**。循环往复，直到所有任务执行完毕。Python在3.5版本中引入了关于协程的语法糖async和await，Python 3.7 又进行了优化，**把API分组为高层级API和低层级API**，把EventLoop相关的API归入到低层级API。

多线程还是 Asyncio？如果是 I/O bound，并且 I/O 操作很慢，需要很多任务 / 线程协同实现，那么使用 Asyncio 更合适。如果是 I/O bound，但是 I/O 操作很快，只需要有限数量的任务 / 线程，那么使用多线程就可以了。如果是 CPU bound，则需要使用多进程来提高程序运行效率。I/O 操作 heavy 的场景下，Asyncio 比多线程的运行效率更高。因为 Asyncio 内部任务切换的损耗，远比线程切换的损耗要小；并且 Asyncio 可以开启的任务数量，也比多线程中的线程数量多得多。但需要注意的是，很多情况下，使用 Asyncio 需要特定第三方库的支持。

使用 asyncio 并不是将代码转换成多线程，它不会导致多条Python指令同时执行，也不会以任何方式让你避开所谓的全局解释器锁（Global Interpreter Lock，GIL）。有些应用受 IO 速度的限制，即使 CPU 速度再快，也无法充分发挥 CPU 的性能。这些应用花费大量时间从存储或网络设备读写数据，往往需要等待数据到达后才能进行计算，在等待期间，CPU 什么都做不了。asyncio 的目的就是为了给 CPU 安排更多的工作：当前单线程代码正在等待某个事情发生时，另一段代码可以接管并使用 CPU，以充分利用 CPU 的计算性能。**asyncio 更多是关于更有效地使用单核，而不是如何使用多核**。python协程单线程内切换，适用于IO密集型程序中，可以最大化IO多路复用的效果。**协程间完全同步**，不会并行，不需要考虑数据安全。PS：与Go协程不同的地方。

### 事件

`class asyncio.Event`可用于通知多个 asyncio 任务某个事件已发生。asyncio.Event管理一个内部标志，该标志可以使用set() 方法设置为true，并使用clear() 方法重置为false。 wait() 方法阻塞，直到标志设置为 true 。该标志最初设置为false。

### 实现原理

生成器与协程的实现。模组、函数、类都是代码块，编译器会为每个代码块创建代码对象，代码对象描述了代码块的具体内容，包括代码对应的字节码、常数、变量名和其它相关信息。而函数则通过函数对象保存其代码对象、函数名、默认参数以及 `__doc__`属性等信息。生成器函数也是函数，只是其代码对象带有 CO_GENERATOR 标记。用户调用生成器函数时，Python 会检查此标记，如果存在，则不执行函数，而是返回生成器对象。类似地，原生协程函数也是函数，只是代码对象带有 CO_COROUTINE 标记，看到此标记时，Python 会直接返回原生协程对象。执行函数时，Python 会创建函数帧对象，用于保存代码对象的执行状态，包括代码对象本身以及局部变量的值、全局变量与内置变量字典的引用、值栈、指令指针等等。
1. 生成器对象保存着生成器函数的函数帧，以及一些辅助数据，如生成器名称、运行标记等。关键的区别在于，运行普通函数时，每次创建一个新的函数帧，而运行生成器时，使用的是同一个函数帧，因而能保存其运行状态。
2. 协程本质上就是一个类型不同的生成器对象。区别在于 generator 类实现了 `__iter__()` 与 `__next__()` 方法，coroutine 类实现的是 `__await__()` 方法


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

## GIL

GIL，是最流行的 Python 解释器 CPython 中的一个技术术语。它的意思是全局解释器锁，每一个 Python 线程，在 CPython 解释器中执行时，都会锁住GIL，阻止别的线程执行，同样的，每一个线程执行完一段后，会释放 GIL，以允许别的线程开始利用资源（CPython 中还有另一个机制，叫做 check_interval，意思是 CPython 解释器会去轮询检查线程 GIL 的锁住情况。每隔一段时间，Python 解释器就会强制当前线程去释放 GIL，这样别的线程才能有执行的机会）。CPython轮流执行 Python 线程。这样一来，用户看到的就是“伪并行”——Python 线程在交错执行，来模拟真正并行的线程。

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