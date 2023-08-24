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

1. 多线程可以直接共享数据，但至多只能用一个CPU核。Python有一个GIL来保证同一时间只能有一个线程来执行，当然，GIL 在较新的python版本已经退出历史舞台了。
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

## 协程（未完成）

Python还提供了异步IO的模块 asyncio，在单线程内实现并发，asyncio的核心原理，就是一个event-loop（可以理解成 while True 循环）。在event-loop的每一次循环中，线程会遍历各个任务，在前面任务IO等待的过程中，进行后面任务的CPU计算。循环往复，直到所有任务执行完毕。

## ray 并发库
