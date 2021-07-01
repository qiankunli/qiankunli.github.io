---

layout: post
title: netty中的线程池
category: 技术
tags: Netty
keywords: JAVA netty

---

## 前言

* TOC
{:toc}

Java 的标准类库，由于其基础性、通用性的定位，往往过于关注技术模型上的抽象，而不是从一线应用开发者的角度去思考。java引入concurrent 包的一个重要原因就是，应用开发者使用 Thread API 比较痛苦，需要操心的不仅仅是业务逻辑，而且还要自己负责将其映射到 Thread 模型上。Java NIO 的设计也有类似的特点，开发者需要深入掌握线程、IO、网络等相关概念，学习路径很长，很容易导致代码复杂、晦涩，即使是有经验的工程师，也难以快速地写出高可靠性的实现。Netty 的设计强调了 “Separation Of Concerns”，通过精巧设计的事件机制，将业务逻辑和无关技术逻辑进行隔离，并通过各种方便的抽象，一定程度上填补了了基础平台和业务开发之间的鸿沟，更有利于在应用开发中普及业界的最佳实践。PS： 也就是说，**有了Executor，怎么操作Thread 你不用管了。 netty 的api 都提供异步接口，你只需要构造入口对象时传入Executor，怎么用Executor 就不用管了**。

## Executor 家族

![](/public/upload/netty/netty_executor.png)

[异步执行抽象——Executor与Future](http://qiankunli.github.io/2016/07/08/executor_future.html)

对Executor 的扩展 主要体现在几个方面

1. 规范 作业线程的管理，比如ExecutorService
2. 提供 更丰富的 异步处理返回值 ，比如guava 的ListeningExecutorService
3. 优化特定场景，比如netty的SingleThreadEventExecutor，只有一个作业线程
4. 针对特定业务场景，更改作业线程的处理逻辑。比如netty的EventLoopGroup，其作业线程逻辑为 io + task ，并可以根据ioRatio 调整io 与task的cpu 占比。

## SingleThreadEventExecutor

EventExecutorGroup 继承了ScheduledExecutorService接口，对原来的ExecutorService的关闭接口提供了增强，提供了优雅的关闭接口。从接口名称上可以看出它是对多个EventExecutor的集合，提供了对多个EventExecutor的迭代访问接口。 

SingleThreadEventExecutor 作为一个Executor，实现Executor.execute 方法，首先具备Executor 的一般特点

1. 会被各种调用方并发调用 “提交”task
2. 有一个队列保存 来不及执行的task
3. 超出队列容量了，有拒绝策略等
4. 作业线程负责不停地从队列中取出任务并执行

![](/public/upload/netty/ThreadPoolExecutor_execute_sequence.png)

## 提交任务

### ThreadPoolExecutor 

![](/public/upload/netty/ThreadPoolExecutor_execute.png)

### SingleThreadEventExecutor 

![](/public/upload/netty/SingleThreadEventExecutor_execute.png)

## 作业线程



Runnable + Thread 实现了 logic 和 runner 的分离，runner 又进一步扩展为 executor，与ThreadPerTaskExecutor 中的线程不同，线程复用之后，SingleThreadEventExecutor/ThreadPoolExecutor 中的线程必须改造为拥有task处理逻辑的作业线程。

[从操作系统层面分析Java IO演进之路](https://mp.weixin.qq.com/s/KgJFyEmZApF7l5UUJeWf8Q)work的线程数量，取决于初始化时创建了几个epoll，worker的复用本质上是epoll的复用。work之间为什么要独立使用epoll？为什么不共享？
1. 为了避免各个worker之间发生争抢连接处理，netty直接做了物理隔离，避免竞争。各个worker只负责处理自己管理的连接，并且后续该worker中的每个client的读写操作完全由 该线程单独处理，天然避免了资源竞争，避免了锁。
2. worker单线程，性能考虑：worker不仅仅要epoll_wait，还是处理read、write逻辑，加入worker处理了过多的连接，势必造成这部分消耗时间片过多，来不及处理更多连接，性能下降。

### 作业线程的逻辑——取任务并执行

ThreadPoolExecutor 的作业逻辑 由Worker 定义

```java
private final class Worker implements Runnable{
    public void run() {
        try {
            Runnable task = firstTask;
            // 循环从线程池的任务队列获取任务 
            while (task != null || (task = getTask()!= null) {
                // 执行任务 
                runTask(task);
                task = null;
            }
        } finally {
            workerDone(this);
        }
    }
    private void runTask(Runnable task) {         
            task.run();
    }
}
```

SingleThreadEventExecutor的作业逻辑在 自己的run 方法中，是一个抽象方法，`DefaultEventExecutor.run` 是一个具体的实现

```java
protected void run() {
    for (;;) {
        Runnable task = takeTask();
        if (task != null) {
            task.run();
            updateLastExecutionTime();
        }
        if (confirmShutdown()) {
            break;
        }
    }
}
```

### 作业线程的管理

ThreadPoolExecutor 作业线程 由一个HashSet 成员专门持有， 管理/crud大都由调用方线程触发

1. caller thread 提交任务，在特定场景下（核心线程数、最大线程数、任务队列长度），由ThreadFactory 创建新线程（其实还是`new Thread`），
2. caller thread 线程调用shutdown，作业线程在 没有任务或shutdown状态下自动结束

SingleThreadEventExecutor 顾名思义，只有一个线程，还是“租来的”。

```java
private void doStartThread() {
    executor.execute(new Runnable() {
        @Override
        public void run() {
            thread = Thread.currentThread();
            try {
                SingleThreadEventExecutor.this.run();
            } catch (Throwable t) {
                logger.warn("Unexpected exception from an event executor: ", t);
            } finally {
                // Run all remaining tasks and shutdown hooks.
                for (;;) {
                    if (confirmShutdown()) {
                        break;
                    }
                }
            }
        }
    });
}
```

SingleThreadEventExecutor 通过thread成员 持有了对当前线程的引用

1. caller 线程提交任务时，SingleThreadEventExecutor执行 doStartThread，使用 `executor.execute` 将 `SingleThreadEventExecutor.this.run()` **转包**给了 Executor
2. caller thread 线程调用shutdown， 作业线程在 没有任务或shutdown状态下自动结束

## EventLoopGroup

1. EventExecutorGroup 首先具备Executor 作为任务处理器的职能，其execute逻辑通过next()移交给EventExecutor
2. EventLoopGroup register(channel) 方法反应了其与io 处理的关联
3. SingleThreadEventExecutor 封装了单线程场景下的作业处理，并将作业处理逻辑暴露给run 方法
4. NioEventLoop 实现了SingleThreadEventExecutor的run 方法 ，自定义作业处理逻辑，除了taskQueue，在作业处理逻辑中，塞入了 io 的处理逻辑， 并可以根据ioRatio调整io 与 task任务的cpu 占比。

## inEventLoop()

![](/public/upload/netty/channel_write.png)

1. EventExecutor 基本不会作为外部对象直接使用
2. 一个channel的 写操作本质是对 其绑定的 Unsafe.OutputBuffer 的写入，且Unsafe.OutputBuffer 非线程安全，所以只能由一个线程来操作
3. channel.write时，要判断 caller 线程 是否为绑定线程，EventExecutor要提供inEventLoop(Thread) 即 inEventLoop(caller) 方法


channel.write 根据inEventLoop 来判断 caller 线程的性质，以判断是否 可以安全写入outboundBuffer

```java
public abstract class AbstractChannel{
    private volatile EventLoop eventLoop;
    private final Unsafe unsafe;
    protected abstract class AbstractUnsafe implements Unsafe {
        private volatile ChannelOutboundBuffer outboundBuffer
    }
}
```

类似于

```java
function write(msg){
    if(Thread.currentThread() == eventLoop.getThread()){
        write buffer
    }else{
        eventLoop.execute(task(msg));
    }
}
```

![](/public/upload/java/use_executor.png)

EventLoop作为AbstractChannel的成员，承接AbstractChannel 的核心逻辑，支持了AbstractChannel 对外提供异步接口。


