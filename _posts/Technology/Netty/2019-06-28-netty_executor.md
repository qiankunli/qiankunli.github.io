---

layout: post
title: netty中的线程池
category: 技术
tags: Netty
keywords: JAVA netty

---

## 前言（持续更新）

[java concurrent 工具类](http://qiankunli.github.io/2017/05/02/java_concurrent_tool.html)

![](/public/upload/netty/netty_executor.png)

## SingleThreadEventExecutor

EventExecutorGroup 继承了ScheduledExecutorService接口，对原来的ExecutorService的关闭接口提供了增强，提供了优雅的关闭接口。从接口名称上可以看出它是对多个EventExecutor的集合，提供了对多个EventExecutor的迭代访问接口。 

SingleThreadEventExecutor 作为一个Executor，实现Executor.execute 方法，首先具备Executor 的一般特点

1. 会被各种调用方多线程调用 “提交”task
2. 有一个队列保存 来不及执行的task
3. 超出队列容量了，有拒绝策略等

![](/public/upload/netty/SingleThreadEventExecutor_execute.png)

![](/public/upload/netty/ThreadPoolExecutor_execute.png)

## SingleThreadEventExecutor 为什么要传入一个executor

1. SingleThreadEventExecutor 有一个 thread 标记了其 执行任务的thread
2. SingleThreadEventExecutor 传入了一个executor，但这个executor 不是直接 执行SingleThreadEventExecutor.execute 提交的任务。


通过线程池控制代码的并发量