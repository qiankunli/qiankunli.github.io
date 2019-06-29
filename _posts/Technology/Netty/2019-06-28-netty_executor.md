---

layout: post
title: netty中的线程池
category: 技术
tags: Netty
keywords: JAVA netty

---

## 前言（持续更新）

* TOC
{:toc}

## Executor 家族

![](/public/upload/netty/netty_executor.png)

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

### 作业线程的逻辑——取任务并执行

ThreadPoolExecutor 的作业逻辑 由Worker 定义

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

SingleThreadEventExecutor的作业逻辑在 自己的run 方法中，是一个抽象方法，`DefaultEventExecutor.run` 是一个具体的实现

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

### 作业线程的管理

ThreadPoolExecutor 作业线程 由一个HashSet 成员专门持有， 管理/crud大都由调用方线程触发

1. caller thread 提交任务，在特定场景下（核心线程数、最大线程数、任务队列长度），由ThreadFactory 创建新线程（其实还是`new Thread`），
2. caller thread 线程调用shutdown，作业线程在 没有任务或shutdown状态下自动结束

SingleThreadEventExecutor 顾名思义，只有一个线程，还是“租来的”。

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

SingleThreadEventExecutor 通过thread成员 持有了对当前线程的引用

1. caller 线程提交任务时，SingleThreadEventExecutor执行 doStartThread，使用 `executor.execute` 将 `SingleThreadEventExecutor.this.run()` **转包**给了 Executor
2. caller thread 线程调用shutdown， 作业线程在 没有任务或shutdown状态下自动结束


