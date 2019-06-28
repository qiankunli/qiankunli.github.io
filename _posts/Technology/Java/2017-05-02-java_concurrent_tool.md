---

layout: post
title: java concurrent 工具类
category: 技术
tags: Java
keywords: JAVA concurrent

---

## 前言

## ThreadPoolExecutor

[戏（细）说 Executor 框架线程池任务执行全过程（上）](https://www.infoq.cn/article/executor-framework-thread-pool-task-execution-part-01/)

![](/public/upload/java/various_executor.png)

ThreadPoolExecutor.execute 这个方法看着比较简单，但是线程池什么时候创建新的作业线程来处理任务，什么时候只接收任务不创建作业线程，另外什么时候拒绝任务。线程池的接收任务、维护工作线程的策略都要在其中体现。

![](/public/upload/netty/ThreadPoolExecutor_execute.png)


ThreadPoolExecutor 使得addThread 成为特定条件下的受限操作，`Thread t = threadFactory.newThread(w);t.start();` 不再被随意触发了。

    private Thread addThread(Runnable firstTask) {
        // 为当前接收到的任务 firstTask 创建 Worker
        Worker w = new Worker(firstTask);
        Thread t = threadFactory.newThread(w);
        w.thread = t;
        // 将 Worker 添加到作业集合 HashSet<Worker> workers 中，并启动作业 
        workers.add(w);
        t.start();
        return t;
    }

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

worker线程在受限的条件下创建，其工作内容便是 不停的从workQueue 中取出task 并执行。文中中有一个形象的比喻：**经理给组长提任务，并不管组长是自己做还是分派给下面的小伙伴。经理等着组长report 即可，从小伙伴的视角看，每天的“日常”就是不停的从组长那里领取task，组长视情况给任务排期，实在忙不过来便增加人手**。

## ConcurrentHashMap

[探索 ConcurrentHashMap 高并发性的实现机制](https://www.ibm.com/developerworks/cn/java/java-lo-concurrenthashmap/)

文中提到

1. 用分离锁实现多个线程间的并发写操作
2. 用 HashEntery 对象的不变性来降低读操作对加锁的需求
3. 用 Volatile 变量协调读写线程间的内存可见性

在java中，我们通过cas ==> aqs实现高性能的锁，进而通过减小锁的粒度、读写分离、或final等减少锁的使用。

所以并发变成下的高性能，不只着眼于锁。同时，锁默认顺带解决了内存可见性问题，不用锁时，就要直接处理内存可见性问题。

## ImmutableMap

