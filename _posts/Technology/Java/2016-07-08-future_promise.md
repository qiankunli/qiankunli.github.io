---

layout: post
title: Future和Promise
category: 技术
tags: Java
keywords: future

---

## 简介

* TOC
{:toc}

建议先对[不同层面的异步](http://qiankunli.github.io/2017/05/16/async_program.html) 有一点感觉

[JAVA 拾遗--Future 模式与 Promise 模式](https://www.cnkirito.moe/future-and-promise/)

Future jdk源码中的接口定义与描述

    interface Future<T>{
         boolean cancel(boolean mayInterruptIfRunning);
         boolean isCancelled();
         boolean isDone();
         V get() throws InterruptedException, ExecutionException;
         V get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException;
    }
    
1. A Future represents the result of an asynchronous computation. 
2. Future 模式相当于一个占位符，代表一个操作的未来的结果
3. `executorService.submit` 之后立马 `future.get()` 就没什么意思了

        Future<Integer> future = executorService.submit(new Callable<Interger>(){耗时操作})
        timeConsumingOperation();
        System.out.println("计算结果:" + future.get());

4. 自己的理解：凡是异步，必涉及调用方和执行方（通常还有队列），两方必涉及沟通媒介，类似于“句柄” 之类的东东。

## 百花齐放的executor 

![](/public/upload/java/various_executor.png)

[Using as a generic library](https://netty.io/wiki/using-as-a-generic-library.html#wiki-h2-5) 将netty的并发编程库与guava 与jdk8 做了对比，Because **Netty tries to minimize its set of dependencies**, some of its utility classes are similar to those in other popular libraries, such as Guava.

在上图中，netty EventExecutorGroup 的方法返回的是netty 自己实现的`io.netty.util.concurrent.Future extends java.util.concurrent.Future`，guava 则直接一点，ListeningExecutorService 直接返回自己定义的`com.google.common.util.concurrent.ListenableFuture extends java.util.concurrent.Future`

EventExecutorGroup 使用实例（不一定非得netty里才能用）

    EventExecutorGroup group = new DefaultEventExecutorGroup(4); // 4 threads
    Future<?> f = group.submit(new Runnable() { ... });
    f.addListener(new FutureListener<?> {
    public void operationComplete(Future<?> f) {
        ..
    }
    });
    ...

## 百花齐放的future

![](/public/upload/java/various_future.png)

### FutureTask

我们来看一个Futrue的简单使用

    ExecutorService executor = Executors.newFixedThreadPool();
    Future<Integer> future = executor.submit(new MyJob()));
    
跟踪submit方法所属的类，Executors.newFixedThreadPool() ==> ThreadPoolExecutor ==> AbstractExecutorService

    public <T> Future<T> submit(Callable<T> task) {
        if (task == null) throw new NullPointerException();
        RunnableFuture<T> ftask = newTaskFor(task);
        execute(ftask);
        return ftask;
    }
    protected <T> RunnableFuture<T> newTaskFor(Runnable runnable, T value) {
        return new FutureTask<T>(runnable, value);
    }
    
   返回的future是一个FutureTask，FutureTask是`interface RunnableFuture<V> extends Runnable, Future<V>`的实现类。
 

### guava ListenableFuture和AbstractFuture

ListenableFuture的简单使用

    ListeningExecutorService executorService =             MoreExecutors.listeningDecorator(Executors.newCachedThreadPool());
    final ListenableFuture<Integer> listenableFuture = executorService.submit(new MyJob<Integer>());
    // 添加监听事件
    Futures.addCallback(listenableFuture, new FutureCallback() {
        public void onSuccess(Integer result) {
          
        }
        public void onFailure(Throwable thrown) {
          
        }
    });


跟踪submit方法所属的类，ListeningExecutorService ==> AbstractListeningExecutorService ==> AbstractExecutorService

    public abstract class AbstractListeningExecutorService extends AbstractExecutorService{
        protected final <T> ListenableFutureTask<T> newTaskFor(Runnable runnable, T value){
            return ListenableFutureTask.create(runnable, value);
        }
        public <T> ListenableFuture<T> submit(Callable<T> task) {
            return (ListenableFuture)super.submit(task);
        }
    }
    public abstract class AbstractExecutorService implements ExecutorService{
        public <T> Future<T> submit(Callable<T> task) {
            if (task == null) throw new NullPointerException();
            RunnableFuture<T> ftask = newTaskFor(task);
            execute(ftask);
            return ftask;
        }
    }

实际执行的submit方法和上节的submit方法一样一样的，但在submit方法中，上节执行的是`AbstractExecutorService.newTaskFor`返回FutureTask，此处执行的是`AbstractListeningExecutorService.newTaskFor`返回ListenableFutureTask，其实际也是个`java.util.concurrent.FutureTask`。所以一个ListenableFuture具有cancel的能力就不奇怪了。**看来本质上，ListenableFutureTask取消任务的方式还是和FutureTask一样。**

ListenableFuture所具备的addListener方法则是任务挂在一个地方，当run方法执行完毕后，执行这些任务。（不同的guava版本实现代码有很大不同）


## 解决回调地狱——promise

Netty 和 Guava 的扩展都提供了 addListener 这样的接口，用于处理 Callback 调用，但future的Callback容易出现回调地狱的问题，由此衍生出了 Promise 模式来解决这个问题。

但其实 jdk1.8 已经提供了一种更为高级的回调方：CompletableFuture，A Future that may be explicitly completed (setting its value and status), and may be used as a CompletionStage, supporting dependent functions and actions that trigger upon its completion.

[[concurrency-interest] CompletableFuture](http://cs.oswego.edu/pipermail/concurrency-interest/2012-December/010423.html) CompletableFuture 曾经被讨论过以下命名：SettableFuture, FutureValue, Promise, and
probably others.


