---

layout: post
title: 异步编程
category: 技术
tags: Architecture
keywords: 异步

---

## 简介

如何定义一个系统的性能，参见[性能调优攻略](http://coolshell.cn/articles/7490.html)

[Servlet 3.0 实战：异步 Servlet 与 Comet 风格应用程序
](https://www.ibm.com/developerworks/cn/java/j-lo-comet/)

本文从以下角度阐述

1. 为什么异步性能好一点
2. 框架是如何提供异步接口的
3. 上层业务如何使用这些异步接口来组织逻辑

## 为什么异步web可以提高吞吐量

首先，异步不是突然出现一个异步就牛了，而是一系列手段加持的结果


1. 长连接
2. Web 线程不需要同步的、一对一的处理客户端请求，能做到一个 Web 线程处理多个客户端请求。同步的实质是线程数限制了连接数，假设tomcat有100个线程，某个请求比较耗时，那么第101个请求就无法创建连接。

举个例子，一个请求建立连接要1s，请求处理要1s，每秒到达100个请求。

||同步|异步|
|---|---|---|
|第1s|100个请求连接成功|100个请求连接成功|
|第2s|100个请求处理成功|100个请求连接成功，100个请求处理成功|
|第3s|100个请求连接成功|100个请求连接成功，100个请求处理成功|
|第4s|100个请求处理成功|100个请求连接成功，100个请求处理成功|
|4s小计|处理200个请求，50qps|100个请求连接成功，300个请求处理成功，65qps|


## 异步编程的在各个层面的支持

### 操作系统和语言层面

强烈推荐这篇文章[有关异步编程框架的讨论](http://www.jianshu.com/p/c4e63927ead2)，基本要点：

1. 操作系统就像一个大型的中断处理库。cpu响应中断，运行中断处理程序。操作系统为了最大化利用cpu，一个进程在等待I/O时，另一个进程可以利用CPU进行计算。因此，进行阻塞操作时，操作系统会让进程/线程让出对cpu的控制权。

		while(true){
			执行任务 // 因为时间片限定，执行时间不长
			执行任务期间，有中断{
				保存上下文
				执行中断处理程序
			}
			任务完毕，也没中断，等待；
		}

2. 对于业务来说，肯定不想因为一个阻塞操作就放弃cpu，即便用上多进程（线程），线程数毕竟是有上限。比如一个webserver，一个请求需要读取数据库，线程阻塞了，但线程还可以去服务其它请求啊。
3. 如果线程直接执行的代码中，**调用了可能阻塞的系统调用，失去cpu就在所难免。（这或许同步异步的本质点）**

	* golang直接实现了一个调度器，碰上阻塞操作（重写了阻塞操作的逻辑，不会直接调用系统调用），挂起的是goroutine，实际的物理线程执行下调度程序，找下一个goroutine接着跑。
	* python twisted, Java Netty, Nodejs libuv 这些框架没有深入到语言层，没办法推翻重来（r/w或者nio的select操作还是得调用系统调用），而是做一个自己的事件驱动引擎。
4. 事件驱动引擎。业务操作分为阻塞操作（差不多就是io操作）和非阻塞操作

	* 从线程的角度说，代码逻辑成了，监听io（事先注册到selector的io事件，确切的说是select 系统调用），有io就处理io（从selector key中拿到必要数据，read、write等系统调用在nio中设置为不阻塞），没io就处理任务队列里的任务。
	* 从代码角度说，io操作变成了注册selector，非io操作则添加到任务队列。

	也即是说，虽然不能防止io操作的阻塞，但通过重构线程的整体逻辑，实现自由控制阻塞和非阻塞操作的事件。（netty eventloop中有io ratio）
	
**文章最后小结提到：其实从某种程度来说，异步框架是程序试图跳出操作系统界定的同步模型，重新虚拟出一套执行机制，让框架的使用者看起来像一个异步模型。另外通过把很多依赖操作系统实现的笨重功能换到程序内部使用更轻量级的实现。**

## 框架层面——以netty 为例

下文摘自《netty in action》对channel的介绍

a ChannelFuture is returned as part of an i/o operations.Here,`connect()`will return directly without blocking and the call will complete in the background.**When this will happen may depend on several factors but this concern is abstracted away from the code.**(从这个角度看，异步也是一种抽象)Because the thread is not blocked waiting for the operation to complete,it can do other work in the meantime,thus using resources  more efficiently.

intercepting operations and transforming inbound or outbound data on the fly requires only that you provide callbacks or utilize the Futures that are returned by opertations.**This makes chaining operations easy** and efficient and ptomotes the writing of reusable，generic code.

同时还有一个问题，方法的调用形成方法栈，方法调用的基本问题：传递参数，传递返回结果。

对于同步调用来说，传递参数和传递结果，机制和约定就不说了。
	
![](/public/upload/architecture/async_servlet_1.png)
			
而对于异步调用来说，在方法被提交到实际的执行者之前，会经过多次封装调用，也会形成一个方法栈。这个栈上的所有方法都等着实际执行者的反馈，所以，观察netty可以看到

chaining operations

![](/public/upload/architecture/async_servlet_2.png)

1. 异步操作嵌套异步操作，直到碰到一个executor直接执行或提交任务。即一个线程的业务逻辑分为事件驱动引擎和事件提交两个部分，**对外暴露的异步函数本质上是事件（及事件成功、失败处理逻辑）的提交**。此处要强调的一点是，切不可被函数封装所迷惑，比如netty的`bootstrap.connect`，看着像连接操作由调用线程触发，实际不是的。
2. future封装下一层future的处理结果
3. 从图中可以看到，**代码的腾挪 是一种艺术**，说白了将代码交给另一个线程执行，但调用的时候还是同步调用的感觉（要么返回future，要么传入callbck）。脑洞大一点，对于分布式系统来说，干脆就是代码交给 另一台主机的进程/线程执行。

这也解释了，为什么说，异步框架中不要有同步代码？

因为所有的任务代码都是由一个“事件驱动引擎”执行的，换句话说，事件驱动引擎的时间，就好比cpu时间一样， 比较宝贵，要避免为未知的阻塞操作所滞留。

2018.6.30 补充。拿netty 和 go 类比一下，可以看到，调用go语言的阻塞方法（io方法不确定是不是这样），相当于

	read(){
		约定好上下文
		让出goroutine 执行权
	}
	
	
go 中实际的 阻塞 操作实际是go语言层完成的，goroutine 调度本身是对 内核系统调用的模仿。只是软中断变成了方法调用，线程阻塞改成 goroutine 阻塞。
	
netty 因为不能改写 io 语言的系统调用，为此 不敢向你直接暴露 io 方法，封装了一个channel 出来，你调用`channel.write()` 不是真正的`java.nio.channels.SocketChannel.write`。 所以netty 也只能提供 一个 单独的层次，与上图竖着画驱动线程与事件驱动引擎（线程）的关系不同

![](/public/upload/architecture/async_servlet_2.png)

相当于netty 提供了一个全异步 io 操作（也包括一般的任务执行）的抽象层，支持类似AIO的”系统调用“。所以上图竖着画，就是另一番滋味了。

![](/public/upload/netty/netty_io.png)

netty in action 中提到

non-blocking network calls free us from having to wait for the completion of an operation.fully asynchronous i/o builds on this feature and carries it a step further:an  asynchronous method returns  immediately and notifies the user when it is complete,directly or at a later time.

此时的线程不再是一个我们通常理解的：一个请求过来，干活，结束。而是一个不停的运行各种任务的线程，任务的结束不是线程的结束，任何的用户请求都是作为一个任务来提交。这就是java Executors中的线程，如果这个线程既可以处理任务，还可以注册socket 事件，就成了netty的eventloop

我们指定线程运行任务 ==> 提交任务，任务线程自己干自己的，不会眷顾某个任务。那么就产生了与任务线程交互的问题，也就引出了callback、future等组件。java的future可以存储异步操作的结果，但结果要手工检查（或者就阻塞），netty的future则通过listener机制


## 基于 异步接口组织业务逻辑

### 异步框架提供哪些异步接口

传统的接口——回调函数——事件，容易有callback hell问题

javascript 有一套异步编程规范 [Promises/A+](https://promisesaplus.com/) 非常给力，然后java 语言参照着实现了一把[异步编程——Promise](https://github.com/hprose/hprose-java/wiki/%E5%BC%82%E6%AD%A5%E7%BC%96%E7%A8%8B%E2%80%94%E2%80%94Promise)

所谓Promise，**简单说就是一个“容器”**，里面保存着某个未来才会结束的事件（通常是一个异步操作）的结果。 此外 “容器”持有 对返回结果(正常/异常) 的处理函数`promise.then(onFulfilled, onRejected);`

## 异步流程控制模式

[async 异步编程框架](https://www.jianshu.com/p/cdddcd361567)

异步流程控制模式包括

1. 串行(series)，后一个调用参数依赖前一个调用的结果。
2. 并行(parallel)，彼此无关联。 这种可以提供类似 CombinedFuture 的工具类
3. 瀑布(waterfall)等，后一个调用是否执行 + 调用参数 依赖前一个调用的结果。这里有个问题，对于这种类型的业务，或许同步模型可读性会更好点。

	    public static <V1, V2> Future<V2> then(final ListenableFuture<V1> future, final Predicate<V1> function, final AsyncFunction<V1, V2> asyncFunction) {
	        Promise<V2> promise = new Promise<V2>();
	        future.addListener(new Listener() {
	            public void onSuccess() {
	                V1 v1 = future.get();
	                if (function.apply(v1)) {
	                    promise = util(asyncFunction.apply(v1));
	                }
	            }
	            public void onFail(){
	                promise.setFailure()
	            }
	        });
        	return promise;
        }



## 小结

异步为什么性能更好？

1. 在开发层面给予更多选择，可以不让线程让出cpu。开发者往往比操作系统知道怎样更高效利用线程。这也是java、netty中一脉相承的思路，比如AQS、CAS来减少线程切换，common-pool、netty中的arena手动管理内存。
2. 事件驱动引擎是一种编程模型，可以不处理io事件。但加入io事件的处理后，select集中等待，并可以在io任务和cpu任务中控制开销比例，进而可以做到更高效。

## 引用

[性能调优攻略](http://coolshell.cn/articles/7490.html)