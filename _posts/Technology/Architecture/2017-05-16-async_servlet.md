---

layout: post
title: 异步和性能
category: 技术
tags: Architecture
keywords: 异步

---

## 简介

https://www.ibm.com/developerworks/cn/java/j-lo-comet/

如何定义一个系统的性能，参见[性能调优攻略](http://coolshell.cn/articles/7490.html)

[Servlet 3.0 实战：异步 Servlet 与 Comet 风格应用程序
](https://www.ibm.com/developerworks/cn/java/j-lo-comet/)

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

当然，对于一个特定请求，请求处理的耗时时间是不变的。

## 异步也是写框架的一种方式

下文摘自《netty in action》对channel的介绍

a ChannelFuture is returned as part of an i/o operations.Here,`connect()`will return directly without blocking and the call will complete in the background.**When this will happen may depend on several factors but this concern is abstracted away from the code.**(从这个角度看，异步也是一种抽象)Because the thread is not blocked waiting for the operation to complete,it can do other work in the meantime,thus using resources  more efficiently.

intercepting operations and transforming inbound or outbound data on the fly requires only that you provide callbacks or utilize the Futures that are returned by opertations.**This makes chaining operations easy** and efficient and ptomotes the writing of reusable，generic code.

不管同步还是异步，发出命令，得到反馈是coder最基本的需求。只要提供合适的接口，供你描述需求，都可以实现一个系统的运转。而对于异步来说，方法调用、方法调用的处理被当成一个任务提交，真正的执行系统在合适的时机执行它们。

同时还有一个问题，方法的调用形成方法栈，方法调用的基本问题：传递参数，传递返回结果。

对于同步调用来说，传递参数和传递结果，机制和约定就不说了。

	|-- 方法1
		|-- 方法2
			|-- 方法3
				|-- 系统调用
			|-- 处理系统调用结果
		|-- 处理方法3结果
	|-- 处理方法2结果
			
而对于异步调用来说，在方法被提交到实际的执行者之前，会经过多次封装调用，也会形成一个方法栈。这个栈上的所有方法都等着实际执行者的反馈，所以，观察netty可以看到

1. Future opeation(arg);一些对用户的操作
2. void operation(arg,Promise promise) 内部操作

通过promise 将方法栈的方法串起来，chaining operations，

我们知道，一个异步系统的简单模型是：将一个任务提交给实际执行者

	Future operation(arg){
		Future submit(arg)
	}

或者
	
	Future operation(arg){
		Promise promise = x
		submit(args,promise) // 有的系统将arg与promise合二为一， 比如ChannelPromise
	}

在这个最简单模型之上，也可以有多重的异步方法调用，异步方法也可能属于不同的对象，对象之间有复杂的依赖关系，而这些内容和同步调用都是一样的。		
## 其它

netty in action 中提到

non-blocking network calls free us from having to wait for the completion of an operation.fully asynchronous i/o builds on this feature and carries it a step further:an  asynchronous method returns  immediately and notifies the user when it is complete,directly or at a later time.

此时的线程不再是一个我们通常理解的：一个请求过来，干活，结束。而是一个不停的运行各种任务的线程，任务的结束不是线程的结束，任何的用户请求都是作为一个任务来提交。这就是java Executors中的线程，如果这个线程既可以处理任务，还可以注册socket 事件，就成了netty的eventloop

我们指定线程运行任务 ==> 提交任务，任务线程自己干自己的，不会眷顾某个任务。那么就产生了与任务线程交互的问题，也就引出了callback、future等组件。java的future可以存储异步操作的结果，但结果要手工检查（或者就阻塞），netty的future则通过listener机制


## 引用

[性能调优攻略](http://coolshell.cn/articles/7490.html)