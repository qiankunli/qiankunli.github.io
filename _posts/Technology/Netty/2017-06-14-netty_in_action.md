---

layout: post
title: 《netty in action》读书笔记
category: 技术
tags: Netty
keywords: JAVA netty

---

## 前言

## java nio

the earliest versions of java introduced enough of an **object-oriented facade** to hide some of the thornier details.but those first java apis supported only the so-called blocking functions provided by the native system socket libraries.

相当的java库就是对底层c库做object-oriented facade

java nio provide considerably more control over the utiliaztion of network resources:

* Using `setsockopt()`,you can configure sockets so that read/write calls will return immediately if there is no data; 是不是blocking其实就是socket 的一个opt，不用牵扯其它的
* you can register a set of non-blocking sockets using the **system's event notification api** to determine whether any of them have data ready for reading or writing. 

select 其实是一种event notification service

this model provides much better resource management than the blocking i/o model:

* many connecitons can be handled with fewer threads,and thus with far less overhead due to memory management and context-switching. 为每个线程分配栈空间是要占内存的
* threads can be **retargeted** to other tasks when there is no i/o to handle.这个retargeted很传神

## netty

在提交netty的一些组件时，作者提到think of them as domain objects rather than concrete java classes.

### Channel、ChannelPipeline、ChannelHandler和ChannelHandlerContext的一对一、一对多关系

netty is asynchronous and event-driven. asynchronous说的是outbound io operations，event-driven 应该对应的是io inbound operations.

every new channel that is created is assigned a new ChannelPipeline.This association is permanent;the channel can neither attach another ChannelPipeline nor detach the current one.

a ChannelHandlerContext represents an association between a ChannelHandler and ChannelPipeline and is created whenever a ChannelHandler is added to a ChannelPipeline.

the movement from one handler to the next at the ChannelHandler level is invoked on the ChannelHandlerContext.

a ChannelHandler can belong to more than one ChannelPipeline,it can be bound to multiple ChannelHandlerContext instances.

### thread model

in netty3,the threading model used in previous releases guaranteed only that inbound (previously called upstream) events would be executed in the so-called i/o thread(corresponding to netty4's eventloop).all outbound(downstream) events were handled by the calling thread,which might be the i/o thread or any other.netty3的问题就是，处理由多个calling thread线程触发的outbound事件时，要通过加锁等方式解决线程安全问题。

**同步io，线程和连接通常要维持一对一关系。异步io才可以一个线程处理多个io。**

首先，netty的数据处理模型

1. 以event notification 来处理io
2. event分为inbound 和 outbound
3. event 由handler处理，handler形成一个链pipeline

数据处理模型与线程模型的结合，便是

1. channel和eventloop是多对一关系
2. channel的inbound和outbound事件全部由eventloop处理
3. 根据1和2，outbound事件可以由多个calling thread触发，但只能由一个eventloop处理。那么就需要将多线程调用转换为任务队列。

the basic idea of an event loop

	while(!terminated){
		List<Runnable> readyEvents = blockUntilEventsReady();
		for(Runnable ev: readyEvents){
			ev.run();
		}
	}
	
### 创建、分配和销毁

当我说线程模型的时候，我在说什么？**如何创建、分配、运行(运行逻辑)和销毁一个线程，是一个线程模型（线程池）的基本内容。**

1. 线程自生自灭
2. 由一个第三方类来管理或触发，比如Executor和EventloopGroup。
	
java 线程池的常见用法

    ExecutorService executor = Executors.newFixedThreadPool(5);
    executor.execute(worker);
	
||提交|池中单元的执行逻辑|创建与销毁|分配|
|---|---|---|---|---|---|
|java线程池|提交runnable|执行`runnable.run`|所属单元统一创建和销毁|找一个空闲的|
|eventloop group|绑定channel|执行 eventloop 处理inbound事件，如上图所示。执行`runnable.run`处理outbound事件|所属单元统一创建和销毁|创建时建立绑定关系，然后根据绑定关系分配|

也就是说，池中的线程逻辑变了，转换过程如下

	abstract class SingleThreadEventExecutor {
		Executor executor;
		volatile Thread thread;
		void doStartThread(){
			...
			executor.execute(new Runnable() {
				public void run() {
					 SingleThreadEventExecutor.this.run();
				}
			});
			...
		}
		protected abstract void run();
		void execute(Runnable task){
			在第一次执行时，doStartThread
		}
	}

Executor的转换：SingleThreadEventExecutor提供类似java Executor的接口`void execute(Runnable task)`，将任务转了一下，这样做

1. java Executor 只能执行runnable，而基于SingleThreadEventExecutor的NioEventloop可以起到`void execute(Channel channel)`的效果
2. 在表现形式上，因为channel与eventloop的一对多关系，由channel维护eventloop的引用来执行outbound event（同样通过`void execute(Runnable task)`提交）。eventloop通过selector得到channel的引用来处理inbound 事件。
3. SingleThreadEventExecutor 记录 绑定的 java Executor中的Thread，确保所有任务统交由该线程执行，这也是SingleThread的含义。有了绑定关系，就有了绑定关系的建立和释放过程（netty中无需释放）。

## Codec



