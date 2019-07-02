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

netty is asynchronous and event-driven. 

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
	






