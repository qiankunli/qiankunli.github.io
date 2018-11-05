---

layout: post
title: zookeeper三重奏
category: 技术
tags: Java
keywords: zookeeper

---

## 前言（待整理） 

* TOC
{:toc}


## 基本通信

本小节主要从zookeeper源码角度，分析下zookeeper一些功能点的实现

源码分析，github上已有maven组织的项目（zookeeper使用ant组织的），参见[swa19/zookeeperSourceCode](https://github.com/swa19/zookeeperSourceCode)

源码的分析参见 [llohellohe/zookeeper](https://github.com/llohellohe/zookeeper/blob/master/docs/overview.md)

### 客户端

一个比较简单的demo代码

	ZooKeeper zk = new ZooKeeper("192.168.3.7:2181", 10000, new Watcher() {
		@Override
		public void process(WatchedEvent event) {
			System.out.println(event.getType() + " happend");
		}
	});
	zk.create("/lqk", "lqk".getBytes(), Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
	// 打开对该目录的监控
	zk.exists("/lqk", true);
	// 创建一个子目录节点
	zk.create("/lqk/lqk1", "lqk".getBytes(), Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
	zk.close();
	

|分层|从上到下接口变化|从上到下数据变化|概述|本层作用|
|---|---|---|---|---|---|
|api层|zookeeper，提供给用户的操作对象，提供各种api:create,get,exist|string path,byte[] data||提供增删改查监操作api，作为一个操作对象，必须是线程安全的|
|业务层|ClientCnxn.submitRequest/ClientCnxn.queuePacket|RequestHeader,Record, ReplyHeader |包括两个线程子类，SendThead和EventThread。其作用《从paxos到zookeeper》有介绍|1. 通过队列+线程封装异步操作；2. 针对packet中的数据进行有意义的操作|
|transport 层|ClientCnxnSocket.sendpacket|Packet|A ClientCnxnSocket does the lower level communication with a socket implementation.|可以是nio、bio。提供packet抽象，负责数据通信|
|socket层||byte[]|||

从上到下是任务分解，从下到上是协议机制的实现。

**我以前理解的分层，一方面是网络协议栈那种系统级的；另一方面是简单的理解为解封包，将字节流转换为有意义的model。那么观察这种业务级的zk client，可以看到，业务级分层可以实现更丰富的抽象，比如监听（要有协议数据的支持）、异步机制的实现。**


//todo

1. 客户端watcher的原理
2. 如何优雅的处理客户端配置
3. 想使用原生接口一样使用netty


参见[ZooKeeper学习之关于Servers和Watches](http://damacheng009.iteye.com/blog/2085002)

watch是由read operation设置的一次性触发器，由一个特定operation来触发。为了在server端管理watch，ZK的server端实现了watch manager。一个WatchManager类的实例负责管理当前已被注册的watch列表，并负责触发它们。

在server端触发了一个watch，会传播到client。此类使用server cnxn对象来处理（参见ServerCnxn类），此对象管理client和server的连接并实现了Watcher接口。Watch.process方法序列化了watch event，并通过网络发送出去。client接收到了序列化数据，转换成watch event对象，并传递到应用程序。watch只会保存在内存，不会持久化到硬盘。当client断开与server的连接时，它的所有watch会从内存中清除。因为client的库也会维护一份watch的数据，在重连之后watch数据会再次被同步到server端。

### 服务端

基本机制

1. persit机制，所有zk数据，内存 + 硬盘，存储 + 内存/硬盘同步
2. session机制，zookeeper会为每个client分配一个session，类似于web服务器一样。针对session可以有保存一些关联数据。
3. watcher机制

此外，因为是多节点的，zookeeper server还要支持leader选举和数据副本一致性的机制。

### 如何实现watcher机制

总结起来看，监听分为三种情况

1. 简单的监听，即观察者模式
2. 异步监听，比如一个异步工作完成后干某事
3. 跨主机的监听，涉及到io和线程模型的协作

首先，客户端和服务端都会维护一个watcher manager，为什么？因为zookeeper client 与server之间不单纯的交互data数据，也交互watcher event数据（如果注册watcher的话）。

在具体工作流程上，简单地讲，客户端在向zookeeper服务器注册watcher的同时，会将watcher对象存储在客户端的watchmanager中。当发生数据/连接变更

1. 服务端，data operation trigger watcher manager，具体反应是：向客户端发送通知（网络通信）
2. 客户端线程从watchmanager中取出对应的watcher对象，反应是：执行回调逻辑。

换句话说：

1. 客户端watcher manager存储 `<event,callback>`数据
2. 服务端watcher manager存储 `<path,watcher>`数据

说简单点，就是跨主机callback，可以参见本文的callback章节


## Curator

Zookeeper committer Patrick 谈到：Guava is to Java what Curator is to Zookeeper

对于client类的api

1. 操作对象
2. 操作对象提供的操作方法，以此来观察如何使用某个应用
3. 操作方法返回的数据对象


从增强手法上讲

guava对java的封装，主要有以下几个方向：

1. 继承java类功能以增强
2. 提供独立的工具类以增强


curator对zookeeper的增强则有些不同

1. CuratorZookeeperClient 直接聚合的原来的zookeeper
2. 监听事件上，则将zk的事件处理函数和事件对象进行了类似于“继承”的扩展。

		new Watcher(){
			 public void process(WatchedEvent watchedEvent){
                CuratorEvent event = xx
                processEvent(event);
             }
		}
			
### 几个有意思的工具类

RetryLoop
ThreadUtils

## Spring-Curator

相关博客 [Spring and Apache Curator](http://jdpgrailsdev.github.io/blog/2014/02/19/spring_curator.html) github [jdpgrailsdev/spring-curator](https://github.com/jdpgrailsdev/spring-curator)

One of the thinks to know about the Apache Curator client is that you only need one per instance/ensemble of Apache ZooKeeper. Therefore, using Spring to manage the injection of a Singleton bean into a class that needs access to the client is a perfect fit. 

