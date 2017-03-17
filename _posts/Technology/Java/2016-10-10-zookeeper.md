---

layout: post
title: zookeeper的一些分析
category: 技术
tags: Java
keywords: zookeeper

---

## 前言（待整理） 

本文主要从zookeeper源码角度，分析下zookeeper一些功能点的实现

源码分析，github上已有maven组织的项目（zookeeper使用ant组织的），参见[swa19/zookeeperSourceCode](https://github.com/swa19/zookeeperSourceCode)

源码的分析参见 [llohellohe/zookeeper](https://github.com/llohellohe/zookeeper/blob/master/docs/overview.md)

## 基本通信

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
|api层|zookeeper，提供给用户的操作对象，提供各种api:create,get,exist|string path,byte[] data||提供增删改查监操作api|
|业务层|ClientCnxn.submitRequest/ClientCnxn.queuePacket|RequestHeader,Record, ReplyHeader |包括两个线程子类，SendThead和EventThread。其作用《从paxos到zookeeper》有介绍|1. 通过队列+线程封装异步操作；2. 针对packet中的数据进行有意义的操作|
|transport 层|ClientCnxnSocket.sendpacket|Packet|A ClientCnxnSocket does the lower level communication with a socket implementation.|可以是nio、bio。提供packet抽象，负责数据通信|
|socket层||byte[]|||

从上到下是任务分解，从下到上是协议机制的实现。

**我以前理解的分层，一方面是网络协议栈那种系统级的；另一方面是简单的理解为解封包，将字节流转换为有意义的model。那么观察这种业务级的zk client，可以看到，业务级分层可以实现更丰富的抽象，比如监听（要有协议数据的支持）、异步机制的实现。**

## 服务端

基本机制

1. persit机制，所有zk数据，内存 + 硬盘，存储 + 内存/硬盘同步
2. session机制，zookeeper会为每个client分配一个session，类似于web服务器一样。针对session可以有保存一些关联数据。
3. watcher机制

此外，因为是多节点的，zookeeper server还要支持leader选举和数据副本一致性的机制。

## 如何实现watcher机制

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