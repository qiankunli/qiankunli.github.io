---

layout: post
title: zookeeper的一些分析
category: 技术
tags: Java
keywords: zookeeper

---

## 前言（待整理） 

本文主要从zookeeper源码角度，分析下zookeeper一些功能点的实现

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
	

|分层|从上到下发送数据|概述|备注|
|---|---|---|---|
|api层|zookeeper，提供给用户的操作对象，提供各种api|||
|业务层|将api转化为各种业务对象，ClientCnxn.sendpacket|包括两个线程子类，SendThead和EventThread。其作用《从paxos到zookeeper》有介绍||
| transport 层|ClientCnxnSocket.sendpacket|A ClientCnxnSocket does the lower level communication with a socket implementation.|可以是nio、bio|


## 如何实现远程监听

总结起来看，监听分为三种情况

1. 简单的监听，即观察者模式
2. 异步监听，比如一个异步工作完成后干某事
3. 跨主机的监听，涉及到io和线程模型的协作

可以参见zookeeper的watch机制，以下摘自《从paxos到zookeeper》。

Zookeeper的watcher机制主要包括客户端线程、客户端watchmanager和zookeeper服务器三部分。在具体工作流程上，简单地讲，客户端在向zookeeper服务器注册watcher的同时，会将watcher对象存储在客户端的watchmanager中。当zookeeper服务器触发watcher事件后，会向客户端发送通知，客户端线程从watchmanager中取出对应的watcher对象来执行回调逻辑。

说简单点，就是跨主机callback，可以参见本文的callback章节