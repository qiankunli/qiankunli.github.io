---

layout: post
title: 分布式系统小结
category: 技术
tags: Distribute
keywords: 分布式系统

---

## 简介

本文主要聊一下 分布式系统（尤其指分布式计算系统）的 基本组成和理论问题

到目前为止，笔者关于分布式这块的博客

* [现有分布式项目小结](http://qiankunli.github.io/2015/07/14/distributed_project.html)
* [分布式配置系统](http://qiankunli.github.io/2015/08/08/distributed_configure_system.html)
* [分布式事务](http://qiankunli.github.io/2017/07/18/distributed_transaction.html)

有没有一个 知识体系/架构、描述方式，能将分布式的各类知识都汇总起来？之所以有分布式系统，肯定是单机不行，为什么不行？不可靠（单点）；不够用（性能、容量）。多机之后

1. 节点会挂掉，如何使应用比较可靠地运行
2. 如何使跨主机应用之间比较可靠的交互

	* 网络会断开、拥堵、变慢
	* 一个计算分布在多个节点上；一个计算由多个节点提供 ==> 一个应用是多节点的 ==> 应用之间是多对多节点交互，而不是一对一交互 ==> 引入中间角色 ==> 一致性问题。

笔者认为，从上到下，主要有以下几块：

1. 分布式应用系统

	* 比如spark、storm 这些，计算逻辑编写完毕后，在集群范围内分发、执行和监控。设计之初，就是基于集群实现。
	* 微服务，应用分别启动。另有监控、日志收集等旁路系统。更多是基于并行开发部署、模块复用等角度从单机拆成分布式的。

	PS：进而演化出paas平台，屏蔽多机物理资源，提供统一抽象，广泛的支持各种应用系统的运行。
2. 分布式中间件，比如zookeeper、kafka这些。将单机多线程/进程间的通信扩展到多机，**使得虽然跑在多机上，但可以彼此通信（共享内存、管道等）和交互。**要不怎么说叫“中间”件呢。
3. 分布式基本原理，包括共识算法等

此外，可以参见[《左耳听风》笔记](http://qiankunli.github.io/2018/09/08/zuoertingfeng_note.html) 看下陈皓大牛对分布式系统的阐述

## 分布式计算系统的 几个套路

参见[Spark Stream 学习](http://qiankunli.github.io/2018/05/27/spark_stream.html)  中对spark stream 和storm 对比一节，有以下几点：

1. 分布式计算系统，都是用户以代码的方式预定义好计算逻辑，系统将计算 下发到各个节点。这一点都是一样的，不同处是对外提供的抽象不同。比如`rdd.filter(function1).map(function2)`，而在storm 中则可能是 两个bolt
2. 有的计算 逻辑在一个节点即可执行完毕，比如不涉及分区的spark rdd，或分布式运行一个shell。有的计算逻辑则 拆分到不同节点，比如storm和mapreduce，“分段”执行。此时系统就要做好 调度和协调。
3. 分布式系统，总是要涉及到输入源数据的读取、数据在节点间流转、将结果写到输出端。

## 学习分布式的正确姿势

2018.7.16 补充 [漫谈分布式系统、拜占庭将军问题与区块链](http://zhangtielei.com/posts/blog-consensus-byzantine-and-blockchain.html) 作者阐述了分布式系统的核心问题和概念，沿着逻辑上前后一贯的思路，讨论了区块链技术。推荐阅读。

### 不要沉迷与具体的算法

[distributed-systems-theory-for-the-distributed-systems-engineer](http://the-paper-trail.org/blog/distributed-systems-theory-for-the-distributed-systems-engineer/) 文中提到：

My response of old might have been “well, here’s the FLP paper, and here’s the Paxos paper, and here’s the Byzantine generals（拜占庭将军） paper…”, and I’d have prescribed(嘱咐、规定) a laundry list of primary source material which would have taken at least six months to get through if you rushed. **But I’ve come to thinking that recommending a ton of theoretical papers is often precisely the wrong way to go about learning distributed systems theory (unless you are in a PhD program).** Papers are usually deep, usually complex, and require both serious study, and usually significant experience to glean(捡拾) their important contributions and to place them in context. What good is requiring that level of expertise of engineers?

也就是说，具体学习某一个分布式算法用处有限。一个很难理解，一个是你很难  place them in contex（它们在解决分布式问题中的作用）。

### 分布式与一致性

李运华 《从0到1学架构》 关于Robert Greiner 两篇文章的对比 建议细读，要点如下

1. 不是所有的分布式系统都有 cap问题，必须interconnected 和 share data。比如一个简单的微服务系统 没有shar data，便没有cap 问题。
2. 强调了write/read pair 。这跟上一点是一脉相承的。cap 关注的是对数据的读写操作，而不是分布式系统的所有功能。


想要沉迷，可以查看梳理的博客：[串一串一致性协议](http://qiankunli.github.io/2018/09/27/consistency_protocol.html)

## 分布式知识体系

[distributed-systems-theory-for-the-distributed-systems-engineer](http://the-paper-trail.org/blog/distributed-systems-theory-for-the-distributed-systems-engineer/) 

1. Many difficulties that the distributed systems engineer faces can be blamed on two underlying causes:

	* processes may fail
	* there is no good way to tell that they have done so

2. The basic tension of fault tolerance。fault 有两种级别

	* 节点失效
	* 节点返回错误数据（对应拜占庭将军问题中的 叛徒）
3. basic primitives

	* Leader election
	* Consistent snapshotting
	* Consensus
	* Distributed state machine replication

	
问题就是，你如何将 一致性、共识 这些概念 place 到 分布式的 context中。


![](/public/upload/architecture/distributed_system.png)

