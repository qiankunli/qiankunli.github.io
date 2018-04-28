---

layout: post
title: 分布式系统小结
category: 技术
tags: Architecture
keywords: 分布式系统

---

## 简介（未完成）

本文主要聊一下 分布式系统的 基本组成和理论问题

有没有一个 知识体系/架构、描述方式，能将分布式的各类知识都汇总起来

[现有分布式项目小结](http://qiankunli.github.io/2015/07/14/distributed_project.html)
[分布式配置系统](http://qiankunli.github.io/2015/08/08/distributed_configure_system.html)
[分布式事务](http://qiankunli.github.io/2017/07/18/distributed_transaction.html)

## 学习分布式的正确姿势

### 不要沉迷与具体的算法

[distributed-systems-theory-for-the-distributed-systems-engineer](http://the-paper-trail.org/blog/distributed-systems-theory-for-the-distributed-systems-engineer/) 文中提到：

My response of old might have been “well, here’s the FLP paper, and here’s the Paxos paper, and here’s the Byzantine generals paper…”, and I’d have prescribed a laundry list of primary source material which would have taken at least six months to get through if you rushed. But I’ve come to thinking that recommending a ton of theoretical papers is often precisely the wrong way to go about learning distributed systems theory (unless you are in a PhD program). Papers are usually deep, usually complex, and require both serious study, and usually significant experience to glean their important contributions and to place them in context. What good is requiring that level of expertise of engineers?

也就是说，具体学习某一个分布式算法用处有限，你很难  place them in contex

### 一致性和共识

[被误用的“一致性”](http://blog.kongfy.com/2016/08/%E8%A2%AB%E8%AF%AF%E7%94%A8%E7%9A%84%E4%B8%80%E8%87%B4%E6%80%A7/) 一致性和算法的误区

我们常说的“一致性（Consistency）”在分布式系统中指的是副本（Replication）问题中对于同一个数据的多个副本，其对外表现的数据一致性，如线性一致性、因果一致性、最终一致性等，都是用来描述副本问题中的一致性的。

[分布式共识(Consensus)：Viewstamped Replication、Raft以及Paxos](http://blog.kongfy.com/2016/05/%E5%88%86%E5%B8%83%E5%BC%8F%E5%85%B1%E8%AF%86consensus%EF%BC%9Aviewstamped%E3%80%81raft%E5%8F%8Apaxos/)

分布式共识问题，简单说，**就是在一个或多个进程提议了一个值应当是什么后，使系统中所有进程对这个值达成一致意见。** 

这样的协定问题在分布式系统中很常用，比如说选主（Leader election）问题中所有进程对Leader达成一致；互斥（Mutual exclusion）问题中对于哪个进程进入临界区达成一致；原子组播（Atomic broadcast）中进程对消息传递（delivery）顺序达成一致。对于这些问题有一些特定的算法，但是，**分布式共识问题试图探讨这些问题的一个更一般的形式，如果能够解决分布式共识问题，则以上的问题都可以得以解决。**

小结一下就是，一致性是一个结果，共识是一个算法，通常被用于达到一致性的结果。

## 一致性问题 的几个协议及其之间的关系

结果和过程

一致性是一个结果， 那么什么样的结果 称为一致？
如何达成一致？

达成一致的节点

1. 只是失效
2. 会伪造非法信息


[从CAP理论到Paxos算法](http://blog.longjiazuo.com/archives/5369?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io) 基本要点：

1. cap 彼此的关系。提高分区容忍性的办法就是一个数据项复制到多个节点上，那么出现分区之后，这一数据项就可能分布到各个区里。分区容忍就提高了。然而，要把数据复制到多个节点，就会带来一致性的问题，就是多个节点上面的数据可能是不一致的。要保证一致，每次写操作就都要等待全部节点写成功，而这等待又会带来可用性的问题。
2. cap 着力点。网络分区是既成的现实，于是只能在可用性和一致性两者间做出选择。在工程上，我们关注的往往是如何在保持相对一致性的前提下，提高系统的可用性。
3. 还是读写问题。[多线程](http://qiankunli.github.io/2014/10/09/Threads.html) 提到，多线程本质是一个并发读写问题，数据库系统中，为了描述并发读写的安全程度，还提出了隔离性的概念。具体到cap 理论上，副本一致性本质是并发读写问题（A主机写入的数据，B主机多长时间可以读到）。2pc/3pc 本质解决的也是如此 ==> A主机写入的数据，成功与失败，B主机多长时间可以读到，然后决定自己的行为。
4. Paxos算法 （未完成）


2pc/3pc/paxos 都需要协调者

### 分布式

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




## 分布式知识体系

![](/public/upload/architecture/distributed_system.png)

