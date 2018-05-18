---

layout: post
title: 分布式系统小结
category: 技术
tags: Architecture
keywords: 分布式系统

---

## 简介

本文主要聊一下 分布式系统的 基本组成和理论问题

[现有分布式项目小结](http://qiankunli.github.io/2015/07/14/distributed_project.html)
[分布式配置系统](http://qiankunli.github.io/2015/08/08/distributed_configure_system.html)
[分布式事务](http://qiankunli.github.io/2017/07/18/distributed_transaction.html)


有没有一个 知识体系/架构、描述方式，能将分布式的各类知识都汇总起来？

## 学习分布式的正确姿势

### 不要沉迷与具体的算法

[distributed-systems-theory-for-the-distributed-systems-engineer](http://the-paper-trail.org/blog/distributed-systems-theory-for-the-distributed-systems-engineer/) 文中提到：

My response of old might have been “well, here’s the FLP paper, and here’s the Paxos paper, and here’s the Byzantine generals（拜占庭将军） paper…”, and I’d have prescribed(嘱咐、规定) a laundry list of primary source material which would have taken at least six months to get through if you rushed. **But I’ve come to thinking that recommending a ton of theoretical papers is often precisely the wrong way to go about learning distributed systems theory (unless you are in a PhD program).** Papers are usually deep, usually complex, and require both serious study, and usually significant experience to glean(捡拾) their important contributions and to place them in context. What good is requiring that level of expertise of engineers?

也就是说，具体学习某一个分布式算法用处有限。一个很难理解，一个是你很难  place them in contex（它们在解决分布式问题中的作用）。

### 一致性和共识

[被误用的“一致性”](http://blog.kongfy.com/2016/08/%E8%A2%AB%E8%AF%AF%E7%94%A8%E7%9A%84%E4%B8%80%E8%87%B4%E6%80%A7/) 一致性和算法的误区

我们常说的“一致性（Consistency）”在分布式系统中指的是副本（Replication）问题中对于同一个数据的多个副本，其对外表现的数据一致性，如线性一致性、因果一致性、最终一致性等，都是用来描述副本问题中的一致性的。

[分布式共识(Consensus)：Viewstamped Replication、Raft以及Paxos](http://blog.kongfy.com/2016/05/%E5%88%86%E5%B8%83%E5%BC%8F%E5%85%B1%E8%AF%86consensus%EF%BC%9Aviewstamped%E3%80%81raft%E5%8F%8Apaxos/)

分布式共识问题，简单说，**就是在一个或多个进程提议了一个值应当是什么后，使系统中所有进程对这个值达成一致意见。** 

这样的协定问题在分布式系统中很常用，比如说选主（Leader election）问题中所有进程对Leader达成一致；互斥（Mutual exclusion）问题中对于哪个进程进入临界区达成一致；原子组播（Atomic broadcast）中进程对消息传递（delivery）顺序达成一致。对于这些问题有一些特定的算法，但是，**分布式共识问题试图探讨这些问题的一个更一般的形式，如果能够解决分布式共识问题，则以上的问题都可以得以解决。**

小结一下就是，一致性是一个结果，共识是一个算法，通常被用于达到一致性的结果。

在《区块链核心算法解析》中，则采用另一种描述方式：对于一组节点，如果所有节点均以相同的顺序执行一个（可能是无限的）命令序列c1,c2,c3...，则这组节点 实现了状态复制。

## 一致性问题 的几个协议及其之间的关系

[从CAP理论到Paxos算法](http://blog.longjiazuo.com/archives/5369?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io) 基本要点：

1. cap 彼此的关系。提高分区容忍性的办法就是一个数据项复制到多个节点上，那么出现分区之后，这一数据项就可能分布到各个区里。分区容忍就提高了。然而，要把数据复制到多个节点，就会带来一致性的问题，就是多个节点上面的数据可能是不一致的。要保证一致，每次写操作就都要等待全部节点写成功，而这等待又会带来可用性的问题。
2. cap 着力点。网络分区是既成的现实，于是只能在可用性和一致性两者间做出选择。在工程上，我们关注的往往是如何在保持相对一致性的前提下，提高系统的可用性。
3. 还是读写问题。[多线程](http://qiankunli.github.io/2014/10/09/Threads.html) 提到，多线程本质是一个并发读写问题，数据库系统中，为了描述并发读写的安全程度，还提出了隔离性的概念。具体到cap 理论上，副本一致性本质是并发读写问题（A主机写入的数据，B主机多长时间可以读到）。2pc/3pc 本质解决的也是如此 ==> A主机写入的数据，成功与失败，B主机多长时间可以读到，然后决定自己的行为。
4. Paxos算法

	* Phase 1
		
		* proposer向网络内超过半数的acceptor发送prepare消息
		* acceptor正常情况下回复promise消息
	* Phase 2
		* 在有足够多acceptor回复promise消息时，proposer发送accept消息
		* 正常情况下acceptor回复accepted消息

只有一个Proposer能进行到第二阶段运行。

目前比较好的通俗解释，以贿选来描述 [如何浅显易懂地解说 Paxos 的算法？ - GRAYLAMB的回答 - 知乎](https://www.zhihu.com/question/19787937/answer/107750652)。

一些补充

1. proposer 和 acceptor，异类交互，同类不交互

	![](/public/upload/architecture/distributed_system_2.png)
	
2. proposer 贿选 不会坚持 让acceptor 遵守自己的提议。出价最高的proposer 会得到大部分acceptor 的拥护（谁贿金高，acceptor最后听谁的。换个说法，acceptor 之间没有之间交互，但），  但会以最快达成一致 为目标，若是贿金高但提议晚，也是会顺从 他人的提议。

下面看下 《区块链核心算法解析》 中的思维线条

1. 两节点

	1. 客户端服务端，如何可靠通信？如何处理消息丢失问题
	2. 请求-确认，客户端一段时间收不到 确认则重发，为数据包标记序列号解决重发导致的重复包问题。这也是tcp 的套路

2. 单客户端-多服务端
3. 多客户端-多服务端

	1. 多服务端前 加一个 单一入口（串行化器）， 所以的客户端先发给 串行化器，再分发给服务端。即主从复制思路==> 串行化器单点问题
	2. 客户端先协调好，由一个客户端发命令

		1. 抽取独立的协调器。2pc/3pc 思路
		2. 客户端向所有的服务端申请锁，谁先申请到所有服务器的锁，谁说了算。缺点：客户端拿到锁后宕机了，尴尬！
		3. 票的概念，弱化形式的锁。paxos 套路（当然，具体细节更复杂）

未完成

1. paxos 无法保证确定性，即理论上存在一直无法达成一致， 不停地投票的情况
2. paxos 假定 参与节点都按规则 运行的基础上

FLP 原理

可以画一个饼图，哪些不可能，哪些可能。理论上不可能，技术上又如何解决？

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

