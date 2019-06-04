---

layout: post
title: 副本一致性
category: 技术
tags: Distribute
keywords: 一致性协议

---

## 简介

* TOC
{:toc}

建议先参见 [串一串一致性协议](http://qiankunli.github.io/2018/09/27/consistency_protocol.html) 了解下背景

![](/public/upload/distribute/consistency.png)

《软件架构设计》有一个关于Paxos、Raft和Zab 的分析对比，包括

1. 复制模型
2. 写入方式：多点/单点写入， 乱序/顺序提交
3. 同步方向：双向/单向同步
4. 心跳检测：有/无


我们使用“一致性”这个字眼太频繁了，国外的 Consistency 被称为一致性、Consensus 也唤作一致性，甚至是 Coherence 都翻译成一致性。比如大名鼎鼎的 Raft 算法和 Paxos 算法。了解它的人都知道它们的作用是在分布式系统中让多个节点就某个决定达成共识，都属于 Consensus Algorithm 一族

## 复制模型

![](/public/upload/distribute/consistency_copy_log.png)

假设KV集群有三台机器，机器之间相互通信，把自己的值传播给其他机器，三个客户端并发的向集群发送三个请求，值X 应该是多少？是多少没关系，135都行，向一个client返回成功、其它client返回失败（或实际值）即可，关键是Node1、Node2、Node3 一致。

||Replicated State Machine|Primary-Backup System|
|---|---|---|
|中文|复制状态机|
|应用的协议|Paxos、Raft|Zab|
|mysql binlog的数据格式|statement<br>存储的是原始的sql语句|raw<br>数据表的变化数据|
|redis持久化的两种方式|AOF<br>持久化的是客户端的set/incr/decr命令|RDB<br>持久化的是内存的快照|
|数据同步次数|客户端的写请求都要在节点之间同步|有时可以合并|

## Replicated State Machine

### 复制状态 == 复制日志 ==> 状态一致 == 日志顺序一致

假设每台机器把收到的请求按日志存下来（包括客户端的请求和其它Node的请求）。不管顺序如何（135/153/315/351/513/531），只要三台机器的日志顺序是一样的，就是所谓的“一致性”。这句话有几个点：

1. 可以看作：Nod2 收到经由Node1 **转发的**来自Client1 的X=1的请求，本质上还是3个Node如何协调一致的问题。
2. 为何要存储日志，直接存储最终的数据不就行了么？

	1. 日志只有一种操作，就是append。而数据状态一直在变化，可以add、delete、update 把三种操作变成一种，便于持久化存储
	2. 数据可能是很复杂的数据结构，比如树、图等，并且状态一直在变化，为保证多机数据一致做数据比对很麻烦。
	3. **任何复杂的数据都可以通过日志生成**。一样的初始状态 + 一样的输入事件 = 一样的最终状态。可见要保证最终状态一致，只需保证多个Node的日志流是一样的即可。
	4. 日志是一个线性序列，比对容易 
	5. Node宕机重启后，只需重放日志即可

### 日志顺序一致 ==> 集群Node对同一个日志位置存储哪个数据协商一致

虽然每个Node 接收到的请求顺序不同，但它们对日志中1号位置、2号位置、3号位置的认知是一样的，大家一起保证1号位置、2号位置、3号位置存储的数据一样。

每个Node 在存储日志前要先问下其它Node（我打算在位置n存xx，你们同意不？若是不同意，你们想存啥），之后再决定把这条日志写到哪个位置（按上次大家的意思，我打算在位置n存xx，你们同意不？若是不同意，重新开始）。


## Paxos——不断循环的2PC

上文所说的“日志位置” 在这里是一个Proposer生成的自增ID，不需要全局有序。

每个Node 同时充当了两个角色：Proposer和Acceptor，在实现过程中， 两个角色在同一个进程里面。

* Prepare阶段——针对一个“位置”的提议，充分的听取大家的意见

	![](/public/upload/distribute/paxos_propose_stage.png)

	1. Proposer 广播 prepare(n)
	2. Proposer 如果收到半数以上yes，则支持在位置n写入 收到的新值（按一定的算法规则）。否则n 自增，重新请求。

* Accept阶段——选取一个“意见”，向大家确认
		
	![](/public/upload/distribute/paxos_accept_stage.png)

	1. Proposer 广播 accept(n,v)
	2. Proposer 如果收到半数以上yes，并且收到的n与accept n一致，则结束。否则n 自增，重新从零开始。

Paxos 是一个“不断循环”的2pc，两个阶段都可能会失败，从0开始，陷入“不断”循环，即“活锁”问题（一直得动，但没有结果）。

目前比较好的通俗解释，以贿选来描述 [如何浅显易懂地解说 Paxos 的算法？ - GRAYLAMB的回答 - 知乎](https://www.zhihu.com/question/19787937/answer/107750652)。

## Raft

![](/public/upload/distribute/raft_copy_log.png)

与Paxos 不断循环的2PC 不同，Raft/Zab 算法分为3个阶段

1. 选举阶段，选举出leader，其他机器为Follower

	1. 处于candidate 状态的节点向所有节点发起广播
	2. 超过半数回复true，则成为Leader，否则重新选举
2. 正常阶段，Leader 接收写请求，然后复制给其他Followers
3. 恢复阶段，旧Leader宕机， 新Leader上任，其他Follower切换到新Leader，开始同步数据。 

## 为什么paxos 更复杂——有一个Leader 能带来多少好处

![](/public/upload/distribute/paxos_vs_raft.png)

1. paxos的一次2pc 类似于 raft的选举， 只不过，paxos一次2pc 是确认一个“值”听谁的，而raft 一次选举是确认以后所有“值”听谁的（谁是老大）。
2. paxos 一次决策分为prepare 和accept 两个阶段，而raft 只需一次即可，因为

	1. 对一个term（第几任leader）而言，一个Follower 只能投一次票， 如果投给了candidate1 就不能再投给 candidate2
	2. 谁的日志最新 谁就是leader，follower 甚至candidate会支持拥有最新日志的candidate 的当leader

3. 乱序提交 vs 顺序提交：paxos 因为每个Node都可以接收请求，导致的一个问题是无法确认请求的时间顺序。raft和zab 都是Leader 接收请求，就一定可以排出先后。日志有了时序的保证，就相当于在全局为每条日志做了个顺序的编号！基于这个编号，就可以做日志的顺序提交、不同节点间的日志比对，回放日志的时候，也可以按照编号从小到大回放。

可见，协商谁当老大比协商一个值更明确，也更容易做出决策。生活中也是如此，谁当老大看拳头硬不硬就行了（吵都不用吵），但一件事怎么办却会吵得的不可开交。


kafka 因为有更明确地业务规则，可以直接复制数据。《软件架构设计》 多副本一致性章节的开篇就 使用kafka 为例讲了一个 做一个强一致的系统有多难。
