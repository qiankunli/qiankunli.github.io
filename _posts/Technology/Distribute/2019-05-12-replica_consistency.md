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

![](/public/upload/distribute/replica_consistency_evolution.jpeg)

|共识算法拆解|含义|basic-paxos|multi-paxos|raft|kafka|
|---|---|---|---|---|---|
|选主||无|node分为Proposer和Acceptor|有<br>random timeout approach|有专门的coordinator|
|日志复制||无|允许并发写log|请求顺序提交|
|安全性|哪些follower有资格成为leader?<br>哪些日志记录被认为是commited?|无|任意|只有最新log的node才能成为leader|

### 复制状态 == 复制日志 ==> 状态一致 == 日志顺序一致

假设每台机器把收到的请求按日志存下来（包括客户端的请求和其它Node的请求）。不管顺序如何（135/153/315/351/513/531），只要三台机器的日志顺序是一样的，就是所谓的“一致性”。这句话有几个点：

1. 可以看作：Node2 收到经由Node1 **转发的**来自Client1 的X=1的请求，本质上还是3个Node如何协调一致的问题。
2. 为何要存储日志，直接存储最终的数据不就行了么？

	1. 日志只有一种操作，就是append。而数据状态一直在变化，可以add、delete、update 把三种操作变成一种，便于持久化存储
	2. 数据可能是很复杂的数据结构，比如树、图等，并且状态一直在变化，为保证多机数据一致做数据比对很麻烦。
	3. **任何复杂的数据都可以通过日志生成**。一样的初始状态 + 一样的输入事件 = 一样的最终状态。可见要保证最终状态一致，只需保证多个Node的日志流是一样的即可。
	4. 日志是一个线性序列，比对容易 
	5. Node宕机重启后，只需重放日志即可

### 日志顺序一致 ==> 集群Node对同一个日志位置存储哪个数据协商一致

虽然每个Node 接收到的请求顺序不同，但它们对日志中1号位置、2号位置、3号位置的认知是一样的，大家一起保证1号位置、2号位置、3号位置存储的数据一样。

每个Node 在存储日志前要先问下其它Node（我打算在位置n存xx，你们同意不？若是不同意，你们想存啥），之后再决定把这条日志写到哪个位置（按上次大家的意思，我打算在位置n存xx，你们同意不？若是不同意，重新开始）。

## basic-paxos——不断循环的2PC

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

### 如何理解ProposerId——逻辑锁

[如何浅显易懂地解说 Paxos 的算法？](https://www.zhihu.com/question/19787937/answer/420487000)Basic Paxos 的两个阶段可以看成多机下的一次读 version number + value（读阶段，原文中的准备阶段），然后拿读到的 version number 和 value 做一次 CAS （略有不同，但思想类似）（写阶段，原文中的接受阶段），非常直观。

**类比一下**，以mvcc 方式并发修改一行记录为例，假设一个student表， 带有version字段，当前version = 0, money = 100。线程1 和 线程2 都给zhangsan打钱（150,180）， 则打钱过程分两步

1. 读取到money = 100，此时version = 0 `select money,version from student where name = zhangsan`
2. 打钱 `update money = 150,version=1 set student where name = zhangsan and version = 0`

    1. 如果在此期间 线程2没有修改过， 则vesion 为0，执行成功
    2. 若修改过，则因为实际version = 1 无法匹配数据记录，修改失败。

假设存在一个全局唯一自增id服务，线程2_id > 线程1_id，也可以换个方式 

|步骤|线程1|线程2|
|---|---|---|
|改version占坑|`update version = 线程1_id where name = zhangsan and version < 线程1_id`||
|改version占坑||`update version = 线程2_id where name = zhangsan and version < 线程2_id`|
|结果||`version = 线程2_id`|
|打钱|`update money = 150 set student where name = zhangsan and version = 线程1_id`||
|打钱||`update money = 180 set student where name = zhangsan and version = 线程2_id`|
|结果|失败|成功|

paxos 与线程安全问题不同的地方是，paxos的node之间 是为了形成一致，而线程安全则是要么成功要么失败，所以有一点差异。 

## multi-paxos

当存在一批提案时，用Basic-Paxos一个一个决议当然也可以，但是每个提案都经历两阶段提交，显然效率不高。Basic-Paxos协议的执行流程针对每个提案（每条redo log）都至少存在三次网络交互：1. 产生log ID；2. prepare阶段；3. accept阶段。

所以，Mulit-Paxos基于Basic-Paxos做了优化，在Paxos集群中利用Paxos协议选举出唯一的leader，在leader有效期内所有的议案都只能由leader发起。

## Raft

![](/public/upload/distribute/raft_copy_log.png)

Raft协议比paxos的优点是 容易理解，容易实现。它强化了leader的地位，**把整个协议可以清楚的分割成两个部分**，并利用日志的连续性做了一些简化：

1. Leader在时。由Leader向Follower同步日志
2. Leader挂掉了，选一个新Leader

## 选主过程对比

不管名词如何变化，paxos 和 raft 都分为两个阶段：先决定听谁的，再将“决议”广播出去形成一致。 

### paxos

**Proposer 之间并不直接交互**，Acceptor除了一个“存储”的作用外，还有一个信息转发的作用。

**从Acceptor的视角看**，basic-paxos 及 multi-paxos 选举过程是协商一个值，每个Proposer提出的value 都可能不一样。

所以第一阶段，先经由Acceptor将**已提交的**ProposerId 最大的value 尽可能扩散到Proposer（即决定哪个Proposer 是“意见领袖”）。第二阶段，再将“多数意见”形成“决议”（Acceptor持久化value）
 
### raft

候选人视角：

1. 处于candidate 状态的节点向所有节点发起广播
2. 超过半数回复true，则成为Leader（同时广播自己是leader），否则重新选举

投票人视角:

1. 在任一任期内，单个节点最多只能投一票
2. first-come-first-served 先到先得。


如果出现平票的情况，那么就延长了系统不可用的时间（没有leader是不能处理客户端写请求的），因此raft引入了randomized election timeouts来尽量避免平票情况。同时，leader-based 共识算法中，节点的数目都是奇数个，尽量保证majority的出现。

## 其它

kafka 因为有更明确地业务规则，有一个专门的coordinator，选举过程进一步简化，复制log的逻辑基本一致。《软件架构设计》 多副本一致性章节的开篇就 使用kafka 为例讲了一个 做一个强一致的系统有多难。
