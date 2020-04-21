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

假设KV集群有三台机器，机器之间相互通信，把自己的值传播给其他机器，三个客户端并发的向集群发送三个请求，值X 应该是多少？是多少没关系，135都行，向一个client返回成功、其它client返回失败（或实际值）也可，关键是Node1、Node2、Node3 一致。

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

每个Node 同时充当了两个角色：Proposer和Acceptor，在实现过程中， 两个角色在同一个进程里面。专门提一个Proposer概念（只是一个角色概念）的好处是：对业务代码没有侵入性，也就是说，我们不需要在业务代码中实现算法逻辑，就可以像使用数据库一样访问后端的数据。

![](/public/upload/distribute/paxos_role.jpg)


||proposer|acceptor||
|---|---|---|---|
|prepare阶段|proposer 生成全局唯一且自增的proposal id，广播propose<br>只广播proposal id即可，无需value|Acceptor 收到 propose 后，做出“两个承诺，一个应答”<br>1. 不再应答 proposal id **小于等于**当前请求的propose<br>2. 不再应答 proposal id **小于** 当前请求的 accept<br>3. 若是符合应答条件，返回已经accept 过的提案中proposal id最大的那个 propose 的value 和 proposal id， 没有则返回空值|proposer 通过prepare请求来发现之前被大多数节点通过的提案|
|accept阶段|提案生成规则<br>1. 从acceptor 应答中选择proposalid 最大的value 作为本次的提案<br>2. 如果所有的应答的天value为空，则可以自己决定value|在不违背“两个承诺”的情况下，持久化当前的proposal id和value|

Paxos 是一个“不断循环”的2pc，两个阶段都可能会失败，从0开始，陷入“不断”循环，即“活锁”问题。

**Proposer 之间并不直接交互**，Acceptor除了一个“存储”的作用外，还有一个信息转发的作用。**从Acceptor的视角看**，basic-paxos 及 multi-paxos 选举过程是协商一个值，每个Proposer提出的value 都可能不一样。所以第一阶段，先经由Acceptor将**已提交的**ProposerId 最大的value 尽可能扩散到Proposer（即决定哪个Proposer 是“意见领袖”）。第二阶段，再将“多数意见”形成“决议”（Acceptor持久化value）

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

生活中面临的任何选择，最后都可以转换为一个问题：你想要什么（以及为此愿意牺牲什么）。任何不一致问题， 本质上都可以转换为一个一致问题。 一个队伍谁当老大 可以各不服气，但大家可以对“多票当选”取得一致，这就是所谓“民主”。**你可以不尊重老大，但必须尊重价值观**。而在basic-poxos 中，底层的“价值观”就是：**谁的ProposerId 大谁的优先级更高**。有了“优先级”/“价值观” 就可以让“无序”的世界变“有序”。 

## multi-paxos

《分布式协议与算法实战》Multi-Paxos 是一种思想，不是算法，缺失实现算法的必须编程细节。而 Multi-Paxos 算法是一个统称，它是指基于 Multi-Paxos 思想，通过多个 Basic Paxos 实例实现一系列值的共识的算法（比如 Chubby 的 Multi-Paxos 实现、Raft 算法等）。

当存在一批提案时，用Basic-Paxos一个一个决议当然也可以，但是每个提案都经历两阶段提交，显然效率不高。Basic-Paxos协议的执行流程针对每个提案（每条redo log）都至少存在三次网络交互：1. 产生log ID；2. prepare阶段；3. accept阶段。所以，Mulit-Paxos基于Basic-Paxos做了优化，引入领导者节点作为唯一提议者，以 Chubby 的 Multi-Paxos 实现为例

1. 主节点作为唯一提议者，这样就不存在多个提议者同时提交提案的情况，也就不存在提案冲突的情况
2. 在 Chubby 中，主节点是通过执行 Basic Paxos 算法，进行投票选举产生的，并且在运行过程中，主节点会通过不断续租的方式来延长租期（Lease）。
3. 在 Chubby 中实现了兰伯特提到的，“当领导者处于稳定状态时，省掉准备阶段，直接进入接受阶段”这个优化机制，减少非必须的协商步骤来提升性能。
4. 在 Chubby 中，为了实现了强一致性，所有的读请求和写请求都由主节点来处理。也就是说，只要数据写入成功，之后所有的客户端读到的数据都是一致的。

## Raft

Raft 算法属于 Multi-Paxos 算法，它是在兰伯特 Multi-Paxos 思想的基础上，做了一些简化和限制，比如增加了日志必须是连续的，只支持领导者、跟随者和候选人三种状态，在理解和算法实现上都相对容易许多。Raft 算法是现在分布式系统开发首选的共识算法。绝大多数选用 Paxos 算法的系统（比如 Cubby、Spanner）都是在 Raft 算法发布前开发的，当时没得选；而全新的系统大多选择了 Raft 算法（比如 Etcd、Consul、CockroachDB）。

![](/public/upload/distribute/raft_copy_log.png)

Raft协议比paxos的优点是 容易理解，容易实现。它强化了leader的地位，**把整个协议可以清楚的分割成两个部分**，并利用日志的连续性做了一些简化：

1. Leader在时。由Leader向Follower同步日志
2. Leader挂掉了，选一个新Leader
    1. 在初始状态下，集群中所有的节点都是跟随者的状态。
    2. Raft 算法实现了随机超时时间的特性。也就是说，每个节点等待领导者节点心跳信息的超时时间间隔是随机的。等待超时时间最小的节点（以下称节点 A），它会最先因为没有等到领导者的心跳信息，发生超时。
    3. 这个时候，节点 A 就增加自己的任期编号，并推举自己为候选人，先给自己投上一张选票，然后向其他节点发送请求投票 RPC 消息，请它们选举自己为领导者。
    4. 如果其他节点接收到候选人 A 的请求投票 RPC 消息，在编号为 1 的这届任期内，也还没有进行过投票，那么它将把选票投给节点 A，并增加自己的任期编号。
    5. 如果候选人在选举超时时间内赢得了大多数的选票，那么它就会成为本届任期内新的领导者。节点 A 当选领导者后，他将周期性地发送心跳消息，通知其他服务器我是领导者，阻止跟随者发起新的选举，篡权。

在 Raft 算法中，约定了很多规则，主要有这样几点

1. 领导者周期性地向所有跟随者发送心跳消息，通知大家我是领导者，阻止跟随者发起新的选举。
2. 如果在指定时间内，跟随者没有接收到来自领导者的消息，那么它就认为当前没有领导者，推举自己为候选人，发起领导者选举。
3. 在一次选举中，赢得大多数选票的候选人，将晋升为领导者。
4. 在一个任期内，领导者一直都会是领导者，直到它自身出现问题（比如宕机），或者因为网络延迟，其他节点发起一轮新的选举。
5. 在一次选举中，每一个服务器节点最多会对一个任期编号投出一张选票，并且按照“先来先服务”的原则进行投票。
6. 日志完整性高的跟随者（也就是最后一条日志项对应的任期编号值更大，索引号更大），拒绝投票给日志完整性低的候选人。
7. 如果一个候选人或者领导者，发现自己的任期编号比其他节点小，那么它会立即恢复成跟随者状态。
8. 如果一个节点接收到一个包含较小的任期编号值的请求，那么它会直接拒绝这个请求。

在议会选举中，常出现未达到指定票数，选举无效，需要重新选举的情况。在 Raft 算法的选举中，也存在类似的问题，那它是如何处理选举无效的问题呢？其实，Raft 算法巧妙地使用随机选举超时时间的方法，把超时时间都分散开来，在大多数情况下只有一个服务器节点先发起选举，而不是同时发起选举，这样就能减少因选票瓜分导致选举失败的情况。**如何避免候选人同时发起选举？**

1. 跟随者等待领导者心跳信息超时的时间间隔，是随机的；
2. 如果候选人在一个随机时间间隔内，没有赢得过半票数，那么选举无效了，然后候选人发起新一轮的选举，也就是说，等待选举超时的时间间隔，是随机的。

## 其它

[分布式系统原理介绍](https://mp.weixin.qq.com/s/3eL7CcwMDhwmPJ6VkQWtRw)副本控制协议指按特定的协议流程控制副本数据的读写行为，使得副本满足一定的可用性和一致性要求的分布式协议。

1. 中心化（centralized）副本控制协议，由一个中心节点协调副本数据的更新、维护副本之间的一致性。所有的副本相关的控制交由中心节点完成。并发控制将由中心节点完成，从而使得一个分布式并发控制问题，简化为一个单机并发控制问题。所谓并发控制，即多个节点同时需要修改副本数据时，需要解决“写写”、“读写”等并发冲突。单机系统上常用加锁等方式进行并发控制。
2. 去中心化（decentralized）副本控制协议

kafka 因为有更明确地业务规则，有一个专门的coordinator，选举过程进一步简化，复制log的逻辑基本一致。《软件架构设计》 多副本一致性章节的开篇就 使用kafka 为例讲了一个 做一个强一致的系统有多难。
