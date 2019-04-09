---

layout: post
title: 串一串一致性协议
category: 技术
tags: Distribute
keywords: 一致性协议

---

## 简介

* TOC
{:toc}

## 什么是一致性

[关于分布式一致性的探究](http://www.hollischuang.com/archives/663)

从client和server的角度看

1. 从客户端来看，一致性主要指的是多并发访问时更新过的数据如何获取的问题。
2. 从服务端来看，则是更新如何复制分布到整个系统，以保证数据最终一致。

数据一致性

1. 在数据库系统中，通常指关联数据之间的逻辑关系是否正确和完整。
2. 在分布式系统中，指的是由于数据的复制，不同数据节点中的数据内容是否完整并且相同。

[分布式系统的CAP理论](http://www.hollischuang.com/archives/666): 一致性指“all nodes see the same data at the same time”，即更新操作成功并返回客户端完成后，所有节点在同一时间的数据完全一致。==> 当更新操作完成之后，任何多个（节点）后续进程或者线程的访问都会返回最新的更新过的值。

[被误用的“一致性”](http://blog.kongfy.com/2016/08/%E8%A2%AB%E8%AF%AF%E7%94%A8%E7%9A%84%E4%B8%80%E8%87%B4%E6%80%A7/) 一致性和算法的误区：我们常说的“一致性（Consistency）”在分布式系统中指的是副本（Replication）问题中对于同一个数据的多个副本，其对外表现的数据一致性，如线性一致性、因果一致性、最终一致性等，都是用来描述副本问题中的一致性的。

[分布式共识(Consensus)：Viewstamped Replication、Raft以及Paxos](http://blog.kongfy.com/2016/05/%E5%88%86%E5%B8%83%E5%BC%8F%E5%85%B1%E8%AF%86consensus%EF%BC%9Aviewstamped%E3%80%81raft%E5%8F%8Apaxos/)分布式共识问题，简单说，**就是在一个或多个进程提议了一个值应当是什么后，使系统中所有进程对这个值达成一致意见。** 

这样的协定问题在分布式系统中很常用，比如说选主（Leader election）问题中所有进程对Leader达成一致；互斥（Mutual exclusion）问题中对于哪个进程进入临界区达成一致；原子组播（Atomic broadcast）中进程对消息传递（delivery）顺序达成一致。对于这些问题有一些特定的算法，但是，**分布式共识问题试图探讨这些问题的一个更一般的形式，如果能够解决分布式共识问题，则以上的问题都可以得以解决。**

小结一下就是，一致性是一个结果，共识是一个算法，通常被用于达到一致性的结果。

在《区块链核心算法解析》中，则采用另一种描述方式：对于一组节点，如果所有节点均以相同的顺序执行一个（可能是无限的）命令序列c1,c2,c3...，则这组节点 实现了状态复制。

[实践丨分布式事务解决方案汇总：2PC、消息中间件、TCC、状态机+重试+幂等](http://www.10tiao.com/html/551/201904/2652561205/1.html)《软件架构设计：大型网站技术架构与业务架构融合之道》把一致性问题分为了两大类：事务一致性和多副本一致性。

李运华 《从0到1学架构》 关于Robert Greiner 两篇文章的对比 建议细读，要点如下

1. 不是所有的分布式系统都有 cap问题，必须interconnected 和 share data。比如一个简单的微服务系统 没有shar data，便没有cap 问题。
2. 强调了write/read pair 。这跟上一点是一脉相承的。cap 关注的是对数据的读写操作，而不是分布式系统的所有功能。

## 一致性理论

以下来自对两个付费专栏的整理

### 左耳听风

![](/public/upload/distribute/cap.PNG)

![](/public/upload/distribute/cap_2.png)

![](/public/upload/distribute/cap_3.PNG)

### 从0到1学架构

下文来自 李运华 极客时间中的付费专栏《从0到1学架构》。

Robert Greiner （http://robertgreiner.com/about/） 对CAP 的理解也经历了一个过程，他写了两篇文章来阐述CAP理论，第一篇被标记为outdated

||第一版|第二版|
|---|---|---|
|地址|http://robertgreiner.com/2014/06/cap-theorem-explained/|http://robertgreiner.com/2014/08/cap-theorem-revisited/|
|理论定义|Any distributed system cannot guaranty C, A, and P simultaneously.|In a distributed system (a collection of interconnected nodes that share data.), you can only have two out of the following three guarantees across a write/read pair: Consistency, Availability, and Partition Tolerance - one of them must be sacrificed.|
|中文||在一个分布式系统（指互相连接并共享数据的节点的集合）中，当涉及读写操作时，只能保证一致性（Consistence）、可用性（Availability）、分区容错性（Partition Tolerance）三者中的两个，另外一个必须被牺牲。|
|一致性|all nodes see the same data at the same time|A read is guaranteed to return the most recent write for a given client 总能读到 最新写入的新值|
|可用性|Every request gets a response on success/failure|A non-failing node will return a reasonable response within a reasonable amount of time (no error or timeout)|
|分区容忍性|System continues to work despite message loss or partial failure|The system will continue to function when network partitions occur|

李运华 文中 关于Robert Greiner 两篇文章的对比 建议细读： 要点如下

1. 不是所有的分布式系统都有 cap问题，必须interconnected 和 share data。比如一个简单的微服务系统 没有share data，便没有cap 问题。
2. 强调了write/read pair 。这跟上一点是一脉相承的。cap 关注的是对数据的读写操作，而不是分布式系统的所有功能。

### 其它材料

[从CAP理论到Paxos算法](http://blog.longjiazuo.com/archives/5369?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io) 基本要点：

1. cap 彼此的关系。提高分区容忍性的办法就是一个数据项复制到多个节点上，那么出现分区之后，这一数据项就可能分布到各个区里。分区容忍就提高了。然而，要把数据复制到多个节点，就会带来一致性的问题，就是多个节点上面的数据可能是不一致的。要保证一致，每次写操作就都要等待全部节点写成功，而这等待又会带来可用性的问题。
2. cap 着力点。**网络分区是既成的现实，于是只能在可用性和一致性两者间做出选择。**在工程上，我们关注的往往是如何在保持相对一致性的前提下，提高系统的可用性。
3. 还是读写问题。[多线程](http://qiankunli.github.io/2014/10/09/Threads.html) 提到，多线程本质是一个并发读写问题，数据库系统中，为了描述并发读写的安全程度，还提出了隔离性的概念。具体到cap 理论上，副本一致性本质是并发读写问题（A主机写入的数据，B主机多长时间可以读到）。2pc/3pc 本质解决的也是如此 ==> A主机写入的数据，成功与失败，B主机多长时间可以读到，然后决定自己的行为。

《反应式设计模式》Eric Brewer评论道：CAP 的这个表述是为了让设计人员的思想敞开至更广泛的系统和权衡。但”3个只能满足2个“的表述总是具有误导性，因为它简化了属性之间的紧密联系。PS：比如系统有几十毫秒的不一致在实践中是可以接受的，进而在CAP三个特性上可以适当放宽。


## 一致性、XA、2pc/3pc paxos的关系

[2PC/3PC到底是啥](http://www.bijishequ.com/detail/49467?p=)

XA 是 X/Open DTP 定义的交易中间件与数据库之间的接口规范（即接口函数），交易中间件用它来通知数据库事务的开始、结束以及提交、回滚等。 XA 接口函数由数据库厂商提供。 
二阶提交协议和三阶提交协议就是根据这一思想衍生出来的，而**分布式事务从广义上来讲也是一致性的一种表现：**事务是数据库特有的概念，分布式事务最初起源于处理多个数据库之间的数据一致性问题，但随着IT技术的高速发展，大型系统中逐渐使用SOA服务化接口替换直接对数据库操作，所以如何保证各个SOA服务之间的数据一致性也被划分到分布式事务的范畴。来自[以交易系统为例，看分布式事务架构的五大演进](http://www.sohu.com/a/134477290_487514)。所以2PC/3PC也可以叫一致性协议。

在真实的应用中，尽管有系统使用2PC/3PC协议来作为数据一致性协议，但是比较少见，更多的是作为实现**数据更新原子性手段**出现。

为什么2PC/3PC没有被广泛用在保证数据的一致性上？主要原因应该还是它本身的缺陷，所有经常看到这句话：there is only one consensus protocol, and that’s Paxos” – all other approaches are just broken versions of Paxos. 意即世上只有一种一致性算法，那就是Paxos。


更新想要被其它node感知到，就要提交更新，各个一致性协议的不同、缺点，也主要体现在提交方式上：

1. 单数据库事务
2. 多数据库事务，一个数据源更新操作已提交，另一个数据源更新操作失败，则数据不一致。so，应该在所有数据源更新操作完之后，再提交。
3. 基于后置提交的多数据库事务，一个数据源提交成功，另一个数据源提交失败，则数据不一致。
3. XA事务，将提交分为两个步骤，预提交、确认提交。前一个步骤“重”，完成大部分提交操作。后一个步骤“轻”，失败概率很低。so，依然会有部分数据源确认提交失败的问题，不过因为概率低，数据量小，可以通过记录日志转向人工处理。
4. 从数据库领域延伸到微服务领域，分布式事务，TCC。
5. 放弃强一致性、达到最终一致性。初步解决一致性问题后，主要通过异步补偿机制进行部分妥协，提高性能。

[分布式服务框架之服务化最佳实践](http://www.infoq.com/cn/articles/servitization-best-practice)


### Paxos

Paxos算法

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

### 思维线条

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
		4. 广播多轮投票。随机共识算法，不准确描述：假设只对01取得共识，第一轮每个节点随机选定一个值，广播给其它所有节点，节点收到超过半数其它节点的值，如果恰好是同一个值，则节点改变自己本轮的“意见”，重新广播该值。

tips

1. paxos 无法保证确定性，即理论上存在一直无法达成一致， 不停地投票的情况
2. paxos/随机共识算法等 假定 参与节点都按规则 运行的基础上
3. FLP 原理：不存在一个确定性算法 总是能解决共识问题。
4. 对于zk，我们常从一个客户端访问者的角度来观察。实际上，zookeeper 工作时，假设存在zk1、zk2和zk3集群，且client1 访问zk1 和client2 访问zk2， 是很容易存在同一时刻，cleint1和client2读写同一个配置的。

拜占庭节点：节点可能不按规则行事，甚至故意发送错误数据，多个拜占庭节点也可能串谋。

基于拜占庭节点达成共识

2. 拜占庭容错（BFT）算法，一系列算法的统称。网络中节点的数量和身份必须是提前确定好的
3. POW，间接共识，先选谁说了算，再达成共识。

两个算法对cap的侧重有所不同

## 分布式系统中最容易被忽视的六大“暗流”

2018.12.2 补充

[分布式系统中最容易被忽视的六大“暗流”](https://mp.weixin.qq.com/s?__biz=MjM5MDE0Mjc4MA==&mid=2651011259&idx=1&sn=fff6663f391ae5cfc175dceeecb67829&chksm=bdbec0e88ac949fe0cf893306a0a61f27f0355d71a61927bfafb223a6663825c4b27e8a070d4&mpshare=1&scene=23&srcid=1202RkLFVYCNUXsR31Wrdvpq%23rd)

1. 网络并不是可靠的，光缆被挖断的事件相信你也看到过不是一两次了，其它的诸如带宽打满 也会导致网络中断
2. 不同节点之间的通信是存在延迟的。由此可得，并不是将一个集中式系统拆分得越散，系统就越快。当中存在的延迟，不容忽视。如果是基于提速为目的的拆分，必须支持并行处理。
3. 带宽是有上限的
4. 分布式并不直接意味着是“敏捷”了,拆分后需要做的额外工作（比如：监控告警系统、配置中心、服务发现，以及批量部署、持续集成，甚至 DevOps 等）如果没做好，可能会导致不是更快，而是更慢。
5. 数据由一份冗余成多份后如何保持一致。由第二点导致。这个概念在软件领域被定义为「数据一致性」。
6. 整个系统的不同部分可能是异构的。可以是装有不同操作系统的服务器、不同的数据库产品、用不同的语言开发的产品等等。你要思考如何通过专制的方式进行标准化，来屏蔽这些差异带来的复杂度影响，使得有更多的精力投入到有价值的地方去。
	

## 从容错性强弱的角度来串一下一致性协议

本小节来自 [漫谈分布式系统、拜占庭将军问题与区块链](http://zhangtielei.com/posts/blog-consensus-byzantine-and-blockchain.html)

理解问题本身比理解问题的答案要重要的多。

||节点|目标|备注|
|---|---|---|---|
|传统分布式一致性问题|可信|在一个去中心化的网络中（考虑到网络延迟、宕机等情况），各个节点之间最终能够对于提议达成一致。
|拜占庭将军问题|不可信|所有忠诚的副官最终都接受相同的命令<br>如果主将是忠诚的，那么所有忠诚的副官都接受主将发出的命令|作者的推理是少有的比较易懂的了，建议看原文|

解决拜占庭问题，笔者感觉有两个点

1. 广播，一个将军的命令发给所有其它将军，
2. 转发，A将收到的B 将军的信息也转发给其他所有将军

尽可能保证所有人都是“知情”的（假设不考虑网络问题，只考虑节点不可信问题），A 既知道B 给自己的意见，也知道B 给其他将军的意见 ==> 如果大家遵循相同的算法 ==> A 既知道自己的决策结果，也知道其他人的决策结果。


从容错性角度来串一下一致性协议

||表现|算法|
|---|---|---|
|非拜占庭错误|节点故障或网络不通，只是收不到它的消息了，而不会收到来自它的错误消息。相反，只要收到了来自它的消息，那么消息本身是「忠实」的。|paxos|
|拜占庭错误|1. 叛徒的恶意行为，在不同的将军看来，叛徒可能发送完全不一致的作战提议。<br>2. 虽然并非恶意，出现故障（比如信道不稳定）导致的随机错误或消息损坏||
|拜占庭将军问题|叛徒发送前后不一致的作战提议，属于拜占庭错误；<br>而不发送任何消息，属于非拜占庭错误。|BTF|

BFT的算法应该可以解决任何错误下的分布式一致性问题，也包括Paxos所解决的问题。那为什么不统一使用BFT的算法来解决所有的分布式一致性问题呢？为什么还需要再费力气设计Paxos之类的一些算法呢？

1. 提供BFT这么强的错误容忍性，肯定需要付出很高的代价。比如需要消息的大量传递。对于运行环境的假设(assumption)
2. 具体到Lamport在论文中给出的解决「拜占庭将军问题」的算法，它还对运行环境的假设(assumption)有更强的要求。比如BTF 有一条： The absence of a message can be detected ==> 依赖某种超时机制 ==> 各节点时钟同步 ==> 同步模型。

区块链到底是什么？有人说是个无法篡改的超级账本，也有人说是个去中心化的交易系统，还有人说它是构建数字货币的底层工具。但是，从技术的角度来说，它首先是个解决了拜占庭将军问题的分布式网络，**在完全开放的环境中，实现了数据的一致性和安全性。而其它的属性，都附着于这一技术本质之上。**
