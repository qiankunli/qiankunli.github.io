---

layout: post
title: 分布式系统小结
category: 技术
tags: Architecture
keywords: 分布式系统

---

## 简介（未完待续）

- 分布式计算
- 分布式文件系统
- 分布式配置系统
- 分布式缓存系统

分布式（distributed）是指在多台不同的服务器中部署不同的服务模块，通过远程调用协同工作，对外提供服务。

集群（cluster）是指在多台不同的服务器中部署相同应用或服务模块，构成一个集群，通过负载均衡设备对外提供服务。


## 一致性问题

数据一致性其实是数据库系统中的概念。我们可以简单的把一致性理解为正确性或者完整性，那么数据一致性通常指**关联数据之间的逻辑关系是否正确和完整**。我们知道，在数据库系统中通常用事务（访问并可能更新数据库中各种数据项的一个程序执行单元）来保证数据的一致性和完整性。而在分布式系统中，数据一致性可以由分布式事务来保证，但往往代价较大，于是人们系统分析归纳了各种一致性类型和级别，根据需要，达到一个弱化后的一致性。

所谓的一致性，就是假设有a,b,c三个节点（关联的数据分别存在abc撒个节点上），a发生了一件事，希望bc能够立刻感知到这件事并作出自己的反应（或者是数据复制、或者完成下一步工作），bc的反应反过来会影响a的反应（比如操作失败，所以a要回滚）

这里有几个难点：

1. a告诉bc做了什么，如何确保bc收到了。bc收到了，a如何知道bc收到了。
2. a告诉bc多个消息，如何确保bc收全了，并且顺序没错（后发的不一定后到喔）

根据场景的不同，一致性问题有一些具体的表现：

1. 将解决一致性问题的组件从整个系统中独立出来（比如zookeeper/etcd等），暴露增删改查监的接口，在每个节点上运行，专门负责维护数据的一致性。就好比古代谈亲事，本人不出面，由双方家长见面商定问题。对应上面的abc三个节点，一致性组件不一定是非得是三个节点，可本地访问，可远程访问。

2. 对于支付——订单——仓库系统这类场景，提供一个协调者的角色。就好比谈亲事时，双方不直接接触，而是通过媒人表达诉求和获知结果。
3. 对于数据库跨机房主备同步，（未完待续）

如果说，两人面对面交流是“理想状态”（一致状态），只要不是两个人面对面交流问题，那么信息的传递和表达就会错漏和失真。

题外话，在mysql中，事务是默认开启的，即执行一条命令就是一个事务。即，命令执行时对数据的独占等环境准备，是事务帮它做的。

## 一致性算法（待整理）

两阶段提交、三阶段提交，paxos

这里重点提一下paxos

[一步一步理解Paxos算法](http://mp.weixin.qq.com/s?__biz=MjM5MDg2NjIyMA==&mid=203607654&idx=1&sn=bfe71374fbca7ec5adf31bd3500ab95a&key=8ea74966bf01cfb6684dc066454e04bb5194d780db67f87b55480b52800238c2dfae323218ee8645f0c094e607ea7e6f&ascene=1&uin=MjA1MDk3Njk1&devicetype=webwx&version=70000001&pass_ticket=2ivcW%2FcENyzkz%2FGjIaPDdMzzf%2Bberd36%2FR3FYecikmo%3D)

paxos侧重于快速且正确的在分布式系统中对某个数据的值达成一致。

首先，paxos将角色分为proposer和acceptor。

proposer提交值，acceptor最终只能批准一个值，最后proposer从acceptor中拿到确定的值。

1. 最简单场景，一个acceptor，每个proposer将自己的值发给acceptor，acceptor看谁选的多，就告诉所有proposer选定的值（或者proposer再问一次）。很明显，acceptor存在单点问题。

	 此问题是2pc和3pc的翻版，2pc和3pc面向的是资源调用方和多个资源提供方，资源提供方告诉资源调用方自己的操作状态，资源调用方告诉它执行还是回滚。只不过2pc和3pc面向的场景更具体，根据acceptor（资源调用方）的反应进行操作，而不是设定值。

2. 多个acceptor，每个proposer将自己的值发给所有的acceptor。acceptor会接到多个值，以谁为准呢？

	a. 第一次接到的值为准

3. 这种单向的提交，无论采取什么策略，都可能acceptor对proposer提交的多个值的批准数一样。所以，proposer向acceptor提交前，先询问acceptor有没有被批准的值，有则提交该值，无则提交自己的值。但因为存在通信延迟的问题，proposer1和proposer2在询问阶段都得知acceptor还没有批准值，因此提交了自己的值。而因为网络延迟，无法判定谁第一次提交。
4. 现在呢，我们可以意识到：

	   1. proposer和acceptor是一个多对多的关系
	   2. proposer和acceptor的策略不再是单一的提交和批准，还涉及到比较复杂的逻辑判断和策略。

	  ![Alt text](/public/upload/architecture/paxos_flow.png)
	  
	 
现在我们在回过头看下paxos解决的问题（paxos议会问题），大家在一个厅里投票和在一个分布的环境的里传信，所面临的问题是有所不同的。proposer之间和acceptor之间不用交互，仅通过proposer和acceptor的多对多交互（类似于2pc）即可协商一致。



## 一些实例

### mesos/mapreduce等

scheduler/executor

### hdfs/glusterfs等

1. 如何记录数据逻辑与物理位置的映像关系。是根据算法，还是用将元数据集中或分布式存储
2. 通过副本来提高可靠性
3. 适合存储大文件还是小文件。小文件过多导致的dfs元数据过多是否会成为性能瓶颈。
4. 如何访问存储在之上的文件，比如是否支持NFS（Network File System）


### zookeeper/etcd等

主从

使用场景

http://blog.csdn.net/miklechun/article/details/32076723

### memcache

一致性哈希算法


http://blog.csdn.net/kongqz/article/details/6695417

### 特点

如果采用主从模式

- slave信息的汇集
- slave有效性检测
- Scheduler与Executor通信（包括task status汇报等）
- 与客户端交互模式

    - master包办一切与客户端的交互
    - client通过master得到元信息，然后直接与slave交互，进行具体的数据操作。




## 引用

[分布式系统的数据一致性和处理顺序问题](http://www.nginx.cn/4331.html)

[初识分布式系统](http://www.hollischuang.com/archives/655)

[关于分布式一致性的探究](http://www.hollischuang.com/archives/663)

[An Introduction to Mesosphere](https://www.digitalocean.com/community/tutorials/an-introduction-to-mesosphere)

[http://mesos.apache.org/documentation/latest/mesos-frameworks/ ](http://mesos.apache.org/documentation/latest/mesos-frameworks/)

[示例代码](https://github.com/qiankunli/mesos/)