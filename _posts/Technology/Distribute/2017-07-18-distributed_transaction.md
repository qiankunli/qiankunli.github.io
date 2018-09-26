---

layout: post
title: 分布式事务
category: 技术
tags: Distribute
keywords: 分布式事务

---

## 简介

笔者相关的其它两篇文章

[JTA与TCC](http://qiankunli.github.io/2016/05/21/tcc.html)

其它引用

[关于分布式事务、两阶段提交协议、三阶提交协议](http://www.hollischuang.com/archives/681)

[分布式事务的典型处理方式:2PC、TCC、异步确保和最大努力型](http://kaimingwan.com/post/fen-bu-shi/fen-bu-shi-shi-wu-de-dian-xing-chu-li-fang-shi-2pc-tcc-yi-bu-que-bao-he-zui-da-nu-li-xing)


github 案例项目:

1. [QNJR-GROUP/EasyTransaction](https://github.com/QNJR-GROUP/EasyTransaction)
2. [moonufo/galaxyLight](https://github.com/moonufo/galaxyLight)

2018.9.26 补充：《左耳听风》中提到：

1. 对于应用层上的分布式事务一致性
	
	* 吞吐量大的最终一致性方案
	* 吞吐量小的强一致性方案：两阶段提交
2. 数据存储层解决这个问题的方式 是通过一些像paxos、raft或是nwr这样的算法和模型来解决。 PS：这可能是因为存储层 主要是副本一致性问题

## 从一致性问题开始

[关于分布式一致性的探究](http://www.hollischuang.com/archives/663)

从client和server的角度看

1. 从客户端来看，一致性主要指的是多并发访问时更新过的数据如何获取的问题。
2. 从服务端来看，则是更新如何复制分布到整个系统，以保证数据最终一致。

数据一致性

1. 在数据库系统中，通常指关联数据之间的逻辑关系是否正确和完整。
2. 在分布式系统中，指的是由于数据的复制，不同数据节点中的数据内容是否完整并且相同。

[分布式系统的CAP理论](http://www.hollischuang.com/archives/666): 一致性指“all nodes see the same data at the same time”，即更新操作成功并返回客户端完成后，所有节点在同一时间的数据完全一致。==> 当更新操作完成之后，任何多个（节点）后续进程或者线程的访问都会返回最新的更新过的值。

## 一致性、XA、2pc/3pc paxos的关系

该话题在另一篇博客 [分布式系统小结](http://qiankunli.github.io/2018/04/16/distributed_system_review.html) 也有阐述

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

## 2PC/3PC

[一致性协议](http://www.cnblogs.com/xrq730/p/4992198.html)

### 过程

正常情况

![](/public/upload/architecture/2pc_3pc.png)

异常情况

![](/public/upload/architecture/2pc_3pc_2.png)

2pc有很多问题，比如单点、同步阻塞等，此处我只讨论数据一致性问题：在二阶段提交协议的阶段二，即执行事务提交的时候，当协调者向所有的参与者发送Commit请求之后，发生了局部网络异常或者是协调者在尚未发送完Commit请求之前自身发生了崩溃，导致最终只有部分参与者收到了Commit请求。于是，这部分收到了Commit请求的参与者就会进行事务的提交，而其他没有收到Commit请求的参与者则无法进行事物提交，于是整个分布式系统便出现了数据不一致的现象。

正常情况

![](/public/upload/architecture/2pc_3pc_3.png)

异常情况

![](/public/upload/architecture/2pc_3pc_4.png)

![](/public/upload/architecture/2pc_3pc_5.png)

![](/public/upload/architecture/2pc_3pc_6.png)

**在分布式环境下，分布式系统的每一次请求和响应，存在特有的三态概念：即成功、失败、超时。**相对于2pc，3pc处理了timeout问题。但3pc在数据一致性上也有问题：在参与者接收到preCommit消息后，如果出现网络分区，此时协调者所在的节点和参与者无法进行正常的网络通信，在这种情况下，参与者依然会进行事物的提交，就可能出现不同节点成功or失败，这必然出现数据的不一致。

### 在代码上的表现

1. 声明式事务
2. 编程式事务

本地事务处理

	Connection conn = null; 
	try{
	    //若设置为 true 则数据库将会把每一次数据更新认定为一个事务并自动提交
	    conn.setAutoCommit(false);
	    // 将 A 账户中的金额减少 500 
	    // 将 B 账户中的金额增加 500 
	    conn.commit();
	}catch(){
	     conn.rollback();
	}

跨数据库事务处理

	UserTransaction userTx = null; 
	Connection connA = null; 
	Connection connB = null; 
	try{
	    userTx.begin();
	    // 将 A 账户中的金额减少 500 
	    // 将 B 账户中的金额增加 500 
	    userTx.commit();
	}catch(){
	     userTx.rollback();
	}
	
代码的封装跟2pc的实际执行过程有所不同，可以参见[JTA与TCC](http://qiankunli.github.io/2016/05/21/tcc.html)

从代码实现上，一般到2pc就可以保证大部分场景的一致性，然后框架着重点开始转向，通过异步、补偿等机制提高调用性能。

## 最终一致性

主线程逻辑

	func run(){
		1. rpc1
		2. send mq
		3. write db
		4. rpc2
	}

强一致性，主线程执行完run方法，即达到一致性，确切的说，是要么都成功，要么都失败。确切的说，对于每一个step，用户实现try、confirm、cancel逻辑，由框架负责这些方法在恰当的时间执行。

对于具体的业务，会有一些变种和要求：

1. 不需要confirm，比如send mq业务，try逻辑发就是了，没啥好confirm的。
2. 主线程要尽可能的快，不能等try rpc1和try rpc2结束，再confirm依次搞完。而是try完了，主线程直接搞别的。至于confirm（可以没有）、cancel由异步线程解决。



异步、补偿

目前主流触发异步数据补偿的方式有两种：

1. 使用消息队列实时触发数据补偿
2. 使用定时任务周期性触发数据补偿

[分布式系统事务一致性解决方案](http://www.infoq.com/cn/articles/solution-of-distributed-system-transaction-consistency)

[以交易系统为例，看分布式事务架构的五大演进](http://www.sohu.com/a/134477290_487514)