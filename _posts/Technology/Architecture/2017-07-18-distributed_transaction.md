---

layout: post
title: 分布式事务
category: 技术
tags: Architecture
keywords: 分布式事务

---

## 简介

笔者相关的其它两篇文章

[JTA与TCC](http://qiankunli.github.io/2016/05/21/tcc.html)

脉络

1. 数据库事务
2. 分布式事务
3. XA
4. 放弃强一致性、达到最终一致性 [分布式服务框架之服务化最佳实践](http://www.infoq.com/cn/articles/servitization-best-practice)


[关于分布式事务、两阶段提交协议、三阶提交协议](http://www.hollischuang.com/archives/681)

[分布式事务的典型处理方式:2PC、TCC、异步确保和最大努力型](http://kaimingwan.com/post/fen-bu-shi/fen-bu-shi-shi-wu-de-dian-xing-chu-li-fang-shi-2pc-tcc-yi-bu-que-bao-he-zui-da-nu-li-xing)


github 案例项目[moonufo/galaxyLight](https://github.com/moonufo/galaxyLight)

## 从一致性问题开始

[关于分布式一致性的探究](http://www.hollischuang.com/archives/663)


[分布式系统的CAP理论](http://www.hollischuang.com/archives/666): 一致性指“all nodes see the same data at the same time”，即更新操作成功并返回客户端完成后，所有节点在同一时间的数据完全一致。==> 当更新操作完成之后，任何多个（节点）后续进程或者线程的访问都会返回最新的更新过的值。

从client和server的角度看

1. 从客户端来看，一致性主要指的是多并发访问时更新过的数据如何获取的问题。
2. 从服务端来看，则是更新如何复制分布到整个系统，以保证数据最终一致。

数据一致性

1. 在数据库系统中，通常指关联数据之间的逻辑关系是否正确和完整。
2. 在分布式系统中，指的是由于数据的复制，不同数据节点中的数据内容是否完整并且相同。

## 一致性、XA、2pc/3pc paxos的关系

[2PC/3PC到底是啥](http://www.bijishequ.com/detail/49467?p=)

XA 是 X/Open DTP 定义的交易中间件与数据库之间的接口规范（即接口函数），交易中间件用它来通知数据库事务的开始、结束以及提交、回滚等。 XA 接口函数由数据库厂商提供。 
二阶提交协议和三阶提交协议就是根据这一思想衍生出来的，而**分布式事务从广义上来讲也是一致性的一种表现，所以2PC/3PC也可以叫一致性协议**。

在真实的应用中，尽管有系统使用2PC/3PC协议来作为数据一致性协议，但是比较少见，更多的是作为实现**数据更新原子性手段**出现。

为什么2PC/3PC没有被广泛用在保证数据的一致性上，主要原因应该还是它本身的缺陷，所有经常看到这句话：there is only one consensus protocol, and that’s Paxos” – all other approaches are just broken versions of Paxos. 意即世上只有一种一致性算法，那就是Paxos。


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

### 在代码上的表现（待整理）

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

tcc事务处理

	