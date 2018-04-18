---

layout: post
title: 现有分布式项目小结
category: 技术
tags: Architecture
keywords: 分布式系统

---

## 简介

- 分布式计算
- 分布式文件系统
- 分布式配置系统
- 分布式缓存系统

分布式（distributed）是指在多台不同的服务器中部署不同的服务模块，通过远程调用协同工作，对外提供服务。

集群（cluster）是指在多台不同的服务器中部署相同应用或服务模块，构成一个集群，通过负载均衡设备对外提供服务。

分布式系统的难点就在于：一个正常的业务系统普遍包含元数据，比如操作系统有文件表供各个进程使用。在一个操作事务结束前，很多的元数据需要在全部节点上保持一致。


## 角色组成

通常在分布式系统中，最典型的是master/slave模式（主备模式），在这种模式中，我们把能够处理所有写操作的机器称为master，把所有通过异步复制方式获取最新数据，并提供读服务的机器称为slave机器。

而有些软件，比如zookeeper，这些概念被颠覆了。zk引入leader，follower和observer三种角色，所有机器通过选举过程选定一台称为leader的机器，为客户端提供读和写服务。follower和observer都能提供读服务，区别在于，observer不参与leader选举过程，也不参与写操作的“过半写成功”策略，因此observer可以在不影响写性能的情况下（leader写入的数据要同步到follower才算写成功）提升集群的读性能。摘自《从paxos到zookeeper》



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