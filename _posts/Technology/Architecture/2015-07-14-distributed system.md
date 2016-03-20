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

## mesos/mapreduce等

scheduler/executor

## hdfs/glusterfs等

1. 如何记录数据逻辑与物理位置的映像关系。是根据算法，还是用将元数据集中或分布式存储
2. 通过副本来提高可靠性
3. 适合存储大文件还是小文件。小文件过多导致的dfs元数据过多是否会成为性能瓶颈。
4. 如何访问存储在之上的文件，比如是否支持NFS（Network File System）


## zookeeper/etcd等

主从

使用场景

http://blog.csdn.net/miklechun/article/details/32076723

## memcache

一致性哈希算法


http://blog.csdn.net/kongqz/article/details/6695417

## 特点

如果采用主从模式

- slave信息的汇集
- slave有效性检测
- Scheduler与Executor通信（包括task status汇报等）
- 与客户端交互模式

    - master包办一切与客户端的交互
    - client通过master得到元信息，然后直接与slave交互，进行具体的数据操作。




## 引用





[An Introduction to Mesosphere]: https://www.digitalocean.com/community/tutorials/an-introduction-to-mesosphere
[http://mesos.apache.org/documentation/latest/mesos-frameworks/ ]: http://mesos.apache.org/documentation/latest/mesos-frameworks/ 
[示例代码]: https://github.com/qiankunli/mesos/