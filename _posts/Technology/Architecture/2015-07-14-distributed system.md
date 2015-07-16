---

layout: post
title: 分布式系统 小结
category: 技术
tags: Architecture
keywords: 分布式系统

---

## 简介

## hdfs

副本

## mesos/mapreduce

scheduler/executor

## zookeeper/etcd

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










[An Introduction to Mesosphere]: https://www.digitalocean.com/community/tutorials/an-introduction-to-mesosphere
[http://mesos.apache.org/documentation/latest/mesos-frameworks/ ]: http://mesos.apache.org/documentation/latest/mesos-frameworks/ 
[示例代码]: https://github.com/qiankunli/mesos/