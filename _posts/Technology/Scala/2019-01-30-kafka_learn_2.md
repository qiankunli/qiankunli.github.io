---

layout: post
title: 《Apache Kafka源码分析》——server
category: 技术
tags: Scala
keywords: Scala  akka

---

## 前言（未完成）

* TOC
{:toc}

建议先阅读下[《Apache Kafka源码分析》——Producer与Consumer](http://qiankunli.github.io/2017/12/08/kafka_learn_1.html)

![](/public/upload/scala/kafka_server_framework.jpg)

对kafka好奇的几个问题：

1. 数据在磁盘上是如何存储的？

## 网络层

kafka client (producer/consumer) 与kafka server通信时使用自定义的协议，一个线程 一个selector 裸调java NIO 进行网络通信。 

面对高并发、低延迟的需求，kafka 服务端使用了多线程+多selector ，参见[java nio的多线程扩展](http://qiankunli.github.io/2015/06/19/java_nio_2.html)

![](/public/upload/netty/kafka_server_nio.jpg)

这是一款常见的服务端 实现

1. 分层实现，网络io 部分负责读写数据，并将数据序列化/反序列化为协议请求。
2. 协议请求交给 上层处理， API层就好比 tomcat 中的servlet

## 日志存储

### 整体思想

接收到客户端的请求之后，不同的系统（都要读写请求数据）有不同的反应

1. redis 读取/写入内存
2. mysql 读取/写入本地磁盘
3. web 系统，转手将请求数据处理后写入数据；或者查询数据库并返回响应信息，其本身就是一个中转站。

读写本地磁盘的系统 一般有考虑几个问题

1. 磁盘只能存储二进制数据或文本数据，文本数据你可以一行一行读取/写入。二进制数据则要求制定一个文件格式，一次性读写特定长度的数据。
2. 如果文件较大，为加快读写速度，还要考虑读写索引文件
3. 内存是否需要缓存热点磁盘数据

建议和mysql 对比学习下  [《mysql技术内幕》笔记1](http://qiankunli.github.io/2017/10/31/inside_mysql1.html)

|逻辑概念|对应的物理概念|备注|
|---|---|---|
|Log|目录|目录的命名规则`<topic_name>_<partition_id>`|
|LogSegment|一个日志文件、一个索引文件|命名规则`[baseOffset].log`和`[baseOffset].index` <br> baseOffset 是日志文件中第一条消息的offset|
|offset|消息在日志文件中的偏移|类似于数据库表中的主键<br> 由索引文件记录映射关系|

![](/public/upload/scala/kafka_index_file.jpg)

### 代码实现








