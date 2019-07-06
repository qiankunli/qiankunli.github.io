---

layout: post
title: 《Apache Kafka源码分析》——Producer与Consumer
category: 技术
tags: Scala
keywords: Scala  akka

---

## 前言

* TOC
{:toc}

## 生产者

The producer connects to any of the alive nodes and requests metadata about the leaders for the partitions of a topic. This allows the producer to put the message directly to the lead broker for the partition.

大纲是什么？

1. 线程模型
2. 发送流程

![](/public/upload/scala/kafka_producer.jpg)

![](/public/upload/scala/kafka_producer_object.png)

1. producer 实现就是 业务线程（可能多个线程操作一个producer对象） 和 io线程（sender，看样子应该是一个producer对象一个）生产消费的过程
2. 从producer 角度看，topic 分区数量以及 leader 副本的分布是动态变化的，Metadata 负责屏蔽相关细节，为producer 提供最新数据
3. 发送消息时有同步异步的区别，其实底层实现相同，都是异步。业务线程通过`KafkaProducer.send`不断向RecordAccumulator 追加消息，当达到一定条件，会唤醒Sender 线程发送RecordAccumulator 中的消息
4. ByteBuffer的创建和释放是比较消耗资源的，为了实现内存的高效利用，基本上每个成熟的框架或工具都有一套内存管理机制，对应到kafka 就是 BufferPool
5. 业务线程和io线程协作靠队列，为什么不直接用队列？

    1. RecordAccumulator acts as a queue that accumulates records into MemoryRecords instances to be sent to the server.The accumulator uses a bounded amount of memory and append calls will block when that memory is exhausted, unless this behavior is explicitly disabled.
    3. 用了队列，才可以batch和压缩。

6. If the request fails, the producer can automatically retry, though since we have specified etries as 0 it won't. Enabling retries also opens up the possibility of duplicates 

### bootstrap.servers

bootstrap.servers 指定了Producer 启动时要连接的 Broker 地址。通常指定 3～4 台就足以了。因为 Producer 一旦连接到集群中的任一台 Broker，就能拿到整个集群的 Broker 信息

在创建 KafkaProducer 实例时，生产者应用会在后台创建并启动一个名为 Sender 的线程，该 Sender线程开始运行时首先会创建与 Broker 的连接。

### 业务线程

producer 在KafkaProducer 与 NetworkClient 之间玩了多好花活儿？

![](/public/upload/scala/kafka_producer_send.png)

### sender 线程

![](/public/upload/scala/kafka_sender.png)


### 值得学习的地方——interceptor

在发送端，record发送和执行record发送结果的callback之前，由interceptor拦截

1. 发送比较简单，record发送前由interceptors操作一把。`ProducerRecord<K, V> interceptedRecord = this.interceptors == null ? record : this.interceptors.onSend(record)`
2. `Callback interceptCallback = this.interceptors == null ? callback : new InterceptorCallback<>(callback, this.interceptors, tp);`

对于底层发送来说，`doSend(ProducerRecord<K, V> record, Callback callback)`interceptors的加入并不影响（实际代码有出入，但大意是这样）。

### 值得学习的地方——反射的另个一好处

假设你的项目，用到了一个依赖jar中的类，但因为策略问题，这个类对有些用户不需要，自然也不需要这个依赖jar。此时，在代码中，你可以通过反射获取依赖jar中的类，避免了直接写在代码中时，对这个jar的强依赖。



## 消费者

While subscribing, the consumer connects to any of the live nodes and requests metadata about the leaders for the partitions of a topic. The consumer then issues a fetch request to the lead broker to consume the message partition by specifying the message offset (the beginning position of the message offset). Therefore, the Kafka consumer works in the pull model and always pulls all available messages after its current position in the Kafka log (the Kafka internal data representation).

[读Kafka Consumer源码](https://www.cnblogs.com/hzmark/p/kafka_consumer.html) 对consumer 源码的实现评价不高

开发人员不必关心与kafka 服务端之间的网络连接的管理、心跳检测、请求超时重试等底层操作，也不必关心订阅Topic的分区数量、分区leader 副本的网络拓扑以及consumer group的Rebalance 等kafka的具体细节。

KafkaConsumer 依赖SubscriptionState 管理订阅的Topic集合和Partition的消费状态，通过ConsumerCoordinator与服务端的GroupCoordinator交互，完成Rebalance操作并请求最近提交的offset。Fetcher负责从kafka 中拉取消息并进行解析，同时参与position 的重置操作，提供获取指定topic 的集群元数据的操作。上述所有请求都是通过ConsumerNetworkClient 缓存并发送的，在ConsumerNetworkClient  中还维护了定时任务队列，用来完成HeartbeatTask 任务和AutoCommitTask 任务。NetworkClient 在接收到上述请求的响应时会调用相应回调，最终交给其对应的XXHandler 以及RequestFuture 的监听器进行处理。

![](/public/upload/scala/consumer_object.png)

![](/public/upload/scala/consumer_poll.png)

Kafka对外暴露了一个非常简洁的poll方法，其内部实现了协作、分区重平衡、心跳、数据拉取等功能，但使用时这些细节都被隐藏了

Kafka provides two types of API for Java consumers:

1. High-level API,  does not allow consumers to control interactions with brokers.
2. Low-level API, is stateless
and provides fine grained control over the communication between Kafka broker and the consumer.

那么consumer 与broker 交互有哪些细节呢？The high-level consumer API is used when only data is needed and the handling of message offsets is not required. This API hides broker details from the consumer and allows effortless communication with the Kafka cluster by providing an abstraction over the low-level implementation. The high-level consumer stores the last offset
(the position within the message partition where the consumer left off consuming the message), read from a specific partition in Zookeeper. This offset is stored based on the consumer group name provided to Kafka at the beginning of the process.

**主动拉取 是kafka 的一个重要特征，不仅是consumer 主动拉取broker， broker partition follower 也是主动拉取leader**。

### consumer group rebalance

为什么要引入consumer group呢？主要是为了提升消费者端的吞吐量。多个consumer实例同时消费，加速整个消费端的吞吐量（TPS）。

当consumer group 新加入一个consumer 时，首要解决的就是consumer 消费哪个分区的问题。这个方案kafka 演化了多次，在最新的方案中，分区分配的工作放到了消费端处理。

所谓的consumer group，指的是多个consumer实例共同组成一个组来消费topic。topic中的每个分区都只会被组内的一个consumer实例消费，其他consumer实例不能消费它。从实现看，consumer 主动拉取的逻辑也不适合 多个consumer 同时拉取一个partition，因为宕机后无法重新消费。另外，一个consumer 一个partition，server 端也无需考虑多线程竞争问题了。

![](/public/upload/scala/kafka_group_ordinator.png)

![](/public/upload/scala/kafka_rebalance_sequence.png)

||broker|consumer|
|---|---|---|
|逻辑|topic|consumer group|
|物理|partition|consumer instance|

两个角色一个过程

1. Consumer Group Co-ordinator

    1. 某一个broker，存储元数据
2. Group Leader 
3. Rebalancing Process


`<分区，位移>` 保存在内部topic中：__consumer_offsets

