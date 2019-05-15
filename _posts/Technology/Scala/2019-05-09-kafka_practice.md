---

layout: post
title: kafka实践
category: 技术
tags: Scala
keywords: Scala  akka

---

## 前言（未完成）

* TOC
{:toc}

建议先阅读下[《Apache Kafka源码分析》——Producer与Consumer](http://qiankunli.github.io/2017/12/08/kafka_learn_1.html)


## 一条消息大小不能超过1M


## 消费端优化

### 多线程 消费

从[spring kafka 源码分析](http://qiankunli.github.io/2019/05/06/kafka_spring_source.html) 可以看到， spring-kafka 仅使用了一个线程来 操作consumer 从broker 拉取消息，一个线程够用么？ 是否可以通过加线程 提高consumer的消费能力呢？


[【原创】探讨kafka的分区数与多线程消费](https://raising.iteye.com/blog/2252456) 一个消费线程可以对应若干个分区，但**一个分区只能被一个consumer 消费 + consumer 对象是线程不安全的==> 一个分区只能被具体某一个消费线程消费**。因此，topic 的分区数必须大于一个（由server.properties 的 num.partitions 控制），否则消费端再怎么折腾，也用不了多线程。

[【原创】Kafka Consumer多线程实例](https://www.cnblogs.com/huxi2b/p/6124937.html)KafkaConsumer和KafkaProducer不同，后者是线程安全的，因此我们鼓励用户在多个线程中共享一个KafkaProducer实例，这样通常都要比每个线程维护一个KafkaProducer实例效率要高。但对于KafkaConsumer而言，它不是线程安全的，所以实现多线程时通常由两种实现方法：

1. 每个线程维护一个KafkaConsumer，多个consumer 可以subscribe 同一个topic `consumer.subscribe(Arrays.asList(topic));`，如果consumer的数量大于Topic中partition的数量就会有的consumer接不到数据。

    ![](/public/upload/scala/kafka_multi_consumer_one_topic.png)

2. 维护一个或多个KafkaConsumer，同时维护多个事件处理线程(worker thread)

    ![](/public/upload/scala/kafka_one_consumer_multi_worker.png)



### 什么时候commit 消息

## 生产端优化

批量发送