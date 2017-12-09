---

layout: post
title: 《Apache Kafka源码分析》小结
category: 技术
tags: Scala
keywords: Scala  akka

---

## 前言

### 宏观概念

producer ==> topic ==> consumer

## 具体细节

### 加入interceptor

在发送端，record发送和执行record发送结果的callback之前，由interceptor拦截

1. 发送比较简单，record发送前由interceptors操作一把。`ProducerRecord<K, V> interceptedRecord = this.interceptors == null ? record : this.interceptors.onSend(record)`
2. `Callback interceptCallback = this.interceptors == null ? callback : new InterceptorCallback<>(callback, this.interceptors, tp);`

对于底层发送来说，`doSend(ProducerRecord<K, V> record, Callback callback)`interceptors的加入并不影响（实际代码有出入，但大意是这样）。

### 反射的另个一好处

假设你的项目，用到了一个依赖jar中的类，但因为策略问题，这个类对有些用户不需要，自然也不需要这个依赖jar。此时，在代码中，你可以通过反射获取依赖jar中的类，避免了直接写在代码中时，对这个jar的强依赖。

### 为什么要zookeeper，因为关联业务要交换元数据

今天笔者在学习kafka源码，kafka的主体是`producer ==> topic ==> consumer`，topic只是一个逻辑概念，topic包含多个分区，每个分区数据包含多个副本（leader副本，slave副本）。producer在发送数据之前，首先要确定目的分区（可能变化），其次确定目的分区的leader副本所在host，知道了目的地才能发送record，这些信息是集群的meta信息。producer每次向topic发送record，都要`waitOnMetadata(record.topic(), this.maxBlockTimeMs)`以拿到最新的metadata。

producer面对的是一个broker集群，这个meta信息找哪个broker要都不方便，也不可靠，本质上，还是从zookeeper获取比较方便。zookeeper成为producer与broker集群解耦的工具。

关联业务之间需要交换元数据，当然，数据库也可以承担这个角色，但数据库没有副本等机制保证可靠性

