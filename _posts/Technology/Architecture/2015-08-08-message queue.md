---

layout: post
title: 消息队列
category: 技术
tags: Architecture
keywords: 消息队列 rabbitmq kafka

---

## 简介


## linux的消息队列

进程间通信的三大手段之一（共享内存、信号和消息队列），对于不同的通信手段，进程消费“通信数据”的方式不同。

消息队列和管道的区别：

- 管道的中间介质是文件。
- 消息队列是消息缓冲区，在内存中以队列的形式组织，传输时以消息（一个数据结构）为单位。

## 消息队列系统

基于内存或磁盘的队列，为分布式应用提供通信手段。

个人观察，使用消息队列系统一个原因是：现代大型系统，大多由多个子系统组成。以A,B两个子系统为例，假定，A系统的输出由B系统处理。一般情况下，B系统的处理能力可以满足A系统的要求。然而，当A系统负载突然增大时，如果AB两系统以RPC方式连接，则会引起连锁反应:B系统也扛不住了。使用消息队列后，因为队列的缓冲功能，B系统可以按照自己的节奏处理数据，提高系统的可靠性。


## 消息模型

消息模型可以分为两种， 队列和发布-订阅式。 

1. 队列,一组消费者从服务器读取消息，一条消息只有其中的一个消费者来处理。
2. 发布-订阅模型，消息被广播给所有的消费者，接收到消息的消费者都可以处理此消息。


rabbitmq

producer ==> exchange ==> queue  ==> consumer

一个消费者消费一个队列，通过改变exchange类型来达到广播、组播和单播效果。

kafka

producer ==> topic =1:n=> consumer group =1:n=> consumer

一个topic消息广播给所有consumer group，consumer轮流处理consumer group 接到的消息(kafka有分区的概念，不同的consumer会消费不同的分区)。通过调整topic、consumer group、consumer三者关系来达到广播、组播和单播效果。




## 其它

消息队列系统有很多，主要有以下不同：

1. 消息队列存在哪？内存、磁盘？
2. 是否支持分布式
3. 消息的格式是什么？
4. 系统是否能够确保：消息一定会被接收，消息未被正确处理时通知发送方。


至于kafka、flume和rabbitmq等系统，网上资料很多，此处不再赘述。


## 引用

[Kafka快速入门](http://colobu.com/2014/08/06/kafka-quickstart/)

[RabbitMQ AMQP 消息模型攻略](https://segmentfault.com/a/1190000007123977)

