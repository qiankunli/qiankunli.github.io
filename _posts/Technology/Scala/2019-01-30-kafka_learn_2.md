---

layout: post
title: 《Apache Kafka源码分析》——server
category: 技术
tags: Scala
keywords: Scala  akka

---

## 前言

* TOC
{:toc}

建议先阅读下[《Apache Kafka源码分析》——Producer与Consumer](http://qiankunli.github.io/2017/12/08/kafka_learn_1.html)

## 网络层

kafka client (producer/consumer) 与kafka server通信时使用自定义的协议，一个线程 一个selector 裸调java NIO 进行网络通信。 

kafka 服务端 也这么玩就不够用了，所以

![](/public/upload/scala/kafka_server_framework.jpg)

