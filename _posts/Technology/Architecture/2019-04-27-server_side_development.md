---

layout: post
title: 服务器端编程
category: 架构
tags: Architecture
keywords: server programming

---

## 简介

* TOC
{:toc}

## 通信协议

[网络通信协议](http://qiankunli.github.io/2019/04/20/network_communication_protocol.html)

## 整体结构

![](/public/upload/architecture/network_communication.png)

## 1+N+M 模型

来自《软件架构设计》 将一个请求的处理分为3道工序：监听、IO、业务逻辑处理。worker 线程还可以继续拆分成编解码、业务逻辑计算等环节，进一步提高并发度。请求与请求之间是并行的，一次请求处理的多个环节之间也是并行的。

![](/public/upload/architecture/server_side_1nm.png)


1. 监听线程，负责accept 事件的注册和处理
2. io线程，负责每个socket rw事件的注册和实际的socket 读写
3. worker线程，纯粹的业务线程，没有socket的读写操作

不同的系统实现方式会有一些差异，比如Tomcat6 的NIO网络模型

![](/public/upload/architecture/tomcat6_1nm.png)

## 服务端案例

[Redis源码分析](http://qiankunli.github.io/2019/04/20/redis_source.html)

[《Apache Kafka源码分析》——server](http://qiankunli.github.io/2019/01/30/kafka_learn_2.html)

[netty（六）netty回顾](http://qiankunli.github.io/2016/07/25/Java-Netty6.html)

## 客户端访问案例

[Jedis源码分析](http://qiankunli.github.io/2016/06/07/jedis_source.html)