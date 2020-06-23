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

## io

一般来说服务端程序有几个角色：Acceptor、Selector 和 Processor。

1. Acceptor 负责接收新连接，也就是 accept；
2. Selector 负责检测连接上的 I/O 事件，也就是 select；
3. Processor 负责数据读写、编解码和业务处理，也就是 read、decode、process、encode、send。

Acceptor 在接收连接时，可能会阻塞，为了不耽误其他工作，一般跑在单独的线程里；而 Selector 在侦测 I/O 事件时也可能阻塞，但是它一次可以检测多个 Channel（连接），其实就是用阻塞它一个来换取大量业务线程的不阻塞，那 Selector 检测 I/O 事件到了，是用同一个线程来执行 Processor，还是另一个线程来执行呢？不同的场景又有相应的策略。

比如 Netty 通过 EventLoop 将 Selector 和 Processor 跑在同一个线程。一个 EventLoop 绑定了一个线程，并且持有一个 Selector。而 Processor 的处理过程被封装成一个个任务，一个 EventLoop 负责处理多个 Channel 上的所有任务，而一个 Channel 只能由一个 EventLoop 来处理，这就保证了任务执行的线程安全，并且用同一个线程来侦测 I/O 事件和读写数据，可以充分利用 CPU 缓存。请你注意，**这要求 Processor 中的任务能在短时间完成**，否则会阻塞这个 EventLoop 上其他 Channel 的处理。因此在 Netty 中，可以设置业务处理和 I/O 处理的时间比率，超过这个比率则将任务扔到专门的业务线程池来执行，这一点跟 Jetty 的 EatWhatYouKill 线程策略有异曲同工之妙。

而 Kafka 把 Selector 和 Processor 跑在不同的线程里，因为 Kafka 的业务逻辑大多涉及与磁盘读写，处理时间不确定，所以 Kafka 有专门的业务处理线程池来运行 Processor。与此类似，Tomcat 也采用了这样的策略