---

layout: post
title: 远程调用
category: 技术
tags: Architecture
keywords: JAVA rpc rmi

---

## 前言 ##

首先，远程服务间的通讯基于网络通信的基本原理，将流从一台计算机传输到另一台计算机，基于传输协议（tcp和udp等）和通信模型（bio、nio和aio等）。

那么，跨主机进程如何通讯？

1. A主机应用将请求转换为流数据
2. 通过计算机网络传输到B主机上
3. B接收数据，并将流还原为请求
4. B处理请求，并将结果转换为流，将流发送至A主机
5. A主机接收流，并将其转换为结果


整个过程涉及到如下问题：

1. 通讯协议（tcp、http等）
2. 通讯模型（NIO、AIO等）
3. 数据与流如何转换

无论哪种远程调用技术，基本的道理都是一样的：服务端监听请求，处理请求，返回结果；客户端调用代理类，访问调用服务。

![Alt text](/public/upload/architecture/remotecall.png)

在远程通讯领域中，涉及的知识点还是相当的多的，例如：

- 通信协议(Socket/tcp/http/udp /rmi/xml-rpc etc.)
- 消息机制
- 网络IO（BIO/NIO/AIO）、MultiThread
- 本地调用与远程调用的透明化方案（涉及java classloader、Dynamic Proxy、Unit Test etc.）
- 异步与同步调用
- 网络通信处理机制（自动重连、广播、异常、池处理等等）
- Java Serialization (各种协议的私有序列化机制等)

## 分布式key-value存储系统

跨主机间进程通信，彼此除了直接或间接用tcp联系外，还可以使用分布式key-value存储系统，比如etcd和zookeeper等工具。进程只需与本机的etcd通信，另一主机的进程通过本机的etcd即可感知到数据变化。

当然，这种方式在通信效率等方面，不如专门的解决方案。


## 引用

[远程调用原理与对比RMI、MINA、ESB、Burlap、Hessian、SOAP、EJB][]






[远程调用原理与对比RMI、MINA、ESB、Burlap、Hessian、SOAP、EJB]: http://blog.sina.com.cn/s/blog_5f53615f01014xfj.html
