---

layout: post
title: 远程调用
category: 技术
tags: Architecture
keywords: JAVA rpc rmi

---

## 前言（未完待续） ##

java跨节点通信方式：

1. 通过Java的Socket+Java序列化的方式进行跨界点调用（这种方式，其实只是约定了数据传输，并没有约定调用细节）
2. 通过RMI进行远程服务调用
3. 利用一些开源的RPC框架进行远程服务调用，比如Thrift
4. 利用标准的公有协议进行跨节点服务调用，比如Http，RESTful + JSON等 


## 共同点

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
3. 数据与流如何转换(即数据的编解码)

无论哪种远程调用技术，基本的道理都是一样的：服务端监听请求，处理请求，返回结果；客户端调用代理类，访问调用服务。

![Alt text](/public/upload/architecture/remotecall.png)

在远程通讯领域中，涉及的知识点还是相当的多的，例如：

- 通信协议(socket/tcp/http/udp /rmi/xml-rpc etc.)
- 网络IO（biO/nio/aio）以及对应线程模型
- 本地调用与远程调用的透明化方案（涉及java classloader、Dynamic Proxy etc.）
- 异步与同步调用
- 网络通信处理机制（自动重连、广播、异常、池处理等等）
- Java Object Serialization (各种协议的私有序列化机制等)



## 引用

[远程调用原理与对比RMI、MINA、ESB、Burlap、Hessian、SOAP、EJB][]

[远程调用原理与对比RMI、MINA、ESB、Burlap、Hessian、SOAP、EJB]: http://blog.sina.com.cn/s/blog_5f53615f01014xfj.html
