---

layout: post
title: 学习网络
category: 技术
tags: Network
keywords: network

---

## 简介

* TOC
{:toc}

用思考的方式去记忆，而不是用记忆的方式去思考。

[读完这篇文章，就再也不怕遇到网络问题啦！](https://mp.weixin.qq.com/s/Tnerf7M_a6HUC4ucaOWzeg)

[快速界定故障：Socket Tracer网络监控实践](https://mp.weixin.qq.com/s/0w5t_KkHRLXkEY1_qbdTtw)

狭义上的网络，只包含交换机、路由器、防火墙、负载均衡等环节，没什么重传，也不丢包，更不影响应用消息本身。如果是广义的网络，那就包含了至少以下几个领域：
1. 对应用层协议的理解；
2. 对传输层和应用层两者协同的理解；如何把应用层问题锚定到网络层数据包。
    1. 应用现象跟网络现象之间的鸿沟：你可能看得懂应用层的日志，但是不知道网络上具体发生了什么。如果日志里说某个 HTTP 请求耗时很长，你是无法知道网络上到底什么问题导致了这么长的耗时，是丢包引起了重传？还是没有丢包，纯粹是传输速度慢呢？ 
    2. 工具提示跟协议理解之间的鸿沟：你看得懂 Wireshark、tcpdump 这类工具的输出信息的含义，但就是无法真正地把它们跟你对协议的理解对应起来。应用层只看到日志connection reset by peer，如何用wireshark 排查原因
3. 对操作系统的网络部分的理解。

## 网络协议

![](/public/upload/network/network_protocol.png)

从计算机 分层和网络分层的feel 来审视上图

1. 物理层。可以理解为网络设备的原生能力，它定义了硬件层次来看的基础网络协议。PS：类似于cpu 指令
2. 在单机体系，操作系统是一台计算机真正可编程的开始。同样地，互联网世界的体系中，IP 网络是互联网 “操作系统” 的核心，是互联网世界可编程的开始。TCP/UDP 传输层。它也是互联网 “操作系统” 的重要组成部分，和 IP 网络一起构成互联网 “操作系统” 的内核。IP 网络解决的是网如何通的问题，而传输层解决的是如何让互联网通讯可信赖的问题，从而大幅降低互联网应用程序开发的负担。

![](/public/upload/network/internet_layer.png)

在网络的多个层次中，IP层处于核心地位，其之上或之下每一层都有多个协议实现，而只有网络层，只有一个IP协议（细腰结构）。

## 网络传输

![](/public/upload/network/network_transmission.png)

[网络设计核心思想](https://mp.weixin.qq.com/s/0XXNDid8lXDbcSZ9fmWOZg)

## 其它

![](/public/upload/network/troubleshoot_network_lesson.jpg)