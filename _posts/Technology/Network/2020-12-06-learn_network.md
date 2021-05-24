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



