---

layout: post
title: Docker网络二,libnetwork
category: 技术
tags: Docker
keywords: Docker,libnetwork

---

## 前言（未完待续）

我们搭建一个网络环境，一般遵循一定的网络拓扑结构。由于Linux可以模拟相应的网络设备，并可以创建“虚拟机”（也就是容器），因此在Linux系统内，我们也可以遵循一定的网路拓扑结构，设计一个“内网”，实现容器之间的通信。

Libnetwork是Docker团队将Docker的网络功能从Docker核心代码中分离出去，形成一个单独的库。 Libnetwork通过**插件的形式**为Docker提供网络功能。 使得用户可以根据自己的需求实现自己的Driver来提供不同的网络功能。 

官方目前计划实现以下Driver：

1. Bridge ： 这个Driver就是Docker现有网络Bridge模式的实现。 （基本完成，主要从之前的Docker网络代码中迁移过来）
2. Null ： Driver的空实现，类似于Docker 容器的None模式。
3. Overlay ： 隧道模式实现多主机通信的方案。 

“Libnetwork所要实现的网络模型（网络拓扑结构）基本是这样的： 用户可以创建一个或多个网络（一个网络就是一个网桥或者一个VLAN ），一个容器可以加入一个或多个网络。 同一个网络中容器可以通信，不同网络中的容器隔离。”**我觉得这才是将网络从docker分离出去的真正含义，即在创建容器之前，我们可以先创建网络（即创建容器与创建网络是分开的），然后决定让容器加入哪个网络。**

## Libnetwork定义的容器网络模型

![Alt text](/public/upload/docker/libnetwork.jpeg)

- Sandbox：对应一个容器中的**网络环境**（没有实体），包括相应的网卡配置、路由表、DNS配置等。CNM很形象的将它表示为网络的『沙盒』，因为这样的网络环境是随着容器的创建而创建，又随着容器销毁而不复存在的； 
- Endpoint：实际上就是一个容器中的虚拟网卡，在容器中会显示为eth0、eth1依次类推； 
- Network：指的是一个能够相互通信的容器网络，加入了同一个网络的容器直接可以直接通过对方的名字相互连接。它的实体本质上是主机上的虚拟网卡或网桥。

    
## 参考文献

[聊聊Docker 1.9的新网络特性][]

[聊聊Docker 1.9的新网络特性]: http://mt.sohu.com/20160118/n434895088.shtml