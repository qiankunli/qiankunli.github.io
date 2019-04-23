---

layout: post
title: Codis/Jodis源码分析
category: 技术
tags: Go
keywords: Go

---

## 前言

* TOC
{:toc}

github 地址[CodisLabs/codis](https://github.com/CodisLabs/codis) 基于go 语言开发，是一个很好的了解go 及 分布式开发的项目。

[Codis 使用文档](https://github.com/CodisLabs/codis/blob/release3.2/doc/tutorial_zh.md)
Codis 是一个分布式 Redis 解决方案, 对于上层的应用来说, 连接到 Codis Proxy 和连接原生的 Redis Server 没有显著区别 ([不支持的命令列表](https://github.com/CodisLabs/codis/blob/release3.2/doc/unsupported_cmds.md)), 上层应用可以像使用单机的 Redis 一样使用, Codis 底层会处理请求的转发, 不停机的数据迁移等工作, 所有后边的一切事情, 对于前面的客户端来说是透明的, 可以简单的认为后边连接的是一个内存无限大的 Redis 服务。

## 集群解决方案——Smart Client VS Proxy

[为什么大厂都喜欢用 Codis 来管理分布式集群？](https://juejin.im/post/5c132b076fb9a04a08218eef)

[Codis作者黄东旭细说分布式Redis架构设计和踩过的那些坑们](https://my.oschina.net/u/658658/blog/500499)（未读完）

Redis多实例通常有3个使用途径

1. 客户端静态分片，一致性哈希；也称为Smart Client
2. 通过Proxy分片，即Twemproxy；
3. 官方的Redis Cluster

服务端的改造来自官方，暂时不考虑， 所以一般争论也在Smart Client 和 Proxy 之间。

需求目标

1. 支持分片
2. Zookeeper ==> Proxy 无状态
3. 平滑扩容/缩容
4. 扩容对用户透明
5. 图形化监控一切

Proxy拥有更好的监控和控制，同时其后端信息亦不易暴露，易于升级；而Smart Client拥有更好的性能（因为没有中间层），及更低的延时，但是升级起来却比较麻烦。从各种大厂的方案看，都比较推崇Proxy

## 整体结构

![](/public/upload/go/codis_architecture.png)

1. codis-proxy 。客户端连接的Redis代理服务，本身实现了Redis协议，表现很像原生的Redis （就像 Twemproxy）。一个业务可以部署多个 codis-proxy，其本身是无状态的。
2. codis-server。Codis 项目维护的一个Redis分支，加入了slot的支持和原子的数据迁移指令。

### 源码分析

主要是两个package

cmd 命令入口，即通过命令行 启动 socket server 程序等
pkg 各个组件的源码文件

## Jodis - Java client for codis

[Jodis - Java client for codis](https://github.com/CodisLabs/jodis) 功能特性

1. Use a round robin policy to balance load to multiple codis proxies.
2. Detect proxy online and offline automatically.

从目前看，主要是解决多proxy 的服务发现




