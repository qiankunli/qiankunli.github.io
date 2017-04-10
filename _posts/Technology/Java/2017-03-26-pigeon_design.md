---

layout: post
title: 通用transport层框架pigeon
category: 技术
tags: Java
keywords: JAVA netty transport

---

## 前言

笔者最近在学习netty在框架中的使用，并顺手实现了一个基于netty的通用transport层框架pigeon，在实现的过程中碰到了一些问题，在此与大家分享。

pigeon的作用，是在一个更复杂的、依赖通信框架中（比如dubbo、zk等）充当底层的数据传输角色，相关介绍参见上一篇博文[netty（六）netty在框架中的使用套路](http://qiankunli.github.io/2017/03/18/netty_in_framework.html)

## 心跳机制的实现

### 连接的可靠性保障

1. 第一连接时，连接失败后，定时重试
2. 心跳机制，发现连接断开后，立即重试

心跳机制有以下几种方案

1. 使用TCP协议层面的keepalive机制。该机制有一些问题，参见[浅析 Netty 实现心跳机制与断线重连](https://segmentfault.com/a/1190000006931568)
2. 在应用层上实现自定义的心跳机制，本文只谈netty有关的。

	1. 心跳包独立定时发送
	2. 当系统一定时间内没有收发数据时，才发送，需要和IdleStateHandler结合使用。

### 心跳对协议设计的影响

心跳包model（因为只需提供一个字段表示该消息为心跳消息即可，所以甚至都不需要model）是独立的，还是和业务model兼容在一起？

||优点|缺点|
|---|---|---|
|独立|业务无感知|表示packet类型的字段必须非常靠前|
|兼容在一起|model解析和处理部分逻辑通用|业务model可能比较复杂，如果要求一个packet必须传很多字段时，心跳packet size会较大|

在笔者的pigeon实践中，业务model定义如下

	class Packet{
		PacketType;	// 序列化和反序列化时，必须是第一个字段
		Id;
		header;
		body;
	}
	
心跳包的发送和接收，则是直接收发Bytebuf

	 ByteBuf buf = ctx.alloc().buffer(4);
    buf.writeInt(4);
    buf.writeInt(PacketType.ping.getValue());
    ctx.writeAndFlush(buf);

## 连接数管理

### 基本实现

### 连接数管理对心跳机制的影响
	


## 配置管理

要有一个配置类贯穿组件的各个方面

1. 读取配置文件配置
2. 配置过多，而一个功能类可能依赖多个属性，导致功能类的构造方法参数比较多。当然，构造方法参数多（并且还容易变化）的问题，可以通过builder模式解决。