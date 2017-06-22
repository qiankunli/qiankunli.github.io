---

layout: post
title: 基于netty的协议解析
category: 技术
tags: Netty
keywords: JAVA spi xsd

---

## 前言（待整理）

我们决定使用netty作为通讯框架时，必须要考虑如何将netty与协议结合的问题


## 协议的复杂性对使用netty的影响

### 简单协议

协议字段完全固定，并且协议的目的就是简单的传输数据。

||model|编解码||
|---|---|---|---|
|业务层|业务model|继承message，实现encode、decode| 调用transport层提供的send接口 |
|transport层| message |抽象方法 encode、decode| transport 只是实现message的收发|


### 复杂协议

1. 大部分协议分为header和data部分，data部分经常随着header部分变化，比如http协议`transfer-encoding:chunked`或者multipart
2. 需要复杂的初始化，比如ssl，在工作之前，要进行ssl握手。
3. 一次语义动作的完成，需要多次通信。通常表现在，既有控制指令，又有数据指令。控制指令通常作为数据指令的上下文，需要协议框架进行存储。


## http协议


[浅谈基于HTTP2推送消息到APNs](http://www.linkedkeeper.com/detail/blog.action?bid=167)


## netty与ssl 的整合

很多公司要对外提供开放接口，比如支付宝的支付接口、apns。开放接口通常对性能和安全性要求极高，此时netty 和 ssl就是一个比较好的选择。so，这也意味着，客户端通常也要支持ssl。

[Netty系列之Netty安全性](http://www.infoq.com/cn/articles/netty-security)

## 基本套路

第三方netty协议框架，将代码独立抽取成一个jar，包括

1. 协议model
2. model encoder/decoder

### encoder

encoder除了encode，进一步提供write逻辑

### decoder

![alt text](/public/upload/netty/netty_tcp_stream.png)

参见[netty对http协议解析原理(一)](http://blog.csdn.net/hetaohappy/article/details/52008120)

集成Byte2MessageDecoder，流式读取数据。**netty数据可读时，就会触发该方法的执行**，`void decode(ChannelHandlerContext ctx, ByteBuf buffer, List<Object> out)`，这也是tcp流式处理数据决定的。





