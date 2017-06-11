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

简单协议，协议字段完全固定，并且协议的目的就是简单的传输数据。

||model|编解码||
|---|---|---|---|
|业务层|业务model|继承message，实现encode、decode| 调用transport层提供的send接口 |
|transport层| message |抽象方法 encode、decode| transport 只是实现message的收发|


复杂协议

1. 大部分协议分为header和data部分，data部分经常随着header部分变化，比如http协议`transfer-encoding:chunked`或者multipart
2. 需要复杂的初始化，比如ssl，在工作之前，要进行ssl握手。


## http协议


从代码上看


1. 协议的encoder/decoder上直接继承MessageToMessageEncoder/ByteToMessageDecoder
2. 定义一系列协议对应的model


## netty与ssl 的整合

很多公司要对外提供开放接口，比如支付宝的支付接口。开放接口通常对性能和安全性要求极高，此时netty 和 ssl就是一个比较好的选择。


[Netty系列之Netty安全性](http://www.infoq.com/cn/articles/netty-security)
