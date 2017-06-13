---

layout: post
title: apple 推送的问题
category: 技术
tags: Other
keywords: apns

---

## 前言（待整理）

## 影响推送的因素

用户方面

1. 大量用户关闭推送
2. 非锁屏该状态下，用户推送的提醒样式设置为none
3. 用户正在使用app时无法收到推送

苹果方面

1. 一系列的原因导致apns关闭连接，尤其是invalid token，而invalid token无法避免

	If APNs decides to terminate an established HTTP/2 connection, it sends a GOAWAY frame. The GOAWAY frame includes JSON data in its payload with a reason key, whose value indicates the reason for the connection termination. **For a list of possible values for the reason key, see [Table 8-6](https://developer.apple.com/library/content/documentation/NetworkingInternet/Conceptual/RemoteNotificationsPG/CommunicatingwithAPNs.html#//apple_ref/doc/uid/TP40008194-CH11-SW17).**

2. 如果设备离线，假设某个时间间隔收到两个推送，则apns只会发送most recent notification。参见[Local and Remote Notification Programming Guide
](https://developer.apple.com/library/content/documentation/NetworkingInternet/Conceptual/RemoteNotificationsPG/APNSOverview.html#//apple_ref/doc/uid/TP40008194-CH8-SW1)的“Quality of Service, Store-and-Forward, and Coalesced Notifications”章节。

系统方面

1. http2协议，中国与苹果推送服务器(api.push.apple.com)连接不稳定，建连时间长，经常handshake fail


其它情况参见[xuduo/socket.io-push](https://github.com/xuduo/socket.io-push/blob/master/readmes/notification-keng.md)

## 大批量推送时带来的一些问题

还未找到切实的原因，因此只能描述一些现象

1. 单推（unicast）和少批量multicast时，没有问题。全局推送时，会出现断连
2. 虽然苹果返回了accepted，但很多用户还是没有收到

网络环境差时，发送大批量推送，推送数据缓冲在socket send buffer，进而堆积在apns的receive buffer，apns receive buffer堆积到一定程度，那么整个处理过程就停止了。linux tcp的机制是，一定时间收不到ack就开会resend，resend次数超过阈值就会断连接。参见[Closing connection due to write timeout](https://github.com/relayrides/pushy/issues/433)和[JAVA Socket超时浅析](http://blog.csdn.net/sureyonder/article/details/5633647)中的"socket写超时"部分。[几种TCP连接中出现RST的情况](https://my.oschina.net/costaxu/blog/127394)

## 和前端协作

1. 无论是否可以拿到token，都要发请求，告诉我客户端是否关了推送

## 涉及的技术栈

1. netty及其涉及的java nio
2. http2
3. ssl
4. netty与ssl、http2的结合问题

## 办法

数据方面

1. 尽可能多的采集用户token
2. 维护token的有效性，减少invalid token的存在，

	* 定时发送静默notification测试和优化
	* 发送推送时，invalid token 随时发现、随时记录

数据处理方面

1. 根据发送失败的原因，进行重试
2. 发送框架本身要稳定，不能发生oom、direct oom之类的异常。[Direct memory exhausted after sending many notifications](https://github.com/relayrides/pushy/issues/142)
3. 维护多个与apns的connection，在一个connection断开后不影响推送。同时，推送任务要尽可能均匀的分布在各个连接上。




## pushy使用要注意的问题

1. 用它自带的eventloop，然后，direct oom就没有了
2. netty-tcnative-boringssl-static 的版本要紧跟 pushy版本，否则也会造成oom



## 其它资料

[浅谈基于HTTP2推送消息到APNs](http://www.linkedkeeper.com/detail/blog.action?bid=167)
