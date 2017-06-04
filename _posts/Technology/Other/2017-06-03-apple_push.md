---

layout: post
title: apple 推送的问题
category: 技术
tags: Other
keywords: apns

---

## 前言（待整理）

## 出现的问题

用户方面

1. 大量用户关闭推送

苹果方面

1. token unregister  导致的连接关闭

系统方便

1. http2协议，中国与苹果推送服务器(api.push.apple.com)连接不稳定，建连时间长，经常handshake fail



其它情况参见[xuduo/socket.io-push](https://github.com/xuduo/socket.io-push/blob/master/readmes/notification-keng.md)


## 需要和前端确认的事情

1. 无论是否可以拿到token，都要发请求，告诉我客户端是否关了推送


## 办法

1. 尽可能多的采集用户token
2. 维护token的有效性，减少invalid token的存在
3. 根据发送失败的原因，进行重试
4. 发送框架本身要稳定，不能发生oom、direct oom之类的异常


[Direct memory exhausted after sending many notifications](https://github.com/relayrides/pushy/issues/142)

## pushy 的问题

1. 用它自带的eventloop，然后，direct oom就没有了
2. netty-tcnative-boringssl-static 的版本要紧跟 pushy版本，否则也会造成oom
3. 


