---

layout: post
title: 异步和性能
category: 技术
tags: Architecture
keywords: 异步

---

## 简介

https://www.ibm.com/developerworks/cn/java/j-lo-comet/

如何定义一个系统的性能，参见[性能调优攻略](http://coolshell.cn/articles/7490.html)

[Servlet 3.0 实战：异步 Servlet 与 Comet 风格应用程序
](https://www.ibm.com/developerworks/cn/java/j-lo-comet/)

## 为什么异步web可以提高吞吐量

首先，异步不是突然出现一个异步就牛了，而是一系列手段加持的结果


1. 长连接
2. Web 线程不需要同步的、一对一的处理客户端请求，能做到一个 Web 线程处理多个客户端请求。同步的实质是线程数限制了连接数，假设tomcat有100个线程，某个请求比较耗时，那么第101个请求就无法创建连接。

举个例子，一个请求建立连接要1s，请求处理要1s，每秒到达100个请求。

||同步|异步|
|---|---|---|
|第1s|100个请求连接成功|100个请求连接成功|
|第2s|100个请求处理成功|100个请求连接成功，100个请求处理成功|
|第3s|100个请求连接成功|100个请求连接成功，100个请求处理成功|
|第4s|100个请求处理成功|100个请求连接成功，100个请求处理成功|
|4s小计|处理200个请求，50qps|100个请求连接成功，300个请求处理成功，65qps|

当然，对于一个特定请求，请求处理的耗时时间是不变的。




## 引用

[性能调优攻略](http://coolshell.cn/articles/7490.html)