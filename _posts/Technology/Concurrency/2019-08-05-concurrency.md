---

layout: post
title: 那些年追过的并发
category: 架构
tags: Concurrency
keywords: concurrency

---

## 简介

* TOC
{:toc}


![](/public/upload/architecture/concurrency.png)

## 并发模型

[多线程设计模式/《Concurrency Models》笔记](http://qiankunli.github.io/2015/06/19/Threads_Pattern.html)

[从Go并发编程模型想到的](http://qiankunli.github.io/2017/02/04/go_concurrence.html)

[java系并发模型的发展](http://qiankunli.github.io/2017/09/05/akka.html)

## 并发读写

《软件架构设计》要让各式各样的业务功能与逻辑最终在计算机系统里实现，只能通过两种操作：读和写。

![](/public/upload/architecture/high_concurrency.png)

并发竞争的几种处理

1. 靠锁把并发搞成顺序的
1. 发现有人在操作数据，就先去干点别的，比如自旋、sleep 一会儿
2. 发现有人在操作数据，找个老版本数据先用着，比如mvcc
2. 相办法不共享数据

## 异步

[异步执行抽象——Executor与Future](http://qiankunli.github.io/2016/07/08/executor_future.html)

[netty中的线程池](http://qiankunli.github.io/2019/06/28/netty_executor.html)

## 服务器端编程

[服务器端编程](http://qiankunli.github.io/2019/04/27/server_side_development.html)