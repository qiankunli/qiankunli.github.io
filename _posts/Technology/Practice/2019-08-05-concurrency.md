---

layout: post
title: 那些年追过的并发
category: 架构
tags: Practice
keywords: window

---

## 简介

* TOC
{:toc}


![](/public/upload/architecture/concurrency.png)

## 硬件对软件的影响

[硬件对软件设计的影响](http://qiankunli.github.io/2018/01/07/hardware_software.html)

[AQS1——并发相关的硬件与内核支持](http://qiankunli.github.io/2016/03/13/aqs.html)

## 共享内存 vs 消息传递

[如何理解 Golang 中“不要通过共享内存来通信，而应该通过通信来共享内存”？ - 赵丙峰的回答 - 知乎](https://www.zhihu.com/question/58004055/answer/155244333)

无论是共享内存还是消息，本质都是不同实体之间的如何协调信息，以达成某种一致。**直接共享内存基于的通讯协议由硬件和OS保证**，这种保证是宽泛的，事实上可以完成任何事情，同样**也带来管理的复杂和安全上的妥协**。而消息是高级的接口，可以通过不同的消息定义和实现把大量的控制，安全，分流等相关的复杂细节封装在消息层，免除上层代码的负担。所以，**这里其实是增加了一层来解决共享内存存在的问题**，实际上印证了另一句行业黑话：计算机科学领域所有的问题都可以通过增加一个额外的间接层来解决。

然而其实还有另一句话：计算机可以领域大多数的性能问题都可以通过删除不必要的间接层来解决。不要误解这句话，这句话不过是说，针对领域问题的性能优化可以使用不同于通用问题的办法，因为通用的办法照顾的是大多数情况下的可用性而不是极端情况下的性能表现。诟病消息系统比共享内存性能差其实是一个伪问题。当二者都不存在的时候，自然共享内存实现直接而简单，成熟的消息系统则需要打磨并且设计不同情况下的策略。人们自然选择快而脏的共享内存。

然而，技术进步的意义就在于提供高层次的选择的灵活性。当二者都存在的时候，选择消息系统一般是好的，而且绝大多数的性能问题可以通过恰当的策略配置得以解决。针对遗留系统，则可以选择使用消息系统模拟共享内存。这种灵活性，是共享内存本身不具备的。

对这种编程哲学，golang提供语言层面的支持无疑是好的，可以推动良好设计的宣传和广泛使用。

如果程序设计成通过通信来共享数据的话，那么通信的两端是不是在同一个物理设备上就无所谓了，只有这样才能实现真正的分布式计算。

### 共享内存

[JVM3——java内存模型](http://qiankunli.github.io/2017/05/02/java_memory_model.html)

### 共享内存数据结构

[无锁数据结构和算法](http://qiankunli.github.io/2018/10/15/lock_free.html)

## 并发模型

[多线程设计模式/《Concurrency Models》笔记](http://qiankunli.github.io/2015/06/19/Threads_Pattern.html)

[从Go并发编程模型想到的](http://qiankunli.github.io/2017/02/04/go_concurrence.html)

[java系并发模型的发展](http://qiankunli.github.io/2017/09/05/akka.html)

## 异步

[异步执行抽象——Executor与Future](http://qiankunli.github.io/2016/07/08/executor_future.html)

[netty中的线程池](http://qiankunli.github.io/2019/06/28/netty_executor.html)

## 服务器端编程

[服务器端编程](http://qiankunli.github.io/2019/04/27/server_side_development.html)