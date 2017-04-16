---

layout: post
title: network channel
category: 技术
tags: Java
keywords: JAVA netty channel

---

## 前言（未完成）

借用知乎上"者年"关于java nio技术栈的回答[如何学习Java的NIO](https://www.zhihu.com/question/29005375/answer/43021911)


0. 计算机体系结构和组成原理 中关于中断，关于内存，关于 DMA，关于存储 等关键知识点
1. 操作系统 中 内核态用户态相关部分，  I/O 软件原理
2. 《UNIX网络编程（卷1）：套接字联网API（第3版）》([美]史蒂文斯，等)一书中 IO 相关部分。
3. [Java I/O底层是如何工作的？](http://www.importnew.com/14111.html)
4. [存储之道 - 51CTO技术博客 中的《一个IO的传奇一生》](http://alanwu.blog.51cto.com/3652632/d-8)
5. [Operating Systems: I/O Systems4. ](https://www.cs.uic.edu/~jbell/CourseNotes/OperatingSystems/13_IOSystems.html)
6. [多种I/O模型及其对socket效率的改进](http://mickhan.blog.51cto.com/2517040/1586370)


## java nio channel

java源码中channel的注释：A nexus for I/O operations。很多时候，作者在源码上的注释，本身就很精练、直接，比博客上赘述要好很多。

Channel用于在字节缓冲区和位于通道另一侧的实体（通常是一个文件或套接字）之间有效地传输数据。

java buffer <==> java channel <==> system socket send/receive buffer

[Java NIO Tutorial](http://tutorials.jenkov.com/java-nio/index.html)
In the standard IO API you work with byte streams and character streams. In NIO you work with channels and buffers. Data is always read from a channel into a buffer, or written from a buffer to a channel.

看完这一段，笔者突然有一种nio类似于erlang/golang基于消息的线程协作的感觉。

||表现|
|---|---|
|nio|channel ==> buffer,buffer ==> chanel|
|erlang/golang基于消息的线程协作|channel ==> 变量,channel <== 变量|

golang基于消息的线程协作本来没什么，但是将其与线程调度等机制结合起来，就可以很明显的提高效率。



## netty channel

A nexus（连结、连系） to a network socket or a component which is capable of I/O
operations such as read, write, connect, and bind.

1. All I/O operations are asynchronous.
2. Channels are hierarchical<

一个client可以有多个channel，多个channel可以共享一个socket

所有的io操作都是异步，那么有channelFuture，和channel pipeline就是理所当然的。因为即便是异步，也得告诉你什么时候写完了，什么时候读完了。只是，对于异步来说，可能A线程触发读操作，而实际处理读到数据的，可能是B线程。

就像，并发是现代操作系统各种复杂性的根源一样，异步也可以理解为netty各种复杂性的根源。