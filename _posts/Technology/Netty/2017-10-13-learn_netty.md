---

layout: post
title: 为什么netty比较难懂？
category: 技术
tags: Netty
keywords: JAVA netty

---

## 前言

到目前为止，笔者关于netty写了十几篇博客，内容非常零碎，笔者一直想着有一个总纲的东西来作为主干，将这些零碎place it in context。所以梳理了一张图，从上往下“俯视”看，netty有哪些东西?

![](/public/upload/netty/learn_netty.png)

当这个图画了一个雏形时，笔者突然有了一个意外收获，心中很久以来的疑惑也得到了解答，那就是为什么很多人会觉得学习netty代码比较难（这也是笔者最初的感受）？**因为对于大部分人来说，是先接触了netty，才第一次接触nio、同步操作异步化 等技术/套路，也就是上图“核心” 部分的东西，除了要理解netty代码本身的抽象之外，还需理解很多新概念。**并且，极易形成的错误观念是：因为netty是这样用的，便容易认为这些“新概念特性”只能这样用。比如，一想起nio便认为只能像netty那样用，但hadoop中传输文件块的代码 便与netty对nio的应用方式有所不同。 

有了上图为总纲，先了解上图左侧的核心部分，再有针对性的阅读笔者先前的博客或netty的源码，心中有“成竹”而不是一堆琐碎，剩下的便是右侧的netty实现细节了，感受上应该会容易许多。

reactor pattern 理念 参见 [Understanding Reactor Pattern: Thread-Based and Event-Driven](https://dzone.com/articles/understanding-reactor-pattern-thread-based-and-eve)，并且建议你读三遍。任何框架，一定都是先有了理念和思想，然后体现在代码上。看代码之前，找到那个理念和思想。


1. 阻塞io 无论怎么玩，Unfortunately, there is always a one-to-one relationship between connections and threads
1. Event-driven approach can separate threads from connections, which only uses threads for events on specific callbacks/handlers.
2. An event-driven architecture consists of event creators and event consumers. The creator, which is the source of the event, only knows that the event has occurred. Consumers are entities that need to know the event has occurred. They may be involved in processing the event or they may simply be affected by the event.
3. The reactor pattern is one implementation technique of the event-driven architecture. **In simple words, it uses a single threaded event loop blocking on resources emitting events and dispatches them to corresponding handlers/callbacks.**
4. **There is no need to block on I/O, as long as handlers/callbacks for events are registered to take care of them.** Events are like incoming a new connection, ready for read, ready for write, etc.
5. This pattern decouples modular application-level code from reusable reactor implementation.
6. The purpose of the Reactor design pattern is to avoid the common problem of creating a thread for each message/request/connection.Avoid this problem is to avoid the famous and known problem C10K.

《反应式设计模式》 基于事件的系统通常建立在一个事件循环上。任何时刻只要发生了事情， 对应的事件就会被追加到一个队列中。事件循环持续的从队列中拉取事件，并执行绑定在事件上的回调函数。每一个回调函数通常都是一段微小的、匿名的、响应特定事件（例如鼠标点击）的过程。回调函数也可能产生新事件，这些事件随后也会被追加到队列里面等待处理。

个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)