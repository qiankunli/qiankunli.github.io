---

layout: post
title: netty回顾
category: 技术
tags: Netty
keywords: JAVA netty review

---

## 前言 

* TOC
{:toc}

[从操作系统层面分析Java IO演进之路](https://mp.weixin.qq.com/s/KgJFyEmZApF7l5UUJeWf8Q) netty 和 kernel 交互图
![](/public/upload/netty/netty_kernel.png)

## netty做了什么工作

Java 的标准类库，由于其基础性、通用性的定位，往往过于关注技术模型上的抽象，而不是从一线应用开发者的角度去思考。java nio类库的三个基本组件bytebuffer、channel、selector。java只是将这三个组件赤裸裸的提供给你，线程模型由我们自己决定采用，数据协议由我们自己制定并解析。

从功能角度
1. 从网络协议的角度，Netty 除了支持传输层的 UDP、TCP、SCTP协议，也支持 HTTP(s)、WebSocket 等多种应用层协议，它并不是单一协议的 API。
2. 在应用中，需要将数据从 Java 对象转换成为各种应用协议的数据格式，或者进行反向的转换，Netty 为此提供了一系列扩展的编解码框架，与应用开发场景无缝衔接，并且性能良好。
3. 扩展了 Java NIO Buffer，提供了自己的 ByteBuf 实现，并且深度支持 Direct Buffer 等技术，甚至 hack 了 Java 内部对 Direct Buffer 的分配和销毁等。同时，Netty 也提供了更加完善的 Scatter/Gather 机制实现。

单独从性能角度，Netty 在基础的 NIO 等类库之上进行了很多改进，例如：
1. 更加优雅的 Reactor 模式实现、灵活的线程模型、利用 EventLoop 等创新性的机制，可以非常高效地管理成百上千的 Channel。
2. 充分利用了 Java 的 Zero-Copy 机制，并且从多种角度，“斤斤计较”般的降低内存分配和回收的开销。例如，使用池化的 Direct Buffer 等技术，在提高 IO 性能的同时，减少了对象的创建和销毁；**利用反射等技术直接操纵 SelectionKey**，使用数组而不是 Java 容器等。
3. 使用更多本地代码。例如，直接利用 JNI 调用 Open SSL 等方式，获得比 Java 内建 SSL 引擎更好的性能。
4. 在通信协议、序列化等其他角度的优化。



||java.nio|netty|
|---|---|---|
|bytebuf||netty的bytebuf提供的接口与nio的bytebuffer是一致的，只是功能的增强，**bytebuf只有在编解码器中才会用到**|
|selector||完全隐藏|
|channel||完全重写|
|线程模型||固定好了|

## 分层视角

![](/public/upload/netty/netty_layer.png)

1. 网络通信层：它执行网络 I/O 操作，核心组件包含 BootStrap、ServerBootStrap、Channel。——Channel 通道，提供了基础的 API 用于操作网络 IO，比如 bind、connect、read、write、flush 等等。它以 JDK NIO Channel 为基础，提供了更高层次的抽象，同时屏蔽了底层 Socket 的复杂性。Channel 有多种状态，比如连接建立、数据读写、连接断开。随着状态的变化，Channel 处于不同的生命周期，背后绑定相应的事件回调函数。
2. 事件调度层：它的核心组件包含 EventLoopGroup、EventLoop。——EventLoop 本质是一个线程池，主要负责接收 Socket I/O 请求，并分配事件循环器来处理连接生命周期中所发生的各种事件。
3. 服务编排层：它的职责实现网络事件的动态编排和有序传播——ChannelPipeline 基于责任链模式，方便业务逻辑的拦截和扩展；本质上它是一个双向链表将不同的 ChannelHandler 链接在一块，当 I/O 读写事件发生时, 会依次调用 ChannelHandler 对 Channel(Socket) 读取的数据进行处理。



## channel

netty channel: A nexus to a network socket or a component which is capable of I/O operations such as read, write, connect, and bind.

1. All I/O operations are asynchronous.
2. Channels are hierarchical,   channel有一套继承结构

channel如何重写呢？AbstractChannel类特别能说明问题。**AbstractChannel聚合了所有channel使用到的能力对象，由AbstractChannel提供初始化和统一封装，如果功能和子类强相关，则定义成抽象方法，由子类具体实现。**
```java
AbstractChannel{
    Channel parent;
    Unsafe unsafe;
    // 读写操作全部转到pipeline上
    DefaultChannelPipeline pipeline;
    EventLoop eventloop;
    // 因为channel 提供纯异步结构，除了正常的io通信外，close 等操作也需要提供 xxFuture 返回，接收close future回调并执行
    SuccessedFuture,ClosedFuture,voidPromise,unsafeVoidPromise
    localAddress,remoteAddress
}
```
    
## pipeline

filter能够**以声明的方式**插入到http请求响应的处理过程中。

inbound事件通常由io线程触发，outbound事件通常由用户主动发起。

ChannelPipeline的代码相对比较简单，**内部维护了一个ChannelHandler的容器和迭代器**（pipeline模式都是如此），可以方便的进行ChannelHandler的增删改查。

1. ChannelPipeline
2. DefaultChannelPipeline
3. ChannelHandler
4. ChannelHandlerContext,**Enables a ChannelHandler to interact with its ChannelPipeline and other handlers. ** A handler can notify the next ChannelHandler in the ChannelPipeline,modify the ChannelPipeline it belongs to dynamically.

几个类之间的关系

channelpipeline保有channelhandler的容器，这在java里实现办法可就多了

1. channelpipeline直接保有一个list（底层实现可以是array或者list）
2. 链表实现，Channelpipeline只保有一个header引用（想支持特性更多的话，就得加tail）。只不过这样有一个问题，handler本身要保有一个next引用。如果既想这么做，又想让handler干净点，那就得加一个channelhandlercontext类，替handler保有next引用。

代码如下

```java
channelpipeline{
    channelhandlercontext header;
}
channelhandlercontext{
    channelhandler handler;
    channelhandlercontext next;
    EventExecutor executor;
    @Override
    public ChannelHandlerContext fireChannelActive() {
        final AbstractChannelHandlerContext next = findContextInbound();
        EventExecutor executor = next.executor();
        if (executor.inEventLoop()) {
            next.invokeChannelActive();
        } else {
            executor.execute(new OneTimeTask() {
                @Override
                public void run() {
                    next.invokeChannelActive();
                }
            });
        }
        return this;
    }
    private void invokeChannelActive() {
        try {
            ((ChannelInboundHandler) handler()).channelActive(this);
        } catch (Throwable t) {
            notifyHandlerException(t);
        }
    }
}
```


从这就可以看到，Channelhandlercontext不只是替Channelhandler保有下next指针，将pipeline的fireChannelxxx 转化为channelhandler的channelxxx方法。

A list of ChannelHandlers which handles or intercepts inbound events and outbound operations of a Channel.  ChannelPipeline implements an advanced form of the Intercepting Filter pattern to give a user full control over how an event is handled and how the ChannelHandlers in a pipeline interact with each other.

inbound 的被称作events， outbound 的被称作operations。

![Alt text](/public/upload/java/netty_pipeline.png)

从这个图就可以佐证，pipeline作为数据处理模型，不介入io模型，也不介入线程模型。

## 小结

回过头来再看，java nio类库的三个基本组件bytebuffer、channel、selector，数据的读写就是这三个组件的相互作用，线程模型的选择留给用户。netty则是使用eventloop隐藏了selector（将selector和线程绑在一起），使用pipeline封装了数据的处理，**在它们复杂关系的背后，它们的起点，或许还是那个最简单的NIOServer程序。**







