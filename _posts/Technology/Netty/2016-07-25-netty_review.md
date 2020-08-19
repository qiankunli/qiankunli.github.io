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

## 几种io模型代码的直观感受

《Netty权威指南》开篇使用各种io模型实现了一个TimeServer和TimeClient

BIO的实现

```java	
public class TimeServer{
    public static void main(String[] args){
        ServerSocket serverSocket = null;
        ...
        while(true){
            Socket socket = serverSocket.accept();
            new Thread(new TimeServerHandler()).start();
        }
    }
}
public class TimeServerHandler implements Runnable{
    private Socket socket;
    public void run(){
        BufferedReader in = null;
        PrintWriter out = null;
        try{...}catch(Exception e){...}
    }
}
```
    
NIO的实现

```java
public class TimeServerHandler implements Runnable{
    private selector selector;
    private ServerSocketChannel servChannel
    public void run(){...}
}
```
    
AIO的实现

```java
public class TimeServerHandler implements Runnable{
    AsynchronousServerSocketChannel asyncServerSocketChannel;
    public void run(){
        CountDownLatch latch = new CountDownLatch(1);
        asyncServerSocketChannel.accept(this,new CompletionHandler(){
            public void completed(AsynchronousSocketChannel channel,TimeServerHandler attachment){
                channel opt...
            }
        });
        latch.await();
    }
}
```

网络数据读写，一方是用户线程，一方是内核处理，AIO、NIO和BIO，正体现了生产者和消费者两方线程的几种交互方式。从TimeServerHandler类成员的不同，就可以看到使用方式的差异。**AIO和NIO都需要我们显式的提供线程去驱动数据的读写和处理**，AIO由jdk底层的线层池负责回调，并驱动读写操作。

## java原生NIO类库

java nio类库的三个基本组件bytebuffer,channel,selector, 它们是spi接口，java并不提供详细的实现（由jvm提供），java只是将这三个组件赤裸裸的提供给你，线程模型由我们自己决定采用，数据协议由我们自己制定并解析。

首先我们要了解java nio原生的类体系。以Channel interface为例，Channel,InterruptibleChannel,SelectableChannel等interface逐步扩展了Channel的特性。java源码中channel的注释：A nexus（连结、连系） for I/O operations。

[Java NIO Tutorial](http://tutorials.jenkov.com/java-nio/index.html)In the standard IO API you work with byte streams and character streams. In NIO you work with channels and buffers. Data is always read from a channel into a buffer, or written from a buffer to a channel.

```java
public interface Channel extends Closeable {
    public boolean isOpen();
    public void close() throws IOException;
}
// 并没有新增方法，只是说明，实现这个接口的类，要支持Interruptible特性。
public interface InterruptibleChannel
    extends Channel
    public void close() throws IOException;
}
```

A channel that can be asynchronously closed and interrupted. A channel that implements this interface is asynchronously closeable: **If a thread is blocked in an I/O operation on an interruptible channel then another thread may invoke the channel's close method.  This will cause the blocked thread to receive an AsynchronousCloseException.**

这就解释了，好多类携带Interruptible的含义。

```java 
public abstract class SelectableChannel extends AbstractInterruptibleChannel implements Channel{
        // SelectorProvider，Service-provider class for selectors and selectable channels.
    public abstract SelectorProvider provider();
    public abstract int validOps();
    public abstract boolean isRegistered();
    public abstract SelectionKey register(Selector sel, int ops, Object att)
        throws ClosedChannelException;
    public final SelectionKey register(Selector sel, int ops)
        throws ClosedChannelException{
        return register(sel, ops, null);
    }
    public abstract SelectableChannel configureBlocking(boolean block)
        throws IOException;
    public abstract boolean isBlocking();
    public abstract Object blockingLock();
}
```

In order to be used with a selector, an instance of this class must first be registered via the register method.  This method returns a new SelectionKey object that represents the channel's registration with the selector.
  
通过以上接口定义，我们可以知道，Channel接口定义的比较宽泛，理论上bio也可以实现Channel接口。所以，**我们在分析selector和Channel的关系时，准确的说是分析selector与selectableChannel的关系:它们是相互引用的。**selector和selectableChannel是多对多的关系，数据库中表示多对多关系，需要一个中间表。面向对象表示多对多关系则需要一个中间对象，SelectionKey。selector和selectableChannel都持有这个selectionkey集合。

## netty做了什么工作

Java 的标准类库，由于其基础性、通用性的定位，往往过于关注技术模型上的抽象，而不是从一线应用开发者的角度去思考。java nio类库的三个基本组件bytebuffer、channel、selector。java只是将这三个组件赤裸裸的提供给你，线程模型由我们自己决定采用，数据协议由我们自己制定并解析。

单独从性能角度，Netty 在基础的 NIO 等类库之上进行了很多改进，例如：
1. 更加优雅的 Reactor 模式实现、灵活的线程模型、利用 EventLoop 等创新性的机制，可以非常高效地管理成百上千的 Channel。
2. 充分利用了 Java 的 Zero-Copy 机制，并且从多种角度，“斤斤计较”般的降低内存分配和回收的开销。例如，使用池化的 Direct Buffer 等技术，在提高 IO 性能的同时，减少了对象的创建和销毁；**利用反射等技术直接操纵 SelectionKey**，使用数组而不是 Java 容器等。
3. 使用更多本地代码。例如，直接利用 JNI 调用 Open SSL 等方式，获得比 Java 内建 SSL 引擎更好的性能。
4. 在通信协议、序列化等其他角度的优化。

从功能角度
1. 从网络协议的角度，Netty 除了支持传输层的 UDP、TCP、SCTP协议，也支持 HTTP(s)、WebSocket 等多种应用层协议，它并不是单一协议的 API。
2. 在应用中，需要将数据从 Java 对象转换成为各种应用协议的数据格式，或者进行反向的转换，Netty 为此提供了一系列扩展的编解码框架，与应用开发场景无缝衔接，并且性能良好。
3. 扩展了 Java NIO Buffer，提供了自己的 ByteBuf 实现，并且深度支持 Direct Buffer 等技术，甚至 hack 了 Java 内部对 Direct Buffer 的分配和销毁等。同时，Netty 也提供了更加完善的 Scatter/Gather 机制实现。

||java.nio|netty|
|---|---|---|
|bytebuf||netty的bytebuf提供的接口与nio的bytebuffer是一致的，只是功能的增强，**bytebuf只有在编解码器中才会用到**|
|selector||完全隐藏|
|channel||完全重写|
|线程模型||固定好了|

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
    
为什么重写后的Channel会有这么多成员呢？这事儿得慢慢说。

## how netty works

笔者曾经读过一本书《how tomcat works》，从第一个例子十几行代码开始讲述tomcat是如何写出来的，此处也用类似的风格描述下。

我们先从一个最简单的NIOServer代码示例开始，单线程模型：

	public class NIOServer {
        public static void main(String[] args) throws IOException {
            Selector selector = Selector.open();
            ServerSocketChannel serverSocketChannel = ServerSocketChannel.open();
            serverSocketChannel.configureBlocking(false);
            serverSocketChannel.socket().bind(new InetSocketAddress(8080));
            serverSocketChannel.register(selector, SelectionKey.OP_ACCEPT);
            while (true) {
            	   // 所有连接、所有事件阻塞在一处 
                selector.select(1000);
                Set<SelectionKey> selectedKeys = selector.selectedKeys();
                Iterator<SelectionKey> it = selectedKeys.iterator();
                SelectionKey key = null;
                while (it.hasNext()) {
                    key = it.next();
                    it.remove();
                    handleKey(key);
                }
            }
        }
        public static void handleKey(SelectionKey key) throws IOException {
            if (key.isAcceptable()) {
                // Accept the new connection
                ServerSocketChannel ssc = (ServerSocketChannel) key.channel();
                SocketChannel sc = ssc.accept();
                sc.configureBlocking(false);
                // Add the new connection to the selector
                sc.register(key.selector(), SelectionKey.OP_READ | SelectionKey.OP_WRITE);
                System.out.println("accept...");
            } else if (key.isReadable()) {
                SocketChannel sc = (SocketChannel) key.channel();
                ByteBuffer readBuffer = ByteBuffer.allocate(1024);
                // handle buffer
                int count = sc.read(readBuffer);
                if (count > 0) {
                    String receiveText = new String(readBuffer.array(), 0, count);
                    System.out.println("服务器端接受客户端数据--:" + receiveText);
                }
            }
        }
	}

**以下忽略main方法的书写。**
    
我们对上述代码进行简单的抽取，将`while(it.hasNext()){..}，handleKey(){...}`抽取到一个worker线程中。**这样的线程有个学名，叫eventloop，**于是

	class NIOServer{
		main(){
			ServerSocketChannel ...
       	while(true){
       		selector.select(1000);
       		new Worker(SelectionKey).start();
       	}
		}
    }
    
当然了，大家都提倡将acceptable和read/write event分开，我们可以换个方式抽取原始代码:boss和worker线程都执行`while(true){selector.select(1000);...}`,只不过boss专门处理acceptable事件，worker只处理r/w事件。

	class NIOServer{
    	ServerSocketChannel ...
        Selector selectror = ...
        new Boss(selector).start();
    	 new Worker(selector).start();
    }
    
    
boss和worker共享一个selector虽然简单，但是扩展性太低，因此让boss和worker各用各的selector，boss thread accept得到的socketchannel通过queue传给worker，worker从queue中取下socketChannel"消费"（将socketChannel注册到selector上，interest读写事件）。简单实现如下：

```java
class NIOServer{
    Queue<SocketChannel> queue = ...
    new Boss(queue).start();
    new Worker(queue).start();
}
```
    
除了共享queue，传递新accept的socket channel另一种方法是，boss thread保有worker thread的引用，worker thread除了run方法，还提供registerSocketChannel等方法。这样，boos thread就可以通过`worker.registerSocketChannel`把得到的SocketChannel注册到worker thread 的selector。

**说句题外话，笔者以前分解的代码都是静态的，简单的说就是将一个类分解为多个类。本例中，代码分解涉及到了线程，线程对象不只有一个run方法，还可以具备registerChannel的功能。所以，在nio中，线程模型与nio通信代码的结合，不只是new Thread(runnable).start()去驱动代码执行，还深入到了代码的分解与抽象中。**

然后再将Boss和worker线程池化，是不是功德圆满了呢？还没有.

nio类库提供给用户的三个基本操作类bytebuffer,channel,selector，虽然抽象程度低，但简单明了，直接提供read/write data的接口。以我们目前的抽象，netty程序的驱动来自boss和worker thread，问题来了？读取的数据怎么处理（尤其是复杂的处理），我们如何主动地写入数据呢？总得给用户一个**入口对象**。(任何框架，总得有一个入口对象供用户使用，比如fastjson，JSON对象就是其对应的入口对应。比如rabbitMQ，messageListner是其读入口对象，rabbitTemplate是其写入口对象)。

netty选择将channel作为写的入口对象，将channel从worker thread中提取出来，channel提出来之后，worker thread便需要提供自己（内部的selector）与channel交互的手段，比如register方法。

channel提出来之后，读写数据的具体逻辑代码也要跟着channel提取出来，这样worker thread中的代码可以更简洁。但本质上还是`worker.handlekey`才知道什么时候读到了数据，什么时候可以写数据。因此，channel支持触发数据的读写，但读写数据的时机还是由work thread决定。我们要对channel作一定的封装。伪代码如下

```java
ChannelFacade{
    channel	// 实际的channel
    writeBuffer	// 发送缓冲区 
    handleReadData(Buffer){}	// 如何处理读到的数据，由worker thread触发	
    write()					// 对外提供的写数据接口
    doWrite()			// 实际写数据，由workerThread触发
    workerThread		// 对应的Channel
}
class NIOServer{
        ServerSocektChannel srvSocketChannel = ...
    new Boss(srvSocketChannel){};
    new Worker().start();
}
class Boss extends Thread{
    public void run(){
        SocketChannel socketChannel = srvSocketChannel.accept();
        ChannelFacade cf = facade(socketChannel);
        worker.register(cf); //如果cf保有workerThread引用的话，也可以
        cf.register();
    }
} 
```

将channel与其对应的reactor线程剥离之后，一个重要的问题是：**如何确保channel.read/write是线程安全的。**一段代码总在一个线程下执行，那么这段代码就是线程安全的，每个channel（或channel对应的channelhandler，ChannelhandlerContext）持有其约定reactor线程的引用，每次执行时判断下：如果在绑定的reactor线程，则直接执行，如果不在约定线程，则向约定线程提交本任务。

**channelhandler一门心思处理业务数据，channelhandlercontenxt触发事件函数的调用，并保证其在绑定的reactor线程下执行**

这样，我们就以《how tomcat works》的方式，猜想了netty的主要实现思路，当然，netty的实现远较这个复杂。但复杂在提高健壮性、丰富特性上，主要的思路应该是这样的。

## 读写事件的处理

我们提到，将channel从work thread抽取出来后，channel和 work thread的交互方式。

1. read由work thread驱动，work thread 通过select.select()得到selectkey中拿到channel和niosocketchannel（保存在attachment中），就可以调用netty socketchannel的读方法。
2. write 由netty socketchannel直接驱动，但问题是,socketchannel作为入口对象，`socketchanel.write`可能在多个线程中被调用，多个线程同时执行`channel.write`，同样都是目的缓冲区，你写点，我写点，数据就乱套了。**重复一下** 解决方法就是，为每个channel绑定一个work thread（一个work thread可以处理多个channel，一个channel却只能被同一个work thread处理）即netty socketchannel持有了work thread引用，执行chanel.write时先判断现在是不是在自己绑定的work thread，是，则直接执行；如果不是，则向work thread提交一个任务，work thread在合适的时机处理（work thread有一个任务队列）。


read的处理过程:worker thread触发`unsafe.read ==>  pipeline.fireChannelRead ==> head(channelhandlercontext).fireChannelRead`

```java
if ((readyOps & (SelectionKey.OP_READ | SelectionKey.OP_ACCEPT)) != 0 || readyOps == 0) {
    unsafe.read();
    if (!ch.isOpen()) {
        // Connection already closed - no need to handle write.
        return;
    }
}
```
            
write分为两条线：

1. worker thread在可写的时候，调用`unsafe.forceFlush() == AbstractUnsafe.flush0() ==> doWrite(outboundBuffer)`，将写缓冲区数据发出。
2. 用户ctx.write的时候，一直运行到`headContext.write ==> unsafe.write()`，将数据加入到写缓冲区中。
    ```java
    AbstractUnsafe{
        ChannelOutboundBuffer outboundBuffer	// 写缓冲区
        write(msg)		将数据加入到outboundBuffer中
        dowrite()	// 实际的发送数据
    }
    if ((readyOps & SelectionKey.OP_WRITE) != 0) {
        // Call forceFlush which will also take care of clear the OP_WRITE once there is nothing left to write
        ch.unsafe().forceFlush();
    }
    ```

DefaultChannlePipeline有一个HeadContext和TailContext，是默认的pipeline的头和尾，outbound事件会从tail outbound context开始，一直到headcontenxt。

```java
@Override
public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise) throws Exception {
    unsafe.write(msg, promise);
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


从这就可以看到，Channelhandlercontext不只是替Channelhandler保有下next指针，将pipeline的fireChannelxxx 转化为channelhandler的channelxxx方法。

A list of ChannelHandlers which handles or intercepts inbound events and outbound operations of a Channel.  ChannelPipeline implements an advanced form of the Intercepting Filter pattern to give a user full control over how an event is handled and how the ChannelHandlers in a pipeline interact with each other.

inbound 的被称作events， outbound 的被称作operations。


![Alt text](/public/upload/java/netty_pipeline.png)

从这个图就可以佐证，pipeline作为数据处理模型，不介入io模型，也不介入线程模型。

## unsafe

Unsafe operations that should never be called from user-code. These methods
are only provided to implement the actual transport, and must be invoked from an I/O thread except for the
following methods:localAddress(),remoteAddress(),closeForcibly(),register(EventLoop, ChannelPromise）,voidPromise()

每一个channel都大体对应一个unsafe内部类/接口。
	
|channel |unsafe|
|---|---|
|Channel|Unsafe|
|AbstractChannel|AbstractUnsafe|
|AbstractNioChannel|NioUnsafe,AbstractNioUnsafe|
|AbstractNioByteChannel|NioByteUnsafe|

## ChannelPool

上层接口

1. ChannelPoolHandler，Handler which is called for various actions done by the  ChannelPool.ChannelPool的很多操作完成后会触发
2. ChannelHealthChecker， Called before a Channel will be returned via  ChannelPool.acquire() or ChannelPool.acquire(Promise). 在用户通过acquire方法获取channel的时候，确保返回的channel是健康的。
3. ChannelPool，Allows to acquire and release Channel and so act as a pool of these.
4. ChannelPoolMap,Allows to map  ChannelPool implementations to a specific key.将channelpool映射到一个特殊的key上。这个key通常是InetSocketAddress，记一个地址映射多个channel。

        public interface ChannelPool extends Closeable {
            Future<Channel> acquire();
            Future<Channel> acquire(Promise<Channel> promise);
            Future<Void> release(Channel channel);
            Future<Void> release(Channel channel, Promise<Void> promise);
            void close();
        }


ChannelPool有两个简单实现simplechannelpool和FixedChannelPool，后者可以控制Channel的最大个数。但相对于common-pool，其在minActive，minIdle等控制上还是不足的。所以笔者在实现时，最终还是选择基于common-pool2实现基于netty的channel pool。
    
基于common-pool2实现基于netty的channel pool需要注意的是：

1. 空闲Channel的连接保持。一个简单的解决方案是心跳机制，即向channel的pipeline中添加发送与接收心跳请求与响应的Handler。
2. common-pool 池的存储结构选择先进先出的队列，而不是先进后出的堆栈。

## 小结

回过头来再看，java nio类库的三个基本组件bytebuffer、channel、selector，数据的读写就是这三个组件的相互作用，线程模型的选择留给用户。netty则是使用eventloop隐藏了selector（将selector和线程绑在一起），使用pipeline封装了数据的处理，**在它们复杂关系的背后，它们的起点，或许还是那个最简单的NIOServer程序。**







