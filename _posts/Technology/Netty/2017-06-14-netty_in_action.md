---

layout: post
title: 《netty in action》读书笔记
category: 技术
tags: Netty
keywords: JAVA netty

---

## 前言

## java nio

the earliest versions of java introduced enough of an **object-oriented facade** to hide some of the thornier details.but those first java apis supported only the so-called blocking functions provided by the native system socket libraries.

相当的java库就是对底层c库做object-oriented facade

java nio provide considerably more control over the utiliaztion of network resources:

* Using `setsockopt()`,you can configure sockets so that read/write calls will return immediately if there is no data; 是不是blocking其实就是socket 的一个opt，不用牵扯其它的
* you can register a set of non-blocking sockets using the **system's event notification api** to determine whether any of them have data ready for reading or writing. 

select 其实是一种event notification service

this model provides much better resource management than the blocking i/o model:

* many connecitons can be handled with fewer threads,and thus with far less overhead due to memory management and context-switching. 为每个线程分配栈空间是要占内存的
* threads can be **retargeted** to other tasks when there is no i/o to handle.这个retargeted很传神

## netty

在提交netty的一些组件时，作者提到**think of them as domain objects rather than concrete java classes**.

### Channel、ChannelPipeline、ChannelHandler和ChannelHandlerContext的一对一、一对多关系

netty is asynchronous and event-driven. 

every new channel that is created is assigned a new ChannelPipeline.This association is permanent;the channel can neither attach another ChannelPipeline nor detach the current one.

a ChannelHandlerContext represents an association between a ChannelHandler and ChannelPipeline and is created whenever a ChannelHandler is added to a ChannelPipeline.

the movement from one handler to the next at the ChannelHandler level is invoked on the ChannelHandlerContext.

a ChannelHandler can belong to more than one ChannelPipeline,it can be bound to multiple ChannelHandlerContext instances.

### thread model


**同步io，线程和连接通常要维持一对一关系。异步io才可以一个线程处理多个io。**

首先，netty的数据处理模型

1. 以event notification 来处理io
2. event分为inbound 和 outbound
3. event 由handler处理，handler形成一个链pipeline

数据处理模型与线程模型的结合，便是

1. channel和eventloop是多对一关系
2. channel的inbound和outbound事件全部由eventloop处理
3. 根据1和2，outbound事件可以由多个calling thread触发，但只能由一个eventloop处理。那么就需要将多线程调用转换为任务队列。

the basic idea of an event loop

	while(!terminated){
		List<Runnable> readyEvents = blockUntilEventsReady();
		for(Runnable ev: readyEvents){
			ev.run();
		}
	}
	
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

