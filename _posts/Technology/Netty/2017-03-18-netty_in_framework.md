---

layout: post
title: netty在框架中的使用套路
category: 技术
tags: Netty
keywords: JAVA in framework

---

## 前言

服务端网络开发的基本套路

![](/public/upload/architecture/network_communication.png)

简单的netty client demo是

	public class TimeClient {
	    public static void main(String[] args) throws InterruptedException {
	        String host = "127.0.0.1";
	        int port = 8080;
	        new TimeClient().connect(host, port);
	    }
	    public void connect(String host,int port) throws InterruptedException{
	        EventLoopGroup workerGroup = new NioEventLoopGroup();
	        try {
	            Bootstrap b = new Bootstrap();
	            b.group(workerGroup).channel(NioSocketChannel.class).option(ChannelOption.SO_KEEPALIVE, true)
	                    .handler(new ChildChannelHandler());
	            ChannelFuture f = b.connect(host, port).sync();
	            // 此处，你其实可以直接使用f.writeAndFlush发送数据
	            // 等待关闭
	            f.channel().closeFuture().sync();
	        } finally {
	            workerGroup.shutdownGracefully();
	        }
	    }
	    private class ChildChannelHandler extends ChannelInitializer<SocketChannel> {
	        protected void initChannel(SocketChannel arg0) throws Exception {
	            arg0.pipeline().addLast(new TimeClientHandler());
	        }
	    }
	}
	
首先，该代码启动一个进程，进程的目的是启动netty。而通常框架中，netty以及其实现的网络通信，只是框架功能的一个基础部分。我们如何对netty进行封装，使其“返璞归真”，回归到java socket原来的api的使用感觉：`socket.write(byte[])`，甚至于借助netty的特性，提供异步操作的api。

或者说，**一个通用的通信分层框架是一个什么样的结构，而上述netty client demo代码如何分散或适配在这个框架中，这是一个很有意思的部分。**

最近在学习zookeeper的源码，zk client的transport层提供java原生nio和netty两种实现。基于zk中netty使用方式的借鉴和自己的思考，我实现了一个基于netty的、通用的transport层框架，参见[qiankunli/pigeon
](https://github.com/qiankunli/pigeon)

## 定义netty transport层与上层的边界

nio/netty的一些特点

1. **nio/netty本质上是异步的，同步接口需要另外包装。**为何？因为nio或者netty本身用了底层OS的异步特性，可以控制读写的逻辑，却无法控制读写的时机。==> 只能使用缓冲区收发数据，或者说，**缓冲区成为业务程序和nio/os底层交互的媒介。**
2. 对于网络数据传输，需要制定一个通信协议。尤其是，如何定义一段有意义的数据的开始与结束。这就需要事先定义好协议model、以及对协议model的编解码。

具体的说，在实现一个transport层框架之前，我们要想清楚，什么是业务层要传入的，什么是transport层要解决的。

业务层要传入的

1. 业务协议数据请求model、响应model（请求和响应model可以是同一个）定义及其序列化逻辑。对应zk就是CreateRequest、DeleteRequest、CreateResponse等
2. 对于transport层server端，需要业务层传入协议数据处理逻辑，即将根据请求model返回响应model。

transport层负责的

1. 通用数据请求model、响应model（可以是同一个）定义及其序列化逻辑。对应zk就是Packet。**为什么transport层还需要一个通用的model？**因为数据model的收发需要一些辅助字段，比如客户端收到一个响应model，要和其对应的请求model关联起来，这就需要一个id字段。**而transport层model通常和业务层协议model不是同一个**，因为层次之间共用model会导致transport层字段暴露到上层。

	
2. 连接的可靠性检测，比如收发ping/pong消息，如有异常，及时反馈到上层
3. transport client 接口异步机制的实现，callback/future，这个通常有两个实现方案

	* 维护一个map及全局id产生器，每次请求`put<id,request>`，从响应中得到一个`<id,response>`,通过id将request和response关联起来。
	* 


我们经常说，分层，但分层的关键在哪里，如果层之间的接口设计不好，不仅上层会侵染下层，下层也会侵染上层，比如netty数据的读取是在回调方法中，此时上层要想获得响应

|上下层交互|具体形式|对协议model的影响|
|---|---|---|
|推的方式|上层对下层传入callback,下层存储`<id,callback>`映射。在netty读取到响应的回调方法中，根据返回数据id找到并调用callback|request和response packet共用一个id维持关联关系|
|拉的方式|上层与下层共用一个`<id,packet>`，这个map是上下层的接口之一。在netty读取到响应的回调方法中，根据id找到并给packet 的状态字段赋值。上层轮询packet的状态字段值|packet中要有一个状态字段|


[浅谈TCP/IP网络编程中socket的行为](http://www.cnblogs.com/promise6522/archive/2012/03/03/2377935.html)提到：无论是磁盘io还是网络io，应用程序乃至r/w系统调用都不负责数据实际的读写（接收/发送），这些控制皆发生在TCP/IP栈中，对应用程序是透明的。系统调用及之上的应用程序和tcp协议栈就是通过send buffer和receive buffer沟通。

而对于netty和上层框架来说，**netty本身是一个死循环的系统**：等待io事件然后处理，外界可以提交任务（比如写数据）交给这个循环系统执行。这个循环系统也会在读到数据时，执行设定的handler。执行设定的handler，里面可以做的文章就比较多了。

## zk 使用netty的一些特别之处

zookeeper中采用“拉的方式”，但transport层并没有维护`<id，packet>`。因为zookeeper client确保了发送数据请求（ping等请求是另一种逻辑）的有序性，因此上下层共用一个packet queue即可。

zk transport层提供了两种方案：nio和netty。即ClientCnxnSocket的两个实现类ClientCnxnSocketNIO和ClientCnxnSocketNetty。**netty比直接使用nio强的地方在于（或者说netty做了哪些工作）：**固化了线程模型与nio的结合方式，同时将编解码的过程、pipeline的思想融入处理过程中，使得“nio与线程结合”的方式，由"百家争鸣"（hadoop传文件块对nio的使用 VS zk对nio的使用）变成“独尊儒术”。

zk client 实现中，netty收到数据后，只是简单的将字节流写入到zk自定义的缓冲区，并未将编解码过程融入到netty运行过程中。最开始我以为zk这样做的目的是nio和netty的实现共用一些逻辑(自己手动对自定义缓冲区数据做编解码)。在我自己实现[qiankunli/pigeon
](https://github.com/qiankunli/pigeon)
的过程中，发现zk client的抽象接口是`ReplyHeader submitRequest(RequestHeader h, Record request,
                Record response, WatchRegistration watchRegistration)`，response对象是事先创建好的。若套用了netty的编解码流程，response对象将由netty框架生成，再利用其为用户创建的response对象赋值，就多费了一番波折，并且不是很有必要。
                
## pushy

Pushy is a Java library for sending APNs (iOS, OS X, and Safari) push notifications.

pushy 对netty的使用和zk有所不同

1. zk client本身具备复杂的业务逻辑，netty只是作为transport
2. 和apns交互用的是http2协议，而http2的编解码是可以作为netty 的一个handler“插件”存在的，同时，pushy对外提供的接口也完全是异步的。


	    ApnsPayloadBuilder payloadBuilder = new ApnsPayloadBuilder();
	    payloadBuilder.setAlertBody("Example!");
	    String payload = payloadBuilder.buildWithDefaultMaximumLength();
	    String token = TokenUtil.sanitizeTokenString("<efc7492 bdbd8209>");
	    SimpleApnsPushNotification  pushNotification = new SimpleApnsPushNotification(token, "com.example.myApp", payload);
		Future<PushNotificationResponse<SimpleApnsPushNotification>> sendNotificationFuture = apnsClient.sendNotification(pushNotification);


## 一些技巧

### 使用内部类

一个类如果只是被某一个类引用的话，做成内部类也无妨，虽然外围类代码看起来长了点，但少传了很多参数。比如，下文中ZKClientHandler 的messageReceived方法就像是ClientCnxnSocketNetty的方法一样操作incomingBuffer，便于清晰的观察数据读写的来龙去脉。

	ClientCnxnSocketNetty{
		protected ByteBuffer incomingBuffer = lenBuffer;	// extend from ClientCnxnSocket
		ZKClientHandler extends SimpleChannelUpstreamHandler{
			public void messageReceived(ChannelHandlerContext ctx,
	                                    MessageEvent e) throws Exception {
	                                    	handle incomingBuffer;
	                                    }
		}
	}

### 连接数管理

你是不是觉得，一个netty client只能用一个channel？

通过本文开头的netty client demo，人很容易想当然的认为，netty client 与 channel是一对一关系。事实上不是的，我们可以为client 维护一个channel pool，进而提高通信效率。毕竟只有一个channel，再怎么异步，性能还是有限的。

对于服务端来说，能够支持的总连接数是有限的，如果一个客户端建立了大量的连接，将严重限制服务端可以服务的客户端数，因此服务端针对一个具体的客户端，要有一个连接数（主要是上限）管理。

相关细节参见[通用transport层框架pigeon](http://qiankunli.github.io/2017/03/26/pigeon_design.html)