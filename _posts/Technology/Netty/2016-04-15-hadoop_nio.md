---

layout: post
title: 向Hadoop学习NIO的使用
category: 技术
tags: Netty
keywords: netty hadoop nio

---

## 前言 ##

如果没有专门进行过网络开发，那么科班出身的程序员对java网络编程的了解到bio就为止了，即便知道nio，也是一个简单地eventloop的例子，这个demo代码应付实际的业务是远远不够的。于是更多的人用了netty，netty将java nio封装的很好，也许封装的太好了，但netty也有它的局限性，比如对大文件传输的支持不是很好。

所以，不管从了解java nio的角度，还是为了在java网络编程时有更多的选择，都应该了解下java nio开发的一些套路。

## 一切都来自SelectorProvider

java nio中几个基本操作对象SocketChannel、ServerSocketChannel和Selector，它们的创建本质上都是调用了SelectorProvider。其OpenJDK实现如下


    public static SelectorProvider create() {  
          PrivilegedAction pa = new GetPropertyAction("os.name");  
          String osname = (String) AccessController.doPrivileged(pa);  
          if ("SunOS".equals(osname)) {//1、如果SunOS  
              return new sun.nio.ch.DevPollSelectorProvider();  
          }  
          //2、Linux 内核>=2.6  
          // use EPollSelectorProvider for Linux kernels >= 2.6  
          if ("Linux".equals(osname)) {  
              pa = new GetPropertyAction("os.version");  
              String osversion = (String) AccessController.doPrivileged(pa);  
              String[] vers = osversion.split("\\.", 0);  
              if (vers.length >= 2) {  
                  try {  
                      int major = Integer.parseInt(vers[0]);  
                      int minor = Integer.parseInt(vers[1]);  
                      if (major > 2 || (major == 2 && minor >= 6)) {  
                          return new sun.nio.ch.EPollSelectorProvider();  
                      }  
                  } catch (NumberFormatException x) {  
                      // format not recognized  
                  }  
              }  
          }  
          return new sun.nio.ch.PollSelectorProvider();  
    }  
    
可以简单归纳如下:

1. 在Solaris系统下将会使用DevPollSelectorProdiver;
2. 在Linux系统下(2.6+版本),将会使用EPollSelectorProvider;
3. 否则将会使用PollSelectorProvider.

对于SunJDK,DefaultSelectorProvider将直接返回WindowsSelectorProvider.这是一种基于Poll机制.

本节引用自[NIO中Channel.spi学习][]

## 牛逼的org.apache.hadoop.net包

众所周知，hadoop将文件分块存储，每块默认有3个副本，那hadoop内部肯定有文件（块）传输操作，这在我们自己实现文件传输时有很大的借鉴意义。相关的代码在`org.apache.hadoop.net`中，主要涉及到四个类SocketInputStream，SocketOutputStream，SocketIOWithTimeout，StandardSocketFactory。笔者初看完这4个类的实现，只有一个感觉：鬼斧神工。

StandardSocketFactory负责创建Socket对象，SocketIOWithTimeout（封装Socket）提供带超时设置的io和connect操作。SocketInputStream和SocketOutputStream有个内部类作为SocketIOWithTimeout的子类，负责对外提供读写方法。很明显，SocketIOWithTimeout是核心，SocketIOWithTimeout的核心是`int doIO(ByteBuffer buf, int ops)`方法，其简化逻辑是

    int doIO(ByteBuffer,ops){
        buffer// 对于read操作，就要要填充的buffer，对于写操作，就是要输出的buffer
		while(buffer.hasremaing()){
			// channel尝试读写，performIO留给子类实现（决定是读还是写）
			int n = performIO(buffer)
			// 如果读到就返回
			if(n != 0){
			    return n;
			}
			// 如果设定的时间内不能readable/wriable
			count = selectorPool.select(channel, ops, timeout)
			if(count == 0){
			    抛出SocketTimeoutException
			}
		}
		return 0
	}


由此可以看到nio使用bytebuffer的一个优势，那就是操作的统一。不管read和write，不管有没有读到或写入数据，不管读取或写入了一个字节还是多个字节，buffer都是参数，不像bio，本质上是`int read()`和`write(int b)`，一次只能一个字节，并且这个字节，在read操作中作为返回值，在write操作中作为参数。读者可以参见InputStream和OutputStream抽象类的源码，`read(byte[])`和`read(byte[],off,len)`本质上是操作read实现的。

## SelectorPool

SocketIOWithTimeout中另一个比较厉害的操作就是`selectorPool.select(channel, ops, timeout)`。selectorPool是一个selector池，调用`select(SelectableChannel channel, int ops, long timeout)`方法就为channel分配一个空闲的selector监听ops操作，一个selector在某个时刻只负责一个channel（（如果监听多个的话，timeout就不准了）），用完回收，如果一个selector空闲时间太长，就关闭它。select方法简化逻辑如下


    int select(SelectableChannel channel, int ops, long timeout){
        // 为channel选择一个空闲的selector，如果没有则创建一个新的selector
        while(true){
            channel.register(selector, ops);
            int ret = selector.select(timeout);
            if (ret != 0) {
			    return ret;
			}
			// 记录循环的耗时，如果timeout了return 0
        }
        // 释放一些资源，回收selector
    }
    
SelectorPool提供的select方法，意义还是蛮大的

1. 以前一个selector负责多个socketchannel，现在一对多变成一对一
2. 我们知道，nio设置成非阻塞后，没有超时一说（只是buffer中有没有数据的区别），而它提供了一种判断超时的机制。
3. 隐藏了selector的创建和销毁。这样，**nio的数据读写就不用eventloop**，编写代码的感觉回归到以socketchannel为主角，跟bio就比较像了。这为我们实现基于nio的socket pool提供了便利。

        public class TestSocketOutputStream {
        	public static void main(String[] args) throws UnknownHostException, IOException {
        		SocketFactory socketFactory = StandardSocketFactory.getDefault();
        		Socket socket = socketFactory.createSocket("127.0.0.1", 8080);
        		SocketOutputStream outputStream = new SocketOutputStream(socket, 5000);
        		outputStream.write("abc".getBytes());
        	}
        }
        

## 怎么又是流

SocketIOWithTimeout虽然是`org.apache.hadoop.net`的核心，但不抛头露面，对外干活的是SocketInputStream和SocketOutputStream，此处以SocketOutputStream为例。

奇怪的是，SocketIOStream继承了OutputStream。本来java nio中，使用channel和buffer后没有stream的事了，为什么到最后又封装成了stream了呢？我估计不是hadoop作者对stream有什么偏爱，而是Stream是一套设计模式。SocketIOStream其实只是解决了“支持超时的读写操作”，把底层实现换成了nio。就好像一个骑两个轮子上班的人，后来开上四个轮子，虽然很爽，但终归还要自己亲自开车，对于习惯司机接送的人来说，还是不够方便。SocketIOStream的不方便之处在于，我要为它准备一个byte[]。

- 如果写入的对象比较复杂，要自己序列化。
- 如果byte[]比较大，会占用很大的内存。如果想占用的少一点，那只能攒一点写一点，自己控制调用write的节奏。


此时，stream的一套东西就很有用，`DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new SocketOutputStream(WritableByteChannel,timeout),BUFFER_SIZE));`

1. SocketOutputStream封装channel的读写操作，支持timeout
2. BufferedOutputStream支持缓冲，用户只管往里写，凑够了BUFFER_SIZE的数据会自动触发SocketOutputStream的write操作。
3. DataOutputStream封装了数据的序列化，免去了字节数组和数据之间的转换过程

《财富论》的开篇讨论了分工，在代码中，对于底层方法而言，要提供一个字节数组，凑够一个约定的数量把它发出去。对于上层调用者而言，它理解的调用时机和数据格式是：我需要发送一个int，发送一个string。这两者的差异，或者就是人与机器的差异。

中间的转换，需要分工去解决，不得不服java的博大精深。

[NIO中Channel.spi学习]: http://shift-alt-ctrl.iteye.com/blog/1841511