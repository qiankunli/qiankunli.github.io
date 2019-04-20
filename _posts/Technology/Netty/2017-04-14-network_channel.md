---

layout: post
title: network channel
category: 技术
tags: Netty
keywords: JAVA netty channel

---

## 前言

* TOC
{:toc}

借用知乎上"者年"关于java nio技术栈的回答[如何学习Java的NIO](https://www.zhihu.com/question/29005375/answer/43021911)

0. 计算机体系结构和组成原理 中关于中断，关于内存，关于 DMA，关于存储 等关键知识点
1. 操作系统 中 内核态用户态相关部分，  I/O 软件原理
2. 《UNIX网络编程（卷1）：套接字联网API（第3版）》([美]史蒂文斯，等)一书中 IO 相关部分。
3. [Java I/O底层是如何工作的？](http://www.importnew.com/14111.html)
4. [存储之道 - 51CTO技术博客 中的《一个IO的传奇一生》](http://alanwu.blog.51cto.com/3652632/d-8)
5. [Operating Systems: I/O Systems4. ](https://www.cs.uic.edu/~jbell/CourseNotes/OperatingSystems/13_IOSystems.html)
6. [多种I/O模型及其对socket效率的改进](http://mickhan.blog.51cto.com/2517040/1586370)

根据这个思路，笔者整理了下相关的知识，参见[java io涉及到的一些linux知识](http://qiankunli.github.io/2017/04/16/linux_io.html)

服务端网络开发的基本套路

![](/public/upload/architecture/network_communication.png)


## netty 与 tomcat 对比理解

有一个本书叫《how tomcat works》，从几行代码开始，一步一步的叙述tomcat的实现。我们通过对照，就可以发现此类网络通信框架，要处理或涉及的一些共同问题。

1. io 模型
2. 线程模型，以及io模型与线程模型的结合问题
3. 组件化的数据处理模型，对netty是channelhandler，对tomcat是fillter和tomcat
4. 元数据配置，此类系统都需要传入大量的参数，配置文件的形式是什么，如何读取，又如何在各个组件之间流转。
5. 即使功能已经非常强大，仍然要上层框架支持以完成复杂的业务逻辑。对tomcat是spring mvc等，对netty则是一系列的rpc框架。


## java nio channel

java源码中channel的注释：A nexus（连结、连系） for I/O operations。

不管是os提供的系统调用，还是c提供的一些库api，本质都是一系列函数，移植到java中需要一个对象来负责使用这些函数。PS：很多时候，作者在源码上的注释，本身就很精练、直接，比博客上的赘述要好很多。

Channel用于在字节缓冲区和位于通道另一侧的实体（通常是一个文件或套接字）之间有效地传输数据。

|client|server|
|---|---|
|java buffer <==> java channel <==> system socket send/receive buffer|system socket send/receive buffer <==> java channel <==> java buffer|

[Java NIO Tutorial](http://tutorials.jenkov.com/java-nio/index.html)
In the standard IO API you work with byte streams and character streams. In NIO you work with channels and buffers. Data is always read from a channel into a buffer, or written from a buffer to a channel.

channel的io操作本身是非阻塞的，要想不无的放矢，就要借助selector。

## netty channel

A nexus to a network socket or a component which is capable of I/O
operations such as read, write, connect, and bind.

1. All I/O operations are asynchronous.
2. Channels are hierarchical

下文以AbstractChannel为例来说明这个两个特点：

	AbstractChannel{
		Unsafe unsafe;
		DefaultChannelPipeline pipeline;
		EventLoop eventLoop;
		
		
		ChannelFuture succeededFuture;
		CloseFuture closeFuture;
	}

### asynchronous

所有的io操作都是异步，那么有channelFuture就是理所当然了。因为即便是异步，也得告诉你什么时候写完了，什么时候读完了，以及r/w之后做什么，这时就需要一个nexus对象（callback或future）。并且，对于异步来说，可能A线程触发读操作，而实际处理读到数据的是B线程。

单纯的nio，是不支持异步的（channel只是搬运buffer罢了，你并不能告诉channel，数据写完了通知你一声儿），这就意味一系列的封装操作。并且netty channel不同方法的封装方式是不一样的。比如，AbstractChannel有一些future成员，为什么呢？read、write、connect等操作的future的返回是AbstractChannel具体负责io的unsafe、pipeline等成员负责。剩下的close（channel作为一个facade类，聚合了多个组件，close操作涉及到多个组件）等操作，由AbstractChannel亲自维护future。

### hierarchical

Channels are hierarchical有以下个含义

1. channel具有父子逻辑，比如一个socketChannel的父可能是一个serverSocketChannel
2. channel有一套继承结构
3. channel聚合了大量功能类，在接口层采用facade模式统一封装
		
从这里就可以看到，netty的channel不再是java nio channel那个只负责r/w操作的纯洁的小伙儿了，这个channel更应该叫Endpoint(Client/server)或channelFacade更合适。

### ChannelPool

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


## channel 和 unsafe的关系

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

netty channel作为对nio channel的增强，有两种增强方式：

1. 继承nio channel。但java nio channel是spi类接口，扩展不易。
2. 重写，在实际读写的位置调用nio channel，最简单的方案：包含

		write(buffer){
			business1
			java.nio.channel.write(buffer)
			business2
		}

而netty channel却采用了聚合的方式，将实际的读写交给unsafe，有以下几个好处：

   1. 不跟netty channel（作为facade）的其它代码、成员放在一起。unsafe更纯粹的组织读写代码。
   2. netty channel除了网络io，还有线程模型(eventloop)、数据处理模型(pipeline)，rw代码交给unsafe后，netty读写就是unsafe、eventloop和pipeline的三国杀，netty channel在外边统筹兼顾。否则，netty channel和eventloop、pipeline就糅合在一起，纠缠不清了。
	
因此，理解netty channel，就要理解unsafe、pipeline和eventloop的三国杀。我们反过来想想，正是理清了这三者的关系，unsafe、pipeline和eventloop才有了自己的继承体系，最后被netty channel揉和在一起。

## pipeline

A list of ChannelHandlers which handles or intercepts inbound events and outbound operations of a Channel.  ChannelPipeline implements an advanced form of the Intercepting Filter pattern to give a user full control over how an event is handled and how the ChannelHandlers in a pipeline interact with each other.

inbound 的被称作events， outbound 的被称作operations。


![Alt text](/public/upload/java/netty_pipeline.png)

从这个图就可以佐证，pipeline作为数据处理模型，不介入io模型，也不介入线程模型。

## eventloop

[Netty 核心组件 EventLoop 源码解析](https://www.jianshu.com/p/2b42954c2983)

### 存在形式

![](/public/upload/netty/NioEventLoop_Inheritance.PNG)

使用红框标出了重点部分：PS： 下面的思维方式很重要

1. ScheduledExecutorService 接口表示是一个定时任务接口，EventLoop 可以接受定时任务。
2. EventLoop 接口：Netty 接口文档说明该接口作用：一旦 Channel 注册了，就处理该Channel对应的所有I/O 操作。
3. SingleThreadEventExecutor 表示这是一个单个线程的线程池。

一个 EventLoop 将由一个永远都不会改变的Thread 驱动，同时任务（Runnable 或者 Callable）可以直接提交给 EventLoop 实现，以立即执行或者调度执行。根据配置和可用核心的不同，可能会创建多个 EventLoop 实例用以优化资源的使用，并且单个 EventLoop 可能会被指派用于服务多个 Channel（PS： 但一个channel只归属一个EventLoop，以线程安全）。

`eventloop.execute(task)` 实现，**主要这段代码由调用线程执行**

    public void execute(Runnable task) {
        boolean inEventLoop = inEventLoop();
        if (inEventLoop) {
            addTask(task);
        } else {
            startThread();
            addTask(task);
            if (isShutdown() && removeTask(task)) {
                reject();
            }
        }
        if (!addTaskWakesUp && wakesUpForTask(task)) {
            wakeup(inEventLoop);
        }
    }

首先判断该 EventLoop 的线程是否是当前线程，如果是，直接添加到任务队列中去，如果不是，则尝试启动线程（但由于线程是单个的，因此只能启动一次），随后再将任务添加到队列中去。如果线程已经停止，并且删除任务失败，则执行拒绝策略，默认是抛出异常。

### EventLoop 中的 Loop 到底是什么？

    protected void run() {
        for (;;) {
			switch (...) {
				case SelectStrategy.CONTINUE:
					...
				case SelectStrategy.SELECT:
					this.select(this.wakenUp.getAndSet(false));
					...
				default:
				    cancelledKeys = 0;
					needsToSelectAgain = false;
					final int ioRatio = this.ioRatio;
					if (ioRatio == 100) {
						try {
							processSelectedKeys();
						} finally {
							runAllTasks();
						}
					} else {
						final long ioStartTime = System.nanoTime();
						try {
							processSelectedKeys();
						} finally {
							// Ensure we always run tasks.
							final long ioTime = System.nanoTime() - ioStartTime;
							runAllTasks(ioTime * (100 - ioRatio) / ioRatio);
						}
					}
			}			
            // Always handle shutdown even if the loop processing threw an exception.
			...
        }
    }

整个 run 方法做了3件事情：

1. selector 获取感兴趣的事件。
2. processSelectedKeys 处理事件。
3. runAllTasks 执行队列中的任务。

在 Netty 中，有2种任务，一种是 IO 任务，一种是非 IO 任务，ioRatio 的作用就是限制执行任务队列的时间。

	private void select(boolean oldWakenUp) throws IOException {
        Selector selector = this.selector;
		int selectCnt = 0;
		long currentTimeNanos = System.nanoTime();
		long selectDeadLineNanos = currentTimeNanos + this.delayNanos(currentTimeNanos);
		while(true) {
			long timeoutMillis = (selectDeadLineNanos - currentTimeNanos + 500000L) / 1000000L;
			if (timeoutMillis <= 0L) {
				if (selectCnt == 0) {
					selector.selectNow();
					selectCnt = 1;
				}
				break;
			}
			if (this.hasTasks() && this.wakenUp.compareAndSet(false, true)) {
				selector.selectNow();
				selectCnt = 1;
				break;
			}
			int selectedKeys = selector.select(timeoutMillis);
			++selectCnt;
			if (selectedKeys != 0 || oldWakenUp || this.wakenUp.get() || this.hasTasks() || this.hasScheduledTasks()) {
				break;
			}
			if (Thread.interrupted()) {
				selectCnt = 1;
				break;
			}
			long time = System.nanoTime();
			if (time - TimeUnit.MILLISECONDS.toNanos(timeoutMillis) >= currentTimeNanos{
				selectCnt = 1;
			} else if (SELECTOR_AUTO_REBUILD_THRESHOLD > 0 && selectCnt >= SELECTOR_AUTO_REBUILD_THRESHOLD) {
				this.rebuildSelector();
				selector = this.selector;
				selector.selectNow();
				selectCnt = 1;
				break;
			}
		}
    }

整段代码都在弄一个事情：是`selector.selectNow();` 还是 `selector.select(timeoutMillis)`

这段逻辑可不孤单，[Redis源码分析](http://qiankunli.github.io/2019/04/20/redis_source.html) 中的eventloop 异曲同工

## channel 以及 unsafe pipeline eventloop 三国杀

**Channel是netty提供的操作对象，其能力是通过聚合unsafe、pipeline和eventloop来实现的。 也即是说，channel只是外皮，分析unsafe、pipeline和eventloop的相互作用才是学习Netty的关键。**

eventloop 有一个selector 成员，selector.select() 得到selectorKey，selectorKey 可以获取到channel（channel执行 `SelectionKey register(Selector sel, int ops, Object att)`时会将自己作为attr传入），进而通过channel得到unsafe进行读写操作。（channel提供的操作能力，一方面供coder使用，一方面供eventloop使用）

对于读取，是eventloop ==> channel.unsafe.read ==>              channel.pipeline.fireChannelRead(byteBuf);

对于实际的写，eventloop ==> unsafe.forceFlush()

也就是说，三国杀里，eventloop只需负责在合适的时间通过channel操作unsafe==>pipeline或pipeline ==> unsafe即可。

对于写，则是 channel.write ==> pipeline.writeAndFlush(msg) ==> HeadContext.write ==> unsafe.write ==> outboundBuffer.addMessage


java nio channel每次读写都要缓冲区。对于netty channel来说（具体是unsafe），读缓冲区是读取时临时申请一个buffer，写则事先分配的一个缓冲区。

一个channel自打注册到selector后，不是一直interest r/w事件的，比如out buffer有数据了才关心，没数据了就remove interest，这样可以提高性能。

## 小结

一个稍微复杂的框架，必然伴随几个抽象以及抽象间的依赖关系，那么依赖的关系的管理，可以选择spring（像大多数j2ee项目那样），也可以硬编码。这就是我们看到的，每个抽象对象有一套自己的继承体系，然后抽象对象子类之间又彼此复杂的交织。比如Netty的eventloop、unsafe和pipeline，channel作为最外部操作对象，聚合这三者，根据聚合的子类的不同，Channel也有多个子类来体现。

同时，做一个粗略的对应

|模型|代码抽象|
|---|---|
|io模型|unsafe|
|线程模型|eventloop|
|数据处理模型|pipeline|
