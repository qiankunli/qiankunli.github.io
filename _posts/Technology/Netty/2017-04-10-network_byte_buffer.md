---

layout: post
title: network byte buffer
category: 技术
tags: Netty
keywords: JAVA netty ByteBuffer

---

## 前言

笔者最近在重温netty的实现，收获很多，主要有两个方面：

1. 看到很多以前不太重视的部分
2. 以前以一个初学者的角度来学习netty，更多看重术的部分；现在以一个熟练使用者的角度来看，更多看中里面的一些思想。这些解决问题的思想，体现了解决问题的思路。

## linux 文件操作系统调用接口

 1. open：打开文件。
 2. read：从已打开的文件中读取数据。
 3. write：向已打开的文件中写入数据。
 4. close：关闭已打开的文件。
 5. ioctl：向文件传递控制信息或发出控制命令。

以r/w为例

	ssize_t read(int fd, void *buf, size_t count);
	ssize_t write(int fd, const void *buf, size_t count);
	
另外还有一个不常备注意的

 	int ioctl(int fd, int request, ...);
 
ioctl操作用于向文件发送控制命令，这些命令不能被视为是输入输出流的一部分，而只是影响文件的操作方式。

与java io/nio通过socket/channel 作为文件io系统调用的nexus不同，java nio /netty byte buffer只是作为数据的载体。
	

## byte buffer

### java nio byte buffer


对于java nio，ByteBuffer有两个子类HeapByteBuffer和DirectByteBuffer的不同。

|ByteBuffer子类|底层数据存储|分配释放|使用场景|优缺点|
|---|---|---|---|---|
|HeapByteBuffer|byte[]|由堆分配和释放|消息编解码||
|DirectByteBuffer|long base = sun.misc.Unsafe.allocateMemory(size)|java对象本身归gc管理，其对应的内存由单独的组件负责释放|io收发|io效率更高,分配和释放效率低|

### netty ByteBuf


对于netty ByteBuffer呢，除了Heap和Direct的不同，为了减少高负载环境下ByteBuffer创建和销毁的GC时间（ByteBuffer用作io`缓冲`，而为了复用ByteBuffer，进而减少内存分配和释放，用了一个更大的`缓存`——对象池），有了Unpooled和Pooled的区别，这样就有了四种组合。

Pooled有助于提高效率，奈何也有瑕疵，参加下文。


## netty的buffer管理

为什么netty的buffer需要管理，java io/nio时期，怎么没有这样的事儿？因为，java io/nio对外的接口就是 `socket/channel.write(byte[]/byte buffer)`,byte[] 由jvm直接管理。这一部分，可以参见zk client的ClientCnxnSocketNIO实现。而netty则是`channel.unsafe.write(Object msg, ChannelPromise promise)`，bytebuffer直接由netty负责，因此netty可以直接采取一些手法，使其更高效.


[深入浅出Netty内存管理：PoolChunk](http://blog.jobbole.com/106001/)开篇说的太好了：多年之前，从C内存的手动管理上升到java的自动GC，是历史的巨大进步。然而多年之后，netty的内存实现又曲线的回到了手动管理模式，正印证了马克思哲学观：社会总是在螺旋式前进的，没有永远的最好。的确，就内存管理而言，GC给程序员带来的价值是不言而喻的，不仅大大的降低了程序员的负担，而且也极大的减少了内存管理带来的Crash困扰，不过也有很多情况，可能手动的内存管理更为合适。

因为没有专门的gc线程，因此**buffer的回收是手动的**，只是netty替我们封装了我们看不到，但在一些复杂的业务逻辑里，还是需要我们注意忘记释放的问题。

引用计数负责确定buffer回收的时机，池负责buffer对象的管理。

### netty内存管理

PooledArena和Recycler 是什么关系？

	PooledByteBufAllocator.newHeapBuffer{
	  	PoolArena.allocate {
	  		PooledByteBuf<T> buf = newByteBuf(maxCapacity); ==> PooledHeapByteBuf.newInstance(maxCapacity); {
	  			 PooledHeapByteBuf buf = RECYCLER.get();
	  			 ...
	  		}
	  		allocate(cache, buf, reqCapacity);
	  	}
	}
	
Recycler负责对象的分配与回收，PooledArena负责buffer对象引用内存的分配与回收。

对象池、内存池


### 对象池 基本实现

Recycler ： Light-weight object pool based on a thread-local stack.

参见[Netty轻量级对象池实现分析](http://www.cnblogs.com/hzmark/p/netty-object-pool.html)

文章中提到几个核心的点：

1. Stack相当于是一级缓存，同一个线程内的使用和回收都将使用一个Stack
2. 每个线程都会有一个自己对应的Stack，如果回收的线程不是Stack的线程，将元素放入到Queue中
3. 所有的Queue组合成一个链表，Stack可以从这些链表中回收元素（实现了多线程之间共享回收的实例）

我比较关注的点是，为什么不使用common pool？

1. common pool 从数据结构上限定线程安全，
2. Recycler 则是通过ThreadLocal实现线程安全性，线程可以简单直接的访问该线程域下的Stack。

### diect memory

[Understanding Java heap memory and Java direct memory](http://fibrevillage.com/sysadmin/325-understanding-java-heap-memory-and-java-direct-memory)

diect memory不是在Java heap上，那么这块内存的大小是多少呢？默认是一般是64M，可以通过参数：-XX:MaxDirectMemorySize来控制。

直接内存的释放并不是由你控制的，而是由full gc来控制的，直接内存会自己检测情况而调用system.gc()

## 引用计数

### java内存回收的局限性

为什么要有引用计数器？[netty的引用计数](http://www.cnblogs.com/gaoxing/p/4249119.html)解释的很清楚。

总的来说，java gc实现了gc的自动化。但其gc机制是通用的，针对一些具体的场景，比如网络通信中的编解码（大量的对象创建与释放），力有不逮。并且，full gc比较耗时，所以，在一些性能要求比较高的程序中，还是要不嫌麻烦，主动进行对象回收，以避免jvm进行full gc。参见[Netty轻量级对象池实现分析](http://www.cnblogs.com/hzmark/p/netty-object-pool.html)

### 引用计数的局限性

使用引用计数之后，我们就有了自己的一套对象池、以及对象管理与分配机制（netty中学名叫Arena），就会出现内存泄漏的可能：即java gc将对象回收了（java gc有自己的回收机制，不管Arena的引用计数是否为0），但以Arena角度看，该对象的引用计数不是0，故其占用的内存不会被Arena重新分配。参见[Netty文档之引用计数对象](http://www.wolfbe.com/detail/201609/377.html#)

## netty内存泄露的防止

[Netty之有效规避内存泄漏](http://calvin1978.blogcn.com/articles/netty-leak.html)

直接内存是IO框架的绝配，但直接内存的分配销毁不易，所以使用内存池能大幅提高性能，也告别了频繁的GC。但，**要重新培养被Java的自动垃圾回收惯坏了的惰性。**


## 引用

[Netty学习之旅----源码分析内存分配与释放原理](http://46aae4d1e2371e4aa769798941cef698.devproxy.yunshipei.com/prestigeding/article/details/54692464)

[《Netty官方文档》引用计数对象](http://ifeve.com/reference-counted-objects/)

[Netty文档之引用计数对象](http://www.wolfbe.com/detail/201609/377.html#)

[netty的引用计数](http://www.cnblogs.com/gaoxing/p/4249119.html)