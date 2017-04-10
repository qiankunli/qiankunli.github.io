---

layout: post
title: network byte buffer
category: 技术
tags: Java
keywords: JAVA netty ByteBuffer

---

## 前言（未完成）

笔者最近在重温netty的实现，收获很多，主要有两个方面：

1. 看到很多以前不太重视的部分
2. 以前以一个初学者的角度来学习netty，更多看重术的部分；现在以一个熟练使用者的角度来看，更多看中里面的一些思想。这些解决问题的思想，体现了解决问题的思路。

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


## 引用计数

### java内存回收的局限性

为什么要有引用计数器？[netty的引用计数](http://www.cnblogs.com/gaoxing/p/4249119.html)解释的很清楚。

总的来说，java gc实现了gc的自动化。但其gc机制是通用的，针对一些具体的场景，比如网络通信中的编解码（大量的对象创建与释放），力有不逮。并且，full gc比较耗时，所以，在一些性能要求比较高的程序中，还是要不嫌麻烦，主动进行对象回收，以避免jvm进行full gc。参见[Netty轻量级对象池实现分析](http://www.cnblogs.com/hzmark/p/netty-object-pool.html)

### 引用计数的局限性

使用引用计数之后，我们就有了自己的一套对象池（netty中学名叫Arena）、以及对象管理与分配机制，就会出现内存泄漏的可能：即java gc将对象回收了（java gc有自己的回收机制，不管Arena的引用计数是否为0），但以Arena角度看，该对象的引用计数不是0，故其占用的内存不会被Arena重新分配。参见[Netty文档之引用计数对象](http://www.wolfbe.com/detail/201609/377.html#)


## 引用

[《Netty官方文档》引用计数对象](http://ifeve.com/reference-counted-objects/)

[Netty文档之引用计数对象](http://www.wolfbe.com/detail/201609/377.html#)

[netty的引用计数](http://www.cnblogs.com/gaoxing/p/4249119.html)