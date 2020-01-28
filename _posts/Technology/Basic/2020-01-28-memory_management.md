---

layout: post
title: 内存管理
category: 技术
tags: Basic
keywords: Permission

---

## 简介

* TOC
{:toc}

一个优秀的通用内存分配器应具有以下特性:

1. 额外的空间损耗尽量少
2. 分配速度尽可能快
3. 尽量避免内存碎片
4. 缓存本地化友好
5. 通用性，兼容性，可移植性，易调试

**内存管理可以分为三个层次**，自底向上分别是：

1. 操作系统内核的内存管理
2. glibc层使用系统调用维护的内存管理算法。glibc/ptmalloc2，google/tcmalloc，facebook/jemalloc
3. 应用程序从glibc动态分配内存后，根据应用程序本身的程序特性进行优化， 比如netty的arena

## 操作系统

![](/public/upload/linux/linux_memory_management.png)

[Linux内核基础知识](http://blog.zhifeinan.top/2019/05/01/linux_kernel_basic.html)

![](/public/upload/linux/linux_virtual_address.png)

## 语言/运行时

进程在推进运行的过程中会调用一些数据，可能中间会产生一些数据保留在堆栈段，栈空间 一般在编译时确定， 堆空间 运行时动态管理。对于堆具体的说，由runtime 在启动时 向操作 系统申请 一大块内存，然后根据 自定义策略进行分配与回收（手动、自动，分代不分代等），必要时向操作系统申请内存扩容。

### jvm

[java gc](http://qiankunli.github.io/2016/06/17/gc.html)

[JVM1——jvm小结](http://qiankunli.github.io/2014/10/27/jvm.html)极客时间《深入拆解Java虚拟机》垃圾回收的三种方式（免费的其实是最贵的）

1. 清除sweep，将死亡对象占据的内存标记为空闲。
2. 压缩，将存活的对象聚在一起
3. 复制，将内存两等分， 说白了是一个以空间换时间的思路。

### go runtime

**与java 多个线程、也不管对象大小  直接地操作一个堆相比，go 的内存分配 就精细多了**。

Golang的内存分配器原理与tcmalloc类似，简单的说就是维护一块大的全局内存，每个线程(Golang中为P)维护一块小的私有内存，私有内存不足再从全局申请。

以64位系统为例，Golang程序启动时会向系统申请的内存如下图所示：

![](/public/upload/go/go_memory_layout.jpg)

预申请的内存划分为spans、bitmap、arena三部分。其中arena即为所谓的堆区，应用中需要的内存从这里分配。其中spans和bitmap是为了管理arena区而存在的。

1. arena的大小为512G，为了方便管理把arena区域划分成一个个的page，每个page为8KB,一共有512GB/8KB个页；
2. spans区域存放span的指针，每个指针对应一个page，所以span区域的大小为(512GB/8KB)*指针大小8byte = 512M
3. bitmap区域大小也是通过arena计算出来，不过主要用于GC。


span是用于管理arena页的关键数据结构（src/runtime/mheap.go type span struct），每个span中包含1个或多个连续页。

从需求侧来说，根据对象大小，划分了一系列class，每个class都代表一个固定大小的对象。

每个span用于管理特定的class对象, 根据对象大小，span将一个或多个页拆分成多个块进行管理。

有了管理内存的基本单位span，还要有个数据结构来管理span，这个数据结构叫mcentral。

    type mcentral struct {
        lock      mutex     //互斥锁
        spanclass spanClass // span class ID
        nonempty  mSpanList // non-empty 指还有空闲块的span列表
        empty     mSpanList // 指没有空闲块的span列表
        nmalloc uint64      // 已累计分配的对象个数
    }

从mcentral数据结构可见，每个mcentral对象只管理特定的class规格的span。事实上每种class都会对应一个mcentral,这个mcentral的集合存放于mheap数据结构中。

    type mheap struct {
        lock      mutex
        spans []*mspan
        bitmap        uintptr 	//指向bitmap首地址，bitmap是从高地址向低地址增长的
        arena_start uintptr		//指示arena区首地址
        arena_used  uintptr		//指示arena区已使用地址位置
        central [67*2]struct {
            mcentral mcentral
            pad      [sys.CacheLineSize - unsafe.Sizeof(mcentral{})%sys.CacheLineSize]byte
        }
    }

各线程需要内存时从mcentral管理的span中申请内存，为了避免多线程申请内存时不断的加锁，Golang为每个线程分配了span的缓存，这个缓存即是cache。

    type mcache struct {
        alloc [67*2]*mspan // 按class分组的mspan列表
    }

![](/public/upload/go/go_memory_alloc.jpg)


[图解 Go 内存分配器](https://www.infoq.cn/article/IEhRLwmmIM7-11RYaLHR)内存分配策略很多，但核心本质上是一致的：**针对不同大小的对象，在不同的 cache 层中，使用不同的内存结构；将从系统中获得的一块连续内存分割成多层次的 cache，以减少锁的使用以提高内存分配效率；申请不同类大小的内存块来减少内存碎片，同时加速内存释放后的垃圾回收**。

![](/public/upload/go/go_memory_strategy.png)

## 中间件

### netty arena

代码上的体现 以netty arena 为例 [netty内存管理](http://qiankunli.github.io/2017/04/10/network_byte_buffer.html)

### kafka BufferPool

[《Apache Kafka源码分析》——Producer与Consumer](http://qiankunli.github.io/2017/12/08/kafka_learn_1.html)

### redis

[Redis源码分析](http://qiankunli.github.io/2019/04/20/redis_source.html)

