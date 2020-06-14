---

layout: post
title: 内存管理玩法汇总
category: 技术
tags: Basic
keywords: memory management

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
2. glibc层使用系统调用维护的内存管理算法。glibc/ptmalloc2，google/tcmalloc，facebook/jemalloc。C 库内存池工作时，会预分配比你申请的字节数更大的空间作为内存池。比如说，当主进程下申请 1 字节的内存时，Ptmalloc2 会预分配 132K 字节的内存（Ptmalloc2 中叫 Main Arena），应用代码再申请内存时，会从这已经申请到的 132KB 中继续分配。当我们释放这 1 字节时，Ptmalloc2 也不会把内存归还给操作系统。
3. 应用程序从glibc动态分配内存后，根据应用程序本身的程序特性进行优化， 比如netty的arena

![](/public/upload/basic/memory_pool.jpg)

进程/线程与内存：Linux 下的 JVM 编译时默认使用了 Ptmalloc2 内存池，64 位的 Linux 为每个线程的栈分配了 8MB 的内存，还预分配了 64MB 的内存作为堆内存池。在多数情况下，这些预分配出来的内存池，可以提升后续内存分配的性能。但也导致创建很多线程时会占用大量内存。

## 操作系统

![](/public/upload/linux/linux_memory_management.png)

[Linux内核基础知识](http://blog.zhifeinan.top/2019/05/01/linux_kernel_basic.html)

![](/public/upload/linux/linux_virtual_address.png)


## 语言/运行时

进程在推进运行的过程中会调用一些数据，可能中间会产生一些数据保留在堆栈段，栈空间 一般在编译时确定（**在编译器和操作系统的配合下，栈里的内存可以实现自动管理**）， 堆空间 运行时动态管理。对于堆具体的说，由runtime 在启动时 向操作 系统申请 一大块内存，然后根据 自定义策略进行分配与回收（手动、自动，分代不分代等），必要时向操作系统申请内存扩容。

从堆还是栈上分配内存？

如果你使用的是静态类型语言，那么，不使用 new 关键字分配的对象大都是在栈中的，通过 new 或者 malloc 关键字分配的对象则是在堆中的。对于动态类型语言，无论是否使用 new 关键字，内存都是从堆中分配的。

**为什么从栈中分配内存会更快？**由于每个线程都有独立的栈，所以分配内存时不需要加锁保护，而且栈上对象的尺寸在编译阶段就已经写入可执行文件了，执行效率更高！性能至上的 Golang 语言就是按照这个逻辑设计的，即使你用 new 关键字分配了堆内存，但编译器如果认为在栈中分配不影响功能语义时，会自动改为在栈中分配。

当然，在栈中分配内存也有缺点，它有功能上的限制。一是， 栈内存生命周期有限，它会随着函数调用结束后自动释放，在堆中分配的内存，并不随着分配时所在函数调用的结束而释放，它的生命周期足够使用。二是，栈的容量有限，如 CentOS 7 中是 8MB 字节，如果你申请的内存超过限制会造成栈溢出错误（比如，递归函数调用很容易造成这种问题），而堆则没有容量限制。

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

## gc

[深入浅出垃圾回收（一）简介篇](https://liujiacai.net/blog/2018/06/15/garbage-collection-intro/) 是一个系列，垃圾回收的很多机制是语言无关的。

![](/public/upload/jvm/gc.png)

### gc 策略

![](/public/upload/basic/gc_strategy.png)

### 最基本的 mark-and-sweep 算法伪代码

mutator 通过 new 函数来申请内存

    new():
        ref = allocate()
        if ref == null
            collect()
            ref = allocate()
            
            if ref == null
                error "Out of memory"
        return ref

collect 分为mark 和 sweep 两个基本步骤

    atomic collect():  // 这里 atomic 表明 gc 是原子性的，mutator 需要暂停
        markFromRoots()
        sweep(heapStart, heapEnd)

从roots 对象开始mark
        
    markFromRoots():
        initialize(worklist)
        for each reference in Roots  // Roots 表示所有根对象，比如全局对象，stack 中的对象
            if ref != null && !isMarked(reference)
                setMarked(reference)
                add(worklist, reference)
                mark()          // mark 也可以放在循环外面
                       
    initialize():
        // 对于单线程的collector 来说，可以用队列实现 worklist
        worklist = emptyQueue()

    //如果 worklist 是队列，那么 mark 采用的是 BFS（广度优先搜索）方式来遍历引用树                
    mark():
        while !isEmpty(worklist):
            ref = remove(worklist)  // 从 worklist 中取出第一个元素
            for each field in Pointers(ref)  // Pointers(obj) 返回一个object的所有属性，可能是数据，对象，指向其他对象的指针
                child = *field
                if child != null && !isMarked(child)
                    setMarked(child)
                    add(worklist, child)

sweep逻辑

    sweep(start, end):
        scan = start
        while scan < end
            if isMarked(scan)
                unsetMarked(scan)
            else
                free(scan)
            scan = nextObject(scan)
