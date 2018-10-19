---

layout: post
title: 硬件对软件设计的影响
category: 技术
tags: Architecture
keywords: Permission

---

## 简介

物质基础决定上层建筑，在计算机世界里，硬件作为“物质基础”深刻的影响着软件设计和性能。

## 缓存

绝大多数时候，两个线程通过内存共享数据，但每个cpu都会保有一份儿共享数据的副本——缓存，这就容易引起不一致。

### 缓存行

[高性能队列——Disruptor](https://tech.meituan.com/disruptor.html)

cpu和内存的速度差异 ==> 缓存 ==> 多级缓存 ==> Cache是由很多个cache line组成的。每个cache line通常是64字节，并且它有效地引用主内存中的一块儿地址。CPU每次从主存中拉取数据时，会把相邻的数据也存入同一个cache line。也就是说，假设一个cache line 对应内存地址是0x1000，存着一个volatile变量，你改了这个变量，那么跟它挨着的另一个变量（地址为0x1008）也会失效（假设它们同属于一个java对象内存结构，或都是某个数组的元素）因为整个cache line被标记为失效了。下次访问第二个变量时，便需要从内存中加载到缓存，再加载到cpu。

因此，缓存行中的64byte 数据，一个失效全失效，有时会带来一些性能问题。

## 内存 + cpu 二级结构

1. 所谓线程 安全，最后可以归结到 并发读写 问题。参见 [多线程](http://qiankunli.github.io/2014/10/09/Threads.html)
2. 所谓数据结构，最后可以归结到 读写效率的 权衡问题。 参见[hbase 泛谈](http://qiankunli.github.io/2018/04/08/hbase.html) 数据信息 和 结构信息（提高读写效率）混在一起，因为磁盘的缘故， 顺序写即可提高读效率。而查询/读效率 的提高花活儿就比较多了，并且通常 会降低写效率。 

## 内存屏障

Compiler 和 cpu 经常搞一些 optimizations，这种单线程视角下的优化在多线程环境下是不合时宜的，为此要用 memory barriers 来禁止 Compiler 和 cpu 搞这些小动作。 For purposes here, I assume that the compiler and the hardware don't introduce funky optimizations (such as eliminating some "redundant" variable reads, a valid optimization under a single-thread assumption).

[老司机谈技术天花板——做自己的破壁人](https://mp.weixin.qq.com/s?__biz=MzA4MDc5OTg5MA==&mid=2650585155&idx=3&sn=30392c82e2003ca54e248b6a7abbee88&mpshare=1&scene=1&srcid=0331lAZn3kCrRoyxDwVkfS7P#rd)

硬件为了加快速度，会弄各种缓存，然后就有一个缓存一致性问题，也会定一些一致性规则（什么时候同步之类）。但基本就是，你不明确要求，硬件就当缓存是有效的。那么就有了打掉缓存的指令（即强制同步），然后编译器/runtime 支持该指令，最终反映在语言层面上。

## 资源的有限性和需求的无限性


||计算能力|需求||备注|
|---|---|---|---|---|
|软硬件|cpu|创建线程的业务是无限的|用一个数据结构 表示和存放你要执行的任务/线程/进程，任尓干着急，我调度系统按既有的节奏来。|
|java线程池|线程池管理的线程|要干的活儿是无限的|用一个runnable对象表示一个任务，线程池线程依次从队列中取出任务来执行|线程池管理的线程数可以扩大和缩小|
|goroutine|goroutine调度器管理的线程|要干的活儿是无限的|用协程表示一个任务，线程从各自的队列里取出任务执行|A线程干完了，还可以偷B线程队列的活儿干|


## 为什么会有人觉得优化没有必要，因为他们不理解有多耗时


[Teach Yourself Programming in Ten Years](http://norvig.com/21-days.html)

Remember that there is a "computer" in "computer science". Know how long it takes your computer to execute an instruction, fetch a word from memory (with and without a cache miss), read consecutive words from disk, and seek to a new location on disk.


Approximate timing for various operations on a typical PC:

||耗时|
|---|---|
|execute typical instruction|	1/1,000,000,000 sec = 1 nanosec|
|fetch from L1 cache memory|	0.5 nanosec|
|branch misprediction|	5 nanosec|
|fetch from L2 cache memory|	7 nanosec|
|Mutex lock/unlock|	25 nanosec|
|fetch from main memory|	100 nanosec|
|send 2K bytes over 1Gbps network	|20,000 nanosec|
|read 1MB sequentially from memory|	250,000 nanosec|
|fetch from new disk location (seek)|	8,000,000 nanosec|
|read 1MB sequentially from disk	|20,000,000 nanosec|
|send packet US to Europe and back	|150 milliseconds = 150,000,000 nanosec|

