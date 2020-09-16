---

layout: post
title: 硬件对软件设计的影响
category: 技术
tags: Basic
keywords: hardware software

---

## 简介

* TOC
{:toc}

物质基础决定上层建筑，在计算机世界里，硬件作为“物质基础”深刻的影响着软件设计和性能。

国外一个大牛的博客 [Mechanical Sympathy](https://mechanical-sympathy.blogspot.com/) Hardware and software working together in harmony 讲的是底层硬件是如何运作的，以及与其协作而非相悖的编程方式。[剖析Disruptor:为什么会这么快？（二）神奇的缓存行填充](http://ifeve.com/disruptor-cacheline-padding/) 作为一个开发者你可以逃避不去了解CPU、数据结构或者大O符号 —— 而我用了10年的职业生涯来忘记这些东西。但是现在看来，如果你知道这些知识并应用它，你能写出一些非常巧妙和非常快速的代码。

2019.4.12 补充：[进程管理信息数据结构](http://qiankunli.github.io/2017/02/14/linux_art_review.html) 二进制文件分段 ==> 进程分段 ==> 指令操作码/操作数 ==> cpu运算单元/数据单元 ==>  cpu代码段寄存器/数据段寄存器/堆栈段寄存器等，从这个视角看，又有一种软硬件融合的味道。


## 并行计算能力

[Java和操作系统交互细节](https://mp.weixin.qq.com/s/fmS7FtVyd7KReebKzxzKvQ)

![](/public/upload/linux/cpu_work.jpeg)

### CPU 的指令执行

一行代码能够执行，必须要有可以执行的上下文环境，包括：指令寄存器、数据寄存器、栈空间等内存资源。

**为什么可以流水线/乱序执行？**我们通常会把 CPU 看做一个整体，把 CPU 执行指令的过程想象成，依此检票进站的过程，改变不同乘客的次序，并不会加快检票的速度。所以，我们会自然而然地认为改变顺序并不会改变总时间。但当我们进入 CPU 内部，会看到 CPU 是由多个功能部件构成的。**一条指令执行时要依次用到多个功能部件，分成多个阶段**，虽然每条指令是顺序执行的，但每个部件的工作完成以后，就可以服务于下一条指令，从而达到并行执行的效果。比如典型的 RISC 指令在执行过程会分成前后共 5 个阶段。

1. IF：获取指令；
2. ID（或 RF）：指令解码和获取寄存器的值；
3. EX：执行指令；
4. ME（或 MEM）：内存访问（如果指令不涉及内存访问，这个阶段可以省略）；
5. WB：写回寄存器。

**在执行指令的阶段，不同的指令也会由不同的单元负责**。所以在同一时刻，不同的功能单元其实可以服务于不同的指令。


[从JVM并发看CPU内存指令重排序(Memory Reordering)](http://ifeve.com/jvm-memory-reordering/)我们知道现代CPU的主频越来越高，与cache的交互次数也越来越多。当CPU的计算速度远远超过访问cache时，会产生cache wait，过多的cache wait就会造成性能瓶颈。针对这种情况，多数架构（包括X86）采用了一种将cache分片的解决方案，即将一块cache划分成互不关联地多个 slots (逻辑存储单元，又名 [Memory Bank](https://en.wikipedia.org/wiki/Memory_bank) 或 Cache Bank)，CPU可以自行选择在多个 idle bank 中进行存取。这种 SMP(指在一个计算机上汇集了一组处理器,各CPU之间共享内存子系统以及总线结构) 的设计，显著提高了CPU的并行处理能力，也回避了cache访问瓶颈。

Memory Bank的划分：一般 Memory bank 是按cache address来划分的。比如 偶数adress 0×12345000分到 bank 0, 奇数address 0×12345100分到 bank1。

重排序的种类

* 编译期重排。编译源代码时，编译器依据对上下文的分析，对指令进行重排序，使其更适合于CPU的并行执行。
* 运行期重排，CPU在执行过程中，动态分析依赖部件的效能（CPU0检查 bank0 的可用性，发现 bank0 处于 busy 状态，那么本来写入cache bank0的数据操作会延后），对指令做重排序优化。

前者是编译器进行的，不同语言不同。后者是cpu 层面的，所有使用共享内存模型进行线程通信的语言都要面对的。

### cpu/服务器 三大体系numa smp mpp

1. SMP(Symmetric Multi-Processor) 所谓对称多处理器结构，是指服务器中多个CPU对称工作，无主次或从属关系。各CPU共享相同的物理内存，每个 CPU访问内存中的任何地址所需时间是相同的，因此SMP也被称为一致存储器访问结构(UMA：Uniform Memory Access)。SMP服务器的主要特征是共享，系统中所有资源(CPU、内存、I/O等)都是共享的。也正是由于这种特征，导致了SMP服务器的主要问题，那就是它的扩展能力非常有限。对于SMP服务器而言，每一个共享的环节都可能造成SMP服务器扩展时的瓶颈，而最受限制的则是内存。由于每个CPU必须通过相同的内存总线访问相同的内存资源，因此随着CPU数量的增加，内存访问冲突将迅速增加，最终会造成CPU资源的浪费，使 CPU性能的有效性大大降低。实验证明，SMP服务器CPU利用率最好的情况是2至4个CPU。      
2. NUMA(Non-Uniform Memory Access)基本特征是具有多个CPU模块，每个CPU模块由多个CPU(如4个)组成，并且具有独立的本地内存、I/O槽口等。由于其节点之间可以通过互联模块(如称为Crossbar Switch)进行连接和信息交互，因此每个CPU可以访问整个系统的内存(这是NUMA系统与MPP系统的重要差别)。显然，访问本地内存的速度将远远高于访问远地内存(系统内其它节点的内存)的速度，这也是非一致存储访问NUMA的由来。由于这个特点，为了更好地发挥系统性能，开发应用程序时需要尽量减少不同CPU模块之间的信息交互。
3. MPP(Massive Parallel Processing)其基本特征是由多个SMP服务器(每个SMP服务器称节点)通过节点互联网络连接而成，每个节点只访问自己的本地资源(内存、存储等)，是一种**完全无共享(Share Nothing)结构**，因而扩展能力最好，理论上其扩展无限制。在MPP系统中，每个SMP节点也可以运行自己的操作系统、数据库等。但和NUMA不同的是，它不存在异地内存访问的问题。换言之，每个节点内的CPU不能访问另一个节点的内存。节点之间的信息交互是通过节点互联网络实现的，这个过程一般称为数据重分配(Data Redistribution)。但是MPP服务器需要一种复杂的机制来调度和平衡各个节点的负载和并行处理过程。

## 多层次内存结构

RAM 分为动态和静态两种，静态 RAM 由于集成度较低，一般容量小，速度快，而动态 RAM 集成度较高，主要通过给电容充电和放电实现，速度没有静态 RAM 快，所以一般将动态 RAM 做为主存，而静态 RAM 作为 CPU 和主存之间的高速缓存 （cache），用来屏蔽 CPU 和主存速度上的差异，也就是我们经常看到的 L1 ， L2 缓存。

一个 CPU 处理器中一般有多个运行核心，我们把一个运行核心称为一个物理核，每个物理核都可以运行应用程序。每个物理核都拥有私有的一级缓存（Level 1 cache，简称 L1 cache），包括一级指令缓存和一级数据缓存，以及私有的二级缓存（Level 2 cache，简称 L2 cache）。**L1 和 L2 缓存是每个物理核私有的**，不同的物理核还会共享一个共同的三级缓存。另外，现在主流的 CPU 处理器中，每个物理核通常都会运行两个超线程，也叫作逻辑核。同一个物理核的逻辑核会共享使用 L1、L2 缓存。


### 缓存速度的差异

![](/public/upload/basic/cpu_cache.jpg)

|从CPU到|	大约需要的 CPU 周期|	大约需要的时间|
|---|---|---|
|主存||		约60-80纳秒|
|QPI 总线传输(between sockets, not drawn)|		|约20ns|
|L3 cache|	约40-45 cycles,|	约15ns|
|L2 cache|	约10 cycles,|	约3ns|
|L1 cache|	约3-4 cycles,|	约1ns|
|寄存器|	1 cycle|	

当CPU执行运算的时候，它先去L1查找所需的数据，再去L2，然后是L3，最后如果这些缓存中都没有，所需的数据就要去主内存拿。走得越远，运算耗费的时间就越长。如果你的目标是让端到端的延迟只有 10毫秒，而其中花80纳秒去主存拿一些未命中数据的过程将占很重的一块。**如果你在做一些很频繁的事，你要确保数据在L1缓存中**。

当然，缓存命中率是很笼统的，具体优化时还得一分为二。比如，你在查看 CPU 缓存时会发现有 2 个一级缓存，这是因为，CPU 会区别对待指令与数据。虽然在冯诺依曼计算机体系结构中，代码指令与数据是放在一起的，但执行时却是分开进入指令缓存与数据缓存的，因此我们要分开来看二者的缓存命中率。

1. 提高数据缓存命中率，考虑cache line size
2. 提高指令缓存命中率，CPU含有分支预测器，如果分支预测器可以预测接下来要在哪段代码执行（比如 if 还是 else 中的指令），就可以提前把这些指令放在缓存中，CPU 执行时就会很快。例如，如果代码中包含if else，不要让每次执行if else 太过于随机。

在一个 CPU 核上运行时，应用程序需要记录自身使用的软硬件资源信息（例如栈指针、CPU 核的寄存器值等），我们把这些信息称为运行时信息。同时，应用程序访问最频繁的指令和数据还会被缓存到 L1、L2 缓存上，以便提升执行速度。但是，在多核 CPU 的场景下，一旦应用程序需要在一个新的 CPU 核上运行，那么，运行时信息就需要重新加载到新的 CPU 核上。而且，新的 CPU 核的 L1、L2 缓存也需要重新加载数据和指令，这会导致程序的运行时间增加。因此，操作系统（调度器）提供了将进程或者线程绑定到某一颗 CPU 上运行的能力（PS：就好像将pod 调度到上次运行它的node）。建议绑定物理核，以防止绑到一个逻辑核时，因为任务较多导致目标线程迟迟无法被调度的情况。

### 缓存的存取——缓存行

[高性能队列——Disruptor](https://tech.meituan.com/disruptor.html)

cpu和内存的速度差异 ==> 缓存 ==> 多级缓存 ==> Cache是由很多个cache line组成的。每个cache line通常是64字节，并且它有效地引用主内存中的一块儿地址。CPU每次从主存中拉取数据时，会把相邻的数据也存入同一个cache line。也就是说，假设一个cache line 对应内存地址是0x1000，存着一个volatile变量，你改了这个变量，那么跟它挨着的另一个变量（地址为0x1008）也会失效（假设它们同属于一个java对象内存结构，或都是某个数组的元素）因为整个cache line被标记为失效了。下次访问第二个变量时，便需要从内存中加载到缓存，再加载到cpu。从某种程度上可以说：**cpu一直是批量访问缓存/内存的**。

因此，缓存行中的64byte 数据，一个失效全失效，有时会带来一些性能问题。

[JVM4——《深入拆解java 虚拟机》笔记
2018年07月20日](http://qiankunli.github.io/2018/07/20/jvm_note.html)因为 缓存行，jvm 使用了字段内存对齐机制。

volatile 字段和缓存行也有一番故事

### 缓存一致性问题——内存屏障

Compiler 和 cpu 经常搞一些 optimizations，这种单线程视角下的优化在多线程环境下是不合时宜的，为此要用 memory barriers 来禁止 Compiler 和 cpu 搞这些小动作。 For purposes here, I assume that the compiler and the hardware don't introduce funky optimizations (such as eliminating some "redundant" variable reads, a valid optimization under a single-thread assumption).

[老司机谈技术天花板——做自己的破壁人](https://mp.weixin.qq.com/s?__biz=MzA4MDc5OTg5MA==&mid=2650585155&idx=3&sn=30392c82e2003ca54e248b6a7abbee88&mpshare=1&scene=1&srcid=0331lAZn3kCrRoyxDwVkfS7P#rd)

硬件为了加快速度，会弄各种缓存，然后就有一个缓存/副本一致性问题，也会定一些一致性规则（什么时候同步之类）。但基本就是，你不明确要求，硬件就当缓存是有效的。那么就有了打掉缓存的指令（即强制同步），然后编译器/runtime 支持该指令，最终反映在语言层面上。

[Java和操作系统交互细节](https://mp.weixin.qq.com/s/fmS7FtVyd7KReebKzxzKvQ)插入内存屏障的指令，会根据指令类型不同有不同的效果，例如在 monitorexit 释放锁后会强制刷新缓存，而 volatile 对应的内存屏障会在每次写入后强制刷新到主存，并且由于 volatile 字段的特性，编译器无法将其分配到寄存器，所以每次都是从主存读取，所以 volatile 适用于读多写少得场景


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
|上下文切换|数千个CPU时钟周期，1微秒|

## 单核CPU技术瓶颈 ==> CPU 向多核发展 ==> 多台服务器

2019.3.28 补充

1. 语言层面，golang协程、java9 支持反应式等
2. 架构层面，全异步化、反应式架构、分布式计算

通过语言层、框架层提出新的模型，引导写出并行度高的代码