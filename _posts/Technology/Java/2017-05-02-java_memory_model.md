---

layout: post
title: java内存模型
category: 技术
tags: Java
keywords: JAVA memory model

---

## 前言（未完成）

详见[JSR 133 (Java Memory Model) FAQ](https://www.cs.umd.edu/~pugh/java/memoryModel/jsr-133-faq.html)及其译文[Java内存模型FAQ（一） 什么是内存模型](http://ifeve.com/memory-model/)，[深入理解Java内存模型（一）——基础](http://www.infoq.com/cn/articles/java-memory-model-1)系列文章

首先，什么是内存模型，为何引入内存模型？ one or more layers of memory cache。缓存能够大大提升性能，但是它们也带来了许多挑战。例如，当两个CPU同时检查相同的内存地址时会发生什么？在什么样的条件下它们会看到相同的值？

||表现|
|---|---|
|在cpu层面上|操作因为缓存的原因是否可见|
|在编译器层面上|重排序| 

The Java Memory Model describes what behaviors are legal in multithreaded code, and how threads may interact through memory（java 内存模型 描述了在多线程环境下哪些行为是合法的，线程之间如何通过共享内存的方式来通信）. It describes the relationship between variables in a program and the low-level details of storing and retrieving them to and from memory or registers in a real computer system. It does this in a way that can be implemented correctly using a wide variety of hardware and a wide variety of compiler optimizations.

Java includes several language constructs, including volatile, final, and synchronized, **which are intended to help the programmer describe a program’s concurrency requirements to the compiler.** The Java Memory Model defines the behavior of volatile and synchronized, and, more importantly, ensures that a correctly synchronized Java program runs correctly on all processor architectures.


## 重排序

### 为什么会出现重排序

重排序的背景，参见[从JVM并发看CPU内存指令重排序(Memory Reordering)](http://ifeve.com/jvm-memory-reordering/)

单cpu时代，因为流水线技术，多个指令同时开始执行，因每个指令的耗时不同，会出现后一个指令先于前一个指令执行完毕的情况。

我们知道现代CPU的主频越来越高，与cache的交互次数也越来越多。当CPU的计算速度远远超过访问cache时，会产生cache wait，过多的cache wait就会造成性能瓶颈。
针对这种情况，多数架构（包括X86）采用了一种将cache分片的解决方案，即将一块cache划分成互不关联地多个 slots (逻辑存储单元，又名 [Memory Bank](https://en.wikipedia.org/wiki/Memory_bank) 或 Cache Bank)，CPU可以自行选择在多个 idle bank 中进行存取。这种 SMP(指在一个计算机上汇集了一组处理器,各CPU之间共享内存子系统以及总线结构) 的设计，显著提高了CPU的并行处理能力，也回避了cache访问瓶颈。

Memory Bank的划分

一般 Memory bank 是按cache address来划分的。比如 偶数adress 0×12345000分到 bank 0, 奇数address 0×12345100分到 bank1。

重排序的种类

* 编译期重排。编译源代码时，编译器依据对上下文的分析，对指令进行重排序，使其更适合于CPU的并行执行。

* 运行期重排，CPU在执行过程中，动态分析依赖部件的效能（CPU0检查 bank0 的可用性，发现 bank0 处于 busy 状态，那么本来写入cache bank0的数据操作会延后），对指令做重排序优化。

前者是编译器进行的，不同语言不同。后者是cpu 层面的，所有使用共享内存模型进行线程通信的语言都要面对的。

### 重排序的影响

主要体现在两个方面，详见[Java内存访问重排序的研究](http://tech.meituan.com/java-memory-reordering.html)

1. 对代码执行的影响

	常见的是，一段未经保护的代码，因为多线程的影响可能会乱序输出。**少见的是，重排序也会导致乱序输出。**

2. 对编译器、runtime的影响，这体现在两个方面：

	1. 运行期的重排序是完全不可控的，jvm经过封装，要保证某些场景不可重排序（比如数据依赖场景下）。提炼成理论就是：happens-before规则（参见《Java并发编程实践》章节16.1），Happens-before的前后两个操作不会被重排序且后者对前者的内存可见。
	2. 提供一些关键字（主要是加锁、解锁），也就是允许用户介入某段代码是否可以重排序。这也是**"which are intended to help the programmer describe a program’s concurrency requirements to the compiler"** 的部分含义所在。

[Java内存访问重排序的研究](http://tech.meituan.com/java-memory-reordering.html)文中提到，内存可见性问题也可以视为重排序的一种。比如，在时刻a，cpu将数据写入到memory bank，在时刻b，同步到内存。cpu认为指令在时刻a执行完毕，我们呢，则认为代码在时刻b执行完毕。


## 内存模型与语言

函数式编程 和 命令式编程中的 消息传递 模型，线程之间不会共享内存，也就没有 内存模型的问题。

JMM属于语言级的内存模型，它确保在不同的编译器和不同的处理器平台之上，通过禁止特定类型的编译器重排序和处理器重排序，为程序员提供一致的内存可见性保证。

[Java内存模型FAQ（二） 其他语言，像C++，也有内存模型吗？](http://ifeve.com/java-faq-otherlanguages/)大部分其他的语言，像C和C++，都没有被设计成直接支持多线程。这些语言对于发生在编译器和处理器平台架构的重排序行为的保护机制会严重的依赖于程序中所使用的线程库（例如pthreads），编译器，以及代码所运行的平台所提供的保障。也就是，语言上没有final、volatile 关键字这些，可以对编译器和处理器重排序 施加影响。

## 内存模型对代码和编程的影响

[Java内存模型FAQ（六）没有正确同步的含义是什么？](http://ifeve.com/jmm-faq-incorrectlysync/)
