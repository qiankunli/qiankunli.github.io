---

layout: post
title: java内存模型
category: 技术
tags: JVM
keywords: JAVA memory model

---

## 前言

* TOC
{:toc}

[Java Memory Model](http://tutorials.jenkov.com/java-concurrency/java-memory-model.html)  它是一系列文章 [Java Concurrency](http://tutorials.jenkov.com/java-concurrency/) 中的一篇文章。

[Java和操作系统交互细节](https://mp.weixin.qq.com/s/fmS7FtVyd7KReebKzxzKvQ)

## 为什么会有内存模型一说

### 硬件层面的内存模型

[Java内存模型深入分析](https://mp.weixin.qq.com/s/0H9yfiYvWGQByjFT-fj-ww)曾经，计算机的世界远没有现在复杂，那时候的cpu只有单核，我们写的程序也只会在单核上按代码顺序依次执行，根本不用考虑太多。后来，随着技术的发展，cpu的执行速度和内存的读写速度差异越来越大，人们很快发现，如果还是按照代码顺序依次执行的话，cpu会花费大量时间来等待内存操作的完成，这造成了cpu的巨大浪费。为了弥补cpu和内存之间的速度差异，计算机世界的工程师们在cpu和内存之间引入了缓存，虽然该方法极大的缓解了这一问题，但追求极致的工程师们觉得这还不够，他们又想到了一个点子，就是**通过合理调整内存的读写顺序来进一步缓解这个问题**。
1. 比如，在编译时，我们可以把不必要的内存读写去掉，把相关连的内存读写尽量放到一起，充分利用缓存。
2. 比如，在运行时，我们可以对内存提前读，或延迟写，这样使cpu不用总等待内存操作的完成，充分利用cpu资源，避免计算能力的浪费。

这一想法的实施带来了性能的巨大提升，但同时，它也带来了一个问题，就是内存读写的乱序，比如原本代码中是先写后读，但在实际执行时却是先读后写，怎么办呢？为了避免内存乱序给上层开发带来困扰，这些工程师们又想到了可以**通过分析代码中的语义，把有依赖关系，有顺序要求的代码保持原有顺序，把剩余的没有依赖关系的代码再进行性能优化，乱序执行**，通过这样的方式，就可以屏蔽底层的乱序行为，使代码的执行看起来还是和其编写顺序一样，完美。

多核时代的到来虽然重启了计算机世界新一轮的发展，但也带来了一个非常严峻的问题，那就是多核时代如何承接单核时代的历史馈赠。单核运行不可见的乱序，在多核情况下都可见了，且此种乱序已经严重影响到了多核代码的正确编写。**默认乱序执行，在关键节点保证有序**，这种方式不仅使单核时代的各种乱序优化依然有效，也使多核情况下的乱序行为有了一定的规范。基于此，各种硬件平台提供了自己的方式给上层开发，约定好只要按我给出的方式编写代码，即使是在多核情况下，该保证有序的地方也一定会保证有序。这套在多核情况下，依然可以让开发者指定哪些代码保证有序执行的规则，就叫做内存模型。

内存模型的英文是memory model，或者更精确的来说是memory consistency model，它其实就是一套方法或规则，用于描述如何在多核乱序的情况下，通过一定的方式，来保证指定代码的有序执行。它是介于硬件和软件之间，以一种协议的形式存在的。对硬件来说，它描述的是硬件对外的行为规范，对软件来说，它描述的是编写多线程代码的一套规则。这就衍生出了一个问题，就是不同硬件上的内存模型差异很大，完全不兼容。比如应用于桌面和服务器领域的x86平台用的是x86 tso内存模型。比如应用于手机和平板等移动设备领域的arm平台用的是weakly-ordered内存模型。比如最近几年大火的riscv平台用的是risc-v weak memory ordering内存模型。

### 语言层面的内存模型

由于Java的目标是write once, run anywhere，所以它不仅创造性的提出了字节码中间层，让字节码运行在虚拟机上，而不是直接运行在物理硬件上，它还在语言层面内置了对多线程的跨平台支持，也为此提出了Java语言的内存模型，这样，当我们用Java写多线程项目时，只要按照Java的内存模型规范来编写代码，Java虚拟机就能保证我们的代码在所有平台上都是正确执行的。在语言层面支持多线程在现在看来不算什么，但在那个年代，这也算是一项大胆的创举了，**它也成为了首个主流编程语言中，内置支持多线程编码的语言**。

JMM属于语言级的内存模型，它确保在不同的编译器和不同的处理器平台之上，通过禁止特定类型的编译器重排序和处理器重排序，为程序员提供一致的内存可见性保证。

[Java内存模型FAQ（二） 其他语言，像C++，也有内存模型吗？](http://ifeve.com/java-faq-otherlanguages/)大部分其他的语言，像C和C++，都没有被设计成直接支持多线程。这些语言对于发生在编译器和处理器平台架构的重排序行为的保护机制会严重的依赖于程序中所使用的线程库（例如pthreads），编译器，以及代码所运行的平台所提供的保障。也就是，语言上没有final、volatile 关键字这些，可以对编译器和处理器重排序 施加影响。

[Java内存模型FAQ（六）没有正确同步的含义是什么？](http://ifeve.com/jmm-faq-incorrectlysync/)

## java memory model 与 harware memory Architecture

![](/public/upload/java/jvm_memory_model_2.png)

![](/public/upload/java/jvm_memory_model_3.png)

![](/public/upload/java/jvm_memory_model_4.png)

这几张图从粗到细，逐步引出了jvm 内存组成，栈的组成，堆的组成，栈和堆内数据的关系。逐步介绍了 thread stack、call stack（方法栈、栈帧）等概念

![](/public/upload/java/jvm_memory_model_5.png)

cpu ==> 寄存器 ==> cpu cache ==> main memory，cpu cache 由cache line 组成，cache line 是 与 main memory 沟通的基本单位，就像mysql innodb 读取 一行数据时 实际上不是 只读取一行，而是直接读取一页到内存一样。

![](/public/upload/java/jvm_memory_model_1.png)

The hardware memory architecture does not distinguish between thread stacks and heap. On the hardware, both the thread stack and the heap are located in main memory. Parts of the thread stacks and heap may sometimes be present in CPU caches and in internal CPU registers. jvm 和 物理机 对“内存/存储” 有不同的划分，jvm 中没有cpu、cpu core 等抽象存在，也没有寄存器、cpu cache、main memory 的区分，因此 stack、heap 数据 可能分布在 寄存器、cpu cache、main memory 等位置。

When objects and variables can be stored in various different memory areas in the computer, certain problems may occur. The two main problems are:

1. Visibility of thread updates (writes) to shared variables. 可以用volatile 关键字解决
2. Race conditions when reading, checking and writing shared variables. 让两个线程 不要同时执行同一段代码，可以用synchronized block 解决，本质就是将竞争转移（从竞争同一个变量 到去竞争 同一个锁）。或者使用cas 保证竞争是原子的。

![](/public/upload/java/jvm_memory_model_6.png)

就着上图 去理解《java并发编程实战》中的有序性、原子性及可见性 ，会有感觉很多。

可以脑补一下 基于jvm 内存模型，多线程执行 访问 对象的局部变量 的图，直接的观感是jvm 是从内存（heap）中直接拿数据的，会有原子性问题，但没有可见性问题。但实际上，你根本搞不清楚，从heap 中拿到的对象变量的值 是从寄存器、cpu cache、main memory 哪里拿到的，写入问题类似。jvm 提供volatile 等微操工具，介入两种内存模型的映射过程，来确保预期与实际一致，从这个角度看，jvm 并没有完全屏蔽硬件架构的特性（当然，也是为了提高性能考虑），不过确实做到了屏蔽硬件架构的差异性。


## java 内存模型与并发读写控制

[Java内存模型深入分析](https://mp.weixin.qq.com/s/0H9yfiYvWGQByjFT-fj-ww)如果程序中存在对同一变量的多个访问操作，且至少有一个是写操作，则这些访问操作被称为是conflicting操作，如果这些conflicting操作没有被happens-before规则约束，则这些操作被称为data race，有data race的程序就不是correctly synchronized，运行时也就无法保证sequentially consistent特性，没有data race的程序就是correctly synchronized，运行时可保证sequentially consistent特性。

happens-before规则由两部分组成，一部分是program order，即单线程中代码的编写顺序，另一部分是synchronizes-with，即多线程中的各种同步原语。也就是说，在单线程中，代码编写的前后顺序之间有happens-before关系，在多线程中，有synchronizes-with关联的代码之间也有happens-before关系。
1. program order，即单线程中代码的字面顺序
2. synchronizes-with，即各种同步操作，比如synchronized关键字，volatile关键字，线程的启动关闭操作等。定义多线程之间操作的顺序




极客时间《深入拆解Java虚拟机》

1. happens-before 关系是用来描述两个操作的内存可见性的。如果操作 X happens-before 操作 Y，那么 X 的结果对于 Y 可见。
2. **规定的happens-before 关系**：Java 内存模型定义了六七种线程间的 happens-before 关系。比如 线程的启动操作（即 Thread.starts()） happens-before 该线程的第一个操作。
3. **可以手动控制的happens-before 关系**：Java 内存模型通过定义了一系列的 happens-before 操作（包括锁、volatile 字段、final 字段与安全发布），让应用程序开发者能够轻易地表达不同线程的操作之间的内存可见性。
2. Java 内存模型是通过内存屏障来禁止重排序的。对于即时编译器来说，内存屏障将限制它所能做的重排序优化。对于处理器来说，内存屏障会导致缓存的刷新操作。

**法无禁止即允许，在遵守happens-before规则的前提下，即时编译器以及底层体系架构能够调整内存访问操作（也就是重排序），以达到性能优化的效果。**

[《mysql技术内幕》笔记2](http://qiankunli.github.io/2017/11/12/inside_mysql2.html) 提到 数据库一共会发生11种异常现象，脏读、不可重复读、幻读只是其中三种，数据库提出隔离性的概念，用这三种异常现象的出现情况来描述并发读写的安全程度。java 有可见性的概念，提供关键字（而不是配置，比如隔离级别是mysql的一种配置）给用户来描述期望的可见性。

||为什么提出|实现原理|
|---|---|---|
|隔离性|实现mysql 需要大量彼此关联的数据结构，并发读写|锁|
|java内存模型|java内存模型与硬件内存模型的映射，并发读写 + 编译器、cpu重排序|happens-before 关系 + 内存屏障|


## 其它材料

[JSR 133 (Java Memory Model) FAQ](https://www.cs.umd.edu/~pugh/java/memoryModel/jsr-133-faq.html)及其译文[Java内存模型FAQ（一） 什么是内存模型](http://ifeve.com/memory-model/)，[深入理解Java内存模型（一）——基础](http://www.infoq.com/cn/articles/java-memory-model-1)系列文章


Java includes several language constructs, including volatile, final, and synchronized, **which are intended to help the programmer describe a program’s concurrency requirements to the compiler.** The Java Memory Model defines the behavior of volatile and synchronized, and, more importantly, ensures that a correctly synchronized Java program runs correctly on all processor architectures.


volatile的写操作是发生在后续的读操作之前：volatile保证的有序性其实是在跨线程之间建立了一条happens-before规则，即volatile的写操作发生在后续的volatile读操作之前，它只建立了这一条有序关系。所以说volatile保证的有序是帮助串联起跨线程之间操作的有序。在x86平台上，volatile的读操作没有任何消耗，volatile的写操作使用的是 lock 汇编指令。








