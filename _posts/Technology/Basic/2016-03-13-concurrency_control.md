---

layout: post
title: 并发控制相关的硬件与内核支持
category: 技术
tags: Basic
keywords: AQS

---

## 简介

* TOC
{:toc}

## 为什么要做并发控制

线程同步出现的根本原因是访问公共资源需要多个操作，而这多个操作的执行过程不具备原子性，被任务调度器分开了，而其他线程会破坏共享资源，所以需要在临界区做线程的同步，这里我们先明确一个概念，就是临界区，临界区是指多个任务访问共享资源如内存或文件时候的指令，**临界区是指令并不是受访问的资源**。

1. 一个资源 同一时间只能被一个进程操作，在current进程操作完成之前，其它进程不可以操作。
2. 操作资源 的代码 可以认为是一个有状态服务，被调度到某个进程去执行。

[聊聊原子变量、锁、内存屏障那点事](http://0xffffff.org/2017/02/21/40-atomic-variable-mutex-and-memory-barrier/)

![](/public/upload/java/cpu_cache_memory.png)

cpu 也可以是一个主机的两个进程、两台机器，想对同一个数据进行操作。

**共享资源就像cpu，只能分时共享**。 cpu 如何执行分时共享呢？cpu 执行完每条指令后，会检查下有没有中断，若有则执行中断处理程序。对应到进程/线程调度上，就是时间片中断。进程每次访问共享资源之前，本质上也是先去查询一个变量，若允许则执行，不允许，则让出cpu。

## 硬件对并发控制的支持

提供的原子操作：

* 关中断指令、内存屏障指令、停止相关流水线指令
* 对于单核cpu，禁止抢占。
* 对于SMP，提供lock 指令。lock指令是一种前缀，它可与其他指令联合，用来维持总线的锁存信号直到与其联合的指令执行完为止。比如基于AT&T的汇编指令`LOCK_PREFIX xaddw %w0,%1`，xaddw 表示先交换源操作数和目的操作数，然后两个操作数求和，存入目的寄存器。为了防止这个过程被打断，加了LOCK_PREFIX的宏（修饰lock指令）。

这些指令，辅助一定的算法，就可以包装一个自旋锁、读写锁、顺序锁出来。os中，lock一词主要指自旋锁。注意，自旋锁时，线程从来没有停止运行过。
	
## 操作系统对并发控制的支持

POSIX表示可移植操作系统接口（Portable Operating System Interface of UNIX，缩写为 POSIX ），POSIX标准定义了操作系统应该为应用程序提供的接口标准。POSIX 定义了五种同步对象，互斥锁，条件变量，自旋锁，读写锁，信号量。有些时候，名词限制了我们对事物的认识。我们谈到锁、信号量这些，分析它们如何实现，或许走入了一个误区。[Locks, Mutexes, and Semaphores: Types of Synchronization Objects](https://www.justsoftwaresolutions.co.uk/threading/locks-mutexes-semaphores.html) 中认为锁是一个抽象概念，包括：

1. 竞态条件。只有一个进入，还是多个进入，还是读写进入，线程能否重复获取自己已占用的锁
2. 获取锁失败时的反应。提示失败、自旋还是直接阻塞，阻塞能不能被打断

按照这些不同，人们给它mutex、信号量等命名

除了提供工具给上层用之外，操作系统内部，本就存在对资源的并发访问。

通过对硬件指令的包装，os提供原子整数操作等系统调用。本质上，通过硬件指令，可以提供对一个变量的独占访问。

	int atomic_read(const atomic_t *v)
	// 将v设置为i
	int atomic_set(atomic_t *v,int id);

通过对硬件指令的封装，操作系统可以封装一个自旋锁出来。

1. 获取锁spin_lock_irq：用变量标记是否被其他线程占用，变量的独占访问，发现占用后自旋，结合关中断指令。
2. 释放锁spin_unlock_irq：独占的将变量标记为未访问状态

那么在获取锁与释放锁之间，可以实现一个临界区。

	spin_lock_irq();
	// 临界区
	spin_unlock_irq();


操作系统中，semaphore与自旋锁类似的概念，只有得到信号量的进程才能执行临界区的代码；不同的是获取不到信号量时，进程不会原地打转而是进入休眠等待状态（自己更改自己的状态位）

	struct semaphore{
		spinlock_t lock;
		unsigned int count;
		struct list_head wait_list;
	}
	// 获取信号量，会导致睡眠
	void down(struct semaphore *sem);
	// 获取信号量，会导致睡眠，但睡眠可被信号打断
	int down_interruptible(struct semaphore *sem);
	// 无论是否获得，都立即返回，返回值不同，不会导致睡眠
	int down_trylock(struct semaphore *sem);
	// 释放信号量
	void up(struct semaphore *sem))
	

通过自旋锁，os可以保证count 修改的原子性。线程尝试修改count的值，根据修改后count值，决定是否挂起当前进程，进而提供semaphore和mutex（类似于semaphore=1）等抽象。**也就是说，semaphore = 自旋锁 + 线程挂起/恢复。**

[大话Linux内核中锁机制之原子操作、自旋锁](http://blog.sina.com.cn/s/blog_6d7fa49b01014q7p.html)

[大话Linux内核中锁机制之内存屏障、读写自旋锁及顺序锁](http://blog.sina.com.cn/s/blog_6d7fa49b01014q86.html)

[大话Linux内核中锁机制之信号量、读写信号量](http://blog.sina.com.cn/s/blog_6d7fa49b01014q8y.html)

[大话Linux内核中锁机制之完成量、互斥量](http://blog.sina.com.cn/s/blog_6d7fa49b01014q9b.html)

### 内存屏障

[剖析Disruptor:为什么会这么快？(三)揭秘内存屏障](http://ifeve.com/disruptor-memory-barrier/)

什么是内存屏障？它是一个CPU指令

1. 插入一个内存屏障，相当于告诉CPU和编译器先于这个命令的必须先执行，后于这个命令的必须后执行
2. 强制更新一次不同CPU的缓存。例如，一个写屏障会把这个屏障前写入的数据刷新到缓存，这样任何试图读取该数据的线程将得到最新值

volatile，Java内存模型将在写操作后插入一个写屏障指令，在读操作前插入一个读屏障指令。

说白了，这是除cas 之外，又一个暴露在 java 层面的指令。

volatile 也是有成本的 [剖析Disruptor:为什么会这么快？（二）神奇的缓存行填充](http://ifeve.com/disruptor-cacheline-padding/)

|从CPU到|大约需要的 CPU 周期|大约需要的时间|
|---|---|---|
|主存||约60-80纳秒|
|QPI 总线传输(between sockets, not drawn)||约20ns|
|L3 cache|约40-45 cycles|约15ns|
|L2 cache|约10 cycles|约3ns|
|L1 cache|约3-4 cycles|约1ns|
| 寄存器 |1 cycle||

[聊聊并发（一）深入分析Volatile的实现原理](http://ifeve.com/volatile/)

		
##  linux 线程

[Understanding Linux Process States](https://access.redhat.com/sites/default/files/attachments/processstates_20120831.pdf)

|进程的基本状态|运行|就绪|阻塞|退出|
|---|---|---|---|---|
|Linux| TASK_RUNNING ||TASK_INTERRUPTIBLE、TASK_UNINTERRUPTIBLE|TASK_STOPPED/TASK_TRACED、TASK_DEAD/EXIT_ZOMBIE|
|java|| RUNNABLE | BLOCKED、WAITING、TIMED_WAITING|TERMINATED|

操作系统提供的手段：

||可以保护的内容|临界区描述|执行体竞争失败的后果|
|---|---|---|---|
|硬件|一个内存的值|某时间只可以执行一条指令|没什么后果，继续执行|
|os-自旋|变量/代码|多用于修改变量（毕竟lock指令太贵了）|自旋|
|os-信号量|变量/代码|代码段不可以同时执行|挂起（修改状态位）|

一个复杂项目由n行代码实现，一行代码由n多系统调用实现，一个系统调用由n多指令实现。那么从线程安全的角度看：锁住系统总线，某个时间只有一条指令执行 ==> **安全的修改一个变量 ==> 提供一个临界区**。通过向上封装，临界区的粒度不断地扩大。

反过来说，无锁的代码仅仅是不需要显式的Mutex来完成，但是存在数据竞争（Data Races）的情况下也会涉及到同步（Synchronization）的问题。从某种意义上来讲，所谓的无锁，仅仅只是颗粒度特别小的“锁”罢了，从代码层面上逐渐降低级别到CPU的指令级别而已，总会在某个层级上付出等待的代价，除非逻辑上彼此完全无关

## 一个博客系列的整理

[[并发系列-0] 引子](http://kexianda.info/page/2/)

并发的核心：

1. 一个是有序性，可见性，原子性. 从底层角度, 指令重排和内存屏障,CPU的内存模型的理解.
2. 另一个是线程的管理, 阻塞, 唤醒, 相关的线程队列管理(内核空间或用户空间)

并发相关的知识栈

1. 硬件只是、cpu cache等
2. 指令重排序、内存屏障，cpu 内存模型等
3. x86_64 相关的指令：lock、cas等
4. linux 进程/线程的实现，提供的快速同步/互斥机制 futex(fast userspace muTeXes)
5. 并发基础原语 pthread_mutex/pthread_cond 在 glibc 的实现。这是C++ 的实现基础
6. java 内存模型，java 并发基础原语 在 jvm hotspot 上的实现
7. java.util.concurrent

从中可以看到

1. 内存模型，有cpu 层次的，java 层次的
2. 并发原语，有cpu 层次的，linux 层次的，glibc/c++ 层次的，java 层次的。 首先cpu 层次根本没有 并发的概念，限定的是cpu 核心。glibc 限定的是pthread，java 限定的是Thread

**所有这一切，讲的都是共享内存模式的并发**。 所以 go 的协程让程序猿 少学多少东西。

