---

layout: post
title: AQS2——一个博客系列的整理
category: 技术
tags: Java
keywords: AQS

---

## 简介

建议结合 [AQS1——并发相关的硬件与内核支持](http://qiankunli.github.io/2016/03/13/aqs.html) 一起看

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


[[并发系列-1] 由wait/sleep例子开始](http://kexianda.info/2017/07/10/%E5%B9%B6%E5%8F%91%E7%B3%BB%E5%88%97-1-%E7%94%B1wait-sleep%E4%BE%8B%E5%AD%90%E5%BC%80%E5%A7%8B/#more)

1. [Why wait(), notify(), notifyAll() must be called inside a synchronized method/block?](http://www.xyzws.com/javafaq/why-wait-notify-notifyall-must-be-called-inside-a-synchronized-method-block/127)
2. 粗略的说，monitor = lock + entry list(新进来的线程发现锁被占用，进来排队) + wait list（线程发现自己缺点啥东西，主动掉wait，进入该队列）
2. object.wait = 

	1. 将线程 放入wait list 用户空间
	2. 释放锁 	用户空间
	3. 前两步代码中 涉及到 OrderAccess 等，估计是限定指令重排序的
	4. pthread_cond_wait 进入内核空间
	6. linux futex 系统调用 一个新的同步机制，了解下。
	7. switch_to 线程A中执行switch_to，则linux 保存上下文，执行其它线程。相当于让出cpu 的效果
	
		
[[并发系列-2] 为什么要有Condition Variable?](http://kexianda.info/2017/07/15/%E5%B9%B6%E5%8F%91%E7%B3%BB%E5%88%97-2-%E4%B8%BA%E4%BB%80%E4%B9%88Condition-Variable/)

1. 为什么有了mutex，还需要condition？ mutex 是保护资源的，condition 是限定 线程执行顺序的
2. 为什么condition 要跟锁一起用？但“判断条件，加入休息队列”两个操作之间，consumer 前脚刚判断没啥消费，还没加入呢，producer生产了一个产品，notify了一下，这就尴尬了（学名就是：条件变量的 判断过程 不能有data racing（数据竞争））。


[[并发系列-3] 从AQS到futex(一): AQS和LockSupport](http://kexianda.info/2017/08/13/%E5%B9%B6%E5%8F%91%E7%B3%BB%E5%88%97-3-%E4%BB%8EAQS%E5%88%B0futex%E4%B9%8B%E4%B8%80-AQS%E5%92%8CLockSupport/)

aqs 包括两个队列，同步队列和wait队列，这点和synchronized实现基本是一致的。

AQS/同步器的基本构成，第一和第三jvm/java层处理，第二个委托os层处理

1. 同步状态(原子性)；
2. 线程的阻塞与解除阻塞(block/unblock)；
3. 队列的管理；

[[并发系列-4] 从AQS到futex(二): HotSpot的JavaThread和Parker](http://kexianda.info/2017/08/16/%E5%B9%B6%E5%8F%91%E7%B3%BB%E5%88%97-4-%E4%BB%8EAQS%E5%88%B0futex-%E4%BA%8C-JVM%E7%9A%84Thread%E5%92%8CParker/)

JDK中的LockSupport只是用来block(park,阻塞)/unblock(unpark,唤醒)线程, 线程队列的管理是JDK的AQS处理的. 从Java层来看, 只需要(glibc或os层)mutex/cond提供个操作系统的block/unblock API即可.

[[并发系列-5] 从AQS到futex(三): glibc(NPTL)的mutex/cond实现](http://kexianda.info/2017/08/17/%E5%B9%B6%E5%8F%91%E7%B3%BB%E5%88%97-5-%E4%BB%8EAQS%E5%88%B0futex%E4%B8%89-glibc-NPTL-%E7%9A%84mutex-cond%E5%AE%9E%E7%8E%B0/)

[[并发系列-6] 从AQS到futex(四): Futex/Critical Section介绍](http://kexianda.info/2017/08/19/%E5%B9%B6%E5%8F%91%E7%B3%BB%E5%88%97-6-%E4%BB%8EAQS%E5%88%B0futex-%E5%9B%9B-Futex-Critical-Section%E4%BB%8B%E7%BB%8D/)

[[并发系列-7] CAS的大致成本](http://kexianda.info/2017/11/12/%E5%B9%B6%E5%8F%91%E7%B3%BB%E5%88%97-7-CAS%E7%9A%84%E5%A4%A7%E8%87%B4%E6%88%90%E6%9C%AC/)


aqs 作者关于aqs的论文[The java.util.concurrent Synchronizer Framework](http://gee.cs.oswego.edu/dl/papers/aqs.pdf) 中文版 [The j.u.c Synchronizer Framework翻译(一)背景与需求](http://ifeve.com/aqs-1/)

文章从AQS到HotSpot, 再到glibc, 最后到内核的futex. 纵向一条线深入下来, 粗略地了解下各个层次的实现。小结起来，有以下几个点：

1. 同步器的基本概念及基本组成
2. 各平台(glibc、java)为效率考虑，并未直接使用内核提供的同步机制。都试图将同步器的一部分放在自己语言层面，另一部分交给内核。java 既不用glibc的，也不用内核的。

	1. 内核陷入成本较高，cas的成本都有人嫌高
	2. 很多时候，竞争状态不是很剧烈，一些简单的check 操作就省去了block 的必要
	3. 内核提供的接口功能不够丰富，比如block 时间、可中断等等
	
3. aqs 维护一个同步状态值，线程的block 依靠glibc/内核，block 操作本质上是靠内核的mutex等实现，但此时，内核mutex 状态值跟 aqs的状态值就不是一个意思了。内核 mutex 的状态值单纯的标记了是否被占用。同步相关的 waiting list 和 mutet 的队列 含义也不同。

类似的底层已实现的能力不用，非要亲自实现一下的情况：linux 内核中，semaphore = 自旋锁 + 线程挂起/恢复，自旋锁通过自旋 也有 线程挂起的效果，但semaphore 只用自旋锁保护 count 变量设置的安全性，挂起的效果自己实现。为何呀？也是嫌spinlock_t 代价太高。

	struct semaphore{
		spinlock_t lock;
		unsigned int count;
		struct list_head wait_list;
	}
	
## futex

[futex内核实现源码分析（1）](https://www.jianshu.com/p/8f4b8dd37cbf)

	#include <linux/futex.h>
    #include <sys/time.h>
    int futex (int *uaddr, int op, int val, const struct timespec *timeout,int *uaddr2, int val3);
    #define __NR_futex              240	// 系统调用号

1. uaddr，用户态下共享内存的地址，里面存放的是一个对齐的整型计数器 参见futex 原理
2. op：存放着操作类型，如最基本的两种 FUTEX_WAIT和FUTEX_WAKE。
3. 具体含义由操作类型op决定

	* FUTEX_WAKE 时，val 表示唤醒 val 个 等待在uaddr 上的进程
	* FUTEX_WAIT，原子性的检查 uaddr 计数器的值是否为 val

		* 如果是，则让进程休眠，直到FUTEX_WAKE 或 timeout


## aqs 为什么自己 也要维护一个队列

虽说aqs 只需要 glibc/linux 提供 线程的block/unblock 的能力即可。但实际上，linux block 线程时也顺带 将其加入了 内核级的等待队列中，为什么？

1. glibc 层面上的mutex/condition 也提供队列管理 等待线程，aqs 不直接用？
2. 内核 也将线程加入到等待队列了？aqs 何必再加入？

原因

1. jvm 层面上java 线程有一个数据结构，glibc 上线程也有一个数据结构，linux 内核 则是 task_struct，不同层级维护的信息不同。再则，jvm 层级的 等待队列，jvm 可以管理和访问，易于做一些事儿
2. aqs 使用的是 clh 队列，有一些独到之处，还有待分析。

## 小结

锁和同步器的几个关系

1. 同步器 通常 = 锁 + 状态量 + 队列
2. 一元 同步器  可以当成 锁使用，或者说锁的实现 是一元同步器，比如java 的lock
3. 同步器、锁 在java、glibc、linux、cpu 等层次 都有实现。并且，都是核心的阻塞线程的功能 由下一层实现，队列管理由自己 负责。每个层次有新的抽象

	* cpu 有锁内存总线的等指令， 但cpu 不知道 “线程” 等概念
	* linux 提供 锁、同步器等系统调用，但linux 不管 指令重排序
	* glibc 和 jvm 也在各自的层次上 做了线程的抽象
4. 撇开锁、同步这类并发场景，java/glibc/linux 提供启动、中止、中断一个线程的能力，虽然底层还是用到了 锁或同步器