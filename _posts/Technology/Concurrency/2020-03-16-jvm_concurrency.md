---

layout: post
title: jvm 线程实现
category: 技术
tags: Concurrency
keywords: jvm concurrency

---

## 简介

![](/public/upload/jvm/jvm_thread.png)

## jvm层实现

![](/public/upload/jvm/hospot_thread_object.png)

[从Java到C++,以JVM的角度看Java线程的创建与运行](https://www.jianshu.com/p/3ce1b5e5a55e)

1. JavaThread: JVM中C++定义的类，一个JavaThread的instance代表了在JVM中的java.lang.Thread的instance, 它维护了线程的状态，并且维护一个指针指向java.lang.Thread创建的对象(oop)。它同时还维护了一个指针指向对应的OSThread，来获取底层操作系统创建的osthread的状态
2. OSThread: JVM中C++定义的类，代表了JVM中对底层操作系统的osthread的抽象，它维护着实际操作系统创建的线程句柄handle，可以获取底层osthread的状态
3. VMThread: JVM中C++定义的类，这个类和用户创建的线程无关，是JVM本身用来进行虚拟机操作的线程，比如GC

![](/public/upload/jvm/hospot_thread_sequence.png)

[聊聊 Java 并发——基石篇（上）](https://www.infoq.cn/article/Nwq2WyKWevl0mGk_g96C)在创建一个 Thread 对象的时候，除了一些初始化设置之外就没有什么实质性的操作，真正的工作其实是在 start 方法调用中产生的。

## 线程的状态

[Understanding Linux Process States](https://access.redhat.com/sites/default/files/attachments/processstates_20120831.pdf)

|进程的基本状态|Linux|Java|
|---|---|---|
|运行|TASK_RUNNING||
|就绪||RUNNABLE|
|阻塞|TASK_INTERRUPTIBLE<br>TASK_UNINTERRUPTIBLE|BLOCKED<br>WAITING<br>TIMED_WAITING|
|退出|TASK_STOPPED/TASK_TRACED<br>TASK_DEAD/EXIT_ZOMBIE|TERMINATED|

在 POSIX 标准中（POSIX标准定义了操作系统应该为应用程序提供的接口标准），thread_block 接受一个参数 stat ，这个参数也有三种类型，TASK_BLOCKED， TASK_WAITING， TASK_HANGING，而调度器只会对线程状态为 READY 的线程执行调度，另外一点是线程的阻塞是线程自己操作的，相当于是线程主动让出 CPU 时间片，所以等线程被唤醒后，他的剩余时间片不会变，该线程只能在剩下的时间片运行，如果该时间片到期后线程还没结束，该线程状态会由 RUNNING 转换为 READY ，等待调度器的下一次调度。

![](/public/upload/java/thread_status.jpg)

[Java和操作系统交互细节](https://mp.weixin.qq.com/s/fmS7FtVyd7KReebKzxzKvQ)对进程而言，就三种状态，就绪，运行，阻塞，而在 JVM 中，阻塞有四种类型，**我们可以通过 jstack 生成 dump 文件查看线程的状态**。

1. BLOCKED （on object monitor)  通过 synchronized(obj) 同步块获取锁的时候，等待其他线程释放对象锁，dump 文件会显示 waiting to lock <0x00000000e1c9f108>
2. TIMED WAITING (on object monitor) 和 WAITING (on object monitor) 在获取锁后，调用了 object.wait() 等待其他线程调用 object.notify()，两者区别是是否带超时时间
3. TIMED WAITING (sleeping) 程序调用了 thread.sleep()，这里如果 sleep(0) 不会进入阻塞状态，会直接从运行转换为就绪
4. TIMED WAITING (parking) 和 WAITING (parking) 程序调用了 Unsafe.park()，线程被挂起，等待某个条件发生，waiting on condition

从linux内核来看， BLOCKED、WAITING、TIMED_WAITING都是等待状态。做这样的区分，是jvm出于管理的需要（两个原因的线程放两个队列里管理，如果线程运行出了synchronized这段代码，jvm只需要去blocked队列放一个线程出来。而某人调用了notify()，jvm只需要去waitting队列里取个出来。），本质上是：who when how唤醒线程。

[Java线程中wait状态和block状态的区别? - 赵老师的回答 - 知乎](
https://www.zhihu.com/question/27654579/answer/128050125)

|从上到下|常规java code|synchronized java code|volatile java code|
|---|---|---|---|
|编译|编译器加点私货|monitor enter/exist|除了其变量定义的时候有一个Volatile外，之后的字节码跟有无Volatile完全一样|
||class 字节码 |扩充后的class 字节码 ||
|运行|jvm加点私货|锁升级：自旋/偏向锁/轻量级锁/重量级锁 ||
||机器码|扩充后的机器码| 加入了lock指令，查询IA32手册，它的作用是使得本CPU的Cache写入了内存，该写入动作也会引起别的CPU invalidate其Cache |
|用户态|||
||系统调用|mutex系统调用|
|内核态|||
|||可能用到了 semaphore struct|
|||线程加入等待队列 + 修改自己的状态 + 触发调度|

## 设置多少线程数

[程序设计的5个底层逻辑，决定你能走多快](https://mp.weixin.qq.com/s/ar3BRRjAgGXShZ0tuFdubQ)

关于应用程序中设置多少线程数合适的问题，我们一般的做法是设置 CPU 最大核心数 * 2 ，我们编码的时候可能不确定运行在什么样的硬件环境中，可以通过 Runtime.getRuntime（).availableProcessors() 获取 CPU 核心。

但是具体设置多少线程数，主要和线程内运行的任务中的阻塞时间有关系，如果任务中全部是计算密集型，那么只需要设置 CPU 核心数的线程就可以达到 CPU 利用率最高，如果设置的太大，反而因为线程上下文切换影响性能，如果任务中有阻塞操作，而在阻塞的时间就可以让 CPU 去执行其他线程里的任务，我们可以通过 线程数量=内核数量 / （1 - 阻塞率）这个公式去计算最合适的线程数，阻塞率我们可以通过计算任务总的执行时间和阻塞的时间获得。

目前微服务架构下有大量的RPC调用，所以利用多线程可以大大提高执行效率，**我们可以借助分布式链路监控来统计RPC调用所消耗的时间，而这部分时间就是任务中阻塞的时间**，当然为了做到极致的效率最大，我们需要设置不同的值然后进行测试。

