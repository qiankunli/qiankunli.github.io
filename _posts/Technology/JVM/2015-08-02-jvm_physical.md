---

layout: post
title: JVM内存与执行
category: 技术
tags: JVM
keywords: jvm

---

## 前言

jvm 作为 a model of a whole computer，便与os 有许多相似的地方，包括并不限于：

1. 针对os 编程的可执行文件，主要指其背后代表的文件格式、编译、链接、加载 等机制
2. 可执行文件 的如何被执行，主要指 指令系统及之上的 方法调用等
3. 指令执行依存 的内存模型

这三者是三个不同的部分，又相互关联，比如jvm基于栈的解释器与jvm 内存模型 相互依存。

## 进程内存布局

[Linux内核基础知识](http://qiankunli.github.io/2019/05/01/linux_kernel_basic.html)进程内存布局

![](/public/upload/linux/virtual_memory_space.jpg)

左右两侧均表示虚拟地址空间，左侧以描述内核空间为主，右侧以描述用户空间为主。右侧底部有一块区域“read from binary image on disk by execve(2)”，即来自可执行文件加载，jvm的方法区来自class文件加载，那么 方法区、堆、栈 便可以一一对上号了。

![](/public/upload/java/jvm_process.png)

## JVM内存区域新画法 

![](/public/upload/java/jvm_memory_layout.jpg)

我们以线程的视角重新画一下

![](/public/upload/java/jvm_memory.png)

一个cpu对应一个线程，一个线程一个栈，或者反过来说，一个栈对应一个线程，所有栈组成栈区。我们从cpu的根据pc指向的指令的一次执行开始：

1. cpu执行pc指向方法区的指令
2. 指令=操作码+操作数，jvm的指令执行是基于栈的，所以需要从栈帧中的“栈”区域获取操作数，栈的操作数从栈帧中的“局部变量表”和堆中的对象实例数据得到。
3. 当在一个方法中调用新的方法时，根据栈帧中的对象引用找到对象在堆中的实例数据，进而根据对象实例数据中的方法表部分找到方法在方法区中的地址。根据方法区中的数据在当前线程私有区域创建新的栈帧，切换PC，开始新的执行。

### PermGen ==> Metaspace

[Permgen vs Metaspace in Java](https://www.baeldung.com/java-permgen-metaspace)PermGen (Permanent Generation) is a special heap space separated from the main memory heap.

1. The JVM keeps track of loaded class metadata in the PermGen. 
2. all the static content: static methods,primitive variables,references to the static objects
3. bytecode,names,JIT information
4. before java7,the String Pool

**With its limited memory size, PermGen is involved in generating the famous OutOfMemoryError**. [What is a PermGen leak?](https://plumbr.io/blog/memory-leaks/what-is-a-permgen-leak)

Metaspace is a new memory space – starting from the Java 8 version; it has replaced the older PermGen memory space. The garbage collector now automatically triggers cleaning of the dead classes once the class metadata usage reaches its maximum metaspace size.with this improvement, JVM **reduces the chance** to get the OutOfMemory error.

## “可执行文件的” 执行

### 基于栈和基于寄存器

[Virtual Machine Showdown: Stack Versus Registers](https://www.usenix.org/legacy/events/vee05/full_papers/p153-yunhe.pdf)

[虚拟机随谈（一）：解释器，树遍历解释器，基于栈与基于寄存器，大杂烩](http://rednaxelafx.iteye.com/blog/492667)

1. 典型的RISC架构会要求除load和store以外，其它用于运算的指令的源与目标都要是寄存器。 `a += b; ` 对应 x86 就是` add a, b` 
2. 基于栈的解释器，`a += b; `对应jvm 就是`iconst_1  iconst_2  iadd  istore_0  ` 

![](/public/upload/java/jvm_os_1.gif)

大多数的处理器架构，都有实现**硬件栈**。有专门的栈指针寄存器，以及特定的硬件指令来完成 入栈/出栈 的操作。当然，硬件栈的主要作用是支持函数调用而不是所有的命令处理。

### 基于栈的解释器

![](/public/upload/jvm/run_java_code.png)

java是一种跨平台的编程语言，为了跨平台，jvm抽象出了一套内存模型和基于栈的解释器，进而创建一套在该模型基础上运行的字节码指令。（这也是本文不像其它书籍一样先从"class文件格式"讲起的原因）

1. 为了跨平台，不能假定平台特性，因此抽象出一个新的层次来屏蔽平台特性，因此推出了基于栈的解释器，与以往基于寄存器的cpu执行有所区别。

2. `字节码指令 = 操作码 + 操作数`,（操作数可以是立即数，可以存在栈中，也可以是指向堆的引用（引用存在栈中）） `传统的指令 = 操作码 + 操作数`,（操作数据可以是立即数，可以存在寄存器中，也可以是指向内存的引用）此处jvm的栈，说的是栈帧，跟传统的栈还是有不同的，**粗略的讲，我们可以说在jvm体系中，用栈替代了原来寄存器的功能。**

    这句话的不准确在于，对于传统cpu执行，线程之间共用的寄存器，只不过在线程切换时，借助了pcb（进程控制块或线程控制块，存储在线程数据所在内存页中），pcb保存了现场环境，比如寄存器数据。轮到某个线程执行时，恢复现场环境，为寄存器赋上pcb对应的值，cpu按照pc指向的指令的执行。
    
    而在jvm体系中，每个线程的栈空间是私有的，栈一直在内存中（无论其对应的线程是否正在执行），轮到某个线程执行时，线程对应的栈（确切的说是栈顶的栈帧）成为“当前栈”（无需重新初始化），执行pc指向的方法区中的指令。
    
3. 类似的编译优化策略

    同一段代码，编译器可以给出不同的字节码（最后执行效果是一样的），还可以在此基础上进行优化。比如，对于传统os，将内存中的变量缓存到寄存器。对于jvm，将堆中对象的某个实例属性缓存到线程对应的栈。而c语言和java语言是通过共享内存，而不是共享寄存器或线程的私有栈来进行线程“交流”的，因此就会带来线程安全问题。因此，c语言和java语言都有volatile关键字。（虽然这不能完全解决线程安全问题）

## 线程的状态

[Understanding Linux Process States](https://access.redhat.com/sites/default/files/attachments/processstates_20120831.pdf)

|进程的基本状态|运行|就绪|阻塞|退出|
|---|---|---|---|---|
|Linux| TASK_RUNNING ||TASK_INTERRUPTIBLE、TASK_UNINTERRUPTIBLE|TASK_STOPPED/TASK_TRACED、TASK_DEAD/EXIT_ZOMBIE|
|java|| RUNNABLE | BLOCKED、WAITING、TIMED_WAITING|TERMINATED|

在 POSIX 标准中（POSIX标准定义了操作系统应该为应用程序提供的接口标准），thread_block 接受一个参数 stat ，这个参数也有三种类型，TASK_BLOCKED， TASK_WAITING， TASK_HANGING，而调度器只会对线程状态为 READY 的线程执行调度，另外一点是线程的阻塞是线程自己操作的，相当于是线程主动让出 CPU 时间片，所以等线程被唤醒后，他的剩余时间片不会变，该线程只能在剩下的时间片运行，如果该时间片到期后线程还没结束，该线程状态会由 RUNNING 转换为 READY ，等待调度器的下一次调度。

![](/public/upload/java/thread_status.jpg)

[Java和操作系统交互细节](https://mp.weixin.qq.com/s/fmS7FtVyd7KReebKzxzKvQ)对进程而言，就三种状态，就绪，运行，阻塞，而在 JVM 中，阻塞有四种类型，我们可以通过 jstack 生成 dump 文件查看线程的状态。

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

## 重排序

### 为什么会出现重排序

[CPU 的指令执行](http://qiankunli.github.io/2018/01/07/hardware_software.html)

### 重排序的影响

主要体现在两个方面，详见[Java内存访问重排序的研究](http://tech.meituan.com/java-memory-reordering.html)

1. 对代码执行的影响

	常见的是，一段未经保护的代码，因为多线程的影响可能会乱序输出。**少见的是，重排序也会导致乱序输出。**

2. 对编译器、runtime的影响，这体现在两个方面：

	1. 运行期的重排序是完全不可控的，jvm经过封装，要保证某些场景不可重排序（比如数据依赖场景下）。提炼成理论就是：happens-before规则（参见《Java并发编程实践》章节16.1），Happens-before的前后两个操作不会被重排序且后者对前者的内存可见。
	2. 提供一些关键字（主要是加锁、解锁），也就是允许用户介入某段代码是否可以重排序。这也是**"which are intended to help the programmer describe a program’s concurrency requirements to the compiler"** 的部分含义所在。

[Java内存访问重排序的研究](http://tech.meituan.com/java-memory-reordering.html)文中提到，内存可见性问题也可以视为重排序的一种。比如，在时刻a，cpu将数据写入到memory bank，在时刻b，同步到内存。cpu认为指令在时刻a执行完毕，我们呢，则认为代码在时刻b执行完毕。






