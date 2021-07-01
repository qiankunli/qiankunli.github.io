---

layout: post
title: 为什么要有堆栈
category: 技术
tags: Basic
keywords: 堆栈

---

## 简介

* TOC
{:toc}

电影胶卷的一个胶片叫做一帧，电影的播放就是代码执行一个个函数一样，一帧一帧过去。

## 应用程序的内存划分

操作系统把磁盘上的可执行文件加载到内存运行之前，会做很多工作，其中很重要的一件事情就是把可执行文件中的代码，数据放在内存中合适的位置，并分配和初始化程序运行过程中所必须的堆栈，所有准备工作完成后操作系统才会调度程序起来运行。来看一下程序运行时在内存（虚拟地址空间）中的布局图：

![](/public/upload/basic/os_load_paragram.png)


1. 代码区，包括能被CPU执行的机器代码（指令）和只读数据比如字符串常量，程序一旦加载完成代码区的大小就不会再变化了。
2. 数据区，包括程序的全局变量和静态变量（c语言有静态变量，而go没有），与代码区一样，程序加载完毕后数据区的大小也不会发生改变。
3. 堆，程序运行时动态分配的内存都位于堆中，这部分内存由内存分配器负责管理。该区域的大小会随着程序的运行而变化。传统的c/c++代码就必须小心处理内存的分配和释放，而在go语言中，有垃圾回收器帮助我们
4. 函数调用栈，**简称栈**。不管是函数的执行还是函数调用，栈都起着非常重要的作用
    1. 保存函数的局部变量；
    2. 向被调用函数传递参数；
    3. 返回函数的返回值；
    4. 保存函数的返回地址。返回地址是指从被调用函数返回后调用者应该继续执行的指令地址

每个函数在执行过程中都需要使用一块栈内存用来保存上述这些值，我们称这块栈内存为某函数的栈帧(stack frame)。当发生函数调用时，因为调用者还没有执行完，其栈内存中保存的数据还有用，所以被调用函数不能覆盖调用者的栈帧，只能把被调用函数的栈帧“push”到栈上，等被调函数执行完成后再把其栈帧从栈上“pop”出去，这样，栈的大小就会随函数调用层级的增加而生长，随函数的返回而缩小。**栈的生长和收缩都是自动的，由编译器插入的代码自动完成**，因此位于栈内存中的函数局部变量所使用的内存随函数的调用而分配，随函数的返回而自动释放，所以程序员不管是使用有垃圾回收还是没有垃圾回收的高级编程语言都不需要自己释放局部变量所使用的内存，这一点与堆上分配的内存截然不同。

AMD64 Linux平台下，栈是从高地址向低地址方向生长的，为什么栈会采用这种看起来比较反常的生长方向呢，具体原因无从考究，不过根据前面那张进程的内存布局图可以猜测，当初这么设计的计算机科学家是希望尽量利用内存地址空间，才采用了堆和栈**相向生长**的方式，因为程序运行之前无法确定堆和栈谁会消耗更多的内存，如果栈也跟堆一样向高地址方向生长的话，栈底的位置不好确定，离堆太近则堆内存可能不够用，离堆太远栈又可能不够用，于是乎就采用了现在这种相向生长的方式。


## 为什么需要栈

《揭秘Java虚拟机》
1. 函数内部会定义局部变量，变量要占内存，由于位于同一函数中，很自然的想法是将其内存空间放在一起，有利于整体申请和释放。所以栈帧首先是一个容器。
2. 一个函数一帧，多个函数多个帧。如何组织呢？ PS： 是散列不行，所以选择了线性，队列不合适，选择了栈。
    1. 散列。一个函数的栈帧随意在内存中分配，栈帧地址与函数名建立关联关系 ==> 调用方与被调用方要保存对方的堆栈空间地址。
    2. 线性。又分为队列和栈两种结构。函数调用符合栈的特性，就用栈了。
3. 操作系统明显的将程序空间区分为堆和栈，堆栈的增长方向不同（相向），为何呢？如果堆栈都从地址0开始分配，则栈空间和堆空间相互交错，则栈实际内存空间的连续性就被破坏了（堆没有这个要求）
4. 系统每次保存栈顶和栈底地址比较麻烦， 因此硬件上使用SP 和BP 寄存器来加速这个过程。`sub $32,%sp` 开辟栈空间，`add $32, $sp`释放栈空间。
4. 在操作系统环境下，操作系统会为一个进程/线程单独划分一个栈空间，还是会让所有进程/线程共同使用同一个栈空间呢？因为栈帧必须是连续的，所以只能是前者。

[Memory Management/Stacks and Heaps](https://en.m.wikibooks.org/wiki/Memory_Management/Stacks_and_Heaps)

1. The system stack,  are used most often to provide **frames**. A frame is a way to localize information about subroutines（可以理解为函数）. 
2. In general, a subroutine must have in its frame the return address (where to jump back to when the subroutine completes), the function's input parameters. When a subroutine is called, all this information is pushed onto the stack in a specific order. When the function returns, all these values on the stack are popped back off, reclaimed to the system for later use with a different function call. In addition, subroutines can also use the stack as storage for local variables.

[Demystifying memory management in modern programming languages](https://deepu.tech/memory-management-in-programming/)The stack is used for static memory allocation and as the name suggests it is a last in first out(LIFO) stack. the process of storing and retrieving data from the stack is **very fast** as **there is no lookup required**, you just store and retrieve data from the **topmost block** on it.

根据上文可以推断：**为什么需要栈？为了支持函数**。OS设计体现了对进程、线程的支持，直接提供系统调用创建进程、线程，但就进程/线程内部来说，os 认为代码段是一个指令序列，最多jump几下，指令操作的数据都是事先分配好的（数据段主要容纳全局变量，且是静态分配的），**没有直接体现对函数的支持**（只是硬件层面上提供了栈指针寄存器，编译器实现函数参数、返回值压栈、出栈）。**没有函数，代码会重复，有了函数，才有局部变量一说，有了局部变量才有了数据的动态申请与分配一说**。函数及其局部变量 是最早的 代码+数据的封装。



[Go 垃圾回收（二）——垃圾回收是什么？](https://zhuanlan.zhihu.com/p/104623357)GC 不负责回收栈中的内存，为什么呢？主要原因是**栈是一块专用内存，专门为了函数执行而准备的**，存储着函数中的局部变量以及调用栈。除此以外，栈中的数据都有一个特点——简单。比如局部变量就不能被函数外访问，所以**这块内存用完就可以直接释放**。正是因为这个特点，栈中的数据可以通过简单的编译器指令自动清理，也就不需要通过 GC 来回收了。

函数调用栈相关的指令和寄存器

1. rsp 栈顶寄存器和rbp栈基址寄存器：这两个寄存器都跟函数调用栈有关，其中rsp寄存器一般用来存放函数调用栈的栈顶地址，而rbp寄存器通常用来存放函数的栈帧起始地址，编译器一般使用这两个寄存器加一定偏移的方式来**访问函数局部变量或函数参数**。比如：`mov    0x8(%rsp),%rdx`
2. `call 目标地址`指令执行函数调用。CPU执行call指令时首先会把rip寄存器中的值入栈，然后设置rip值为目标地址，又因为rip寄存器决定了下一条需要执行的指令，所以当CPU执行完当前call指令后就会跳转到目标地址去执行。一条call指令修改了3个地方的值：rip寄存器、rsp和栈。
3. ret指令从被调用函数返回调用函数，它的实现原理是把call指令入栈的返回地址弹出给rip寄存器。一条call指令修改了3个地方的值：rip寄存器、rsp和栈。

一些有意思的表述：

1. 函数调用后的返回地址会保存到堆栈中。jmp跳过去就不知道怎么回来了，而通过call跳过去后，是可以通过ret指令直接回来的。**call指令保存eip的地方叫做栈**，在内存里，ret指令执行的时候是直接取出栈中保存的eip值 并恢复回去，达到返回的效果。PS： rsp 和 rsp 就好像一个全局变量，一经改变，call/ret/push/pop 这些指令的行为就都改变了。
2. 函数的局部状态也可以保存到堆栈中。在汇编环境下，寄存器是全局可见的，不能用于充当局部变量。借鉴call指令保存返回地址的思路，如果，在每一层函数中都将当前比较关键的寄存器保存到堆栈中，然后才去调用下一层函数，并且，下层的函数返回的时候，再将寄存器从堆栈中恢复出来，这样也就能够保证下层的函数不会破坏掉上层函数的状了。

[go调度器源代码情景分析之九：操作系统线程及线程调度](https://mp.weixin.qq.com/s/OvGlI5VvvRdMRuJegNrOMg)**线程是什么？**进程切换、线程切换、协程切换、函数切换，cpu 只有一个，寄存器也只有一批，想要并发，就得对硬件分时（cpu和寄存器）使用，分时就得save/load，切换时保留现场，返回时恢复现场。线程调度时操作系统需要保存和恢复的寄存器除了通用寄存器之外，还包括指令指针寄存器rip以及与栈相关的栈顶寄存器rsp和栈基址寄存器rbp，rip寄存器决定了线程下一条需要执行的指令，2个栈寄存器确定了线程执行时需要使用的栈内存。所以恢复CPU寄存器的值就相当于改变了CPU下一条需要执行的指令，同时也切换了函数调用栈，因此从调度器的角度来说，线程至少包含以下3个重要内容：

1. 一组通用寄存器的值（切换时涉及到save和load）
2. 将要执行的下一条指令的地址
3. 栈

||寄存器|栈等资源||
|---|---|---|---|
|函数调用|PC和栈寄存器|如果有参数或返回值的话需要几次用户栈操作|不到1ns|
|协程切换|少数几个寄存器|协程栈切换|
|系统调用|SS、ESP、EFLAGS、CS和EIP寄存器|同一进程用户态和内核态栈切换|1000条左右的cpu指令<br>200ns到10+us|
|线程切换|寄存器|用户态和内核态栈切换到另一线程|
|进程切换|几乎所有寄存器|用户态和内核态栈切换到另一线程，切换虚拟内存，全局变量等资源|

系统调用：在内核栈的最高地址端，存放的是一个结构pt_regs，这个结构的作用是：当系统调用从用户态到内核态的时候，首先要做的第一件事情，就是将用户态运行过程中的CPU上下文也即各种寄存器保存在这个结构里。这里两个重要的两个寄存器SP和IP，SP里面是用户态程序运行到的栈顶位置，IP里面是用户态程序运行到的某行代码。

所以操作系统对线程的调度可以简单的理解为内核调度器对不同线程所使用的寄存器和栈的切换。最后，我们对操作系统线程下一个简单且不准确的定义：操作系统线程是由内核负责调度且**拥有自己私有的一组寄存器值和栈的执行流**。

[函数运行时在内存中是什么样子？](https://mp.weixin.qq.com/s/fyrnqiK8ucGjmUxuHeakNQ)进程和线程的运行体现在函数执行上，函数的执行除了函数内部执行的顺序执行还有子函数调用的控制转移以及子函数执行完毕的返回。函数调用是一个First In Last Out 的顺序，天然适用于栈这种数据结构来处理。当函数在运行时每个函数也要有自己的一个“小盒子”，这个小盒子中保存了函数运行时的各种信息，这些小盒子通过栈这种结构组织起来，这个小盒子就被称为栈帧，stack frames，也有的称之为call stack。当函数A调用函数B的时候，控制从A转移到了B，所谓控制其实就是指CPU执行属于哪个函数的机器指令。当函数A调用函数B时，我们只要知道：函数A对于的机器指令执行到了哪里 (我从哪里来，返回 ret)；函数B第一条机器指令所在的地址 (要到哪里去，跳转 call)。

**编译成机器码的函数有什么特点呢？**在被调用者的函数体内，通常会分为三个部分。头尾两个部分叫做序曲（prelude）和尾声（epilogue），分别做一些初始化工作和收尾工作。在序曲里会保存原来的栈指针，以及把自己应该保护的寄存器存到栈里、设置新的栈指针等，接着执行函数的主体逻辑。最后，到尾声部分，要根据调用约定把返回值设置到寄存器或栈，恢复所保护的寄存器的值和栈顶指针，接着跳转到返回地址。

jump 和call 的区别是：jump 之后不用回来，而call（也就是函数调用是需要回来的）。有没有一个可以不跳转回到原来开始的地方，来实现函数的调用呢？
1. 把调用的函数指令，直接插入在调用函数的地方，替换掉对应的 call 指令，然后在编译器编译代码的时候，直接就把函数调用变成对应的指令替换掉。这个方法有些问题：如果函数 A 调用了函数 B，然后函数 B 再调用函数 A，我们就得面临在 A 里面插入 B 的指令，然后在 B 里面插入 A 的指令，这样就会产生无穷无尽地替换。PS：当然，内联函数完全可以这样做
2. **把后面要跳回来执行的指令地址给记录下来**。但是在多层函数调用里，简单只记录一个地址也是不够的。像我们一般使用的 Intel i7 CPU 只有 16 个 64 位寄存器，调用的层数一多就存不下了。最终，计算机科学家们想到了一个比单独记录跳转回来的地址更完善的办法。我们在内存里面开辟一段空间，用栈这个后进先出（LIFO，Last In First Out）的数据结构。

函数通过自身的指令遥控其对应的方法栈，可以往里面放入数值，从里面读取数据，也可以将数值移动到其它地方，从调用者的方法栈里取值。**函数执行的切换，不只是切换pc，还有切换sp 和bp ，因为函数代码中有很多 指令引用了 sp 和bp**。PS：就好比搬家不是换个住处，还有孩子的学校、户籍等等。

[一文教你搞懂 Go 中栈操作](https://mp.weixin.qq.com/s/H9ZYnJevZAnFaNsIH2wbjQ)In computer science, a call stack is a stack data structure that stores information about the active subroutines of a computer program.
In computer programming, a subroutine is a sequence of program instructions that performs a specific task, packaged as a unit.A stack frame is a frame of data that gets pushed onto the stack. In the case of a call stack, a stack frame would represent a function call and its argument data.栈是一种栈数据结构，用于存储有关计算机程序的活动 subroutines 信息。栈帧stack frame又常被称为帧frame是在调用栈中储存的函数之间的调用关系，每一帧对应了函数调用以及它的参数数据。

## 为什么需要堆？

Heap is used for dynamic memory allocation(data with dynamic size ) and unlike stack, the program needs to look up the data in heap using pointers.  

光有栈，对于面向过程的程序设计还远远不够，因为栈上的数据在函数返回的时候就会被释放掉，所以**无法将数据传递至函数外部**。而全局变量没有办法动态地产生，只能在编译的时候定义，有很多情况下缺乏表现力，在这种情况下，堆（Heap）是一种唯一的选择。The heap is an area of dynamically-allocated memory that is **managed automatically by the operating system or the memory manager library**. Memory on the heap is allocated, deallocated, and resized regularly during program execution, and this can lead to a problem called fragmentation. 堆适合管理生存期较长的一些数据，这些数据在退出作用域以后也不会消失。

如果堆上有足够的空间的满足我们代码的内存申请，内存分配器可以完成内存申请无需内核参与，否则将通过操作系统调用（brk）进行扩展堆，通常是申请一大块内存。（对于 malloc 大默认指的是大于 MMAP_THRESHOLD 个字节 - 128KB）。

任何线程都可以在堆上申请空间（全局的），因此每次申请堆内存都必须进行同步处理，竞争十分激烈，必然会出现线程阻塞。在java 中，为每个线程在eden 区分配一块TLAB 空间，线程在各自的TLAB内存区域申请空间，无需加锁，这是一种典型的空间换时间的策略。



## 其它

[为什么需要 GC](https://liujiacai.net/blog/2018/06/15/garbage-collection-intro/)
1. 在计算机诞生初期，在程序运行过程中没有栈帧（stack frame）需要去维护，所以内存采取的是静态分配策略，这虽然比动态分配要快，但是其一明显的缺点是程序所需的数据结构大小必须在编译期确定，而且不具备运行时分配的能力，这在现在来看是不可思议的。
2. 在 1958 年，Algol-58 语言首次提出了块结构（block-structured），块结构语言通过在内存中申请栈帧来实现按需分配的动态策略。在过程被调用时，帧（frame）会被压到栈的最上面，调用结束时弹出。PS：一个block 内的变量要么都可用 要么都回收，降低了管理成本。但是后进先出（Last-In-First-Out, LIFO）的**栈限制了栈帧的生命周期不能超过其调用者**，而且由于每个栈帧是固定大小，所以一个过程的返回值也必须在编译期确定。所以诞生了新的内存管理策略——堆（heap）管理。
3. 堆分配运行程序员按任意顺序分配/释放程序所需的数据结构——**动态分配的数据结构可以脱离其调用者生命周期的限制**，这种便利性带来的问题是垃圾对象的回收管理。

