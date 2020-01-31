---

layout: post
title: 软件机制
category: 技术
tags: Basic
keywords: software mechanism

---

## 简介

* TOC
{:toc}

## 为什么要有堆和栈


[为什么需要 GC](https://liujiacai.net/blog/2018/06/15/garbage-collection-intro/)
1. 在计算机诞生初期，在程序运行过程中没有栈帧（stack frame）需要去维护，所以内存采取的是静态分配策略，这虽然比动态分配要快，但是其一明显的缺点是程序所需的数据结构大小必须在编译期确定，而且不具备运行时分配的能力，这在现在来看是不可思议的。
2. 在 1958 年，Algol-58 语言首次提出了块结构（block-structured），块结构语言通过在内存中申请栈帧来实现按需分配的动态策略。在过程被调用时，帧（frame）会被压到栈的最上面，调用结束时弹出。**栈分配策略赋予程序员极大的自由度，局部变量在不同的调用过程中具有不同的值**，这为递归提供了基础。但是后进先出（Last-In-First-Out, LIFO）的**栈限制了栈帧的生命周期不能超过其调用者**，而且由于每个栈帧是固定大小，所以一个过程的返回值也必须在编译期确定。所以诞生了新的内存管理策略——堆（heap）管理。
3. 堆分配运行程序员按任意顺序分配/释放程序所需的数据结构——**动态分配的数据结构可以脱离其调用者生命周期的限制**，这种便利性带来的问题是垃圾对象的回收管理。

通常可执行程序有一定的格式：代码段＋数据段。**但程序的一次执行过程是动态的，为了增加动态性，肯定会增加一些自己的数据结构**。进程在推进运行的过程中会调用一些数据，可能中间会产生一些数据保留在堆栈段，所以是动态的。

进程的典型内存布局

1. 代码段
2. 初始化数据段/数据段 ，包括全局变量和静态变量，在编程时就已经被初始化
3. 未初始化数据段/BSS段
4. 堆栈段
5. 上下文信息

### 为什么需要栈

[Memory Management/Stacks and Heaps](https://en.m.wikibooks.org/wiki/Memory_Management/Stacks_and_Heaps)

1. The system stack,  are used most often to provide **frames**. A frame is a way to localize information about subroutines（可以理解为函数）. 
2. In general, a subroutine must have in its frame the return address (where to jump back to when the subroutine completes), the function's input parameters. When a subroutine is called, all this information is pushed onto the stack in a specific order. When the function returns, all these values on the stack are popped back off, reclaimed to the system for later use with a different function call. In addition, subroutines can also use the stack as storage for local variables.

​栈 (stack) 是现代计算机程序里最为重要的概念之一，几乎每一个程序都使用了栈，**没有栈就没有函数，没有局部变量，也就没有我们如今能够看见的所有的计算机语言**。在数据结构中，栈被定义为一个特殊的容器，先进后出。在计算机系统中，栈则是一个具有以上属性的动态内存区域。栈在程序运行中具有举足轻重的地位。**最重要的，栈保存了一个函数调用所需要的维护信息**，这常常被称为堆栈帧(Stack Frame)。

根据上文可以推断：**为什么需要栈？为了支持函数**。OS设计体现了对进程、线程的支持，直接提供系统调用创建进程、线程，但就进程/线程内部来说，os 认为代码段是一个指令序列，最多jump几下，指令操作的数据都是事先分配好的（数据段主要容纳全局变量，且是静态分配的），**没有直接体现对函数的支持**（只是硬件层面上提供了栈指针寄存器，编译器实现函数参数、返回值压栈、出栈）。**没有函数，代码会重复，有了函数，才有局部变量一说，有了局部变量才有了数据的动态申请与分配一说**。函数及其局部变量 是最早的 代码+数据的封装。

### 为什么需要堆？

光有栈，对于面向过程的程序设计还远远不够，因为栈上的数据在函数返回的时候就会被释放掉，所以**无法将数据传递至函数外部**。而全局变量没有办法动态地产生，只能在编译的时候定义，有很多情况下缺乏表现力，在这种情况下，堆（Heap）是一种唯一的选择。The heap is an area of dynamically-allocated memory that is **managed automatically by the operating system or the memory manager library**. Memory on the heap is allocated, deallocated, and resized regularly during program execution, and this can lead to a problem called fragmentation. 




## 调度的本质

[万字长文深入浅出 Golang Runtime](https://zhuanlan.zhihu.com/p/95056679)**CPU 在时钟的驱动下, 根据 PC 寄存器从程序中取指令和操作数, 从 RAM 中取数据, 进行计算, 处理, 跳转, 驱动执行流往前. CPU 并不关注处理的是线程还是协程, 只需要设置 PC 寄存器, 设置栈指针等(这些称为上下文), 那么 CPU 就可以欢快的运行这个线程或者这个协程了**.

**调度的本质： 给CPU找活儿干 ==> PC 及 栈指针 能用即可  ==> 多任务 就得维护多份儿 PC及栈指针（栈空间） ==> PC（当前执行位置） 栈空间 等聚合成一个数据结构，称为协程/线程/进程 ==> CPU层次的切换PC/SP 变成了切换 这个struct**

**真正运行的实体是CPU and 线程/协程是一个数据结构。**线程的运行其实是**被运行**.阻塞其实是切换出调度队列, 不再去调度执行这个执行流. 其他执行流满足其条件, 便会把被移出调度队列的执行流重新放回调度队列.协程同理, **协程其实也是一个数据结构**, 记录了要运行什么函数, 运行到哪里了.
