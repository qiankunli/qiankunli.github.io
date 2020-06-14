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

## 为什么需要反码和补码

### 有界/越界/溢出与取模

在数学的理论中，数字可以有无穷大，也有无穷小。现实中的计算机系统不可能表示无穷大或者无穷小的数字，都有一个上限和下限。加法加越界了就成了 取模运算。 

### 符号位

在实际的硬件系统中，**计算机 CPU 的运算器只实现了加法器88，而没有实现减法器。那么计算机如何做减法呢？我们可以通过加上一个负数来达到这个目的。如何让计算机理解哪些是正数，哪些是负数呢？人们把二进制数分为有符号数（signed）和无符号数（unsigned）。如果是有符号数，那么最高位就是符号位。如果是无符号数，那么最高位就不是符号位，而是二进制数字的一部分。有些编程语言，比如 Java，它所有和数字相关的数据类型都是有符号位的；而有些编程语言，比如 C 语言，它有诸如 unsigned int 这种无符号位的数据类型。

### 比取模更“狠”——有符号数的溢出 

对于 n 位的数字类型，符号位是 1，后面 n-1 位全是 0，我们把这种情形表示为 -2^(n-1)。n 位数字的最大的正值，其符号位为 0，剩下的 n-1 位都1，再增大一个就变为了符号位为 1，剩下的 n-1 位都为0。也就是**n位 有符号最大值 加1 就变成了 n位有符号数界限范围内最小的负数**——上溢出之后，又从下限开始

是不是有点扑克牌的意思， A 可以作为10JQKA 的最大牌，也可以作为A23456 的最小牌。 

||下限|上限|
|---|---|---|
|n位无符号数|0|2^n-1|
|n位有符号数|-2^(n-1)|2^(n-1)-1|

**取模 可以将（最大值+1） 变成下限值**，对于无符号数是0 ，对于有符号数是负数。

### 减法靠补码

原码就是我们看到的二进制的原始表示，是不是可以直接使用负数的原码来进行减法计算呢？答案是否定的，因为负数的原码并不适用于减法操作（加负数操作）

因为取模的特性，我们知道 `i = i + 模数`。 那么 `i-j = i-j + 模数` 也是成立的，进而`i-j = i + (模数 -j)`。`模数 -j` 即补码 可以对应到计算机的 位取反  和 加 1 操作

本质就是

1. 加法器不区分 符号位和数据位
2. 越界 等于 取模，对于有符号位的取模，可以使得 正数 变成负数

我们经常使用朴素贝叶斯算法 过滤垃圾短信，`P(A|B)=P(A) * P(B/A) / P(B)` 这个公式在数学上平淡无奇，但工程价值在于：实践中右侧数据比 左侧数据更容易获得。 cpu减法器也是类似的道理，`减法器 = CPU 位取反 + 加法器  `

## 为什么要有堆和栈


[为什么需要 GC](https://liujiacai.net/blog/2018/06/15/garbage-collection-intro/)
1. 在计算机诞生初期，在程序运行过程中没有栈帧（stack frame）需要去维护，所以内存采取的是静态分配策略，这虽然比动态分配要快，但是其一明显的缺点是程序所需的数据结构大小必须在编译期确定，而且不具备运行时分配的能力，这在现在来看是不可思议的。
2. 在 1958 年，Algol-58 语言首次提出了块结构（block-structured），块结构语言通过在内存中申请栈帧来实现按需分配的动态策略。在过程被调用时，帧（frame）会被压到栈的最上面，调用结束时弹出。**栈分配策略赋予程序员极大的自由度，局部变量在不同的调用过程中具有不同的值**，这为递归提供了基础。但是后进先出（Last-In-First-Out, LIFO）的**栈限制了栈帧的生命周期不能超过其调用者**，而且由于每个栈帧是固定大小，所以一个过程的返回值也必须在编译期确定。所以诞生了新的内存管理策略——堆（heap）管理。
3. 堆分配运行程序员按任意顺序分配/释放程序所需的数据结构——**动态分配的数据结构可以脱离其调用者生命周期的限制**，这种便利性带来的问题是垃圾对象的回收管理。

通常可执行程序有一定的格式：代码段＋数据段。**但程序的一次执行过程是动态的，为了增加动态性，肯定会增加一些自己的数据结构**。进程在推进运行的过程中会调用一些数据，可能中间会产生一些数据保留在堆栈段，所以是动态的。

![](/public/upload/basic/os_memory_view.jpg)

指令中的地址总要有落脚的地方，只有代码区是肯定不行的，静态数据区也不够灵活（待补充）。

### 为什么需要栈

[Memory Management/Stacks and Heaps](https://en.m.wikibooks.org/wiki/Memory_Management/Stacks_and_Heaps)

1. The system stack,  are used most often to provide **frames**. A frame is a way to localize information about subroutines（可以理解为函数）. 
2. In general, a subroutine must have in its frame the return address (where to jump back to when the subroutine completes), the function's input parameters. When a subroutine is called, all this information is pushed onto the stack in a specific order. When the function returns, all these values on the stack are popped back off, reclaimed to the system for later use with a different function call. In addition, subroutines can also use the stack as storage for local variables.

​栈 (stack) 是现代计算机程序里最为重要的概念之一，几乎每一个程序都使用了栈，**没有栈就没有函数，没有局部变量，也就没有我们如今能够看见的所有的计算机语言**。在数据结构中，栈被定义为一个特殊的容器，先进后出。在计算机系统中，栈则是一个具有以上属性的动态内存区域。栈在程序运行中具有举足轻重的地位。**最重要的，栈保存了一个函数调用所需要的维护信息**，这常常被称为堆栈帧(Stack Frame)。

根据上文可以推断：**为什么需要栈？为了支持函数**。OS设计体现了对进程、线程的支持，直接提供系统调用创建进程、线程，但就进程/线程内部来说，os 认为代码段是一个指令序列，最多jump几下，指令操作的数据都是事先分配好的（数据段主要容纳全局变量，且是静态分配的），**没有直接体现对函数的支持**（只是硬件层面上提供了栈指针寄存器，编译器实现函数参数、返回值压栈、出栈）。**没有函数，代码会重复，有了函数，才有局部变量一说，有了局部变量才有了数据的动态申请与分配一说**。函数及其局部变量 是最早的 代码+数据的封装。

加入线程的因素：**每个线程有独立的栈**（栈一般作为线程独占的内存空间，使用时也就无需担心并发安全），而栈既保留了变量的值，也保留了函数的调用关系、参数和返回值。

### 为什么需要堆？

光有栈，对于面向过程的程序设计还远远不够，因为栈上的数据在函数返回的时候就会被释放掉，所以**无法将数据传递至函数外部**。而全局变量没有办法动态地产生，只能在编译的时候定义，有很多情况下缺乏表现力，在这种情况下，堆（Heap）是一种唯一的选择。The heap is an area of dynamically-allocated memory that is **managed automatically by the operating system or the memory manager library**. Memory on the heap is allocated, deallocated, and resized regularly during program execution, and this can lead to a problem called fragmentation. 堆适合管理生存期较长的一些数据，这些数据在退出作用域以后也不会消失。

## 调度系统设计精要

[调度系统设计精要](https://mp.weixin.qq.com/s/R3BZpYJrBPBI0DwbJYB0YA)在计算机科学中，调度就是一种将任务（Work）分配给资源的方法。任务可能是虚拟的计算任务，例如线程、进程或者数据流，这些任务会被调度到硬件资源上执行，例如：处理器 CPU 等设备。调度模块的核心作用就是对有限的资源进行分配以实现最大化资源的利用率或者降低系统的尾延迟，**调度系统面对的就是资源的需求和供给不平衡的问题**。

||任务|资源|描述|
|---|---|---|---|
|操作系统|线程|cpu|
|Go|协程|cpu线程|
|Kubernetes|pod|node|为待运行的工作负载 Pod 绑定运行的节点 Node|
|CDN的资源调度||
|订单调度||
|离线任务调度||

![](/public/upload/basic/scheduler_design.png)




