---

layout: post
title: 软件机制
category: 技术
tags: Basic
keywords: Permission

---

## 简介

* TOC
{:toc}

## 为什么要有堆和栈

[Memory Management/Stacks and Heaps](https://en.m.wikibooks.org/wiki/Memory_Management/Stacks_and_Heaps)

1. The system stack,  are used most often to provide **frames**. A frame is a way to localize information about subroutines（可以理解为函数）. 
2. In general, a subroutine must have in its frame the return address (where to jump back to when the subroutine completes), the function's input parameters. When a subroutine is called, all this information is pushed onto the stack in a specific order. When the function returns, all these values on the stack are popped back off, reclaimed to the system for later use with a different function call. In addition, subroutines can also use the stack as storage for local variables.

通常可执行程序有一定的格式：代码段＋数据段。**但程序的一次执行过程是动态的，为了增加动态性，肯定会增加一些自己的数据结构**。进程在推进运行的过程中会调用一些数据，可能中间会产生一些数据保留在堆栈段，所以是动态的。

进程的典型内存布局

1. 代码段
2. 初始化数据段/数据段 ，包括全局变量和静态变量，在编程时就已经被初始化
3. 未初始化数据段/BSS段
4. 堆栈段
5. 上下文信息


​栈 (stack) 是现代计算机程序里最为重要的概念之一，几乎每一个程序都使用了栈，**没有栈就没有函数，没有局部变量，也就没有我们如今能够看见的所有的计算机语言**。在数据结构中，栈被定义为一个特殊的容器，先进后出。在计算机系统中，栈则是一个具有以上属性的动态内存区域。栈在程序运行中具有举足轻重的地位。**最重要的，栈保存了一个函数调用所需要的维护信息**，这常常被称为堆栈帧(Stack Frame)。PS： **为什么需要栈？为了支持函数**。OS设计体现了对进程、线程的支持，直接提供系统调用创建进程、线程，但就进程/线程内部来说，os 认为代码段是一个指令序列，最多jump几下，指令操作的数据都是事先分配好的（数据段主要容纳全局变量，且是静态分配的），**没有直接体现对函数的支持**（只是硬件层面上提供了栈指针寄存器，编译器实现函数参数、返回值压栈、出栈）。**没有函数，代码会重复，有了函数，才有局部变量一说，有了局部变量才有了数据的动态申请与分配一说**。

函数及其局部变量 是最早的 代码+数据的封装。

为什么需要堆？光有栈，对于面向过程的程序设计还远远不够，因为栈上的数据在函数返回的时候就会被释放掉，所以**无法将数据传递至函数外部**。而全局变量没有办法动态地产生，只能在编译的时候定义，有很多情况下缺乏表现力，在这种情况下，堆（Heap）是一种唯一的选择。The heap is an area of dynamically-allocated memory that is **managed automatically by the operating system or the memory manager library**. Memory on the heap is allocated, deallocated, and resized regularly during program execution, and this can lead to a problem called fragmentation. 

笔者一直以来的困惑就是：为何进程有了数据段，还要有堆栈段？因为数据段 + 代码段 + 指令寄存器 也只是支持程序run起来，数据段主要容纳全局变量，且是静态分配的，跨线程的。数据段远远不是现代程序的主角了。

