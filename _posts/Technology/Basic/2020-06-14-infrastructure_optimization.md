---

layout: post
title: 基础设施优化
category: 技术
tags: Basic
keywords: infrastructure optimization

---

## 简介

* TOC
{:toc}

## 零拷贝

把文件内容发送到网络。这个过程发生在用户空间，文件和网络socket属于硬件资源，读取磁盘或者操作网卡都由操作系统内核完成。在操作系统内部，整个过程为:

![](/public/upload/basic/copy_file_to_socket.jpg)

在Linux Kernal 2.2之后出现了一种叫做“零拷贝(zero-copy)”系统调用机制，就是跳过“用户缓冲区”的拷贝，建立一个磁盘空间和内存空间的直接映射，数据不再复制到“用户态缓冲区”，系统上下文切换减少2次，可以提升一倍性能

![](/public/upload/basic/copy_file_to_socket_in_kernel.jpg)

如果网卡支持 SG-DMA（The Scatter-Gather Direct Memory Access）技术，还可以再去除 Socket 缓冲区的拷贝，这样一共只有 2 次内存拷贝。

![](/public/upload/basic/copy_file_to_socket_sg_dma.jpg)

[零拷贝及一些引申内容](https://mp.weixin.qq.com/s/l_MRLyRW8lxvjtsKapT6HA)

## 用户态与内核态切换有什么代价呢？

用户态的程序只能通过调用系统提供的API/系统调用来申请并使用资源，比如有个read 系统调用 用户态程序不能直接调用read，而是要`systemcall read系统调用号`。为了避免用户态程序绕过操作系统，直接执行对于硬件的控制和操作，**操作系统利用CPU所提供的特权机制**，封锁一些指令，并且将内存地址进行虚拟化（Ring 3无法执行一些指令，访问一些地址），使得存储有关键数据（比如IO映射）的部分物理内存无法从用户态进程进行访问。PS: 就好像你永远只能给运维提交工单，而不能直接操作一样。

我们的应用程序运行在 Ring 3（我们通常叫用户态，cpu的状态），而操作系统内核运行在 Ring 0（我们通常叫内核态）。所以一次中断调用，不只是“函数调用”，更重要的是改变了执行权限，从用户态跃迁到了内核态。

[Understanding User and Kernel Mode](https://blog.codinghorror.com/understanding-user-and-kernel-mode/)

1. 在操作系统中，In Kernel mode, the executing code has complete and unrestricted access to the underlying hardware. It can execute any CPU instruction and reference any memory address. 而用户态可以访问的指令和地址空间是受限的
2. 用户态和内核态的切换通常涉及到内存的复制，比如内核态read 得到的数据返回给 用户态，因为用户态访问不了内核态的read 返回数据。
3. jvm 则再插一腿，因为jvm 数据不仅在用户态，jvm 还希望数据是由jvm heap管理，所以对于read 操作来讲，数据从内核态 ==> 用户态  ==> jvm heap 经历了两次复制，netty 中允许使用堆外内存（对于linux来说，jvm heap和堆外内存都在进程的堆内存之内） 减少一次复制
4. linux 和jvm 都可以使用 mmap来减少用户态和内核态的内存复制，但一是应用场景有限，二是代码复杂度提升了好多。

