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

## 用户态与内核态切换有什么代价呢？

[Understanding User and Kernel Mode](https://blog.codinghorror.com/understanding-user-and-kernel-mode/)

1. 在操作系统中，In Kernel mode, the executing code has complete and unrestricted access to the underlying hardware. It can execute any CPU instruction and reference any memory address. 而用户态可以访问的指令和地址空间是受限的
2. 用户态和内核态的切换通常涉及到内存的复制，比如内核态read 得到的数据返回给 用户态，因为用户态访问不了内核态的read 返回数据。
3. jvm 则再插一腿，因为jvm 数据不仅在用户态，jvm 还希望数据是由jvm heap管理，所以对于read 操作来讲，数据从内核态 ==> 用户态  ==> jvm heap 经历了两次复制，netty 中允许使用堆外内存（对于linux来说，jvm heap和堆外内存都在进程的堆内存之内） 减少一次复制
4. linux 和jvm 都可以使用 mmap来减少用户态和内核态的内存复制，但一是应用场景有限，二是代码复杂度提升了好多。

