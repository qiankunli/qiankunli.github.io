---

layout: post
title: 内存管理
category: 技术
tags: Basic
keywords: Permission

---

## 简介

* TOC
{:toc}

一个优秀的通用内存分配器应具有以下特性:

1. 额外的空间损耗尽量少
2. 分配速度尽可能快
3. 尽量避免内存碎片
4. 缓存本地化友好
5. 通用性，兼容性，可移植性，易调试

**内存管理可以分为三个层次**，自底向上分别是：

1. 操作系统内核的内存管理
2. glibc层使用系统调用维护的内存管理算法。glibc/ptmalloc2，google/tcmalloc，facebook/jemalloc
3. 应用程序从glibc动态分配内存后，根据应用程序本身的程序特性进行优化， 比如netty的arena

## 操作系统

![](/public/upload/linux/linux_memory_management.png)

[Linux内核基础知识](http://blog.zhifeinan.top/2019/05/01/linux_kernel_basic.html)

![](/public/upload/linux/linux_virtual_address.png)

## jvm

[java gc](http://qiankunli.github.io/2016/06/17/gc.html)

[JVM1——jvm小结](http://qiankunli.github.io/2014/10/27/jvm.html)极客时间《深入拆解Java虚拟机》垃圾回收的三种方式（免费的其实是最贵的）

1. 清除sweep，将死亡对象占据的内存标记为空闲。
2. 压缩，将存活的对象聚在一起
3. 复制，将内存两等分， 说白了是一个以空间换时间的思路。

## netty arena

代码上的体现 以netty arena 为例 [netty内存管理](http://qiankunli.github.io/2017/04/10/network_byte_buffer.html)

## kafka BufferPool

[《Apache Kafka源码分析》——Producer与Consumer](http://qiankunli.github.io/2017/12/08/kafka_learn_1.html)

## redis

[Redis源码分析](http://qiankunli.github.io/2019/04/20/redis_source.html)