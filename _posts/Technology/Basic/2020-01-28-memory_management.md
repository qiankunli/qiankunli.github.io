---

layout: post
title: 内存管理玩法汇总
category: 技术
tags: Basic
keywords: memory management

---

## 简介

* TOC
{:toc}

[Demystifying memory management in modern programming languages](https://deepu.tech/memory-management-in-programming/)是一个系列，建议细读。推荐极客时间教程《编程高手必学的内存知识》


## 内存分配

内存分配一般有三种方式：静态存储区（根对象、静态变量、常量）、栈（函数中的临时局部变量）、堆（malloc、new等）；

内存的分配和使用其是一个增加系统中熵的过程，内存分配器只包含线性内存分配器（Sequential Allocator）和空闲链表内存分配器（Free-list Allocator）两种，其它内存分配器其实都是上述两种不同分配器的变种。
1. 线性内存分配器。**“用到哪了标识到哪”**，维护一个指向内存特定位置的指针，当用户程序申请内存时，分配器只需要检查剩余的空闲内存、返回分配的内存区域并修改指针在内存中的位置。PS： 假设先分配A后分配B，则在回收B之前无法回收A
    1. 已分配的内存无法直接回收，需要与**拷贝方式**的垃圾回收算法配合使用，比如标记压缩（Mark-Compact）、复制回收（Copying GC）和分代回收（Generational GC）等，所以 C 和 C++ 等需要直接对外暴露指针的语言就无法使用该策略。
2. 空闲链表内存分配器。**将内存按照块的结构来进行划分**，维护一个类似链表的数据结构。当用户程序申请内存时，空闲链表分配器会依次遍历空闲的内存块，找到足够大的内存，然后申请新的资源并修改链表。因为分配内存时需要遍历链表，所以它的时间复杂度就是 O(n)，为了提高效率，**将内存分割成多个链表**，每个链表中的内存块大小相同（不同链表不同），申请内存时先找到满足条件的链表，再从链表中选择合适的内存块，减少了需要遍历的内存块数量。

![](/public/upload/go/mm_envole.png)

## 内存回收/gc

独立成文。

## 整体描述

[内存管理设计精要](https://mp.weixin.qq.com/s/lx0vX-4wlQWxxfj017T9HA)虽然多数系统都会将内存管理拆分成多个复杂的模块并引入一些中间层提供缓存和转换的功能，但是内存管理系统实际上都可以简化成两个模块，即内存分配器（Allocator）、垃圾收集器（Collector）。在研究内存管理时都会引入第三个模块 — 用户程序（Mutator/App）

![](/public/upload/basic/mm_overview.jpg)

一个优秀的通用内存分配器应具有以下特性:

1. 额外的空间损耗尽量少
2. 分配速度尽可能快
3. 尽量避免内存碎片
4. 缓存本地化友好。每个线程有自己的栈，每个线程有自己的堆缓存（减少对全局堆内存管理结构的争用）
5. 通用性，兼容性，可移植性，易调试

我们可以将内存管理简单地分成手动管理和自动管理两种方式，手动管理内存一般是指由工程师在需要时通过 malloc 等函数手动申请内存并在不需要时调用 free 等函数释放内存；自动管理内存由编程语言的内存管理系统自动管理，在编程语言的编译期或者运行时中引入，最常见的自动内存管理机制就是垃圾回收，一些编程语言也会使用自动引用计数辅助内存的管理，自动内存管理会带来额外开销并影响语言的运行时性能。

![](/public/upload/basic/memory_pool.jpg)

**内存管理可以分为三个层次**，自底向上分别是：

1. 操作系统内核的内存管理
2. glibc层使用系统调用维护的内存管理算法。glibc/ptmalloc2，google/tcmalloc，facebook/jemalloc。C 库内存池工作时，会预分配比你申请的字节数更大的空间作为内存池。比如说，当主进程下申请 1 字节的内存时，Ptmalloc2 会预分配 132K 字节的内存（Ptmalloc2 中叫 Main Arena），应用代码再申请内存时，会从这已经申请到的 132KB 中继续分配。当我们释放这 1 字节时，Ptmalloc2 也不会把内存归还给操作系统。
3. 应用程序从glibc动态分配内存后，根据应用程序本身的程序特性进行优化， 比如netty的arena

进程/线程与内存：Linux 下的 JVM 编译时默认使用了 Ptmalloc2 内存池，64 位的 Linux 为**每个线程**的栈分配了 8MB 的内存，还（为每个线程？）预分配了 64MB 的内存作为堆内存池。在多数情况下，这些预分配出来的内存池，可以提升后续内存分配的性能。但也导致创建很多线程时会占用大量内存。

栈空间由编译器 + os管理。如果堆上有足够的空间的满足我们代码的内存申请，内存分配器可以完成内存申请无需内核参与，否则将通过操作系统调用（brk）进行扩展堆，通常是申请一大块内存。（对于 malloc 大默认指的是大于 MMAP_THRESHOLD 个字节 - 128KB）。但是，内存分配器除了更新 brk address 还有其他职责。其中主要的一项就是如何减少 内部（internal）和外部（external）碎片和如何快速分配当前块。我们该如何减少内存碎片化呢 ？答案取决是使用哪种内存分配算法，也就是使用哪个底层库。


## 操作系统

[Linux内核基础知识](http://blog.zhifeinan.top/2019/05/01/linux_kernel_basic.html)

![](/public/upload/basic/mmu.png)

[Linux 内存管理知识总结（一）](https://zhuanlan.zhihu.com/p/366957562)

![](/public/upload/linux/linux_virtual_address.png)

[30张图带你领略glibc内存管理精髓](https://mp.weixin.qq.com/s/pdv5MMUQ9ACpeCpyGnxb1Q)Linux 系统在装载 elf 格式的程序文件时，会调用 loader 把可执行文件中的各个段依次载入到从某一地址开始的空间中。用户程序可以直接使用系统调用来管理 heap 和mmap 映射区域，但更多的时候程序都是使用 C 语言提供的 malloc()和 free()函数来动态的分配和释放内存。stack区域是唯一不需要映射，用户却可以访问的内存区域，这也是利用堆栈溢出进行攻击的基础。

1. 对于heap的操作，操作系统提供了brk()函数，c运行时库提供了sbrk()函数。
2. 对于mmap映射区域的操作，操作系统提供了mmap()和munmap()函数。

![](/public/upload/basic/glibc_mm.png)

内存的延迟分配，只有在真正访问一个地址的时候才建立这个地址的物理映射，这是 Linux 内存管理的基本思想之一。Linux 内核在用户申请内存的时候，只是给它分配了一个线性区（也就是虚拟内存），并没有分配实际物理内存；只有当用户使用这块内存的时候，内核才会分配具体的物理页面给用户，这时候才占用宝贵的物理内存。内核释放物理页面是通过释放线性区，找到其所对应的物理页面，将其全部释放的过程。

```c
struct mm_struct {
    ...
    unsigned long (*get_unmapped_area) (struct file *filp,
    unsigned long addr, unsigned long len,
    unsigned long pgoff, unsigned long flags);
    ...
    unsigned long mmap_base; /* base of mmap area */
    unsigned long task_size; /* size of task vm space */
    ...
    unsigned long start_code, end_code, start_data, end_data;   
    unsigned long start_brk, brk, start_stack;
    unsigned long arg_start, arg_end, env_start, env_end;
    ...
    // [start_code,end_code)表示代码段的地址空间范围
    // [start_data,end_start)表示数据段的地址空间范围
    // [start_brk,brk)分别表示heap段的起始空间和当前的heap指针
    // [start_stack,end_stack)表示stack段的地址空间范围
    // mmap_base表示memory mapping段的起始地址
}
```
C语言的动态内存分配基本函数是 malloc()，在 Linux 上的实现是通过内核的 brk 系统调用。brk()是一个非常简单的系统调用， 只是简单地改变mm_struct结构的成员变量 brk 的值。

## 语言/运行时

进程在推进运行的过程中会调用一些数据，可能中间会产生一些数据保留在堆栈段，栈空间 一般在编译时确定（**在编译器和操作系统的配合下，栈里的内存可以实现自动管理**）， 堆空间 运行时动态管理。对于堆具体的说，由runtime 在启动时 向操作 系统申请 一大块内存，然后根据 自定义策略进行分配与回收（手动、自动，分代不分代等），必要时向操作系统申请内存扩容。

从堆还是栈上分配内存？如果你使用的是静态类型语言，那么，不使用 new 关键字分配的对象大都是在栈中的，通过 new 或者 malloc 关键字分配的对象则是在堆中的。对于动态类型语言，无论是否使用 new 关键字，内存都是从堆中分配的。

**为什么从栈中分配内存会更快？**由于每个线程都有独立的栈，所以分配内存时不需要加锁保护，而且栈上对象的size在编译阶段就已经写入可执行文件了，执行效率更高！性能至上的 Golang 语言就是按照这个逻辑设计的，即使你用 new 关键字分配了堆内存，但编译器如果认为在栈中分配不影响功能语义时，会自动改为在栈中分配。

当然，在栈中分配内存也有缺点，它有功能上的限制。
1. 栈内存生命周期有限，它会随着函数调用结束后自动释放，在堆中分配的内存，并不随着分配时所在函数调用的结束而释放，它的生命周期足够使用。
2. 栈的容量有限，如 CentOS 7 中是 8MB 字节，如果你申请的内存超过限制会造成栈溢出错误（比如，递归函数调用很容易造成这种问题），而堆则没有容量限制。

[Go 内存对齐的那些事儿](https://mp.weixin.qq.com/s/cg0pq6X1eGlm2lbD14F_bA)

### jvm

### go runtime

[Visualizing memory management in Golang](https://deepu.tech/memory-management-in-golang/#:~:text=collection%20comes%20in.-,Go%20Memory%20management,is%20well%20optimized%20and%20efficient.) 有动图建议细读


## 中间件

os、语言运行时等层面 对内存分配已经做了很多优化，但是很多框架 依然会使用对象池，为什么呢？可能申请内存这个路径  的耗时并没有减少多少，但是因为用完 通过gc 回收的，大量小对象对 gc 压力还是挺大的。因此，基于gc 类语言的做高性能框架，可以把 对象池 可以作为一个通用的优化点。对象池 borrow/returnObject 手动管理内存还是有用武之地的。 

1. netty arena
2. kafka BufferPool
3. redis
4. go hertz RequestContext 池

## 其它

![](/public/upload/linux/linux_memory_management.png)