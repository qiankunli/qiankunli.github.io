---

layout: post
title: 硬件对软件设计的影响
category: 技术
tags: Architecture
keywords: Permission

---

## 简介

物质基础决定上层建筑，在计算机世界里，硬件作为“物质基础”深刻的影响着软件设计和性能。

## 缓存

绝大多数时候，两个线程通过内存共享数据，但每个cpu都会保有一份儿共享数据的副本——缓存，这就容易引起不一致。

### 缓存行

[高性能队列——Disruptor](https://tech.meituan.com/disruptor.html)

cpu和内存的速度差异 ==> 缓存 ==> 多级缓存 ==> Cache是由很多个cache line组成的。每个cache line通常是64字节，并且它有效地引用主内存中的一块儿地址。CPU每次从主存中拉取数据时，会把相邻的数据也存入同一个cache line。也就是说，假设一个cache line 对应内存地址是0x1000，存着一个volatile变量，你改了这个变量，那么跟它挨着的另一个变量（地址为0x1008）也会失效（假设它们同属于一个java对象内存结构，或都是某个数组的元素）因为整个cache line被标记为失效了。下次访问第二个变量时，便需要从内存中加载到缓存，再加载到cpu。

因此，缓存行中的64byte 数据，一个失效全失效，有时会带来一些性能问题。

## 内存 + cpu 二级结构

1. 所谓线程 安全，最后可以归结到 并发读写 问题。参见 [多线程](http://qiankunli.github.io/2014/10/09/Threads.html)
2. 所谓数据结构，最后可以归结到 读写效率的 权衡问题。