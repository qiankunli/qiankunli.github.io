---

layout: post
title: 容器问题排查
category: 技术
tags: Docker
keywords: debug

---

## 简介（未完成）

* TOC
{:toc}


[干货携程容器偶发性超时问题案例分析（一）](https://mp.weixin.qq.com/s/bSNWPnFZ3g_gciOv_qNhIQ)

[干货携程容器偶发性超时问题案例分析（二）](https://mp.weixin.qq.com/s/7ZZqWPE1XNf9Mn_wj1HjUw)

两篇文章除了膜拜之外，最大的体会就是：业务、docker daemon、网关、linux内核、硬件  都可能会出问题，都得有手段去支持自己做排除法

## perf

![](/public/upload/linux/perf_event.png)

[perf](https://perf.wiki.kernel.org/index.php/Main_Page) began as a tool for using the performance counters subsystem in Linux, and has had various enhancements to add tracing capabilities. linux 有一个performance counters，perf 是这个子系统的外化。

Performance counters are CPU hardware registers that count hardware events such as instructions executed, cache-misses suffered, or branches mispredicted. They form a basis for **profiling applications** to trace dynamic control flow and identify hotspots. perf provides rich generalized abstractions over hardware specific capabilities. Among others, it provides per task, per CPU and per-workload counters, sampling on top of these and source code event annotation.

Tracepoints are instrumentation points placed at logical locations in code, such as for system calls, TCP/IP events, file system operations, etc. These have negligible(微不足道的) overhead when not in use, and can be enabled by the perf command to collect information including timestamps and stack traces. perf can also dynamically create tracepoints using the kprobes and uprobes frameworks, for kernel and userspace dynamic tracing. The possibilities with these are endless. 基于已有的或动态创建的tracepoints 跟踪数据 

The userspace perf command present a simple to use interface with commands like:

1. perf stat: obtain event counts
2. perf record: record events for later reporting
3. perf report: break down events by process, function, etc.
4. perf annotate: annotate assembly or source code with event counts
5. perf top: see live event count
6. perf bench: run different kernel microbenchmarks

[perf Tutorial](https://perf.wiki.kernel.org/index.php/Tutorial)

[Perf -- Linux下的系统性能调优工具，第 1 部分](https://www.ibm.com/developerworks/cn/linux/l-cn-perf1/index.html) 值得再细读一遍，尤其是如何从linxu及硬件 视角优化性能。

当算法已经优化，代码不断精简，人们调到最后，便需要斤斤计较了。cache 啊，流水线啊一类平时不大注意的东西也必须精打细算了。

[干货携程容器偶发性超时问题案例分析（一）](https://mp.weixin.qq.com/s/bSNWPnFZ3g_gciOv_qNhIQ)[干货携程容器偶发性超时问题案例分析（二）](https://mp.weixin.qq.com/s/7ZZqWPE1XNf9Mn_wj1HjUw) 这两篇文章中，使用perf sched 发现了物理上存在的调度延迟问题。

### perf stat

 perf stat 针对程序 t1 的输出

    $perf stat ./t1 
    Performance counter stats for './t1': 
    
    262.738415 task-clock-msecs # 0.991 CPUs 
    2 context-switches # 0.000 M/sec 
    1 CPU-migrations # 0.000 M/sec 
    81 page-faults # 0.000 M/sec 
    9478851 cycles # 36.077 M/sec (scaled from 98.24%) 
    6771 instructions # 0.001 IPC (scaled from 98.99%) 
    111114049 branches # 422.908 M/sec (scaled from 99.37%) 
    8495 branch-misses # 0.008 % (scaled from 95.91%) 
    12152161 cache-references # 46.252 M/sec (scaled from 96.16%) 
    7245338 cache-misses # 27.576 M/sec (scaled from 95.49%) 
    
    0.265238069 seconds time elapsed 

perf stat 给出了其他几个最常用的统计信息：

1. Task-clock-msecs：CPU 利用率，该值高，说明程序的多数时间花费在 CPU 计算上而非 IO。
2. Context-switches：进程切换次数，记录了程序运行过程中发生了多少次进程切换，频繁的进程切换是应该避免的。
3. Cache-misses：程序运行过程中总体的 cache 利用情况，如果该值过高，说明程序的 cache 利用不好
4. CPU-migrations：表示进程 t1 运行过程中发生了多少次 CPU 迁移，即被调度器从一个 CPU 转移到另外一个 CPU 上运行。
5. Cycles：处理器时钟，一条机器指令可能需要多个 cycles，
6. Instructions: 机器指令数目。
7. IPC：是 Instructions/Cycles 的比值，该值越大越好，说明程序充分利用了处理器的特性。
8. Cache-references: cache 命中的次数
9. Cache-misses: cache 失效的次数。

通过指定 -e 选项，可以改变 perf stat 的缺省事件 ( 可以通过 perf list 来查看 )。假如你已经有很多的调优经验，可能会使用 -e 选项来查看您所感兴趣的特殊的事件。

### perf sched

[Perf -- Linux下的系统性能调优工具，第 2 部分](https://www.ibm.com/developerworks/cn/linux/l-cn-perf2/index.html)未读

perf sched 使用了转储后再分析 (dump-and-post-process) 的方式来分析内核调度器的各种事件