---

layout: post
title: go编译器
category: 技术
tags: Go
keywords: go compiler

---

## 前言(持续更新)

* TOC
{:toc}

[Go语言编译器简介](https://github.com/gopherchina/conference/blob/master/2020/2.1.5%20Go%E8%AF%AD%E8%A8%80%E7%BC%96%E8%AF%91%E5%99%A8%E7%AE%80%E4%BB%8B.pdf) 未读完
1. N种语言+M种机器=N+M个任务，有几种方案
    1. 其它语言 ==> C ==> 各个机器
    2. 各个语言 ==> x86 ==> 各个机器
2. 通用编译器方案
    ![](/public/upload/basic/general_compiler.png)

SSA-IR（Single Static Assignment）是一种介于高级语言和汇编语言的中间形态的伪语言，从高级语言角度看，它是（伪）汇编；而从真正的汇编语言角度看，它是（伪）高级语言。顾名思义，SSA（Single Static Assignment）的两大要点是：
1. Static：每个变量只能赋值一次（因此应该叫常量更合适）；
2. Single：每个表达式只能做一个简单运算，对于复杂的表达式a*b+c*d要拆分成："t0=a*b; t1=c*d; t2=t0+t1;"三个简单表达式；

## go编译器

![](/public/upload/go/go_compiler.png)

[漫谈Go语言编译器（01）](https://mp.weixin.qq.com/s/0q0k8gGX56SBKJvfMquQkQ) 


[一个95分位延迟要求5ms的场景，如何做性能优化](https://mp.weixin.qq.com/s/BUpsa22bQhK1pQKW8fUVOw)Golang 的生态中相关工具我们能用到的有 pprof 和 trace。pprof 可以看 CPU、内存、协程等信息在压测流量进来时系统调用的各部分耗时情况。而 trace 可以查看 runtime 的情况，比如可以查看协程调度信息等。代码层面的优化，是 us 级别的，而针对业务对存储进行优化，可以做到 ms 级别的，所以优化越靠近应用层效果越好。对于代码层面，优化的步骤是：

1. 压测工具模拟场景所需的真实流量
2. pprof 等工具查看服务的 CPU、mem 耗时
3. 锁定**平顶山逻辑**，看优化可能性：异步化，改逻辑，加 cache 等
4. 局部优化完写 benchmark 工具查看优化效果
5. 整体优化完回到步骤一，重新进行 压测+pprof 看效果，看 95 分位耗时能否满足要求(如果无法满足需求，那就换存储吧~。

火焰图中圈出来的大平顶山都是可以优化的地方

![](/public/upload/go/go_profiler.png)

另外推荐一个不错的库，这是 Golang 布道师 Dave Cheney 搞的用来做性能调优的库，使用起来非常方便：https://github.com/pkg/profile，可以看 pprof和 trace 信息。有兴趣读者可以了解一下。
