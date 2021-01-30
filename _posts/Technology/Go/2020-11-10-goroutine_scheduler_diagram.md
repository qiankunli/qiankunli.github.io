---

layout: post
title: 图解Goroutine 调度
category: 技术
tags: Go
keywords: Go goroutine scheduler

---

## 前言

* TOC
{:toc}

goroutine一些重要设计：
1. 堆栈开始很小（只有 4K），但可按需自动增长；
2. 坚决干掉了 “线程局部存储（TLS）” 特性的支持，让执行体更加精简；
3. 提供了同步、互斥和其他常规执行体间的通讯手段，包括大家非常喜欢的 channel；
4. 提供了几乎所有重要的系统调用（尤其是 IO 请求）的包装。

[调度系统设计精要](https://mp.weixin.qq.com/s/Ge9YgIi9jwrTEOrz3cnvdg)

本文内容来自 张万波大佬 [go语言调度器源代码情景分析](https://mp.weixin.qq.com/mp/homepage?__biz=MzU1OTg5NDkzOA==&hid=1&sn=8fc2b63f53559bc0cee292ce629c4788&scene=25#wechat_redirect)。在此再次表达 对大佬的膜拜。

## 预备知识

系统调用是指使用类似函数调用的方式调用操作系统提供的API。虽然从概念上来说系统调用和函数调用差不多，但本质上它们有很大的不同（call vs int/syscall）

1. 操作系统的代码位于内核地址空间，而CPU在执行用户代码时特权等级很低，无权访问需要最高优先级才能访问的内核地址空间的代码和数据，所以**不能通过简单的call指令直接调用操作系统提供的函数**，而需要使用特殊的指令进入操作系统内核完成指定的功能。
2. 用户代码调用操作系统API不是根据函数名直接调用，而是需要根据操作系统为每个API提供的一个**整型编号**来调用，AMD64 Linux平台约定在进行系统调用时使用rax寄存器存放系统调用编号（PS：还有专门的寄存器），同时约定使用rdi, rsi, rdx, r10, r8和r9来传递前6个系统调用参数。

线程调度：操作系统什么时候会发起调度呢？总体来说操作系统必须要得到CPU的控制权后才能发起调度，那么**当用户程序在CPU上运行时如何才能让CPU去执行操作系统代码从而让内核获得控制权呢？**一般说来在两种情况下会从执行用户程序代码转去执行操作系统代码：
1. 用户程序使用系统调用进入操作系统内核；
2. 硬件中断。硬件中断处理程序由操作系统提供，所以当硬件发生中断时，就会执行操作系统代码。硬件中断有个特别重要的时钟中断，这是操作系统能够发起抢占调度的基础。

## 与函数的关系

20 世纪 60 年代高德纳（Donald Ervin Knuth）总结两种子过程（Subroutine）：一种是我们常见的函数调用的方式，而另一种就是协程。和函数的区别是，函数调用时，调用者跟被调用者之间像是一种上下级的关系；而在协程中，调用者跟被调用者更像是互相协作的关系，比如一个是生产者，一个是消费者。

和函数的区别是，函数调用时，调用者跟被调用者之间像是一种上下级的关系；当我们使用函数的时候，**简单地保持一个调用栈就行了**。当 fun1 调用 fun2 的时候，就往栈里增加一个新的栈帧，用于保存 fun2 的本地变量、参数等信息；这个函数执行完毕的时候，fun2 的栈帧会被弹出（恢复栈顶指针 sp），并跳转到返回地址（调用 fun2 的下一条指令），继续执行调用者 fun1 的代码。

而在协程中，调用者跟被调用者更像是互相协作的关系，比如一个是生产者，一个是消费者。如果调用的是协程 coroutine1，该怎么处理协程的栈帧呢？因为协程并没有执行完，显然还不能把它简单地丢掉。这种情况下，程序可以从堆里申请一块内存，保存协程的活动记录，包括本地变量的值、程序计数器的值（当前执行位置）等等。这样，当下次再激活这个协程的时候，可以在栈帧和寄存器中恢复这些信息。

1. Stackful Coroutine，每个协程，都有一个自己专享的协程栈。可以在协程栈的任意一级，暂停协程的运行。可以从一个线程脱离，附加到另一个线程上。
2. Stackless Coroutine，在主栈上运行协程的机制，会被绑定在创建它的线程上

## 初始化和调度循环

在主线程第一次被调度起来执行第一条指令之前，主线程的函数栈如下图所示：

![](/public/upload/go/go_scheduler_thread_init.jpg)

初始化全局变量g0，g0的主要作用是提供一个栈供runtime代码执行。 PS：代码执行就得有一个栈结构存在？

![](/public/upload/go/go_scheduler_g0.jpg)

把m0和g0绑定在一起，这样，之后在主线程中通过get_tls可以获取到g0，通过g0的m成员又可以找到m0，于是这里就实现了m0和g0与主线程之间的关联。

![](/public/upload/go/go_scheduler_m0.jpg)

初始化p和allp

![](/public/upload/go/go_scheduler_p.jpg)

创建 main goroutine。多了一个我们称之为newg的g结构体对象，该对象也已经获得了从堆上分配而来的2k大小的栈空间，newg的stack.hi和stack.lo分别指向了其栈空间的起止位置。

![](/public/upload/go/go_scheduler_newg.jpg)

调整newg的栈空间，把goexit函数的第二条指令的地址入栈，伪造成goexit函数调用了fn，从而使fn执行完成后执行ret指令时返回到goexit继续执行完成最后的清理工作；重新设置newg.buf.pc 为需要执行的函数的地址，即fn，我们这个场景为runtime.main函数的地址。修改newg的状态为_Grunnable并把其放入了运行队列

![](/public/upload/go/go_scheduler_newg_runnable.jpg)

gogo函数也是通过汇编语言编写的，这里之所以需要使用汇编，是因为goroutine的调度涉及不同执行流之间的切换，前面我们在讨论操作系统切换线程时已经看到过，执行流的切换从本质上来说就是**CPU寄存器以及函数调用栈的切换**（PS：栈的切换没有在之前的意识里），然而不管是go还是c这种高级语言都无法精确控制CPU寄存器的修改，因而高级语言在这里也就无能为力了，只能依靠汇编指令来达成目的。

1. 保存g0的调度信息，主要是保存CPU栈顶寄存器SP到g0.sched.sp成员之中；
2. 调用schedule函数寻找需要运行的goroutine，我们这个场景找到的是main goroutine;
3. 调用gogo函数首先从g0栈切换到main goroutine的栈，然后从main goroutine的g结构体对象之中取出sched.pc的值并使用JMP指令跳转到该地址去执行；
4. main goroutine执行完毕直接调用exit系统调用退出进程。


入口函数是runtime.main，runtime.main函数主要工作流程如下：

1. 启动一个sysmon系统监控线程，该线程负责整个程序的gc、抢占调度以及netpoll等功能的监控
2. 执行runtime包的初始化；
3. 执行main包以及main包import的所有包的初始化；
4. 执行main.main函数；
5. 从main.main函数返回后调用exit系统调用退出进程；

非main goroutine执行完成后就会返回到goexit继续执行，而main goroutine执行完成后整个进程就结束了，这是main goroutine与其它goroutine的一个区别。


```go
func g2(n int, ch chan int) {
    ch <- n*n
}
func main() {
    ch := make(chan int)
    go g2(100, ch)
    fmt.Println(<-ch)
}
```

```
schedule()->execute()->gogo()->用户协程->goexit()->goexit1()->mcall()->goexit0()->schedule()
```

一轮调度是从调用schedule函数开始的，然后经过一系列代码的执行到最后又再次通过调用schedule函数来进行新一轮的调度，从一轮调度到新一轮调度的这一过程我们称之为一个调度循环，这里说的调度循环是指某一个工作线程的调度循环，而同一个Go程序中可能存在多个工作线程，每个工作线程都有自己的调度循环，也就是说每个工作线程都在进行着自己的调度循环。

![](/public/upload/go/go_scheduler_cycle.jpg)

## 调度策略

所谓的goroutine调度，是指程序代码按照一定的算法在适当的时候挑选出合适的goroutine并放到CPU上去运行的过程。这句话揭示了调度系统需要解决的三大核心问题：

1. 调度时机：什么时候会发生调度？
    1. 被动调度，goroutine执行某个操作因条件不满足需要等待而发生的调度，比如读取channel
    2. 主动调度，正在运行的goroutine通过直接调用runtime.Gosched()函数暂时放弃运行而发生的调度。是用户代码自己控制的
    3. 抢占调度。见下文
2. 调度策略：使用什么策略来挑选下一个进入运行的goroutine？
3. 切换机制：如何把挑选出来的goroutine放到CPU上运行？

抢占调度

1. 因运行时间过长而导致的抢占调度。监控线程负责给被抢占的goroutine设置抢占标记，被抢占的goroutine再在函数的的入口处检查g的stackguard0成员决定是否需要调用morestack_noctxt函数，从而最终调用到newstack函数处理抢占请求；
2. 因进入系统调用时间过长而发生的抢占调度。而对于系统调用执行时间过长的goroutine，调度器并没有暂停其执行，只是剥夺了正在执行系统调用的工作线程所绑定的p，要等到工作线程从系统调用返回之后绑定p失败的情况下该goroutine才会真正被暂停运行。
