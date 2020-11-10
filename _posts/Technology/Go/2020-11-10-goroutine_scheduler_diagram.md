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

本文内容来自 张万波大佬 [go语言调度器源代码情景分析](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/mp/homepage%3F__biz%3DMzU1OTg5NDkzOA%3D%3D%26hid%3D1%26sn%3D8fc2b63f53559bc0cee292ce629c4788%26scene%3D25%23wechat_redirect)。在此再次表达 对大佬的膜拜。

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
schedule()->execute()->gogo()->g2()->goexit()->goexit1()->mcall()->goexit0()->schedule()
```

一轮调度是从调用schedule函数开始的，然后经过一系列代码的执行到最后又再次通过调用schedule函数来进行新一轮的调度，从一轮调度到新一轮调度的这一过程我们称之为一个调度循环，这里说的调度循环是指某一个工作线程的调度循环，而同一个Go程序中可能存在多个工作线程，每个工作线程都有自己的调度循环，也就是说每个工作线程都在进行着自己的调度循环。

![](/public/upload/go/go_scheduler_cycle.jpg)



所谓的goroutine调度，是指程序代码按照一定的算法在适当的时候挑选出合适的goroutine并放到CPU上去运行的过程。这句话揭示了调度系统需要解决的三大核心问题：

1. 调度时机：什么时候会发生调度？
2. 调度策略：使用什么策略来挑选下一个进入运行的goroutine？
3. 切换机制：如何把挑选出来的goroutine放到CPU上运行？

对这三大问题的解决构成了调度器的所有工作，因而我们对调度器的分析也必将围绕着它们所展开。
