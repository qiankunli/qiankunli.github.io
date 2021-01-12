---

layout: post
title: golang 系统调用与阻塞处理
category: 技术
tags: Go
keywords: Go goroutine scheduler

---

## 前言

* TOC
{:toc}

异步系统调用 G 会和MP分离（G挂到netpoller），同步系统调用 GM 会和P分离（P另寻M），生动的说明了GPM相对GM的精妙之处。

## 阻塞

在 Go 里面阻塞主要分为以下 4 种场景：
1. 由于原子、互斥量或通道操作调用导致  Goroutine  阻塞，调度器将把当前阻塞的 Goroutine 切换出去，重新调度 LRQ 上的其他 Goroutine；
2. 由于网络请求和 IO 操作导致  Goroutine  阻塞。Go 程序提供了网络轮询器（NetPoller）来处理网络请求和 IO 操作的问题，其后台通过 kqueue（MacOS），epoll（Linux）或  iocp（Windows）来实现 IO 多路复用。通过使用 NetPoller 进行网络系统调用，调度器可以防止  Goroutine  在进行这些系统调用时阻塞 M。这可以让 M 执行 P 的  LRQ  中其他的  Goroutines，而不需要创建新的 M。执行网络系统调用不需要额外的 M，网络轮询器使用系统线程，它时刻处理一个有效的事件循环，有助于减少操作系统上的调度负载。用户层眼中看到的 Goroutine 中的“block socket”，实现了 goroutine-per-connection 简单的网络编程模式。实际上是通过 Go runtime 中的 netpoller 通过 Non-block socket + I/O 多路复用机制“模拟”出来的。
3. 当调用一些系统方法的时候（如文件 I/O），如果系统方法调用的时候发生阻塞，这种情况下，网络轮询器（NetPoller）无法使用，而进行系统调用的  G1  将阻塞当前 M1。调度器引入 其它M 来服务 M1 的P。
4. 如果在 Goroutine 去执行一个 sleep 操作，导致 M 被阻塞了。Go 程序后台有一个监控线程 sysmon，它监控那些长时间运行的 G 任务然后设置可以强占的标识符，别的 Goroutine 就可以抢先进来执行。

## 系统调用

Go 语言通过 Syscall 和 Rawsyscall 等使用汇编语言编写的方法封装了操作系统提供的所有系统调用，其中 Syscall 在 Linux 386 上的实现如下：

```
TEXT ·Syscall(SB),NOSPLIT,$0-28
    CALL	runtime·entersyscall(SB)
    MOVL	trap+0(FP), AX	// syscall entry
    MOVL	a1+4(FP), BX
    MOVL	a2+8(FP), CX
    MOVL	a3+12(FP), DX
    MOVL	$0, SI
    MOVL	$0, DI
    INVOKE_SYSCALL
    CMPL	AX, $0xfffff001
    JLS	ok
    MOVL	$-1, r1+16(FP)
    MOVL	$0, r2+20(FP)
    NEGL	AX
    MOVL	AX, err+24(FP)
    CALL	runtime·exitsyscall(SB)
    RET
ok:
    MOVL	AX, r1+16(FP)
    MOVL	DX, r2+20(FP)
    MOVL	$0, err+24(FP)
    CALL	runtime·exitsyscall(SB)
    RET
```

[Golang - 调度剖析](https://segmentfault.com/a/1190000016611742)

[Go: Goroutine, OS Thread and CPU Management](https://medium.com/a-journey-with-go/go-goroutine-os-thread-and-cpu-management-2f5a5eaf518a) Go optimizes the system calls — whatever it is blocking or not — by wrapping them up in the runtime. This wrapper will automatically **dissociate** the P from the thread M and allow another thread to run on it.

### 异步系统调用

通过使用网络轮询器进行网络系统调用，调度器可以防止 Goroutine 在进行这些系统调用时阻塞M。这可以让M执行P的 LRQ 中其他的 Goroutines，而**不需要创建新的M**。有助于减少操作系统上的调度负载。

G1正在M上执行，还有 3 个 Goroutine 在 LRQ 上等待执行

![](/public/upload/go/go_scheduler_async_systemcall_1.png)

接下来，G1想要进行网络系统调用，因此它被移动到网络轮询器并且处理异步网络系统调用。然后，M可以从 LRQ 执行另外的 Goroutine。

![](/public/upload/go/go_scheduler_async_systemcall_2.png)

最后：异步网络系统调用由网络轮询器完成，G1被移回到P的 LRQ 中。一旦G1可以在M上进行上下文切换，它负责的 Go 相关代码就可以再次执行。

![](/public/upload/go/go_scheduler_async_systemcall_3.png)

### 同步系统调用

G1将进行同步系统调用以阻塞M1

![](/public/upload/go/go_scheduler_sync_systemcall_1.png)

调度器介入后：识别出G1已导致M1阻塞，此时，调度器将M1与P分离，同时也将G1带走。然后调度器引入新的M2来服务P。

![](/public/upload/go/go_scheduler_sync_systemcall_2.png)

阻塞的系统调用完成后：G1可以移回 LRQ 并再次由P执行。如果这种情况需要再次发生，M1将被放在旁边以备将来使用。

![](/public/upload/go/go_scheduler_sync_systemcall_3.png)


## sysmon 协程

![](/public/upload/go/go_scheduler_sysmon.jpg)

在 linux 内核中有一些执行定时任务的线程, 比如定时写回脏页的 pdflush, 定期回收内存的 kswapd0, 以及每个 cpu 上都有一个负责负载均衡的 migration 线程等.在 go 运行时中也有类似的协程 sysmon. 它会每隔一段时间**检查 Go 语言runtime**，确保程序没有进入异常状态。


系统监控的触发时间就会稳定在 10ms，功能比较多: 

1. 检查死锁runtime.checkdead 
2. 运行计时器 — 获取下一个需要被触发的计时器；
3. 定时从 netpoll 中获取 ready 的协程
4. 抢占运行时间较长的或者处于系统调用的 Goroutine；基本流程是 sysmon 协程标记某个协程运行过久, 需要切换出去, 该协程在运行函数时会检查栈标记, 然后进行切换.
5. 在满足条件时触发垃圾收集回收内存；
6. 打印调度信息,归还内存等定时任务.

