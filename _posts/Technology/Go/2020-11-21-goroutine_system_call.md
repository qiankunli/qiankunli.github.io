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

Please remember that at the end of the day, all programs that work on UNIX machines end up using C system calls to communicate with the UNIX kernel and perform most of their tasks. 所有在 UNIX 系统上运行的程序最终都会通过 C 系统调用来和内核打交道。用其他语言编写程序进行系统调用，方法不外乎两个：一是自己封装，二是依赖 glibc、或者其他的运行库。Go 语言选择了前者，把系统调用都封装到了 syscall 包。封装时也同样得通过汇编实现。

当M一旦进入系统调用后，会脱离go runtime的控制。试想万一系统调用阻塞了呢，此时又无法进行抢占，是不是整个M也就罢工了。所以为了维持整个调度体系的高效运转，必然要在进入系统调用之前要做点什么以防患未然。
1. 异步系统调用 G 会和MP分离（G挂到netpoller）
2. 同步系统调用 MG 会和P分离（P另寻M），当M从系统调用返回时，不会继续执行，而是将G放到run queue。

生动的说明了GPM相对GM的精妙之处。

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

梳理：如果 G 被阻塞在某个 channel 操作或网络 I/O 操作上时，G 会被放置到某个等待（wait）队列中，而 M 会尝试运行 P 的下一个可运行的 G。如果这个时候 P 没有可运行的 G 供 M 运行，那么 M 将解绑 P，并进入挂起状态。当 I/O 操作完成或 channel 操作完成，在等待队列中的 G 会被唤醒，标记为可运行（runnable），并被放入到某 P 的队列中，绑定一个 M 后继续执行。

### 同步系统调用

G1将进行同步系统调用以阻塞M1

![](/public/upload/go/go_scheduler_sync_systemcall_1.png)

调度器介入后：识别出G1已导致M1阻塞，此时，调度器将M1与P分离（**正因为M 和P 可能分离，所以mcache 挂在P上**），同时也将G1带走。然后调度器引入新的M2来服务P。

![](/public/upload/go/go_scheduler_sync_systemcall_2.png)

阻塞的系统调用完成后：G1可以移回 LRQ 并再次由P执行。如果这种情况需要再次发生，M1将被放在旁边以备将来使用。

![](/public/upload/go/go_scheduler_sync_systemcall_3.png)

梳理：如果 G 被阻塞在某个系统调用（system call）上，那么不光 G 会阻塞，执行这个 G 的 M 也会解绑 P，与 G 一起进入挂起状态。如果此时有空闲的 M，那么 P 就会和它绑定，并继续执行其他 G；如果没有空闲的 M，但仍然有其他 G 要去执行，那么 Go 运行时就会创建一个新 M（线程）。当系统调用返回后，阻塞在这个系统调用上的 G 会尝试获取一个可用的 P，如果没有可用的 P，那么 G 会被标记为 runnable，之前的那个挂起的 M 将再次进入挂起状态。

## sysmon 协程

Go 程序启动时，运行时会去启动一个名为 sysmon 的 M（一般称为监控线程），这个 M 的特殊之处在于它不需要绑定 P 就可以运行（以 g0 这个 G 的形式）。

```go
// $GOROOT/src/runtime/proc.go
// The main goroutine.
func main() {
     ... ...
    systemstack(func() {
        newm(sysmon, nil)
    })
    .... ...
}
// Always runs without a P, so write barriers are not allowed.
// go:nowritebarrierrec
func sysmon() {
    // If a heap span goes unused for 5 minutes after a garbage collection,
    // we hand it back to the operating system.
    scavengelimit := int64(5 * 60 * 1e9)
    ... ...
    if  .... {
        ... ...
        // retake P's blocked in syscalls and preempt long running G's
        if retake(now) != 0 {
            idle = 0
        } else {
            idle++
        }
       ... ...
    }
}
```

![](/public/upload/go/go_scheduler_sysmon.jpg)

在 linux 内核中有一些执行定时任务的线程, 比如定时写回脏页的 pdflush, 定期回收内存的 kswapd0, 以及每个 cpu 上都有一个负责负载均衡的 migration 线程等.在 go 运行时中也有类似的协程 sysmon. sysmon 运行在 M，且不需要 P。它会每隔一段时间**检查 Go 语言runtime**，确保程序没有进入异常状态。

sysmon 每 20us~10ms 启动一次，功能比较多: 

1. 检查死锁runtime.checkdead 
2. 运行计时器 — 获取下一个需要被触发的计时器；
3. 将长时间未处理的 netpoll 结果添加到任务队列；
4. 向长时间运行的 G 任务发出抢占调度（retake 方法）；[Go 的抢占式调度](https://mp.weixin.qq.com/s/d7FdGBc0S0V3S4aRL4EByA)当 sysmon 发现 M 已运行同一个 G（Goroutine）10ms 以上时，它会将该 G 的内部参数 preempt 设置为 true。然后，在函数序言中，当 G 进行函数调用时，G 会检查自己的 preempt 标志，如果它为 true，则它将自己与 M 分离并推入“全局队列”。由于它的工作方式（函数调用触发），在 `for{}` 的情况下并不会发生抢占，如果没有函数调用，即使设置了抢占标志，也不会进行该标志的检查。Go1.14 引入抢占式调度（使用信号的异步抢占机制），sysmon 仍然会检测到运行了 10ms 以上的 G（goroutine）。然后，sysmon 向运行 G 的 P 发送信号（SIGURG）。Go 的信号处理程序会调用P上的一个叫作 gsignal 的 goroutine 来处理该信号，将其映射到 M 而不是 G，并使其检查该信号。gsignal 看到抢占信号，停止正在运行的 G。
6. 打印调度信息,归还内存等定时任务.
7. 释放闲置超过 5 分钟的 span 内存；如果超过 2 分钟没有垃圾回收，强制执行；
7. 收回因 syscall 长时间阻塞的 P；



