---

layout: post
title: Goroutine 调度模型
category: 技术
tags: Go
keywords: Go goroutine scheduler

---

## 前言

* TOC
{:toc}

[万字长文深入浅出 Golang Runtime](https://zhuanlan.zhihu.com/p/95056679)调度在计算机中是分配工作所需资源的方法，linux的调度为CPU找到可运行的线程，而Go的调度是为M（线程）找到P（内存、执行票据）和可运行的G。

## goroutine调度模型的四个抽象及其数据结构

goroutine调度模型4个重要结构，分别是M、G、P、Sched，前三个定义在runtime.h中，Sched定义在proc.c中。

- Sched结构就是调度器，它维护有存储M和G的队列（全局的）以及调度器的一些状态信息等。
- M代表内核级线程，一个M就是一个线程，goroutine就是跑在M之上 ；M是一个很大的结构，里面维护小对象内存cache（mcache）、当前执行的goroutine、随机数发生器等等非常多的信息。
- P全称是Processor，处理器，表示调度的上下文，它可以被看做一个运行于线程 M 上的本地调度器，所以它维护了一个goroutine队列（环形链表），里面存储了所有需要它来执行的goroutine。
- G就是goroutine实现的核心结构了，G维护了goroutine需要的栈、程序计数器以及它所在的M等信息。一个协程代表了一个执行流，执行流有需要执行的函数(startpc)，有函数的入参，有当前执行流的状态和进度(对应 CPU 的 PC 寄存器和 SP 寄存器)，当然也需要有保存状态的地方，用于执行流恢复。

`$GOROOT/src/runtime/runtime2.go`

![](/public/upload/go/go_scheduler_object.png)

1. 结构体 g 的字段 atomicstatus 就存储了当前 Goroutine 的状态，可选值为
    1. _Gidle, 刚刚被分配并且还没有被初始化
    2. _Grunnable, 没有执行代码、没有栈的所有权、存储在运行队列中 
    3. _Grunning, 可以执行代码、拥有栈的所有权，被赋予了内核线程 M 和处理器 P
    4. _Gsyscall, 正在执行系统调用、拥有栈的所有权、没有执行用户代码，被赋予了内核线程 M 但是不在运行队列上
    5. _Gwaiting, 由于运行时而被阻塞，没有执行用户代码并且不在运行队列上，但是可能存在于 Channel 的等待队列上
    6. _Gdead, 	没有被使用，没有执行代码，可能有分配的栈
    7. _Gcopystack, 栈正在被拷贝、没有执行代码、不在运行队列上
2. 虽然 Goroutine 在运行时中定义的状态非常多而且复杂，但是我们可以将这些不同的状态聚合成最终的三种：等待中(比如正在执行系统调用或同步操作)、可运行、运行中（占用M），在运行期间我们会在这三种不同的状态来回切换。
3. runhead、runqtail、runq 以及 runnext 等字段表示P持有的运行队列，该运行队列是一个使用数组构成的环形链表，其中最多能够存储 256 个指向Goroutine 的指针，除了 runq 中能够存储待执行的 Goroutine 之外，runnext 指向的 Goroutine 会成为下一个被运行的 Goroutine
4. p 结构体中的状态 status 可选值

    1. _Pidle	处理器没有运行用户代码或者调度器，运行队列为空
    2. _Prunning	被线程 M 持有，并且正在执行用户代码或者调度器
    3. _Psyscall	没有执行用户代码，当前线程陷入系统调用
    4. _Pgcstop	被线程 M 持有，当前处理器由于垃圾回收被停止
    5. _Pdead	当前处理器已经不被使用

## 调度模型的演化

### GM模型

go1.1 之前都是该模型

![](/public/upload/go/go_scheduler_gm.jpg)

### GPM模型

几个问题

1. 为什么引入Processor 的概念
2. 为什么把全局队列打散. 对该队列的操作均需要竞争同一把锁, 导致伸缩性不好.
新生成的协程也会放入全局的队列, 大概率是被其他 m运行了, 内存亲和性不好. 
3. mcache 为什么跟随 P。 参见[内存管理](http://qiankunli.github.io/2020/01/28/memory_management.html) 了解mcache
4. 为什么 P 的个数默认是 CPU 核数: Go 尽量提升性能, 那么在一个 n 核机器上, 如何能够最大利用 CPU 性能呢? 当然是同时有 n 个线程在并行运行中, 把 CPU 喂饱, 即所有核上一直都有代码在运行.

![](/public/upload/go/go_scheduler_gpm.jpg)

![](/public/upload/go/go_scheduler_goroutine_status.jpg)

## 函数运行

[Go 语言设计与实现 Goroutine](https://draveness.me/golang/docs/part3-runtime/ch06-concurrency/golang-goroutine/)

`$GOROOT/src/runtime/proc.go`

### goroutine 创建

go 关键字在编译期间通过 stmt 和 call 两个方法将该关键字转换成 newproc 函数调用，代码的路径和原理与 defer 关键字几乎完全相同。

我们向 newproc 中传入一个表示函数的指针 funcval，在这个函数中我们还会获取当前调用 newproc 函数的 Goroutine 以及调用方的程序计数器 PC，然后调用 newproc1 函数：

    func newproc(siz int32, fn *funcval) {
        argp := add(unsafe.Pointer(&fn), sys.PtrSize)
        gp := getg()
        pc := getcallerpc()
        newproc1(fn, (*uint8)(argp), siz, gp, pc)
    }

newproc1 函数的主要作用就是创建一个运行传入参数 fn 的 g 结构体，并对其各个成员赋值。

    func newproc1(fn *funcval, argp *uint8, narg int32, callergp *g, callerpc uintptr) {
        _g_ := getg()
        siz := narg
        siz = (siz + 7) &^ 7
        _p_ := _g_.m.p.ptr()
        // 获取或创建一个 g struct
        newg := gfget(_p_)
        if newg == nil {
            newg = malg(_StackMin)
            casgstatus(newg, _Gidle, _Gdead)
            allgadd(newg)
        }
        // 获取新创建 Goroutine 的堆栈并直接通过 memmove 将函数 fn 需要的参数全部拷贝到栈中
        totalSize := 4*sys.RegSize + uintptr(siz) + sys.MinFrameSize
        totalSize += -totalSize & (sys.SpAlign - 1)
        sp := newg.stack.hi - totalSize
        spArg := sp
        if narg > 0 {
            memmove(unsafe.Pointer(spArg), unsafe.Pointer(argp), uintptr(narg))
        }
        // 初始化新 Goroutine 的栈指针、程序计数器、调用方程序计数器等属性
        memclrNoHeapPointers(unsafe.Pointer(&newg.sched), unsafe.Sizeof(newg.sched))
        newg.sched.sp = sp
        newg.stktopsp = sp
        newg.sched.pc = funcPC(goexit) + sys.PCQuantum
        newg.sched.g = guintptr(unsafe.Pointer(newg))
        gostartcallfn(&newg.sched, fn)
        newg.gopc = callerpc
        newg.startpc = fn.fn
        if isSystemGoroutine(newg, false) {
            atomic.Xadd(&sched.ngsys, +1)
        }
        // 将新 Goroutine 的状态从 _Gdead 切换成 _Grunnable 并设置 Goroutine 的标识符（goid）
        casgstatus(newg, _Gdead, _Grunnable)

        newg.goid = int64(_p_.goidcache)
        _p_.goidcache++
        // runqput 函数会将新的 Goroutine 添加到处理器 P 的运行队列上
        runqput(_p_, newg, true)
        // 如果符合条件，当前函数会通过 wakep 来添加一个新的 p 结构体来执行 Goroutine
        if atomic.Load(&sched.npidle) != 0 && atomic.Load(&sched.nmspinning) == 0 && mainStarted {
            wakep()
        }
    }

### 协程切换入口——gopark

[从源码角度看 Golang 的调度](https://studygolang.com/articles/20651)

![](/public/upload/go/go_scheduler_sequence.png)

协程切换的原因一般有以下几种情况：


1. 系统调用；Go 语言通过 Syscall 和 Rawsyscall 等使用汇编语言编写的方法封装了操作系统提供的所有系统调用
2. 同步和编配；如果原子、互斥量或通道操作调用将导致 Goroutine 阻塞，调度器可以将之切换到一个新的 Goroutine 去运行。一旦 Goroutine 可以再次运行，它就可以重新排队，并最终在M上切换回来。
3. 抢占式调度时间片结束；
4. 垃圾回收

**所有触发 Goroutine 调度的方式最终都会调用 gopark 函数让出当前处理器 P 的控制权**。就好像linux 进程会主动调用schedule() 触发调度让出cpu 控制权，只是linux 多了时间片中断主动触发调度而已。


    func gopark(unlockf func(*g, unsafe.Pointer) bool, lock unsafe.Pointer, reason waitReason, traceEv byte, traceskip int) {
        mp := acquirem()
        gp := mp.curg
        mp.waitlock = lock
        mp.waitunlockf = unlockf
        gp.waitreason = reason
        mp.waittraceev = traceEv
        mp.waittraceskip = traceskip
        releasem(mp)
        mcall(park_m)
    }


gopark 函数中会更新当前处理器(mp)的状态并在处理器上设置该 Goroutine 的等待原因。gopark中调用的 park_m 函数会将当前 Goroutine 的状态从 _Grunning 切换至 _Gwaiting 并调用 waitunlockf 函数进行解锁

    func park_m(gp *g) {
        _g_ := getg()

        casgstatus(gp, _Grunning, _Gwaiting)
        dropg()

        if fn := _g_.m.waitunlockf; fn != nil {
            ok := fn(gp, _g_.m.waitlock)
            _g_.m.waitunlockf = nil
            _g_.m.waitlock = nil
            if !ok {
                casgstatus(gp, _Gwaiting, _Grunnable)
                execute(gp, true) // Schedule it back, never returns.
            }
        }
        schedule()
    }

在大多数情况下都会调用 schedule 触发一次 Goroutine 调度，这个函数的主要作用就是从不同的地方查找待执行的 Goroutine：

    func schedule() {
        _g_ := getg()

    top:
        var gp *g
        var inheritTime bool

        // 有一定几率会从全局的运行队列中选择一个 Goroutine；
        if gp == nil {
            if _g_.m.p.ptr().schedtick%61 == 0 && sched.runqsize > 0 {
                lock(&sched.lock)
                gp = globrunqget(_g_.m.p.ptr(), 1)
                unlock(&sched.lock)
            }
        }
        // 从当前处理器本地的运行队列中查找待执行的 Goroutine；
        if gp == nil {
            gp, inheritTime = runqget(_g_.m.p.ptr())
            if gp != nil && _g_.m.spinning {
                throw("schedule: spinning with local work")
            }
        }
        // 尝试从其他处理器上取出一部分 Goroutine，如果没有可执行的任务就会阻塞直到条件满足；
        if gp == nil {
            gp, inheritTime = findrunnable() // blocks until work is available
        }

        execute(gp, inheritTime)
    }

findrunnable 函数会再次从本地运行队列、全局运行队列、网络轮询器和其他的处理器中获取待执行的任务，该方法一定会返回待执行的 Goroutine，否则就会一直阻塞。

获取可以执行的任务之后就会调用 execute 函数执行该 Goroutine，执行的过程中会先将其状态修改成 _Grunning、与线程 M 建立起双向的关系并调用 gogo 触发调度。

    func execute(gp *g, inheritTime bool) {
        _g_ := getg()

        casgstatus(gp, _Grunnable, _Grunning)
        gp.waitsince = 0
        gp.preempt = false
        gp.stackguard0 = gp.stack.lo + _StackGuard
        if !inheritTime {
            _g_.m.p.ptr().schedtick++
        }
        // 与线程 M 建立起双向的关系
        _g_.m.curg = gp
        gp.m = _g_.m


        gogo(&gp.sched)
    }

gogo 在不同处理器架构上的实现都不相同，但是不同的实现其实也大同小异，下面是该函数在 386 架构上的实现：

    TEXT runtime·gogo(SB), NOSPLIT, $8-4
        MOVL	buf+0(FP), BX		// gobuf
        MOVL	gobuf_g(BX), DX
        MOVL	0(DX), CX		// make sure g != nil
        get_tls(CX)
        MOVL	DX, g(CX)
        MOVL	gobuf_sp(BX), SP	// restore SP
        MOVL	gobuf_ret(BX), AX
        MOVL	gobuf_ctxt(BX), DX
        MOVL	$0, gobuf_sp(BX)	// clear to help garbage collector
        MOVL	$0, gobuf_ret(BX)
        MOVL	$0, gobuf_ctxt(BX)
        MOVL	gobuf_pc(BX), BX
        JMP	BX

这个函数会从 gobuf 中取出 Goroutine 指针、栈指针、返回值、上下文以及程序计数器并将通过 JMP 指令跳转至 Goroutine 应该继续执行代码的位置。

## sysmon 协程

![](/public/upload/go/go_scheduler_sysmon.jpg)

在 linux 内核中有一些执行定时任务的线程, 比如定时写回脏页的 pdflush, 定期回收内存的 kswapd0, 以及每个 cpu 上都有一个负责负载均衡的 migration 线程等.在 go 运行时中也有类似的协程, sysmon.功能比较多: 定时从 netpoll 中获取 ready 的协程, 进行抢占, 定时 GC,打印调度信息,归还内存等定时任务.

协作式抢占：基本流程是 sysmon 协程标记某个协程运行过久, 需要切换出去, 该协程在运行函数时会检查栈标记, 然后进行切换. PS： 有点类似linux 的时间片中断。

## 系统调用

Go 语言通过 Syscall 和 Rawsyscall 等使用汇编语言编写的方法封装了操作系统提供的所有系统调用，其中 Syscall 在 Linux 386 上的实现如下：

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

[Golang - 调度剖析](https://segmentfault.com/a/1190000016611742)

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

## 补充

笔者今日学习Joe Armstrong的博士论文《面对软件错误构建可靠的分布式系统》，文中提到“在构建可容错软件系统的过程中要解决的本质问题就是故障隔离。”操作系统进程本身就是一种天然的故障隔离机制，当然从另一个层面，进程间还是因为共享cpu和内存等原因相互影响。进程要想达到容错性，就不能与其他进程有共享状态；它与其他进程的唯一联系就是由内核消息系统传递的消息。 

## 参考文献


[goroutine与调度器](http://blog.csdn.net/chanshimudingxi/article/details/40855467)