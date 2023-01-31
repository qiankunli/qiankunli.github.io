---

layout: post
title: golang性能分析及优化
category: 技术
tags: Go
keywords: Go io

---

## 前言

* TOC
{:toc}

我们或许都有这样的体会，自己思考明白，设计出来的程序，可以很清晰明了的将细节解释明白，对功能的增删改也是可以做到灵活应对。可是让我们一下子去修改别人写的功能或模块的时候，很多时候会一脸懵逼，这也不敢动，那也不敢动，在不理解的情况下，有疑问，一定要问清楚原理和逻辑，否则搞不好就是线上问题。如上情况，最重要的一个原因就是自己对当前模块/功能的熟悉程度，以及自己的**思维模型是否可迁移**。


## 性能优化流程

1. 理清待优化代码的常用逻辑与场景
2. 根据实际场景编写压测用例
3. 使用pprof 或者火焰图等工具取得数据
4. 找到热点代码重点优化

有两种类型的 profiler ：
1. 追踪型：任何时候触发提前设定的事件就会做测量，例如：函数调用，函数退出，等等
2. 采样型：常规时间间隔做一次测量

Go CPU profiler 是一个采样型的 profiler。也有一个追踪型的 profiler，Go 执行追踪器，用来追踪特定事件像请求锁，GC 相关的事件，等等。

采样型 profiler 通常包含两个主要部分：

1. 采样器：一个在时间间隔触发的回调，一个堆栈信息一般会被收集成 profiling data。不同的 profiler 用不同的策略去触发回调。
2. 数据收集：这个是 profiler 收集数据的地方：它可能是内存占用或者是调用统，基本上跟堆栈追踪相关的数据

## Profiling

pprof 是用于可视化和分析性能分析数据的工具。

![](/public/upload/go/pprof.png)

Profiling 这个词比较难翻译，一般译成画像。比如在案件侦破的时候会对嫌疑人做画像，从犯罪现场的种种证据，找到嫌疑人的各种特征，方便对嫌疑人进行排查；还有就是互联网公司会对用户信息做画像，通过了解用户各个属性（年龄、性别、消费能力等），方便为用户推荐内容或者广告。在计算机性能调试领域里，profiling 就是对应用的画像，这里画像就是应用使用 CPU 和内存的情况。也就是说应用使用了多少 CPU 资源？都是哪些部分在使用？每个函数使用的比例是多少？有哪些函数在等待 CPU 资源？知道了这些，我们就能对应用进行规划，也能快速定位性能瓶颈。

CPU Profiling 是如何工作的？stack trace + statistics 的模型。当我们准备进行 CPU Profiling 时，通常需要选定某一**时间窗口**，在该窗口内，CPU Profiler 会向目标程序注册一个定时执行的 hook（有多种手段，譬如 SIGPROF 信号），在这个 hook 内我们每次会获取业务线程此刻的 **stack trace**。我们将 hook 的执行频率控制在特定的数值，譬如 100hz，这样就做到每 10ms 采集一个业务代码的调用栈样本。当时间窗口结束后，我们将采集到的所有样本进行聚合，最终得到每个函数被采集到的次数，相较于总样本数也就得到了每个函数的**相对占比**。借助此模型我们可以发现占比较高的函数，进而定位 CPU 热点。

Heap Profiling 也是stack trace + statistics 的模型。数据采集工作并非简单通过定时器开展，而是需要侵入到内存分配路径内，即直接将自己集成在内存分配器内，当应用程序进行内存分配时拿到当前的 stack trace，最终将所有样本聚合在一起，这样我们便能知道每个函数直接或间接地内存分配数量了。由于 Heap Profiling 也是采样的（默认每分配 512k 采样一次），所以**展示的内存大小要小于实际分配的内存大小**。同 CPU Profiling 一样，这个数值仅仅是用于计算**相对占比**，进而定位内存分配热点。

## 实现

[深究 Go CPU profiler](https://mp.weixin.qq.com/s/DRQWcU2dN-FycoyFZfnklA)在Linux中，Go runtime 使用setitimer/timer_create/timer_settime API 来设置 SIGPROF 信号处理器。这个处理器在runtime.SetCPUProfileRate 控制的周期内被触发，默认为100Mz（10ms）。一旦 pprof.StartCPUProfile 被调用，Go runtime 就会在特定的时间间隔产生SIGPROF 信号。内核向应用程序中的一个运行线程发送 SIGPROF 信号。**由于 Go 使用非阻塞式 I/O，等待 I/O 的 goroutines 不被计算为运行**，Go CPU profiler 不捕获这些。顺便提一下：这是实现 fgprof 的基本原因。fgprof 使用 runtime.GoroutineProfile来获得等待和非等待的 goroutines 的 profile 数据。

一旦一个随机运行的 goroutine 收到 SIGPROF 信号，它就会被中断，然后信号处理器的程序开始运行。被中断的 goroutine 的堆栈追踪在这个信号处理器的上下文中被检索出来，然后和当前的 profiler 标签一起被保存到一个无锁的日志结构中（每个捕获的堆栈追踪都可以和一个自定义的标签相关联，你可以用这些标签在以后做过滤）。这个特殊的无锁结构被命名为 profBuf ，它被定义在 runtime/profbuf.go 中，它是一个单一写、单一读的无锁环形缓冲 结构，与这里发表的结构相似。writer 是 profiler 的信号处理器，reader 是一个 goroutine(profileWriter)，定期读取这个缓冲区的数据，并将结果汇总到最终的 hashmap。这个最终的 hashmap 结构被命名为 profMap，并在 runtime/pprof/map.go中定义。PS：goroutine 堆栈信息 ==> sigProfHandler ==write==> profBuf ==read==> profWriter ==> profMap

## 如何看懂火焰图（以从下到上为例）

[如何看懂火焰图](https://cloud.tencent.com/developer/article/1873597)火焰图的调用顺序从下到上，每个方块代表一个函数，它上面一层表示这个函数会调用哪些函数，方块的大小代表了占用 CPU 使用的长短。火焰图的配色并没有特殊的意义，默认的红、黄配色是为了更像火焰而已。
1. 每一列代表一个调用栈，每一个格子代表一个函数
2. 纵轴展示了栈的深度，按照调用关系从下到上排列。最顶上格子代表采样时，正在占用 cpu 的函数。
3. 横轴的意义是指：火焰图将采集的多个调用栈信息，通过按字母横向排序的方式将众多信息聚合在一起。需要注意的是它并不代表时间。
4. 横轴格子的宽度代表其在采样中出现频率，所以一个格子的宽度越大，说明它是瓶颈原因的可能性就越大。

![](/public/upload/go/flame_graph.jpg)

总的来说

1. 颜色本身没有什么意义
2. 纵向表示调用栈的深度
3. 横向表示消耗的时间

## 实操

```go
import (
    ... ...
    "net/http"
    _ "net/http/pprof"  // 会自动注册handler到http server，方便通过http 接口获取程序运行采样报告
    ... ...
)
... ...
func main() {
    go func() {
        http.ListenAndServe(":6060", nil)
    }()
    ... ...
}

```
以空导入的方式导入 net/http/pprof 包，并在一个单独的 goroutine 中启动一个标准的 http 服务，就可以实现对 pprof 性能剖析的支持。pprof 工具可以通过 6060 端口采样到我们的 Go 程序的运行时数据。然后就可以通过 `http://192.168.10.18:6060/debug/pprof` 查看程序的采样信息，但是可读性非常差，需要借助pprof 的辅助工具来分析。


```sh
// 192.168.10.18为服务端的主机地址
$ go tool pprof -http=:9090 http://192.168.10.18:6060/debug/pprof/profile?seconds=30
Fetching profile over HTTP from http://192.168.10.18:6060/debug/pprof/profile
Saved profile in /Users/tonybai/pprof/pprof.server.samples.cpu.004.pb.gz
Serving web UI on http://localhost:9090
```

## trace

虽然CPU分析器做了一件很好的工作，告诉你什么函数占用了最多的CPU时间，但它并不能帮助你确定是什么阻止了goroutine运行。这里可能的原因：被syscall阻塞 、阻塞在共享内存(channel/mutex etc)、阻塞在运行时(如 GC)、甚至有可能是运行时调度器不工作导致的。这种问题使用pprof很难排查。

go tool trace 能够跟踪捕获各种执行中的事件，例如：
1. Goroutine 的创建/阻塞/解除阻塞。
2. Syscall 的进入/退出/阻止，GC 事件。
3. Heap 的大小改变。
4. Processor 启动/停止等等。

![](/public/upload/go/trace_view.jpg)

1. 时间线: 显示执行的时间单元 根据时间的纬度不同 可以调整区间
2. stats 区域
    1. 堆，显示执行期间内存的分配和释放情况（折线图）
    2. 协程(Goroutine)，显示每个时间点哪些Goroutine在运行 哪些goroutine等待调度 ，其包含 GC 等待（GCWaiting）、可运行（Runnable）、运行中（Running）这三种状态。
    3. Threads，显示在执行期间有多少个线程在运行，其包含正在调用 Syscall（InSyscall）、运行中（Running）这两种状态。
3. proc区域，虚拟处理器Processor：每个虚拟处理器显示一行，虚拟处理器的数量一般默认为系统内核数。数量由环境变量GOMAXPROCS控制。 分两层
    1. 上一层表示Processor上运行的goroutine的信息，一个P 某个时刻只能运行一个G，选中goroutine 可以查看特定时间点 特定goroutine的执行堆栈信息以及关联的事件信息
    2. 下一层表示processor附加的事件比如SysCall 或runtime system events

## 其它

Linux perf 使用 PMU（Performance Monitor Unit）计数器进行采样。你指示 PMU 在某些事件发生 N 次后产生一个中断。一个例子，可能是每 1000 个 CPU 时钟周期进行一次采样。一旦数据收集回调被定期触发，剩下的就是收集堆栈痕迹并适当地汇总。Linux perf 使用 `perf_event_open(PERF_SAMPLE_STACK_USER,...)` 来获取堆栈追踪信息。捕获的堆栈痕迹通过 mmap'd 环形缓冲区写到用户空间。