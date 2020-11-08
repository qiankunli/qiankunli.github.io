---

layout: post
title: Go并发机制及语言层工具
category: 技术
tags: Go
keywords: Go Concurrence

---

## 一 前言

本文算是对《Go并发编程实战》一书的小结。

我们谈go的优点时，并发编程是最重要的一块。因为go基于新的并发编程模型：不用共享内存的方式来通信，作为替代，以通信作为手段来共享内存。（goroutine共享channel，名为管道，实为内存）。

为解释这个优势，本文提出了四个概念：交互方式、手段、类型、目的（不一定对，只是为了便于描述）。并在不同的并发粒度上（进程、线程、goroutine）对这几个概念进行了梳理。

## 几个概念

|不同层面的并行化支持|表现|
|---|---|
|硬件并行化|多核心、多cpu|
|操作系统并行化|并行抽象（进程、线程）以及交互手段的提供|
|编程语言并行化|java启动一个线程要extends thread，而go只需一个`go func(){}`|

并发编程思想来自多任务操作系统，随后，人们逐渐将并发编程思想凝练成理论，同时开发出了一套关于它的描述方法。之后，人们把这套理论融入到编程语言当中。

并发程序内部有多个串行程序，串行程序之间有交互的需求，**交互方式**有同步和异步；不同的方式有各自的**交互手段**（整体分为共享内存和通讯两个**交互类型**）；**交互目的**分为：互斥访问共享资源、协调进度。

交互目的中，比较重要的是对共享资源的访问。这个共享资源，对进程是文件等，对线程是共享内存等。共享就容易发生干扰，<font color="red">一切问题的起点是：OS为了支持并发执行，会中断进程</font>==> 中断前要保存现场，中断后要恢复现场。**这个现场小了说是寄存器数据，大了说，就是进程关联的所有资源的状态（比如进程打开的fd）** ==> 要确保进程“休息”期间，别的进程不能“动它的奶酪”，否则，现场就被破坏了。==> 解决办法有以下几个：

1. 访问共享资源的操作不能被中断（原子操作）；
2. 可中断，但资源互斥访问（临界区）;
3. 资源本身就是不变的（比如常量）

临界区为什么不都弄成原子操作呢？因为一个操作执行起来没办法中断，也意味着很大的风险。所以，内核只提供针对二进制位和整数的原子操作。

交互类型中，通信比共享内存要简单。因为，把数据放在共享内存区供多个线程访问，这种方式的基本思想非常简单，却使并发访问控制变得复杂，要做好各种约束和限制，才能使看似简单的方法得以正确实施。比如，当线程离开临界区时，不仅要放弃对临界区的锁定（设置互斥量），还要通知其它等待进入该临界区的线程（操作条件变量）。**同步工具的引入（互斥量和条件变量等）增加了业务无关代码，其本身的正确使用也有一定的学习曲线**。而通讯就简单了，收到消息就往下走，收不到就等待，自得其乐，不用管其它人。

针对并发粒度的不同，我们把上述概念梳理一下：

|并发粒度|交互手段|同步/异步|交互类型|
|---|---|---|---|
|进程|管道、信号、socket|同步异步都有|只支持通信|
|线程|共享内存|1. 互斥量+条件变量 支持同步；2. 程序层面通过模拟signal弄出的futrue模式支持异步|只支持共享内存，高层抽象支持通信，比如java的blockingQueue|
|goroutine|channel|1. channel支持同步；2. 程序层面提供异步|只支持通信，高层抽象支持共享内存，比如go的sync包|

PS,routine is a set sequence of steps, part of larger computer program.


## 同步原语

[Go 语言设计与实现-同步原语与锁](https://draveness.me/golang/docs/part3-runtime/ch06-concurrency/golang-sync-primitives/)Go 语言在 sync 包中提供了用于同步的一些基本原语，包括常见的 sync.Mutex、sync.RWMutex、sync.WaitGroup、sync.Once 和 sync.Cond。这些基本原语提高了较为基础的同步功能，但是它们是一种相对原始的同步机制，在多数情况下，我们都应该使用**抽象层级的更高的** Channel 实现同步。

### Mutex

```go
type Mutex struct {
	state int32     // 表示当前互斥锁的状态，最低三位分别表示 mutexLocked、mutexWoken 和 mutexStarving，剩下的位置用来表示当前有多少个 Goroutine 等待互斥锁的释放
	sema  uint32
}
func (m *Mutex) Lock() {
	if atomic.CompareAndSwapInt32(&m.state, 0, mutexLocked) {
		return
    }
	m.lockSlow()
}
// src/runtime/sema.go
func semacquire1(addr *uint32, lifo bool, profile semaProfileFlags, skipframes int) {
    gp := getg()
    s := acquireSudog()
    ...
	for {
		// Add ourselves to nwait to disable "easy case" in semrelease.
		atomic.Xadd(&root.nwait, 1)
		...
		root.queue(addr, s, lifo)       // 加入等待队列
		goparkunlock(&root.lock, waitReasonSemacquire, traceEvGoBlockSync, 4+skipframes)    
		if s.ticket != 0 || cansemacquire(addr) {
			break
		}
	}
	...
	releaseSudog(s)
}
// src/runtime/proc.go
func park_m(gp *g) {
	_g_ := getg()
	casgstatus(gp, _Grunning, _Gwaiting)
	dropg()
	if fn := _g_.m.waitunlockf; fn != nil {
		ok := fn(gp, _g_.m.waitlock)
		...
		if !ok {
			casgstatus(gp, _Gwaiting, _Grunnable)       // 改变goroutine 状态
			execute(gp, true) 
		}
	}
	schedule()  // 触发调度器 调度
}
```

Mutex.Lock 有一个类似jvm 锁膨胀的过程（go 调度器运行在 用户态，因此实现比java synchronized 关键字更简单），Goroutine 会自旋、休眠自己，也会修改 mutex 的state

Goroutine修改自己的行为/状态

1. 锁空闲则加锁；
2. 锁占用  + 普通模式则执行 `sync.runtime_doSpin`进入自旋，执行30次PAUSE 指令消耗CPU时间；
3. 锁占用  + 饥饿模式则执行 `sync.runtime_SemacquireMutex`进入休眠状态

Goroutine修改 mutex 的状态

1. 如果当前 Goroutine 等待锁的时间超过了 1ms，当前 Goroutine 会将互斥锁切换到饥饿模式
2. 如果当前 Goroutine 是互斥锁上的最后一个等待的协程或者等待的时间小于 1ms，当前 Goroutine 会将互斥锁切换回正常模式；


### 读写锁

```go
var (
    pcodes         = make(map[string]string)
    mutex          sync.RWMutex
    ErrKeyNotFound = errors.New("Key not found in cache")
)
func Add(address, postcode string) {
    // 写入的时候要完全上锁
    mutex.Lock()
    pcodes[address] = postcode
    mutex.Unlock()
}
func Value(address string) (string, error) {
    // 读取的时候，只用读锁就可以
    mutex.RLock()
    pcode, ok := pcodes[address]
    mutex.RUnlock()
    if !ok {
        return "", ErrKeyNotFound
    }
    return pcode, nil
}
func main() {
    Add("henan", "453600")
    v, err := Value("henan")
    if err == nil {
        fmt.Println(v)
    }
}
```

### 超时

java 的`future.get(timeout)` 体现在channel 上是

```go
select {
    case ret:= <- ch: 
        xx
    case <- time.After(time.Second*1):
        fmt.Println("time out") 
}
```
java的Combine Future 体现在channel 上是
```go
select {
    case ret:= <- ch1: 
        xx
    case ret:= <- ch2: 
        xx
    default:
        fmt.Println("no one returned")
}
```

## 取消/中断goroutine 执行的工具——context

[深度解密Go语言之context](https://mp.weixin.qq.com/s/GpVy1eB5Cz_t-dhVC6BJNw)Go 1.7 标准库引入 context，中文译作“上下文”（其实这名字叫的不好）

在 Go 的 server 里，通常每来一个请求都会启动若干个 goroutine 同时工作：有些去数据库拿数据，有些调用下游接口获取相关数据……这些 goroutine 需要共享这个请求的基本数据，例如登陆的 token，处理请求的最大超时时间（如果超过此值再返回数据，请求方因为超时接收不到）等等。当请求被取消或超时，所有正在为这个请求工作的 goroutine 需要快速退出，因为它们的“工作成果”不再被需要了。context 包就是为了解决上面所说的这些问题而开发的：在 一组 goroutine 之间传递共享的值、取消信号、deadline……



[Go 语言设计与实现——上下文 Context](https://draveness.me/golang/docs/part3-runtime/ch06-concurrency/golang-context/)主要作用还是在多个 Goroutine 组成的树中同步取消信号以减少对资源的消耗和占用，**虽然它也有传值的功能，但是这个功能我们还是很少用到**。在真正使用传值的功能时我们也应该非常谨慎，使用 context.Context 进行传递参数请求的所有参数一种非常差的设计，比较常见的使用场景是传递请求对应用户的认证令牌以及用于进行分布式追踪的请求 ID。


### 为什么有 context？

一个goroutine启动后是无法控制它的，大部分情况是等待它自己结束，如何主动通知它结束呢？

```go
func main() {
	stop := make(chan bool)
	go func() {
		for {
			select {
			case <-stop:    // 有点类似Thread.interrupt() 的感觉
				fmt.Println("监控退出，停止了...")
				return
			default:
				fmt.Println("goroutine监控中...")
				time.Sleep(2 * time.Second)
			}
		}
	}()
	time.Sleep(10 * time.Second)
	fmt.Println("可以了，通知监控停止")
	stop<- true
}
```

`chan+select`是比较优雅的结束goroutine的方式，不过这种方式也有局限性，如果有很多goroutine都需要控制结束？如果这些goroutine又衍生了其他更多的goroutine怎么办呢？goroutine的关系链导致了这些场景非常复杂

```go
func main() {
	ctx, cancel := context.WithCancel(context.Background())
	go func(ctx context.Context) {
		for {
			select {
			case <-ctx.Done():
				fmt.Println("监控退出，停止了...")
				return
			default:
				fmt.Println("goroutine监控中...")
				time.Sleep(2 * time.Second)
			}
		}
	}(ctx)
	time.Sleep(10 * time.Second)
	fmt.Println("可以了，通知监控停止")
	cancel()    // context.WithCancel 返回的cancel 方法
}
```

### 父 goroutine 创建context

1. 根Context，通过`context.Background()/context.TODO()` 创建
2. 子Context
    ```go
    func WithCancel(parent Context) (ctx Context, cancel CancelFunc)
    // 和WithCancel差不多，会多传递一个截止时间参数，即到了这个时间点会自动取消Context，也可以通过cancel函数提前取消。
    func WithDeadline(parent Context, deadline time.Time) (Context, CancelFunc)    
    // 和WithDeadline基本上一样，多少时间后自动取消Context
    func WithTimeout(parent Context, timeout time.Duration) (Context, CancelFunc)   
    // 和取消Context无关，绑定了一个kv数据的Context，kv可以通过Context.Value方法访问到
    func WithValue(parent Context, key interface{}, val interface{}) Context    
    ```
3. 当前Context 被取消时，基于他的子context 都会被取消

Goroutine的创建和调用关系总是像层层调用进行的，就像人的辈分一样，而更靠顶部的Goroutine应有办法主动关闭其下属的Goroutine的执行但不会影响 其上层Goroutine的执行（不然程序可能就失控了）。为了实现这种关系，**Context结构也应该像一棵树**，叶子节点须总是由根节点衍生出来的。

![](/public/upload/go/context_tree.png)

如上左图，代表一棵 context 树。当调用左图中标红 context 的 cancel 方法后，该 context 从它的父 context 中去除掉了：实线箭头变成了虚线。且虚线圈框出来的 context 都被取消了，圈内的 context 间的父子关系都荡然无存了。

### 子 goroutine  使用context

```go
type Context interface {
	Deadline() (deadline time.Time, ok bool)    // 获取设置的截止时间
	Done() <-chan struct{}      // 如果该方法返回的chan可以读取，则意味着parent context已经发起了取消请求
	Err() error                 // 返回取消的原因，在 Done 返回的 Channel 被关闭时返回非空的值；如果 context.Context 被取消，会返回 Canceled 错误；如果 context.Context 超时，会返回 DeadlineExceeded 错误；
	Value(key interface{}) interface{}
}
```

![](/public/upload/go/context_object.png)

```go
// golang.org/x/net/context/pre_go17.go
type cancelCtx struct {
	Context
	done chan struct{}          // closed by the first cancel call.
	mu       sync.Mutex
	children map[canceler]bool  // child 会被加入 parent 的 children 列表中，等待 parent 释放取消信号；
	err      error             
}
func (c *cancelCtx) cancel(removeFromParent bool, err error) {
	c.mu.Lock()
	if c.err != nil {c.mu.Unlock() return}
	c.err = err
	if c.done == nil {
		c.done = closedchan
	} else {
		close(c.done)
	}
	for child := range c.children {
		child.cancel(false, err)
	}
	c.children = nil
	c.mu.Unlock()
	if removeFromParent {
		removeChild(c.Context, c)
	}
}
```

### 使用建议

官方对于使用 context 提出了几点建议：

1. 不要将 Context 塞到结构体里。直接将 Context 类型作为函数的第一参数，而且一般都命名为 ctx。
2. 不要向函数传入一个 nil 的 context，如果你实在不知道传什么，标准库给你准备好了一个 context：todo。
3. 不要把本应该作为函数参数的类型塞到 context 中，context 存储的应该是一些共同的数据。例如：登陆的 session、cookie 等。
4. 同一个 context 可能会被传递到多个 goroutine，别担心，context 是并发安全的。

context 取消和传值示例

```go
func main() {
    // cancel 是一个方法
	ctx, cancel := context.WithCancel(context.Background())
	valueCtx := context.WithValue(ctx, key, "add value")
	go watch(valueCtx)
	time.Sleep(10 * time.Second)
	cancel()
	time.Sleep(5 * time.Second)
}
func watch(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			//get value
			fmt.Println(ctx.Value(key), "is cancel")
			return
		default:
			//get value
			fmt.Println(ctx.Value(key), "in goroutine")
			time.Sleep(2 * time.Second)
		}
	}
}
```










