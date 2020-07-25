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



## Mechanics

共享内存

1. sync.Mutex,           类似lock
2. sync.WaitGroup,       类似CountDown       

CSP channel

```go
func AsyncService() chan string {
	retCh := make(chan string, 1)
	//retCh := make(chan string, 1)
	go func() {
		ret := service()
		fmt.Println("returned result.")
		retCh <- ret
		fmt.Println("service exited.")
	}()
	return retCh
}
func TestAsynService(t *testing.T) {
	retCh := AsyncService()
	fmt.Println(<-retCh)
	time.Sleep(time.Second * 1)
}
```

从上述代码的感觉看，`channel string`像极了`Future<String>`

### 读写锁

```go
package main
import (
    "errors"
    "fmt"
    "sync"
)
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

```java
public Future<String> AsyncService(){
    Future<String> future = Executer.submit(new Callable<String>(){
        public String call(){
            return xx.service();
        }
    });
    return future;
}
```
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

从上述视角看，与Future 相比，channel 也更像一个“受体”。

### 取消

所有的channel 接收者（通常是一个goroutine） 都会在channel关闭时，立刻从阻塞等待中返回且 `v,ok <- ch` ok值为false。这个广播机制常被利用， 进行向多个订阅者同时发送信号，如：退出信号。PS：有点类似`Thread.interrupt()` 的感觉。

```go
func isCancelled(cancelChan chan struct{}) bool {
	select {
	case <-cancelChan:
		return true
	default:
		return false
	}
}
func TestCancel(t testing.T){
    cancelChan := make(chan struct{},0)
    go func(cancelChan chan struct{}){
        for{
            if isCancelled(cancelChan){
                break
            }
            time.Sleep(time.Millisecond * 5)
        }
        fmt.Println("Cancelled")
    }(cancelChan)
    // 类似java future.cancel()
    close(cancelChan)
    time.Sleep(time.Second * 1)
}
```

### 只执行一次

GetSingletonObj 可以被多次并发调用， 但只执行一次（可比java 的单例模式清爽多了）
```go
var once sync.Once
func GetSingletonObj() *SingletonObj{
    once.Do(func(){
        fmt.Println("Create Singleton obj.")
        obj = &SingletonObj{}
    })
    return obj
}
```

**通过buffered channel 可以变相实现对象池的效果**。

## context

[深度解密Go语言之context](https://mp.weixin.qq.com/s/GpVy1eB5Cz_t-dhVC6BJNw)Go 1.7 标准库引入 context，中文译作“上下文”，准确说它是 goroutine 的上下文，包含 goroutine 的运行状态、环境、现场等信息。

![](/public/upload/go/context_object.png)

### 为什么有 context？

在 Go 的 server 里，通常每来一个请求都会启动若干个 goroutine 同时工作：有些去数据库拿数据，有些调用下游接口获取相关数据……这些 goroutine 需要共享这个请求的基本数据，例如登陆的 token，处理请求的最大超时时间（如果超过此值再返回数据，请求方因为超时接收不到）等等。当请求被取消或超时，所有正在为这个请求工作的 goroutine 需要快速退出，因为它们的“工作成果”不再被需要了。context 包就是为了解决上面所说的这些问题而开发的：在 一组 goroutine 之间传递共享的值、取消信号、deadline……

goroutine **主动**检查 Context 的状态并作出正确的响应。PS： **从这个视角看，context 跟 惯用的stopChannel 差不多**

### 为什么是context 树

1. 根Context，通过`context.Background()` 创建
2. 子Context，`context.WithCancel(parentContext)` 创建
    ```go
    func WithCancel(parent Context) (ctx Context, cancel CancelFunc)
    func WithDeadline(parent Context, deadline time.Time) (Context, CancelFunc)
    func WithTimeout(parent Context, timeout time.Duration) (Context, CancelFunc)
    func WithValue(parent Context, key interface{}, val interface{}) Context
    ```
3. 当前Context 被取消时，基于他的子context 都会被取消
4. 接收取消通知 `<-ctx.Done()`

Goroutine的创建和调用关系总是像层层调用进行的，就像人的辈分一样，而更靠顶部的Goroutine应有办法主动关闭其下属的Goroutine的执行但不会影响 其上层Goroutine的执行（不然程序可能就失控了）。为了实现这种关系，**Context结构也应该像一棵树**，叶子节点须总是由根节点衍生出来的。

![](/public/upload/go/context_tree.png)

如上左图，代表一棵 context 树。当调用左图中标红 context 的 cancel 方法后，该 context 从它的父 context 中去除掉了：实线箭头变成了虚线。且虚线圈框出来的 context 都被取消了，圈内的 context 间的父子关系都荡然无存了。


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










