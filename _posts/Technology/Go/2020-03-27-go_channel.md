---

layout: post
title: go channel
category: 技术
tags: Go
keywords: Go defer

---

## 前言

* TOC
{:toc}


## channel

[Go 语言设计与实现-Channel](https://draveness.me/golang/docs/part3-runtime/ch06-concurrency/golang-channel/)Channel 在运行时的内部表示是 runtime.hchan，该结构体中包含了一个用于保护成员变量的互斥锁，从某种程度上说，**Channel 是一个用于同步和通信的有锁队列**。社区有一些无锁Channel 的提案，但还在不停的优化中。PS：**本质上还是共享内存**

```go
type hchan struct {
	qcount   uint           // Channel 中的元素个数；
	dataqsiz uint           // Channel 中的循环队列的长度；
	buf      unsafe.Pointer // Channel 的缓冲区数据指针；
	elemsize uint16         // 当前 Channel 能够收发的元素大小
	closed   uint32
	elemtype *_type         // 当前 Channel 能够收发的元素类型
	sendx    uint           // Channel 的发送操作处理到的位置；
    recvx    uint           // Channel 的接收操作处理到的位置；
	recvq    waitq          // sendq 和 recvq 当前 Channel 由于缓冲区空间不足而阻塞的 Goroutine 列表
	sendq    waitq

	lock mutex
}
```

### 发送数据

当我们想要向 Channel 发送数据时，就需要使用 `ch <- i` 语句，**编译器**会经过一系列处理后调用runtime.chansend，这个函数负责了发送数据的全部逻辑，如果我们在调用时将 block 参数设置成 true，那么就表示当前发送操作是一个阻塞操作

```go
func chansend(c *hchan, ep unsafe.Pointer, block bool, callerpc uintptr) bool {
    lock(&c.lock)
    // 如果 Channel 已经关闭
	if c.closed != 0 {
		unlock(&c.lock)
		panic(plainError("send on closed channel"))
    }
    // 如果目标 Channel 没有被关闭并且已经有处于读等待的 Goroutine，那么 runtime.chansend 函数会从接收队列 recvq 中取出最先陷入等待的 Goroutine 并直接向它发送数据
    if sg := c.recvq.dequeue(); sg != nil {
		send(c, sg, ep, func() { unlock(&c.lock) }, 3)
		return true
    }
    // 如果创建的 Channel 包含缓冲区并且 Channel 中的数据没有装满
    if c.qcount < c.dataqsiz {
		qp := chanbuf(c, c.sendx)
		typedmemmove(c.elemtype, qp, ep)
		c.sendx++
		if c.sendx == c.dataqsiz {
			c.sendx = 0
		}
		c.qcount++
		unlock(&c.lock)
		return true
    }
    // 当 Channel 没有接收者能够处理数据时，向 Channel 发送数据就会被下游阻塞
    if !block {
		unlock(&c.lock)
		return false
	}
	gp := getg()
	mysg := acquireSudog()
	mysg.elem = ep
	mysg.g = gp
	mysg.c = c
	gp.waiting = mysg
	c.sendq.enqueue(mysg)
	goparkunlock(&c.lock, waitReasonChanSend, traceEvGoBlockSend, 3)    // 触发 Goroutine 让出处理器的使用权
	gp.waiting = nil
	gp.param = nil
	mysg.c = nil
	releaseSudog(mysg)
	return true
}
```

发送数据时会调用 runtime.send，该函数的执行可以分成两个部分：

1. 调用 `runtime.sendDirect` 函数将发送的数据直接拷贝到 `x = <-c` 表达式中变量 x 所在的内存地址上；
2. 调用 `runtime.goready` 将等待接收数据的 Goroutine 标记成可运行状态 Grunnable 并把该 Goroutine 放到发送方所在的处理器的 runnext 上等待执行，该处理器在下一次调度时就会立刻唤醒数据的接收方；

### 接收数据

```go
i <- ch
i, ok <- ch
```
不同的接收方式会被转换成 `runtime.chanrecv1` 和 `runtime.chanrecv2` 两种不同函数的调用，但是这两个函数最终还是会调用 `runtime.chanrecv`。

```go
func chanrecv(c *hchan, ep unsafe.Pointer, block bool) (selected, received bool) {
    // 当我们从一个空 Channel 接收数据时会直接调用 runtime.gopark 直接让出处理器的使用权
	if c == nil {
		if !block {
			return
		}
		gopark(nil, nil, waitReasonChanReceiveNilChan, traceEvGoStop, 2)
		throw("unreachable")
	}
    lock(&c.lock)
    // 如果当前 Channel 已经被关闭并且缓冲区中不存在任何的数据，那么就会清除 ep 指针中的数据并立刻返回。
	if c.closed != 0 && c.qcount == 0 {
		unlock(&c.lock)
		if ep != nil {
			typedmemclr(c.elemtype, ep)
		}
		return true, false
    }
    // 当存在等待的发送者时，通过 runtime.recv 直接从阻塞的发送者或者缓冲区中获取数据；
    if sg := c.sendq.dequeue(); sg != nil {
		recv(c, sg, ep, func() { unlock(&c.lock) }, 3)
		return true, true
    }
    // 当Channel 的缓冲区中已经包含数据时，从 Channel 中接收数据会直接从缓冲区中 recvx 的索引位置中取出数据进行处理
    if c.qcount > 0 {
		qp := chanbuf(c, c.recvx)
		if ep != nil {
			typedmemmove(c.elemtype, ep, qp)
		}
		typedmemclr(c.elemtype, qp)
		c.recvx++
		if c.recvx == c.dataqsiz {
			c.recvx = 0
		}
		c.qcount--
		return true, true
    }
    // 当 Channel 的发送队列中不存在等待的 Goroutine 并且缓冲区中也不存在任何数据时，从管道中接收数据的操作会变成阻塞操作，然而不是所有的接收操作都是阻塞的，与 select 语句结合使用时就可能会使用到非阻塞的接收操作
    if !block {
		unlock(&c.lock)
		return false, false
	}

	gp := getg()
	mysg := acquireSudog()
	mysg.elem = ep
	gp.waiting = mysg
	mysg.g = gp
	mysg.c = c
	c.recvq.enqueue(mysg)
	goparkunlock(&c.lock, waitReasonChanReceive, traceEvGoBlockRecv, 3)

	gp.waiting = nil
	closed := gp.param == nil
	gp.param = nil
	releaseSudog(mysg)
    return true, !closed
}
```

### 关闭

编译器会将用于关闭管道的 close 关键字转换成 OCLOSE 节点以及 runtime.closechan 的函数调用。该函数在最后会为所有被阻塞的 Goroutine 调用 runtime.goready 触发调度。

## select

[面向信仰编程-select](https://draveness.me/golang/docs/part2-foundation/ch05-keyword/golang-select/)C 语言中的 select 关键字可以同时监听多个文件描述符的可读或者可写的状态，Go 语言中的 select 关键字也能够让 Goroutine 同时等待多个 Channel 的可读或者可写，在多个文件或者 Channel 发生状态改变之前，select 会一直阻塞当前线程或者 Goroutine。

```go
func fibonacci(c, quit chan int) {
	x, y := 0, 1
	for {
		select {
		case c <- x:
			x, y = y, x+y
		case <-quit:
			fmt.Println("quit")
            return
        default:
		    println("default")
		}
	}
}
```

1. select 是一种与 switch 相似的控制结构，与 switch 不同的是，select 中虽然也有多个 case，但是这些 case 中的表达式必须都是 Channel 的收发操作。
2. 上述控制结构会等待 `c <- x` 或者 `<-quit` 两个表达式中任意一个的返回。无论哪一个表达式返回都会立刻执行 case 中的代码，当 select 中的两个 case 同时被触发时，就会随机选择一个 case 执行。
3. 非阻塞的 Channel ：比如，我们只是想看看 Channel 的可读或者可写状态，不希望Channel收发阻塞当前 Goroutine。此时可以为select 添加default 分支，当某次循环 不存在可以收发的 Channel 时，会直接执行 default 中的代码并返回

## 广播channel

channels在多个writer，一个reader的模型下面工作的很好，但是却不能很容易的处理多个reader等待获取一个writer发送的数据的情况。处理这样的情况，可能的一个go api原型如下：

```go
type Broadcaster interface{
    func NewBroadcaster() Broadcaster
    func (b Broadcaster) Write(v interface{})
    func (b Broadcaster) Listen() chan interface{}
}
```

broadcast channel通过NewBroadcaster创建，通过Write函数进行数据广播。为了监听这个channel的信息，我们使用Listen，该函数返回一个新的channel去接受Write发送的数据。这套解决方案需要一个中间process用来处理所有reader的注册。当调用Listen创建新的channel之后，该channel就被注册，通常该中间process的主循环如下：

```go
for {
    select {
        case v := <-inc:
            for _, c := range(listeners) {
                c <- v
            }
        case c := <- registeryc:
            listeners.push(c)
    }
}
```

这是一个通常的做法，但是该process在处理数据广播的时候会阻塞，直到所有的readers读取到值。一个可选的解决方式就是reader的channel是有buffer缓冲的，缓冲大小我们可以按需调节。或者当buffer满的时候我们将数据丢弃。

[SierraSoftworks/multicast](https://github.com/SierraSoftworks/multicast) 解决了这个问题，The multicast module provides single-writer, multiple-reader semantics around Go channels. It attempts to maintain semantics similar to those offered by standard Go channels while guaranteeing parallel delivery (slow consumers won't hold up delivery to other listeners) and guaranteeing delivery to all registered listeners when a message is published. 

![](/public/upload/go/channel_multicast.png)

示例代码

```go
import (
    "fmt"
    "github.com/SierraSoftworks/multicast"
)
func main() {
    c := multicast.New()
	go func() {
		l := c.Listen()
		for msg := range l.C {
			fmt.Printf("Listener 1: %s\n", msg)
		}
        fmt.Println("Listener 1 Closed")
	}()
	go func() {
		l := c.Listen()
		for msg := range l.C {
			fmt.Printf("Listener 2: %s\n", msg)
		}
        fmt.Println("Listener 2 Closed")
	}()
    // 据笔者实践，此处最好加上  time.Sleep(1000)
	c.C <- "Hello World!"
	c.Close()
}
```