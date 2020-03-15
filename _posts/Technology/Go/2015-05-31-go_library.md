---

layout: post
title: Go常用的一些库
category: 技术
tags: Go
keywords: Go library

---

## 一 前言

* TOC
{:toc}

本文主要阐述一下golang中常用的库。

## go runtime

提到 runtime, 大家可能会想起 java, python 的 runtime. 不过 go 和这两者不太一样, java, python 的 runtime 是虚拟机, 而 go 的 runtime 和用户代码一起编译到一个可执行文件中.用户代码和 runtime 代码除了代码组织上有界限外, 运行的时候并没有明显的界限. 一些常用的关键字被编译成 runtime 包下的一些函数调用.


Golang runtime 是go语言运行所需要的基础设施
1. 协程调度、内存分配、GC
2. 操作系统及cpu 相关的操作的封装（信号处理、系统调用、寄存器操作、原子操作等）CGO。go 对系统调用进行了封装，可不依赖glibc
3. pprof,trace,race 检测的支持
4. map,channel,string 等内置类型及反射的实现

## unsafe

相比于 C 语言中指针的灵活，Go 的指针多了一些限制。

1. Go的指针不能进行数学运算。
2. 不同类型的指针不能相互转换。
3. 不同类型的指针不能使用==或!=比较。
4. 不同类型的指针变量不能相互赋值。

**为什么有 unsafe？**Go 语言类型系统是为了安全和效率设计的，有时，安全会导致效率低下。有了 unsafe 包，高阶的程序员就可以利用它**绕过**类型系统的低效。Package unsafe contains operations that step around the **type safety** of Go programs.

`$GOROOT/src/unsafe/unsafe.go` 里只有一个文件，内容只有几行

```go
package unsafe
type ArbitraryType int
type Pointer *ArbitraryType
func Sizeof(x ArbitraryType) uintptr
func Offsetof(x ArbitraryType) uintptr
func Alignof(x ArbitraryType) uintptr
```

以上三个函数返回的结果都是 uintptr 类型，这和 unsafe.Pointer 可以相互转换。三个函数都是在编译期间执行，它们的结果可以直接赋给 const型变量。另外，因为三个函数执行的结果和操作系统、编译器相关，所以是不可移植的。Packages that import unsafe may be non-portable and are not protected by the Go 1 compatibility guidelines.

unsafe 包提供了 2 点重要的能力：

1. 任何类型的指针和 unsafe.Pointer 可以相互转换。
2. uintptr 类型和 unsafe.Pointer 可以相互转换。

pointer 不能直接进行数学运算，但可以把它转换成 uintptr，对 uintptr 类型进行数学运算，再转换成 pointer 类型。uintptr 并没有指针的语义，意思就是 uintptr 所指向的对象会被 gc 无情地回收。而 unsafe.Pointer 有指针语义，可以保护它所指向的对象在“有用”的时候不会被垃圾回收。

## context

Go 1.7 标准库引入 context，中文译作“上下文”，准确说它是 goroutine 的上下文，包含 goroutine 的运行状态、环境、现场等信息。

![](/public/upload/go/context_object.png)

### 为什么有 context？

在 Go 的 server 里，通常每来一个请求都会启动若干个 goroutine 同时工作：有些去数据库拿数据，有些调用下游接口获取相关数据……这些 goroutine 需要共享这个请求的基本数据，例如登陆的 token，处理请求的最大超时时间（如果超过此值再返回数据，请求方因为超时接收不到）等等。当请求被取消或超时，所有正在为这个请求工作的 goroutine 需要快速退出，因为它们的“工作成果”不再被需要了。context 包就是为了解决上面所说的这些问题而开发的：在 一组 goroutine 之间传递共享的值、取消信号、deadline……

### 为什么是context 树

Goroutine的创建和调用关系总是像层层调用进行的，就像人的辈分一样，而更靠顶部的Goroutine应有办法主动关闭其下属的Goroutine的执行但不会影响 其上层Goroutine的执行（不然程序可能就失控了）。为了实现这种关系，**Context结构也应该像一棵树**，叶子节点须总是由根节点衍生出来的。

![](/public/upload/go/context_tree.png)

如上左图，代表一棵 context 树。当调用左图中标红 context 的 cancel 方法后，该 context 从它的父 context 中去除掉了：实线箭头变成了虚线。且虚线圈框出来的 context 都被取消了，圈内的 context 间的父子关系都荡然无存了。

要创建Context树，第一步就是要得到根节点，context.Background函数的返回值就是根节点：

```go
func Background() Context
```

有了根节点，又该怎么创建其它的子节点，孙节点呢？context包为我们提供了多个函数来创建他们：

```go
func WithCancel(parent Context) (ctx Context, cancel CancelFunc)
func WithDeadline(parent Context, deadline time.Time) (Context, CancelFunc)
func WithTimeout(parent Context, timeout time.Duration) (Context, CancelFunc)
func WithValue(parent Context, key interface{}, val interface{}) Context
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

## Go代码中的依赖注入

[Go中的依赖注入](https://www.jianshu.com/p/cb3682ad34a7) 推荐使用 [uber-go/dig](https://github.com/uber-go/dig) 
A reflection based dependency injection toolkit for Go.

依赖注入是你的组件（比如go语言中的structs）在创建时应该接收它的依赖关系。PS：这个理念在java、spring 已经普及多年。这与在初始化期间构建其自己的依赖关系的组件的相关反模式相反。

**设计模式分为创建、结构和行为三大类，如果自己构造依赖关系， 则创建 与 行为 两个目的的代码容易耦合在一起， 代码较长，给理解造成困难。**

## 读写锁

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
    
## command line application

go 可执行文件没有复杂的依赖（java依赖jvm、python 依赖python库），特别适合做一些命令行工具

大概的套路都是

1. 定义一个Command对象
2. Command 对象一般有一个 name，多个flag（全写和简写） 以及一个处理函数

### [urfave/cli](https://github.com/urfave/cli)

cli is a simple, fast, and fun package for building command line apps in Go. The goal is to enable developers to write fast and distributable command line applications in an expressive way.

Things like generating help text and parsing command flags/options should not hinder productivity when writing a command line app.This is where cli comes into play. cli makes command line programming fun, organized, and expressive!

### [spf13/cobra](https://github.com/spf13/cobra) 

这个库牛就牛在k8s 用的也是它

The best applications will read like sentences when used(命令执行起来应该像句子一样). Users will know how to use the application because they will natively understand how to use it.

The pattern to follow is `APPNAME VERB NOUN --ADJECTIVE`. or `APPNAME COMMAND ARG --FLAG`

A flag is a way to modify the behavior of a command 这句说的很有感觉


