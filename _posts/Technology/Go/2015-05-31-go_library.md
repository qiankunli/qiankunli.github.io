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

[Go 大杀器之性能剖析 PProf](https://eddycjy.gitbook.io/golang/di-9-ke-gong-ju/go-tool-pprof) 未读
[Go 大杀器之跟踪剖析 trace](https://eddycjy.gitbook.io/golang/di-9-ke-gong-ju/go-tool-trace) 未读
[用 GODEBUG 看调度跟踪](https://eddycjy.gitbook.io/golang/di-9-ke-gong-ju/godebug-sched) 未读
[用 GODEBUG 看GC](https://eddycjy.gitbook.io/golang/di-9-ke-gong-ju/godebug-gc) 未读

## klog

[logr](https://github.com/go-logr/logr) logr offers an(other) opinion on how Go programs and libraries can do logging without becoming coupled to a particular logging implementation. This is not an implementation of logging - it is an API. 用户使用Logger type，Log提供方实现 LogSink interface. This decoupling allows application and library developers to write code in terms of logr.Logger (which has very low dependency fan-out) while the implementation of logging is managed "up stack" (e.g. in or near main().) Application developers can then switch out implementations as necessary. 

klog是著名google开源C++日志库glog的golang版本
1. 支持四种日志等级INFO < WARING < ERROR < FATAL，不支持DEBUG等级。
2. 每个日志等级对应一个日志文件，低等级的日志文件中除了包含该等级的日志，还会包含高等级的日志。
3. 日志文件可以根据大小切割，但是不能根据日期切割。
4. **程序开始时必须调用`flag.Parse()`解析命令行参数**，退出时必须调用`glog.Flush()`确保将缓存区日志输出。
5. Info()与Infoln()没有区别，因为glog为了保证每行只有一条log记录，会主动check末尾是否有换行符，如果没有的话，会自动加上。

```go
func main() {
    klog.InitFlags(nil)
    //Init the command-line flags.
    flag.Parse()
    ...
    // Flushes all pending log I/O.
    defer klog.Flush()
}
```
通常而言，日志打印会按错误级别进行打印，如:【Fatal,Error,Warning,Info】等级别。但在实际项目开发过程中，还会涉及到一些内部状态切换、基础库以及框架的使用。这些信息显然不能按照错误级别进行打印。所以对 Info 级别日志进行二次分级是必要的。info 又可细化为多个级别：0~10，信息的重要性依次降低。

![](/public/upload/go/klog_vlog.png)

```go
if glog.V(2) {
    glog.Info("log this")
}
// Equals
glog.V(2).Info("log this")
```

[如何在 Go 函数中获取调用者的函数名、文件名、行号...](https://mp.weixin.qq.com/s/Z2CCgpG7WSqgxE7ky-CBVg)跟踪调用链的神器。 
```go
pc, file, lineNo, ok := runtime.Caller(1)
fmt.Println(pc)
fmt.Println(file)
fmt.Println(lineNo)
fmt.Println(ok)
```

## unsafe

[深度解密Go语言之unsafe](https://mp.weixin.qq.com/s/OO-kwB4Fp_FnCaNXwGJoEw)相比于 C 语言中指针的灵活，Go 的指针多了一些限制。

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

以上三个函数返回的结果都是 uintptr 类型，这和 `unsafe.Pointer` 可以相互转换。三个函数都是在编译期间执行，它们的结果可以直接赋给 const型变量。另外，因为三个函数执行的结果和操作系统、编译器相关，所以是不可移植的。Packages that import unsafe may be non-portable and are not protected by the Go 1 compatibility guidelines.

unsafe 包提供了 2 点重要的能力：

1. 任何类型的指针和 unsafe.Pointer 可以相互转换。
2. uintptr 类型和 unsafe.Pointer 可以相互转换。

||`unsafe.Pointer`|pointer|
|---|---|---|
|数学运算|不能|可以|
|指针的语义|有|无<br>uintptr 所指向的对象会被 gc 无情地回收|

在java 中经常有一种场景

```java
class Business{
    private validate Object data;
    // sync 会被定时执行
    void sync()(
        Object newData = 从db 拉到新数据 构造data
        synchronized(this){
            this.data = newData
        }
    )
}
```
对应到go 中可以 `atomic.StorePointer($data,unsafe.Pointer(&newData))`

## sync.pool

[深度解密Go语言之sync.pool](https://mp.weixin.qq.com/s/O8EY0M4_Rt_0BLCcM8OwFw)

sync.Pool 是 sync 包下的一个组件，可以作为保存临时取还对象的一个“池子”。它的名字有一定的误导性，与java 里的pool 不同，sync.Pool 里装的对象可以被无通知地被回收（GC 发生时清理未使用的对象，Pool 不可以指定⼤⼩，⼤⼩只受制于 GC 临界值），可能 sync.Cache 是一个更合适的名字。

```go
var pool *sync.Pool
type Person struct {
  Name string
}
func main() {
  pool = &sync.Pool {
    // 用于在 Pool 里没有缓存的对象时，创建一个。这点就很像guava cache
    New: func()interface{} {
      fmt.Println("Creating a new Person")
      return new(Person)
    },
  }
  // 当调用 Get 方法时，如果池子里缓存了对象，就直接返回缓存的对象。如果没有存货，则调用 New 函数创建一个新的对象。
  p := pool.Get().(*Person)
  p.Name = "first"
  // 处理p
  // 将对象放回池中
  p.Name = ""   // 将对象清空
  pool.Put(p)
}
```

1. Go 语言内置的 fmt 包，encoding/json 包都可以看到 sync.Pool 的身影；gin，Echo 等框架也都使用了 sync.Pool。
2. Pool 里对象的生命周期受 GC 影响，不适合于做连接池，因为连接池需要自己管理对象的生命周期。
3. 不要对 Get 得到的对象有任何假设，更好的做法是归还对象时，将对象“清空”。


```go
type Pool struct {
    noCopy noCopy
    // 本地池，本地池数量与P 相等，一个P同时只能执行一个G，因此操作本地池时没有并发问题
    local     unsafe.Pointer  // local fixed-size per-P pool, actual type is [P]poolLocal
    localSize uintptr         // size of the local array
    victim     unsafe.Pointer // local from previous cycle
    victimSize uintptr        // size of victims array
    // New optionally specifies a function to generate a value when Get would otherwise return nil.It may not be changed concurrently with calls to Get.
    New func() interface{}
}
```

## Go代码中的依赖注入

[Go中的依赖注入](https://www.jianshu.com/p/cb3682ad34a7) 推荐使用 [uber-go/dig](https://github.com/uber-go/dig) 
A reflection based dependency injection toolkit for Go.

[Alibaba/IOC-golang 正式开源 ——打造服务于go开发者的IOC框架](https://mp.weixin.qq.com/s/Ar-JdkrQ5NnCWcGOoCuVgg)在面向对象编程的思路下，开发者需要直接关心对象之间的依赖关系、对象的加载模型、对象的生命周期等等问题。对于较为复杂的业务应用系统，随着对象数目增长，**对象之间的拓扑关系呈指数级增加**，如果这些逻辑全部由开发人员手动设计和维护，将会在应用内保存较多业务无关的冗余代码，影响开发效率，提高代码学习成本，增加了模块之间的耦合度，容易产生循环依赖等等问题。按照常规的应用开发模式，在一个“开发单元”内，开发者需要关注哪些事情？我们习惯于编写一个构造函数返回需要的对象，这个构造函数的入参，**包含了一些参数以及下游依赖**，我们在构造函数中会把这些对象和参数拼接成一个结构，再执行初始化逻辑，最后返回。

```go
// +ioc:autowire=true
// +ioc:autowire:type=singleton
type App struct {
    ServiceImpl1  ServiceInterface   `singleton:"main.ServiceImpl1"` // inject ServiceInterface 's ServiceImpl1 implementation
}
```

依赖注入是你的组件（比如go语言中的structs）在创建时应该接收它的依赖关系。PS：这个理念在java、spring 已经普及多年。这与在初始化期间构建其自己的依赖关系的组件的相关反模式相反。

**设计模式分为创建、结构和行为三大类，如果自己构造依赖关系， 则创建 与 行为 两个目的的代码容易耦合在一起， 代码较长，给理解造成困难。**

![](/public/upload/go/go_ioc_layer.png)

[为什么依赖注入只在 Java 技术栈中流行，在 go 和 cpp 没有大量使用？](https://www.zhihu.com/question/521822847/answer/2451020694)

    
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

## 其它

组合一个struct 以方便我们给它加方法。比如 net.IP 是go 库的struct，想对ip 做一些扩充。就可以提供两个ip 的转换方法，以及扩充方法。

```go
// Sub class net.IPNet so that we can add JSON marshalling and unmarshalling.
type IPNet struct {
	net.IPNet
}
// MarshalJSON interface for an IPNet
func (i IPNet) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}
func (i *IPNet) UnmarshalJSON(b []byte) error {...}
// Version returns the IP version for an IPNet, or 0 if not a valid IP net.
func (i *IPNet) Version() int {
	if i.IP.To4() != nil {
		return 4
	} else if len(i.IP) == net.IPv6len {
		return 6
	}
	return 0
}
func ParseCIDR(c string) (*IP, *IPNet, error) {...}
```


