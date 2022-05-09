---

layout: post
title: go设计哲学
category: 技术
tags: Go
keywords: go 设计哲学

---

## 前言

* TOC
{:toc}

[深入剖析对 Go 的成功作出巨大贡献的设计决策](https://mp.weixin.qq.com/s/zXOjaIuvu4XrWSGRqndOiw)

## 组合

《Go语言第一课》在 Go 语言设计层面，Go 设计者为开发者们提供了正交的语法元素，以供后续组合使用
1. Go 语言无类型层次体系，各类型之间是相互独立的，没有子类型的概念；
2. 每个类型都可以有自己的方法集合，类型定义与方法实现是正交独立的；
3. 实现某个接口时，无需像 Java 那样采用特定关键字修饰；
4. 包之间是相对独立的，没有子包的概念。

我们可以看到，无论是包、接口还是一个个具体的类型定义，Go 语言其实是为我们呈现了这样的一幅图景：**一座座没有关联的“孤岛”**，但每个岛内又都很精彩。那么现在摆在面前的工作，就是**在这些孤岛之间以最适当的方式建立关联，并形成一个整体**。而 Go 选择采用的组合方式，也是最主要的方式。

**组合原则的应用实质上是塑造了 Go 程序的骨架结构**。类型嵌入为类型提供了垂直扩展能力，而接口是水平组合的关键，它好比程序肌体上的“关节”，给予连接“关节”的两个部分各自“自由活动”的能力，而整体上又实现了某种功能。

当我们通过垂直组合将一个个类型建立完毕后，就好比我们已经建立了整个应用程序骨架中的“器官”，比如手、手臂等，那么这些“器官”之间又是通过什么连接在一起的呢？关节! `func Save(f *os.File, data []byte) error` 发现 Save **函数所在的“器官”**与 **os.File 所在的“器官”**之间采用了一种硬连接的方式，而以 os.File 这样的结构体作为“关节”让它连接的两个“器官”丧失了相互运动的自由度，让它与它连接的两个“器官”构成的联结体变得“僵直”。 `func Save(w io.Writer, data []byte) error`，用 io.Writer 接口类型替换掉了 *os.File，这样一来就符合了接口分离原则。PS： 通过嵌入定义基本类型（器官），通过接口作为方法的参数 作为类型/器官之间的关节（或者说方法连接器官），编程就是定义器官（组成部分）以及器官之间的关系/关节。

### 垂直组合

Go 语言为支撑组合的设计提供了类型嵌入（Type Embedding）。通过类型嵌入，我们可以将已经实现的功能嵌入到新类型中，以快速满足新类型的功能需求（**用于新类型的定义**），这种方式有些类似经典面向对象语言中的“继承”机制，但在原理上却与面向对象中的继承完全不同，这是一种 Go 设计者们精心设计的“**语法糖**”。**被嵌入的类型和新类型两者之间没有任何关系，甚至相互完全不知道对方的存在，更没有经典面向对象语言中的那种父类、子类的关系，以及向上、向下转型（Type Casting）**。通过新类型实例调用方法时，**方法的匹配主要取决于方法名字，而不是类型**。这种组合方式，我称之为**垂直组合**，即通过类型嵌入，快速让一个新类型“复用”其他类型已经实现的能力，实现功能的垂直扩展。

poolLocal 这个结构体类型中嵌入了类型 Mutex，这就使得 poolLocal 这个类型具有了互斥同步的能力，我们可以通过 poolLocal 类型的变量，直接调用 Mutex 类型的方法 Lock 或 Unlock。
```go
// $GOROOT/src/sync/pool.go
type poolLocal struct {
    private interface{}   
    shared  []interface{}
    Mutex               
    pad     [128]byte  
}
```
标准库通过嵌入接口类型的方式来实现接口行为的聚合，**组成大接口**，这种方式在标准库中尤为常用，并且已经成为了 Go 语言的一种惯用法。
```go
// $GOROOT/src/io/io.go
type ReadWriter interface {
    Reader
    Writer
}
```

### 水平组合

垂直组合本质上是一种“能力继承”，采用嵌入方式定义的新类型继承了嵌入类型的能力。Go 还有一种常见的组合方式，叫水平组合。和垂直组合的能力继承不同，水平组合是一种能力委托（Delegate），我们通常使用接口类型来实现水平组合。

Go 语言中的接口是一个创新设计，它只是方法集合，并且它与实现者之间的关系无需通过显式关键字修饰，它让程序内部各部分之间的耦合降至最低，同时它也是连接程序各个部分之间“纽带”。PS：与孤岛 联系起来，找找感觉

水平组合的模式有很多，比如一种常见方法就是，通过接受接口类型参数的普通函数进行组合，如以下代码段所示：

```go
// $GOROOT/src/io/ioutil/ioutil.go
func ReadAll(r io.Reader)([]byte, error)
// $GOROOT/src/io/io.go
func Copy(dst Writer, src Reader)(written int64, err error)
```

也就是说，函数 ReadAll 通过 io.Reader 这个接口，将 io.Reader 的实现与 ReadAll 所在的包低耦合地水平组合在一起了，从而达到从任意实现 io.Reader 的数据源读取所有数据的目的。类似的水平组合“模式”还有点缀器、中间件等。PS：就是尽量别在方法参数中直接用struct

## 并发

1. 在语法层面提供了并发原语支持。在提供了开销较低的 goroutine 的同时，Go 还在语言层面内置了辅助并发设计的原语：channel 和 select。开发者可以通过语言内置的 channel 传递消息或实现同步，并通过 select 实现多路 channel 的并发控制。相较于传统复杂的线程并发模型，Go 对并发的原生支持将大大降低开发人员在开发并发程序时的心智负担。
2. 并发与组合的哲学是一脉相承的，并发是一个更大的组合的概念，它在程序设计的全局层面对程序进行拆解组合，再映射到程序执行层面上：goroutines 各自执行特定的工作，通过 channel+select 将 goroutines 组合连接起来。并发的存在鼓励程序员在程序设计时进行独立计算的分解，而对并发的原生支持让 Go 语言也更适应现代计算环境。

## 其它

Go 不是一门面向对象的语言，但它可以模拟面向对象语言的某些功能。

[go语言设计哲学](https://studygolang.com/articles/2944)go 没有像JAVA一样，宗教式的完全面向对象设计；
1. 完全面向对象设计就是一刀切的宗教式的设计，但其并不能很好的表述这个世界，这就导致其表现力不足，最后通过设计模式和面向切面等设计技巧来弥补语言方面的缺陷； JAVA就好比：手里握着是锤子，看什么都是钉子，什么都是类的对象，这个和现实世界不符，类表示单个事物还可以，一旦表示多个事物及其交互，其表现力也就会遇到各种挑战。
2. go是面向工程的实用主义者，其糅合了面向对象的设计，函数式设计和过程式设计的优点；原来通过各种设计模式的设计通过函数、接口、组合等简单方式就搞定了；go有更多胶水的东西比如：全局变量、常量，函数，闭包等等，可以轻松的的把模块衔接和驱动起来；

[Is Go an object-oriented language?](https://golang.org/doc/faq#Is_Go_an_object-oriented_language)Yes and no. Although Go has types and methods and allows an object-oriented style of programming, there is no type hierarchy. The concept of “interface” in Go provides a different approach that we believe is easy to use and in some ways more general. There are also ways to embed types in other types to provide something analogous—but not identical—to subclassing. Moreover, methods in Go are more general than in C++ or Java: they can be defined for any sort of data, even built-in types such as plain, “unboxed” integers. They are not restricted to structs (classes).

与常见编程语言的不同之处：

1. 接口是非侵入性的/鸭子类型， 实现不需要依赖接口定义。Go 语言只会在传递参数、返回参数以及变量赋值时才会对某个类型是否实现接口进行检查。
2. 不支持 “继承“，“子类”通过聚合“父类” 可以调用 ”父类“的方法，但无法重载，“子类”也无法直接赋值给 “父类”变量。
3. 倾向于使用小的接口定义， 很多接口只包含一个方法。较大的接口定义，可以由多个小接口定义组合而成

    ```go
    type ReadWriter interface{
        Reader
        Writer
    }
    ```

[程序员技术选型：写Go还是Java？](https://mp.weixin.qq.com/s/v1jMd875d9hvfY2Y-AJO4Q)Go 不是面向对象编程语言。Go 没有类似 Java 的继承机制，因为它没有通过继承实现传统的多态性。实际上，它没有对象，只有结构体。它可以通过接口和让结构体实现接口来模拟一些面向对象特性。此外，你可以在结构体中嵌入结构体，但内部结构体无法访问外部结构体的数据和方法。Go 使用组合而不是继承将一些行为和数据组合在一起。

Go 是一种命令式语言，Java 是一种声明式语言。Go 没有依赖注入，我们需要显式地将所有东西包装在一起。因此，**在使用 Go 时尽量少用“魔法”之类的东西**。一切代码对于代码评审人员来说都应该是显而易见的。Go 程序员应该了解 Go 代码如何使用内存、文件系统和其他资源。Java 要求开发人员更多地地关注程序的业务逻辑，知道如何创建、过滤、修改和存储数据。系统底层和数据库方面的东西都是通过配置和注解来完成的（比如通过 Spring Boot 等通用框架）。我们尽可能把枯燥乏味的东西留给框架去做。这样做很方便，但控制也反转了，限制了我们优化整个过程的能力。

![](/public/upload/go/go_first_class.jpg)