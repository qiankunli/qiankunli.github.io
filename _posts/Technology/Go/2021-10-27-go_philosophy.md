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

个人感受
1. golang 太多的匿名函数、函数套函数，缺点是不好懂，优点是用匿名函数消掉了很多类，大部分可以塞到一个struct里，十个八个struct就能实现一个 功能还算复杂的框架。
2. 给go 很多函数 起一个 xxFactory 等面向对象意义的名字，真的让人很错乱。

## 流派

[Go，基于连接与组合的语言（上）](https://www.infoq.cn/article/go-based-on-connection-combination-language-1)Go 语言是非常简约的语言。简约的意思是少而精。Go 语言极力追求语言特性的最小化，如果某个语法特性只是少些几行代码，但对解决实际问题的难度不会产生本质的影响，那么这样的语法特性就不会被加入。Go 语言更关心的是如何解决程序员开发上的心智负担。如何减少代码出错的机会，如何更容易写出高品质的代码，是 Go 设计时极度关心的问题。
1. Go 语言支持面向对象，但将特性最小化。Go 语言中有结构体（类似面向对象中的类），结构体可以有方法，这就是 Go 对面向对象支持的所有内容。
2. 面向消息编程是个比较小众的编程流派，因为分布式与并发编程的强烈诉求而崛起。面向消息编程的主体思想是推荐基于消息而不是基于锁和共享内存进行并发编程。Erlang 语言是面向消息编程的代表。Go 语言中有面向消息的影子。因为 Go 语言中有 channel，但 channel 只是 Go 语言的基础语法特性，Go 并没有杜绝锁和共享内存，所以它并不能算面向消息编程流派。
3. 函数式编程也是一个小众的流派，函数式编程中有些概念如：闭包、柯里化、变量不可变等。Haskell、Erlang 都是这个流派的代表。函数式编程之所以小众，个人认为最重要的原因，是理论基础不广为人知。我们缺乏面向函数式编程的数据结构学。因为变量不可变，数据结构学需要用完全不同思维方式来表达。比如在传统命令式的编程方式中，数组是最简单的基础数据结构，但函数式编程中，数组这样的数据结构很难提供（修改数组的一个元素成本太高，Erlang 语言中数组这个数据结构很晚才引入，用 tree 来模拟数组）。Go 语言除了支持闭包外，没有太多函数式的影子。
Go 语言有以上每一流派的影子，但都只是把这些流派的最基础的概念吸收，这些特性很基础，很难作为一个流派的关键特征来看。所以从编程范式上来说，个人认为 Go 语言不属于以上任何流派。如果非要说一个流派，Go 语言类似 C++，应该算“多范式”流派的。C++ 是主流语言中，几乎是唯一一门大力宣扬多范式编程理念的语言。C++ 主要支持的编程范式是过程式编程、面向对象编程、泛型编程（我们上面没有把泛型编程列入讨论的流派之中）。C++ 对这些流派的主要特性支持都很完整，说“多范式”名副其实。但 Go 不一样的是，每个流派的特性支持都很基础，这些特性只能称之为功能，并没有形成范式。**Go 语言在吸收这些流派精华的基础上，开创了自己独特的编程风格：一种基于连接与组合的语言**。
1. 连接，指的是组件的耦合方式，也就是组件是如何被串联起来的。非侵入式的interface，抽象的io.Reader,io.Writer和pipe
2. 组合，是形成复合对象的基础。连接与组合都是语言中非常平凡的概念，但 Go 语言恰恰是在平凡之中见神奇。匿名组合、指针组合、接口组合。


## 组合

《Go语言第一课》在 Go 语言设计层面，Go 设计者为开发者们提供了正交的语法元素，以供后续组合使用
1. Go 语言无类型层次体系，各类型之间是相互独立的，没有子类型的概念；
2. 每个类型都可以有自己的方法集合，类型定义与方法实现是正交独立的；
3. 实现某个接口时，无需像 Java 那样采用特定关键字修饰；
4. 包之间是相对独立的，没有子包的概念。

我们可以看到，无论是包、接口还是一个个具体的类型定义，Go 语言其实是为我们呈现了这样的一幅图景：**一座座没有关联的“孤岛”**，但每个岛内又都很精彩。那么现在摆在面前的工作，就是**在这些孤岛之间以最适当的方式建立关联，并形成一个整体**。而 Go 选择采用的组合方式，也是最主要的方式。

**组合原则的应用实质上是塑造了 Go 程序的骨架结构**。类型嵌入为类型提供了垂直扩展能力，而接口是水平组合的关键，它好比程序肌体上的“关节”，给予连接“关节”的两个部分各自“自由活动”的能力，而整体上又实现了某种功能。

当我们通过垂直组合将一个个类型建立完毕后，就好比我们已经建立了整个应用程序骨架中的“器官”，比如手、手臂等，那么这些“器官”之间又是通过什么连接在一起的呢？关节! `func Save(f *os.File, data []byte) error` 发现 Save **函数所在的“器官”**与 **os.File 所在的“器官”**之间采用了一种硬连接的方式，而以 os.File 这样的结构体作为“关节”让它连接的两个“器官”丧失了相互运动的自由度，让它与它连接的两个“器官”构成的联结体变得“僵直”。 `func Save(w io.Writer, data []byte) error`，用 io.Writer 接口类型替换掉了 *os.File，这样一来就符合了接口分离原则。PS： 通过嵌入定义基本类型（器官），通过接口作为方法的参数 作为类型/器官之间的关节（或者说方法连接器官），编程就是定义器官（组成部分）以及器官之间的关系/关节。

组合优于继承：我们有一只鸭子和一只鸡，他们工作得很好。我们发现鸭子和鸡有很多重复的地方，他们都会飞，都有两只脚两个翅膀，都会唧唧或者嘎嘎叫。于是我们抽象出鸟这个父类，鸭子和鸡都继承了鸟这个父类， 当我们想要在飞的时候额外做点什么，只需要修改鸟就好了，代码得到了缩减，维护起来看似方便了。鸟工作得也很好。我们业务不断扩展，企鹅出现了。 它不会飞，但是会游泳。鸟的工作出了问题，于是我们把飞行这个功能被下沉到了会飞的鸟类，企鹅继承自一个不会飞的鸟类。接下来橡皮鸭子出现了，人们对于它究竟是不是鸟有了争议。开始浪费时间大量的讨论什么是鸟，鸟该做些什么。但我们的生活中没有鸟（请注意这句话），鸟是一个抽象， 我们生活中有鸡，有鸭。我们觉得他们有一些相同的地方，于是把拥有这些相同点的东西叫做鸟，**但永远不知道下一个遇见的，能不能算鸟，鸟的定义要不要修改**。这就是继承不适用的原因，让我们看看组合会怎么样。我们找到了鸡和鸭的共同点， 会飞，两只脚，两个翅膀，会叫。 这些东西加上其他的特质『组合』成了鸡或鸭。 会飞这个能力就能提出来，使用在每一个需要飞行能力的地方。 当我遇到企鹅，就不用拿飞行来『组合』它。飞行，不应该是鸡或鸭从父类继承的能力，而应该是『飞行能力』组合成了鸡鸭的一部分。

### 垂直组合

Go 语言为支撑组合的设计提供了类型嵌入（Type Embedding）。通过类型嵌入，我们可以将已经实现的功能嵌入到新类型中，以快速满足新类型的功能需求（**用于新类型的定义**），这种方式有些类似经典面向对象语言中的“继承”机制，但在原理上却与面向对象中的继承完全不同，这是一种 Go 设计者们精心设计的“**语法糖**”。**被嵌入的类型和新类型两者之间没有任何关系，甚至相互完全不知道对方的存在，更没有经典面向对象语言中的那种父类、子类的关系，以及向上、向下转型（Type Casting）**。通过新类型实例调用方法时，**方法的匹配主要取决于方法名字，而不是类型**。这种组合方式，我称之为**垂直组合**，即通过类型嵌入，快速让一个新类型“复用”其他类型已经实现的能力，实现功能的垂直扩展。

poolLocal 这个结构体类型中嵌入了类型 Mutex，这就使得 poolLocal 这个类型具有了互斥同步的能力，我们可以通过 poolLocal 类型的变量，直接调用 Mutex 类型的方法 Lock 或 Unlock。PS： 这个语法糖 至少使得在代码量上让组合跟继承差不多，组合优于继承现实依据之一。
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
比如很多tcp 的网络库会基于  net.Conn 提供自己的 Conn strurt。框架能力就是通过 “嵌入”层层堆叠起来的，这也是我们理解go 框架的突破口。
```go
type Conn interface {
	net.Conn
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


## 鸭子类型和小接口

Go 语言之父 Rob Pike 说过“接口越大，抽象程度越弱”。越偏向业务层，抽象难度就越高。所以Go 标准库小接口（1~3 个方法）占比略高于 Docker 和 Kubernetes 的原因。**Go 接口是构建 Go 应用骨架（对应血肉）的重要元素**。抽象的时机：在实际真正需要的时候才对程序进行抽象。再通俗一些来讲，就是不要为了抽象而抽象。接口的确可以实现解耦，但它也会引入“抽象”的副作用，或者说接口这种抽象也不是免费的，是有成本的，除了会造成运行效率的下降之外，也会影响代码的可读性。PS：**接口由使用方按需定义，而不用事前规划**。


与常见编程语言的不同之处：

1. 侵入式接口：在Java中，接口主要作为不同组件之间的契约存在。**对契约的实现是强制的，你必须声明你的确实现了该接口**。
1. 接口是非侵入性的/鸭子类型， 实现不需要依赖接口定义。Go 语言只会在传递参数、返回参数以及变量赋值时才会对某个类型是否实现接口进行检查。
2. 不支持 “继承“，“子类”通过聚合“父类” 可以调用 ”父类“的方法，但无法重载，“子类”也无法直接赋值给 “父类”变量。
3. 倾向于使用小的接口定义， 很多接口只包含一个方法。较大的接口定义，可以由多个小接口定义组合而成

    ```go
    type ReadWriter interface{
        Reader
        Writer
    }
    ```

Go 的类型系统不太常见，而且非常简单。内建类型包括结构体、函数和接口。 任何实现了接口的方法的类型都可以称为实现了该接口。类型可以被隐式的从表达式中推导， 而且不需要被显式的指定。 有关接口的特殊处理以及隐式的类型推导使得 Go 看起来**像是**一种轻量级的动态类型语言。

鸭子类型，是动态编程语言的一种对象推断策略，它更关注对象能如何被使用（观察对象有没有实现所需的方法、签名、语义），而不是对象的类型本身（比如用isintancece去判断）。Go 语言作为一门现代静态语言，是有后发优势的。它引入了动态语言的便利，同时又会进行静态语言的类型检查。 [Go是如何判断实现了interface](https://mp.weixin.qq.com/s/qH9HDEelHGi96u-tkiOPdQ)鸭子类型使得开发者可以不使用继承体系来灵活地实现一些“约定”，**尤其是使得混合不同来源、使用不同对象继承体系的代码成为可能**。有更低的耦合度，避免了继承带来的强耦合，不必先有父类/父接口，再有子类
1. 可以直接使用第三方库的类型。PS：比如你定义了一个Hello interface，第三方库有个类有hello方法，这个类就可以作为Hello interface使用
2. 不需要修改现有代码就能支持新类型
3. 再配合”组合>继承“，当我们需要扩展接口时，可以定义一个ExtendHello包含hello interface，而不会破坏现有的代码。

[万字长文复盘导致Go语言成功的那些设计决策](https://mp.weixin.qq.com/s/Ca72d8-A0UoiIv-EquT8rA)避免接口和实现之间的显式关联，允许Go程序员定义小型、灵活以及临时性的接口，而不是将它们作为复杂类型层次结构的基础构件。**它鼓励捕捉开发过程中出现的关系和操作，而不是需要提前计划和定义它们**。这对大型程序尤其有帮助，因为在刚开始开发时，最终的结构是很难看清楚的。初次学习Go的开发者常常担心一个类型会意外地实现一个接口。虽然很容易建立起这样的假设，但在实践中，不太可能为两个不兼容的操作选择相同的名称和签名，而且我们从未在实际的Go程序中看到这种情况发生。

[深入剖析对 Go 的成功作出巨大贡献的设计决策](https://mp.weixin.qq.com/s/zXOjaIuvu4XrWSGRqndOiw)
1. Go 不定义类，但允许将方法绑定到任何类型，包括结构、数组、slice、map 甚至是整数等基本类型。它没有类型的层次结构；我们认为继承往往会使程序在成长过程中更难适应。相反，Go 鼓励类型的组合。Go 通过其接口类型提供了面向对象的多态性。
2. 避免接口和实现之间的显式关联允许 Go 程序员定义小的、灵活的、通常是 ad hoc 接口，而不是将它们用作复杂类型层次结构中的基础块。它鼓励在开发过程中捕获关系和操作，而不需要提前计划和定义它们。这尤其有助于大型程序，在这些程序中，**刚开始开发时，最终的结构更加难以看清**。无需声明实现的方式鼓励使用精确的、一种或两种方法的接口，例如 Writer、Reader、Stringer（类似于 Java 的 toString 方法）等，这些接口遍布标准库。PS： 写代码的时候，觉得A这里可能会扩展，就定一个interface 先用着（可能只有一个方法），觉得BC需要扩展也类似，后续实现上，如果ABC 可以联动，就定义一个struct 实现ABC，也可以定义一个struct 实现A，另一个struct 实现BC。**一段程序确定的逻辑写成代码，不确定的逻辑留出interface**，自由实现和扩展。从这个视角看，interface 是扩展的手段，而不是在设计阶段就充分使用。

还有一种 先定义具体实现，后定义扩展的情况，比如k8s生态里，Pod 等core object 肯定是先出的，controller-runtime 相关的object是后出的，但因为 Pod 实现了 metav1.Object 和 runtime.Object，Pod 也就实现了 controller-runtime Object，controller-runtime 就可以拿着Object 去指代 任意k8s 对象了，定义方和实现方 不需要明确的由 extend/implement关键字来建立关系（只可以扩展子类，却无法**扩展父类**）。PS：或者说，在go里，一般由调用方定义接口（只有调用方知道使用什么方法，**interface 按调用方需要定义**），定义方（被调用方）提供struct 实现。比如标准库中 net/http/fs.go 中需要使用文件系统，定义了File 和 FileSystem 两个interface，os包只有File struct（os包没有提供 File 和 FileSystem interface）

```go
// controller-runtime/pkg/client/object.go
type Object interface {
	metav1.Object           // interface k8s.io/apimachinery/pkg/runtime/interfaces.go
	runtime.Object          // interface k8s.io/apimachinery/pkg/apis/meta/v1/meta.go
}
// k8s.io/api/core/v1/types.go
type Pod struct {
	metav1.TypeMeta 
	metav1.ObjectMeta 
	Spec PodSpec 
	Status PodStatus 
}
```
更绝得是，一个map 给扩展实现了  Registry interface。
```go
// kubernetes/pkg/scheduler/framework/runtime/registry.go
type Registry map[string]PluginFactory
// Register adds a new plugin to the registry. If a plugin with the same name exists, it returns an error.
func (r Registry) Register(name string, factory PluginFactory) error {
	if _, ok := r[name]; ok {
		return fmt.Errorf("a plugin named %v already exists", name)
	}
	r[name] = factory
	return nil
}
// Unregister removes an existing plugin from the registry. If no plugin with the provided name exists, it returns an error.
func (r Registry) Unregister(name string) error {
	if _, ok := r[name]; !ok {
		return fmt.Errorf("no plugin named %v exists", name)
	}
	delete(r, name)
	return nil
}
```
也因此呢，go 很多代码里，`var _ framework.FilterPlugin = &Sample{} ` 声明一个变量，又不用，一般是干什么作用呢？就是开发的时候验证下 Sample struct有没有完全实现 FilterPlugin/PreBindPlugin接口。
```go
var _ framework.FilterPlugin = &Sample{}
var _ framework.PreBindPlugin = &Sample{}
type Sample struct {
	handle framework.Handle
}
```

## 不得不提的面向对象

Go 不是一门面向对象的语言，但它可以模拟面向对象语言的某些功能。

[go语言设计哲学](https://studygolang.com/articles/2944)go 没有像JAVA一样，宗教式的完全面向对象设计；
1. 完全面向对象设计就是一刀切的宗教式的设计，但其并不能很好的表述这个世界，这就导致其表现力不足，最后通过设计模式和面向切面等设计技巧来弥补语言方面的缺陷； JAVA就好比：手里握着是锤子，看什么都是钉子，什么都是类的对象，这个和现实世界不符，类表示单个事物还可以，一旦表示多个事物及其交互，其表现力也就会遇到各种挑战。
2. **go是面向工程的实用主义者**，其糅合了面向对象的设计，函数式设计和过程式设计的优点；原来通过各种设计模式的设计通过函数、接口、组合等简单方式就搞定了；go有更多胶水的东西比如：全局变量、常量，函数，闭包等等，可以轻松的的把模块衔接和驱动起来；

[Is Go an object-oriented language?](https://golang.org/doc/faq#Is_Go_an_object-oriented_language)Yes and no. Although Go has types and methods and allows an object-oriented style of programming, there is no type hierarchy. The concept of “interface” in Go provides a different approach that we believe is easy to use and in some ways more general. There are also ways to embed types in other types to provide something analogous—but not identical—to subclassing. Moreover, methods in Go are more general than in C++ or Java: they can be defined for any sort of data, even built-in types such as plain, “unboxed” integers. They are not restricted to structs (classes).


[程序员技术选型：写Go还是Java？](https://mp.weixin.qq.com/s/v1jMd875d9hvfY2Y-AJO4Q)Go 不是面向对象编程语言。Go 没有类似 Java 的继承机制，因为它没有通过继承实现传统的多态性。实际上，它没有对象，只有结构体。它可以通过接口和让结构体实现接口来模拟一些面向对象特性。此外，你可以在结构体中嵌入结构体，但内部结构体无法访问外部结构体的数据和方法。Go 使用组合而不是继承将一些行为和数据组合在一起。

Go 是一种命令式语言，Java 是一种声明式语言。Go 没有依赖注入，我们需要显式地将所有东西包装在一起。因此，**在使用 Go 时尽量少用“魔法”之类的东西**。一切代码对于代码评审人员来说都应该是显而易见的。Go 程序员应该了解 Go 代码如何使用内存、文件系统和其他资源。Java 要求开发人员更多地地关注程序的业务逻辑，知道如何创建、过滤、修改和存储数据。系统底层和数据库方面的东西都是通过配置和注解来完成的（比如通过 Spring Boot 等通用框架）。我们尽可能把枯燥乏味的东西留给框架去做。这样做很方便，但控制也反转了，限制了我们优化整个过程的能力。

### 《Go codding in java way》 

评价好坏的标准只有两条：系统是不是稳定，是不是能快速响应变化。
1. go 版threadlocal
2. web开发：全局ExceptionHandler，Requestmapping
3. Stream programing
4. 不要尝试在golang中用面向对象的方法写你的业务代码。现在是微服务当道的年代，服务已经代替了类成为主要的建模工具，现在很少关注一个服务里面是不是按照ddd拆分的，类是贫血的还是充血的，接到的业务代码最后都写成了pipeline。 PS：单个模块的复杂度降低了。其次是，如果你有足够强的数据库、缓存等等，其实业务代码没多少的。
5. goburrow/cache 类似 java的guava cache
6. go-funk 类似CollectonUtils/MapUtils

## 其它

[一些设计模式的Go实现](https://mp.weixin.qq.com/s/S1BQ55yZgBlB4AkfPX-gIg)


[万字长文复盘导致Go语言成功的那些设计决策](https://mp.weixin.qq.com/s/Ca72d8-A0UoiIv-EquT8rA)Go语言提供了内置的字符串、hash map和动态大小的数组等易于使用的数据类型。如前面所述，这些对于大多数Go程序来说已经足够了。其结果是Go程序之间有了更大的互操作性--例如，没有产生竞争性的字符串或hash map的实现来分裂包的生态系统。Go包含的goroutines和channel是另一种形式的完整性。这些功能提供了现代网络程序中所需要的核心并发功能。Go直接在语言中提供这些功能，而不是在库中提供，这样可以更容易地调整语法、语义和实现，使其尽可能地轻量和易于使用，同时为所有用户提供统一的方法。

[Go语言之道](https://mp.weixin.qq.com/s/ZwbTcaRFwDhOA6hMFToIxw)

[Docker 之父：Go、Rust 为什么会成为云原生的主导语言？](https://mp.weixin.qq.com/s/yg_gqi7bMldhi95ZAIR9fA)
1. 我们之前都是用 Python 和 C 编写分布式系统的开发人员，对 Python 在实际生产中的应用已经非常熟悉了，所以大家都很讨厌 Python 的类型问题。最终，这些本该被早点发现的问题就暴露在运行时中，再也没法更改。
2. 我们设想中的 Docker 不仅会是款成功的工具，还会是个成功的开源项目，因此语言的选择对于后期建立社区非常重要。比方说，我们得保证选择的语言有足够高的人气，保证语言本身不会成为理解源代码、贡献新代码的障碍。Go 的好处就是它的语法比较平易近人，会写 C 或者 Python 的人肯定能很快上手 Go。


[Golang与Java全方位对比总结](https://mp.weixin.qq.com/s/-N4eqdXb9a93uvOWfE4ScQ)Java在运行时相比Golang多占用了一些内存。原因在于：
1. Java运行态中包含了一个完整的解释器、一个JIT编译期以及一个垃圾回收器，这会显著地增加内存。
2. Golang语言直接编译到机器码，**运行态只包含机器码和一个垃圾回收器**。
因此Golang的运行态相对消耗内存较少。

![](/public/upload/go/go_first_class.jpg)