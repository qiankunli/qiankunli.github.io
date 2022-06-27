---

layout: post
title: Scala初识
category: 技术
tags: Scala
keywords: Scala 

---

## 前言

* TOC
{:toc}

阅读材料

[从Java走进Scala（Scala经典读物）](http://developer.51cto.com/art/200909/154717.htm)

[Scala 初学指南](https://www.gitbook.com/book/windor/beginners-guide-to-scala/details)

[A Scala Tutorial for Java programmers](http://www.scala-lang.org/docu/files/ScalaTutorial-zh_CN.pdf)

[怎样最高效地学习Scala](http://blog.csdn.net/chszs/article/details/51693175)

## 为什么会有Scala

[Strategic Scala Style: Principle of Least Power](http://www.lihaoyi.com/post/StrategicScalaStylePrincipleofLeastPower.html) ， Scala的语言特性不算多，但是语言特性之间过于正交，一方面你把语言特性组合起来之后可以变得很复杂，写出各种其他语言的范式，另一方面容易玩脱。不想玩脱的话，就得优先选用功能最弱的功能（尽量少用scala的高级特性）。只在弱的功能解决不了你面临的问题时，才用更强的功能。Scala的创始人Martin Odersky也在李浩毅博客下面举双手赞成。

《Scala实用指南》Scala不是难，而是很难，难在缺乏一本浅近易学、循序渐进的图书。社区有时候弥漫的风气会让你觉得：代码写得太平实，就不能表现出真正的实力。因此，无论国内还是国外，大家热衷分享的都是一些第一眼看过去不知所云，第二眼看过去竟会让你不知所措的代码片段。而我要说的是，这并非日常，也并不值得推崇。**我个人比较推崇编写贴近Java风格的Scala代码，并适度地利用Scala的语言特性简化代码**。因为Java语言表现力有限，所以我们需要使用各种设计模式提高代码的抽象能力，固化编码逻辑。Scala这门语言在设计之初就借鉴了大量现存的语言特性，并吸取了许多设计模式中的精华，因此表现力非常强大。

典型的企业级应用受困于各种问题——烦琐的代码难以维护，可变性增加了程序出错的可能，而共享的可变状态也让并发编程的乐趣变成了炼狱。我们一再深陷主流编程语言拙劣抽象能力的泥潭中。Scala引入了合理的特性并规避了一些陷阱。Scala及其类库让我们能够更多地关注问题领域，而不是陷入各种底层基础设施（如多线程与同步）实现细节的泥沼之中。

Scala并不拘泥于一种编程风格。我们可以面向对象编程，也可以使用函数式风格，甚至可以结合两者的优点将它们混合使用。
1. 面向对象编程是Java程序员熟悉的舒适区。Scala是面向对象和静态类型的，并在这两方面都比Java走得更远。对于初学Scala的我们，这是个好消息，因为我们在面向对象编程上多年的投入不会浪费，而是化作宝贵的经验红利。在创建传统的应用程序时，我们可以倾向于使用Scala提供的面向对象风格。我们可以像使用Java那样编写代码，利用抽象、封装、继承尤其是多态的能力。与此同时，当这些能力无法满足需求时，我们也并不受限于这种编程模型。
2. 函数式编程风格越来越受关注，而Scala也已支持这种风格。使用Scala，我们更容易趋向不可变性，创建纯函数，降低不可预期的复杂度，并且应用函数的组合和惰性求值（lazy evaluation）策略。**在函数式风格的助益下**，我们可以用Scala创建高性能的单线程和多线程应用程序。

《Scala并发编程》并发计算无处不在。随着消费者市场中多核处理器的崛起，人们对并发计算的需求已经在开发者世界中掀起巨大波澜。并发计算其实涉及多方面的问题，难以在单一的形式化体系中很好地表达出来。**也许并不存在解决所有并发问题的高招**，对不同的需求需要采用不同的方案。比如，利用异步计算来响应事件和数据流，在消息通信时使用自发而独立的实体，为状态可变的数据中心定义事务，或者用并发计算提高性能。每一个任务都有着相应的更为合适的抽象方式：Future、响应式流（reactivestream）、角色、事务性内存或并行容器。**于是有了 Scala 和本书**。并发计算中有用的抽象模型如此之多，将它们拼凑到一门语言中似乎并不明智。不过，Scala的目标就是让用户能在编码时更方便地定义各种高层抽象，并以此来构建代码库。


## 基本语法

Scala还提供了一些Java中不支持的特性，如元组、多重赋值、命名参数、默认值、隐式参数、多行字符串、字符串插值以及更加灵活的访问修饰符。

### 数据类型

1. Scala是一门静态类型的编程语言，通过静态类型，编译器充当了抵御错误的第一道防线。它们可以验证当前的对象是否就是想要的类型。这是一种在编译时强制接口约定的方式。不幸的是，在一些主流的静态类型编程语言中，使用静态类型就意味着更多的手指键入。Scala是一门静态类型的编程语言，但是值得庆幸的是，它偏向于使用类型推断。在绝大多数的情况下，我们都不必提及类型信息——Scala将智能地从上下文中推断出必要的细节。与此同时，它也没有进行过度的类型推断，从而导致晦涩难懂或者难以维护代码。
2. 在Java中是先指定变量的类型，然后是变量名，而在Scala中，恰好做了相反的操作，这样做有两个原因：首先，通过要求将类型放在变量名之后，Scala暗示，选择一个好的变量名比标注类型更加重要；其次，类型信息是可选的。
6. Java的世界观是分裂的——其原始类型（如int和double）和对象截然不同。从Java 5开始，利用自动装箱（autoboxing）机制，可以将原始类型视为对象。然而，Java的原始类型不允许方法调用，如`2.toString()`。另外，自动装箱还涉及类型转换的开销，会带来一些负面的影响。和Java不同，Scala将所有的类型都视为对象。这就意味着，和调用对象上的方法一样，也可以在字面量上进行方法调用。**Scala会自动应用intWrapper()方法将Int转换为scala.runtime.RichInt**。诸如RichInt、RichDouble和RichBoolean这些类，可称为富包装类（rich wrapperclass）。Scala能够自动将String转化为scala.runtime.RichString。这种转换给String新增了一些有用的方法，如capitalize()、lines()和reverse()方法。PS：比Java 把int 转为Integer 提供了更多便捷的方法
7. 字符串插值。在Java中以输出或者消息的形式创建一个字符串非常麻烦，比如`String message = String.format("A discount of %d has been applied",discount)`，对应到Scala `val message = s"A discount of %discount has been applied"`，在双引号前面的s的意思是s插值器（s-interpolator），它会找到字符串中的表达式，并将其替换成对应的值。**在字符串声明处的作用域中**的任何变量都可以在表达式中使用（PS：作用域内搜索跟隐式变量有点像）。对于更复杂的表达式，可以把它们放在大括号中
    ```
    val price = 90
    val totalPrice = s"The amount of discount is ${price * discount / 100} dollars"
    ```
    为了对输出做格式化，可以使用f插值器（f-interpolator），Scala还有一个raw插值器（raw-interpolator），它会把其中的表达式换成值，但是会保留任何不可打印的字符，如换行符。除了这3个内置的插值器，你还可以创建自定义的插值器（涉及到隐式类）
3. 假设我们想要一个列表，其元素是原始列表中的值的两倍。 
    ```scala
    val values = List(1,2,3)
    val doubleValues = values.map(_ * 2) # _ * 2是一个匿名函数，下划线（_）表示传递给此函数的参数，函数本身作为参数值传递给了map()函数。
    ```
4. Scala要求变量在使用前必须初始化，如果初始化时不确定值，可以用下划线初始化var变量，表示相应类型的默认值。用val声明的变量就没法使用下划线这种方便的初始化方法了，因为val变量创建后就无法修改了。
3. 类型系统 [浅谈编程语言的类型系统](http://blog.csdn.net/ce123_zhouwei/article/details/8976652) 尽管可以在Scala中使用Java的任何类型，但同时也可以享受到由Scala提供的一些原生类型。
	
	* Any、AnyRef、AnyVal，Any类型是所有类型的超类型，Any类型的直接后裔是AnyVal和AnyRef类型。AnyVal是Scala中所有值类型（如Int、Double等）的基础类型，并映射到了Java中的原始类型，而AnyRef是所有引用类型的基础类型。尽管AnyVal没有什么额外的方法，但是AnyRef包含了Java的Object的方法，如notify()、wait()、finalize()等。
	* bottom type：Null、Nothing，Nothing是一切类型的子类型。
    * 当一个函数调用的结果可能存在也可能不存在时，Option类型很有用。有时候，你可能希望从一个函数中返回两种不同类型的值之一，Scala的Either类型就派上用场了。

### 函数

4. 为参数提供默认值。在Java中，我们可以用重载方法的方式省略一个或者多个参数，以达到灵活的效果。
    ```scala
    def mail(destination: String = "head office",mailClass: String = "first"): Unit = println(s"sending to $destination by $mailClass class")
    # 调用
    mail()
    mail("Boston office")
    mail("Houston office","Priority")
    ```
    为省略的参数补上默认值这个操作是在编译时完成的。不过在重载方法的时候，需要特别小心。如果一个方法在基类中用了一个默认值，而在其派生类的相应重载方法中却使用了另一个默认值，就会让人感到困惑，到底选用哪个默认值。对于多参数的方法，如果对于其中一个参数，你选择使用它的默认值，你就不得不让这个参数后面的所有参数都使用默认值。当然，使用命名参数可以打破这个限制。PS：命名参数提高了代码的可读性。
5. 隐式参数，默认值是由函数的创建者决定的，而不是由调用者决定。Scala还提供另外一种赋默认值的方法，可以由调用者来决定所传递的默认值，而不是由函数的定义者来决定。我们来看一个利用隐式参数的例子。我们随身携带着各种智能手机和移动设备，它们总是需要连接不同的网络：家庭网络、办公网络、机场候机厅的公共网络等。我们的操作是相同的——连接到一个网络，但是我们所连接的网络依赖于我们所处的环境。我们不想每一次都去指定网络，这很无聊。与此同时，我们也不希望每一次都是同一个默认值生效。这时，我们可以用一个名为隐式参数的特殊参数来解决这个问题。
8. **句末分号在Scala中是可选的**；根据上下文，**点操作符（.）和括号都是可选的**。因此，我们可以编写`s1 equals s2`来替代`s1.equals(s2)`。通过去除分号、点号和括号，代码获得了较高的**信噪比**，使得编写领域特定语言变得更加容易。Scala程序员把空括号视为噪声，如果方法没有参数，或者只有一个参数，就可以省略点号（.）和括号。如果一个方法带多个参数，则必须使用括号，但点号仍然是可选的。
1. 返回值类型推断，只有当你使用等号（=）将方法的声明和方法的主体部分区分开时，Scala的返回值类型推断才会生效。否则，该方法将会被视为返回一个Unit，等效于Java中的Void。如果一个函数的主体是一个简单表达式或者复合表达式，就可以删除大括号。
10. 元组是一个不可变的对象序列，创建时使用逗号分隔。例如，`("Venkat","Subramaniam","venkats@agiledeveloper.com")`表示一个3个对象的元组。可以采用info._1这种语法形式访问其中的第一个元素，第二个元素则是info._2，以此类推。下划线加数字这种模式，如_1，表示我们在元组中想访问的元素的索引或位置。与集合不同，**访问元组的索引是从1开始的**。另一个和集合的差异点在于，如果指定的索引越界，则会在编译期而不是在运行时出错。
11. 在Java中，方法可以接受多个参数，但是只能返回一个结果。在Java中返回多个结果需要使用拙劣的变通方案。**Scala的元组与多重赋值（multiple assignment）结合，可以将返回多个值变成小菜一碟**。在并发编程时，Actor之间也将元组以数据值列表的形式作为消息进行传递，而且**元组的不可变性正好契合这种场景**。PS：这也是为什么元组这种结果 在语言中有存在价值？
11. return是可选的。假定最后一个求值的表达式能够匹配方法所声明的返回类型，那么这个表达式的求值结果将会自动作为方法调用的结果值返回。
13. 操作符重载，技术上说，Scala没有操作符，所以操作符重载的意思就是重载诸如+、-等符号在Scala中，这些都是方法名。操作符利用了Scala宽松的方法调用语法——Scala不强制在对象引用和方法名中间使用点号（.）。**这两个特性结合在一起就给人一种操作符重载的幻觉**。因此，当调用ref1 + ref2，实际上写的是ref1.+(ref2)，是在ref1上面调用+()方法。
    ```scala
    class Complex(val real:Int,val imaginary:Int){
        def +(operand: Complex):Complex = {
            new Complex(real + operand.real,imaginary + operand.real)
        }
    }
    ```
14. Scala没有操作符这个事实非常有趣。然而，没有操作符并不能免去处理操作符优先级的需要。Scala没有在操作符上定义优先级，但是它在方法上定义了优先级。方法的第一个字符用来决定它们的优先级。如果在一个表达式中两个字符的优先级相同，那么在左边的方法优先级更高。

## 面向对象

1. 类和方法默认就是公开的，所以你无须显式使用public关键字。
2. Scala默认会导入两个包、scala.Predef对象以及它们相应的类和成员。只用类名就可以从这些预导入的包中引用相应的类。因为java.lang已经自动导入，所以无须额外的导入就可以在脚本中使用通用的Java类型。
3. 通常，Java的包中只含有接口、类、枚举和注解类型。Scala更进一步，包中还可以有变量和函数。它们都被放在一个称为包对象（package object）的特殊的单例对象中。如果你发现自己创建一个类，仅仅是为了保留在同一个包中的其他类之间共享的一组方法，那么包对象就能避免创建并重复引用这样一个额外的类的负担。Scala会在相应的包中将包对象编译为名为package的类。


### 构造器

在scala中，主构造器（还有辅助构造器）是整个类体，构造器所需的所有参数都被罗列在类名称后面。从概念上讲，`class Name(var value:String)`（如果类定义没有主体，就没有必要使用大括号）和下面的代码是等价的：

```scala
class Name(s:String){
	private var _value:String =s
	def value:String = _value							 // get方法
	def value_= (newValue:String):Unit = _value=newValue // set方法
}
```

其中set 方法`def value_= (newValue:String):Unit = _value=newValue`，其中`value_=`是一个方法名（绝不绝），该名的函数类型是`(newValue:String):Unit`，函数的具体实现是`_value=newValue`

Scala会执行主构造器中任意表达式和直接内置在类定义中的可执行语句。


### 伴生类和伴生对象

Java代码中通常充斥着很多样板代码——getter、setter、访问修饰符、处理受检异常的代码等。Scala具有非常高的**代码密度**——输入少量代码就可以完成许多功能。

**伴生对象是伴随一个类的单例**，单例是一种非常常用的设计模式，但在Java中其实很难实现，幸运的是，**在Scala中这个问题在编程语言层面就已经解决了**。创建一个单例要使用关键字object而不是class。因为不能实例化一个单例对象，所以不能传递参数给它的构造器。

*  using parentheses(圆括号) on the instance of a class actually calls the apply method defined on this class. This approach is widely used in the standard library as well as in third-party libraries.`val joe = new Person("zhangsan")`，joe.apply() 等同于 `joe()`. **apply 方法是一个语法糖**。
*  Scala doesn’t have the static keyword but it does have syntax for defining singletons. **If you need to define methods or values that can be accessed on a type rather than an instance**, use the object keyword.
	```scala
	object RandomUtils {
		def random100 = Math.round(Math.random * 100)
	}
	```
	After RandomUtils is defined this way, you will be able to use method random100 without creating any instances of the class: RandomUtils.random100
*  If an object has the same name as a class, then it’s called a companion object. Companion objects are often used in Scala for defining additional methods and implicit values. 

伴生类和伴生对象：由于static定义的类和对象破坏了 面向对象编程的规范完整性，因此scala 在设计之初就没有static关键字概念，类相关的静态属性都放在伴生对象object中。当同一个文件内同时存在`object x`和`class x`的声明时：我们称`class x`称作`object x`的伴生类。其`object x`称作`class x`的伴生对象。
1. 使用object 随便定义一个独立对象（没有类和它重名）称为独立对象，单纯只是一个单例模式类。可以选择将一个单例关联到一个类，其名字和对应类的名字一致，因此被称为伴生对象（companion object）。相应的类被称为伴生类。
2. 类和伴生对象之间没有界限——它们可以互相访问彼此的private字段和private方法。
4. 只有伴生对象中可以定义main函数，类似于static修饰

```scala
object Greeter{
    def greet(): Unit = println("hello world")
}
```
在字节码层面上，单例中方法会被创建为static方法。这从与Java的互操作性上讲，是一个好消息。
```java
public final class Greeter{
    public static void greet (){
        ...
    }
}
```

### case 和 锅炉板模式

[Java正在“Kotlin化”](https://mp.weixin.qq.com/s/ut6l7ipdkN3O-9rIELuEcQ)Java record 是我们长期以来一直要求的一项特性，我相信你早就多次遇到这样的场景了，那就是极不情愿地实现 toString、hashCode、equals 方法以及每个字段的 getter。Kotlin 提供了数据类（data class）来解决这个问题，Java 也通过发布 record 类来解决了这个问题，同样的问题，Scala 是通过 case 类来解决的。这些类的主要目的是在对象中保存不可变的数据。PS：**有点类似DDD中的值对象。或者说，语言设计正在想业务模型的需要靠拢**。

[Boilerplate code](https://en.wikipedia.org/wiki/Boilerplate_code)

In computer programming, boilerplate code or boilerplate refers to sections of code that have to be included in many places with little or no alteration. It is often used when referring to languages that are considered verbose, i.e. the programmer must write a lot of code to do minimal jobs.

比如，setter/getter 就是典型的锅炉板代码，尽管每个类的setter/getter 细节不一样，但setter/getter 代码 占了Pet 类代码量的一半。

```java
public class Pet {
    private String name;
    private Person owner;
    public Pet(String name, Person owner) {
        this.name = name;
        this.owner = owner;
    }
    public String getName() {
        return name;
    }
    public void setName(String name) {
        this.name = name;
    }
    public Person getOwner() {
        return owner;
    }
    public void setOwner(Person owner) {
        this.owner = owner;
    }
}
```

To reduce the amount of boilerplate, many frameworks have been developed, e.g. Lombok for Java.

```java	
@AllArgsConstructor
@Getter
@Setter
public class Pet {
    private String name;
    private Person owner;
}
```
	
scala 就更简洁了

```scala
case class Pet(var name: String, var owner: Person)
```

从jdk14 开始支持 java record 也有类似效果

```java
public record EmployeeRecord(String firstName, String surname,
 int age, AddressRecord address, double salary) {  
}
```

boilerplate code 理念扩展下层次 就成了 boilerplate pattern， 比如配置中心、服务发现等。单纯配置中心或服务发现功能需要大量的代码，但对于不同业务方来说，不同之处可能就是几个参数，这也是spring cloud的重要理念。

case class 和 case object可以和match配合使用

## 函数式编程

1. 函数是first-class成员，我们可以用代码块来增强另一个函数的功能，可以用代码块来指定谓词、查询以及在方法中实现逻辑约束，可以用代码块来改变方法的控制流，如遍历集合。
6. trait和java interface的异同，trait本质上就是组合，只不过是一种在语法上更简洁一些的语法糖。例如，要做一个关于朋友的抽象建模，我们可以将一个Friend trait混入任何的类中，如Man、Woman、Dog等，而又不必让所有这些类都继承同一个公共基类（狗是人类的好朋友，但我们不能为此让Dog继承Human）。trait类似于一个带有部分实现的接口，trait可以是独立的，也可以扩展自某个类。我们在trait中定义并初始化的val和var变量，将会在混入了该trait的类的内部被实现。任何已定义但未被初始化的val和var变量都被认为是抽象的，混入这些trait的类需要实现它们。
    1. 如果一个类没有扩展任何其他类，则使用extends关键字来混入trait。我们可以混入任意数量的trait。如果要混入额外的trait，要使用with关键字。
    2. 如果一个类已经扩展了另外一个类，那么我们也可以使用with关键字来混入第一个trait。
   trait要求混入了它们的类去实现在特质中已经声明但尚未初始化的（抽象的）变量（val和var）。其次，trait的构造器不能接受任何参数。trait连同对应的实现类被编译为Java中的接口，实现类中保存了trait中已经实现的所有方法.
7. akka和golang并发机制的异同
4. 丰富的模式匹配支持，Pattern matching is a mechanism for checking a value（可以是任何类型） against a pattern


## 构建工具/sbt

[SBT构建一个基本工程](http://www.jianshu.com/p/db903ad4781d)

从sbt 的build 文件名`build.sbt`就可以 看到 sbt 和 ant（ant是 build.xml） 的近亲关系。

## 应用框架/play

[给Java开发者的Play Framework(2.4)介绍 Part1：Play的优缺点以及适用场景](http://skaka.me/blog/2015/07/27/play1/)

文章重点：JVM阵营在Web领域逐渐落后主要有三个原因：编译的锅，技术栈的锅和语言的锅。

1. Java源代码需要编译之后才能运行，直接结果是每次修改源代码都需要重启Web服务器才能看到效果。
2. Servlet技术栈： Web容器（比如tomcat）存在的必要性也被越来越多的人质疑。原因就在于人为的将应用与容器剥离， 虽然这种做法本意是好的，但是结果就是给开发测试部署带来一系列集成的问题，现在越来越多的项目开始使用内嵌的Jetty或Tomcat。
3. 基于Netty实现了自己的 请求响应接口（Request/Result），Promise已经被Scala，JavaScript等语言大量使用，Actor模型也已经遍地开花， 这些你都可以直接在Play中使用

笔者读完这一段的反应就是：原先感觉还不错的java技术栈竟被说的这么不堪。

[玩转 Java Web 应用开发：Play 框架](https://www.ibm.com/developerworks/cn/java/j-lo-play/)


[Scala Cookbook 关于xml、test和play的pdf](https://resources.oreilly.com/examples/9781449339616-files/blob/master/Scala_Cookbook_bonus_chapters.pdf)

1.  A terrific thing about the Play architecture is that Play templates are compiled to Scala functions

## slick

[浅谈Slick（1）－ 基本功能描述](http://www.cnblogs.com/tiger-xc/p/5891758.html)


## 小结

《scala程序设计》：scala在架构层面提倡的方法是：小处用函数式编程，大处用面向对象编程。用函数式实现算法、操作数据、以及规范的管理状态，是减少bug、压缩代码行数和降低项目延期风险的最好办法。另一方面，scala的oo模型提供很多工具，可用来设计可组合可复用的模块，这对于较大的应用程序是必不可少的。


项目推荐

[快学Scala+Playframework之增删改查——项目搭建（一）](https://beacelee.com/post/play-framework-scala-userlist.html) 对应项目代码
[BeAce/scala-and-playframework-userlist](https://github.com/BeAce/scala-and-playframework-userlist)

[Play framework, Slick and MySQL Tutorial](http://pedrorijo.com/blog/play-slick/) 对应项目代码 [pedrorijo91/play-slick3-steps](https://github.com/pedrorijo91/play-slick3-steps/tree/step2)

