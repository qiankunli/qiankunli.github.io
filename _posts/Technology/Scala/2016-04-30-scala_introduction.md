---

layout: post
title: Scala初识
category: 技术
tags: Scala
keywords: Scala 

---

## 前言

阅读材料

[从Java走进Scala（Scala经典读物）](http://developer.51cto.com/art/200909/154717.htm)

[Scala 初学指南](https://www.gitbook.com/book/windor/beginners-guide-to-scala/details)

[A Scala Tutorial for Java programmers](http://www.scala-lang.org/docu/files/ScalaTutorial-zh_CN.pdf)

[怎样最高效地学习Scala](http://blog.csdn.net/chszs/article/details/51693175)

[Strategic Scala Style: Principle of Least Power](http://www.lihaoyi.com/post/StrategicScalaStylePrincipleofLeastPower.html) ， Scala的语言特性不算多，但是语言特性之间过于正交，一方面你把语言特性组合起来之后可以变得很复杂，写出各种其他语言的范式，另一方面容易玩脱。不想玩脱的话，就得优先选用功能最弱的功能（尽量少用scala的高级特性）。只在弱的功能解决不了你面临的问题时，才用更强的功能。Scala的创始人Martin Odersky也在李浩毅博客下面举双手赞成

## 一些概念

共同点

1. 基本也是一个类一个文件，每个类包含成员和方法。如何分解一个复杂的业务逻辑，是学习一门语言的第一课。

不同点

1. 方法嵌套
2. 泛型的表示不同
3. 类型系统 [浅谈编程语言的类型系统](http://blog.csdn.net/ce123_zhouwei/article/details/8976652) 

	* 丰富的模式匹配支持，Pattern matching is a mechanism for checking a value（可以是任何类型） against a pattern
	* Any、AnyRef、AnyVal
	* bottom type：Null、Nothing
6. trait和java interface的异同
7. akka和golang并发机制的异同

### 构造器

在scala中，主构造器（还有辅助构造器）是整个类体，构造器所需的所有参数都被罗列在类名称后面。从概念上讲，`class Name(var value:String)`和下面的代码是等价的：

```scala
class Name(s:String){
	private var _value:String =s
	def value:String = _value							 // get方法
	def value_= (newValue:String):Unit = _value=newValue // set方法
}
```


`def value_= (newValue:String):Unit = _value=newValue`，其中`value_=`是一个方法名（绝不绝），该名的函数类型是`(newValue:String):Unit`，函数的具体实现是`_value=newValue`

### 伴生类和伴生对象

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
1. 伴生类和伴生对象需要同名。
2. 类和伴生对象之间没有界限——它们可以互相访问彼此的private字段和private方法。
3. 没有class，只有object则是单例模式类。
4. 只有伴生对象中可以定义main函数，类似于static修饰

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

## sbt

[SBT构建一个基本工程](http://www.jianshu.com/p/db903ad4781d)

从sbt 的build 文件名`build.sbt`就可以 看到 sbt 和 ant（ant是 build.xml） 的近亲关系。

## play

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

就scala的学习路线来说，笔者还是建议粗略的了解语法后，尽快转入源码学习和项目实践。

项目推荐

[快学Scala+Playframework之增删改查——项目搭建（一）](https://beacelee.com/post/play-framework-scala-userlist.html) 对应项目代码
[BeAce/scala-and-playframework-userlist](https://github.com/BeAce/scala-and-playframework-userlist)

[Play framework, Slick and MySQL Tutorial](http://pedrorijo.com/blog/play-slick/) 对应项目代码 [pedrorijo91/play-slick3-steps](https://github.com/pedrorijo91/play-slick3-steps/tree/step2)

## 引用

[Scala 初学指南](https://windor.gitbooks.io/beginners-guide-to-scala/content/index.html)