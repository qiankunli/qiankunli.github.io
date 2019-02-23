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

[函数式思维:为什么函数式编程越来越受关注](https://www.ibm.com/developerworks/cn/java/j-ft20/index.html)

1. 因为runtime的能力变得更强：一个猜测是：比如并发执行for循环，那么如果还在语言中写for循环的话，jvm就不知道是否可以并发执行）
2. 并且语言获得了更强大的抽象（比如异步future、缓存等下沉到语言层面）。大家想下，一个异步框架 给你的抽象 是不是实现一个handler 就行了。当语言将这个框架的能力 集成进来以后，可以将这个 handler 的接口做的更简洁。

所以开发世界变得更加函数化，这使开发人员可以花费更多的时间来思考结果的影响，而不是思考如何生成结果。比如以前语言不支持 异步也不用框架的话，就难免异步代码 和 业务代码 混淆在一起。由于高阶函数等抽象出现在语言中，它们将成为高度优化的操作的自定义机制。

因此，**函数式编程的流行，跟语言的发展有关系，以前不流行，不是不想，是语言暂时没办法提供那个抽象能力**。

||简化|
|---|---|
|c/c++ ==> java|内存管理|
|java ==> scala|类型推断，迭代|

[A Scala Tutorial for Java programmers](http://www.scala-lang.org/docu/files/ScalaTutorial-zh_CN.pdf)

[怎样最高效地学习Scala](http://blog.csdn.net/chszs/article/details/51693175)

**scala将面向对象和函数式编程糅合在了一起，因此我们讨论scala时，主要也从这两个方面入手。**

## 面向对象特性/和java异同点

共同点

1. 基本也是一个类一个文件，每个类包含成员和方法。如何分解一个复杂的业务逻辑，是学习一门语言的第一课。

不同点

1. 方法嵌套
2. 泛型的表示不同
3. 类型系统 [浅谈编程语言的类型系统](http://blog.csdn.net/ce123_zhouwei/article/details/8976652) 

	* 丰富的模式匹配支持，Pattern matching is a mechanism for checking a value（可以是任何类型） against a pattern
	* Any、AnyRef、AnyVal
	* bottom type：Null、Nothing
	
4. 类和对象
	
	* case class和enum
	* object 与静态类

5. 在scala中，主构造器（还有辅助构造器）是整个类体，构造器所需的所有参数都被罗列在类名称后面。
6. trait和java interface的异同
7. akka和golang并发机制的异同

代码的简写

	class Name(var value:String)
	
从概念上讲，上述代码和下面的代码是等价的：

	class Name(s:String){
		private var _value:String =s
		def value:String = _value	// get方法
		def value_= (newValue:String):Unit = _value=newValue // set方法
	}

`def value_= (newValue:String):Unit = _value=newValue`，其中`value_=`是一个方法名（绝不绝），该名的函数类型是`(newValue:String):Unit`，函数的具体实现是`_value=newValue`


[浅谈编程语言的类型系统](http://blog.csdn.net/ce123_zhouwei/article/details/8976652) 的基本内容

1. 编程语言的本质在于回答两个问题：如何表示信息；如何处理信息
2. 宇宙虽然鬼斧神工，丰富多彩，但是在微观上，整个世界仅仅是由少数寥寥几种基本粒子构成的。程序繁杂的外表之下，骨子里都是由一些“基本粒子”，按照一定的组合方式构成的。那么究竟有哪些基本粒子，又允许进行何种组合？一门语言定义了一套基本类型的“集合”，这个集合就作为一个整体被称为类型系统。
3. 坦白讲，“系统”是一个非常模糊的概念。我们会说操作系统、消化系统、生态系统……各种各样的系统，然而对于系统本身是什么，在不同的科学领域有截然不同的定义。通常我们所说的系统中，**存在一些基本要素**（软件模块、细胞、物种等等），然后**存在一定的相互作用关系**（函数调用、细胞连接、捕食与被捕食等等），在此基础上**实现一定的功能**（完成金融计算、排解人体毒素、完成有机物的自然循环等等）。那么我们就把这些基本元素，以及其构成方式，统称为一个系统。

[函数式编程](http://qiankunli.github.io/2018/09/12/functional_programming.html)

## 其它

对于scala和go来说，通过variable:Type来声明变量类型，方法也是一种变量，方法的类型就是方法的返回值。

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