---

layout: post
title: 函数式编程
category: 技术
tags: Architecture
keywords: functional programming

---

## 简介（未完成）

## 函数式编程

|||函数的概念|其它|
|---|---|---|---|
|函数式编程||一种东西和另一种东西之间的对应关系。|至于什么柯里化、什么数据不可变，都只是外延体现而已。|
|命令式编程|面向过程、面向对象等|一个步骤||

	object HelloWorld  {  
	  def main(args: Array[String]): unit = {  
	    args.filter( (arg:String) => arg.startsWith("G") )  
	        .foreach( (arg:String) => Console.println("Found " + arg) )  
	  }  
	} 
	
函数式的代码是“对映射的描述”，查看scala中filter函数的源码，或许更能体会对映射的描述的感觉。

	class Array[A]{  
	    // ...  
	   def filter  (p : (A) => Boolean) : Array[A] = ... // not shown  
	} 


[什么是函数式编程思维？](https://www.zhihu.com/question/28292740/answer/100284611)

当我在说函数也是一个对象时，我在说什么：函数作为参数进行传递、把它们存贮在变量中、或者当作另一个函数的返回值

**函数不是方法**，方法指的是类或对象中定义的函数，**方法中的this引用会隐性的指向某一对象**。比如`(s:String) => s.toUpperCase()`便是一个函数，它是独立的，没有通过this与一个对象相关联。函数不是方法，但在某些时候，会将方法归入函数。

scala函数中，有偏函数一说，限定了函数的输入。还有柯里化函数，A way to write functions with multiple parameter lists. For instance  
`def f(x: Int)(y: Int)` is a curried function with two parameter  
lists. A curried function is applied by passing several arguments  
lists, as in: f(3)(4). However, it is also possible to write a partial  
application of a curried function, such as f(3).  

比如`"hello world"`可以称为String字面量，那么`(s:String) => s.toUpperCase`可以称为函数字面量。

函数没有副作用，比如没有全局对象被修改。当一个函数采用其它函数作为变量或返回值时，它被称为高阶函数。

函数可以内嵌。

一个方法的极简定义`def id = Match.random`，完整形式`def def():Unit = Match.random`

定一个函数类型的值（注意是函数不是方法）`var greetVar:String => String = (name) => "Hello" + name`， 分为三个部分
 
1. var greetVar
2. 类型 String => String
3. "Hello" + name

一个方法可以赋给一个函数类型的变量`val gr = greet _`，此处greet是一个方法名。

在该小节中，笔者因为水平和对scala的认识有限，还未能精确描述函数式编程的精髓

1. 从基本组成和相互关系的角度来描述，面向对象的基本组成是类和对象，基本组成的相互关系是继承、聚合等。函数式编程基本组成是函数、偏函数等，函数与函数关系可以是调用、内嵌、返回等。**当然，面向对象和函数编程是两个维度的东西，对比不太合适。**
2. 笔者学习spark的时候，了解到，你对data set的一系列函数式调用，并不会被立即执行。比如对一个数据集，你先过滤男性、又将值翻番。从执行角度看，不会是把数据拿出来，先for循环一把过滤，再for循环一把每个值乘以2。一系列的函数调用更像是一个执行计划，spark会优化这些计划的执行，最终可能一次for循环就好了。


