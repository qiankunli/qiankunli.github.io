---

layout: post
title: Scala初识
category: 技术
tags: Scala
keywords: Scala 

---

## 前言（未完待续）

阅读材料

[从Java走进Scala（Scala经典读物）](http://developer.51cto.com/art/200909/154717.htm)


||简化|
|---|---|
|c/c++ ==> java|内存管理|
|java ==> scala|类型推断|

## 和java的一些共同点

1. 基本也是一个类一个文件，每个类包含成员和方法。如何分解一个复杂的业务逻辑，是学习一门语言的第一课。

## 函数式编程

|||函数的概念|其它|
|---|---|---|---|
|函数式编程||一种东西和另一种东西之间的对应关系。|至于什么科里化、什么数据不可变，都只是外延体现而已。|
|命令式编程|面向过程、面向对象等|一个步骤||

	object HelloWorld  
	{  
	  def main(args: Array[String]): unit = {  
	    args.filter( (arg:String) => arg.startsWith("G") )  
	        .foreach( (arg:String) => Console.println("Found " + arg) )  
	  }  
	} 
	
函数式的代码是“对映射的描述”，查看scala中filter函数的源码，或许更能体会对映射的描述的感觉。

	class Array[A]  
	{  
	    // ...  
	   def filter  (p : (A) => Boolean) : Array[A] = ... // not shown  
	} 


[什么是函数式编程思维？](https://www.zhihu.com/question/28292740/answer/100284611)

## 其它

对于scala和go来说，通过variable:Type来声明变量类型，方法也是一种变量，方法的类型就是方法的返回值。

	

## 引用

https://windor.gitbooks.io/beginners-guide-to-scala/content/index.html