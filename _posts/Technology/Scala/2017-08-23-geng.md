---

layout: post
title: Scala的一些梗
category: 技术
tags: Scala
keywords: Scala 

---

## 前言

1. object and class

	*  using parentheses on the instance of a class actually calls the apply method defined on this class. This approach is widely used in the standard library as well as in third-party libraries.`val joe = new Person("zhangsan")`，joe.apply() 等同于 `joe()`
	*  Scala doesn’t have the static keyword but it does have syntax for defining singletons14. **If you need to define methods or values that can be accessed on a type rather than an instance**, use the object keyword.`object RandomUtils {def random100 = Math.round(Math.random * 100)}` After RandomUtils is defined this way, you will be able to use method random100 without creating any instances of the class: RandomUtils.random100
	*  If an object has the same name as a class, then it’s called a companion object. Companion objects are often used in Scala for defining additional methods and implicit values. You will see a concrete example when we get to serializing objects into JSON.多做了一些工作，但代码中呈现的名字是一样的。
	*  称呼泛型为Type parametrization或 parametrized class，这个比较贴切
2. 代码简化

	* String interpolation，在字符串中使用变量，scala会自动完成替换
	* 调用函数时不写参数，函数参数若有默认值，调用函数时可以不写参数。函数参数设置为implicit，相同scope下有implicit变量，调用函数时可以不写参数。
	* yield [Scala 的 yield 例子 (for 循环和 yield 的例子)](https://unmi.cc/scala-yield-samples-for-loop/)
	* try、for都有返回值

3. 函数编程

	* If your function accepts only one argument, then you are free to call your function using curly braces instead of parentheses
	* `def sum(a: Int, b: Int): Int = a + b` =Currying=> `def sum(a: Int)(b: Int): Int = a + b`

## 其它

[函数式思维和函数式编程](http://www.vaikan.com/programming-thinking-functional-way/)