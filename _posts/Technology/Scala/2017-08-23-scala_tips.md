---

layout: post
title: Scala的一些梗
category: 技术
tags: Scala
keywords: Scala 

---

## 前言

1. object and class

	*  using parentheses(圆括号) on the instance of a class actually calls the apply method defined on this class. This approach is widely used in the standard library as well as in third-party libraries.`val joe = new Person("zhangsan")`，joe.apply() 等同于 `joe()`
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

## case 和 锅炉板模式

[Java正在“Kotlin化”](https://mp.weixin.qq.com/s/ut6l7ipdkN3O-9rIELuEcQ)Java record 是我们长期以来一直要求的一项特性，我相信你早就多次遇到这样的场景了，那就是极不情愿地实现 toString、hashCode、equals 方法以及每个字段的 getter。Kotlin 提供了数据类（data class）来解决这个问题，Java 也通过发布 record 类来解决了这个问题，同样的问题，Scala 是通过 case 类来解决的。这些类的主要目的是在对象中保存不可变的数据。PS：**有点类似DDD中的值对象。或者说，语言设计正在想业务模型的需要靠拢**。

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

## 其它

[函数式思维和函数式编程](http://www.vaikan.com/programming-thinking-functional-way/)

[怎么避免把scala程序写成java？ - 杨博的回答 - 知乎](https://www.zhihu.com/question/64568400/answer/222581715) 李浩毅写了一篇[Strategic Scala Style: Principle of Least Power](http://www.lihaoyi.com/post/StrategicScalaStylePrincipleofLeastPower.html) ， Scala的语言特性不算多，但是语言特性之间过于正交，一方面你把语言特性组合起来之后可以变得很复杂，写出各种其他语言的范式，另一方面容易玩脱。不想玩脱的话，就得优先选用功能最弱的功能（尽量少用scala的高级特性）。只在弱的功能解决不了你面临的问题时，才用更强的功能。Scala的创始人Martin Odersky也在李浩毅博客下面举双手赞成