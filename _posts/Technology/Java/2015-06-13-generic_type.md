---

layout: post
title: Java泛型
category: 技术
tags: Java
keywords: JAVA Generic Type 泛型

---

## 前言 ##

[一文详解Java泛型设计](https://mp.weixin.qq.com/s/VbkzJb45EPFY3Yu_iSQ25A)
在没有泛型之前，必须使用Object编写适用于多种类型的代码，同时由于数组的存在，设计者为了让其可以比较通用的进行处理，也让数组允许协变，这又为程序添加了一些天然的不安全因素。为了解决这些情况，Java的设计者终于在Java5中引入泛型。类型参数（Type parameter）使得ArrayList以及其他可能用到的集合类能够方便的**指示虚拟机其包含元素的类型**。运行时可能出现的各种类型转换错误得以在编译阶段就被阻止。

虚拟机没有泛型类型对象，“泛型”只在程序源码中存在，在编译后的字节码文件中，就已经被替换为原来的原始类型（Raw Type，也称为裸类型）了，并且在相应的地方插入了强制转型代码，因此对于运行期的Java语言来说，ArrayList<int>与ArrayList<String>就是同一个类。所以说泛型技术实际上是Java语言的一颗语法糖，Java语言中的泛型实现方法称为类型擦除（erased），基于这种方法实现的泛型被称为伪泛型。最大好处是编绎期的类型错误检查。PS： 类型擦除是Java泛型的一种实现机制。这意味着在编译时，泛型类型会被替换为它的限定类型（如果没有明确指定，那就是Object），并且在字节码中并不保留泛型信息。类型擦除最大的优点就是保证了与老版本Java代码的兼容性，因为在引入泛型之前的Java代码都是没有泛型信息的。但类型擦除毕竟是不得已为之，会有一些缺点，比如无法执行某些类型检查、导致方法签名冲突等。

C++的泛型是在编绎期实现的，为每一个类型都生成一份代码，所以C++的泛型容易让编绎后的代码出现膨胀。

[Java中如何获得A<T>泛型中T的运行时类型及原理探究](https://mp.weixin.qq.com/s/Yn9CIfgLozZNs_xfSo1ZmA) 未读

## 实例对比

泛型类是有一个或者多个类型变量的类，类型变量在整个类上定义就是用于指定方法的返回类型以及字段的类型。

```java
public class A{  
    public static void main(String[] args) {  
        A a = new A();  
        a.test();  
            
        String r = a.test();  
    }  
        
    public <T> T test() {  
        return (T) new Object();  
    }  
} 
```
    

`javap -c A`对应的字节码


    public class A {
      public A();
        Code:
           0: aload_0
           1: invokespecial #1                  // Method java/lang/Object."<init>":()V
           4: return
    
      public static void main(java.lang.String[]);
        Code:
           0: new           #2                  // class A
           3: dup
           4: invokespecial #3                  // Method "<init>":()V
           7: astore_1
           8: aload_1
           9: invokevirtual #4                  // Method test:()Ljava/lang/Object;
          12: checkcast     #5                  // class java/lang/String
          15: astore_2
          16: return
    
      public <T extends java/lang/Object> T test();
        Code:
           0: new           #6                  // class java/lang/Object
           3: dup
           4: invokespecial #1                  // Method java/lang/Object."<init>":()V
           7: areturn
    }

Java的泛型是伪泛型。在编译期间，所有的泛型信息都会被擦除掉。

    public class A{  
        public static void main(String[] args) {  
            A a = new A();  
            a.test();  
              
            String r = (String)a.test();  
        }  
          
        public  Object test() {  
            return new Object();  
        }  
    } 


对应的字节码

    public class A {
      public A();
        Code:
           0: aload_0
           1: invokespecial #1                  // Method java/lang/Object."<init>":()V
           4: return
    
      public static void main(java.lang.String[]);
        Code:
           0: new           #2                  // class A
           3: dup
           4: invokespecial #3                  // Method "<init>":()V
           7: astore_1
           8: aload_1
           9: invokevirtual #4                  // Method test:()Ljava/lang/Object;
          12: pop
          13: aload_1
          14: invokevirtual #4                  // Method test:()Ljava/lang/Object;
          17: checkcast     #5                  // class java/lang/String
          20: astore_2
          21: return
    
      public java.lang.Object test();
        Code:
           0: new           #6                  // class java/lang/Object
           3: dup
           4: invokespecial #1                  // Method java/lang/Object."<init>":()V
           7: areturn
    }
    
可见，两个类的字节码文件基本一摸一样。

```java

import java.util.ArrayList;
class Playground {
    public static void main(String[ ] args) {
        ArrayList<Integer> int_list = new ArrayList<Integer>();
        ArrayList<String> str_list = new ArrayList<String>();
        System.out.println(int_list.getClass() == str_list.getClass());
    }
}
```
关于 C++ 泛型的实现，ArrayList 和 ArrayList 应该是不同的两种类型。java这段代码的输出是 true。
```java
import java.util.ArrayList;
class Playground {
    public static void main(String[ ] args) {
        System.out.println("Hello World");
    }
    public static void sayHello(ArrayList<String> list) {}
    public static void sayHello(ArrayList<Integer> list) {}
}
```
我们知道，方法的重载的基本条件是两个同名方法的参数列表并不相同。但是当我们尝试编译上述程序的时候，却会得到这样的错误提示：
```
Playground.java:12: error: name clash: sayHello(ArrayList<Integer>) and sayHello(ArrayList<String>) have the same erasure
    public static void sayHello(ArrayList<Integer> list) {
                       ^
1 error
```
这是因为当对泛型进行擦除以后，两个 sayHello 方法的参数类型都变成了 ArrayList，从而变成了同名方法，所以就会出现命名冲突报错。
    
## 引用

[java泛型（一）、泛型的基本介绍和使用][]
    
[java泛型（一）、泛型的基本介绍和使用]: http://blog.csdn.net/lonelyroamer/article/details/7864531