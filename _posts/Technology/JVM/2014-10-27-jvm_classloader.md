---
layout: post
title: JVM类加载
category: 技术
tags: JVM
keywords: JAVA JVM
---

## 前言

* TOC
{:toc}

## 字节码结构

C在开发层面的平台相关性：C语言实现系统兼容性的思路很简单，那就是通过在不同的硬件平台和操作系统上开发各自特定的编译器，从而将相同的C语言源代码翻译为底层平台相关的硬件指令。虽然这种思路很棒，但是仍然有明显的缺点，当涉及系统调用时，开发者仍然要关注具体底层系统的API。在Linux平台上，开发者需要知道Linux平台所提供的创建线程的接口是`pthread_create()`；而在Windows平台上，开发者需要知道Windows平台所提供的创建线程的接口是`CreateThread()`。另外，在Linux和Windows平台上，C程序需要引用不同的头文件，并且所调用的创建线程的两种API的入参和返回值也不相同。所以在开发层面上屏蔽底层差异的关键就是**中间语言**，C可以run anywhere，但不能write once。

以类似C struct 的方式来表达java 字节码文件的结构。

![](/public/upload/jvm/class_code.jpg)

常量池（对应Hotspot C++ constantPoolOop）里放的是字面常量和符号引用
1. 字面常量主要包含文本串以及被声明为final的常量。
2. 符号引用包含类和接口的全局限定名、字段的名称和描述符、方法的名称和描述符，因为Java语言在编译的时候没有连接这一步，所有的引用都是运行时动态加载的，所以就需要把这些引用的信息保存在class文件里。

## 字节码生成——ASM

从编译原理的层面看，生成 LLVM 的 IR 时，可以得到 LLVM 的 API 的帮助。字节码就是另一种 IR，而且比 LLVM 的 IR 简单多了，有ASM/Apache BCEL/Javassist 这个工具为我们生成字节码。ASM是一个开源的字节码生成工具/**字节码操纵框架**。Grovvy 语言就是用它来生成字节码的，它还能解析 Java 编译后生成的字节码，从而进行修改。

ASM 解析字节码的过程，有点像 XML 的解析器解析 XML 的过程：先解析类，再解析类的成员，比如类的成员变量（Field）、类的方法（Mothod）。在方法里，又可以解析出一行行的指令。


### 部分生成的字节码

Spring 采用的代理技术有两个：一个是 Java 的动态代理（dynamic proxy）技术；一个是采用 cglib 自动生成代理，cglib 采用了 asm 来生成字节码。Java 的动态代理技术，只支持某个类所实现的接口中的方法。如果一个类不是某个接口的实现，那么 Spring 就必须用到 cglib，从而用到字节码生成技术来生成代理对象的字节码。

### 系统的根据编程语言代码AST生成字节码

基于 AST 生成 JVM 的字节码的逻辑还是比较简单的，比生成针对物理机器的目标代码要简单得多，为什么这么说呢？主要有以下几个原因：

1. 不用太关心指令选择的问题。针对 AST 中的每个运算，基本上都有唯一的字节码指令对应，直白地翻译就可以了，不需要用到树覆盖这样的算法。
2. 不需要关心寄存器的分配，因为 JVM 是使用操作数栈的；
3. 指令重排序也不用考虑，因为指令的顺序是确定的，按照逆波兰表达式的顺序就可以了；
4. 优化算法，暂时也不用考虑。

## 类加载——按类名加载

加载的本质，从磁盘上加载，得到的是一个字节数组，然后按照自己的内存模型，把字节数组中对应的数据放到进程内存对应的地方。并对数据进行校验，转化解析和初始化，最终形成可以被虚拟机直接使用的java类型，这就是虚拟机的类加载机制。

[JVM类加载器与ClassNotFoundException和NoClassDefFoundError](http://arganzheng.life/jvm-classloader-ClassNotFoundException-NoClassDefFoundError.html)在”加载“阶段，虚拟机需要完成以下三件事：

1. 通过一个类的全限定名来获取此类的二进制字节流。类似的 [maven的基本概念](http://qiankunli.github.io/2019/02/14/maven_concept.html) 中提到`URL construction scheme` 概念，根据一个jar 的groupId + artifactId + version 即可构造一个http url ，从maven remote Repository 下载jar 文件。
2. 将字节流代表的静态存储结构转换为方法区的运行时数据结构 
3. 在内存中创建一个代表此类的java.lang.Class对象，作为方法区此类的各种数据的访问入口。

![](/public/upload/java/class_load_process.png)

![](/public/upload/java/jvm_class_loader_reference.png)

### 类加载器的双亲委派模型

ClassLoader源码注释：The ClassLoader class uses a delegation model to search for classes and resources. 

![](/public/upload/java/classloader_object.png)

双亲委派模型要求除了顶层的启动类加载器外，其余的类加载器都必须有自己的父类加载器，类加载器间的父子关系不会以继承关系实现，而是以组合的方式来复用父类加载的代码。通过这种**层次模型**，可以避免类的重复加载，也可以避免核心类被不同的类加载器加载到内存中造成冲突和混乱，从而保证了Java核心库的安全。

双亲委派模型的工作过程：当一个类加载器收到类加载请求的时候，它会首先把这个请求委托给父类加载器去执行，因此所有的类加载请求最终都会传送到顶层的启动类加载器中，只有当父类加载器也无法找到时才会交给自己去加载。


[Java类加载器 — classloader 的原理及应用](https://mp.weixin.qq.com/s/YzIlIx4t0uqb-fm9rA9EvQ)使用场景：

1. 热部署
2. 热加载 spring boot devtools。
2. 代码加密。基于java开发编译产生的jar包是由.class字节码组成，由于字节码的文件格式是有明确规范的。因此对于字节码进行反编译，就很容易知道其源码实现了。jar包加密的本质，还是对字节码文件进行加密操作。但是JVM虚拟机加载class的规范是统一的，因此在加载class文件之前通过自定义classloader先进行反向的解密操作，然后再按照标准的class文件标准进行加载
4. 依赖冲突。阿里 pandora(潘多拉）通过自定义类加载器，为每个中间件自定义一个加载器来 解决依赖冲突

### 延迟加载

    class X{
        static{   System.out.println("init class X..."); }
        int foo(){ return 1; }
        Y bar(){ return new Y(); }
    }

The most basic API is ClassLoader.loadClass(String name, boolean resolve)

    Class classX = classLoader.loadClass("X", resolve);

If resolve is true, it will also try to load all classes referenced by X. In this case, Y will also be loaded. If resolve is false, Y will not be loaded at this point.

### ClassNotFoundException和NoClassDefFoundError

[Why am I getting a NoClassDefFoundError in Java?](https://stackoverflow.com/questions/34413/why-am-i-getting-a-noclassdeffounderror-in-java)

1. ClassNotFoundException This exception indicates that the class was not found on the classpath.
1. NoClassDefFoundError, This is caused when there is a class file that your code depends on and it is present at compile time but not found at runtime. Look for differences in your build time and runtime classpaths. 引起的原因比较少见，还未掌握到精髓。

### ClassLoader 隔离

在 Java 虚拟机中，类的唯一性是由类加载器实例以及类的全名一同确定的（即便是同一串字节流，经由不同的类加载器加载，也会得到两个不同的类。猜测一下，如果是一致的，ClassLoader 该如何实现呢？

1. ClassLoader `Class<?> defineClass(String name, byte[] b, int off, int len)` 时，如果发现name 相同， 可以直接返回。但ClassLoader 是可以自定义实现的，很难约束开发必须遵守这个规则。
2. defineClass 时直接覆盖，那问题就更严重了，开发就有机会恶意覆盖 一些已有的java 库中的类的实现。

在大型应用中，**往往借助这一特性，来运行同一个类的不同版本**。tomcat 在类加载方面就有很好的实践 [Tomcat源码分析](http://qiankunli.github.io/2019/11/26/tomcat_source.html)

## Java对象在内存中的表示

|新建对象的方式||
|---|---|
|new|通过构造器来初始化实例字段|
|反射|通过构造器来初始化实例字段|
|Object.clone|直接复制已有的数据，来初始化新建对象的实例字段|
|反序列化|直接复制已有的数据，来初始化新建对象的实例字段|
|Unsafe.allocateInstance|未初始化实例字段|

### java 对象的C++ 类表示——oop-klass model

当C、C++和Delphi等程序被编译成二进制程序后，原来所定义的高级数据结构都不复存在了，当Windows/Linux等操作系统(宿主机)加载这些二进制程序时，是不会加载这些语言中所定义的高级数据结构的，宿主机压根儿就不知道原来定义了哪些数据结构、哪些类，所有的数据结构都被转换为对特定内存段的偏移地址。例如C中的Struct结构体，被编译后不复存在，汇编和机器语言中没有与之对应的数据结构的概念，CPU更不知道何为结构体。C++和Delphi中的类概念被编译后也不复存在，所谓的类最终变成内存首地址。而JVM虚拟机在加载字节码程序时，会记录字节码中所定义的所有类型的原始信息(元数据)，JVM知道程序中包含了哪些类，以及每个类中所关联的字段、方法、父类等信息（类型结构信息被带到了运行期）。这是JVM虚拟机与操作系统最大的区别所在。

```c
struct iphone6s {
    int length;
    int width;
    int height;
    int weight;
    int ram;
    int rom;
    int pixel;
}
int main(){
    struct iphone6s iphone; // 定义变量
    iphone.length = 138;
    iphone.weight = 64;
    ...
    return 0
}
// 编译为汇编
main:
    pushl %ebp
    movel%esp, %ebp
    subl$32, %esp
    
    movel$138, -28(%ebp)
    movel$67, -24(%ebp)
    ...
    movel$0,%eax
    leave
    ret
```

[深入理解多线程（二）—— Java的对象模型](https://juejin.im/post/5b7625aa6fb9a009910e641d)HotSpot是基于c++实现，而c++是一门面向对象的语言，本身具备面向对象基本特征，所以Java中的对象表示，最简单的做法是为每个Java类生成一个c++类与之对应。但HotSpot JVM并没有这么做，而是设计了一个OOP-Klass Model。
1. OOP（Ordinary Object Pointer）用来描述对象实例信息
2. Klass 用来描述java类，是虚拟机内部Java类型结构的对等体
为什么HotSpot要设计一套oop-klass model呢？答案是：HotSopt JVM的设计者不想让每个对象中都含有一个vtable（虚函数表）。oop的职能主要在于表示对象的实例数据，所以其中不含有任何虚函数。而klass为了实现虚函数多态，所以提供了虚函数表。

![](/public/upload/java/oop_kclass_model.png)

**在Java程序运行过程中，每创建一个新的对象，在JVM内部就会相应地创建一个对应类型的OOP对象。**JVM内部定义了各种oop-klass，在JVM看来，不仅Java类是对象，Java方法也是对象，字节码常量池也是对象，一切皆是对象。JVM使用不同的oop-klass模型来表示各种不同的对象。

![](/public/upload/java/hotspot_oop.png)

在HotSpot中，根据JVM内部使用的对象业务类型，具有多种oopDesc的子类。除了oppDesc类型外，opp体系中还有很多instanceOopDesc、arrayOopDesc 等类型的实例，他们都是oopDesc的子类。

![](/public/upload/java/hotspot_kclass.png)

**一个Java对象，它的存储是怎样的？**

1. 一般很多人会回答：对象存储在堆上。
2. 稍微好一点的人会回答：对象存储在堆上，对象的引用存储在栈上。
3. 一个更加显得牛逼的回答：对象的实例（instantOopDesc)保存在堆上，对象的元数据（instantKlass）保存在方法区，对象的引用保存在栈上。

### 内存布局

```c++
class oopDesc {
 private:
  volatile markWord _mark;
  union _metadata {
    Klass*      _klass;
    narrowKlass _compressed_klass;
  } _metadata;
}
```

![](/public/upload/java/java_object_memory_layout.png)

每个 Java 对象都有一个对象头 （object header） ，由标记字段和类型指针构成。java对象头信息是跟对象自身定义的数据结构无关的，**这些信息所记录的状态是用于JVM对对象的管理的**（比如并发访问与gc）。

1. 标记字段用来存储对象的哈希码， GC 信息， 持有的锁信息。
2. 类型指针指向该对象的类 Class。

在 64 位操作系统中，标记字段占有 64 位，而类型指针也占 64 位，也就是说一个  Java  对象在什么属性都没有的情况下要占有 16 字节的空间，当前 JVM 中默认开启了压缩指针，这样类型指针可以只占 32 位，所以对象头占 12 字节， 压缩指针可以作用于对象头，以及引用类型的字段。

以 Integer 类为例，它仅有一个 int 类型的私有字段，占 4 个字节。因此，每一个 Integer 对象的额外内存开销至少是 400%。这也是为什么 Java 要引入基本类型的原因之一。

默认情况下，Java 虚拟机堆中对象的起始地址需要对齐至 8的倍数。如果一个对象用不到 8N 个字节，那么剩下的就会被填充。这些浪费掉的空间我们称之为对象间的填充（padding）。

1. 对象内存对齐， 这样对象的地址 就可以压缩一下，比如address * 8 得到对象的实际地址。
2. 对象字段内存对齐（有六七个对齐规则），让字段只出现在同一 CPU 的缓存行中。如果字段不是对齐的，那么就有可能出现跨缓存行的字段。也就是说，该字段的读取可能需要替换两个缓存行，而该字段的存储也会同时污染两个缓存行。
3. Java 虚拟机重新分配字段的先后顺序，以达到内存对齐的目的



## java 对象在缓存中的读写

![](/public/upload/jvm/field_align.png)

通过内存对齐可以避免一个字段同时存在两个缓存行里的情况，但还是无法完全规避缓存伪共享的问题，也就是一个缓存行中存了多个变量，而这几个变量在多核 CPU 并行的时候，会导致竞争缓存行的写权限，当其中一个 CPU 写入数据后，这个字段对应的缓存行将失效，导致这个缓存行的其他字段也失效。

在 Disruptor 中，通过填充几个无意义的字段，让对象的大小刚好在 64 字节，一个缓存行的大小为64字节，这样这个缓存行就只会给这一个变量使用，从而避免缓存行伪共享，但是在 jdk7 中，由于无效字段被清除导致该方法失效，只能通过继承父类字段来避免填充字段被优化，而 jdk8 提供了注解@Contended 来标示这个变量或对象将独享一个缓存行，使用这个注解必须在 JVM 启动的时候加上 `-XX:-RestrictContended` 参数，其实也是用**空间换取时间**。

## 其它

《揭秘Java虚拟机:JVM设计原理与实现》Java选择具备运行时类型识别的特性本身便从一个十分隐晦的层面制约了Java必须选择成为一门面向对象的编程语言，为何？类型本身就是一种“闭包”的技术手段，只有先从语法层面实现了“闭包”，才能实现“对象”的概念，否则，何来的属性、成员变量、类方法一说？类型是实现将若干属性和动作打包成为一个整体对象进行统一识别的策略。如果Java像C++那样，类型不作为属性和方法封装的唯一手段，开发者可以随心所欲地在类的外面定义变量和函数，那么对于这部分数据的“运行时识别”必然是一个难题，可能需要通过类似namespace或者filename这样的机制去实现动态反射了，但是这种反射想想都让人头大，不容易啊！

当一门编程语言实现了完全的闭包语法策略(使用类型包装可以认为是闭包的一种)，便自然而然具备了自动内存管理的技术基础，或者说实现自动内存管理更加容易。所以闭包便成为很多具备自动内存回收特性的编程语言的语法基础，例如GO语言、Phthon、JavaScript等，虽然大家具体实现闭包的手段不同，但是殊途同归，都是为了能够让虚拟机在自动回收内存时尽量简单。




