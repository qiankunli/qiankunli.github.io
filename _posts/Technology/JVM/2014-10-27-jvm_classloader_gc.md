---
layout: post
title: JVM类加载与垃圾收集
category: 技术
tags: JVM
keywords: JAVA JVM
---

## 前言

* TOC
{:toc}

[Java crashes](https://confluence.atlassian.com/confkb/java-crashes-235669496.html)The virtual machine is responsible for emulating a CPU, managing memory and devices, just like the operating system does for native applications (MS Office, web browsers etc).

断断续续的在java内存、多线程、io这块写了几篇博客，跨度有两三年，回顾起来，发现它们正好构成了看待jvm的绝佳方式。

![](/public/upload/java/jvm.png)

1. jvm 内存模型与物理机/os内存模型的映射 [JVM3——java内存模型](http://qiankunli.github.io/2017/05/02/java_memory_model.html)
2. jvm 线程 与 linux 进程/线程模型的映射 [AQS1——并发相关的硬件与内核支持](http://qiankunli.github.io/2016/03/13/aqs.html)
3. java io 与 linux io 模型的映射 [java io涉及到的一些linux知识](http://qiankunli.github.io/2017/04/16/linux_io.html)
4. jvm 一些高级指令（以支持java 高级语法） 对 机器指令的映射 [JVM4——《深入拆解java 虚拟机》笔记](http://qiankunli.github.io/2018/07/20/jvm_note.html)


## jvm 在java 体系中的位置

![](/public/upload/java/jdk_jre_jvm.png)

jdk 安装目录含义

![](/public/upload/java/jdk_install_directory.png)

[Class Loaders in Java](https://www.baeldung.com/java-classloaders)

Class loaders are responsible for loading Java classes during runtime dynamically to the JVM (Java Virtual Machine). Also, they are part of the JRE (Java Runtime Environment). Hence, the JVM doesn’t need to know about the underlying files or file systems in order to run Java programs thanks to class loaders. 潜台词：Class loaders 是jre 类库的一部分但不是JVM 的一部分

## “可执行文件”

在linux中，可执行文件没有唯一的后缀名，本文以"可执行文件"统称。

||java|os|
|---|---|---|
||jvm|linux os|
||class 文件|可执行文件|

两者有很多相象的地方，但毕竟机理不同，class文件和可执行文件的不同正是两个os机理不同的反映。而本质上的不同，则要追溯到java的起源：面向网络，为了让“可执行文件”在网络上传输并在不同的系统上执行，发散出很多机制。

### class文件格式

因为指令中包含了操作数，可执行文件不只是指令的堆砌。

操作数大部分是地址引用，寄存器（或栈）成了存储引用的地方，作为cpu和内存的“中转站”。还有一些符号引用，需要在指令之前，描述这些符号引用。

class文件中包含方法和属性信息，这些数据为反射机制提供的基础。

### class文件的加载

加载的本质，从磁盘上加载，得到的是一个字节数组，然后按照自己的内存模型，把字节数组中对应的数据放到对应的地方。

程序和可执行文件  本身，都将“方法之类”的数据共享，“数据之类”的数据保存多份。

## 类加载——按类名加载

与c/c++语言不同，c的二进制代码是c代码 + 库函数 编译链接的结果，运行时直接被加载到内存，当然也可以先加载一部分，通过缺页机制按页加载，加载哪一页跟地址有关系。而对于java，实际的“可执行文件”是jvm，像shell一样是个解释器，jvm加载java 代码开始执行，就像shell读入人的指令开始执行。**换句话说，如果条件允许，jvm启动起来，像shell一样空转都是可以的。**

所以java有一个类加载过程，按名称找到Class文件并加载到内存，并对数据进行校验，转化解析和初始化，最终形成可以被虚拟机直接使用的java类型，这就是虚拟机的类加载机制。

[JVM类加载器与ClassNotFoundException和NoClassDefFoundError](http://arganzheng.life/jvm-classloader-ClassNotFoundException-NoClassDefFoundError.html)在”加载“阶段，虚拟机需要完成以下三件事：

1. 通过一个类的全限定名来获取此类的二进制字节流。类似的 [maven的基本概念](http://qiankunli.github.io/2019/02/14/maven_concept.html) 中提到`URL construction scheme` 概念，根据一个jar 的groupId + artifactId + version 即可构造一个http url ，从maven remote Repository 下载jar 文件。
2. 将字节流代表的静态存储结构转换为方法区的运行时数据结构 
3. 在内存中创建一个代表此类的java.lang.Class对象，作为方法区此类的各种数据的访问入口。

![](/public/upload/java/class_load_process.png)

![](/public/upload/java/jvm_class_loader.png)

![](/public/upload/java/jvm_class_loader_reference.png)

### 类加载器的双亲委派模型

ClassLoader源码注释：The ClassLoader class uses a delegation model to search for classes and resources. 

![](/public/upload/java/classloader_object.png)

双亲委派模型要求除了顶层的启动类加载器外，其余的类加载器都必须有自己的父类加载器，类加载器间的父子关系不会以继承关系实现，而是以组合的方式来复用父类加载的代码。

双亲委派模型的工作过程：当一个类加载器收到类加载请求的时候，它会首先把这个请求委托给父类加载器去执行，因此所有的类加载请求最终都会传送到顶层的启动类加载器中，只有当父类加载器也无法找到时才会交给自己去加载。

双亲委派模型的关键就是定义了类的加载过程，先尝试用父类加载器加载，再使用自定义加载器加载，以确保关键的类不被篡改。

使用场景：

1. 热部署
2. 代码加密
3. 类层次划分

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

笔者曾写过一个框架，用户在代码中通过注解使用。注解参数包括类的全类名（用户自定义的策略类），框架通过注解拿到用户的全类名，加载类，然后调用执行。

但当框架给scala小组使用时，scala小组因使用的play框架的classloader是spring classload的子类。用户自定义策略类是scala实现的，写在用户的项目中。

框架实现主流程，其中的某个环节，load 用户自定义的策略类执行。此时，框架代码`Class.forName(class name)`去load scala class name就力不从心了。为何呀？

[Java中隔离容器的实现](http://codemacro.com/2015/09/05/java-lightweight-container/)

1. 当在class A中使用了class B时，JVM默认会用class A的class loader去加载class B。
2. 每个class loader 有一个自己的search class 文件的classpath 范围。
3. class的 加载不是一次性加载完毕的，而是根据需要延迟加载的（上文提到过）。
4. 如果class B 不在class loader的classpath search 范围，则会报ClassNotFoundException

与Spring ioc 隔离的对比 [Spring IOC 级联容器原理探究](https://gitbook.cn/gitchat/activity/5b4d716d6b1c4569aa703e49)。PS：有意思的是，**classloader 和 spring ioc 都称之为容器，都具有隔离功能，这背后是否有一个统一的逻辑在？都是class loader，只是class 来源不同，加载后的组织方式不同**

在 Java 虚拟机中，类的唯一性是由类加载器实例以及类的全名一同确定的（即便是同一串字节流，经由不同的类加载器加载，也会得到两个不同的类。猜测一下，如果是一致的，ClassLoader 该如何实现呢？

1. ClassLoader `Class<?> defineClass(String name, byte[] b, int off, int len)` 时，如果发现name 相同， 可以直接返回。但ClassLoader 是可以自定义实现的，很难约束开发必须遵守这个规则。
2. defineClass 时直接覆盖，那问题就更严重了，开发就有机会恶意覆盖 一些已有的java 库中的类的实现。

在大型应用中，**往往借助这一特性，来运行同一个类的不同版本**。tomcat 在类加载方面就有很好的实践 [Tomcat源码分析](http://qiankunli.github.io/2019/11/26/tomcat_source.html)

## Java对象的内存布局

|新建对象的方式||
|---|---|
|new|通过构造器来初始化实例字段|
|反射|通过构造器来初始化实例字段|
|Object.clone|直接复制已有的数据，来初始化新建对象的实例字段|
|反序列化|直接复制已有的数据，来初始化新建对象的实例字段|
|Unsafe.allocateInstance|未初始化实例字段|

![](/public/upload/java/java_object_memory_layout.png)

每个 Java 对象都有一个对象头 （object header） ，由标记字段和类型指针构成，标记字段用来存储对象的哈希码， GC 信息， 持有的锁信息，而类型指针指向该对象的类 Class ，在 64 位操作系统中，标记字段占有 64 位，而类型指针也占 64 位，也就是说一个  Java  对象在什么属性都没有的情况下要占有 16 字节的空间，当前 JVM 中默认开启了压缩指针，这样类型指针可以只占 32 位，所以对象头占 12 字节， 压缩指针可以作用于对象头，以及引用类型的字段。

以 Integer 类为例，它仅有一个 int 类型的私有字段，占 4 个字节。因此，每一个 Integer 对象的额外内存开销至少是 400%。这也是为什么 Java 要引入基本类型的原因之一。

默认情况下，Java 虚拟机堆中对象的起始地址需要对齐至 8的倍数。如果一个对象用不到 8N 个字节，那么剩下的就会被填充。这些浪费掉的空间我们称之为对象间的填充（padding）。

内存对齐

1. 对象内存对齐， 这样对象的地址 就可以压缩一下，比如address * 8 得到对象的实际地址。
2. 对象字段内存对齐（有六七个对齐规则），让字段只出现在同一 CPU 的缓存行中。如果字段不是对齐的，那么就有可能出现跨缓存行的字段。也就是说，该字段的读取可能需要替换两个缓存行，而该字段的存储也会同时污染两个缓存行。
3. Java 虚拟机重新分配字段的先后顺序，以达到内存对齐的目的

## java 对象在缓存中的读写

![](/public/upload/jvm/field_align.png)

通过内存对齐可以避免一个字段同时存在两个缓存行里的情况，但还是无法完全规避缓存伪共享的问题，也就是一个缓存行中存了多个变量，而这几个变量在多核 CPU 并行的时候，会导致竞争缓存行的写权限，当其中一个 CPU 写入数据后，这个字段对应的缓存行将失效，导致这个缓存行的其他字段也失效。

在 Disruptor 中，通过填充几个无意义的字段，让对象的大小刚好在 64 字节，一个缓存行的大小为64字节，这样这个缓存行就只会给这一个变量使用，从而避免缓存行伪共享，但是在 jdk7 中，由于无效字段被清除导致该方法失效，只能通过继承父类字段来避免填充字段被优化，而 jdk8 提供了注解@Contended 来标示这个变量或对象将独享一个缓存行，使用这个注解必须在 JVM 启动的时候加上 `-XX:-RestrictContended` 参数，其实也是用**空间换取时间**。

## 垃圾收集算法

有一个梗：说在食堂里吃饭，吃完把餐盘端走清理的是 C++ 程序员，吃完直接就走的是 Java 程序员。

不同的区域存储不同性质的数据，除了程序计数器区域不会OOM外，其它的都有可能因为存储本区域数据过多而OOM。

jvm 提供自动垃圾回收机制，但免费的其实是最贵的，一些追求性能的框架会自己进行内存管理。[资源的分配与回收——池](http://qiankunli.github.io/2016/06/17/pool.html)

### 如何判断对象已经死亡

说白了，判断还有“引用”引用它么？

1. 引用计数法

    记录对象被引用的次数
  
2. 可达性分析算法

    以一系列GC Roots对象作为起点，从这写节点向下检索，当GC Roots到这些对象不可达时，则证明此对象是不可用的。

GC Roots

1. 虚拟机栈（栈帧中的本地变量表）中引用的对象
2. 方法区中类静态属性引用的对象
3. 方法区中常量引用的对象
4. 本地方法栈中 JNI（即一般说的 Native 方法）引用的对象

### 回收已死对象所占内存区域

当我们知道哪些对象可以回收时，它们分散在堆的各个地方，如何提高回收效率呢？一次回收完成后，理想状态是：内存是“整齐”的，活着的在一边，空闲的在一边。

1. 标记-清除算法

    - 实现： 第一遍，标记堆中哪些对象需要被回收；第二遍，回收被标记的对象。
    - 特点：效率不高，一次回收后，堆区碎片化

2. 复制算法

    - 实现：将区域分成两块（或多块），先紧着一块使用，这块用完后，将活着的对象复制到另一块，然后回收这一整块。
    - 特点：一部分区域会被浪费，如果对象都是“朝生夕死”的，则非常适合
3. 标记-整理算法

    - 实现：让所有活着的对象都向边界一端移动，清理端边界以外的堆区域
4. 分代收集算法

    - 实现：将堆区根据对象生存期分为几块，比如分为新生代和老年代，新生代采用“复制算法”，老年代采用“标记-清理”或“标记-整理”算法。

极客时间《深入拆解Java虚拟机》垃圾回收的三种方式

1. 清除sweep，将死亡对象占据的内存标记为空闲。
2. 压缩，将存活的对象聚在一起
3. 复制，将内存两等分， 说白了是一个以空间换时间的思路。

基本假设：部分的 Java 对象只存活一小段时间，而存活下来的小部分对象则会存活很长一段时间。**这个假设造就了 Java 虚拟机的分代回收思想**。PS：想提高效率就要限定问题域（优化都是针对特定场景的优化），限定问题域就要充分的发掘待解决问题的特征。

上面三种回收算法，各有各的优缺点，既然优缺点不可避免，那就是将它们用在特定的场合扬长避短。java 虚拟机将堆分为新生代和老年代，并且对不同代采用不同的垃圾回收算法


