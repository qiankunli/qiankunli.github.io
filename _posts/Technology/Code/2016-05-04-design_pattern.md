---

layout: post
title: 换个角度看待设计模式
category: 技术
tags: Code
keywords: java design pattern

---

## 前言

世事纷扰，但总有几个线头。兵法多样，就那几个套路。

||设计模式|军事|
|---|---|---|
|战术|二十三个设计模式|三十六计|
|战略|4个原则|以正合以奇胜|

**以正合，以奇胜**。以奇胜，被人们误读为奇袭得胜，还是贪巧求速的心理作怪。以奇胜的奇，不念qi，念ji，是个数学词汇，奇数、偶数的奇，古人又称为“余奇”，多余的部分，正兵安排好了，余下来的就是奇兵，关键的时候用。  简单地说，奇（ji）兵，不是出奇制胜的部队，是预备队。孙子的意思是：不要一下子把所有的牌都打完了，留一张在手上，关键时候打出去。 “奇正之变，不可胜穷也。奇正相生，如环之无端。” 奇正之间怎么相互转化呢？其实很简单，已经投入战斗的，是正兵；预备队，是奇兵。预备队投上去，就变为正兵了。正在打的部队撤下来，又变成奇兵。

## 形而上

[Object-oriented programming](https://en.wikipedia.org/wiki/Object-oriented_programming#Object-orientation_and_databases)

本文主要来自wiki 百科 [Software design pattern](https://en.wikipedia.org/wiki/Software_design_pattern) 

It is not a finished design that can be transformed directly into source or machine code（代码不是设计）. It is a description or template for how to solve a problem that can be used in many different situations. （设计是一个通用问题的解决方案）

Object-oriented design patterns typically show relationships and interactions between classes or objects,  Patterns that imply mutable state may be unsuited for functional programming languages,object-oriented patterns are not necessarily suitable for non-object-oriented languages. 设计模式跨语言有一定的局限性。

Design patterns may be viewed as a structured approach（套路化的方法） to computer programming intermediate between the levels of a programming paradigm（范例） and a concrete algorithm. 一说levels 想起了 [分层那些事儿](http://qiankunli.github.io/2017/03/16/layer.html)

Patterns originated as an architectural concept by Christopher Alexander (1977/79). In 1987,Kent Beck and Ward Cunningham began experimenting with the idea of applying patterns to programming. 互联网公司讲究敏捷，需求一定就开干。复杂的业务会有一份儿设计文档，但粒度很粗且主要体现业务。**大部分人一说编程，还是在说写代码 且是具体的 controller-service-dao， 在项目设计 和 jdbc/redis/rabbitmq/kafka code 之间缺了一个代码设计的layer和设计过程，具体业务问题和pattern的匹配过程**

In order to achieve flexibility, design patterns usually introduce additional levels of indirection, which in some cases may complicate the resulting designs and hurt application performance.

A famous aphorism of David Wheeler goes: "All problems in computer science can be solved by another level of indirection" (the "[fundamental theorem of software engineering](https://en.wikipedia.org/wiki/Fundamental_theorem_of_software_engineering)"). This is often deliberately mis-quoted with "abstraction layer" substituted for "level of indirection".  为了灵活性  ==> 不能太直接 ==> 没有什么问题是加一层解决不了的 ==> 分层。

By definition, a pattern must be programmed anew into each application that uses it. Since some authors see this as a step backward from software reuse as provided by components, researchers have worked to turn patterns into components. pattern 应用的多了就会 进化为一个组件。

## 抽取一个类

### 简单抽取

假设原来有一个比较复杂的类

    class A{
        void func(){
            1.xx
            2.xx
            3.xx
            4.xx
            5.xx
        }
    }
    
现在我们代码重构，要将步骤234抽出一个类B来，类B需要A的数据初始化，类A需要类B的计算结果。一般有两种方案

1. 方案1

        class A{
            void func(){
                1.xx
                2.B b = new B(xx);    // b作为A的类成员跟这个差不多
                3.xx = b.func();
                4.xx
            }
        }
    
2. 方案2

        class A{
            void func(){
                1. xx
                2. xx
            }
        }
        class B{
            void func(A a){
                1. xx = a.getxx();
                2. xx
                3. a.setxx();
            }
        }
        class Main{
            main{
                A a = new A();
                B b = new B();
                b.func(a);
            }
        }
    
比如spring ioc初始化的一段代码便是如此

    // 定义配置文件    
    ClassPathResource res = new ClassPathResource(“beans.xml”);
    // 创建bean工厂
    DefaultListableBeanFactory factory = new DefaultListableBeanFactory();
    // 定义读取配置文件的类
    XmlBeanDefinitionReader reader = new XmlBeanDefinitionReader(factory);
    // 加载文件中的信息到bean工厂中
    reader.loadBeanDefinitions(res);
    

两种方式的不同在于：

1. 前者只是将相关代码抽取为一个函数，然后到了另一个类里。（本质上只算是抽取了一个函数）
2. 后者将相关代码完全抽出来，A类中不用保有任何痕迹，可以算是抽取出了一个类

### 回调在拆分中的作用

假设我们要抽出的步骤是不连续的，比如将235抽取成一个函数（或B类），那么就要用到回调。（在实际的应用中，会出现“14使用了同样的数据，并且从语义上它们更适合在一起（或在A类）”的情况）

    class A{
        void func(){
            1.xx
            2.xx
            3.xx
            4.xx
            5.xx
        }
    }
    
     class A{
        void func(){
            1.xx
            func2(arg,new Callback(){
                void callback(){
                    4. xx
                }
            })  
        }
        void func2(arg,callback){
            2. xx
            3. xx
            4. callback()
            5. xx
        }
    }
    
当然，回调主要用在两个线程交互中。

## 从拆分的角度看

如果所有功能写在了一个函数里，我们如何拆分它

1. 能并行的并行。（设计模式已经不再局限于三大类型，还扩展到多线程模型，io通信模型）
2. 无关的代码拆成独立的类
3. 可能会变的代码拆成独立的类

拆分的方向有两种

1. 将不同的代码拆分成父类和子类，代理类和被代理类，装饰类和被装饰类等，有种类似垂直关系的感觉
2. 将不同的代码拆分成类似水平关系的类


## 接口重于实现——避免直接使用直接干活的类

操作系统是为了避免我们直接使用硬件，编程语言是为了避免我们直接使用系统调用，笔者一个很强烈的感觉就是，设计模式（创建型，行为型，结构型）为了避免我们直接使用一个类。

明明可以new一个，偏偏要用工厂。平明可以`obj.func`,偏偏要`proxyObj.func`

这一切都应了《设计模式之禅》中的思想，所有思想的基本原则就是：对扩展开放，对修改关闭。

一切可能变化的部分都不应该让程序猿直接调用（或者抽象成参数），为了应对变化，把一个类拆成多个类（按照变化的可能性拆分，按照上层接口聚合在一起），甚至不惜把变化本身单独拆成一个类。


## 分类方式

《Pattern Oriented Software Architecture》中提到，将模式分为三种类型：

1. 体系结构模式，比如mvc
2. 设计模式
3. 惯用法，比如引用计数法

Design patterns were originally grouped into the categories: creational patterns, structural patterns, and behavioral patterns, and described using the concepts of delegation, aggregation, and consultation

Design patterns are composed of several sections . Of particular interest are the Structure, Participants, and Collaboration sections. These sections describe a design motif:

1. a prototypical **micro-architecture** that developers copy and adapt to their particular designs to solve the recurrent problem described by the design pattern. 
2. A micro-architecture is a set of program constituents (e.g., classes, methods...) and their relationships. 
3. Developers use the design pattern by introducing in their designs this prototypical micro-architecture, which means that micro-architectures in their designs will have structure and organization similar to the chosen design motif.

`http://design-patterns.readthedocs.io/zh_CN/latest/` 传统分类方式

- 创建型模式，创建型模式(Creational Pattern)对类的实例化过程进行了抽象，能够**将软件模块中对象的创建和对象的使用分离**单一职责原则，仅一个对象的创建独立出来，催生了spring的ioc，减少了代码中的创建类代码）。在创建什么(What)，由谁创建(Who)，何时创建(When)等方面都为软件设计者提供了尽可能大的灵活性。
- 结构型模式(Structural Pattern)描述如何将类或者对 象结合在一起形成**更大的结构**.结构型模式可以分为类结构型模式和对象结构型模式

	- 类结构型模式关心类的组合,一般只存在继承关系和实现关系.
	- 对象结构型模式关心类与对象的组合，通过关联关系使得在一 个类中定义另一个类的实例对象(也就是成员变量)，然后通过该对象调用其方法。
- 行为型模式(Behavioral Pattern)**是对在不同的对象之间划分责任和算法的抽象化**（所以一般先定义好高层接口，定义好交互关系）。行为型模式不仅仅关注类和对象的结构（涉及到结构设计，但不是重点），而且重点关注它们之间的相互作用。

    行为型模式分为类行为型模式和对象行为型模式两种：

    - 类行为型模式：类的行为型模式使用继承关系在几个类之间分配行为，类行为型模式主要通过多态等方式来分配父类与子类的职责。
    - 对象行为型模式：对象的行为型模式则使用对象的聚合关联关系来分配行为，对象行为型模式主要是通过对象关联等方式来分配两个或多个类的职责。

根据“合成复用原则”，系统中要尽量使用关联关系来取代继承关系，因此大部分结构/行为型设计模式都属于对象结构/行为型设计模式。

## 成为通用术语

[从技术演变的角度看互联网后台架构](https://mp.weixin.qq.com/s/7Qc8irbh0rz43OPWKbO2Ag)20多年前的经典著作DesignPatterns中讲过学习设计模式的意义：学习设计模式并不是要你学习一种新的技术或者编程语言，而是建立一种交流的共同语言和词汇，在方案设计时方便沟通，同时也帮助人们从更抽象的层次去分析问题本质，而不被一些实现的细枝末节所困扰。同时，当我们能把很多问题抽象出来之后，也能帮我们更深入更好地去了解现有系统。

[​圣杯与银弹——没用的设计模式](https://mp.weixin.qq.com/s/3TbunRkouM7PtCQrC52brQ)设计模式作为通用的术语确实可以增加不同工程师之间的沟通效率，但是降低沟通成本的前提是双方对同术语有着相同的并且正确的认识，如果双方的理解有差异，反而会制造更多的困惑。我们可以将 23 种不同的设计模式分成两部分来分析，其中一部分是单例模式、抽象工厂模式这些被广泛接受并理解的模式，另一部分是迭代子模型、命令模式和解释器模式等不容易被理解的复杂模式。从单例模式以及观察者模式的命名，我们就能猜到它们想要解决的问题，使用类似的术语也很难造成歧义，确实能够起到提高沟通效率的作用；不过，**对于复杂的设计模式想要正确理解就非常困难，更不用说用来沟通了**。

## 其它

[ddd(一)——领域驱动理念入门](http://qiankunli.github.io/2017/12/25/ddd.html)提到，面向对象设计的一种概述是：提取抽象以及抽象之间的关系（巧了，数据结构说的是数据与数据之间的关系；马克思说的是 生产力与生产力之间的关系）。针对具体的业务特性，二十几种设计模式，**每一种设计模式（尤其是行为型设计模式）都在指导我们如何划分对象以及对象之间的关系**。**领会设计模式，是我们领会面向对象设计思想的一个入口**。所以当我们想着如何提高自己的“面向对象”的设计能力时，直观的做法尽量多用符合业务场景的设计模式。

[​圣杯与银弹——没用的设计模式](https://mp.weixin.qq.com/s/3TbunRkouM7PtCQrC52brQ)软件系统中处处都是设计，学习设计模式无法让我们成为优秀的工程师，如果我们错误的理解了这本书的目的，以为自己学到了软件设计或者面向对象设计的精髓，那就大错特错了。软件设计的能力并不是一朝一夕就能培养出来的，它需要我们不断对代码进行思考，**理解可能存在的扩展点**并设计合理的抽象。PS：面向扩展点设计。[​圣杯与银弹——没用的设计模式](https://mp.weixin.qq.com/s/3TbunRkouM7PtCQrC52brQ) 对设计模式做了一定的批评，对“单元测试”推崇有加，提升项目单元测试覆盖率的过程会让我们思考如何写出更利于测试的代码，虽然软件工程中没有银弹，但是单元测试不是银弹可能也所差无几了。

[​圣杯与银弹——没用的设计模式](https://mp.weixin.qq.com/s/3TbunRkouM7PtCQrC52brQ)抽象的设计模式是从不同具体项目中总结出来的通用经验，从具体到抽象的过程相对容易，然而**从抽象的模式套用到具体场景却很困难**，如果没有足够的经验或者思考只会做出拙劣的设计。而且并不是居高临下的架构设计才是系统设计，每个包、方法甚至代码中的空行中都体现了作者的设计思路，抽象的理论和模式能够起到指导的作用，但是真正让设计融入系统的还是工程师的丰富经验和深入思考。

21 世纪诞生的一些编程语言与过去的编程语言有着很大的不同，不仅新的编程语言开始接收函数式编程中的一些思想和设计，上个世纪诞生的编程语言也在吸纳不同的编程范式，函数和方法成为了语言中的一等公民，我们可以直接**向函数中传递函数来简化过去复杂的类关系**。比如观察者模式[函数式编程的设计模式](http://qiankunli.github.io/2018/12/15/functional_programming_patterns.html)

```
object.OnUpdate(func(u *updates) {
    ...
})
```
