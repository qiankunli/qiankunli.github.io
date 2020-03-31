---

layout: post
title: 形而上之设计模式
category: 架构
tags: Practice
keywords: 设计模式

---

## 简介

* TOC
{:toc}

[ddd(一)——领域驱动理念入门](http://qiankunli.github.io/2017/12/25/ddd.html)提到，面向对象设计的一种概述是：提取抽象以及抽象之间的关系（巧了，数据结构说的是数据与数据之间的关系；马克思说的是 生产力与生产力之间的关系）。针对具体的业务特性，二十几种设计模式，**每一种设计模式（尤其是行为型设计模式）都在指导我们如何划分对象以及对象之间的关系**。**领会设计模式，是我们领会面向对象设计思想的一个入口**。所以当我们想着如何提高自己的“面向对象”的设计能力时，直观的做法尽量多用符合业务场景的设计模式。也因此，我们在分析 apollo client 时，其核心结构 便是一个观察者模式。

当然，本文说的design pattern 不单指 Object-oriented design patterns，我们通常在一个编程范式下讨论设计模式，但严格上两者不具备对应关系。

[​圣杯与银弹——没用的设计模式](https://mp.weixin.qq.com/s/3TbunRkouM7PtCQrC52brQ)软件系统中处处都是设计，学习设计模式无法让我们成为优秀的工程师，如果我们错误的理解了这本书的目的，以为自己学到了软件设计或者面向对象设计的精髓，那就大错特错了。软件设计的能力并不是一朝一夕就能培养出来的，它需要我们不断对代码进行思考，**理解可能存在的扩展点**并设计合理的抽象。PS：面向扩展点设计。[​圣杯与银弹——没用的设计模式](https://mp.weixin.qq.com/s/3TbunRkouM7PtCQrC52brQ) 对设计模式做了一定的批评，对“单元测试”推崇有加，提升项目单元测试覆盖率的过程会让我们思考如何写出更利于测试的代码，虽然软件工程中没有银弹，但是单元测试不是银弹可能也所差无几了。

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

## 形而下

[​圣杯与银弹——没用的设计模式](https://mp.weixin.qq.com/s/3TbunRkouM7PtCQrC52brQ)抽象的设计模式是从不同具体项目中总结出来的通用经验，从具体到抽象的过程相对容易，然而**从抽象的模式套用到具体场景却很困难**，如果没有足够的经验或者思考只会做出拙劣的设计。而且并不是居高临下的架构设计才是系统设计，每个包、方法甚至代码中的空行中都体现了作者的设计思路，抽象的理论和模式能够起到指导的作用，但是真正让设计融入系统的还是工程师的丰富经验和深入思考。

21 世纪诞生的一些编程语言与过去的编程语言有着很大的不同，不仅新的编程语言开始接收函数式编程中的一些思想和设计，上个世纪诞生的编程语言也在吸纳不同的编程范式，函数和方法成为了语言中的一等公民，我们可以直接**向函数中传递函数来简化过去复杂的类关系**。比如观察者模式[函数式编程的设计模式](http://qiankunli.github.io/2018/12/15/functional_programming_patterns.html)

```
object.OnUpdate(func(u *updates) {
    ...
})
```

## 成为通用术语

[从技术演变的角度看互联网后台架构](https://mp.weixin.qq.com/s/7Qc8irbh0rz43OPWKbO2Ag)20多年前的经典著作DesignPatterns中讲过学习设计模式的意义：学习设计模式并不是要你学习一种新的技术或者编程语言，而是建立一种交流的共同语言和词汇，在方案设计时方便沟通，同时也帮助人们从更抽象的层次去分析问题本质，而不被一些实现的细枝末节所困扰。同时，当我们能把很多问题抽象出来之后，也能帮我们更深入更好地去了解现有系统。

[​圣杯与银弹——没用的设计模式](https://mp.weixin.qq.com/s/3TbunRkouM7PtCQrC52brQ)设计模式作为通用的术语确实可以增加不同工程师之间的沟通效率，但是降低沟通成本的前提是双方对同术语有着相同的并且正确的认识，如果双方的理解有差异，反而会制造更多的困惑。我们可以将 23 种不同的设计模式分成两部分来分析，其中一部分是单例模式、抽象工厂模式这些被广泛接受并理解的模式，另一部分是迭代子模型、命令模式和解释器模式等不容易被理解的复杂模式。从单例模式以及观察者模式的命名，我们就能猜到它们想要解决的问题，使用类似的术语也很难造成歧义，确实能够起到提高沟通效率的作用；不过，**对于复杂的设计模式想要正确理解就非常困难，更不用说用来沟通了**。

## Classification 

Design patterns are composed of several sections . Of particular interest are the Structure, Participants, and Collaboration sections. These sections describe a design motif:

1. a prototypical micro-architecture that developers copy and adapt to their particular designs to solve the recurrent problem described by the design pattern. 
2. A micro-architecture is a set of program constituents (e.g., classes, methods...) and their relationships. 
3. Developers use the design pattern by introducing in their designs this prototypical micro-architecture, which means that micro-architectures in their designs will have structure and organization similar to the chosen design motif.

设计模式有好多种不同的分类方式（不同的划分维度）

1. Design patterns were originally grouped into the categories: creational patterns, structural patterns, and behavioral patterns, and described using the concepts of delegation, aggregation, and consultation
2. Domain-specific patterns，比如 [Concurrency patterns](https://en.wikipedia.org/wiki/Concurrency_pattern), Web Presentation Patterns: Model View Controller 

