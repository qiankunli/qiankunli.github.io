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

[Object-oriented programming](https://en.wikipedia.org/wiki/Object-oriented_programming#Object-orientation_and_databases)

本文主要来自wiki 百科 [Software design pattern](https://en.wikipedia.org/wiki/Software_design_pattern) 


## 形而上

It is not a finished design that can be transformed directly into source or machine code（代码不是设计）. It is a description or template for how to solve a problem that can be used in many different situations. （设计是一个通用问题的解决方案）

Object-oriented design patterns typically show relationships and interactions between classes or objects,  Patterns that imply mutable state may be unsuited for functional programming languages,object-oriented patterns are not necessarily suitable for non-object-oriented languages. 设计模式跨语言有一定的局限性。

Design patterns may be viewed as a structured approach（套路化的方法） to computer programming intermediate between the levels of a programming paradigm（范例） and a concrete algorithm. 一说levels 想起了 [分层那些事儿](http://qiankunli.github.io/2017/03/16/layer.html)

Patterns originated as an architectural concept by Christopher Alexander (1977/79). In 1987,Kent Beck and Ward Cunningham began experimenting with the idea of applying patterns to programming. 互联网公司讲究敏捷，需求一定就开干。复杂的业务会有一份儿设计文档，但粒度很粗且主要体现业务。**大部分人一说编程，还是在说写代码 且是具体的 controller-service-dao， 在项目设计 和 jdbc/redis/rabbitmq/kafka code 之间缺了一个代码设计的layer和设计过程，具体业务问题和pattern的匹配过程**


In order to achieve flexibility, design patterns usually introduce additional levels of indirection, which in some cases may complicate the resulting designs and hurt application performance.

A famous aphorism of David Wheeler goes: "All problems in computer science can be solved by another level of indirection" (the "[fundamental theorem of software engineering](https://en.wikipedia.org/wiki/Fundamental_theorem_of_software_engineering)"). This is often deliberately mis-quoted with "abstraction layer" substituted for "level of indirection".  为了灵活性  ==> 不能太直接 ==> 没有什么问题是加一层解决不了的 ==> 分层。

By definition, a pattern must be programmed anew into each application that uses it. Since some authors see this as a step backward from software reuse as provided by components, researchers have worked to turn patterns into components. pattern 应用的多了就会 进化为一个组件。


[从技术演变的角度看互联网后台架构](https://mp.weixin.qq.com/s/7Qc8irbh0rz43OPWKbO2Ag)20多年前的经典著作DesignPatterns中讲过学习设计模式的意义：学习设计模式并不是要你学习一种新的技术或者编程语言，而是建立一种交流的共同语言和词汇，在方案设计时方便沟通，同时也帮助人们从更抽象的层次去分析问题本质，而不被一些实现的细枝末节所困扰。同时，当我们能把很多问题抽象出来之后，也能帮我们更深入更好地去了解现有系统

## Classification 

Design patterns are composed of several sections . Of particular interest are the Structure, Participants, and Collaboration sections. These sections describe a design motif:

1. a prototypical micro-architecture that developers copy and adapt to their particular designs to solve the recurrent problem described by the design pattern. 
2. A micro-architecture is a set of program constituents (e.g., classes, methods...) and their relationships. 
3. Developers use the design pattern by introducing in their designs this prototypical micro-architecture, which means that micro-architectures in their designs will have structure and organization similar to the chosen design motif.


## 分类

设计模式有好多种不同的分类方式（不同的划分维度）

1. Design patterns were originally grouped into the categories: creational patterns, structural patterns, and behavioral patterns, and described using the concepts of delegation, aggregation, and consultation
2. Domain-specific patterns，比如 [Concurrency patterns](https://en.wikipedia.org/wiki/Concurrency_pattern), Web Presentation Patterns: Model View Controller 



