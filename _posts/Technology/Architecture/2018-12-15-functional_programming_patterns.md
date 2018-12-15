---

layout: post
title: 函数式编程的设计模式
category: 技术
tags: Architecture
keywords: functional programming patterns

---

## 简介

可预先看下 [函数式编程](http://qiankunli.github.io/2018/09/12/functional_programming.html)  

在面向对象的观念里: 一切皆对象。但知道这句话并没有什么用，大部分人还是拿着面向对象的写着面向过程的代码，尤其是结合spring + springmvc 进行controller-service-dao 的业务开发。所以，看一个代码是不是“面向对象” 一个立足点就是对java 设计模式的应用。

对应的，函数式编程时知道一切皆函数的意义也很有限，应用函数式编程的一个重要立足点就是：函数式编程中的设计模式。 

思维训练是一辈子的事儿，德扑的时候， 

时间管理，你知道轻重缓急，就是心里着急

排查bug，狄仁杰探案

## Functional Programming Design Patterns

本小节膝盖给 ScottWlaschin 大神，其slide 有一幅图[Functional Programming Patterns (NDC London 2014)](https://www.slideshare.net/ScottWlaschin/fp-patterns-ndc-london2014) 其在youtube 有对应的演讲。

![](/public/upload/architecture/function_programming_patterns.jpg)

[Gang of Four Patterns in a Functional Light: Part 1
](https://www.voxxed.com/2016/04/gang-four-patterns-functional-light-part-1/)

a simple exercise of grammatical analysis. Consider a sentence like: “smoking is unhealthy” or even “running is tiring”. What are “smoking” and “running” in this context? In English, the -ing suffix transforms verbs like to smoke or to run into nouns. The biggest part of the design patterns listed in the Gang of Four book, especially the ones classified as behavioural patterns, follow exactly the same approach. Like the -ing suffix, they turn verbs into nouns – or in this case, functions into objects.