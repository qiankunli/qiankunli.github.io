---

layout: post
title: ProxyFactoryBean
category: 技术
tags: Java
keywords: proxyFactoryBean

---

## 简介（未完待续）

流程写好，将中间的一个个性化环节暴漏出来，这是框架常做的工作。换个说法，就是，流程 + 个性化的操作，如何织入？

再换个表述，一边是spring（确切的说，是spirng的ioc倡导的pojo编程范式），一边是某个领域的具体解决方案（复杂的父子关系），两者如何整合（我就曾经碰到过这样的问题，用spring写多线程程序，怎么看怎么怪）。

这个时候，我们就能看到aop的用武之地，spring的事务，rmi实现等都依赖的aop，而aop的关键组件，proxyFactoryBean



## 引用

