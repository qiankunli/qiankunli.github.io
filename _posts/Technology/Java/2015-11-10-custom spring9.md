---

layout: post
title: 自己动手写spring（九） 总结
category: 技术
tags: Java
keywords: Java Spring

---

## 前言

作为一个使用轮子的程序员第一次写了轮子，感觉还是满兴奋的。笔者在看很多项目的源码时，经常是看了忘，忘了看。在反思这个现象时，笔者认为：很多代码固然很精巧，但因为读者没有设身处地考虑过实际的情景，对为什么要这样做没有直接的感觉，导致印象不深刻。

还有一个重要原因是，大部分框架的代码在发展过程中都经过重新设计，这固然必要，但判空、处理异常，以及复杂的父子关系等掩盖了最初的思路，容易将读者带入到细节中。

最后，笔者前一阵在学习《How tomcat works》，对作者讲述tomcat的方式深以为然，因此就萌生了自己实现一个框架的想法。尽量少的判空，几乎没有异常处理（方法全部是throw Exception），一次将最初的思路最直接的呈现在读者面前，然大家了解到spring也是这样一步步写出来的。


## 该框架的重点

**重要的是三个map**

框架中两个品类的组件，BeanFactory和BeanProcessor全部是基于这个三个map在做处理，理解了这三个map，就理解了整个框架。

配置文件（包括注解）描述下这个bean是什么样的，spring便可以维护一个map来管理创建的bean，并将其应用到需要的地方。

**构建bean的艺术**

书上讲`依赖注入`,`AOP`（其实还可以加上`工厂bean`）是spring的两个基本特性。其实是一个，它们都是一个关于如何构建bean的技术，beanFactory隐藏了创建Bean的细节。

假设BeanFactory是一个真的工厂，它的产品是bean，该工厂主要有以下工艺来生产Bean：

1. clazz.newInstance
2. factoryBean.getObject，向其它工厂订货
3. Proxy.xxxx，由Proxy组装
