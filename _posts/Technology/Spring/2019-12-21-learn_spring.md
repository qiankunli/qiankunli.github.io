---

layout: post
title: 学习Spring
category: 技术
tags: Spring
keywords: springboot

---

## 简介

* TOC
{:toc}


![](/public/upload/spring/spring_value.png)

![](/public/upload/spring/spring_features.png)

## 先有思想后有的支持思想的feature

常规的说法是：AOP的实现用到了动态代理技术。但更准确的说：**动态代理 是java 内嵌的对面向切面编程的支持**

## 元编程

当我们刚刚开始学习和了解编程这门手艺或者说技巧时，一切的知识与概念看起来都非常有趣，随着学习的深入和对语言的逐渐了解，我们可能会发现原来看起来无所不能的编程语言成为了我们的限制，我们只能一遍一遍地写重复的代码来解决本可以轻松搞定的问题。


元编程（Metaprogramming）是计算机编程中一个非常重要、有趣的概念，维基百科 上将元编程描述成一种计算机程序可以将代码看待成数据的能力。

Metaprogramming is a programming technique in which computer programs have the ability to treat programs as their data.

如果能够将代码看做数据，那么代码就可以像数据一样在运行时被修改、更新和替换；元编程赋予了编程语言更加强大的表达能力，通过编译期间的展开生成代码或者允许程序在运行时改变自身的行为。归根结底就是一种使用代码生成代码的思想，消灭重复的代码，极大地增强编程语言的表达能力。

## 编程模型

面向对象编程

1. 契约接口，Aware,BeanPostProcessor...。 ioc 约定了会对这些接口类进行处理
2. 设计模式，观察者模式（ApplicationEvent）、组合模式（Composite*）、模板模式（JdbcTemplate/RestTemplate）...
3. 对象继承，Abstract*类

面向切面编程

1. 动态代理，JdkDynamicAopProxy
2. 字节码提升，ASM、CGLib、AspectJ...

面向元编程

1. 注解，模式注解（@Component/@Service/@Respository...）
2. 配置，Environment抽象、PropertySources、BeanDefinition...
3. 泛型，GenericTypeResolver、ResolvableType...

因为java 注解不允许继承，所以会在“子注解”上进行源标注。

函数驱动

1. 函数接口，ApplicationEventPublisher
2. Reactive,Spring WebFlux

模块驱动

1. Maven Artifacts
2. OSGI Bundies
3. Java 9 Automatic Modules
4. Spring @Enable*

