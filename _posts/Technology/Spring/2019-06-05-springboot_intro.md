---

layout: post
title: springboot 入门
category: 技术
tags: Spring
keywords: springboot

---

## 简介


## spring 当前的问题

Spring的组件代码是轻量级的，但它的配置却是重量级的

1. 一开始，Spring用XML配置， 而且是很多XML配置。
2. Spring 2.5引入了基于注解的组件扫描，这消除了大量针对应用程序自身组件的显式XML配置。
3. Spring 3.0引入了基于Java的配置，这是一种类型安全的可重构配置方式， 可以代替XML。
4. 开启某些Spring特性时，比如事务管理和Spring MVC，还是需要用XML或Java进行显式配置。启用第三方库时也需要显式配置，比如基于 Thymeleaf的Web视图。


项目的依赖管理也是件吃力不讨好的事情。决定项目里要用哪些库就已经够让人 头痛的了，你还要知道这些库的哪个版本和其他库不会有冲突，这难题实在太棘手。