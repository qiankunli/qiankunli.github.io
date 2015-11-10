---

layout: post
title: 自己动手写spring 总结
category: 技术
tags: Java
keywords: Java Spring

---

## 前言

作为一个使用轮子的程序员第一次写了轮子。

1. 重要的是拿三个map
2. 书上讲`依赖注入`,`AOP`（其实还可以加上`工厂bean`）是spring的两个基本特性。其实是一个，它们都是一个关于如何构建bean的技术。配置文件（包括注解）描述下这个bean是什么样的，spring便可以维护一个map来管理创建的bean，并将其应用到需要的地方。


（BeanFactory）如何创建bean，该Factory主要有以下工艺：

1. clazz.newInstance
2. factoryBean.getObject
3. Proxy.xxxx
