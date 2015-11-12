---

layout: post
title: 自己动手写spring（六） 工厂bean的实现
category: 技术
tags: Java
keywords: Java Spring

---

## 前言

为什么需要工厂bean？回答这个问题，首先要回答下为什么需要工厂模式？简单说，就是不用`new xxx()`了。用`new xxx()`有什么不好呢？简单说就是：不能适应变化。对于传统的`new xxx()`,假设这个代码有10个地方使用，那么当`xxx()`改变时，你便要更改10个地方的代码（好吧，我承认，借助于ide，这个优势不是很明显）。

`Object obj = xxxFactory.getBean(xxx)`体现到配置文件就是



