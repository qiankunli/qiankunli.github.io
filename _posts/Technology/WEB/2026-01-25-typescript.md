---

layout: post
title: typescript学习
category: 技术
tags: WEB
keywords:  typescript

---

## 简介（未完成）

JavaScript 作为一门拥有悠久历史的脚本语言，几乎无处不在，而 TypeScript 作为其超集，它们之间最核心的区别在于 静态类型系统（意味着变量的类型在程序中的任何时候都不能改变）。 TypeScript 代码需要先编译成 JavaScript 才能在浏览器中运行。

## 基础知识
1. 在 TypeScript 中，any 类型被称为 top type。所谓的 top type 可以理解为通用父类型，也就是能够包含所有值的类型。any 类型本质上是类型系统的一个逃生舱口，TypeScript 允许我们对 any 类型的值执行任何操作，而无需事先执行任何形式的检查。如果代码里使用了大量的 any，那 TypeScript 也就失去了意义，所以我们应该尽量避免使用 any 。
2. 为了解决 any 类型存在的安全隐患，在 TypeScript 3.0 时，引入一个新的 top type —— unknown 类型。同 any 一样，你也可以把任何值赋给 unknown 类型的变量。两者有啥区别呢？「any 类型：我不在乎它的类型，unknown 类型：我不知道它的类型。」你可以把它理解成类型安全的 any 类型。相比 any 类型，TypeScript 会对 unknown 类型的变量执行类型检查