---

layout: post
title: javascript应用在哪里
category: 技术
tags: WEB
keywords: javascript

---

## 简介

## 进程虚拟机和系统虚拟机

[一篇给小白看的 JavaScript 引擎指南](http://web.jobbole.com/84351/)

“系统虚拟机”提供了一个可以运行操作系统的完整仿真平台。Mac 用户很熟悉的 Parallels 就是一个允许你在 Mac 上运行 Windows系统虚拟机。

另一方面，“进程虚拟机”不具备全部的功能，能运行一个程序或者进程。Wine 是一个允许你在 Linux 机器上运行 Windows 应用的进程虚拟机，但是并不在 Linux 中提供完整的 Windows 操作系统。

JavaScript 虚拟机是一种进程虚拟机，专门设计来解释和执行的 JavaScript 代码。

|语言|引擎|类型|
|---|---|---|
|java|jvm|编译|
|python|python 解释器|解释|
|javascript|javascript引擎|解释|

**预处理：**有时为了提高运行效率，会将javascript源码先编译成字节码，然后解释器（引擎）逐行执行字节码。java也有类似的目的，比如代码引用另外一个jar，因为jar都是class文件，所以运行时就省了编译jar java源码的负担。

## javascript的一些特性

[JavaScript 运行机制详解：再谈Event Loop](http://www.ruanyifeng.com/blog/2014/10/event-loop.html)

主要有以下基本特点：

1. JavaScript是单线程
2. 任务队列
 
## 浏览器环境
 
JavaScript代码嵌入网页的方法

1. `<script>`标签
2. 事件属性：代码写入HTML元素的事件处理属性，比如onclick或者onmouseover

网页加载流程

1. 浏览器一边下载HTML网页，一边开始解析
2. 解析过程中，发现`<script>`标签
3. 暂停解析，网页渲染的控制权转交给JavaScript引擎
4. 如果`<script>`标签引用了外部脚本，就下载该脚本，否则就直接执行
5. 执行完毕，控制权交还渲染引擎，恢复往下解析HTML网页

浏览器的核心是两部分：渲染引擎和JavaScript解释器（又称JavaScript引擎）。渲染引擎的主要作用是，将网页代码渲染为用户视觉可以感知的平面文档。渲染引擎根据JavaScript提供的桥接接口提供给JavaScript访问DOM的能力.


## native 环境

[React Native 初探（iOS）](http://www.hotobear.com/?p=1015)

JavaScript（引擎）在浏览器中的应用几乎是尽人皆知的。实际上，JavaScript技术也可以使用在非浏览器应用程序当中，从而让应用程序具有自动的脚本功能。

javascript引擎一般由C/c++实现。

我们听说过，在java中弄一个groovy引擎+groovy脚本、lua引擎+lua脚本、javascript引擎+javascript脚本 来处理一些经常变化的需求。**在ios上，用object-c嵌入javascript引擎，可以解释执行JavaScript代码（object-c与js彼此交互）。如果这个javascript 代码占了大头，带来的一个影响就是：开发ios app，从感觉上，由写object-c程序，变成了在写java script。**

## V8 执行 JS 的过程
1. 首先对 JS 源代码进行词法分析，将源代码拆分成一个个简单的词语（即 Token）；然后，以这些 Token 为输入流进行语法分析，形成一棵抽象语法树（即 AST），并检查其语法上的错误；最后，由语法树生成字节码，由 JS 解析器运行。
2. 在代码中，常常会有同一部分代码，被多次调用，同一部分代码如果每次都需要解释器转二进制代码再去执行，效率上来说，会有些浪费，所以在 V8 模块中会有专门的监控模块，来监控同一代码是否多次被调用，如果被多次调用，那么就会被标记为热代码。当存在热代码的时候，V8 会借助 TurboFan (优化编译器)将为热代码的字节码转为机器码并缓存下来，这样一来，当再次调用热代码时，就不再需要将字节码转为机器码。
