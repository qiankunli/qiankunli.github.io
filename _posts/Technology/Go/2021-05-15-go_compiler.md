---

layout: post
title: go编译器
category: 技术
tags: Go
keywords: go compiler

---

## 前言(持续更新)

* TOC
{:toc}

[Go语言编译器简介](https://github.com/gopherchina/conference/blob/master/2020/2.1.5%20Go%E8%AF%AD%E8%A8%80%E7%BC%96%E8%AF%91%E5%99%A8%E7%AE%80%E4%BB%8B.pdf) 未读完
1. N种语言+M种机器=N+M个任务，有几种方案
    1. 其它语言 ==> C ==> 各个机器
    2. 各个语言 ==> x86 ==> 各个机器
2. 通用编译器方案
    ![](/public/upload/basic/general_compiler.png)

SSA-IR（Single Static Assignment）是一种介于高级语言和汇编语言的中间形态的伪语言，从高级语言角度看，它是（伪）汇编；而从真正的汇编语言角度看，它是（伪）高级语言。顾名思义，SSA（Single Static Assignment）的两大要点是：
1. Static：每个变量只能赋值一次（因此应该叫常量更合适）；
2. Single：每个表达式只能做一个简单运算，对于复杂的表达式a*b+c*d要拆分成："t0=a*b; t1=c*d; t2=t0+t1;"三个简单表达式；

## go编译器

![](/public/upload/go/go_compiler.png)

[漫谈Go语言编译器（01）](https://mp.weixin.qq.com/s/0q0k8gGX56SBKJvfMquQkQ) 