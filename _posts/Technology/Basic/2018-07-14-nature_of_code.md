---

layout: post
title: 《编程的本质》笔记
category: 架构
tags: Basic
keywords: nature  code

---

## 简介

本文主要来自陈皓 的两篇文章 以及付费专栏《左耳听风》，作者的水平很高，语句也很精炼，实在是没什么可提炼整理的，干脆就弄成读书笔记了。

## 编程的本质

### 两篇论文

Algorithms + Data Structures

1. 如果数据结构设计的好，算法会简单
2. 好的通用算法 应该用在不同的数据结构上

An algorithm can be regarded as consisting of a logic component, which specifies the knowledge to be used in solving problems, and a control component, which determines the problem-solving strategies by means of which that knowledge is used. The logic component determines the meaning of the algorithm whereas the control component only affects its efficiency. The efficiency of an algorithm can often be improved by improving the control component without changing the logic of the algorithm. We argue that computer programs would be more often correct and more easily improved and modified if their logic and control aspects were identified and separated in the program text.

Algorithm = Logic + Control

1. Logic 解决问题
2. Control 只影响效率
3. Logic 和 Control 没有关系
4. Logic 和 Control 如果分开，代码更容易改进和维护

算法的效率往往可以通过提高控制部分的效率来实现，而无须改变逻辑部分

### 揉和一下

[左耳朵耗子：编程的本质是什么？](http://www.gzhshoulu.wang/article/2101593) 

![](/public/upload/architecture/nature_of_code.png)

1. 算法的效率往往可以通过提高控制部分的效率来实现，而无须改变逻辑部分。**就像函数式编程中的 Map/Reduce/Filter，它们都是一种控制。而传给这些控制模块的那个 lambda 表达式才是我们要解决的问题的逻辑，它们共同组成了一个算法。最后，我再把数据放在数据结构里进行处理，最终就成为了我们的程序。**

2. 控制一个程序流转的方式，即程序执行的方式，并行还是串行，同步还是异步，以及调度不同执行路径或模块，数据之间的存储关系，这些和业务逻辑没有关系。
3. 代码复杂度的本质：

	* 业务逻辑的复杂度决定了代码的复杂度；
	* 控制逻辑的复杂度 + 业务逻辑的复杂度 ==> 程序代码的混乱不堪；
	* 绝大多数程序复杂混乱的根本原因：业务逻辑与控制逻辑的耦合。



## 程序的本质复杂性和元语言抽象

[程序的本质复杂性和元语言抽象](https://coolshell.cn/articles/10652.html)

1. 逻辑就是问题的定义，比如，对于排序问题来讲，逻辑就是“什么叫做有序，什么叫大于，什么叫小于，什么叫相等”？控制就是如何合理地安排时间和空间资源去实现逻辑。比如java collections 的sort 方法`public static <T> void sort(List<T> list, Comparator<? super T> comparator)`
2. 如果目标还是代码“简短、优雅、易理解、易维护”，那么代码优化是否有一个理论极限？这个极限是由什么决定的？**普通代码比起最优代码多出来的“冗余部分”到底干了些什么事情？程序的本质复杂性就是逻辑，非本质复杂性就是控制**。逻辑决定了代码复杂性的下限，也就是说不管怎么做代码优化，Office程序永远比Notepad程序复杂，这是因为前者的逻辑就更为复杂。如果要代码简洁优雅，任何语言和技术所能做的只是尽量接近这个本质复杂性，而不可能超越这个理论下限。
3. 理解”程序的本质复杂性是由逻辑决定的”从理论上为我们指明了代码优化的方向：让逻辑和控制这两个维度保持**正交关系**。**绝大多数程序不够简洁优雅的根本原因：逻辑与控制耦合**
4. 每种组件形式都代表了特定的抽象维度，组件复用只能在其维度上进行抽象层次的提升。比如，我们可以把常用的HashMap等功能封装为类库，但是不管怎么封装复用类永远是类，封装虽然提升了代码的抽象层次，但是它永远不会变成Lambda，而实际问题所代表的抽象维度往往与之并不匹配。
5. 逻辑决定了程序的本质复杂性，但接口不是表达逻辑的通用方式，那么是否存在表达逻辑的通用方式呢？ 通过元语言抽象让逻辑和控制彻底解耦！有两种方式：元编程（比如thrift、定义了thrift文件，并提供一个thrfit 编译器）；元驱动编程，类似于下文的通用检查用户注册信息的逻辑。那么我们编写代码时，如何从业务中发现“元”（逻辑），是一个很有意义的问题，可以从`Collections.sort(xx,comparator)`开始。

	

		var meta_create_user = {
		    form_id : 'create_user',
		    fields : [
		        { id : 'name', type : 'text', min_length : 3 },
		        { id : 'password', type : 'password', min_length : 8 },
		        { id : 'repeat-password', type : 'password', min_length : 8 },
		        { id : 'email', type : 'email' }
		    ]
		};




陈皓在给《代码整洁之道》中的序文提到：无论微观世界的代码，还是宏观层面的架构，无论是三种编程范式还是微服务架构，它们都在解决一个问题：**分离控制和逻辑**。所谓控制，就是对程序流转的与业务无关的代码或系统的控制（如多线程、异步、服务发现、部署、弹性伸缩等），所谓逻辑则是实实在在的业务逻辑，是解决用户问题的逻辑。控制和逻辑控制了整体的软件复杂度，有效的分离控制和逻辑会让你的系统得到最大的简化。