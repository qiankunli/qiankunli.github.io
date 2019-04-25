---

layout: post
title: 重新看面向对象设计
category: 架构
tags: Architecture
keywords: 面向对象

---

## 简介

本文最初由[apollo client源码分析](http://qiankunli.github.io/2017/09/23/apollo_client_source.html) 引发，后来加入了 陈皓《左耳听风》教程的相关内容。

我一直有一个比方，如果把程序员写程序比作史官写历史，那么面向过程就是编年体通史，而面向对象更像是纪传体通史。编年体通史以时间（或一段时间）为中心，

    公元xxx年，xxx攻打xxx，克xx城
    公元xx年，xx卒
    
而纪传体通史则以人物为中心，譬如《史记》的《高祖本纪》等。类似的，面向过程多认为程序由一个个函数组成（依照顺序先后调用），譬如历史由一件件大事组成。而面向对象倾向于认为程序由一个个对象组成，譬如历史由一个个人的故事组成。

## 喷一喷面向对象

2019.1.2 补充

[如此理解面向对象编程](https://coolshell.cn/articles/8745.html)，有一个需求：代码检查操作系统类型，若是linux 输出：linux很不错；若是windows，输出windows 很差

1. 过程化的方案
2. 一般面向对象方案（一个os 抽象class，一个具体os 对应一个子class）
3. 面向对象进化：不仅弄子类，还弄一map 保存os 和 子类的关系
4. 大神 Rob Pike 对此的评论是：根本就不需要什么Object，只需要一张小小的配置表格，里面配置了对应的操作系统和你想输出的文本。这不就完了。所谓的代码进化相当疯狂和愚蠢的，这个完全误导了对编程的认知。


还有的人喜欢用Object来替换所有的if-else语句，他们甚至还喜欢把函数的行数限制在10行以内 [programming in the
twenty-first century](https://prog21.dadgum.com/156.html)
6. **那23个经典的设计模式和OO半毛钱关系没有**，只不过人家用OO来实现罢了。设计模式就三个准则：1）中意于组合而不是继承，2）依赖于接口而不是实现，3）高内聚，低耦合。你看，这完全就是Unix的设计准则。


[Don't Distract New Programmers with OOP](https://prog21.dadgum.com/93.html)

The shift from procedural to OO brings with it a shift from thinking about problems and solutions to thinking about architecture. That's easy to see just by comparing a procedural Python program with an object-oriented one. The latter is almost always longer, full of extra interface and indentation and annotations. The temptation（诱惑） is to start moving trivial bits of code into classes and adding all these little methods and anticipating（预料） methods that aren't needed yet but might be someday. 封装对象、类、接口等对很多简单代码来说是不必要的。

When you're trying to help someone learn how to go from a problem statement to working code, the last thing you want is to get them sidetracked（转移话题） by faux（人造的）-engineering busywork（作业、额外工作）. Some people are going to run with those scraps（点滴） of OO knowledge and build crazy class hierarchies and end up not as focused on on what they should be learning. Other people are going to lose interest because there's a layer of extra nonsense（无意义的） that makes programming even more cumbersome（笨重的）.

面向对象逼着你除了思考问题本身外，还要思考结构、设计，很多人无此意识或功力不足， 滥用面向对象的特性，整出大量无意义的代码，使得代码复杂度大大超过了问题本身的复杂度。[函数式编程的设计模式](http://qiankunli.github.io/2018/12/15/functional_programming_patterns.html) 面向对象设计模式经常在搞一件事，把动词转换为名词，但很不幸，这个动作很多时候没有必要。



## 左耳听风

来自陈皓《左耳听风》付费课程，建议先看下[java 语言的动态性](http://qiankunli.github.io/2018/08/15/java_dynamic.html)

在面向对象编程里，计算机程序会被设计成彼此相关的对象，独立而又相互调用。传统程序主张将程序看做一系列指令，或一系列函数。面向对象设计中的每一个对象 都应该能够接受数据、处理数据并将数据传递给其它对象。

面向对象的缺点：通过对象来达到抽象效果， 把代码分散在不同的类里面。那要让它们执行起来，就需要将这些类粘合起来。设计模式以及ioc 等虽然精巧，但不得不怀疑点歪了科技树。一段代码的执行路径 `obj1.func1 ==> obj2.func2 ==> obj3.func3` 在函数式编程中 `func1(func2(func3))` 就解决了。

![](/public/upload/java/object_oriented_3.png)

换个角度想一下，架构设计从单体演化到微服务架构，固然一部分是单机无法负载，另一个原因就是单体 在维护和运维上的困难，比如一个小改动导致整个项目的重启。也就是说，架构设计之初，就越来越考虑可维护性和扩展性。**架构设计不只考虑实现功能，可维护性和扩展性影响了架构设计，对应的，可维护性和扩展性影响了代码结构。**指令序列或函数序列被解构，分散在各个对象中，以减少修改对整体的影响。但从上图可以看到，面向对象的优点 也直接导致了其缺点。

宏观上的系统设计与类设计

![](/public/upload/java/object_oriented_1.png)

可以认为，db 就是controller-service-dao  这些类的状态。

微观上的类实现与系统模块实现

![](/public/upload/java/object_oriented_2.png)

你设计一个controller-service-dao 项目 制定http api 时，肯定会想业务层面会有哪些调用，绝不会一个http api 干了一半的活儿 然后让调用方自己 访问两个 http api 自己聚合数据。对应的，我们在设计类时，根据类持有的数据/状态，一个对象可以访问数据库、可以内部操作线程，但其对外提供的interface function 应该是自洽的（对外隐藏掉不必要的细节）。类似的概念可以看 [ddd(一)——领域驱动理念入门](http://qiankunli.github.io/2017/12/25/ddd.html)


### 一个神奇的例子

	void business(){
		Lock lock = xx;
		if(condition){
			return;
		}
		business code;
		lock.unlock();
	}
	
这里有一个问题， 符合condition 情况下代码提前退出，则没有执行unlock 操作。但如果condition 很多，到处都是`lock.unclock()` 也比较乱。则可以

	class LockGuard{
		Lock lock
		构造函数{
			lock = new Lock();
		}
		析构函数{
			lock.unLock()
		}
	}
	void business(){
		LockGuard lockGuard = new xx;
		if(condition){
			return;
		}
		business code;
	}


## 从分析apollo client 得来的

2018.7.12 补充

### 面向对象和基于对象

大部分人写出的java 代码，可能只是基于对象。

基于对象，通常指的是对数据的封装，以及提供一组方法对封装过的数据操作。比如 C 的 IO 库中的 FILE * 就可以看成是基于对象的。

面向对象的程序设计语言必须有描述对象及其相互之间关系的语言成分。这些程序设计语言可以归纳为以下几类：

1. 系统中一切事物皆为对象；
2. 对象是属性及其操作的封装体；
2. 对象可按其性质划分为类，
3. 对象成为类的实例；
4. 实例关系和继承关系是对象之间的静态关系；
5. 消息传递是对象之间动态联系的唯一形式，也是计算的唯一形式；
6. 方法是消息的序列。

**笔者曾看到一篇如何分析源码的文章，文中提到的一个重要建议就是画类图，并将类图比作一个地图的各个山头，山头虽然不是地图的全部，但却撑起了类图的骨架。然后再由一条 业务逻辑的主线 串起来各个山头（即各个山头的相互调用）**

### 面向对象的渊源

面向对象的编程产生的历史原因：**由于面向过程编程在构造系统时，无法解决重用，维护，扩展的问题，**而且逻辑过于复杂，代码晦涩难懂。PS：笔者第一次看到这句话时没有感觉，后来忘记了，看《左耳听风》时自己总结了这句，再看到这句早就看到的话时，知己二字不能形容。

[《面向对象分析与设计》读书笔记 （1）- 关键的思想](https://zhuanlan.zhihu.com/p/27106866) 要点如下

1. 复杂性是面向对象主要解决的问题,复杂系统的5个属性

	* 层次结构,复杂性常常以层次结构的形式存在,层次结构的形式

		* 组成（”part of“）层次结构
		* 是一种“("is a")层次结构
	* 相对本原，这里是指构建系统的最小单位。你不需要担心基础组件是如何实现的，只要利用其外部行为即可。举个例子，你要盖一个房子，你需要砖，水泥等，这些都是一些基础组件，但是你不要自己去生产砖，水泥。
	* 分离关注,组件内的联系通常比组件间的联系更强。这一事实实际上将组件中高频率的动作（涉及组件的内部结构）和低频率的动作（涉及组件间的相互作用）区分开来
	* 共同模式,复杂系统具有共同的模式。比如小组件的复用，比如常用方案提炼为设计模式等
	* 稳定的中间形式（注意不是中间件），复杂系统是在演变中诞生的，不要一开始就期望能够构建出一个复杂的系统。要从简单系统逐步迭代到复杂的系统。

2. 思考分解的方式

	1. 系统中每个模块代表了某个总体过程的一个主要步骤。邮寄快递时，我们先将物品准备好，找到快递员，填写快递信息，进行邮寄。在这个过程中，我们分成了4个步骤，我们更注重的是事件的顺序，而非主要关注参与者。
	2. 根据问题域中的关键抽象概念对系统进行分解。针对上面的快递邮寄的例子，采用面向对象分解时，我们分解成4个对象：物品，快递单，快递员，我。我拥有物品，然后向快递员发出请求，快递员给我快递单让我填写快递信息。然后快递员进行邮递。

3. 编程风格，Bobrow将编程风格定义为“一种组织程序的方式，基于某种编程概念模型和一种适合的语言，其目的是使得用这种风格编写的程序很清晰”

4. 对象模型的4个主要要素：抽象；封装；模块化；层次结构；3个次要要素：类型、持久、并发

5. Shaw对于抽象的定义："对一个系统的一种简单的描述或指称，强调系统的某些细节或属性同时抑制另一些细节或属性。好的抽象强调了对读者或者用户重要的细节，抑制了那些至少是暂时的非本质细节或枝节" （我以前的思维漏洞 就是不知道 抑制非本质细节）
6. 封装的意义，复杂系统的每一部分，都不应该依赖于其他部分的内部细节。要让抽象能工作，必须将实现封装起来
7. 模块化的意义
8. 层次结构的意义

### 实例分析

关联[apollo client源码分析](http://qiankunli.github.io/2017/09/23/apollo_client_source.html) 阅读

所以，我们讲面向对象，重点不是业务流程（不是说不重要，而是不论怎么写代码，都要按业务顺序执行，这点无疑问）。就好比，apollo client，重点也不是 向服务端 获取数据更新本地数据 这些事儿。而是

1. apollo client 在给开发人员的接口 是什么样的，包括配置 变化时 驱动listener 执行。 
2. 向服务端 申请数据（推拉）以及更新本地数据 这些事儿 如何抽象
3. 两个 抽象域 如何 交集

apollo client 一个牛逼之处 在于 第二个 抽象域，提取出 ConfigRepository 的抽象，RemoteRepository LocalRespository 并标记 每个ConfigRepository 有一个parent

### 多线程与对象的 关系

我以前的认知，线程只是一个 驱动者，驱动代码执行，对象跟线程没啥关系。一个典型的代码是

	class XXThread extends Thread{
		private Business business;
		public XXThread(Business business){
			this.business = business;
		}
		public void run(){
			business code
		}
	}
	
在apollo client 中，RemoteRepository 内部聚合线程 完成配置的周期性拉取，线程就是一个更新数据的手段，只是周期性执行下而已。 

	class Business{
		private Data data;
		public Business(){
			Executors.newSingleThread().execute(new Runnable(){
				public void run(){
					acquireDataTimely();
				}
			});
		}
		public void acquireDataTimely(){}
		public void useData(){}
		public void transferData(){}
	
	}
	
从两段代码 看，线程与对象的 主从关系 完全相反。[程序的本质复杂性和元语言抽象](https://coolshell.cn/articles/10652.html)指出：程序=control + logic。 同步/异步 等 本质就是一个control，只是拉取数据的手段。因此，在我们理解程序时，同步异步不应成为本质的存在。

## 小结

本文着重从面向对象渊源的角度来说明：**面向过程编程在构造系统时，无法解决重用，维护，扩展的问题**，进而说明面向对象将 重用、维护、扩展加入了设计理念中，进而体现在语言的方方面面。

[ddd(一)——领域驱动理念入门](http://qiankunli.github.io/2017/12/25/ddd.html)提到，面向对象设计的一种概述是：提取抽象以及抽象之间的关系。针对具体的业务特性，二十几种设计模式，**每一种设计模式（尤其是行为型设计模式）都在指导我们如何划分对象以及对象之间的关系**。**领会设计模式，是我们领会面向对象设计思想的一个入口**。所以当我们想着如何提高自己的“面向对象”的设计能力时，直观的做法尽量多用符合业务场景的设计模式。也因此，我们在分析 apollo client 时，其核心结构 便是一个观察者模式。

2019.1.2补充：上面一段话叫“设计模式驱动编程”，在 [如此理解面向对象编程](https://coolshell.cn/articles/8745.html) 被批判了。