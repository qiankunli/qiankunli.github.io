---

layout: post
title: 如何看待大牛的经验文
category: 生活
tags: Life
keywords: 认知

---

## 简介

网络上经常会有各种大牛（cto、阿里架构师等）的分享，诸如：

1. 技术人员2年、3年、5年如何提高自己
2. 程序猿自我修炼之路
3. 工作八年、十年总结

会列出各种路线图，分为几类

1. 过程式，即不同阶段要做什么
2. 目标式，即不管干什么，到一定要阶段要掌握xx。比如：

	1. 分析源码，主要指常用设计模式、spring3/4/5、mybatis
	2. 分布式架构，包括原理、中间件以及 应用层框架（计算、微服务、存储等）
	3. 并发编程、性能调优
	4. 开发工具工程化，包括maven、jenkins、sonar、git等
	5. 项目实战，比如一个b2c项目包括：用户认证、店铺商品、订单支付、数据统计分析、通知推送等

技术人员在面对这些攻略，要认识到以下几点，否则这样的文章看的越多就越焦虑和困惑：

1. 有自己的侧重，比如笔者就觉得spring 代码组织的不太好，在买过一本书对其原理有大致体会后便没有深究，springmvc 源码笔者到现在都没有深入看过。

	* 有的东西不理解到细节难受，你知道底层原理，然后可以推知上层所有因果
	* 有的东西不耽误用就行，可以通过博客等把别人二手结论拿来用。比如要实现一个自定义注解，博客说spring xx组件可以实现，你demo 做出来就可以用在项目中。
	* 你对项目的定位（自己学到何种程度）要有自己的判断，当然，这个判断要根据实际情况调整。
2. 有自己的路线图，靠兴趣、“事到临头”来推动。别人是1=>2=>3开始学，你2==>3==>1学也没什么问题，甚或是2.1 ==>1.6 ==>3.2==>2.5。比如jvm调优很有意义，但一则很多人用不上，二则过早接触也看不懂。你先看点，关键时刻知道有这么个事儿就行。**一般来说，只要你追求去做更大和复杂的项目，123终究会体验全的。**
3. 看文章 要**为我所用**，前提是你自己有一套取舍观、方法论和路线图。看文章的目的不是刷新自己，而是吸取自己之前没注意到的知识、观点和方法论，添长处去短板。

为何要想这些东西，因为如果这些东西想不清楚，他们会一次次来占用你的精力、带来困惑和烦扰。以后看到这类“经验文” 应该不会再引起难受了。**有一句话：很多人为了不思考愿意做任何事情。但其实，很多人没认识到该思考这个问题，也没认识到一直拒绝思考导致自己付出了多大的代价。**

**思考是对复杂事务、信息降维处理， 以便于主动规划，而不总是被动应对。你应该先有一套知识图谱、方法论，然后碰到新东西，去充实它们。而不是左支右绌，忙于应对。**

## 知识模型

[开发人员和架构师的知识模型](https://mp.weixin.qq.com/s/C8o7emsIzm7eOoBqWUjbyQ)
1. 作为开发人员，更加关注知识的深度，以便有足够的知识储备满足工作需要。开发人员在职业生涯的早期，应该关注于自身知识储备的增长，并保持技术深度。PS：学习了，深度=你知道自己知道。广度=你知道自己知道+你知道自己不知道。
	![](/public/upload/life/developer_breadth_depth.jpg)
2. 作为架构师，之所以技术的广度比深度更重要，是因为架构师的重要职责之一是进行架构决策。系统架构设计是关于权衡的艺术，在特定的问题域上下文下，架构师需要在诸多可行的解决方案间进行权衡和决策，这也对其技术广度提出了要求。开发人员成长为架构师，应该更加关注知识的广度，并在几个特定领域深耕，以便有足够的知识支撑架构决策。
	![](/public/upload/life/architecture_breadth_depth.jpg)

虽然开发人员和架构师在知识域的关注点上存在差异，但在认知层面都可以统一到Bloom认知层次模型。该模型将认知层次划分为逐步递进的六个层次：
![](/public/upload/life/bloom_taxonomy.jpg)

不论是架构师还是开发人员，Bloom认知层次模型都适用。通过不断的学习扩展自身的知识体系，在识记、理解和应用的同时，要持续的培养分析、评估和创造的能力，逐步向高层次的认知水平提升。但需要注意的是：**知识不等于认知**，避免陷入知识学习的陷阱。知识是无限的，没有人能够以有限的精力去学习无限的知识。不论是开发人员还是架构师，又或者其他角色，不应该只将精力投入在知识边界的扩充，而**应该注重从知识到认知提升的转变**。格物以致知，对表象不断的归纳、演绎直至事物的本象，探寻事物背后的规律，建立更高层的认知。这种认知层次由下及上的跃升有两种方式：
1. 悟：由内向外，通过不断积累、持续思考，由量变到质变，直至 “开悟”
2. 破：自外向内，高层次或不同的思想输入碰撞，加速认知层次的突破

![](/public/upload/life/knowledge_recognize.jpg)