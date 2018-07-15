---

layout: post
title: apollo client源码分析及看待面向对象设计
category: 技术
tags: Java
keywords: apollo configcenter

---

## 前言

[apollo](https://github.com/ctripcorp/apollo) 是携程开源的配置管理平台，本文主要研究其apollo-client源码实现。

apollo-client 的使用参见 [Java客户端使用指南](https://github.com/ctripcorp/apollo/wiki/Java%E5%AE%A2%E6%88%B7%E7%AB%AF%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97)

apollo-client 的主要包括以下几个部分：

1. 提出一套数据结构以存储远端的配置信息
2. 一套同步机制负责本地与远程配置的同步
3. 和spring的整合

从功能上说，主要包括

1. 为性能考虑，使用长连接。为防止长连接失效，支持周期性拉取。同时配置持久化到本地，以防止服务端失联。
2. 支持多种格式的配置文件
3. 配置更新时，要更新内存中bean的属性值，更新本地的缓存文件等。

这么多功能，我要实现的话，都不知到从何入手，这也是分析其实现的动力所在。

## 类图分析

### config

![](/public/upload/java/apollo_client_config.png)

Config 定义外界使用接口，AbstractConfig处理掉Cache并用m_executorService处理掉ConfigChangeListener回调。真正配置的获取工作由DefaultConfig来完成（AbstractConfig并没有实现Config的`String getProperty(name,defaultValue)`），DefaultConfig将获取配置的工作交给ConfigRepository来负责，自己带上RepositoryChangeListener，对Config RepositoryChange事件作出处理。

或者说，从顶层设计上，`DefaultConfigFactory.create`将整体架构分为Config和ConfigRepository两个部分。

	public Config create(String namespace) {
	        DefaultConfig defaultConfig =
	                new DefaultConfig(namespace, createLocalConfigRepository(namespace));
	        return defaultConfig;
	    }

**此处可以有一个问题，为什么是DefaultConfig实现RepositoryChangeListener，而不是AbstractConfig实现RepositoryChangeListener？**因为AbstractConfig重点在cache property和处理ConfigChangeListener回调。 其abstract 方法`String getProperty(name,defaultValue)` 并不要求底层跟ConfigRepository扯上关系，其简单实现可以是加载一个本地properties文件。

### ConfigRepository

![](/public/upload/java/apollo_client_config_repository.png)

ConfigRepository 定义外界使用接口`ConfigRepository`和`addChangeListener(in RepositoryChangeListener listener)`，同时ConfigRepository定义了`setUpstreamRepository(in ConfigRepository upstreamConfigRepository)`自用，以确保ConfigRepository 之间具备关联关系。

AbstractConfigRepository 处理RepositoryChangeListener回调，并提供了一个很有意义抽象trySync。对于LocalFileConfigRepository来说，其关联一个RemoteConfigRepository，sync逻辑就是将RemoteConfigRepository数据视情况同步到本地。对于RemoteConfigRepository，sync逻辑就是拉取远端配置。

### Config和ConfigRepository 的关系

Config和ConfigRepository两个继承体系，在DefaultConfig那里分叉，但又**双向关联**，形成一个完善功能的整体。

||说明|
|---|---|
|正向关联| DefaultConfig 操作 ConfigRepository |
|反向关联| ConfigRepository 操作 RepositoryChangeListener。在Config 继承体系中，由子类DefaultConfig 实现RepositoryChangeListener |

**这是一种常见的设计方式，一个继承体系的顶层接口定义该继承体系的基本功能。其子类在继承上层接口或父类的同时，还继承其它接口，作为与其它继承体系互操作的基本约定。一个类实现很多功能，但面向不同的角色，只希望暴露有限的功能，这是子类继承其它接口的初衷。**从另一个角度说，从一个继承体系抽离出另一个继承体系时，两个继承体系通过一个接口互操作。

### ConfigService

![](/public/upload/java/apollo_client_config_service.png)

`Config config = ConfigService.getConfig(namespace)`是apollo-client的代码接口，ConfigService 缓存 namespace 和 Config的映射关系，**通过map达成缓存和单例效果。** ConfigRegistry负责维护手动注入的namespace和Config映射（作用不明），ConfigFactoryManager 负责维护namespace和ConfigFactory的关系，ConfigFactory负责根据namespace创建Config。

有一个ConfigUtil 贯穿所有类，以在必要的时候获取配置（配置经过简单的处理）。

## 流程梳理

扳机在哪？哪些是主干，哪些是枝节？

首先为优化性能，各种缓存先不考虑。同样，为丰富功能，各种Listener也先省掉。我们发现主干就剩下Config和ConfigRepository，Config抽象配置使用，ConfigRepository抽象配置获取，它们的连接就是ConfigRepository 的`Properties getConfig()`

## 从中学习到的


### 分析代码

一个类有很多种成员，但一定有一个（最多两三个）跟这个类的功能关系最大。

debug代码，从头到尾走一遍（在最后几步停下，看以前的堆栈），把涉及到的几个class diagram画出来。

### 自己从头到尾实现

代码是逐渐演化的，先实现最核心的功能。从配置中心的功能可以看出，最核心的是：从远程拉取配置存在内存中。附加功能涉及到

1. 缓存配置以优化性能
2. 本地持久化，与服务端失联时使用本地配置
3. 单例、线程安全等问题

如何有一个通用的思维方法，可以导到作者的实现上？

**以代码演化的角度看待架构设计**，我们在实现附加功能的时候，不是简单的添加代码，这即涉及架构设计，也涉及到重构

1. 提取超类、子类，纵向扩展继承体系
2. 独立的功能分离，扩展成多个继承体系

作者关于ConfigRepository的设计还是非常精巧，**自关联**，由我来实现的话，或许还是要

	ConfigRepository{
		Properties remoteProperties;	// 缓存远端结果，专职拉取线程拉到配置后存在这里
		Properties getConfig(){
			1. 连接状态正常，则从remoteProperties拿数据
			2. 否则从本地Properties拿数据
		}
		sync(){
			1. 从远端数据拿到后，更新remoteProperties及数据版本
			2. 视数据版本更新本地Properties
		}
	}
	
但其实就本地Properties来说，根据配置读取Properties存储地址，Properties文件更新等，代码角度较多，抽取出来是一个很必要的选择。

## 重新来看待 面向对象

2018.7.12 补充

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

面向对象的编程产生的历史原因：由于面向过程编程在构造系统时，无法解决重用，维护，扩展的问题，而且逻辑过于复杂，代码晦涩难懂。

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

所以，我们讲面向对象，重点不是业务流程（不是说不重要，而是不论怎么写代码，都要按业务顺序执行，这点无疑问）。就好比，apollo client，重点也不是 向服务端 获取数据更新本地数据 这些事儿。而是

1. apollo client 在给开发人员的接口 是什么样的，包括配置 变化时 驱动listener 执行。 
2. 向服务端 申请数据（推拉）以及更新本地数据 这些事儿 如何抽象
3. 两个 抽象域 如何 交集

apollo client 一个牛逼之处 在于 第二个 抽象域，提取出 ConfigRepository 的抽象，RemoteRepository LocalRespository 并标记 每个ConfigRepository 有一个parent

多线程与对象的 关系。我以前的漏洞，线程只是一个 驱动者，驱动代码执行，对象跟线程没啥关系。一个典型的代码是

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
	
从两段代码 看，线程与对象的 主从关系 完全相反。[程序的本质复杂性和元语言抽象](https://coolshell.cn/articles/10652.html)指出：程序=control + logic。 同步/异步 等 本质就是一个control，只是拉取数据的手段。因此，在我们理解程序时，不应成为本质的存在。