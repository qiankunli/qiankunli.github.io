---

layout: post
title: 编写java框架的几大利器
category: 技术
tags: Java
keywords: JAVA spi xsd

---

## 前言（未完成）

## spi

[java中的SPI机制](http://www.cnblogs.com/javaee6/p/3714719.html)

java 平台对spi的支持可以参见java.util.ServiceLoader.A simple service-provider loading facility类。比如对一个maven项目结构


	src
		main
			java
				org.lqk.spi
					IUserService
					UserService
			resources
				META-INF
					services
						org.lqk.spi.IUserService
						

那么在代码中

	public class ServiceBootstrap {
		  public static <S> S loadFirst(Class<S> clazz) {
		    Iterator<S> iterator = loadAll(clazz);
		    if (!iterator.hasNext()) {
		      throw new IllegalStateException(String.format(
		              "No implementation defined in /META-INF/services/%s, please check whether the file exists and has the right implementation class!",
		              clazz.getName()));
		    }
		    return iterator.next();
		  }
		  private static <S> Iterator<S> loadAll(Class<S> clazz) {
		    ServiceLoader<S> loader = ServiceLoader.load(clazz);
		    return loader.iterator();
		  }
	}


`ServiceBootstrap.load(IUserService.class)`即可得到IUserService实例。


类似的ioc工具有[ Plexus，Spring之外的IoC容器](http://blog.csdn.net/huxin1/article/details/6020814)， [google/guice](https://github.com/google/guice)

使用它们而不是spring ioc有一个好处：我们所写的框架，经常应用于很多使用spring的项目中，不使用spring ioc，可以不与业务使用的spring版本冲突。

贴一段guice的简介：Put simply, Guice alleviates the need for factories and the use of new in your Java code. Think of Guice's @Inject as **the new new（一个新的new）.** You will still need to write factories in some cases, but your code will not depend directly on them. Your code will be easier to change, unit test and reuse in other contexts.

## xsd

[ spring框架的XML扩展特性：让spring加载和解析你自定义的XML文件](http://blog.csdn.net/aitangyong/article/details/23592789)


## 通过继承来分解复杂逻辑

笔者对继承的认识，有以下几个过程：

1. 我们以前更加习惯将复杂代码，分在不同的类中，类直接彼此依赖。
2. 一些类的共同代码抽取出来，形成父类

这种逻辑方式，在应对controller-service-dao之类的代码时是够用的，因为逻辑的入口controller通常是由多个url触发的。一个项目虽然有多个controller、service、dao，但可以划分为一系列controller-service-dao主线，交叉的不多。多个主线偶尔有一两个流程交叉，复用一些公共代码。**此时呢，父类自下而上产生，更多的是承载一些公共函数，这些公共函数通常无法反应该系列类的作用。**

而对于框架说，其主线非常明确，往往只有一个入口，一个入口操作对象对外提供服务。此时呢，通常有一个InterfaceA，然后是AbstractA，继而BaseA/DefaultA等，向下开枝散叶（自上而下）。此时呢，通常上级类更容易反映该系列类的作用，制定业务流程，而下级类则往往是某个流程步骤的具体实现了。

还有一种父类，参见netty-codec-http2，父类HttpRequestDecoder处理了http协议所有可能的场景，子类只需为父类某些参数设置特定值即可。构造子类时，不需要传入父类构造函数一样多的参数。

不同的继承形式，就好比不同的家庭。有的是富二代，父类把所有活儿都干完了。有的父类则是下个指示，交给各个子女去执行。甚至于还有，子类忙的要死，将公共的活儿匀给父类干。

## 重新理解工厂模式

A factory class decouples the client and implementing class.

那么当一个类有多层级继承关系时，就有必要为顶层接口/类准备一个工厂了。

## 重新来看观察者模式

对于我这个入门程序猿来说，尽管知道观察者模式什么意思，但在实际应用用还是很少用，因为这个违背了了我的直觉。


	Objector{
		List<Listener> listeners;
		function notify(){
			for(){
				...
			}
		}
		business1(){	// 我的直觉
			business code
			listener.listen();
		}
		business2(){	// 实际
			business code
			notify();
		}
	}
	
所谓的直觉是，类与类既然相互依赖，那么它们的调用就应该是直接的，就像business1那样，哪怕是business2稍微婉转了一下，都感觉别扭，反射弧就这么大，囧！

**所以说，设计模式分为创建型、结构型、行为型，我们在思考类的关系的时候，或许也应先从创建、结构、行为三个方向来着眼。**比如，使用工厂模式的时候，应该想着我要创建一个类，而不是想着我使用了工厂模式。

## 框架方和使用框架方的边界

表现形式

1. 注解，这就要求有一个组件将分散在业务各个位置的注解相关数据重新组织起来
2. 接口

驱动

1. 框架本身有线程启动，start 框架内线程有以下几种方式

	1. 框架线程启动代码注册到web容器中，web容器启动时驱动
	2. 显式初始化框架提供的操作对象
	3. 框架的操作对象在执行第一个任务时启动线程
2. 通过业务调用驱动


## 框架和业务的几点不同

1. 框架应该尽量减少依赖。除了上文提到的spring的版本兼容问题，一些框架功能的实现还依赖zk、mq等，这是要尽量减少的。



## 其它

* IdentityHashMap，This class implements the Map interface with a hash table, using reference-equality in place of object-equality when comparing keys (and values).  In other words, in an IdentityHashMap, two keys k1 and k2 are considered equal if and only if
 (k1==k2)