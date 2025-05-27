---

layout: post
title: 编写java框架的几大利器
category: 技术
tags: Java
keywords: JAVA spi xsd

---

## 前言

笔者最近学习几个框架，其共同特征用maven 项目结构来描述的话就是：

```
xxx
    xxx-core/xxx-common  // 核心抽象与实现
    xxx-spring		     // 负责与spring整合，隐藏掉初始化，加一些注解等操作
```
我们说面向过程，画流程图，流程图中从start到end的代码我们都看的到。但到java中，大部分时候，我们在写回调函数/生命周期函数。即便是我们用代码描述了开始与结束，但其中注解啊什么的，又为我们夹带了不少私货。经常有人说，大部分程序猿拿着面向对象的语言写面向过程的代码（常写controller-service-dao的程序猿对这点感受尤其明显）。那么理解问题、理解框架、理解框架与业务的边界，有利于我们从面向对象的角度去思考问题。

框架一般有以下几个tips：

1. 如何感知用户配置。框架本身的运行需要一些策略和参数
2. 找出共性和个性的边界。比如web的filter、listener，spring的BeanPostProcessor等
3. 框架本身的复杂性如何解决？比如分层、设计模式
4. 对现有框架的扩展
5. 基于现有框架实现新框架。比如基于spring 实现一个自定义注解


一个框架，尤其是一个复杂框架，往往意味着多个复杂抽象以及抽象间复杂的依赖关系。尽量从中找一些共同点或工具，以减少阅读源码时的思维负担。

## spi

[剖析 SPI 在 Spring 中的应用](https://mp.weixin.qq.com/s/6YCCsjWwBMk9pX4LCvlEsA)SPI（Service Provider Interface），是Java内置的一种服务提供发现机制，**可以为接口寻找服务实现**。SPI机制将服务的具体实现转移到了程序外，为框架的扩展和解耦提供了极大的便利。

[阅读 Flink 源码前必知必会 - SPI 和 ClassLoader](https://mp.weixin.qq.com/s/PtmlneRo6AG4Fyb8y-Bvrw)在框架设计中，要遵循的原则是对扩展开放，对修改关闭，保证框架实现对于使用者来说是黑盒。因为框架不可能做好所有的事情，只能把共性的部分抽离出来进行流程化，然后留下一些扩展点让使用者去实现，这样不同的扩展就不用修改源代码或者对框架进行定制。

[SPI 加载慢的解决方法](https://mp.weixin.qq.com/s/CTFcwer2htssKszjhnOXtQ) 好文章

### 概念
[JDK/Dubbo/Spring 三种 SPI 机制，谁更好？](https://mp.weixin.qq.com/s/6SU1BPvNTCv_fhnMx3GhLw) SPI 的本质是将接口实现类的全限定名配置在文件中，并由服务加载器读取配置文件，加载实现类。PS: **本质是类加载的一种应用**。

[java中的SPI机制](http://www.cnblogs.com/javaee6/p/3714719.html)

java 平台对spi的支持可以参见java.util.ServiceLoader.A simple service-provider loading facility类。比如对一个maven项目结构

```
src
    main
        java
            org.lqk.spi
                IUserService
                UserService
        resources
            META-INF
                services
                    org.lqk.spi.IUserService // 内容是org.lqk.spi. UserService
```					

那么在代码中
```java
public class ServiceBootstrap {
    public static void main(String[] args) {
        // 一般 用于加载不属于当前 jar 的 框架库文件内的 类
        ServiceLoader<IUserService> serviceLoader = ServiceLoader.load(IUserService.class);
        serviceLoader.forEach(IUserService::sayHello);
    }
}
```

`ServiceLoader.load(IUserService.class)`即可得到IUserService实例。

```java
public static <S> ServiceLoader<S> load(Class<S> service) {
	// 获取当前线程的上下文类加载器。ContextClassLoader 是每个线程绑定的
	ClassLoader cl = Thread.currentThread().getContextClassLoader();
	return ServiceLoader.load(service, cl);
}
```
Thread.currentThread().getContextClassLoader();  使用这个获取的类加载器是 AppClassLoader，会去加载 classpath 的类。 ServiceLoader 核心的逻辑就在nextService方法里

```java
private S nextService() {
	if (!hasNextService())
		throw new NoSuchElementException();
	String cn = nextName;
	nextName = null;
	Class<?> c = null;
	try {
		// 加载这个类
		c = Class.forName(cn, false, loader);
	} catch (ClassNotFoundException x) {
		fail(service,
				"Provider " + cn + " not found");
	}
	if (!service.isAssignableFrom(c)) {
		fail(service,
				"Provider " + cn  + " not a subtype");
	}
	try {
		// 初始化这个类
		S p = service.cast(c.newInstance());
		providers.put(cn, p);
		return p;
	} catch (Throwable x) {
		fail(service,
				"Provider " + cn + " could not be instantiated",
				x);
	}
	throw new Error();          // This cannot happen
}
private boolean hasNextService() {
	if (nextName != null) {
		return true;
	}
	if (configs == null) {
		try {
			// 寻找 META-INF/services/类
			String fullName = PREFIX + service.getName();
			if (loader == null)
				configs = ClassLoader.getSystemResources(fullName);
			else
				configs = loader.getResources(fullName);
		} catch (IOException x) {
			fail(service, "Error locating configuration files", x);
		}
	}
	while ((pending == null) || !pending.hasNext()) {
		if (!configs.hasMoreElements()) {
			return false;
		}
		// 解析这个类文件的所有内容
		pending = parse(service, configs.nextElement());
	}
	nextName = pending.next();
	return true;
}
```

JDK SPI 在查找扩展实现类的过程中，需要遍历 SPI 配置文件中定义的所有实现类（寻找 META-INF/services/类，解析类的内容，构造 Class），该过程中会将这些实现类全部实例化。PS: 本质上是 扩展了classloader的实现，**SPI 是classloader 的一种应用**。

缺点：JDK SPI 在查找扩展实现类的过程中，需要遍历 SPI 配置文件中定义的所有实现类，该过程中会将这些实现类全部实例化。如果 SPI 配置文件中定义了多个实现类，而我们只需要使用其中一个实现类时，就会生成不必要的对象。

### 与api 对比

[Java SPI思想梳理](https://zhuanlan.zhihu.com/p/28909673)
spi 是与 api 相对应的一个词，代码上会有一个接口类与其对应

||api|spi|
|---|---|---|
|概念上|概念上更接近实现方。实现了一个服务，然后提供接口 给外界调用|概念上更依赖调用方。比如java 的jdbc 规范，是先定义接口，然后交给mysql、oracle 等厂商实现|
|interface 在|实现方所在的包中|调用方所在的包中|
|interface 和 implement|在一个包中| implement位于独立的包中（也可认为在提供方中）|
|如何对外提供服务，证明自己用处在哪里|提供接口 给外界调用|java spi的具体约定为: 当服务的提供者，提供了服务接口的一种实现之后，在jar包的META-INF/services/目录里同时创建一个以服务接口命名的文件。文件内容就是实现该服务接口的具体实现类|

### 与 ioc 对比

类似的ioc工具有[ Plexus，Spring之外的IoC容器](http://blog.csdn.net/huxin1/article/details/6020814)， [google/guice](https://github.com/google/guice)

使用它们而不是spring ioc有一个好处：我们所写的框架，经常应用于很多使用spring的项目中，不使用spring ioc，可以不与业务使用的spring版本冲突。

贴一段guice的简介：Put simply, Guice alleviates the need for factories and the use of new in your Java code. Think of Guice's @Inject as **the new new（一个新的new）.** You will still need to write factories in some cases, but your code will not depend directly on them. Your code will be easier to change, unit test and reuse in other contexts.


1. 面向的对象的设计里，我们一般推荐模块之间基于接口编程
2. ioc 侧重于将一个类的构造过程省掉，调用方只需关心类的使用。因为接口编程的关系，spring **顺带支持**通过scan 等方式 为一个接口 找到（spring 容器中）对应的实现类
3. spi 强调一个接口有多种实现，不想在代码中写死具体使用哪种实现。

## xsd

[ spring框架的XML扩展特性：让spring加载和解析你自定义的XML文件](http://blog.csdn.net/aitangyong/article/details/23592789)


## 通过继承来分解复杂逻辑

笔者对继承的认识，有以下几个过程：

1. 我们以前更加习惯将复杂代码，分在不同的类中，类直接彼此依赖。
2. 一些类的共同代码抽取出来，形成父类

这种逻辑方式，在应对controller-service-dao之类的代码时是够用的，因为逻辑的入口controller通常是由多个url触发的。一个项目虽然有多个controller、service、dao，但可以划分为一系列controller-service-dao主线，交叉的不多。多个主线偶尔有一两个流程交叉，复用一些公共代码。**此时呢，父类自下而上产生，更多的是承载一些公共函数，这些公共函数通常无法反应该系列类的作用。**

而对于框架说，其主线非常明确，往往只有一个入口，一个入口操作对象对外提供服务。此时呢，通常有一个InterfaceA，然后是AbstractA，继而BaseA/DefaultA等，向下开枝散叶（自上而下）。此时呢，通常上级类更容易反映该系列类的作用，制定业务流程，而下级类则往往是某个流程步骤的具体实现了。

还有一种父类，参见netty-codec-http2，父类HttpRequestDecoder处理了http协议所有可能的场景，子类只需为父类某些参数设置特定值即可。构造子类时，不需要传入父类构造函数一样多的参数。

不同的继承形式，就好比不同的家庭。有的是富二代，父类把所有活儿都干完了。有的父类则是下个指示，交给各个子女去执行。随着接入业务逐渐增多继承关系会越来越“胖”或者越来越“高”，当一个新的扩展品类来的时候，我们都需要决策一件事情，新来的品类是继承最根部base类（变胖），还是找一个实现逻辑最相近的类来继承（变高）。变胖带来的后果就是复用性不好，一样的实现逻辑可能会出现在多个流程中；变高带来的后果是父类实现的修改有可能会影响子类的业务。

## 重新理解工厂模式

工厂模式可以用 DRY（Don’t Repeate Yourself）原则来理解，也就是说尽量避免重复的代码，简单地认为它就是“对 new 的封装”。想象一下，如果程序里到处都是“硬编码”的 new，一旦设计发生变动，比如说把“new 苹果”改成“new 梨子”，你就需要把代码里所有出现 new 的地方都改一遍，不仅麻烦，而且很容易遗漏，甚至是出错。如果把 new 用工厂封装起来，就形成了一个“中间层”，隔离了客户代码和创建对象，两边只能通过工厂交互，彼此不知情，也就实现了解耦，由之前的强联系转变成了弱联系。所以，你就可以在工厂模式里拥有对象的“生杀大权”，随意控制生产的方式、生产的时机、生产的内容。重点是“如何创建对象、创建出什么样的对象”，用函数或者类会比单纯用 new 更灵活。

A factory class decouples the client and implementing class. 工厂模式解决的是bean的生产问题，简单工厂模式根据入参生产不同的bean，普通工厂模式针对每个bean都构建一个工厂，此两者各有优劣，看需要。如果每个bean主要的功能都在方法中，不涉及类变量的使用，可以利用spring容器生成的bean（bean作为factory的成员由spring注入）。PS：我们明确地计划不同条件下创建不同实例时。

The factory pattern is a design pattern that is used to encapsulate complex logic in functions that creates the wanted instance, **without the caller knowing anything about the implementation details**.

那么当一个类有多层级继承关系时，就有必要为顶层接口/类准备一个工厂了。

## 重新来看观察者模式

观察者模式定义对象间的一种一对多的依赖关系，当一个对象的状态发生改变时，所有依赖于它的对象都得到通知并被自动更新。

对于我这个入门程序猿来说，尽管知道观察者模式什么意思，但在实际应用用还是很少用，因为这个违背了了我的直觉。

```
Objector{
	# 关键代码：在抽象类里有一个 ArrayList 存放观察者们。
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
```
	
所谓的直觉是，类与类既然相互依赖，那么它们的调用就应该是直接的，就像business1那样，哪怕是business2稍微婉转了一下，都感觉别扭，反射弧就这么大，囧！

**所以说，设计模式分为创建型、结构型、行为型，我们在思考类的关系的时候，或许也应先从创建、结构、行为三个方向来着眼。**

观察者模式，也描述了对象与对象之间的依赖关系。

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

	1. 使用spring 容器时， 挂载一些钩子函数、或实现特定接口，即可将代码挂载容器的启动过程中。


## 框架和业务的几点不同

1. 框架应该尽量减少依赖。除了上文提到的spring的版本兼容问题，一些框架功能的实现还依赖zk、mq等，这是要尽量减少的。
2. 框架应该尽量减少对入参的更改。因为调用方可能复用入参。

## 框架和框架box 类

学习一个框架，要分清楚框架的核心和边缘，比如《netty in action》中提到:**Netty provides an extensive set of predefined handlers that you can use out of the box.**including handlers for protocols such as HTTP and SSL/TLS. Internally ,Channel-Handlers use events and futures themselves,making them consumers of the same abstractions your applications will employ.

## 依赖关系的管理

一个稍微复杂的框架，必然伴随几个抽象以及抽象间的依赖关系，那么依赖的关系的管理，可以选择spring（像大多数j2ee项目那样），也可以硬编码。这就是我们看到的，每个抽象对象有一套自己的继承体系，然后抽象对象子类之间又彼此复杂的交织。比如Netty的eventloop、unsafe和pipeline，channel作为最外部操作对象，聚合这三者，根据聚合合的子类的不同，Channel也有多个子类来体现。

依赖关系管理，也可以考虑使用[google/guice](https://github.com/google/guice)


## 其它

们使用框架，主要是复用框架的能力（function）。而对于框架而言，他最主要的职责是帮用户处理“数据”，对于用户数据，我们在框架中通常叫它们Context（上下文）。我们可以将框架中的Context进一步分为Procedure Context（过程上下文）和Global Context（全局上下文）：
1. 所谓Procedure Context，是指用户每次调用框架所需要携带的数据。这个Context一般被设计为函数参数，在框架内部传递，当调用链结束，即被销毁。简单理解就是Context per request。例如，web容器框架中的每一个http请求都会有一个HttpServletRequest，就属于Procedure Context。
2. 所谓Global Context，一般存储的是用户对框架的配置信息，它是全局共享的，在框架的整个生命周期都有效。比如，web容器中的ServletContext，一个容器只有一个是Global的。