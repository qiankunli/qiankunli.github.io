---

layout: post
title: 学习Spring
category: 技术
tags: Spring
keywords: springboot

---

## 简介

* TOC
{:toc}

[使用Spring特性优雅书写业务代码](https://mp.weixin.qq.com/s/94oe5c_7ouE1GbyiPfNg5g)

[程序员必备的思维能力：抽象思维](https://mp.weixin.qq.com/s/cJ0odiYcphhNBoAVjqpCZQ)我们知道Spring的核心功能是Bean容器，那么在看Spring源码的时候，我们可以着重去看它是如何进行Bean管理的？它使用的核心抽象是什么？不难发现，Spring是使用了BeanDefinition、BeanFactory、BeanDefinitionRegistry、BeanDefinitionReader等核心抽象**实现了Bean的定义、获取和创建**。抓住了这些核心抽象，我们就抓住了Spring设计主脉。除此之外，我们还可以进一步深入思考，它为什么要这么抽象？这样抽象的好处是什么？以及它是如何支持XML和Annotation（注解）这两种关于Bean的定义的。



## 编程模型

面向对象编程

1. 契约接口，Aware,BeanPostProcessor...。 ioc 约定了会对这些接口类进行处理
2. 设计模式，观察者模式（ApplicationEvent）、组合模式（Composite*）、模板模式（JdbcTemplate/RestTemplate）...
3. 对象继承，Abstract*类

面向切面编程

1. 动态代理，JdkDynamicAopProxy
2. 字节码提升，ASM、CGLib、AspectJ...

面向元编程

1. 注解，模式注解（@Component/@Service/@Respository...）
2. 配置，Environment抽象、PropertySources、BeanDefinition...
3. 泛型，GenericTypeResolver、ResolvableType...

因为java 注解不允许继承，所以会在“子注解”上进行源标注。

函数驱动

1. 函数接口，ApplicationEventPublisher
2. Reactive,Spring WebFlux

模块驱动

1. Maven Artifacts
2. OSGI Bundies
3. Java 9 Automatic Modules
4. Spring @Enable*

## Spring是什么？

![](/public/upload/spring/ioc_overview.png)

### 内在本质——component container

[History of Spring Framework and Spring Boot](https://www.quickprogrammingtips.com/spring-boot/history-of-spring-framework-and-spring-boot.html)It currently consists of a large number of modules providing a range of services. These include a component container, aspect oriented programming support for building cross cutting concerns, security framework, data access framework, web application framework and support classes for testing components. **All the components of the spring framework are glued together by the dependency injection architecture pattern**. Dependency injection(also known as inversion of control) makes it easy to design and test loosely coupled software components. 依赖注入的关键就是有一个component container//bean container/IOC container，它持有所有对象的实例，负责所有对象的创建和销毁问题，在创建对象时可以夹一点自己的私货。

||tomcat|spring|
|---|---|---|
|组成|包含Connector和Container|包含ioc和其它特性|
|容器部分|servlet容器|bean容器|
|初始化|根据web.xml文件初始化servlet|根据xml初始化bean|
|Servlet/Bean扩展点|基于同一的Servlet接口定义|基于BeanPostProcessor等类来定义|
|依赖关系|Servlet之间无依赖关系|Bean之间可能有依赖关系|
|扩展||事件、外部资源/配置|

### 外在表现

1. 是一个应用平台，它不像hibernate等只是解决某一个领域的问题，它对企业应用资源（比如数据库、事务处理等）都提供了简化的、模板化的操作方式。类似于os简化了我们对计算机各种硬件资源的使用。
2. 简化了J2EE开发。用户使用POJO或者简单的javabean即可实现复杂的业务需求。POJO类有一些属性及其getter setter方法的类,基本没有业务逻辑，不具有任何特殊角色和不继承或不实现任何其它Java框架的类或接口。(model,dao,service,controller其实都是POJO类)一个项目仅依靠pojo类和几个配置文件来描述，用不着复杂的开发模型和编程模式。

    这种简化还体现在，spring对数据库（mysql或hbase等）、缓存（redis等）、队列（rabbitmq）、协作框架（Zookeeper等）和RPC框架（thrift等）都有着很好的支持。这些组件现在是构建一个大型系统的必备要素。
    
2017.7.27 更新

面向对象出来之后，一个项目的代码通常由一系列对象组成，而理解一个项目的难点变成了：如何理解对象之间复杂的依赖关系。读过netty源码的都知道，channel、pipeline、eventloop三个组件之间，复杂的依赖关系，简直不忍直视。比如A依赖B，B可以作为A的成员、方法参数等，而Spring统一成了一种：B作为A的成员。c、go之类，即便按照面向对象的思路来编程，因为没有类似spring的组件，业务本身的复杂性 + 对象之间的复杂的依赖关系，增加了理解的难度。

## 感受抽象的力量

[深入Spring配置内核，感受抽象的力量](https://mp.weixin.qq.com/s/gTSHekcN427jZ5H1LPfBFg) 值得细读、多次读。
假设现在要让你实现一个类似于Spring的配置框架，你会如何设计？所谓配置是一种通过调整参数，在不改动软件代码的情况下，改变系统行为的方式。所有的配置系统，都需要做三件事情：
1. 配置内容获取： 获取配置内容，是配置系统要做的第一件事情。配置内容（通常是以配置文件的形式）可以从classpath，文件系统或者网络字节流获取。
2. 配置内容解析： 拿到配置内容之后，需要对配置内容进行解析。因为配置可能存在多种不同的格式，比如常见的配置文件格式包括properties，yaml，JSON，XML等等。
3. 配置项赋值： 最后，就是要给需要给配置项赋值。在赋值的过程中，当一个配置项存在于多个配置定义中时，需要有优先级处理规则。
要想实现这样的功能，框架必须要具备一定的灵活性和扩展性。一定要设计得足够抽象，不能写死。比如关于文件格式你如果写死为properties，就没办法支持yaml和xml了。要让设计满足这样的灵活性，有三个核心抽象，你必须要了解，这三个抽象分别是Resource抽象，PropertySource抽象，以及MutablePropertySources抽象。
1. Resource
    ```
    public interface Resource {
        //核心抽象
        InputStream getInputStream() throws IOException;
        ...
    }
    ```
2. PropertySource，这个抽象需要磨平不同的配置内容格式，即不管你的配置文件是properties、yaml还是json，格式有所不同，但解析之后不应该存在差异。这里的抽象很关键，需要深入思考配置内容的本质是什么。配置的本质是给配置项赋值，不管其外在形式如何，其本质形式就是一组key-value pair。Spring对这一组key-value pair的抽象叫PropertySource，其核心方法就是通过配置名称找到配置内容。那么用HashMap数据结构存储配置内容是再适合不过的了，你可以认为MapPropertySource就是对Map的更有业务语义的封装。
    ```
    public abstract class PropertySource<T> {
        protected final String name;
        protected final T source;

        //配置的核心无外乎就是通过name，找到配置内容
        public abstract  Object getProperty(String name);

        ...
    }
    ```

![](/public/upload/spring/spring_configure.jpg)

抛开一些Loader、Resovler、BeanBinder、PropertySourceLocator辅助类，Spring配置的核心抽象就只有Resource、PropertySource和MutablePropertySources。这就是抽象的力量，一个好的抽象是对问题本质的深入洞悉（配置的本质就是KV Pair），是要能经受地起时间的锤炼而保持稳定。

## 其它

先有思想后有的支持思想的feature：常规的说法是：AOP的实现用到了动态代理技术。但更准确的说：**动态代理 是java 内嵌的对面向切面编程的支持**

![](/public/upload/spring/spring_features.png)



