---

layout: post
title: 回头看Spring IOC
category: 技术
tags: Spring
keywords: JAVA Spring

---

## 前言 

* TOC
{:toc}

![](/public/upload/spring/spring_ioc.png)

Spring 容器具象化一点就是 从xml、配置类、依赖jar 等处 通过 `BeanDefinitionRegistry.registerBeanDefinition` 向容器注入Bean信息，然后通过`BeanFactory.getBean` 应用在各个位置。 笔者以前习惯于关注 BeanFactory 初始化后的getBean 部分，忽视了其初始化过程的Bean信息加载部分。

![](/public/upload/spring/ioc_overview.png)

从[谈元编程与表达能力](https://mp.weixin.qq.com/s/SUV6vBaqwu19-xYzkG4SxA)中，笔者收获了对运行时的一个理解：当相应的行为在当前对象上没有被找到时，运行时会提供一个改变当前对象行为的入口。**从这个视角看，Spring 也可以认为是 java 的一个runtime，通过ApplicationContext 获取的bean 拥有 bean代码本身看不到的能力**。

## 什么是容器？

[Inversion of control](https://en.wikipedia.org/wiki/Inversion_of_control) In software engineering, inversion of control (IoC) is a programming principle. IoC inverts the flow of control as compared to traditional control flow. In IoC, custom-written portions of a computer program receive the flow of control from a generic framework. A software architecture with this design inverts control as compared to traditional procedural programming: in traditional programming, the custom code that expresses the purpose of the program calls into reusable libraries to take care of generic tasks, but with inversion of control, it is the framework that calls into the custom, or task-specific, code. traditional control flow 是从开始到结束都是自己“写代码”，IoC 中control flow的发起是由一个framework 触发的。类只是干自己的活儿——“填代码”，然后ioc在需要的时候调用。

IOC设计模式的两个重要支持：

1. **对象间依赖关系的建立和应用系统的运行状态没有很强的关联性**，因此对象的依赖关系可以在启动时建立好，ioc容器（负责建立对象的依赖关系）不会对应用系统有很强的侵入性。
2. 面向对象系统中，除了一部分对象是数据对象外，其他很大一部分是用来处理数据的，这些对象并不经常发生变化，在系统中以单件的形式起作用就可以满足应用的需求。

**控制反转 带来的改变：“解耦”**

## ioc的实现

什么是ioc容器？BeanFactory是最简单的ioc容器，看了BeanFactory接口方法，也许会更直观(主要包括获取bean、查询bean的特性等方法)。

    interface BeanFactory{
        FACTORY_BEAN_PREFIX
        object getBean(String)
        T getBean(String,Class<T>)
        T getBean(Class<T>)
        boolean containsBean(String)
        boolean isSingleton(String)
        boolean isPrototype(String)  
        boolean isTypeMatch(String,Class<T>)   
        class<?> getType(String)
        String[] getAliases    
    }

### BeanFactory

属于spring-beans包

The root interface for accessing a Spring bean container.This is the basic client view of a bean container;The point of this approach is that the BeanFactory is a **central registry** of application components, and **centralizes configuration** of application components (no more do individual objects need to read properties files,for example). 

Note that it is generally better to rely on Dependency Injection("push" configuration) to configure application objects through setters or constructors, rather than use any form of "pull" configuration like a BeanFactory lookup. Spring's Dependency Injection functionality is implemented using this BeanFactory interface and its subinterfaces. 

![](/public/upload/spring/bean_factory_class_diagram.png)

1. 作为一个BeanFactory，要能够getBean、createBean、autowireBean、根据各种Bean的信息检索list bean、支持父子关系，这些能力被分散在各个接口中
2. AbstractBeanFactory 负责实现BeanFactory，同时留了一些抽象方法交给子类实现
3. 如何对 BeanFactory 施加影响？BeanPostProcessor，**Factory hook** that allows for custom modification of new bean instances,e.g. checking for marker interfaces or wrapping them with proxies.


### ApplicationContext

![](/public/upload/spring/application_context_class_diagram.png)

属于spring-context包

AbstractRefreshableApplicationContext 虽然实现了BeanFactory接口，但其实是**组合了一个beanFactory**，这是讨论BeanFactory 和 ApplicationContext 差异时最核心的一点。 也正因此，ApplicationContext is a complete superset of the BeanFactory.

    public abstract class AbstractRefreshableApplicationContext extends AbstractApplicationContext {
        private DefaultListableBeanFactory beanFactory;
        public final ConfigurableListableBeanFactory getBeanFactory() {
            synchronized (this.beanFactoryMonitor) {
                if (this.beanFactory == null) {
                    throw new IllegalStateException("BeanFactory not initialized or already closed - " +
                            "call 'refresh' before accessing beans via the ApplicationContext");
                }
                return this.beanFactory;
            }
        }
    }


In contrast to a **plain BeanFactory**, an ApplicationContext is supposed to detect special beans(BeanFactoryPostProcessor/BeanPostProcessor/ApplicationListener）defined in its internal bean factory

    public abstract class AbstractApplicationContext{
        public void refresh(){
            ...
            prepareBeanFactory(beanFactory);
            ...
            registerBeanPostProcessors(beanFactory);
            ...
        }
        protected void prepareBeanFactory(ConfigurableListableBeanFactory beanFactory) {
            ...
            beanFactory.addBeanPostProcessor(new ApplicationContextAwareProcessor(this));
            ...
        }
    }

`beanFactory.addBeanPostProcessor(new ApplicationContextAwareProcessor(this));`ApplicationContextAwareProcessor是一个BeanPostProcessor，其作用就是当发现 一个类实现了ApplicationContextAware等接口时，为该类注入ApplicationContext 成员。

    public static void registerBeanPostProcessors(...){
        String[] postProcessorNames = beanFactory.getBeanNamesForType(BeanPostProcessor.class, true, false);
        ...
        for (String ppName : postProcessorNames) {
            BeanPostProcessor pp = beanFactory.getBean(ppName, BeanPostProcessor.class);
            priorityOrderedPostProcessors.add(pp);
        }
        ...
        registerBeanPostProcessors(beanFactory,priorityOrderedPostProcessors);
    }

### BeanFactory VS ApplicationContext

ApplicationContexts can autodetect BeanPostProcessor beans in their bean definitions and apply them to any beans subsequently created.Plain bean factories allow for **programmatic registration** of post-processors,applying to all beans created through this factory.

可以说，BeanFactory 基本实现了一个bean container 需要的所有功能，但其一些特性要通过programmatic来支持，ApplicationContext在简单容器BeanFactory的基础上，增加了许多面向框架的特性。《Spring技术内幕》中参照XmlBeanFactory的实现，以编程方式使用DefaultListableBeanFactory，从中我们可以看到Ioc容器使用的一些基本过程。

	ClassPathResource res = new ClassPathResource("beans.xml");
	DefaultListableBeanFactory factory = new DefaultListableBeanFactory();
	XmlBeanDefinitionReader reader = new XmlBeanDefinitionReader(factory);
	reader.loadBeanDefinitions(res);
	
**简单来说，`reader.loadBeanDefinitions(res);`将类信息从`beans.xml` 读取到DefaultListableBeanFactory及其父类的各种map中。然后`factory.getBean`时即可做出对应的反应。**

## 容器启动与停止

AbstractApplicationContext.refresh(): Load or refresh the persistent representation of the configuration such as xml. ApplicationContext 支持的很多特性都可以在个启动方法里找到迹象。

	public void refresh() throws BeansException, IllegalStateException {
		synchronized (this.startupShutdownMonitor) {
			// Prepare this context for refreshing.
			prepareRefresh();
			// Tell the subclass to refresh the internal bean factory.
			ConfigurableListableBeanFactory beanFactory = obtainFreshBeanFactory();
			// Prepare the bean factory for use in this context. 加入了Bean依赖以及非Bean依赖（比如Environment）
			prepareBeanFactory(beanFactory);
			try {
				// Allows post-processing of the bean factory in context subclasses.
				postProcessBeanFactory(beanFactory);
				// Invoke factory processors registered as beans in the context.
				invokeBeanFactoryPostProcessors(beanFactory);
				// Register bean processors that intercept bean creation.
				registerBeanPostProcessors(beanFactory);
				// Initialize message source for this context.
				initMessageSource();
				// Initialize event multicaster for this context.
				initApplicationEventMulticaster();
				// Initialize other special beans in specific context subclasses.
				onRefresh();
				// Check for listener beans and register them.
				registerListeners();
				// Instantiate all remaining (non-lazy-init) singletons.
				finishBeanFactoryInitialization(beanFactory);
				// Last step: publish corresponding event.
				finishRefresh();
			}catch (BeansException ex) {
				// Destroy already created singletons to avoid dangling resources.
				destroyBeans();
				// Reset 'active' flag.
				cancelRefresh(ex);
				// Propagate exception to caller.
				throw ex;
			}finally {
				// Reset common introspection caches in Spring's core, since we
				// might not ever need metadata for singleton beans anymore...
				resetCommonCaches();
			}
		}
	}

停止逻辑

	public void close() {
		synchronized (this.startupShutdownMonitor) {
			doClose();
			// If we registered a JVM shutdown hook, we don't need it anymore now:We've already explicitly closed the context.
			if (this.shutdownHook != null) {
				try {
					Runtime.getRuntime().removeShutdownHook(this.shutdownHook);
				}catch (IllegalStateException ex) {
					// ignore - VM is already shutting down
				}
			}
		}
	}

## ioc 中的对象

|来源|Spring Bean对象|生命周期管理|配置元信息<br>是否lazyload/autowiring等|使用场景|
|---|---|---|---|---|
|Spring BeanDefinition|是|是|有|依赖查找、依赖注入|
|单体对象|是|否|无|依赖查找、依赖注入|
|Resolvable Dependency|否|否|有|依赖注入|

### Bean工厂的养料——BeanDefinition

在BeanFactory 可以getBean之前，必须要先初始化，**从各种源（比如xml配置文件）中加载bean信息**。

![](/public/upload/spring/bean_definition.png)

bean在不同阶段的表现形式

||表现形式|与jvm类比|
|---|---|---|
|配置文件<br>@Configuration注释的class等|`<bean class=""></bean>`|java代码|
|ioc初始化|BeanDefinition|class二进制 ==> 内存中的Class对象|
|getBean|Object|堆中的对象实例|

jvm 基于Class 对象将对象实例化，想new 一个对象，得先有Class 对象，或者使用classLoader 加载，或者动态生成。spring 容器类似，想getBean 对象bean 实例， 就是先register 对应的BeanDefinition，任何来源的bean 通过`BeanDefinitionRegistry.registerBeanDefinition` 都可以纳入到IOC 的管理。

![](/public/upload/spring/bean_definition_xmind.png)

[Spring的Bean生命周期，11 张高清流程图及代码，深度解析](https://mp.weixin.qq.com/s/Rilo9hlkwM1OvfqUx_YJ9A)`BeanFactory.getBean("beanid")`得到bean的实例

![](/public/upload/spring/get_bean.png)

ioc在实例化bean的过程中，还夹了不少“私货”/“钩子”：

![](/public/upload/spring/create_bean.png)

### 内建对象Environment

Interface representing the environment in which the current application is running.Models two key aspects of the application environment: profiles and properties. Environment代表着程序的运行环境，主要包含了两种信息，一种是profiles，用来描述哪些bean definitions 是可用的；一种是properties，用来描述系统的配置，其来源可能是配置文件、jvm属性文件、操作系统环境变量等。

并不是所有的Bean 都会被纳入到ioc管理，A profile is a named, **logical group of bean definitions** to be registered with the container only if the given profile is active. Beans may be assigned to a profile whether defined in XML or via annotations; see the spring-beans 3.1 schema or the  @Profile annotation for syntax details.

Properties play an important role in almost all applications,and may originate from a variety of sources: properties files, JVM system properties, system environment variables, JNDI, servlet context parameters, ad-hoc Properties objects,Maps, and so on. The role of the environment object with relation to properties is to provide the user with a convenient service interface for configuring property sources and resolving properties from them.

![](/public/upload/spring/spring_env.png)