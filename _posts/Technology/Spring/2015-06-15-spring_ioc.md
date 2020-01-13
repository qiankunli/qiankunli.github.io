---

layout: post
title: 回头看Spring IOC
category: 技术
tags: Spring
keywords: JAVA Spring

---

## 前言 ##

* TOC
{:toc}

wiki[Inversion of control](https://en.wikipedia.org/wiki/Inversion_of_control) In software engineering, inversion of control (IoC) is a programming principle. IoC inverts the flow of control as compared to traditional control flow. In IoC, custom-written portions of a computer program receive the flow of control from a generic framework. A software architecture with this design inverts control as compared to traditional procedural programming: in traditional programming, the custom code that expresses the purpose of the program calls into reusable libraries to take care of generic tasks, but with inversion of control, it is the framework that calls into the custom, or task-specific, code. traditional control flow 是从开始到结束都是自己“写代码”，IoC 中control flow的发起是由一个framework 触发的。类只是干自己的活儿——“填代码”，然后ioc在需要的时候调用。


![](/public/upload/spring/spring_ioc.png)

## 控制反转 带来的改变：“解耦”

假设有两个类A和B

    class A{
        private B b;
        void fun(){
            创建b对象
            b.fun2()
        }
        public void setB(B b){
            this.b = b;
        }
    }

    class B{
        void fun2(){}
    }
    
在A的fun方法中，为调用b的fun2方法，需要先创建b对象，如果b对象的创建过程比较复杂（比如B还依赖其它类，那么还要先创建其他类），`a.fun`方法将非常臃肿，并且大部分代码都有点“不务正业”（都花在创建b对象上了），事实上，`a.fun`方法只是想单纯的运行`b.fun2()`。

按照书中的说法：许多非凡的应用都是由两个或多个类通过彼此的合作来实现业务逻辑的，这使得每个对象都需要与其合作的对象的引用（有时候这个获取过程会占用多行代码）。如果这个获取过程要靠自身实现，将导致代码高度耦合并且难以测试。

控制反转后，A类中B类引用的获得，不再是new（很多时候，new是不够的，需要多个操作才能获取一个可用的B实例），而是“别人”调用A类的set方法。**如果，把面向对象编程中需要执行的诸如新建对象、为对象引用赋值等操作交由容器统一完成，这样一来，这些散落在不同代码中的功能相同的部分(也就是上述代码中“创建b对象”和"setB"部分)就集中成为容器的一部分，也就是成为面向对象系统的基础设施的一部分。**（话说，如果大部分框架也使用spring实现该多好啊，至少容易看懂）

IOC设计模式的两个重要支持：

1. **对象间依赖关系的建立和应用系统的运行状态没有很强的关联性**，因此对象的依赖关系可以在启动时建立好，ioc容器（负责建立对象的依赖关系）不会对应用系统有很强的侵入性。
2. 面向对象系统中，除了一部分对象是数据对象外，其他很大一部分是用来处理数据的，这些对象并不经常发生变化，在系统中以单件的形式起作用就可以满足应用的需求。


## ioc的实现

什么是ioc容器？

BeanFactory是最简单的ioc容器，看了BeanFactory接口方法，也许会更直观(主要包括获取bean、查询bean的特性等方法)。

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


当然，ioc在实现上述功能之前，必须要先初始化，从某个源（比如xml配置文件）中加载bean信息。ioc容器对针对bean的不同（比如bean是否是单例），对bean的实例化有不同的处理，下面排除各种特例，描述下最常见的bean实例化过程。

1. ioc初始化

        BeanFactory bf = new XmlBeanFactory(new ClassPathResource("beans.xml"));
        
    - 验证并加载xml文件
    - 依次读取`<bean></bean>`（或扫描所有java类中的注解信息），并将其信息转化为BeanDefinition类（将bean信息由文本存储方式转换为内存存储（以java类的形式存在））


2. 执行`bf.getBean("beanid")`得到bean的实例

    - 根据beanid获取到相应的BeanDefinition
    - 根据BeanDefinition创建bean实例（此时还未注入属性）
    - 属性（包括其依赖对象）注入

    ioc在实例化bean的过程中，还夹了不少“私货”，也称之为装配wire：

    - 在属性或依赖注入逻辑的前后留有处理函数（或回调函数）
    - 如果bean实现了一些接口，ioc将其注入该接口指定的属性

bean在不同阶段的表现形式

||表现形式|与jvm类比|
|---|---|---|
|配置文件|`<bean class=""></bean>`|java代码|
|ioc初始化|BeanDefinition|class二进制 ==> 内存中的Class对象|
|getBean|Object|堆中的对象实例|
    

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

AbstractRefreshableApplicationContext 虽然实现了BeanFactory接口，但其实是**组合了一个beanFactory**，这是讨论BeanFactory 和 ApplicationContext 差异时最核心的一点。

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

可以说，BeanFactory 基本实现了一个bean container 需要的所有功能，但其一些特性要programmatic，ApplicationContext在简单容器BeanFactory的基础上，增加了许多面向框架的特性。《Spring技术内幕》中参照XmlBeanFactory的实现，以编程方式使用DefaultListableBeanFactory，从中我们可以看到Ioc容器使用的一些基本过程。

	ClassPathResource res = new ClassPathResource("beans.xml");
	DefaultListableBeanFactory factory = new DefaultListableBeanFactory();
	XmlBeanDefinitionReader reader = new XmlBeanDefinitionReader(factory);
	reader.loadBeanDefinitions(res);
	
**简单来说，`reader.loadBeanDefinitions(res);`将类信息从`beans.xml` 读取到DefaultListableBeanFactory及其父类的各种map中。然后`factory.getBean`时即可做出对应的反应。**

## 细说BeanDefinition

    public abstract class AbstractBeanDefinition extends BeanMetadataAttributeAccessor
		implements BeanDefinition, Cloneable {
        private volatile Object beanClass;  
    	private String scope = SCOPE_DEFAULT;  
    	private boolean singleton = true;  
    	private boolean prototype = false;  
    	private boolean abstractFlag = false;  
    	private boolean lazyInit = false;  
    	private int autowireMode = AUTOWIRE_NO;  
    	private int dependencyCheck = DEPENDENCY_CHECK_NONE;  
    	private String[] dependsOn;  
    	private boolean autowireCandidate = true;  
    	private boolean primary = false;  
    	private final Map<String, AutowireCandidateQualifier> qualifiers =
    			new LinkedHashMap<String, AutowireCandidateQualifier>(0);  
    	private boolean nonPublicAccessAllowed = true;  
    	private boolean lenientConstructorResolution = true;  
    	private ConstructorArgumentValues constructorArgumentValues;  
    	private MutablePropertyValues propertyValues;  
    	private MethodOverrides methodOverrides = new MethodOverrides();  
    	private String factoryBeanName;  
    	private String factoryMethodName;  
    	private String initMethodName;  
    	private String destroyMethodName;  
    	private boolean enforceInitMethod = true;  
    	private boolean enforceDestroyMethod = true;
    	private boolean synthetic = false;
    	private int role = BeanDefinition.ROLE_APPLICATION;
    	private String description;
    	private Resource resource;
    }
	
首先要将配置文件描述的bean信息加载到内存中，再根据这些信息构建bean实例，这些信息在内存中的存在形式便是BeanDefinition。spring ioc的基本功能可由以下过程实现：

1. 将BeanDefinition（以配置文件，注解形式存在）加载到内存
2. 根据BeanDefinition创建并管理bean实例以及它们之间的依赖

创建bean实例的几种方式：

1. class.newInstance，然后为其各种属性赋值
2. FactoryBean.getBean
3. Proxy.newProxyInstance
4. BeanDefinition

BeanDefinition的几种来源

1. xml解析，对应BeanDefinitionParser

	* `<bean></bean>`直接解析
	* `<tx:annotation-driven transaction-manager="txManager" />` 系统解析到该配置后，会根据配置，手动register若干个BeanDefinition

2. 注解扫描


## Bean的管理Environment

Interface representing the environment in which the current application is running.Models two key aspects of the application environment: profiles and properties. Environment代表着程序的运行环境，主要包含了两种信息，一种是profiles，用来描述哪些bean definitions 是可用的；一种是properties，用来描述系统的配置，其来源可能是配置文件、jvm属性文件、操作系统环境变量等。

并不是所有的Bean 都会被纳入到ioc管理，A profile is a named, **logical group of bean definitions** to be registered with the container only if the given profile is active. Beans may be assigned to a profile whether defined in XML or via annotations; see the spring-beans 3.1 schema or the  @Profile annotation for syntax details.

Properties play an important role in almost all applications,and may originate from a variety of sources: properties files, JVM system properties, system environment variables, JNDI, servlet context parameters, ad-hoc Properties objects,Maps, and so on. The role of the environment object with relation to properties is to provide the user with a convenient service interface for configuring property sources and resolving properties from them.

![](/public/upload/spring/spring_env.png)