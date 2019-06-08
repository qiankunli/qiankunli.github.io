---

layout: post
title: Springboot 启动过程分析
category: 技术
tags: Spring
keywords: springboot

---

## 简介（持续更新）

* TOC
{:toc}

![](/public/upload/spring/springboot.png)

几个问题

1. 如何自动加载配置，将依赖bean 注入到spring ioc
2. 如何自动规范依赖jar？继承 spring-boot-starter-parent
3. tomcat 是如何内置的

![](/public/upload/spring/spring_boot_class_diagram.png)

建议先阅读下 [回头看Spring IOC](http://qiankunli.github.io/2015/06/15/spring_ioc.html) 对IOC 和 ApplicationContext 等概念有所了解。

## 启动过程

![](/public/upload/spring/spring_application_run.png)

	public SpringApplication(ResourceLoader resourceLoader, Class<?>... primarySources) {
		this.resourceLoader = resourceLoader;
		Assert.notNull(primarySources, "PrimarySources must not be null");
		this.primarySources = new LinkedHashSet<>(Arrays.asList(primarySources));
		this.webApplicationType = WebApplicationType.deduceFromClasspath();
		setInitializers((Collection) getSpringFactoriesInstances(
				ApplicationContextInitializer.class));
		setListeners((Collection) getSpringFactoriesInstances(ApplicationListener.class));
		this.mainApplicationClass = deduceMainApplicationClass();
	}

getSpringFactoriesInstances 会从`META-INF/spring.factories`读取key为org.springframework.context.ApplicationContextInitializer的value 并创建ApplicationContextInitializer 实例，ApplicationListener类似。

SpringApplication.run 创建并刷新ApplicationContext，算是开始进入正题了。

    public ConfigurableApplicationContext run(String... args) {
        ConfigurableApplicationContext context = null;
        ...
        SpringApplicationRunListeners listeners = getRunListeners(args);
        listeners.starting();

        ApplicationArguments applicationArguments = new DefaultApplicationArguments(
                args);
        ConfigurableEnvironment environment = prepareEnvironment(listeners,
                applicationArguments);
        ...
        // 创建AnnotationConfigServletWebServerApplicationContext
        context = createApplicationContext();
        ...
        prepareContext(context,environment,listeners,applicationArguments,printedBanner);
        refreshContext(context);
        listeners.started(context);
        callRunners(context, applicationArguments);
        listeners.running(context);
        return context;
    }

1. 所谓的 SpringApplicationRunListeners 就是在SpringApplication.run 方法执行的不同阶段去执行一些操作， SpringApplicationRunListener 也可在`META-INF/spring.factories` 配置
2.  Environment代表着程序的运行环境，主要包含了两种信息，一种是profiles，用来描述哪些bean definitions 是可用的；一种是properties，用来描述系统的配置，其来源可能是配置文件、jvm属性文件、操作系统环境变量等。
3. AnnotationConfigServletWebServerApplicationContext 的默认构造方法中初始化了两个成员变量，类型分别为AnnotatedBeanDefinitionReader 和  ClassPathBeanDefinitionScanner 用来加载Bean 定义。

其中在 prepareContext 中

	private void prepareConte(ConfigurableApplicationContext context,ConfigurableEnvironment environment,SpringApplicationRunListeners listeners,ApplicationArguments applicationArguments, Banner printedBanner) {
        context.setEnvironment(environment);
        postProcessApplicationContext(context);
        applyInitializers(context);
        listeners.contextPrepared(context);
        // Add boot specific singleton beans
        ConfigurableListableBeanFactory beanFactory = context.getBeanFactory();
        beanFactory.registerSingleton("springApplicationArguments", applicationArguments);
        ...
        // Load the sources
        Set<Object> sources = getAllSources();
        load(context, sources.toArray(new Object[0]));
        listeners.contextLoaded(context);
    }

applyInitializers 会执行`initializer.initialize(context)`

## starter 依赖扩展ApplicationContext的入口 ApplicationContextInitializer

## tomcat 是如何内置的——ServletWebServerApplicationContext

![](/public/upload/spring/spring_boot_web_server.png)