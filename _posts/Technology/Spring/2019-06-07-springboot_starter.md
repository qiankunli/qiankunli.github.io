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
2. 

其中在 prepareContext 中


	private void prepareContext(ConfigurableApplicationContext context,
			ConfigurableEnvironment environment, SpringApplicationRunListeners listeners,
			ApplicationArguments applicationArguments, Banner printedBanner) {
		context.setEnvironment(environment);
		postProcessApplicationContext(context);
		applyInitializers(context);
		listeners.contextPrepared(context);
		if (this.logStartupInfo) {
			logStartupInfo(context.getParent() == null);
			logStartupProfileInfo(context);
		}
		// Add boot specific singleton beans
		ConfigurableListableBeanFactory beanFactory = context.getBeanFactory();
		beanFactory.registerSingleton("springApplicationArguments", applicationArguments);
		if (printedBanner != null) {
			beanFactory.registerSingleton("springBootBanner", printedBanner);
		}
		if (beanFactory instanceof DefaultListableBeanFactory) {
			((DefaultListableBeanFactory) beanFactory)
					.setAllowBeanDefinitionOverriding(this.allowBeanDefinitionOverriding);
		}
		// Load the sources
		Set<Object> sources = getAllSources();
		Assert.notEmpty(sources, "Sources must not be empty");
		load(context, sources.toArray(new Object[0]));
		listeners.contextLoaded(context);
	}

applyInitializers