---

layout: post
title: spring cloud 初识
category: 技术
tags: Spring
keywords: spring cloud

---

## 前言（未完成）

官方英文 [Spring Cloud](http://cloud.spring.io/spring-cloud-static/Dalston.SR2/#_spring_cloud_config)

官方中文 [Spring Cloud](https://springcloud.cc/spring-cloud-dalston.html)

## springboot 

Spring Cloud是一个基于Spring Boot实现的云应用（cloud native）开发工具；Spring boot专注于快速、方便集成的单个个体，Spring Cloud基于 spring boot的、关注全局的服务治理框架。因此先简单说下 springboot

[第二篇:Spring boot与Spring cloud 是什么关系？](https://zhuanlan.zhihu.com/p/30211072)

一般的springmvc 项目通常有几个问题

1. 配置复杂
2. 依赖复杂
3. 依赖tomcat 等运行环境

插句题外话：讲真，可能是springmvc 习惯了，不觉得有多复杂。

springboot 解决方法

1. 通过自动配置AutoConfiguration，解决配置复杂问题。[What is Spring Boot Auto Configuration?](http://www.springboottutorial.com/spring-boot-auto-configuration) 如果 classpath 里面有 Spring MAC 的 JAR 包，但没有检查到 web-context.xml 配置，spring boot 会自动配置好 Dispatcher Servlet 
2. 通过starter和依赖管理解决依赖问题。[自定义SpringBoot Starter](http://objcoding.com/2018/02/02/Costom-SpringBoot-Starter/),SpringBoot 项目就是由一个一个 Starter 组成的,一个 Starter 代表该项目的 SpringBoot 起步依赖。自定义Starter 时 通常包含了 如何创建 starter 对应依赖所需的配置（这句话有一点待考证）。
3. 通过内嵌web容器，来解决部署运行问题。


[初学 Spring Boot，你需要了解的 7 样东西](https://juejin.im/post/5a50b189518825732334f713)

1. Spring MVC 提供了诸如 Dispatcher Servlet、ModelAndView 和 View Resolver 等，让编写松散耦合的 Web 应用变得很容易（说实话，用spring mvc 这么久，一下子还说不清楚springmvc 是什么）。对应的web.xml 和 web-context.xml 中也必须有这些资源的基本配置。


## 来源背景

Spring Cloud provides tools for developers to quickly build some of the common patterns in distributed systems (e.g. configuration management, service discovery, circuit breakers, intelligent routing, micro-proxy, control bus). Coordination of distributed systems leads to boiler plate patterns, and using Spring Cloud developers can quickly stand up services and applications that implement those patterns. 这句话有几个要点

1. Coordination of distributed systems 需要很多 common patterns
2. 大量的 common pattern系统（比如配置中心、服务发现）用到了boiler plate pattern。 所谓锅炉板模式，就是比如对方提供一个http rpc 服务。你可以手写http 代码来调用，也可以将http 调用代码封装为一个工具类或jar，还可以用一个注解+参数描述http 调用。因为 对不同业务 http rpc 不同的地方就是几个配置，因此最后一种方法即称为boiler plate pattern（锅炉板模式）。

spring cloud 实现了这些 common patterns，并使用锅炉板 模式简化了它们的使用，因此可以快速用来构建 distributed systems

spring cloud 提供的组件很多，核心 是服务治理，以至于[从架构演进的角度聊聊Spring Cloud都做了些什么？](http://www.ityouknow.com/springcloud/2017/11/02/framework-and-springcloud.html) 直接说“Spring Cloud作为一套微服务治理的框架，几乎考虑到了微服务治理的方方面面”。 服务之间的直接依赖转化为服务对服务中心的依赖

## spring cloud config

[springcloud(六)：配置中心git示例](http://www.ityouknow.com/springcloud/2017/05/22/springcloud-config-git.html)

git 库中 针对每个环境 新建一个文件，比如

	config-dev.properties
	config-test.properties
	config-pro.properties
	
创建配置中心 server 端，将配置文件服务化。假设对外端口是8001，即可通过`http://localhost:8001/config/dev` 来获取`config-dev.properties` 的内容。

客户端即可 通过 @Value 等方式使用 配置文件中的配置。