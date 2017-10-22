---

layout: post
title: log4j学习
category: 技术
tags: Java
keywords: log4j 

---

## 前言

首先明确几个日志框架的基本关系:[slf4j log4j logback关系详解和相关用法](http://www.cnblogs.com/Sinte-Beuve/p/5758971.html) 

文章基本要点：slf4j是一系列的日志接口（slf4j由二十多个类组成，大部分是interface），而log4j logback是具体实现了的日志框架。logback是直接实现了slf4j的接口，是不消耗内存和计算开销的。而log4j不是对slf4j的原生实现，所以slf4j api在调用log4j时需要一个适配层，比如slf4j-log4j12。

源码分析参见：[Log4j源码解析--框架流程+核心解析](http://blog.csdn.net/u011794238/article/details/50736331)，几个基本概念理的比较清楚。

![](/public/upload/java/log4j.jpeg)

从中调整的几个误区

1. LoggerRepository是Logger实例的容器，维护Logger name与Logger的映射关系。
2. Logger 本身并不直接写日志到文件，Logger（准确的说是父类）聚合Appender实现类，Logger将日志信息弄成LogEvent，交给Appender写入。

## 问题

The org.slf4j.Logger interface is the main user entry point of SLF4J API. slf4j 定好了Logger和LoggerFactory实现，最直观的感觉，为什么不是`Logger.debug(),Logger.info()`直接写到文件就好了呢？

为什么Logger需要一个name，为什么Logger要做成多例的？

Logger name是一个Logger日志输入目的地、日志级别及日志格式的描述，在一个系统中有多重配置。

**从中可以学习到：那么类似的一个实例，通过配置文件读取，Logger、LoggerRepository（log4j中貌似功能有点弱化）、LoggerFactory、LoggerManager协同工作，达到`Logger log = LoggerFactory.getLogger(name)`效果。并根据配置文件变化，及时刷新。**

## 配置文案理解

一个全套的logger配置如下

	log4j.logger.logger_name = debug_level1,appender_name1,appender_name2
	log4j.appender.appender_name1= appender_class_name
	log4j.appender.appender_name1.layout= layout_class_name
	...

`log4j.logger.org.springframework=DEBUG`单独出现便不算全套，这里`org.springframework`便表示logger name，其没有配置appender，便复用rootLogger的appender。

从中可以学习到如何用properties文件描述

1. logger与level多对一关系
2. logger与appender的一对多关系
3. appender与layout的一对多关系

## 初始化过程

LogManager

	static {
		1. 若设置DEFAULT_INIT_OVERRIDE_KEY为true，则放弃默认的初始化过程
		2. 按log4j.xml、log4j.properties和log4j.configuration 顺序尝试读取配置文件
		3. 读取环境变量configuratorClassName设置的配置类，默认为PropertyConfigurator
		4. configurator.doConfigure(url, hierarchy);
	}
	
自定义初始化过程

通过初始化过程分析，那么自定义log4j的初始化过程的本质便是：自己触发执行`doConfigure(Properties properties, LoggerRepository hierarchy)`，doConfigure多次执行，会覆盖先前的配置。