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