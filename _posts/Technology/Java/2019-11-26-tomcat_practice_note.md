---

layout: post
title: 《Tomcat底层源码解析与性能调优》笔记
category: 技术
tags: Java
keywords: tomcat

---

## 简介（未完成）

* TOC
{:toc}

## tomcat是一个Servlet 容器？


单纯的思考一下这句话，我们可以抽象出来这么一段代码：

    class Tomcat {
        List<Servlet> sers;
    }

如果Tomcat就长这样，那么它肯定是不能工作的，所以，Tomcat其实是这样：

    class Tomcat {
        Connector connector; // 连接处理器
        List<Servlet> sers;
    }

## Servlet规范与tomcat实现

![](/public/upload/java/servlet_tomcat_object.png)

![](/public/upload/java/tomcat_request.png)

## 启动入口

`/usr/java/jdk1.8.0_191/bin/java -Dxx  -Xxx org.apache.catalina.startup.Bootstrap start`