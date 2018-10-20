---

layout: post
title: mybatis学习
category: 技术
tags: Data
keywords: mysql mybatis

---

## 前言（完成）

What is MyBatis?

MyBatis is a first class persistence framework with support for custom SQL, stored procedures and advanced mappings. MyBatis eliminates almost all of the JDBC code and manual setting of parameters and retrieval of results. MyBatis can use simple XML or Annotations for configuration and map primitives, Map interfaces and Java POJOs (Plain Old Java Objects) to database records.

1. 一般业务模型直接体现在数据库表设计，表设计好了，页面设计确认了，剩下的code就是一个匹配的过程。我以前用持久化框架，都是先创建数据库表，一直忽视从class persistence framework 的角度来看问题，这是实际的数据流动的视角。
2. 基本隐藏了jdbc 实现，条件查询（包括结果返回）也简化了很多
3. 使用xml 或 xml 控制java 类和 database record 的映射过程




[Mybatis framework learning](http://www.codeblogbt.com/archives/303221)

1. Mybatis It is characterized by the use of Xml files to configure SQL statements. Programmers write their own SQL statements. When the requirements change, we only need to modify the configuration files, which is more flexible. 提取 xml  来隔离sql 的变化

2. In addition, Mybatis transfer to the SQL statement can be passed directly into the Java object without the need for a parameter to correspond with the placeholder; the query result set is mapped to a Java object.  sql 的输入输出 都可以是对象

![](/public/upload/data/mybatis_framework.jpg)

1. The global configuration file SqlMapConfig.xml configuring the running environment of Mybatis. The Mapper.xml file is the SQL map file, and the SQL statement in the file is configured in the file. This file needs to be in SqlMLoad in apConfig.xml.

2. SqlSessionFactory is constructed through configuration information such as Mybatis environment, that is, a conversation factory. 这个图的走势很形象，尤其是从配置文件 建builder的感觉。