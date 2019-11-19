---

layout: post
title: influxdb入门
category: 技术
tags: Storage
keywords: influxdb

---

## 前言（未完成）

* TOC
{:toc}

可以先看下同为时序数据库的 [OpenTSDB 入门](http://qiankunli.github.io/2017/08/02/opentsdb.html)，同类产品还可以参见 [Prometheus 入门与实践](https://www.ibm.com/developerworks/cn/cloud/library/cl-lo-prometheus-getting-started-and-practice/index.html) [时序性数据库介绍及对比](http://qiankunli.github.io/2019/02/26/tsdb_intro.html)

[官网](https://www.influxdata.com/)


influxdb 提供统一的TICK 技术栈，The TICK stack - Open Source Components

1. Telegraf - Data collection
2. InfluxDB - Data storage
3. Chronograf - Data visualization
4. Kapacitor - Data processing and events

[InfluxDB学习系列教程，InfluxDB入门必备教程](https://www.cnblogs.com/waitig/p/5673564.html)


InfluxDB自带web管理界面，在浏览器中输入 http://服务器IP:8083 即可进入web管理页面。


## schema

[时序性数据库介绍及对比](http://qiankunli.github.io/2019/02/26/tsdb_intro.html)

### 与传统数据库中的名词做比较

|influxDB中的名词|	传统数据库中的概念|
|---|---|
|database|	数据库」
|measurement|	数据库中的表|
|points|	表里面的一行数据|
 
### InfluxDB中独有的概念

Point由时间戳（time）、数据（field）、标签（tags）组成。

Point相当于传统数据库里的一行数据，如下表所示：

|Point属性|	传统数据库中的概念|
|---|---|
|time|	每个数据记录时间，是数据库中的主索引(会自动生成)|
|fields|	各种记录值（没有索引的属性）也就是记录的值：温度， 湿度|
|tags|	各种有索引的属性：地区，海拔|


