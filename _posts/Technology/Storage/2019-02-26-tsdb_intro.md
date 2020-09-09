---

layout: post
title: 时序性数据库介绍及对比
category: 技术
tags: Storage
keywords: tsdb

---

## 前言（未完成）

* TOC
{:toc}

建议看下前文 [OpenTSDB 入门](http://qiankunli.github.io/2017/08/02/opentsdb.html) [influxdb 入门](http://qiankunli.github.io/2019/02/26/influxdb_intro.html)


[下一代监控系统大阅兵-基于TSDB的监控系统定制开发需求调研](https://zhuanlan.zhihu.com/p/35978607)

## 什么是TSDB？

wikipedia：A time series database (TSDB) is a software system that is optimized for handling time series data, arrays of numbers indexed by time (a datetime or a datetime range).

其中，时序列数据可以定义如下：

1. 可以唯一标识的序列名/ID（比如cpu.load.1）及meta-data；
2. 一组数据点 point {timestamp, value}。timestamp是一个Unix时间戳，一般精度会比较高，比如influxdb里面是nano秒。一般来说这个精度都会在秒以上。
3. 额外的，OpenTSDB提出了为 metric 增加 tag（key-value 键值对） 的方法来实现更方便和强大的查询语法，为influxdb 所沿袭

一般时序列数据都具备如下两个特点：

1. 数据结构简单
2. 数据量大

所谓的结构简单，可以理解为某一度量指标在某一时间点只会有一个值，没有复杂的结构（嵌套、层次等）和关系（关联、主外键等）。

在实现上

1. OpenTSDB schema 中 metric、ts、value、tag， OpenTSDB 没有表的概念，估计是受hbase的影响。 [Prometheus COMPARISON TO ALTERNATIVES](https://prometheus.io/docs/introduction/comparison/)  time series are identified by a set of arbitrary key-value pairs (OpenTSDB tags are Prometheus labels). **All data for a metric is stored together**
2. influxdb schema 中 measurement、ts、value、tag，influxdb measurement 类似于metric，但单独成表

## 方案选型

### push/pull

对部署方式会产生影响

## Prometheus VS InfluxDB VS OpenTSDB

[Prometheus VS InfluxDB](https://blog.csdn.net/u011537073/article/details/80305804)

OpenTSDB：基于 Hadoop and HBase 的时间序列数据库

Prometheus 和InfluxDB 、 OpenTSDB最大的区别可以理解成：后两者仅仅是数据库，而 Prometheus 是一个监控系统，它不仅仅包含了时间序列数据库，还有全套的抓取、检索、绘图、报警的功能

Prometheus 官网有个专门的对比 [Prometheus COMPARISON TO ALTERNATIVES](https://prometheus.io/docs/introduction/comparison/)


## 注意事项

### tag Cardinality 基数不能太大 

[series cardinality](https://docs.influxdata.com/influxdb/v1.7/concepts/glossary/#series-cardinality)

[Tags with high cardinality](https://community.influxdata.com/t/tags-with-high-cardinality/1557) 

1. influxdb 会为tag 与 metric 建立反向索引
2. tag 的基数说的是可选值的数量，比如省份的基数就是3x个，直辖市的基数是4个。这个influxdb 表的Cardinality 是所有tag 基数的乘积
3. Cardinality 基数过大，会引起influxdb 查询缓慢。PS：还有待确认


## 其它

RedisTimeSeries 是 Redis 的一个扩展模块。它专门面向时间序列数据提供了数据类型和访问接口，并且支持在 Redis 实例上直接对数据进行按时间范围的聚合计算。

