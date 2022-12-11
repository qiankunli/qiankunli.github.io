---

layout: post
title: TIDB源码分析
category: 技术
tags: Storage
keywords: TIDB

---

## 前言（未完成）

* TOC
{:toc}

[TiDB 源码阅读系列文章（一）序](https://zhuanlan.zhihu.com/p/34109413) 是tidb 官方出发的 源码分析系列文章，前几章是综述，后面的章节是具体模块的源码分析。

[一条 select 语句在 TiDB Server 层都发生了什么](https://mp.weixin.qq.com/s/t3SErHZ1-TeLaaTAsJQI4A) 未读。

## 包的划分

大部分包都以接口的形式对外提供服务，大部分功能也都集中在某个包中

![](/public/upload/data/tidb_source_package.png)


## SQL层

![](/public/upload/data/tidb_sql_layer_architecture.jpg)


