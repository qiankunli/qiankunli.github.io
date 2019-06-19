---

layout: post
title: elasticsearch 初步认识
category: 技术
tags: Data
keywords: elasticsearch

---

## 前言（未完成）

* TOC
{:toc}

Elasticsearch所涉及到的每一项技术都不是创新或革命性的，全文搜索、分析系统以及分布式数据库早已存在了，它的革命性在于将这些独立且有用的技术整合成一个一体化的、实时的应用。

## Comparing document-oriented and relational data

应用中的对象很少只是简单的键值列表，更多时候它拥有复杂的数据结构，比如包含日期、地理位置、另一个对象或数组。总有一天你会想到把这些对象存储在数据库中。将这些数据保存到由行和列组成的关系数据库中，就好比是把一个丰富、信息表现力强的对象拆散了放入一个非常大的表格中：你不得不拆散对象以适应表模式（通常一列表示一个字段），然后又不得不在查询的时候重建它们。

elasticsearch 是面向文档的，**使用JSON作为文档序列化格式**（JSON已经成为NoSQL领域的标准格式），这意味着它可以存储整个对象或文档，然而它不仅仅是存储，还会索引每个文档的内容使之可以被索引、搜索、排序、过滤。elasticsearch官方客户端会自动为你序列化和反序列化Json。 PS：document-oriented 虽然不像relational database 一样有schema，但也是有json格式的。

|称谓|relational db|elasticsearch||
|---|---|---|---|
|数据库|databases|indices/indexes|索引在es里如此自然，以至于数据库都叫索引|
|表|tables|types|
|记录|rows|documents|
|列|columns|fields|document中所有field都拥有一个倒排索引|
|数据库操作|SQL|restful api|
|新增一条记录|create databalse xx;<br> use xx <br>create table xx;<br> insert xx|`PUT /$index/$type/$id`|

es 不适合/不善于频繁更新、复杂关联查询、事务等操作

## 安装和配置

tar.gz 文件解压完

    bin
        elasticsearch
    config
        elasticsearch.yml  
        jvm.options  
        log4j2.properties  
        role_mapping.yml  
        roles.yml  
        users  
        users_roles
    jdk
    lib
    LICENSE.txt
    logs
    modules
    NOTICE.txt
    plugins
    README.textile

`bin/elasticsearch` 前台启动，`bin/elasticsearch -d`后台启动。