---

layout: post
title: 《Elasticsearch权威指南》笔记
category: 技术
tags: Storage
keywords: elasticsearch

---

## 前言（持续更新）

* TOC
{:toc}

面向对象编程语言流行的原因之一，是我们可以用对象来表示和处理现实生活中那些有着潜在关系和复杂结构的实体。但当我们想存储这些实体的时候问题便来了，传统上，我们以行和列的形式把数据存储在关系型数据库中，相当于使用电子表格。这种固定的存储方式导致对象的灵活性不复存在了。PS：或许是ddd存在的动因之一 [ddd(一)——领域驱动理念入门](http://qiankunli.github.io/2017/12/25/ddd.html)

如何能以对象的形式存储对象呢？相对于围绕表格去为我们的程序建模，我们可以专注使用数据，把对象本来的灵活性找回来。对象是一种语言相关，记录在内存的数据结构。为了在网络间发送或者存储它，我们需要一些标准的格式来表示它。JSON 是一种可读的以文本来表示对象的方式，已经成为NoSQL领域的事实标准格式。

Elasticsearch所涉及到的每一项技术都不是创新或革命性的，全文搜索、分析系统以及分布式数据库早已存在了，它的革命性在于将这些独立且有用的技术整合成一个一体化的、实时的应用。

## 直接存储对象

应用中的对象很少只是简单的键值列表，更多时候它拥有复杂的数据结构，比如包含日期、地理位置、另一个对象或数组。总有一天你会想到把这些对象存储在数据库中。将这些数据保存到由行和列组成的关系数据库中，就好比是把一个丰富、信息表现力强的对象拆散了放入一个非常大的表格中：你不得不拆散对象以适应表模式（通常一列表示一个字段），然后又不得不在查询的时候重建它们。

elasticsearch 是面向文档的，**使用JSON作为文档序列化格式**，这意味着它可以存储整个对象或文档，然而它不仅仅是存储，还会索引每个文档的内容使之可以被索引、搜索、排序、过滤。elasticsearch官方客户端会自动为你序列化和反序列化Json。 PS：document-oriented 虽然不像relational database 一样有schema，但也是有json格式的。

|称谓|relational db|elasticsearch||
|---|---|---|---|
|数据库|databases|indices/indexes|索引在es里如此自然，以至于数据库都叫索引|
|表|tables|types|document代表的对象的类|
|记录|rows|documents|document是以唯一ID标识并存储与es的对象的json数据|
|列|columns|fields|document中所有field都拥有一个（倒排）索引|
|数据库操作|SQL|restful api|
|新增一条记录|create databalse xx;<br> use xx <br>create table xx;<br> insert xx|`PUT /$index/$type/$id` + json body|
|更新某个字段|`update xx set xx=xx`|`update /$index/$type/$id`|es是整体更新|
|并发控制|隔离级别|乐观锁|

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

## 索引透明/自动化——数据插入即可被搜索

插入一条记录 `curl -XPUT -H 'Content-Type:application/json' http://ip:9200/school/student/1 -d '{"name":"zhangsan"}'`

查看插入的记录 `curl http://192.168.62.212:9200/school/student/1`

    {
        "_index":"school",
        "_type":"student",
        "_id":"1",
        "_version":1,   ## cud都会使_version增加
        "_seq_no":0,
        "_primary_term":1,
        "found":true,
        "_source":{
            "name":"zhangsan"
        }
    }

只返回source部分 `curl http://192.168.62.212:9200/school/student/1/_source`

    {
        "name":"zhangsan"
    }
    

使用Query DSL 通过json 请求body查询 `curl -H 'Content-Type:application/json' http://192.168.62.212:9200/school/student/_search -d '{"query":{"match":{"name":"lisi"}}}'`

增强的“select”

1. 全文搜索，凡是出现某个关键字的json 都被检索，并给出相关性评分
2. 短语搜索，检索同时出现多个关键字的 json
3. 高亮我们的搜索
4. 聚合aggregations

最关键的是：**无需配置，只需要添加数据然后开始搜索**。

## 分布式概念透明化

elasticsearch 在分布式概念（cluster/node/shard）上做了很大程度的透明化，很多操作自动完成

1. 将json记录分区到不同的shard中，
2. 将shard均匀的分配到不同节点，对索引和搜索做负载均衡
3. 冗余每一个shard
4. 将集群任意一个节点的请求路由到相应数据所在的节点
5. 无论增加删除节点，分片都可以做到无缝的扩展和迁移

集群中有一个node被选举为master，它将临时管理集群级别的一些变更，例如新建/删除database（es叫index）、增加移除node等，master不参与记录（es叫document）级别的变更和搜索。

database(es叫索引)只是一个用来指向多个shard（默认一个index被分配5个shard）的逻辑命名空间，一个shard 是最小级别的工作单元，就是一个Lucene实例，本身就是一个完整的搜索引擎。只不过应用程序直接与 index（mysql叫database）而不是shard 通信罢了。shard 分为primary shard 和 replica shard，后者用于冗余数据并提供读请求。

||es|kafka||
|---|---|---|---|
|逻辑概念|database/index|topic|
|工作单位|shard|partition|
|副本机制|primary-replica|leader-follower|es副本还可对外服务|
|集群发现|广播|zk|
|复制数据|复试请求到各个shard|pull|es可以通过参数调整复制策略<br>es建议sync复制，这估计是其不适合大量写入的原因吧|

## 映射

倒排索引由document/json记录中出现的唯一的单词列表，以及对每个单词在文档中的位置组成。

为了能够把日期字段处理为日期，把数字字段处理成数字，把字符串字段处理成全文本(full-text)或精确地字符串值，elasticsearch 需要知道每个字段里都包含了什么类型。索引中每个ducument 都有一个type，每个type 拥有自己的mapping或schema definition。一个mapping 定义了field的数据类型， 以及field被elasticsearch处理的方式。


    curl -XPUT -H 'Content-Type:application/json' http://ip:9200/school/student/3 -d '{
        "name":"lisi",
        "age":18,
        "interests":["sports","music"],
        "address":{
            "country":"china",
            "city":"shanghai"
        }
    }'

`curl http://ip:9200/school/_mapping\?pretty` 查看返回

    {
        "school": {
            "mappings": {
                "properties": {
                    "address": {
                        "properties": {
                            "city": {xx}, 
                            "country": {xx}
                        }
                    }, 
                    "age": {
                        "type": "long"
                    }, 
                    "interests": {
                        "type": "text", 
                        "fields": {}
                    }, 
                    "name": {
                        "type": "text", 
                        "fields": {}
                    }
                }
            }
        }
    }

## 其它金句

自然语言实际上是高度结构化的，问题是自然语言的规则是如此复杂，计算机难以正确解析，于是常常被视为“非结构化数据”。
