---

layout: post
title: 《Elasticsearch权威指南》笔记
category: 技术
tags: Storage
keywords: elasticsearch

---

## 前言

* TOC
{:toc}

面向对象编程语言流行的原因之一，是我们可以用对象来表示和处理现实生活中那些有着潜在关系和复杂结构的实体。但当我们想存储这些实体的时候问题便来了，传统上，我们以行和列的形式把数据存储在关系型数据库中，相当于使用电子表格。这种固定的存储方式导致对象的灵活性不复存在了。PS：另一个理由是mysql like不够用。

如何能以对象的形式存储对象呢？相对于围绕表格去为我们的程序建模，我们可以专注使用数据，把对象本来的灵活性找回来。对象是一种语言相关，记录在内存的数据结构。为了在网络间发送或者存储它，我们需要一些标准的格式来表示它。JSON 是一种可读的以文本来表示对象的方式，已经成为NoSQL领域的事实标准格式。

Elasticsearch所涉及到的每一项技术都不是创新或革命性的，全文搜索、分析系统以及分布式数据库早已存在了，它的革命性在于将这些独立且有用的技术整合成一个一体化的、实时的应用。

## 核心概念

es核心概念
1. 单机核心引擎：Apache Lucene。
    2. 数据模型相关概念，索引、类别（在Elasticsearch 7.0以后，类别逐渐被弃用）、文档。文档是一个JSON格式的数据对象。对应关系型数据库中的数据行。优势在于提供了更高的自由度，不需要预先定义严格的数据库模式，文档中可以方便地新增减字段，**多个文档间也不要求字段完全一致**。同时，文档也保留了一部分结构化存储的特性，对存储的数据进行了一定的结构化封装，而没有像K-V非关系型数据库那样完全抛弃数据的结构化。
2. 分布式存储相关概念，集群、节点、分片、副本。
    1. 一个 Shard 本质上就是一个功能完备、独立的 Lucene 实例。
3. 分析检索能力相关概念：倒排索引，分析器。

应用中的对象很少只是简单的键值列表，更多时候它拥有复杂的数据结构，比如包含日期、地理位置、另一个对象或数组。总有一天你会想到把这些对象存储在数据库中。将这些数据保存到由行和列组成的关系数据库中，就好比是把一个丰富、信息表现力强的对象拆散了放入一个非常大的表格中：你不得不拆散对象以适应表模式（通常一列表示一个字段），然后又不得不在查询的时候重建它们。

elasticsearch 是面向文档的，**使用JSON作为文档序列化格式**，这意味着它可以存储整个对象或文档，然而它不仅仅是存储，还会索引每个文档的内容使之可以被索引、搜索、排序、过滤。elasticsearch官方客户端会**自动为你序列化和反序列化Json**。 PS：document-oriented 虽然不像relational database 一样有schema，但也是有json格式的。

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


[Internal Working of ElasticSearch : Deep Dive](https://medium.com/%40ByteCodeBlogger/internal-working-of-elasticsearch-deep-dive-34a87bbf0404)
1. Index ==> table, document ==> record
2. Shard ==> replica
  1. An index can be divided into multiple pieces called shards. This allows Elasticsearch to distribute and parallelize operations across a cluster.
  2. Each shard is a **fully functional and independent "index"** that can be hosted on any node in the cluster.
  3. A replica is a copy of a shard. Replicas provide redundancy and high availability. If a node fails, the data can still be served from its replica.
3. 入库逻辑，找到index ==> 找到shard ==> shard 开始存
  1. When you index (add) a document to Elasticsearch, the document is assigned to a specific index.
  2. Elasticsearch routes the document to a specific shard based on a hashing mechanism.
  3. The shard processes the document and stores it on disk. This involves analyzing the document, creating an inverted index, and storing both the original document and its searchable representation.
4. 查询逻辑，找到index+相关shard, shard检索 ，数据汇总
  - When a search query is received, Elasticsearch determines which indices and shards to query.
  - The search request is **broadcast** to all relevant shards in the cluster.
  - Each shard performs a local search and returns its results.
  - Elasticsearch aggregates these local results into a final set of results and returns them to the client.

es 不适合/不善于频繁更新、复杂关联查询、事务等操作。

## 安装和配置

tar.gz 文件解压完

```
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
```

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

## 查询

### dsl

[浅谈Elasticsearch的入门与实践](https://mp.weixin.qq.com/s/wlh2AHpNLrz9dHxPw9UrkQ)基于以上的index+document+倒排索引+分析器等概念，Elasticsearch通过分布式存储结构和分析检索能力，支持并提供了多种不同类型的查询能力，用于满足各种检索需求。
1. 单词级别查询，
    1. Term Query（精确）；把输入字符串全部看作一个完整的单词，然后去倒排索引表里面找。
    2. Fuzzy Query（模糊）；带编辑距离的term查询。具体实现：给定一个模糊度（编辑距离），ES会根据这个编辑距离，对原始的单词进行拓展，生成一系列候选的新单词。对每一个编辑距离内的新单词，做term查询。
2. 全文级别查询，全文级别查询是对多个/多种单词级别查询的封装。
    1. match，查询的主要步骤：检查字段类型，查看字段是analyzed（对输入进行分词）还是not_analyzed；分析查询字符串，将输入字符串进行分词，对分出来的每个单词，根据是否设置了模糊度参数fuzziness，选择走term query或者fuzzy query；文档评分计算。
    2. match_phrase，在match查询的基础上，保证输入的单词之间的顺序不变才会命中，性能相比match会差一些。
3. Bool查询。用于实现复杂的组合查询逻辑，具体有四种：should（或）must（且）非（must _not）filter（类似must），逻辑完备性：足够数量的或且非，可以实现任何逻辑。

[Query DSL](https://opensearch.org/docs/latest/query-dsl/)

1. leaf queries:   Leaf queries search for a specified value in a certain field or fields. PS：可以看做是基本的筛选语句/表达式，后续都是最这些表达式的组合。
    1. Term-level queries: term
	2. Full-text queries: match
    3. ...
2. Compound queries: Compound queries serve as wrappers for multiple leaf or compound clauses either to combine their results or to modify their behavior.
	1. bool 
		a. must，对应 and 。表示查询条件必须匹配。如果有多个must条件，文档必须同时满足所有这些条件。
		b. must_not，对应not。表示查询条件必须不匹配。
		c. should，对应or，匹配的条数越多分越高.表示查询条件是首选的，但不是必需的。should查询可以有多个条件，文档至少满足其中一个条件就可以被包含在结果中。
		d. filter。与must类似，filter条件也是必须匹配的，但它用于结构化查询，如范围查询、存在查询等，并且对性能有优化。可以用于作为查询中的前置过滤条件，must类似，好处是它不会参与计算相关性分数。
	2. function_score
	3. hybrid
        1. queries. An array of one or more query clauses that are used to match documents. A document must match at least one query clause in order to be returned in the results. The documents’ relevance scores from all query clauses are combined into one score by applying a search pipeline. The maximum number of query clauses is 5.

比如 query 里一个match 用的好好的，如果你想加一个过滤，就一下子变成

```json
{
  "query": {
    "match": {
      "from": "raptor"
    }
  }
}
```
```json
{
  "query": {
        "bool":{
            "must": "match": {"from": "raptor"}
            "filter": {
                "range": {
                    "year": { "gt": 2020}    
                }
            }
        }
}
```

### 得分

Term Query的文档相关度得分计算方式：利用倒排索引，对于输入的单词，考虑每个文档的以下指标：
1. TFIDF 目的：用文档中的一个单词，在一堆文档中区分出该文档；
    1. TFIDF = TF * IDF；
    2. TF（term frequency）：词频。表示单词在该文本中出现的频率（单词在该文本中出现的多不多）；
    3. IDF（inverse document frequency）：反向文档频率。 表示单词在整个文本集合中出现的频率（有多少文本包含了这个词）的倒数，IDF越大表示该词的重要性越高，反映了单词是否具有distinguish其所在文本的能力。
2. 字段的长度。字段越短相关度越高；
综合这两个指标得出每个文档的相关度评分_score。

## 其它金句

自然语言实际上是高度结构化的，问题是自然语言的规则是如此复杂，计算机难以正确解析，于是常常被视为“非结构化数据”。

Elasticsearch 背后的搜索原理，则是先分词，然后再使用倒排索引。简单来说，就是把上面的“气质小清新拼接百搭双肩斜挎包”这样的商品名称，拆分成“气质”“小清新”“拼接”“百搭”“双肩”“斜挎包”。每个标题都是这样切分。然后，建立一个索引，比如“气质”这个词，出现过的标题的编号，都按编号顺序跟在气质后面。其他的词也类似。然后，当用户搜索的时候，比如用户搜索“气质背包”，也会拆分成“气质”和“背包”两个词。然后就根据这两个词，找到包含这些词的标题，根据出现的词的数量、权重等等找出一些商品。
