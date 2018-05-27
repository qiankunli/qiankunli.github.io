---

layout: post
title: Spark Stream 学习
category: 技术
tags: Data
keywords: Spark

---

## 前言 



## spark 和strorm

storm做的是流式处理，就是数据不停地来，storm一直在运行，就是kafka、rabbit可以不停地接消息，storm可以不停地处理消息一样。

hadoop和spark等则是批处理，其数据源是一个明确的文件，输出也是一个名明确的文件。

（待整理）实时性和大数据量其实是不可兼得的，比如数据量很大的话，就要移动计算（而不是移动数据），这时实现实时性就比较困难。


## spark stream（未完成） 

spark stream 是微批处理。

spark，数据流进来，根据时间分隔成一小段，一小段做处理1、处理2、处理3。
storm，一个消息进来，一个消息做处理1、处理2、处理3

