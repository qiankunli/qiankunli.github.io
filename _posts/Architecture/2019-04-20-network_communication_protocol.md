---

layout: post
title: 网络通信协议
category: 架构
tags: Architecture
keywords: project

---

## 简介（持续更新）

* TOC
{:toc}






## 协议的制定

1. 文本协议 vs 二进制协议

### 二进制安全

示例：[图解Redis通信协议](https://www.jianshu.com/p/f670dfc9409b)Redis客户端和服务端之间使用一种名为RESP(REdis Serialization Protocol)的二进制安全文本协议进行通信

[Binary-safe](https://en.wikipedia.org/wiki/Binary-safe) is a computer programming term mainly used in connection with string manipulating functions.
 A binary-safe function is essentially one that treats its input as a 
raw stream of data without any specific format. It should thus work with
 all 256 possible values that a character can take (assuming 8-bit characters).

二进制安全是一种主要用于字符串操作函数相关的计算机编程术语。一个二进制安全功能（函数），其本质上将操作输入作为原始的、无任何特殊格式意义的数据流。其在操作上应包含一个字符所能有的256种可能的值（假设为8为字符）。

何为特殊格式呢？Special characters：Most functions are not binary safe when using 
any special or markup characters, such as escape(转义) codes or those that 
expect null-terminated strings.

c中的strlen函数就不算是binary safe的，因为它依赖于特殊的字符'\0'来判断字符串是否结束。而在php中，strlen函数是binary safe的，因为它不会对任何字符（包括'\0'）进行特殊解释。


## 协议层的实现

### http/http2

http 太普遍以至于我们都不感觉到它们是协议了

[netty对http2协议的解析](http://qiankunli.github.io/2017/06/12/netty_http2.html)

### 自定义二进制

[《Apache Kafka源码分析》——server](http://qiankunli.github.io/2019/01/30/kafka_learn_2.html)

[Thrift基本原理与实践（一）](http://qiankunli.github.io/2016/07/13/thrift.html)

### java/c/golang 

[为什么netty比较难懂？](http://qiankunli.github.io/2017/10/13/learn_netty.html)