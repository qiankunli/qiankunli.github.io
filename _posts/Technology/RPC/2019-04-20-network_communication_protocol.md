---

layout: post
title: 网络通信协议
category: 架构
tags: RPC
keywords: network communication protocol

---

## 简介

* TOC
{:toc}

[Dubbo 3.0 前瞻之：常用协议对比及 RPC 协议新形态探索](https://mp.weixin.qq.com/s/8zgvuwdNrT3GOhZZZTfEaQ)一个简单的协议需要定义数据交换格式，协议格式和请求方式
1. 数据交换格式，在 RPC 中也叫做序列化格式，评价序列化优劣一般从三个维度：
    1. 序列化后的字节数组大小
    2. 序列化和反序列化速度
    3. 序列化后的可读性
2. 协议格式。
    1. 紧凑型协议，只提供用于调用的简单元数据和数据内容。
    2. 复合型协议，会携带框架层的元数据用来提供功能上的增强
3. 请求方式，和协议格式息息相关，常见的请求格式有同步 Request/Response 和异步 Request/Response，区别是客户端发出一个请求后，是否需要同步等待响应返回。如果不需要等待响应，一个链接上就可以同时存在多个未完成的请求，这也被叫做多路复用。另外的请求模型有 Streaming ，在一次完整的业务调用中存在多次 RPC，每次都传输一部分数据，适合流数据传输。

为什么RPC 框架不像消息队列一样，采用性能更好的专用的序列化实现呢？这个原因很简单，消息队列它需要序列化数据的类型是固定的，只是它自己的内部通信的一些命令。但 RPC 框架，它需要序列化的数据是，用户调用远程方法的参数，这些参数可能是各种数据类型，所以必须使用通用的序列化实现，确保各种类型的数据都能被正确的序列化和反序列化。

## 协议的几个套路


### 如何表达长度

古文就好比流，一个问题就是没有标点符号

1. 在协议的起始位置(或其它约定位置)直接说长度
2. 在协议的起始位置标识有几个元素，然后每个元素分别说明自己的长度。比如redis请求的协议格式

在流中也加上“标点符号”来断句不就行了？
1. 这个办法是可行的，也有很多传输协议采用这种方法，比如 HTTP1 协议，它的分隔符是换行（\r\n）。但是，这个办法有一个问题比较难处理，在自然语言中，标点符号是专用的，它没有别的含义，和文字是有天然区分的。在数据传输的过程中，无论你定义什么字符作为分隔符，理论上，它都有可能会在传输的数据中出现。为了区分“数据内的分隔符”和真正的分隔符，你必须得在发送数据阶段，加上分隔符之前，把数据内的分隔符做转义，收到数据之后再转义回来。这是个比较麻烦的过程，还要损失一些性能。
2. 更加实用的方法是，我们给每句话前面加一个表示这句话长度的数字

### 简单协议

1. 请求和回复协议格式一致

### 复杂协议

1. 大部分协议分为header和data部分，data部分经常随着header部分变化，比如http协议transfer-encoding:chunked或者multipart
2. 需要复杂的初始化，比如ssl，在工作之前，要进行ssl握手。
3. 一次语义动作的完成，需要多次通信。通常表现在，既有控制指令，又有数据指令。控制指令通常作为数据指令的上下文，需要协议框架进行存储。

## 协议的特点

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

### 文本协议

http 太普遍以至于我们都不感觉到它们是协议了

[netty对http2协议的解析](http://qiankunli.github.io/2017/06/12/netty_http2.html)

[Redis 学习](http://redisdoc.com/topic/protocol.html)

Redis 这个文本协议的实现性能仍然可以和二进制协议一样快。因为 Redis 协议将数据的长度放在数据正文之前， 所以程序无须像 JSON 那样， 为了寻找某个特殊字符而扫描整个 payload 

### 自定义二进制

[《Apache Kafka源码分析》——server](http://qiankunli.github.io/2019/01/30/kafka_learn_2.html)

[Thrift基本原理与实践（一）](http://qiankunli.github.io/2016/07/13/thrift.html)

又包括两个分类：

1. 传统的变长协议，比如bolt/dubbo
2. 基于帧的协议，比如http2

### java/c/golang 

[为什么netty比较难懂？](http://qiankunli.github.io/2017/10/13/learn_netty.html)

## 其它
思考：为什么不用http作为rpc协议？当然，springcloud就是用的http。 HTTP 协议的数据包大小相对请求数据本身要大很多；**HTTP 协议属于无状态协议**，客户端无法对请求和响应进行关联，每次请求都需要重新建立连接，响应完成后再关闭连接。

