---

layout: post
title: mosn有的没的
category: 技术
tags: Mesh
keywords: mosn detail

---

## 前言

* TOC
{:toc}

之前mosn 不太懂的时候，整了很多有的没的，画了很多图，舍不得删，先放在这里。

[Service Mesh 双十一后的探索和思考(上)](https://mp.weixin.qq.com/s/-OH9WONueWhydfZlNFsguw)值得细读。

## 与envoy 对比

Envoy 支持四层的读写过滤器扩展、基于 HTTP 的七层读写过滤器扩展以及对应的 Router/Upstream 实现。如果想要基于 Envoy 的扩展框架实现 L7 协议接入，目前的普遍做法是基于 L4 filter 封装相应的 L7 codec，在此基础之上再实现对应的协议路由等能力，无法复用 HTTP L7 的扩展框架。


envoy 对应逻辑 [深入解读Service Mesh的数据面Envoy](https://sq.163yun.com/blog/article/213361303062011904)

![](/public/upload/mesh/envoy_new_connection.jpg)

[深入解读Service Mesh的数据面Envoy](https://sq.163yun.com/blog/article/213361303062011904)下文以envoy 实现做一下类比 用来辅助理解mosn 相关代码的理念：

![](/public/upload/mesh/envoy_on_data.jpg)

对于每一个Filter，都调用onData函数，咱们上面解析过，其中HTTP对应的ReadFilter是ConnectionManagerImpl，因而调用ConnectionManagerImpl::onData函数。ConnectionManager 是协议插件的处理入口，**同时也负责对整个处理过程的流程编排**。

![](/public/upload/mesh/envoy_data_parse.jpg)

## 补充细节（之前没懂的时候画了很多图）

[SOFAMosn Introduction](https://github.com/sofastack/sofastack-doc/blob/master/sofa-mosn/zh_CN/docs/Introduction.md) 

![](/public/upload/go/mosn_io_process.png)

不同颜色 表示所处的 package 不同

![](/public/upload/go/mosn_object.png)

一次http1协议请求的处理过程（绿色部分表示另起一个协程）

![](/public/upload/go/mosn_http_read.png)

代码的组织（pkg/stream,pkg/protocol,pkg/proxy）  跟架构是一致的

![](/public/upload/go/mosn_layer.png)

1. `pkg/types/connection.go` Connection
2. `pkg/types/stream.go` StreamConnection is a connection runs multiple streams
3. `pkg/types/stream.go` Stream is a generic protocol stream
4. 一堆listener 和filter 比较好理解：Method in listener will be called on event occur, but not effect the control flow.Filters are called on event occurs, it also returns a status to effect control flow. Currently 2 states are used: Continue to let it go, Stop to stop the control flow.