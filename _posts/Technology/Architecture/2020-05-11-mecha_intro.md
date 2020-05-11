---

layout: post
title: mecha 理念学习
category: 架构
tags: Architecture
keywords: system design principle 

---

## 简介（未完成）

* TOC
{:toc}

两篇文章很多深度，需要多次重复阅读。 

[Mecha：将Mesh进行到底](https://mp.weixin.qq.com/s/sLnfZoVimiieCbhtYMMi1A)

[[译] 多运行时微服务架构](https://skyao.io/post/202003-multi-runtime-microservice-architecture/)  

rpc 有mesh，db、mq、redis 都搞mesh，mesh 的未来一定不是更多的sidecar， 运维根本受不了。必然需要出现新的形态来解决 Sidecar 过多的问题，合并为一个或者多个 Sidecar 就会成为必然。


现代分布式应用的对外需求分为四种类型（生命周期，网络，状态，绑定）。

![](/public/upload/architecture/four_needs_of_app.jpg)

单机时代，我们习惯性认为 应用 ==> systemcall ==> 应用。  但实际上，换个视角（以应用为中心），应用 对外的需求由systemcall 抽象，最终由内核提供服务。那么在分布式时代，就缺一个类似systemcall 的分布式原语，把分布式的能力 统一标准化之后 给到应用。

![](/public/upload/architecture/mecha_overview.png)

API 和配置的制订以及标准化，预计将会是 Mecha 成败的关键。PS：历史一次次的告诉我们：产品不重要，协议才重要，协议才是最直接反应理念的东西

1. 数据库产品不重要，牛逼的是sql
2. istio 还好， 牛逼的是xds


## 以dapr 为例

以微软开源的 dapr 为例，应用所有与 外界的交互（消息队列、redis、db、rpc） 都通过dapr http api

1. 消息队列：
    * 发布消息 `http://localhost:daprport/v1.0/publish/<topic>`
    * 订阅消息  dapr 询问app 要订阅哪些topic，dapr 订阅topic， 当收到topic 消息时，发给app

2. rpc : 请求远程服务 `http://localhost:daprport/v1.0/invoke/<appId>/method/<method-name>`

为了进一步简化调用的过程（毕竟发一个最简单的 HTTP GET 请求也要应用实现 HTTP 协议的调用 / 连接池管理等），dapr 提供了各个语言的 SDK，如 java / go / python / dotnet / js / cpp / rust 。另外同时提供 HTTP 客户端和 gRPC 客户端。我们以 Java 为例，java 的 client API 接口定义如下：

```java
public interface DaprClient {  
   Mono<Void> publishEvent(String topic, Object event);
   Mono<Void> invokeService(Verb verb, String appId, String method, Object request);
    ......
}
```

## Mecha 架构中

![](/public/upload/architecture/mecha_intro.png)

1. 所有分布式能力使用的过程（包括访问内部生态体系和访问外部系统）都被 Runtime 接管和屏蔽实现
2. 通过 CRD/ 控制平面实现声明式配置和管理（类似 Servicemesh）
3. 部署方式上 Runtime 可以部署为 Sidecar 模式，或者 Node 模式，取决于具体需求，不强制