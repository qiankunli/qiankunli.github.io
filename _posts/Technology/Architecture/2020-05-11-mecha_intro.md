---

layout: post
title: mecha 架构学习
category: 架构
tags: Architecture
keywords: system design principle 

---

## 简介

* TOC
{:toc}

两篇文章很多深度，需要多次重复阅读。 

[Mecha：将Mesh进行到底](https://mp.weixin.qq.com/s/sLnfZoVimiieCbhtYMMi1A)

[Multi-Runtime Microservices Architecture](https://www.infoq.com/articles/multi-runtime-microservice-architecture/)[[译] 多运行时微服务架构](https://skyao.io/post/202003-multi-runtime-microservice-architecture/)  

[云原生时代，Java危矣？](https://mp.weixin.qq.com/s/fVz2A-AmgfhF0sTkz8ADNw)不可变基础设施的内涵已不再局限于方便运维、程序升级和部署的手段，而是升华一种为**向应用代码隐藏环境复杂性的手段**，是分布式服务得以成为一种可普遍推广的普适架构风格的必要前提。

rpc 有mesh，db、mq、redis 都搞mesh，mesh 的未来一定不是更多的sidecar， 运维根本受不了。必然需要出现新的形态来解决 Sidecar 过多的问题，合并为一个或者多个 Sidecar 就会成为必然。


现代分布式应用的对外需求分为四种类型（生命周期，网络，状态，绑定）。

![](/public/upload/architecture/four_needs_of_app.jpg)

单机时代，我们习惯性认为 应用 ==> systemcall ==> 内核。  但实际上，换个视角（以应用为中心），应用 对外的需求由systemcall 抽象，最终由内核提供服务。那么在分布式时代，就缺一个类似systemcall 的分布式原语，把分布式的能力 统一标准化之后 给到应用。

![](/public/upload/architecture/mecha_overview.png)

当前的项目开发，开发人员就像老妈子一样，把db、redis、mq 等资源聚在一起，还得考虑他们的容量、负载、连接池等。后续，它们 会向水电一样，支持项目随取随用。

API 和配置的制订以及标准化，预计将会是 Mecha 成败的关键。PS：历史一次次的告诉我们：产品不重要，协议才重要，协议才是最直接反应理念的东西

1. 数据库产品不重要，牛逼的是sql
2. istio 还好， 牛逼的是xds

## Kubernetes 是基础，但单靠Kubernetes是不够的

对于Kubernetes，要管理的最小原语是容器，它专注于在容器级别和流程模型上交付分布式原语。这意味着它在管理应用的生命周期，健康检查，恢复，部署和扩展方面做得很出色，但是在容器内的分布式应用的其他方面却没有做得很好，例如灵活的网络，状态管理和绑定。 

## 应用运行时——以dapr 为例

[在云原生的时代，我们到底需要什么样的应用运行时？](https://mp.weixin.qq.com/s/PwPC1ZWNZvzQoOZvOAF2Qw)

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

## Mecha 架构

![](/public/upload/architecture/mecha_intro.png)

1. 所有分布式能力使用的过程（包括访问内部生态体系和访问外部系统）都被 Runtime 接管和屏蔽实现
2. 通过 CRD/ 控制平面实现声明式配置和管理（类似 Servicemesh）
3. 部署方式上 Runtime 可以部署为 Sidecar 模式，或者 Node 模式，取决于具体需求，不强制

## 从云原生中间件的视角

[无责任畅想：云原生中间件的下一站](https://mp.weixin.qq.com/s/EWR6xwbf53XHZ8O3m2bdVA)在云原生时代，不变镜像作为核心技术的 docker 定义了不可变的单服务部署形态，统一了容器编排形态的 k8s 则定义了不变的 service 接口，二者结合定义了服务可依赖的不可变的基础设施。有了这种完备的不变的基础设置，就可以定义不可变的中间件新形态 -- 云原生中间件。云原生时代的中间件，包含了不可变的缓存、通信、消息、事件(event) 等基础通信设施，应用只需通过本地代理即可调用所需的服务，无需关心服务能力来源。

除了序列化协议和通信协议，微服务时代的中间件体系大概有如下技术栈：

* RPC，其代表是 Dubbo/Spring Cloud/gRPC 等。
* 限流熔断等流控，如 hystrix/sentinel 等。
* Cache，其代表是 Redis。
* MQ，其代表有 kafka/rocketmq 等。
* 服务跟踪，如兼容 Opentracing 标准的各种框架。
* 日志收集，如 Flume/Logtail 等。
* 指标收集，如 prometheus。
* 事务框架，如阿里的 seata。
* 配置下发，如 apollo/nacos。
* 服务注册，如 zookeeper/etcd 等。
* 流量控制，如 hystrix/sentinel 等。
* 搜索，如 ElasticSearch。
* 流式计算，如 spark/flink。

把各种技术栈统一到一种事实上的技术标准，才能反推定义出**不可变的中间件设施的终态**。把上面这些事实上的中间件梳理一番后，整体工作即是：

* 统一定义各服务的标准模型
* 定义这些标准模型的可适配多种语言的 API
* 一个具备通信和中间件标准模型 API 的 Proxy
* 适配这些 API 的业务

![](/public/upload/architecture/application_mesh.png)

Service Proxy 可能是一个集状态管理、event 传递、消息收发、分布式追踪、搜索、配置管理、缓存数据、旁路日志传输等诸多功能于一体的 Proxy， 也可能是分别提供部分服务的多个 Proxy 的集合，但对上提供的各个服务的 API 是不变的。Application Mesh 具有如下更多的收益：

* 更好的扩展性与向后兼容性
* 与语言无关
* 与云平台无关
* 应用与中间件更彻底地解耦
* 应用开发更简单。基于新形态的中间件方案，Low Code 或者 No Code 技术才能更好落地。单体时代的 IDE 才能更进一步 -- 分布式时代的 IDE，基于各种形态中间件的标准 API 之对这些中间件的能力进行组合，以 WYSIWYG 方式开发出分布式应用。
* 更快的启动速度
* 以统一技术形态的 Service Mesh 为基础的云原生中间件技术体系真正发起起来，在其之上的 Serverless 才有更多的落地场景，广大中小企业才能分享云原生时代的技术红利，业务开发人员的编码工作就会越来越少，编程技术也会越来越智能--从手工作坊走向大规模机器自动生产时代。

## 其它

[未来：应用交付的革命不会停止](https://mp.weixin.qq.com/s/x7lTp9fJXav6nIJH_bgVMA)Kubernetes 项目一直在做的，其实是在进一步清晰和明确“应用交付”这个亘古不变的话题。只不过，相比于交付一个容器和容器镜像， Kubernetes 项目正在尝试明确的定义云时代“应用”的概念。在这里，应用是一组容器的有机组合，同时也包括了应用运行所需的网络、存储的需求的描述。而像这样一个“描述”应用的 YAML 文件，放在 etcd 里存起来，然后通过控制器模型驱动整个基础设施的状态不断地向用户声明的状态逼近，就是 Kubernetes 的核心工作原理了。PS: 以后你给公有云一个yaml 文件就可以发布自己的应用了。

[解读容器 2019：把“以应用为中心”进行到底](https://www.kubernetes.org.cn/6408.html)云原生的本质是一系列最佳实践的结合；更详细的说，云原生为实践者指定了一条低心智负担的、能够以可扩展、可复制的方式最大化地利用云的能力、发挥云的价值的最佳路径。这种思想，以一言以蔽之，就是“以应用为中心”。正是因为以应用为中心，云原生技术体系才会无限强调**让基础设施能更好的配合应用**、以更高效方式为应用“输送”基础设施能力，而不是反其道而行之。而相应的， Kubernetes 、Docker、Operator 等在云原生生态中起到了关键作用的开源项目，就是让这种思想落地的技术手段。