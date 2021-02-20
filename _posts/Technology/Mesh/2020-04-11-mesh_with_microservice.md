---

layout: post
title: 与微服务框架整合
category: 技术
tags: Mesh
keywords: mesh microservice

---

## 前言

* TOC
{:toc}

[落地三年，两次架构升级，网易的Service Mesh实践之路](https://mp.weixin.qq.com/s/2UIp6l1haH6z6ISxHM4UjA)

服务框架在微服务架构中占据核心位置，因此，使用 Service Mesh 来替换正在使用的微服务框架，除了需要在 Service Mesh 数据面和控制面组件中对服务注册发现、RPC 协议、配置下发进行扩展之外，还要对现有的上层研发工作台、运维效能平台等支撑平台进行兼容设计

[构建基于Service Mesh 的云原生微服务框架](https://mp.weixin.qq.com/s/toLtpHA9ZbHVzQQbs6rtjA)迁移过程难处多啊，关键是规范和统一。

## 与微服务框架整合

||数据面|控制面|
|---|---|---|
|服务注册发现|||
|RPC 协议|协议编解码等|无|
|配置下发|路由字段支持|路由字段支持|

### 服务发现注册

1. client 查询服务节点

    1. 发现 处在容器环境，并有sidecar 运行。则将请求 导向sidecar 的ip和port。
    2. 否则，访问注册中心，执行路由选择策略等， 选中一个server 节点，建连，发起请求。 
2. server 注册服务 

    1. 直接向注册中心注册服务。注册代码在sdk里，如果后续更换注册中心、注册中心地址变更 或 注册逻辑，仍然需要升级sdk。
    2. 通过 sidecar 向注册中心注册服务。mosn 0.13.0 开始支持dubbo 的注册逻辑。 
    2. 通过 k8s 注册服务。这要求所有的服务 都运行在 k8s 集群
    3. 通过 发布系统 向注册中心注册服务


一位技术大佬的建议：如果定制化的东西比较多的话，我建议直接使用mosn和注册中心配置中心交互是最好落地的，也是最快的。我们是因为没有统一的配置中心，比如网关的配置在一个地方，蓝绿的配置又在另外一个地方，所以我们需要一个统一的配置中心，而且我们希望紧跟社区，这样的话可以享受社区福利，所以我们选择了istio+mosn，因为mosn和istio交互走的是标准的xDS，以后就算是要换envoy也是没问题的。

[如何将第三方服务注册集成到 Istio ？](https://mp.weixin.qq.com/s/EJMk0tcJ457iKNMFbmi3jQ)Istio 对 Kubernetes 具有较强的依赖性，其服务发现就是基于 Kubernetes 实现的。大量现存的微服务项目要么还没有迁移到 Kubernetes 上；要么虽然采用了 Kubernetes 来进行部署和管理，但还是使用了 Consul，Eureka 等其他服务注册解决方案或者自建的服务注册中心。

在 Istio 控制面中，Pilot 组件负责管理服务网格内部的服务和流量策略。Pilot 将服务信息和路由策略转换为 xDS 接口的标准数据结构，下发到数据面的 Envoy。Pilot 自身并不负责网格中的服务注册，而是通过集成其他服务注册表来获取网格中管理的服务。除此以外，Istio 还支持通过 API 向网格中添加注册表之外的独立服务。

[如何将第三方服务注册集成到 Istio ？](https://mp.weixin.qq.com/s/EJMk0tcJ457iKNMFbmi3jQ)
### 支持自定义RPC 协议


### 数据平面

[陌陌 Service Mesh 架构的探索与实践](https://mp.weixin.qq.com/s/EeJTpAMlx_mFZp6mh2i2xw) 

1. 平滑升级
2. Agent 容灾
3. 代理性能

一些细节：

1. [SOFAMesh中的多协议通用解决方案x-protocol介绍系列(1)-DNS通用寻址方案](https://skyao.io/post/201809-xprotocol-common-address-solution/)iptables在劫持流量时，除了将请求转发到localhost的Sidecar处外，还额外的在请求报文的TCP options 中将 ClusterIP 保存为 original dest。在 Sidecar （Istio默认是Envoy）中，从请求报文 TCP options 的 original dest 处获取 ClusterIP
2. [SOFAMesh中的多协议通用解决方案x-protocol介绍系列(2)-快速解码转发](https://skyao.io/post/201809-xprotocol-rapid-decode-forward/)

    1. 转发请求时，由于涉及到负载均衡，我们需要将请求发送给多个服务器端实例。因此，有一个非常明确的要求：就是必须以单个请求为单位进行转发。即单个请求必须完整的转发给某台服务器端实例，负载均衡需要以请求为单位，不能将一个请求的多个报文包分别转发到不同的服务器端实例。所以，拆包是请求转发的必备基础。
    2. 多路复用的关键参数：RequestId。RequestId用来关联request和对应的response，请求报文中携带一个唯一的id值，应答报文中原值返回，以便在处理response时可以找到对应的request。当然在不同协议中，这个参数的名字可能不同（如streamid等）。严格说，RequestId对于请求转发是可选的，也有很多通讯协议不提供支持，比如经典的HTTP1.1就没有支持。但是如果有这个参数，则可以实现多路复用，从而可以大幅度提高TCP连接的使用效率，避免出现大量连接。稍微新一点的通讯协议，基本都会原生支持这个特性，比如SOFARPC，Dubbo，HSF，还有HTTP/2就直接內建了多路复用的支持。

我们可以总结到，对于Sidecar，要正确转发请求：

1. 必须获取到destination信息，得到转发的目的地，才能进行服务发现类的寻址
2. 必须要能够正确的拆包，然后以请求为单位进行转发，这是负载均衡的基础
3. 可选的RequestId，这是开启多路复用的基础

## 周边支撑

## 部署架构

## 其它

笔者在最开始推进 service mesh 落地时，潜意识中过于以“追求技术”来影响方案的取舍（比如偏好使用istio），陌陌这把篇文章很好的指导了技术与实践的取舍[陌陌 Service Mesh 架构的探索与实践](https://mp.weixin.qq.com/s/EeJTpAMlx_mFZp6mh2i2xw)

1. 当前架构的痛点是 SDK 升级与跨语言，而不是缺少控制平面功能；
2. 现阶段我们关注的核心收益均由数据平面产生，因此整体方案中将着重进行数据平面 Agent 的建设

[陌陌 Service Mesh 架构的探索与实践](https://mp.weixin.qq.com/s/EeJTpAMlx_mFZp6mh2i2xw) 这篇文章的思路非常好，从业界方案 ==> 自己痛点 ==> 数据面和控制面的侧重 ==> 提炼数据面的几个关注点 ，条清缕析的给出了一条实践路线。