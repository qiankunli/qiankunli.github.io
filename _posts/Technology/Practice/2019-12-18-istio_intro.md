---

layout: post
title: istio学习
category: 架构
tags: Practice
keywords: window

---

## 简介（未完成）

* TOC
{:toc}

类似产品 [SOFAMesh 介绍](https://www.sofastack.tech/projects/sofa-mesh/overview/)

[Istio 庖丁解牛一：组件概览](https://www.servicemesher.com/blog/istio-analysis-1/)未读

[使用 Istio 实现基于 Kubernetes 的微服务应用](https://www.ibm.com/developerworks/cn/cloud/library/cl-lo-implementing-kubernetes-microservice-using-istio/index.html)

## 整体架构

![](/public/upload/practice/istio.jpg)


控制平面的三大模块，其中的Pilot和Citadel/Auth都不直接参与到traffic的转发流程，因此他们不会对运行时性能产生直接影响。

|组件名|代码|对应进程|
|---|---|---|
|pilot||istio-pilot|

## Envoy

Envoy 是 Istio 中最基础的组件，所有其他组件的功能都是通过调用 Envoy 提供的 API，在请求经过 Envoy 转发时，由 Envoy 执行相关的控制逻辑来实现的。

[浅谈Service Mesh体系中的Envoy](https://yq.aliyun.com/articles/606655)

类似产品 [MOSN](https://github.com/sofastack/sofa-mosn) [MOSN 文档](https://github.com/sofastack/sofa-mosn)

## Mixer

![](/public/upload/practice/istio_mixer.png)

mixer 的变更是比较多的，有v1 architecture 和 v2 architecture，社区还尝试将其与proxy/envoy 合并。

[WHY DOES ISTIO NEED MIXER?](https://istio.io/faq/mixer/#why-mixer)Mixer provides a rich intermediation layer between the Istio components as well as Istio-based services, and the infrastructure backends used to perform access control checks and telemetry capture. Mixer enables extensible policy enforcement and control within the Istio service mesh. It is responsible for insulating（隔离） the proxy (Envoy) from details of the current execution environment and the intricacies of infrastructure backends. 

理解“为什么需要一个Mixer” 的关键就是 理解infrastructure backend， 它们可以是Logging/metric 等，mixer 将proxy 与这些系统隔离（proxy通常是按照无状态目标设计的），代价就是每一次proxy间请求需要两次与mixer的通信 影响了性能，这也是社区想将proxy与mixer合并的动机（所以现在proxy是不是无状态就有争议了）。

[Service Mesh 发展趋势(续)：棋到中盘路往何方 | Service Mesh Meetup 实录](https://www.sofastack.tech/blog/service-mesh-development-trend-2/)

![](/public/upload/practice/istio_mixer_evolution.png)

## pilot

[服务网格 Istio 初探 -Pilot 组件](https://www.infoq.cn/article/T9wjTI2rPegB0uafUKeR)

![](/public/upload/practice/istio_pilot_detail.png)

1. Pilot 的架构，最下面一层是 Envoy 的 API，提供 Discovery Service 的 API，这个 API 的规则由 Envoy 约定，Pilot 实现 Envoy API Server，**Envoy 和 Pilot 之间通过 gRPC 实现双向数据同步**。
2. Pilot 最上面一层称为 Platform Adapter，这一层不是 Kubernetes 调用 Pilot，而是 **Pilot 通过调用 Kubernetes 来发现服务之间的关**系，Pilot 通过在 Kubernetes 里面注册一个 Controller 来监听事件，从而获取 Service 和 Kubernetes 的 Endpoint 以及 Pod 的关系。

Istio 通过 Kubernets CRD 来定义自己的领域模型，使大家可以无缝的从 Kubernets 的资源定义过度到 Pilot 的资源定义。


## 其它

任何软件架构设计，其核心都是围绕数据展开的，基本上如何定义数据结构就决定了其流程的走向，剩下的不外乎加上一些设计手法，抽离出变与不变的部分，不变的部分最终会转化为程序的主流程，基本固化，变的部分尽量保证拥有良好的扩展性、易维护性，最终会转化为主流程中各个抽象的流程节点。

