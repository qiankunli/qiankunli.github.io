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

![](/public/upload/practice/istio.png)



## Envoy

Envoy 是 Istio 中最基础的组件，所有其他组件的功能都是通过调用 Envoy 提供的 API，在请求经过 Envoy 转发时，由 Envoy 执行相关的控制逻辑来实现的。

[浅谈Service Mesh体系中的Envoy](https://yq.aliyun.com/articles/606655)

类似产品 [MOSN](https://github.com/sofastack/sofa-mosn) [MOSN 文档](https://github.com/sofastack/sofa-mosn)

## Mixer

Why does Istio need Mixer? Mixer provides a rich intermediation layer between the Istio components as well as Istio-based services, and the infrastructure backends used to perform access control checks and telemetry capture. 没有Mixer，control plan 的几个组件就要直面 无数个envoy 了。

Mixer enables extensible policy enforcement and control within the Istio service mesh. It is responsible for insulating the proxy (Envoy) from details of the current execution environment and the intricacies of infrastructure backends.

## pilot

[Istio Pilot代码深度解析](https://www.servicemesher.com/blog/201910-pilot-code-deep-dive/)


## 其它

任何软件架构设计，其核心都是围绕数据展开的，基本上如何定义数据结构就决定了其流程的走向，剩下的不外乎加上一些设计手法，抽离出变与不变的部分，不变的部分最终会转化为程序的主流程，基本固化，变的部分尽量保证拥有良好的扩展性、易维护性，最终会转化为主流程中各个抽象的流程节点。

