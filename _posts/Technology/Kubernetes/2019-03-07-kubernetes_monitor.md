---

layout: post
title: Kubernetes监控
category: 技术
tags: Kubernetes
keywords: Kubernetes monitor

---

## 简介

* TOC
{:toc}

## 容器监控和Kubernetes监控不一样

[Kubernetes监控在小米的落地](https://mp.weixin.qq.com/s/ewwD6A3-ClbotdfFmYY3KA) 为了更方便的管理容器，Kubernetes对Container进行了封装，拥有了Pod、Deployment、Namespace、Service等众多概念。与传统集群相比，Kubernetes集群监控更加复杂：

1. 监控维度更多，除了传统物理集群的监控，还包括核心服务监控（apiserver，etcd等）、容器监控、Pod监控、Namespace监控等。
2. 监控对象动态可变，在集群中容器的销毁创建十分频繁，无法提前预置。
3. 监控指标随着容器规模爆炸式增长，如何处理及展示大量监控数据。
4. 随着集群动态增长，监控系统必须具备动态扩缩的能力。

## 监控什么

![](/public/upload/kubernetes/kubernetes_monitor.png)

类似于Prometheus的pull理念，应用要自己暴露出一个/metrics，而不是单纯依靠监控组件从”外面“ 分析它

1. 依靠现成组件提供 通用metric
2. 自己实现

### 自定义监控指标

在过去的很多 PaaS 项目中，其实都有一种叫作 Auto Scaling，即自动水平扩展的功能。只不过，这个功能往往只能依据某种指定的资源类型执行水平扩展，比如 CPU 或者 Memory 的使用值。而在真实的场景中，用户需要进行 Auto Scaling 的依据往往是自定义的监控指标。比如，某个应用的等待队列的长度，或者某种应用相关资源的使用情况。


Metrics server复用了api-server的库来实现自己的功能，比如鉴权、版本等，为了实现将数据存放在内存中吗，去掉了默认的etcd存储，引入了内存存储。因为存放在内存中，因此监控数据是没有持久化的，可以通过第三方存储来拓展


## Metrics Server

![](/public/upload/kubernetes/kubernetes_metric_server.png)

1. Metrics API URI 为 `/apis/metrics.k8s.io/`，在 `k8s.io/metrics` 维护
2. 必须部署 metrics-server 才能使用该 API，metrics-server 通过调用 Kubelet Summary API 获取数据，Summary API 返回的信息，既包括了 cAdVisor 的监控数据，也包括了 kubelet 本身汇总的信息。Pod 的监控数据是从kubelet 的 Summary API （即 `<kubelet_ip>:<kubelet_port>/stats/summary`）采集而来的。
3. Metrics server以Deployment 形式存在，复用了api-server的库来实现自己的功能，比如鉴权、版本等，为了实现将数据存放在内存中吗，去掉了默认的etcd存储，引入了内存存储。因此监控数据是没有持久化的，可以通过第三方存储来拓展

从cadvisor 视角看

![](/public/upload/kubernetes/kubernetes_cadvisor.png)



