---

layout: post
title: kubernetes自动扩容缩容
category: 技术
tags: Kubernetes
keywords: kubernetes yaml

---

## 简介（持续更新）

* TOC
{:toc}

开源框架[kubernetes/autoscaler](https://github.com/kubernetes/autoscaler)

[kubernetes 资源管理概述](https://cizixs.com/2018/06/25/kubernetes-resource-management/)

![](/public/upload/kubernetes/kubernetes_resource_manager.png)


[Kubernetes Autoscaling 101: Cluster Autoscaler, Horizontal Pod Autoscaler, and Vertical Pod Autoscaler](https://medium.com/magalix/kubernetes-autoscaling-101-cluster-autoscaler-horizontal-pod-autoscaler-and-vertical-pod-2a441d9ad231) 未读完

**Kubernetes at its core is a resources management and orchestration tool**. It is ok to focus day-1 operations to explore and play around with its cool features to deploy, monitor and control your pods. However, you need to think of day-2 operations as well. You need to focus on questions like:

1. How am I going to scale pods and applications?
2. How can I keep containers running in a healthy state and running efficiently?
3. With the on-going changes in my code and my users’ workloads, how can I keep up with such changes?

## Cluster Auto Scaler 

[kubernetes 资源管理概述](https://cizixs.com/2018/06/25/kubernetes-resource-management/)

随着业务的发展，应用会逐渐增多，每个应用使用的资源也会增加，总会出现集群资源不足的情况。为了动态地应对这一状况，我们还需要 CLuster Auto Scaler，能够根据整个集群的资源使用情况来增减节点。

对于公有云来说，Cluster Auto Scaler 就是监控这个集群因为资源不足而 pending 的 pod，根据用户配置的阈值调用公有云的接口来申请创建机器或者销毁机器。对于私有云，则需要对接内部的管理平台。

## Horizontal Pod Autoscaler 

## Vertical Pod Autoscaler


## 其它

目前 HPA 和 VPA 不兼容，只能选择一个使用，否则两者会相互干扰。而且 VPA 的调整需要重启 pod，这是因为 pod 资源的修改是比较大的变化，需要重新走一下 apiserver、调度的流程，保证整个系统没有问题。目前社区也有计划在做原地升级，也就是说不通过杀死 pod 再调度新 pod 的方式，而是直接修改原有 pod 来更新。

理论上 HPA 和 VPA 是可以共同工作的，HPA 负责瓶颈资源，VPA 负责其他资源。比如对于 CPU 密集型的应用，使用 HPA 监听 CPU 使用率来调整 pods 个数，然后用 VPA 监听其他资源（memory、IO）来动态扩展这些资源的 request 大小即可。当然这只是理想情况



