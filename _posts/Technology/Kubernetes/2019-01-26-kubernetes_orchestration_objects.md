---

layout: post
title: Kubernetes objects之编排对象 
category: 技术
tags: Kubernetes
keywords: kubernetes stateset

---

## 简介（未完成）

* TOC
{:toc}


![](/public/upload/kubernetes/kubernetes_object.png)

你一定有方法在不使用 Kubernetes、甚至不使用容器的情况下，自己 DIY 一个类似的方案出来。但是，一旦涉及到升级、版本管理等更工程化的能力，Kubernetes 的好处，才会更加凸现。


**Kubernetes 的各种object，就是常规的各个项目组件在 kubernetes 上的表示** [深入理解StatefulSet（三）：有状态应用实践](https://time.geekbang.org/column/article/41217) 充分体现了在我们把服务 迁移到Kubernetes 的过程中，要做多少概念上的映射。

## 集大成者——StatefulSet

StatefulSet 的设计其实非常容易理解。它把真实世界的应用状态，抽象为了两种情况：
1. 拓扑状态，比如应用的主节点 A 要先于从节点 B 启动
2. 存储状态，应用的多个实例分别绑定了不同的存储数据

StatefulSet 的核心功能，就是通过某种方式记录这些状态，然后在 Pod 被重新创建时，能够为新 Pod 恢复这些状态。程序 = 数据结构 + 算法。**新增了一个功能，一定在数据表示上有体现（对应数据结构），一定在原来的工作流程中有体现或者改了工作流程（对应算法）**


StatefulSet 这个控制器的主要作用之一，就是使用Pod 模板创建 Pod 的时候，对它们进行编号，并且按照编号顺序逐一完成创建工作。而当 StatefulSet 的“控制循环”发现 Pod 的“实际状态”与“期望状态”不一致，需要新建或者删除 Pod 进行“调谐”的时候，它会严格按照这些Pod 编号的顺序，逐一完成这些操作。**所以，StatefulSet 其实可以认为是对 Deployment 的改良。**

StatefulSet 里的不同 Pod 实例，不再像 ReplicaSet 中那样都是完全一样的，而是有了细微区别的。比如，每个 Pod 的 hostname、名字等都是不同的、携带了编号的。Kubernetes 通过 Headless Service，为这些有编号的 Pod，在 DNS 服务器中生成带有同样编号的 DNS 记录。StatefulSet 还为每一个 Pod 分配并创建一个同样编号的 PVC。


 [DNS for Services and Pods](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/) “Normal” (not headless) Services are assigned a DNS A record for a name of the form `my-svc.my-namespace.svc.cluster.local`. Headless Service 所代理的所有 Pod 的 IP 地址，都会被绑定一个这样格式的 DNS 记录 `<pod-name>.<svc-name>.<namespace>.svc.cluster.local`

通过 Headless Service 的方式，StatefulSet 为每个 Pod 创建了一个固定并且稳定的 DNS记录，来作为它的访问入口。在部署“有状态应用”的时候，应用的每个实例拥有唯一并且稳定的“网络标识”，是一个非常重要的假设。

Persistent Volume Claim 和 PV 的关系。运维人员创建PV，告知有多少volume。开发人员创建Persistent Volume Claim 告知需要多少大小的volume。创建一个 PVC，Kubernetes 就会自动为它绑定一个符合条件的Volume。即使 Pod 被删除，它所对应的 PVC 和 PV 依然会保留下来。所以当这个 Pod 被重新创建出来之后，Kubernetes 会为它找到同样编号的 PVC，挂载这个 PVC 对应的 Volume，从而获取到以前保存在 Volume 里的数据。

## ConfigMap

## DaemonSet

## Job/CronJob

## 体会

学习rc、deployment、service、pod 这些Kubernetes object 时，因为功能和yaml 有直接的一对一关系，所以体会不深。在学习StatefulSet 和 DaemonSet 时，有几个感觉

1. Kubernetes object 是分层次的，pod 是很基础的层次，然后rc、deployment、StatefulSet 等用来描述如何管理它。

    * 换句话说，pod 的配置更多是给docker看的，deployment 和StatefulSet 等配置更多是给 Kubernetes Controller 看的
    * pod 其实有一份儿配置的全集， DaemonSet 的生效 是背后偷偷改 pod 配置 加上 恰当的时机操作pod api
2. Kubernetes objects是否可以笼统的划分一下，编排对象架构在调度对象之上？

    1. 调度对象pod、service、volume
    2. 编排对象StatefulSet、DaemonSet 和Job/CronJob 等
