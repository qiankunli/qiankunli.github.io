---

layout: post
title: Kubernetes源码分析——kubelet
category: 技术
tags: Kubernetes
keywords: kubernetes 源码分析

---

## 简介

* TOC
{:toc}

前文



kubelet 调用下层容器运行时的执行过程，并不会直接调用Docker 的 API，而是通过一组叫作 CRI（Container Runtime Interface，容器运行时接口）的 gRPC 接口来间接执行的。Kubernetes 项目之所以要在 kubelet 中引入这样一层单独的抽象，当然是为了对 Kubernetes 屏蔽下层容器运行时的差异。实际上，对于 1.6 版本之前的 Kubernetes 来说，它就是直接调用 Docker 的 API 来创建和管理容器的。

![](/public/upload/kubernetes/cri_shim.png)

除了 dockershim 之外，其他容器运行时的 CRI shim，都是需要额外部署在宿主机上的。

cri 接口定义， 可以找找感觉

![](/public/upload/kubernetes/cri_interface.png)


