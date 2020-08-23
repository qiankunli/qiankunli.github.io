---

layout: post
title: knative入门
category: 技术
tags: Kubernetes
keywords: kubernetes knative

---

## 简介（持续更新）

* TOC
{:toc}

曾经有一篇文章， [Kubernetes何时才会消于无形却又无处不在？](https://mp.weixin.qq.com/s?__biz=MzA5OTAyNzQ2OA==&mid=2649699253&idx=1&sn=7f47db06b63c4912c2fd8b4701cb8d79&chksm=88930cd6bfe485c04b99b1284d056c886316024ba4835be8967c4266d9364cffcfedaf397acc&mpshare=1&scene=23&srcid=1102iGdvWF6lcNRaDD19ieRy%23rd)一项技术成熟的标志不仅仅在于它有多流行，还在于它有多不起眼并且易于使用。Kubernetes依然只是一个半成品，还远未达到像Linux内核及其周围操作系统组件在过去25年中所做到的那种“隐形”状态。 

那么knative 的出现应该是对上述问题的解决，且，就像k8s一样， knative带出来一系列规范

[官网](https://knative.dev/)


##  入门

[Knative入门——构建基于 Kubernetes 的现代化Serverless应用](https://www.servicemesher.com/getting-started-with-knative/)

Knative 是以 Kubernetes 的一组自定义资源类型（CRD）的方式来安装的

![](/public/upload/kubernetes/knative_xmind.png)

## kubernetes serving

![](/public/upload/kubernetes/knative_serving.jpg)

创建示例Configuration`kubectl apply -f configuration.yaml`

    apiVersion: serving.knative.dev/v1alpha1
    kind: Configuration
    metadata:
    name: knative-helloworld
    namespace: default
    spec:
    revisionTemplate:
        spec:
        container:
            image: docker.io/gswk/knative-helloworld:latest
            env:
            - name: MESSAGE
                value: "Knative!"

Knative 转换 Configuration 定义为一些 Kubernetes 对象并在集群中创建它们。在启用 Configuration 后，可以看到相应的 Deployment、ReplicaSet 和 Pod

## kubernetes eventing

![](/public/upload/kubernetes/knative_eventing.jpg)