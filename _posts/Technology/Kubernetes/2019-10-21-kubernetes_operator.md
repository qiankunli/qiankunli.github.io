---

layout: post
title: kubernetes operator
category: 技术
tags: Kubernetes
keywords: kubernetes yaml

---

## 简介（未完成）

* TOC
{:toc}

## 在k8s上部署一个etcd

有几种方法

1. 找一个etcd image，组织一个pod，进而写一个 deployment.yaml，kubectl apply 一下
2. 使用helm
3. kubernetes operator [etcd Operator](https://coreos.com/operators/etcd/docs/latest/)

        apiVersion: etcd.database.coreos.com/v1beta2
        kind: EtcdCluster
        metadata:
        name: example
        spec:
            size: 3
            version: 3.2.13

使用etcd operator 有一个比较好玩的地方，仅需调整 size 和 version 配置，就可以控制etcd cluster 个数和版本，比第一种方法方便的多了。

我们都知道在 Kubernetes 上安装应用可以使用 Helm 直接安装各种打包成 Chart 形式的 Kubernetes 应用，但随着 Kubernetes Operator 的流行，Kubernetes 社区又推出了 [OperatorHub](https://operatorhub.io/)，你可以在这里分享或安装 Operator：https://www.operatorhub.io。


## 内涵

An Operator is a method of packaging, deploying and managing a **Kubernetes application**. A Kubernetes application is an application that is both deployed on Kubernetes and managed using the Kubernetes APIs and kubectl tooling.

An Operator is an application-specific controller that extends the Kubernetes API to create, configure and manage instances of complex stateful applications on behalf of a Kubernetes user. It builds upon the basic Kubernetes resource and controller concepts, but also includes domain or **application-specific** knowledge to automate common tasks better managed by computers.

[Redis Enterprise Operator for Kubernetes](https://redislabs.com/blog/redis-enterprise-operator-kubernetes/)Although Kubernetes is good at scheduling resources and recovering containers gracefully from a failure, it does not have primitives that understand the internal lifecycle of a data service.

[使用etcd-operator在集群内部署etcd集群](https://blog.csdn.net/fy_long/article/details/88874373)Operator 本身在实现上，其实是在 Kubernetes 声明式 API 基础上的一种“微创新”。它合理的利用了 Kubernetes API 可以添加自定义 API 类型的能力，然后又巧妙的通过 Kubernetes 原生的“控制器模式”，完成了一个面向分布式应用终态的调谐过程。

我们在看一个redis cluster 的部署过程，可以看到， **operator让你以更贴近redis的特质来部署reids**，而不是在部署deployment和拼装pod。

    apiVersion: app.redislabs.com/v1alpha1
    kind: RedisEnterpriseCluster
    metadata:
    name: redis-enterprise
    spec:
    nodes: 3
    persistentSpec:
        enabled: 'true'
        storageClassName: gp2
    uiServiceType: LoadBalancer
    username: admin@acme.com
    redisEnterpriseNodeResources:
        limits:
        cpu: 400m
        memory: 4 Gi
        requests:
        cpu: 400m
        memory: 4 Gi
    redisEnterpriseImageSpec:
        imagePullPolicy: IfNotPresent
        repository: redislabs/redis
        versionTag: 5.4.0-19

||spring|kubernetes|
|---|---|---|
|核心|ioc模式|声明式api + controller模式|
|常规使用|`<bean>` FactoryBean等|pod 及之上扩展的deployment等|
|扩展|自定义namespace及NamespaceHandler|CRD|
|微创新|比如整合rabbitmq `<rabbit:template>`|etcd/redis operator|

## 原理




