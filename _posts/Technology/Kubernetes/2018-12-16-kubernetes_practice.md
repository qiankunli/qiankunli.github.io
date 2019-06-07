---

layout: post
title: kubernetes实践
category: 技术
tags: Kubernetes
keywords: kubernetes yaml

---

## 简介（持续更新）

* TOC
{:toc}

## 总纲

两个基本工作

1. 应用容器化

    1. 
    2. 
    3. 
    4. 
2. 编排自动化

    1. 
    2. 
    3. 
    4. 

## “划一片地”来测试

假设集群一共有10个节点，我们希望一般任务只调度到其中的8多个节点，另外2个节点用来试验一些新特性。

此时便可以采用Taints and Tolerations 特性

### 默认的都匹配 + 特征匹配 node 加label，pod 加nodeSelector/nodeAffinity

[Assigning Pods to Nodes](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#node-affinity-beta-feature)

### 默认的都不匹配 + “特征”匹配 node 加 Taints，pod 加tolerations

Node affinity, described [here](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#node-affinity-beta-feature), is a property of pods that attracts them to a set of nodes (either as a preference or a hard requirement). Taints are the opposite – they allow a node to repel（击退） a set of pods.

[Taints and Tolerations](https://kubernetes.io/docs/concepts/configuration/taint-and-toleration/)

给节点加Taints

    kubectl taint nodes node1 key=value:NoSchedule

 NoSchedule 是一个effect. This means that no pod will be able to schedule onto node1 unless it has a matching toleration.

给pod 加加tolerations

    tolerations:
    - key: "key"
    operator: "Equal"
    value: "value"
    effect: "NoSchedule"

## 另类“Service”——给服务一个稳定的主机名

容器的ip 总是经常变，针对这个问题， k8s早已有一系列解决方案。

1. k8s 套件内

    1. k8s集群内，Service等
    2. k8s集群外，Nodeport，ingress等
2. 提供一个dns服务器，维护`<容器名,ip>` 映射
3. 改写ipam插件，支持静态ip

## web界面管理

[Qihoo360/wayne](https://github.com/Qihoo360/wayne) Wayne 是一个通用的、基于 Web 的 Kubernetes 多集群管理平台。通过可视化 Kubernetes 对象模板编辑的方式，降低业务接入成本， 拥有完整的权限管理系统，适应多租户场景，是一款适合企业级集群使用的发布平台。

## 扩容缩容

[容器化在一下科技的落地实践](http://www.10tiao.com/html/217/201811/2649699541/1.html)

HPA 扩容缩容范围

健康检查

1. 存活检查
2. 就绪检查

[荔枝运维平台容器化实践](https://mp.weixin.qq.com/s/Q4t5IptqQmQZ6z4vOIhcjQ) 从打包、监控、日志、网络、存储各方面阐述了一下，还比较全面

## 工作流

线上环境上线的镜像是已经上线到测试环境的相同镜像。

笔者个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)
