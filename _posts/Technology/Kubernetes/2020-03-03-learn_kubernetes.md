---

layout: post
title: 如何学习Kubernetes
category: 技术
tags: Kubernetes
keywords: kubernetes yaml

---

## 简介

* TOC
{:toc}

[Kubernetes 学习路径](https://www.infoq.cn/article/9DTX*1i1Z8hsxkdrPmhk)

所谓学习，就是要能回答几个基本问题：

1. [kubectl 创建 Pod 背后到底发生了什么？](https://mp.weixin.qq.com/s/ctdvbasKE-vpLRxDJjwVMw) 这个问题 对于一个java 工程师来说，就像“在浏览器里输入一个url之后发生了什么？” 一样基础。再进一步就是 自己从kubectl 到crd 实现一遍。

2. [github kubernetes community](https://github.com/kubernetes/community/tree/8decfe42b8cc1e027da290c4e98fa75b3e98e2cc/contributors/devel)

好文章在知识、信息概念上一定是跨层次的，既有宏观架构，又有微观佐证。只有原理没有代码总觉得心虚，只有代码没有原理总觉得心累。**从概念过渡到实操，从而把知识点掌握得更加扎实**。

## 从kubernetes 看分布式系统实现

hadoop 是直接把jar 传输到目标节点，其实也可以学习 k8s， 调度工作只是更改etcd 里的资源状态（给pod的nodeName赋值），然后kubelet 自己监控并拉数据处理。


## 通用实现

除apiserver/kubectl 之外（kubelet 类似，但更复杂些），与api server 的所有组件Controller/Scheduler 的业务逻辑几乎一致

1. 组件需要与apiserver 交互，但核心功能组件不直接与api-server 通信，而是抽象了一个Informer 负责apiserver 数据的本地cache及监听。Informer 还会比对 资源是否变更（依靠内部的Delta FIFO Queue），只有变更的资源 才会触发handler。**因为Informer 如此通用，所以Informer 的实现在 apiserver 的 访问包client-go 中**。*在k8s推荐的官方java库中，也支持直接创建Informer 对象*。PS：Informer 对应java+zk 系就是curator
2. 组件 全部采用control loop 逻辑
3. 组件 全部内部维护一个 queue队列，通过注册Informer事件 函数保持 queue数据的更新 或者说 作为队列的生产者，control loop 作为队列的消费者。
4. 通过Informer 提供过的Lister 拥有遍历数据的能力，将操作结果 重新通过kubeclient 写入到apiserver 

![](/public/upload/kubernetes/component_overview.png)

## 一个充分支持扩展的系统

![](/public/upload/kubernetes/kubernetes_extension.png)

