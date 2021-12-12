---

layout: post
title: 在离线业务混部
category: 架构
tags: Kubernetes
keywords:  Kubernetes 混部

---

## 简介（未完成）

* TOC
{:toc}

[数据中心日均 CPU 利用率 45% 的运行之道--阿里巴巴规模化混部技术演进](https://mp.weixin.qq.com/s?__biz=MzUzNzYxNjAzMg==&mid=2247483986&idx=1&sn=44e9ad3c4bc4529a79547ba506773881&chksm=fae5099dcd92808b9af6e8f28a661b8c16284efb4656131479d21e9092922b03728c1140042c&mpshare=1&scene=23&srcid=%23rd)
1. 在线离线混部/高低优先级业务混部
2. 

[阿里大规模业务混部下的全链路资源隔离技术演进](https://mp.weixin.qq.com/s/_DTQ4Q2dC-kN3zyozGf9QA)

在线资源通常要给一个预留，预留的部分就是浪费的部分，所以要挖掘在线业务占用的资源给离线业务用，但是要注意：资源隔离，在线随时可以抢占离线。

## 操作系统级/资源隔离

看着要动操作系统的样子。

## 调度层

[百度混部实践：如何提高 Kubernetes 集群资源利用率？](https://mp.weixin.qq.com/s/12XFN2lPB3grS5FteaF__A)

[历经 7 年双 11 实战，阿里巴巴是如何定义云原生混部调度优先级及服务质量的？](https://mp.weixin.qq.com/s/GrgWzxAfHe2Ml4biwai8eQ)在这些在线和离线的 Pod 之间，我们就需要用不同的调度优先级和服务质量等级，以满足在线和离线的实际运行需求。

### Priority 和 Qos

![](/public/upload/kubernetes/priority_vs_qos.png)