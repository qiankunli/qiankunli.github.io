---

layout: post
title: Kubernetes webhook
category: 架构
tags: Kubernetes
keywords:  Kubernetes event

---

## 简介

* TOC
{:toc}

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
3. 提高资源利用率 [容器化计算资源利用率现象剖析](https://mp.weixin.qq.com/s/8sHsI1pVm-1RX5w1F3uWPg) [资源利用率提升工具大全](https://mp.weixin.qq.com/s/tjpSneIghbGlRpAg1qkhHA)

[美团点评Kubernetes集群管理实践](https://mp.weixin.qq.com/s/lYDYzEUlvXQhCO1xCJ7HAg) 笔者从中得到一个启发就是，整个kubernetes 的实践是分层次的。

![](/public/upload/kubernetes/meituan_kubernetes_practice.png)

[网易数帆云原生故障诊断系统实践与思考](https://zhuanlan.zhihu.com/p/347629491) PS： 我们自己wrench工具的分布式版
1. 规范用户的使用方式，容器 vs 虚拟机
2. 部分集群连接的 APIServer 客户端数量超过了 4000 个，其中不乏一些用户用脚本对 Pod 资源进行全量 LIST 来获取数据。这些集群的 APIServer 消耗接近 100G 的内存以及 50 核的 CPU 算力，并且 APIServer 所在节点的网卡流量达到了 15G。
3. 明确集群稳定性保障以及应用稳定性保障的边界以及有效的评估模型，这种责任边界的不明确带来了交付成本上的增长以及不确定性。PS：出问题无法明确是应用的问题还是k8s的问题

## 排查问题文章汇总

[kubernetes 问题排查: 磁盘 IO 过高导致 Pod 创建超时](https://mp.weixin.qq.com/s/3v84M5idGi-nJ5u8RUzP6A)

