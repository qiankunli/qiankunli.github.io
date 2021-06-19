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
[kubernetes 平台开发者的几个小技巧](https://mp.weixin.qq.com/s/RVYJd_3xzDps-1xFwtl01g)
[内存回收导致关键业务抖动案例分析-论云原生OS内存QoS保障](https://mp.weixin.qq.com/s/m74OgseP3I9AIKvPP6exrg)
[去哪儿容器化落地过程踩过的那些坑](https://mp.weixin.qq.com/s/TEHKO9M1BdkQre2IrIQUlA)Qunar 在做容器化过程中，各个系统 portal 平台、中间件、ops 基础设施、监控等都做了相应的适配改造
1. portal：Qunar 的 PAAS 平台入口，提供CI CD 能力、资源管理、自助运维、应用画像、应用授权(db授权、支付授权、应用间授权)等功能
2. 运维工具：提供应用的可观测性工具, 包括 watcher（监控和报警）、bistoury （java 应用在线 debug）、qtrace （tracing 系统）, loki/elk （提供实时日志/离线日志查看）
3. 中间件：应用用到的所有中间件, mq、配置中心、分布式调度系统 qschedule、dubbo 、mysql sdk等
4. 虚拟化集群：底层的 k8s 和 openstack 集群，多k8s集群管理工具 kubesphere
5. noah：测试环境管理平台，支持应用 kvm / 容器混合部署

![](/public/upload/kubernetes/qunar_overview.png)
