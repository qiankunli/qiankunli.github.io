---

layout: post
title: 以应用为中心
category: 架构
tags: Architecture
keywords: application

---

## 简介（未完成）

* TOC
{:toc}


[解读容器 2019：把“以应用为中心”进行到底](https://www.kubernetes.org.cn/6408.html)云原生的本质是一系列最佳实践的结合；更详细的说，云原生为实践者指定了一条低心智负担的、能够以可扩展、可复制的方式最大化地利用云的能力、发挥云的价值的最佳路径。这种思想，以一言以蔽之，就是“以应用为中心”。正是因为以应用为中心，云原生技术体系才会无限强调**让基础设施能更好的配合应用**、以更高效方式为应用“输送”基础设施能力，而不是反其道而行之。而相应的， Kubernetes 、Docker、Operator 等在云原生生态中起到了关键作用的开源项目，就是让这种思想落地的技术手段。


## k8s 的问题

### K8s 的 API 里其实并没有“应用”的概念

[给 K8s API “做减法”：阿里巴巴云原生应用管理的挑战和实践](https://mp.weixin.qq.com/s/u3CxlpBYk4Fw3L3l1jxh-Q)

Kubernetes API 的设计把研发、运维还有基础设施关心的事情全都糅杂在一起了。这导致研发觉得 K8s 太复杂，运维觉得 K8s 的能力非常凌乱、零散，不好管理。很多 PaaS 平台只允许开发填 Deployment 的极个别字段。为什么允许填的字段这么少？是平台能力不够强吗？其实不是的，本质原因在于业务开发根本不想理解这众多的字段。归根到底，Kubernetes 是一个 Platform for Platform 项目，**它的设计是给基础设施工程师用来构建其他平台用的（比如 PaaS 或者 Serverless），而不是直面研发和运维同学的**。从这个角度来看，Kubernetes 的 API，其实可以类比于 Linux Kernel 的 System Call，这跟研发和运维真正要用的东西（Userspace 工具）完全不是一个层次上的。

### K8s 实在是太灵活了，插件太多了，各种人员开发的 Controller 和 Operator 也非常多


## Open Application Model （OAM）

[给 K8s API “做减法”：阿里巴巴云原生应用管理的挑战和实践](https://mp.weixin.qq.com/s/u3CxlpBYk4Fw3L3l1jxh-Q)

## 未来

在下一代“以应用为中心”的基础设施当中，**业务研发通过声明式 API 定义的不再是具体的运维配置（比如：副本数是 10 个），而是“我的应用期望的最大延时是 X ms”**。接下来的事情，就全部会交给 Kubernetes 这样的应用基础设施来保证实际状态与期望状态的一致，这其中，副本数的设置只是后续所有自动化操作中的一个环节。只有让 Kubernetes 允许研发通过他自己的视角来定义应用，而不是定义 Kubernetes API 对象，才能从根本上的解决 Kubernetes 的使用者“错位”带来的各种问题。