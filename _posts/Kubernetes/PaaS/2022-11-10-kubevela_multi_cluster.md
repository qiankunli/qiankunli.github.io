---

layout: post
title: kubevela多集群
category: 架构
tags: Kubernetes
keywords:  kubevela cue

---

## 简介（未完成）

* TOC
{:toc}

kubevela 与 常规的多集群管理工具 clusternet 与karmada 有以下区别 

1. 一般的多集群方案 都尽量避免用户感知到 api 的变化，比如用户 想向多集群 提交一个deployment，由多集群控制器负责 将deployment 分发到不同的集群上。为了 让deployment 提交后不被 deployment controller 直接感知并立即被处理，clusternet 选择更改 用户提交的deployment 的group，karmada 选择直接 把用户提交的deployment 存放在一个独立的apiserver上。对于kubevela 来说，本来用户提交的就是 Application，就省去了 clusternet 和 karmada 所面对的问题。
2. kubevela 目前的多集群处理 相对简单，更多是把 workload 发布到 多个集群上 作为workflow的step，不像karmada 等还要进行复杂的多集群调度、重调度等工作（当然，后续可能也会有）。