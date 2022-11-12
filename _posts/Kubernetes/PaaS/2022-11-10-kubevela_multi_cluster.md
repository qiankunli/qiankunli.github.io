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

一般的多集群方案 都尽量避免用户感知到 api 的变化，比如用户过去向单集群 提交一个deployment，现在还是提交一个deployment，由多集群控制器负责 将deployment 分发到不同的集群上。为了 让deployment 提交后不被 deployment controller 直接感知掉并被处理，clusternet 选择更改