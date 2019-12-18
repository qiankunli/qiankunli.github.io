---

layout: post
title: istio学习
category: 架构
tags: Practice
keywords: window

---

## 简介（未完成）

* TOC
{:toc}


## 整体架构

![](/public/upload/practice/istio.png)

## Envoy

Envoy 是 Istio 中最基础的组件，所有其他组件的功能都是通过调用 Envoy 提供的 API，在请求经过 Envoy 转发时，由 Envoy 执行相关的控制逻辑来实现的。

[浅谈Service Mesh体系中的Envoy](http://jm.taobao.org/2018/07/05/Mesh%E4%BD%93%E7%B3%BB%E4%B8%AD%E7%9A%84Envoy/)


