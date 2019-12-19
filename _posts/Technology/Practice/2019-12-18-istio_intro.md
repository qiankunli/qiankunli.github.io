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

[浅谈Service Mesh体系中的Envoy](https://yq.aliyun.com/articles/606655)


## 其它

任何软件架构设计，其核心都是围绕数据展开的，基本上如何定义数据结构就决定了其流程的走向，剩下的不外乎加上一些设计手法，抽离出变与不变的部分，不变的部分最终会转化为程序的主流程，基本固化，变的部分尽量保证拥有良好的扩展性、易维护性，最终会转化为主流程中各个抽象的流程节点。

