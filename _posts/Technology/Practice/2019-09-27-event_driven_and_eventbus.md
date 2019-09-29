---

layout: post
title: 事件驱动和事件总线
category: 架构
tags: Practice
keywords: window

---

## 简介（未完成）

* TOC
{:toc}

## 事件驱动

## 事件总线

### 按特性分

按支持的特性分为几种方式

1. 向所有订阅者发送所有事件
2. 向所有订阅者发送所有事件影子，事件本身则持久化
3. 向经过过滤的订阅者发送所有事件影子
4. 按顺序传递事件

### thread mode

## 实现

### guava evnentbus

### EventBus

[greenrobot/EventBus](https://github.com/greenrobot/EventBus)

