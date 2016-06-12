---

layout: post
title: google guava的一些理解
category: 技术
tags: Java
keywords: JTA TCC 

---

## 简介（待完善）

严重推进的文档——官方wiki：`https://github.com/google/guava/wiki`，部分中文翻译问题比较大。

## Future

A traditional Future represents the result of an asynchronous computation: a computation that may or may not have finished producing a result yet. 

一个同步操作往往会有一个结果（立即返回，或阻塞一段后返回），异步操作会立即返回一个结果（不是业务上的结果），确切的说是一个结果的Holder，即Future，异步操作的发起方和执行方约定在这个Future中存放和获取结果。

## 系统的理下异步执行框架

调用线程和执行线程（执行线程可以是一个线程，可以是一个线程池）

异步批量执行


异步执行框架在rpc（网络io）和线程方面的不同实现。