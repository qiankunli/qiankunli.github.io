---

layout: post
title: 自己写分布式系统1——分析一个简单的分布式任务调度系统
category: 技术
tags: Architecture
keywords: 分布式系统

---

## 简介

最近研究一个系统设计方案，学习spark、storm等，包括跟同事交流，有以下几个感觉

1. 我们平时的系统其实也是分布式系统，若是归纳起来， 很多做法跟分布式系统差不多。比如你通过jdbc 访问mysql，spark 也是，spark rdd 做数据处理，我们又何尝不是。因此，特定的业务上，也没必要一定套spark、storm这些，系统的瓶颈有时也不是 spark、storm 可以解决的。
2. 笔者以前熟悉的项目，都是一个个独立的节点，节点是按功能划分的，谈不上主次，几个功能的节点组合形成架构。分布式系统也包括多个节点，但通常有Scheduler和Executor，业务功能都由Executor 完成，Scheduler 监控和调度Executor。
2. spark、storm 这些系统 一个很厉害的地方在于，抽象架设在分布式环境下。比如spark 的rdd，storm的topology/spout/bolt 这些。笔者以前的业务系统也有抽象，但抽象通常在单机节点内。
3. 部署方式上，也跟笔者熟悉的tomcat、springboot jar 有所不通
	
	1. 代码本身是一个进程，即定了main 函数
	2. 通常有一个额外的提交工作比如spark-submit 等

因此笔者萌生了一个想法， 自己动手写一个最简单的 分布式程序，学一学其中的套路。

首先，先看下一个简单的 分布式程序的源码 [ltsopensource/light-task-scheduler](https://github.com/ltsopensource/light-task-scheduler)

## 分析（未完成）