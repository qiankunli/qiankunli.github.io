---

layout: post
title: Scheduler扩展
category: 架构
tags: Kubernetes
keywords: Scheduler Extender
---

## 简介

* TOC
{:toc}





[Scheduler extender](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/scheduling/scheduler_extender.md) 扩展Scheduler 的三种方式

1. by adding these rules to the scheduler and recompiling, [described here](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-scheduling/scheduler.md) 改源码
2. implementing your own scheduler process that runs instead of, or alongside of, the standard Kubernetes scheduler,  另写一个scheduler
3. implementing a "scheduler extender" process that the standard Kubernetes scheduler calls out to as a final pass when making scheduling decisions. 给默认Scheduler 做参谋长

This approach is needed for use cases where scheduling decisions need to be made on resources not directly managed by the standard Kubernetes scheduler. The extender helps make scheduling decisions based on such resources. (Note that the three approaches are not mutually exclusive.) 第三种方案一般用在 调度决策依赖于 非默认支持的资源的场景


## 示例实现

