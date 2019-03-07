---

layout: post
title: 容器日志和监控
category: 技术
tags: Kubernetes
keywords: kuberneteslog

---

## 简介（持续更新）

* TOC
{:toc}



## kubernetes 日志

Kubernetes 里面对容器日志的处理方式，都叫作 cluster-level-logging，即：这个日志处理系统，与容器、Pod 以及 Node 的生命周期都是完全无关的。这种设计当然是为了保证，无论是容器挂了、Pod 被删除，甚至节点宕机的时候，应用的日志依然可以被正常获取到。

1. 第一种，在 Node 上部署 logging agent，将日志文件转发到后端存储里保存起来。
2. 当容器的日志只能输出到某些文件里的时候，我们可以通过一个 sidecar 容器把这些日志文件重新输出到 sidecar的 stdout 和 stderr 上，这样就能够继续使用第一种方案了。
3. 通过一个 sidecar 容器，直接把应用的日志文件发送到远程存储里面去

### 一些观点

[猪八戒网DevOps容器云与流水线](http://mp.weixin.qq.com/s?__biz=MzA5OTAyNzQ2OA==&mid=2649699681&idx=1&sn=9f26d3dc8564fd31be93dead06489a6b&chksm=88930a02bfe48314e1e37873850010656d87650d0adcb1738049638cffb7e6496476b0cc8bac&mpshare=1&scene=23&srcid=121648JGw0qJ73GJs4ZJcIuY#rd)

![](/public/upload/docker/docker_log_collect.PNG)

比较常见的有这么几种，可能也有项目日志直接写入ES集群，不需要容器内收集的。

作者推荐使用第三种收集方案，以DaemonSet的方案部署日志收集组件，做到业务容器的完全无侵入，节省服务器资源，不必为每个业务容器都启动一个日志收集组件。





