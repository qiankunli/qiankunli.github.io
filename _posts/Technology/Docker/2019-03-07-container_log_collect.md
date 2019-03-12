---

layout: post
title: 容器日志采集
category: 技术
tags: Docker
keywords: jib

---

## 简介（未完成）

* TOC
{:toc}

参见 [容器狂打日志怎么办？](http://qiankunli.github.io/2019/03/05/container_log.html)


## kubernetes 日志

Kubernetes 里面对容器日志的处理方式，都叫作 cluster-level-logging，即：这个日志处理系统，与容器、Pod 以及 Node 的生命周期都是完全无关的。这种设计当然是为了保证，无论是容器挂了、Pod 被删除，甚至节点宕机的时候，应用的日志依然可以被正常获取到。

1. 第一种，在 Node 上部署 logging agent，将日志文件转发到后端存储里保存起来。
2. 当容器的日志只能输出到某些文件里的时候，我们可以通过一个 sidecar 容器把这些日志文件重新输出到 sidecar的 stdout 和 stderr 上，这样就能够继续使用第一种方案了。
3. 通过一个 sidecar 容器，直接把应用的日志文件发送到远程存储里面去

## 采集方式

1. 应用直接将数据发往监控系统的收集服务或消息队列
2. 应用将数据发往本地的一个agent
3. 应用将数据以日志形式写到磁盘，用一个本地agent实时的读取日志。这种方式将文件系统作为一个稳定的数据缓存，可以很好的保证数据完整性。当agent 重启或其它原因导致数据丢失时，可以简单地从之前断掉的点重新读取日志内容。迁移到k8s后，这个agent 自然地对应一个DaemonSet

## 一些观点

[猪八戒网DevOps容器云与流水线](http://mp.weixin.qq.com/s?__biz=MzA5OTAyNzQ2OA==&mid=2649699681&idx=1&sn=9f26d3dc8564fd31be93dead06489a6b&chksm=88930a02bfe48314e1e37873850010656d87650d0adcb1738049638cffb7e6496476b0cc8bac&mpshare=1&scene=23&srcid=121648JGw0qJ73GJs4ZJcIuY#rd)

![](/public/upload/docker/docker_log_collect.PNG)

比较常见的有这么几种，可能也有项目日志直接写入ES集群，不需要容器内收集的。

作者推荐使用第三种收集方案，以DaemonSet的方案部署日志收集组件，做到业务容器的完全无侵入，节省服务器资源，不必为每个业务容器都启动一个日志收集组件。




