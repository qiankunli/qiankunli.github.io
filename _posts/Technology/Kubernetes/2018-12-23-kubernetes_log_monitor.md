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

## docker 日志

对于一个容器来说，当应用把日志输出到 stdout 和 stderr 之后，容器项目在默认情况下就会把这些日志输出到宿主机上的一个 JSON 文件里

[「Allen 谈 Docker 系列」之 docker logs 实现剖析](http://blog.daocloud.io/allen_docker01/)

对于应用的标准输出(stdout)日志，Docker Daemon 在运行这个容器时就会创建一个协程(goroutine)，负责标准输出日志。由于此 goroutine 绑定了整个容器内所有进程的标准输出文件描述符，因此容器内应用的所有标准输出日志，都会被 goroutine 接收。goroutine 接收到容器的标准输出内容时，立即将这部分内容，写入与此容器—对应的日志文件中，日志文件位于`/var/lib/docker/containers/<container_id>`，文件名为<container_id>-json.log。

![](/public/upload/docker/docker_log.png)

Docker 则通过 docker logs 命令向用户提供日志接口。`docker logs` 实现原理的本质均基于与容器一一对应的 <container-id>-json.log，`kubectl logs`类似

从这可以看到几个问题

1. app 同时输出文件日志和stdout 是一种浪费
2. stdout 日志在 `/var/lib/docker/containers/<container_id>` 下可以被清理， 也可以配置 docker daemon 设置 log-driver 和 log-opts 参数

		 "log-driver":"json-file",
	  	 "log-opts": {"max-size":"500m", "max-file":"3"}
	  	 
3. 将日志输出到stdout 貌似是容器环境下的方案，这与物理机时代非常不同
4. 你如何限定开发小伙伴不向文件写日志? 限定写文件权限

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





