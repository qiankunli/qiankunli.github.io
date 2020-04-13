---

layout: post
title: 容器日志采集
category: 技术
tags: Container
keywords: container log collect

---

## 简介

* TOC
{:toc}

[直击痛点，详解 K8s 日志采集最佳实践](https://mp.weixin.qq.com/s/PVYbdryPvSegWGdB7evBZg)

## 采集方式

![](/public/upload/container/collect_log.png)

## 日志输出

容器提供标准输出和文件两种方式，

1. 在容器中，标准输出将日志直接输出到 stdout 或 stderr，实际的业务场景中建议大家尽可能使用文件的方式
    1. Stdout 性能问题，从应用输出 stdout 到服务端，中间会经过好几个流程（例如普遍使用的 JSON LogDriver）：应用 stdout -> DockerEngine -> LogDriver -> 序列化成 JSON -> 保存到文件 -> Agent 采集文件 -> 解析 JSON -> 上传服务端。整个流程相比文件的额外开销要多很多，在压测时，每秒 10 万行日志输出就会额外占用 DockerEngine 1 个 CPU 核；
    2. Stdout 不支持分类，即所有的输出都混在一个流中，无法像文件一样分类输出，通常一个应用中有 AccessLog、ErrorLog、InterfaceLog（调用外部接口的日志）、TraceLog 等，而这些日志的格式、用途不一，如果混在同一个流中将很难采集和分析；
    3. Stdout 只支持容器的主程序输出，如果是 daemon/fork 方式运行的程序将无法使用 stdout；
    4. 文件的 Dump 方式支持各种策略，例如同步/异步写入、缓存大小、文件轮转策略、压缩策略、清除策略等，相对更加灵活。
2. 日志打印到文件的方式和虚拟机/物理机基本类似，只是日志可以使用不同的存储方式，例如默认存储、EmptyDir、HostVolume、NFS 等。

## 采集什么

![](/public/upload/container/collect_what.png)

1. 容器文件，比如容器运行了Tomcat，则Tomcat 的启动日志也在采集范围之内
2. 容器 Stdout
3. 宿主机文件
4. Journal
5. Event 

[使用日志服务进行Kubernetes日志采集](https://help.aliyun.com/document_detail/87540.html)其它

1. 支持多种采集部署方式，包括 DaemonSet、Sidecar、DockerEngine LogDriver 等；
2. 支持对日志数据进行富化，包括附加 Namespace、Pod、Container、Image、Node 等信息；
3. 稳定、高可靠，基于阿里自研的 Logtail 采集 Agent 实现，目前全网已有几百万的部署实例；
4. 基于 CRD 进行扩展，可使用 Kubernetes 部署发布的方式来部署日志采集规则，与 CICD 完美集成。

[9 个技巧，解决 K8s 中的日志输出问题](https://mp.weixin.qq.com/s/fLNzHS_6V78pSJ_zqTWhZg)










