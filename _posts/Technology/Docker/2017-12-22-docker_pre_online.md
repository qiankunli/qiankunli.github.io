---

layout: post
title: 线上用docker要解决的问题
category: 技术
tags: Docker
keywords: Docker,macvlan

---


## 简介（持续更新）

* TOC
{:toc}

本文的目的在于梳理脉络，具体的细节会随着实践的深入逐步铺开。

## docker的应用路线图

笔者在 [测试环境docker化实践](http://qiankunli.github.io/2017/03/29/docker_test_environment_practice.html) 提到了测试环境运行docker化的一些实践，一年多以来，对docker以及 docker实施有了很多新的认识：

1. docker 本身的落地 涉及到 网络、编排工具选型 等问题，这个问题在18年已经逐步成熟并清晰。
2. 仅仅是在一个集群上搭建和运行docker/k8s 等，还是非常初级的。推广容器平台从某种程度上讲，自身是一个ToB的业务。你不仅要能用，还得好用，这个好用体现在

	* 要形成一个完整的持续交付体系 [docker环境下的持续构建](http://qiankunli.github.io/2018/11/18/ci_in_docker.html) 以及 [docker环境下的持续发布](http://qiankunli.github.io/2018/11/18/cd_in_docker.html)
	* 对开发同学的友好性 
	* 与现有基础设施的兼容性

3. 实现了前面几点，也只是解决了有无问题，若要在线上使用docker，很多测试环境不太在意的事便必须严肃对待。

线上用docker有哪些风险，要做哪些工作呢？

[Docker在沪江落地的实践](https://hujiangtech.github.io/tech/%E5%90%8E%E7%AB%AF/2017/03/21/Docker.html)


## docker 与 java的亲和性

[docker 环境（主要运行java项目）常见问题](http://qiankunli.github.io/2017/08/25/docker_debug.html)

## docker build 比较慢

[docker环境下的持续构建](http://qiankunli.github.io/2018/11/18/ci_in_docker.html) 
 
## io隔离性

[美团点评Docker容器管理平台](https://mp.weixin.qq.com/s?__biz=MjM5NjQ5MTI5OA==&mid=2651746030&idx=3&sn=f0c97665bb35aca7bc054e9d230baae7&chksm=bd12b7a38a653eb5aca4ca366abee24bad89d1bfab9031e5bf859d15f38d92d6d0755beca225&scene=21#wechat_redirect)

1. 一个服务狂写日志 导致整个磁盘被占满
2. 一个服务狂发数据 导致网卡被打满

## 性能不及物理机

容器平台性能，主要包括两个方面性能：

1. 业务服务运行在容器上的性能。
2. 容器操作（创建、删除等等）的性能。比如 如何加快镜像分发？

如何科学的设置容器的资源配置？

## 容器状态监控

[适配多种监控服务的容器状态采集](https://mp.weixin.qq.com/s?__biz=MjM5NjQ5MTI5OA==&mid=2651746030&idx=3&sn=f0c97665bb35aca7bc054e9d230baae7&chksm=bd12b7a38a653eb5aca4ca366abee24bad89d1bfab9031e5bf859d15f38d92d6d0755beca225&scene=21#wechat_redirect)

## 服务画像

[美团容器平台架构及容器技术实践](https://mp.weixin.qq.com/s?__biz=MjM5NjQ5MTI5OA==&mid=2651749434&idx=1&sn=92dcd59d05984eaa036e7fa804fccf20&chksm=bd12a5778a652c61f4a181c1967dbcf120dd16a47f63a5779fbf931b476e6e712e02d7c7e3a3&mpshare=1&scene=23&srcid=11183r23mQDITxo9cBDHbWKR%23rd)

通过对服务容器实例运行指标的搜集和统计，更好的完成调度容器、优化资源分配。比如可以根据某服务的容器实例的CPU、内存、IO等使用情况，来分辨这个服务属于计算密集型还是IO密集型服务，在调度时尽量把互补的容器放在一起。

## 推广难题


[美团容器平台架构及容器技术实践](https://mp.weixin.qq.com/s?__biz=MjM5NjQ5MTI5OA==&mid=2651749434&idx=1&sn=92dcd59d05984eaa036e7fa804fccf20&chksm=bd12a5778a652c61f4a181c1967dbcf120dd16a47f63a5779fbf931b476e6e712e02d7c7e3a3&mpshare=1&scene=23&srcid=11183r23mQDITxo9cBDHbWKR%23rd)

容器有如下优势：

1. 轻量级：容器小、快，能够实现秒级启动。
2. 应用分发：容器使用镜像分发，开发测试容器和部署容器配置完全一致。
3. 弹性：可以根据CPU、内存等资源使用或者QPS、延时等业务指标快速扩容容器，提升服务能力。进而减少资源预留。
4. Docker对CPU、内存等资源管理是通过Linux的CGroup实现的，调整配置只需要修改容器的CGroup参数，不必重启容器。对于一些有状态的业务，例如数据库和缓存，运行时调整CPU、内存和磁盘。 [美团点评Docker容器管理平台](https://mp.weixin.qq.com/s?__biz=MjM5NjQ5MTI5OA==&mid=2651746030&idx=3&sn=f0c97665bb35aca7bc054e9d230baae7&chksm=bd12b7a38a653eb5aca4ca366abee24bad89d1bfab9031e5bf859d15f38d92d6d0755beca225&scene=21#wechat_redirect)


推广容器平台从某种程度上讲，自身是一个ToB的业务，首先要有好的产品。这个产品要能和客户现有的系统很好的进行集成，而不是让客户推翻所有的系统重新再来。要提供良好的客户支持，（即使有些问题不是这个产品导致的也要积极帮忙解决）。

本文开头，笔者梳理了一个docker应用路线图，docker以及k8s 是一个入门门槛很高的技术，倒不是本身的知识点难学，而是其处于paas层，paas层的变动会影响一个公司的方方面面，持续交付体系、日志收集监控体系等，这些事情做不好或者做一半，很容易给人用了docker之后更麻烦的感觉。


## 支持微服务架构的设计

[美团点评Docker容器管理平台](https://mp.weixin.qq.com/s?__biz=MjM5NjQ5MTI5OA==&mid=2651746030&idx=3&sn=f0c97665bb35aca7bc054e9d230baae7&chksm=bd12b7a38a653eb5aca4ca366abee24bad89d1bfab9031e5bf859d15f38d92d6d0755beca225&scene=21#wechat_redirect)


1. 一个容器除运行服务外，还需要运行多个旁路系统 [一个容器多个进程](http://qiankunli.github.io/2018/11/06/multi_process_per_container.html)
2. 对日志采集的兼容


## 镜像的清理

起初，集群规模较小，harbor 和其它服务部署在一起。后来镜像文件 越来越多，于是将harbor 单独部署。但必然，单机磁盘无法承载 所有镜像，此时

1. 集群改造
2. 清理过期不用的镜像

	清理时，应根据当前服务使用镜像的情况，保留之前几个版本。笔者曾粗暴的清理了一些镜像，刚好赶上断网，直接导致部分服务重新部署时拉不到镜像。

## 网络方案选择与多机房扩展问题

