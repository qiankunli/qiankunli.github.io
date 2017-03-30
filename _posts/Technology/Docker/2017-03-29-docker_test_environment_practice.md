---

layout: post
title: 测试环境docker化实践
category: 技术
tags: Docker
keywords: Docker,macvlan

---


## 简介

为什么要docker化？

1. 标准化，以部署tomcat为例，一个项目一个主机tomcat,tomcat永远位于`/usr/local/tomcat`（或其它你喜欢的位置）下，对外端口是8080，debug端口是8000.
2. api化，通过api接口操作项目的部署（cpu、内存分配、机器分配、实例数管理等），而不是原来物理机环境的的手工命令行操作。
3. 自动化，调度系统可以根据api进行一些策略性的反应，比如自动扩容缩容。

上述工作，基于物理机环境不是不可以做，可是太麻烦，可用性和扩展性都不够好。

||网段|对外抽象|
|---|---|---|
|基本环境：物理环境|192.168.0.0/16|一台台互联互通的物理机，大部分要手工|
|目标：容器环境|172.30.0.0/16|marathon标准化的api，大部分可以自动化|

基于n台物理机搭建容器环境，整个工作的主线：一个项目一个主机 ==> 物理机资源不够 ==> 虚拟化 ==> 轻量级虚拟化 ==> docker ==> 针对docker容器带来的网络、存储等问题 ==> 集群编排 ==> 对CI/CD的影响。

在整个docker化的过程中，笔者认为比较重要的是网络，网络部分参见[docker macvlan实践](http://qiankunli.github.io/2017/01/13/docker_macvlan.html)

## 编排

docker解决了单机的虚拟化，但当一个新部署任务到达，由集群中的哪一个docker执行呢？因此需要一个编排工具，实现集群的资源管理和任务调度。

||优缺点|
|---|---|
|swarm/swarm mode|docker原生，但目前更多是一个docker任务分发工具；换句话说，作为docker分发工具是够用的，但作为集群资源管理和任务调度工具是勉强的|
|k8s|k8s提供的pod、service、replicaController简化了一些问题，但使用起来也相对复杂|
|mesos + marathon（本文采用）|在docker管理和分布式资源管理之间，找到了一个比较好的平衡点|

其实这些工具的根本区别就是：

1. 从一个docker/容器化调度工具， 扩展成一个分布式集群管理工具
2. 从一个分布式资源管理工具 ，增加支持docker的feature

## image的组织

|镜像名|功能|
|---|---|
|alpine|base image|
|alpine+|加上ssh、一些基本的命令|
|jdk6/7/8|新增jdk|
|tomcat6/tomcat7/tomcat8|新增tomcat|

## CI

本质上jenkins如何跟marathon结合的问题。


## 碰到的问题

mesos + marathon + docker的文章很多，其实这才是本文的重点。

1. Base image的影响

  1. 时区、tomcat PermGensize、编码等参数值的修正
  2. base image为了尽可能精简，使用了alpine。其一些文件的缺失，导致一些java代码无法执行。比如，当去掉`/etc/hosts`中ip和容器主机名的映射后，加上`/etc/sysconfig/network`的缺失，导致`InetAddress.getLocalHost()`执行失败。参见[ava InetAddress.getLocalHost() 在linux里实现](http://blog.csdn.net/raintungli/article/details/8191701)
		
2. Safe shutdown以及添加sshd对safe shutdown的影响（需要通过supervisord传导SIGTERM信号）
3. Marathon多机房主备问题
4. ip/hostname变化对服务治理系统、nginx的影响

	1. 比如我司服务发现系统认为`192.168.0.0/16`才是合法网段，导致拥有`172.31.0.0/16`网段ip的服务无法工作

5. 容器的漂移对日志采集、分析系统的影响
6. 对容器提供dns服务，以使其可以正确解析外部服务的hostname
7. 如何更好的推广与应用的问题（这是个大问题，包括分享ppt的写作思路、一些模板的创建等，不比解决技术难题耗费的精力少）
8. 这还只是测试环境，很多问题可以简单粗暴的解决。但到了线上，仅保证可用性就可以再翻一倍的工作量。

## 解决ip变化带来的影响

主要有以下方案

1. 新增组件屏蔽ip变化
2. 提供dns服务（有缓存和多实例问题）
3. 服务定死ip（这个方案非常不优雅）

容器环境运行web服务时，nginx如何感知web服务ip的变化？参见一个大牛的工具[weibocom/nginx-upsync-module](https://github.com/weibocom/nginx-upsync-module)，我为大牛的工具新增了zk支持，参见[qiankunli/nginx-upsync-module-zk](https://github.com/qiankunli/nginx-upsync-module-zk)

## 引用

[Docker 在 Bilibili 的实战：由痛点推动的容器化](http://dockone.io/article/2023)



	
	