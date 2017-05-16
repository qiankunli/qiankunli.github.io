---

layout: post
title: 测试环境docker化实践
category: 技术
tags: Docker
keywords: Docker,macvlan

---


## 简介

为什么要docker化？

1. 标准化

	* 配置标准化，以部署tomcat为例，理想状态下：一个项目一个主机tomcat,tomcat永远位于`/usr/local/tomcat`（或其它你喜欢的位置）下，对外端口是8080，debug端口是8000.

	* 部署标准化，现在云平台越来越流行，同时，也不会立即丢弃物理环境，因此必然存在着同时向云环境和物理环境部署的问题。这就需要一系列工具，能够屏蔽物理环境和云环境的差异，docker在其中扮演重要角色。
	
2. api化，通过api接口操作项目的部署（cpu、内存分配、机器分配、实例数管理等），而不是原来物理机环境的的手工命令行操作。
3. 自动化，调度系统可以根据api进行一些策略性的反应，比如自动扩容缩容。

上述工作，原有的技术手段不是不可以做，可是太麻烦，可用性和扩展性都不够好。

## 几个小目标

1. 业务之间不互相干扰	* 一个项目/war一虚拟机/容器   * Ip-pert-task2. 容器之间、容器与物理机之间可以互通3. 容器编排：健康检查、容器调度等4. 使用方式：通过yaml/json来描述任务，通过api部署


||网段|对外抽象|
|---|---|---|
|基本环境：物理环境|192.168.0.0/16|一台台互联互通的物理机，大部分要手工|
|目标：容器环境|172.30.0.0/16|marathon标准化的api，大部分操作（deploy、scal等）可以自动化|

总结一下，基于n台物理机搭建容器环境，整个工作的主线：一个项目一个主机 ==> 物理机资源不够 ==> 虚拟化 ==> 轻量级虚拟化 ==> docker ==> 针对docker容器带来的网络、存储等问题 ==> 集群编排 ==> 对CI/CD的影响。

## 网络

1. Overlay
	
	* 隧道，通常用到一个专门的解封包工具
	* 路由，每个物理机充作一个路由器，对外，将容器数据路由到目标机器上；对内，将收到的数据路由到目标容器中。

   通常需要额外的维护物理机以及物理机上容器ip（段）的映射关系
2. Underlay，不准确的说，容器的网卡暴露在物理网络中，直接收发，通常由外部设备负责网络的连通性。

经过对比，我们采用了macvlan，一个简单，一个是效率高。关于macvlan，这涉及到lan ==> vlan => macvlan 的发展过程，请读者自行了解。网络部分参见[docker macvlan实践](http://qiankunli.github.io/2017/01/13/docker_macvlan.html)

## 编排

docker解决了单机的虚拟化，但当一个新部署任务到达，由集群中的哪一个docker执行呢？因此需要一个编排工具，实现集群的资源管理和任务调度。

||优缺点|
|---|---|
|swarm/swarm mode|docker原生，但目前更多是一个docker任务分发工具；换句话说，作为docker分发工具是够用的，但作为集群资源管理和任务调度工具是勉强的|
|k8s|k8s提供的pod、service、replicaController简化了一些问题，但使用起来也相对复杂|
|mesos + marathon（本文采用）|在docker管理和分布式资源管理之间，找到了一个比较好的平衡点|

就我的理解，其实这些工具的根本区别就是：

1. 从一个docker/容器化调度工具， 扩展成一个分布式集群管理工具
2. 从一个分布式资源管理工具 ，增加支持docker的feature

到目前为止，根据我们测试环境的实践，发现我司有以下特点

1. 对编排的需求很弱，基本都是单个项目的部署
2. 对web项目、rpc server项目得进行特殊处理，这种处理我们采用macvlan以及公司的一些技术应用有关系。

## image的组织

我们要对镜像的layer进行组织，以最大化的复用layer。

|镜像名|功能|
|---|---|
|alpine|base image|
|alpine+|一些基本的命令|
|jdk6/7/8、ssh|新增jdk|
|tomcat6/tomcat7/tomcat8|新增tomcat|

我们经过一段时间的使用，发现image的分发速度较慢，针对该问题主要由以下方案：

1. 京东对image layer进行压缩的方案
2. 阿里对镜像进行hotfix标识，对于这类镜像，不再创建新容器，而是更新容器

因为我们还只是在测试环境使用，镜像较慢的矛盾还不是太突出，因此这方面并没有做什么工作。

## CI

本质上jenkins如何跟marathon结合的问题，本文不再赘述。

## ip变化带来的问题

使用docker后，物理机的角色弱化成了：单纯的提供计算资源。容器可以在物理机之间自由漂移，但许多系统的正常运行依赖ip，ip不稳定带来一系列的问题。

解决ip的变化问题主要有以下方案

1. 新增组件屏蔽ip变化
2. 提供dns服务（有缓存和多实例问题）
3. 服务定死ip（这个方案非常不优雅）

对于web服务，ip的变化导致要经常更换nginx配置，为此，我们专门改写了一个nginx插件来解决这个问题。参见一个大牛的工具[weibocom/nginx-upsync-module](https://github.com/weibocom/nginx-upsync-module)，我为大牛的工具新增了zk支持，参见[qiankunli/nginx-upsync-module-zk](https://github.com/qiankunli/nginx-upsync-module-zk)

对于rpc服务，我司有自己独立开发的服务治理系统，实现服务注册和发现。但该系统有审核机制，系统认为服务部署在新的机器上时（通过ip识别新机器）应先审核才能对外使用。我们和开发同学协调，在服务上线时，调用了自动审核的接口，来屏蔽掉这个问题。

ip的经常变化，为开发童鞋的调试带来了很大不便。最初，我们提供了一个web console来访问容器，操作步骤为：login ==> find container ==> input console ==> op。但很多童鞋依然认为过于繁琐，并且web console的性能也不理想。而为每个容器配置ssh server，又会对safe shutdown等产生不良影响。因此

1. 登陆测试环境，90%是为了查看日志
2. 和开发约定项目的日志目录，并将其映射到物理机下
2. 每个物理机启动一个固定ip的ssh container，并映射日志目录
3. 使用go语言实现了一个essh工具，`essh -i marathon_app_name`即可访问对应的ssh container实例并查看日志。

当然，日志的问题，也可以通过elk解决，但这需要耗费一定的资源，所以还暂未安排。

## 其它问题

mesos + marathon + docker的文章很多，其实这才是本文的重点。

1. Base image的影响

  1. 时区、tomcat PermGensize、编码等参数值的修正
  2. base image为了尽可能精简，使用了alpine。其一些文件的缺失，导致一些java代码无法执行。比如，当去掉`/etc/hosts`中ip和容器主机名的映射后，加上`/etc/sysconfig/network`的缺失，导致`InetAddress.getLocalHost()`执行失败。参见[ava InetAddress.getLocalHost() 在linux里实现](http://blog.csdn.net/raintungli/article/details/8191701)
		
2. Safe shutdown，部分服务退出时要保存状态数据
3. 支持sshd，以方便大家查看日志文件（web console对查看大量日志时，还是不太好用）
	1. 使用supervisord（管理ssh和tomcat），需要通过supervisord传导SIGTERM信号给tomcat，以确保tomcat可以safeshutdown。该方法比较low，经常发生supervisord让tomcat退出，但自己不退出的情况。
	2. 每台机器启动跟一个专门的容器，映射一些必要的volume，以供大家查看日志等文件
3. Marathon多机房主备问题
5. 容器的漂移对日志采集、分析系统的影响
6. 对容器提供dns服务，以使其可以正确解析外部服务的hostname
7. 如何更好的推广与应用的问题（这是个大问题，包括分享ppt的写作思路、jenkins模板的创建等，不比解决技术难题耗费的精力少）

## 引用

[Docker 在 Bilibili 的实战：由痛点推动的容器化](http://dockone.io/article/2023)



	
	