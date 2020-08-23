---

layout: post
title: Kubernetes介绍
category: 技术
tags: Kubernetes
keywords: kubernetes 

---

## 简介

* TOC
{:toc}

## Kubernetes——集群操作系统

比较 Kubernetes 和单机操作系统，Kubernetes 相当于内核，它负责集群软 硬件资源管理，并对外提供统一的入口，用户可以通过这个入口来使用集群，和集 群沟通。

![](/public/upload/kubernetes/k8s_vs_kernel.png)

||关在“笼子”里的程序|一般运行在操作系统上的程序|
|---|---|---|
||为了让这个程序不依赖于操作系统自身的库文件，<br>我们需要制作容器镜像||
|程序分发|镜像仓库|应用商店|
|系统入口|apiserver|系统调用|

2019.8.13：刘超《趣谈linux操作系统》

最初使用汇编语言的前辈，在程序中需要指定使用的硬件资源，例如，指定使用哪个寄存器、放在内存的哪个位置、写入或者读取那个串口等等。对于这些资源的使用，需要程序员自己心里非常地清楚，要不然一旦 jump 错了位置，程序就无法运行。为了将程序员从对硬件的直接操作中解放出来，我们有了操作系统这一层，用来实现对于硬件资源的统一管理。某个程序使用哪个 CPU、哪部分内存、哪部分硬盘，都由操作系统自行分配和管理。其实操作系统最重要的事情，就是调度（当前任务需要多少资源，当前空闲多少资源，做一个适配）。因此，在内核态就产生这些模块：进程管理子系统、内存管理子系统、文件子系统、设备子系统网络子系统。这些模块通过统一的 API，也就是系统调用，对上提供服务。基于 API，用户态有很多的工具可以帮我们使用好 Linux 操作系统，比如用户管理、软件安装、软件运行、周期性进程、文件管理、网络管理和存储管理。但是到目前为止，我们能管理的还是少数几台机器。当我们面临数据中心成千上万台机器的时候，仍然非常“痛苦”。

![](/public/upload/kubernetes/linux_vs_kubernetes.jpg)

PS：上图很有意思的一点是，将Kubernetes 的各项机制，根据内核态和用户态做了一个梳理。

2019.8.3：《阿里巴巴云原生实践15讲》读后感

![](/public/upload/kubernetes/application_delivery.jpg)

docker 让镜像和容器融合在一起，`docker run` 扣动扳机，实现镜像到 容器的转变。但`docker run` 仍然有太多要表述的，比如`docker run`的各种参数： 资源、网络、存储等，“一个容器一个服务”本身也不够反应应用的复杂性（或者 说还需要额外的信息 描述容器之间的关系，比如`--link`）。

我们学习linux的时候，知道linux 提供了一种抽象：一切皆文件。没有人会质疑基本单元为何是文件不是磁盘块？linux 还提供了一种抽象叫“进程”，没有人会好奇为何 linux 不让你直接操纵cpu 和 内存。一个复杂的系统，最终会在某个层面收敛起来，就好像一个web系统，搞到最后就是一系列object 的crud。类似的，如果我们要实现一个“集群操作系统”，容器的粒度未免太小。也就是说，镜像和容器之间仍然有鸿沟要去填平？kubernetes 叫pod，marathon叫Application，中文名统一叫“应用”。在这里，应用是一组容器的有机组 合，同时也包括了应用运行所需的网络、存储的需求的描述。而像这样一个“描述” 应用的 YAML 文件，放在 etcd 里存起来，然后通过控制器模型驱动整个基础设施的 状态不断地向用户声明的状态逼近，就是 Kubernetes 的核心工作原理了。“有了 Pod 和容 器设计模式，我们的**应用基础设施才能够与应用(而不是容器)进行交互和响应的能力**，实现了“云”与“应用”的直接对接。而有了声明式 API，我们的应用基础而设 施才能真正同下层资源、调度、编排、网络、存储等云的细节与逻辑解耦”。

关于声明式api，举一个调度例子， 假设一个服务有10个实例，要求两个实例不能在同一个机器上。在“命令式”的传统发布系统中，一般会指定每个实例所在的机器（就像指定数据放在哪个集群器、放在内存的哪个位置一样）而对于k8s，“两个实例不能在同一个机器上”就是一个配置而已， 由k8s相机将实例调度在不同的机器上。PS：笔者最近在团队管理上也有一些困惑， 其实，你直接告诉小伙伴怎么做并不是一个很好的方式，他们不自由，你也很心累。比较好的方式是做好目标管理，由他们自己不断去填平目标与现实的鸿沟。

[Reasons Kubernetes is cool](https://jvns.ca/blog/2017/10/05/reasons-kubernetes-is-cool/)once you have a working Kubernetes cluster you really can set up a production **HTTP service** (“run 5 of this application, set up a load balancer, give it this DNS name, done”) with just one configuration file. 对运维来说部署java application 和 部署mysql 是两个事情，但对于k8s 来说，就是一个事情。 

## 赢在orchestrator

一般orchestrator 包括但不限于以下功能：

1. Organizational primitives，比如k8s的label
2. Scheduling of containers to run on a ost
3. Automated health checks to determine if a container is alive and ready to serve traffic and to relaunch it if necessary
4. autoscaling 
5. upgrading strategies,from rolling updates to more sophisticated techniques such as A/B and canary deployments.
6. service discovery to determine which host a scheduled container ended upon,usually including DNS support.

The unit of scheduling in Kubernetes is a pod. Essentially, this is a tightly coupled set of one or more containers that are always collocated (that is, scheduled onto a node as a unit); they cannot be spread over nodes. 

1. The number of running instances of a pod—called replicas—can be declaratively stated and enforced through controllers. 
2. **The logical organization of all resources, such as pods, deployments, or services, happens through labels.** label 的作用不小啊

Kubernetes is highly extensible, from defining new workloads and resource types in general to customizing its user-facing parts, such as the CLI tool kubectl (pronounced cube cuddle).

##  Container Engine cluster

本小节主要来自对[https://cloud.google.com/container-engine/docs](https://cloud.google.com/container-engine/docs)的摘抄，有删减。

本小节主要讲了Container Engine cluster和Pod的概念

A Container Engine cluster is a group of Compute Engine instances running Kubernetes. It consists of one or more node instances, and a Kubernetes master instance. A cluster is the foundation of a Container Engine application—pods,services, and replication controllers all run on top of a cluster.

一个Container Engine cluster主要包含一个master和多个slave节点，它是上层的pod、service、replication controllers的基础。

1. The Kubernetes master, Every cluster has a single master instance. The master provides a unified view into the cluster and, through its publicly-accessible endpoint, is the doorway(途径) for interacting with the cluster.
2. Nodes, A cluster can have one or more node instances. These are managed from the master, and run the services necessary to support Docker containers. Each node runs the Docker runtime and hosts a Kubelet agent（管理docker runtime）, which manages the Docker containers scheduled on the host. Each node also runs a simple network proxy（网络代理程序）.

**The master runs the Kubernetes API server, which services REST requests, schedules pod creation and deletion on worker nodes, and synchronizes pod information (such as open ports and location) with service information.**

1. 提供统一视图
2. service REST requests
3. 调度
4. 控制，使得actual state满足desired state 

## 设计理念

[火得一塌糊涂的kubernetes有哪些值得初学者学习的？](https://mp.weixin.qq.com/s/iI5vpK5bVkKmdbf9sbAGWw)

1. 声明式 VS 命令式, 声明式优点很多，一个很重要的点是：在分布式系统中，任何组件都可能随时出现故障。当组件恢复时，需要弄清楚要做什么，使用命令式 API 时，处理起来就很棘手。但是使用声明式 API ，组件只需查看 API 服务器的当前状态，即可确定它需要执行的操作。
2. 显式的 API, Kubernetes 是透明的，它没有隐藏的内部 API。换句话说 **Kubernetes 系统内部用来交互的 API 和我们用来与 Kubernetes 交互的 API 相同**。这样做的好处是，当 Kubernetes 默认的组件无法满足我们的需求时，我们可以利用已有的 API 实现我们自定义的特性。
3. 无侵入性, 我们的应用达到镜像后, 不需要改动就可以遨游在 Kubernetes 集群中。 Kubernetes 以一种友好的方式将 Secret、Configuration等注入 Pod，减少了大家的工作量，而无需重写或者很大幅度改变原有的应用代码。
4. 有状态的移植, 比如 PersistentVolumeClaim 和 PersistentVolume

## 整体结构

[Kubernetes架构为什么是这样的？](https://mp.weixin.qq.com/s/ps34qFlEzQNYbp6ughkrOA)

简版的

![](/public/upload/kubernetes/k8s_architecture.png)

详细的

![](/public/upload/kubernetes/k8s_framework.PNG)

调整后的架构图

![](/public/upload/kubernetes/k8s_framework_refactor.PNG)

1. etcd，各个组件通信都并不是互相调用 API 来完成的，而是把状态写入 ETCD（相当于写入一个消息），其他组件通过监听 ETCD 的状态的的变化（相当于订阅消息），然后做后续的处理，然后再一次把更新的数据写入 ETCD。
2. api server，各个组件并不是直接访问 ETCD，而是访问一个代理，这个代理是通过标准的RESTFul API，重新封装了对 ETCD 接口调用，除此之外，这个代理还实现了一些附加功能，比如身份的认证、缓存等
3. Controller Manager 是实现任务调度的
4. Scheduler 是用来做资源调度的

## Kubernetes 仍需进一步发展

[Kubernetes何时才会消于无形却又无处不在？](https://mp.weixin.qq.com/s?__biz=MzA5OTAyNzQ2OA==&mid=2649699253&idx=1&sn=7f47db06b63c4912c2fd8b4701cb8d79&chksm=88930cd6bfe485c04b99b1284d056c886316024ba4835be8967c4266d9364cffcfedaf397acc&mpshare=1&scene=23&srcid=1102iGdvWF6lcNRaDD19ieRy%23rd)一项技术成熟的标志不仅仅在于它有多流行，还在于它有多不起眼并且易于使用。Kubernetes依然只是一个半成品，还远未达到像Linux内核及其周围操作系统组件在过去25年中所做到的那种“隐形”状态。

[解读2018：我们处在一个什么样的技术浪潮当中？](https://mp.weixin.qq.com/s?__biz=MjM5MDE0Mjc4MA==&mid=2651011968&idx=1&sn=3d500660f7dd47c9fa4033bd9fa69c2f&chksm=bdbec3d38ac94ac523355e1e21f04af71e47a0841d1af0afedecc528b5eb4a5f9fe83f105a11&mpshare=1&scene=1&srcid=12217gWDeJ0aPl8BVBUycQyh#rd)Kubernetes 还是太底层了，真正的云计算并不应该是向用户提供的 Kubernetes 集群。2014 年 AWS 推出 Lambda 服务，Serverless 开始成为热词。从理论上说，Serverless 可以做到 NoOps、自动扩容和按使用付费，也被视为云计算的未来。Serverless 是我们过去 25 年来在 SaaS 中走的最后一步，因为我们已经渐渐将越来越多的职责交给了服务提供商。——Joe Emison 《为什么 Serverless 比其他软件开发方法更具优势》

[解读容器 2019：把“以应用为中心”进行到底](https://www.kubernetes.org.cn/6408.html)云原生的本质是一系列最佳实践的结合；更详细的说，云原生为实践者指定了一条低心智负担的、能够以可扩展、可复制的方式最大化地利用云的能力、发挥云的价值的最佳路径。这种思想，以一言以蔽之，就是“以应用为中心”。正是因为以应用为中心，云原生技术体系才会无限强调**让基础设施能更好的配合应用**、以更高效方式为应用“输送”基础设施能力，而不是反其道而行之。而相应的， Kubernetes 、Docker、Operator 等在云原生生态中起到了关键作用的开源项目，就是让这种思想落地的技术手段。


