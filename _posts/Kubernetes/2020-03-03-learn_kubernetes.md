---

layout: post
title: 如何学习Kubernetes
category: 技术
tags: Kubernetes
keywords: learn kubernetes 

---

## 简介

* TOC
{:toc}

[Kubernetes 学习路径](https://www.infoq.cn/article/9DTX*1i1Z8hsxkdrPmhk)

所谓学习，就是要能回答几个基本问题：

1. [kubectl 创建 Pod 背后到底发生了什么？](https://mp.weixin.qq.com/s/ctdvbasKE-vpLRxDJjwVMw) 这个问题 对于一个java 工程师来说，就像“在浏览器里输入一个url之后发生了什么？” 一样基础。再进一步就是 自己从kubectl 到crd 实现一遍。

2. [github kubernetes community](https://github.com/kubernetes/community/tree/8decfe42b8cc1e027da290c4e98fa75b3e98e2cc/contributors/devel)

好文章在知识、信息概念上一定是跨层次的，既有宏观架构，又有微观佐证。只有原理没有代码总觉得心虚，只有代码没有原理总觉得心累。**从概念过渡到实操，从而把知识点掌握得更加扎实**。

## 声明式应用管理

[Kubernetes 是一个“数据库”吗？](https://mp.weixin.qq.com/s/QrHpw8PCAhQOxCTLRVeLLg)Kubernetes 里面的绝大多数功能，无论是 kubelet 执行容器、kube-proxy 执行 iptables 规则，还是 kube-scheduler 进行 Pod 调度，以及 Deployment 管理 ReplicaSet 的过程等等，其实从总体设计上都是在遵循着我们经常强调过的“控制器”模式来进行的。即：用户通过 YAML 文件等方式来表达他所想要的期望状态也就是终态（无论是网络、还是存储），然后 Kubernetes 的各种组件就会让整个集群的状态跟用户声明的终态逼近，最终达成两者的完全一致。这个实际状态逐渐向期望状态逼近的过程，就叫做 reconcile（调谐）。而同样的原理，也正是 Operator 和自定义 Controller 的核心工作方式。


声明式应用管理的本质：Infrastructure as Data/Configuration as Data，这种思想认为，基础设施的管理不应该耦合于某种编程语言或者配置方式，而应该是纯粹的、格式化的、系统可读的数据，并且这些数据能够完整的表征使用者所期望的系统状态。这样做的好处就在于，任何时候我想对基础设施做操作，最终都等价于对这些数据的“增、删、改、查”。而更重要的是，我对这些数据进行“增、删、改、查”的方式，与这个基础设施本身是没有任何关系的。

这种好处具体体现在 Kubernetes 上，就是如果我想在 Kubernetes 上做任何操作，我只需要提交一个 YAML 文件，然后对这个 YAML 文件进行增删改查即可。而不是必须使用 Kubernetes 项目的 Restful API 或者 SDK 。这个 YAML 文件里的内容，其实就是 Kubernetes 这个 IaD 系统对应的 Data（数据）。Kubernetes 从诞生起就把它的所有功能都定义成了所谓的“API 对象”，其实就是定义成了一份一份的 Data。**Kubernetes 本质上其实是一个以数据（Data）来表达系统的设定值、通过控制器（Controller）的动作来让系统维持在设定值的调谐系统**。

既然 Kubernetes 需要处理这些 Data，那么 Data 本身不是也应该有一个固定的“格式”这样 Kubernetes 才能解析它们呢？没错，这里的格式在 Kubernetes 中就叫做 API 对象的 Schema。Kubernetes 为你暴露出来的各种 API 对象，实际上就是一张张预先定义好 Schema 的表（Table）。而我们绞尽脑汁编写出的那些 YAML 文件，其实就是对这些表中的数据（Data）进行的增删改查（CURD）。唯一跟传统数据库不太一样的是，Kubernetes 在拿到这些数据之后，并不以把这些数据持久化起来为目的。

如果你从一个“数据库”的角度重新审视 Kubernetes 设计的话
1. 数据模型 -  Kubernetes 的各种 API 对象与 CRD 机制
2. 数据拦截校验和修改机制 - Kubernetes Admission Hook
3. 数据驱动机制 - Kubernetes Controller/Operator
4. 数据监听变更与索引机制 - Kubernetes 的 Informer 机制

随着 Kubernetes 基础设施越来越复杂，第三方插件与能力越来越多，社区的维护者们也发现 Kubernetes 这个“数据库”内置的“数据表”无论从规模还是复杂度上，都正在迎来爆炸式的增长。所以  Kubernetes 社区很早就在讨论如何给 Kubernetes  设计出一个“数据视图（View）”出来。阿里正在推 （OAM）及 oam-kubernetes-runtime



## 通用实现

除apiserver/kubectl 之外（kubelet 类似，但更复杂些），与api server 通信的Controller/Scheduler 的业务逻辑几乎一致

1. 组件需要与apiserver 交互，但核心功能组件不直接与api-server 通信，而是抽象了一个Informer 负责apiserver 数据的本地cache及监听。Informer 还会比对 资源是否变更（依靠内部的Delta FIFO Queue），只有变更的资源 才会触发handler。**因为Informer 如此通用，所以Informer 的实现在 apiserver 的 访问包client-go 中**。*在k8s推荐的官方java库中，也支持直接创建Informer 对象*。
2. 组件 全部采用control loop 逻辑
3. 组件 全部内部维护一个 queue队列，通过注册Informer事件 函数保持 queue数据的更新 或者说 作为队列的生产者，control loop 作为队列的消费者。
4. 通过Informer 提供过的Lister 拥有遍历数据的能力，将操作结果 重新通过kubeclient 写入到apiserver 

![](/public/upload/kubernetes/component_overview.png)

## 从技术实现角度的整体设计

[A few things I've learned about Kubernetes](https://jvns.ca/blog/2017/06/04/learning-about-kubernetes/) 值得读三遍

[47 advanced tutorials for mastering Kubernetes](https://techbeacon.com/top-tutorials-mastering-kubernetes) 设计k8s 的方方面面，未详细读

### 从单机到集群到调度

一个逻辑链条是 kubelet ==> api server ==> scheduler。当你一想k8s有点懵的时候，可以从这个角度去发散。

1. the “kubelet” is in charge of running containers on nodes
2. If you tell the API server to run a container on a node, it will tell the kubelet to get it done (indirectly)
3. the scheduler translates “run a container” to “run a container on node X”

kubelet/api server/scheduler 本身可能会变，但它们的功能以及 彼此的交互接口 不会变的太多，it’s a good place to start

### 源码包结构

[kubernetes源码解读——源码结构](https://blog.csdn.net/ZQZ_QiZheng/article/details/54729869)

![](/public/upload/kubernetes/k8s_source_package.png)

1. pkg是kubernetes的主体代码，里面实现了kubernetes的主体逻辑
2. cmd是kubernetes的所有后台进程的代码，主要是各个子模块的启动代码，具体的实现逻辑在pkg下
3. plugin主要是kube-scheduler和一些插件

源码地址 [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes)

	go get -d k8s.io/kubernetes
	cd $GOPATH/src/k8s.io/kubernetes
	
然后使用ide 工具比如goland 等就可以打开Kubernetes 文件夹查看源码了。

感觉你 `go get github.com/kubernetes` 也没什么错，但因为代码中 都是 `import k8s.io/kubernetes/xxx` 所以推荐前者


## 如何看待k8s

笔者最开始接触k8s 是想用k8s 接管公司的测试环境，所以一直以来对k8s的理解停留在 如何将一个项目运行在k8s上，之后向前是ci/cd，向后是监控报警等。一直以来有两个误区：
1. 虽然知道k8s 支持crd，但只是认为crd 是增强服务发布能力的。**以发布、运行服务为主来先入为主的理解k8s**。
2. k8s 是声明式的，所以crd 也应该是声明式的。

这个印象在openkruise 推出ImagePullJob 之后产生了动摇，这个crd的作用是将 image 下发到指定个规则的node上。[面对不可避免的故障，我们造了一个“上帝视角”的控制台](https://mp.weixin.qq.com/s/QxTMLqf8VspXWy4N4AIFuQ) 中chaosblade 的crd 则支持向某个进程注入一个故障。chaosblade  本身首先是一个故障注入平台，然后才是支持在k8s 平台上支持故障注入（物理机也是支持的）。

当然这个观点在 [面向 K8s 设计误区](https://mp.weixin.qq.com/s/W_UjqI0Rd4AAVcafMiaYGA) 被认为是错误的。几个错误思维：
1. 一切设计皆 YAML；
2. 一切皆合一；
3. 一切皆终态；
4. 一切交互皆 cr。

[谈谈 Kubernetes 的问题和局限性](https://mp.weixin.qq.com/s/ULfmxZh2PBYK-298Xskf2Q)

## 一个充分支持扩展的系统

Kubernetes 本身就是微服务的架构，虽然看起来复杂，但是容易定制化，容易横向扩展。在 Kubernetes 中几乎所有的组件都是无状态化的，状态都保存在统一的 etcd 里面，这使得扩展性非常好，组件之间异步完成自己的任务，将结果放在 etcd 里面，互相不耦合。有了 API Server 做 API 网关，所有其他服务的协作都是基于事件的，**因而对于其他服务进行定制化，对于 client 和 kubelet 是透明的**。

![](/public/upload/kubernetes/kubernetes_extension.png)

|k8s涉及的组件|功能交付方式||
|---|---|---|
|crd|**API**|
|kubectl|binary，用户直接使用|
|kubelet|binary，提供http服务|
|cri-shim|grpc server|
|csi|grpc server|
|cni plugin|binary，程序直接使用|binary 放在约定目录，需要安装到所有Node节点上|
|adminssion controller|webhook|
|Scheduler plugin|被编译到Scheduler中|
|Operator|binary，以容器方式运行在Kubernetes 集群中<br>通过扩展API server支持与kubectl 交互|
|Ingress Controller|pod 部署在 Kubernetes 集群里，根据api server 中Ingress 数据做出处理即可|
|metric server|以Deployment 方式运行|
|cadvisor|以源码包方式被集成在 kubelet里|

单机有 ubuntu centos 操作系统；集群中可以把 Kubernetes 看成云操作系统。单机有计算、存储、网络等驱动；集群有 CNI/CSI/CRI 实现像是集群的驱动。

[你该如何为 Kubernetes 定制特性](https://mp.weixin.qq.com/s/0XZa2KBubdNtN6sJTonluA)

[假如重新设计Kubernetes](https://mp.weixin.qq.com/s/QgROwPVRgpE-jF7vMtjJcQ)
