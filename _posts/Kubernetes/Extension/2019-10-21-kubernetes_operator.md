---

layout: post
title: kubernetes operator
category: 技术
tags: Kubernetes
keywords: kubernetes operator

---

## 简介

* TOC
{:toc}

[面向 K8s 设计误区](https://mp.weixin.qq.com/s/W_UjqI0Rd4AAVcafMiaYGA)

[Kubernetes operator 模式开发实践](https://mp.weixin.qq.com/s/VJVVB5JJ_RlGnbq0hyFd8A)

来看一个Operator的示例：基于Operator 实现微服务的运维 MicroServiceDeploy 的 CR（custom resource） 

```yaml
apiVersion: custom.ops/v1
kind: MicroServiceDeploy
metadata:
  name: ms-sample-v1s0
spec:
  msName: "ms-sample"                     # 微服务名称
  fullName: "ms-sample-v1s0"              # 微服务实例名称
  version: "1.0"                          # 微服务实例版本
  path: "v1"                              # 微服务实例的大版本，该字符串将出现在微服务实例的域名中
  image: "just a image url"               # 微服务实例的镜像地址
  replicas: 3                             # 微服务实例的 replica 数量
  autoscaling: true                       # 该微服务是否开启自动扩缩容功能
  needAuth: true                          # 访问该微服务实例时，是否需要租户 base 认证
  config: "password=88888888"             # 该微服务实例的运行时配置项
  creationTimestamp: "1535546718115"      # 该微服务实例的创建时间戳
  resourceRequirements:                   # 该微服务实例要求的机器资源
    limits:                                 # 该微服务实例会使用到的最大资源配置
      cpu: "2"
      memory: 4Gi
    requests:                               # 该微服务实例至少要用到的资源配置
      cpu: "2"
      memory: 4Gi
  idle: false                             # 是否进入空载状态
```

以上一个 resource 实际上创建了很多其他的 Kubernetes resource，这些 Kubernetes resource 才真正构成了该微服务实际的能力

1. 对于一个微服务而言必备的 Service, ServiceAccount 和 Deployment。
2. 代表运行时配置项的ConfigMap
3. 资源管理 ResourceQuota
4. HPA自动扩缩容

operator 本质上不创造和提供新的服务，它只是已有 Kubernetes API service 的组合，但这种“抽象”大大简化了运维操作，**否则这些逻辑都要由上层发布系统实现**。
**Kubernetes 集群真正的能力（mount目录、操作GPU等）要通过 Kubelet 去支持**。 


## 内涵

### kube-native way of managing the lifecycle of service in Kubernetes

An Operator is a method of packaging, deploying and managing a **Kubernetes application**. A Kubernetes application is an application that is both deployed on Kubernetes and managed using the Kubernetes APIs and kubectl tooling.

An Operator is an application-specific controller that extends the Kubernetes API to create, configure and manage instances of complex stateful applications on behalf of a Kubernetes user. **It builds upon the basic Kubernetes resource and controller concepts**, but also includes domain or **application-specific** knowledge to automate common tasks better managed by computers.

[Redis Enterprise Operator for Kubernetes](https://redislabs.com/blog/redis-enterprise-operator-kubernetes/)**kube-native way of managing the lifecycle of service in Kubernetes**：Although Kubernetes is good at scheduling resources and recovering containers gracefully from a failure, it does not have primitives that understand the internal lifecycle of a data service.

[使用etcd-operator在集群内部署etcd集群](https://blog.csdn.net/fy_long/article/details/88874373)Operator 本身在实现上，其实是在 Kubernetes 声明式 API 基础上的一种“微创新”。它合理的利用了 Kubernetes API 可以添加自定义 API 类型的能力，然后又巧妙的通过 Kubernetes 原生的“控制器模式”，完成了一个面向分布式应用终态的调谐过程。

||spring|kubernetes|
|---|---|---|
|核心|ioc模式|声明式api + controller模式|
|常规使用|`<bean>` FactoryBean等|pod 及之上扩展的deployment等|
|扩展|自定义namespace及NamespaceHandler|CRD|
|微创新|比如整合rabbitmq `<rabbit:template>`|etcd/redis operator|

Linux 提供了一个 vfs 接口，任何资源都是File， 提供Open/Close/Create/Read/Write 5个基本api， 从linux 内置文件、Socket 到外设的各种显示器、键盘、鼠标 都统一到一个vfs体系之下。 PS：开发提供一个fs驱动，用户就可以通过读写文件实现特定的业务逻辑，比如cgroup/内存文件系统/网络通信/共享文件系统等。 

||Linux vfs|Kubernetes 控制器模式|
|---|---|---|
|作用|管理文件、网络和“外设”|管理builtin 及Custom Resource|
|理念|一切皆文件|一切皆Resource|
|注册|注册设备到vfs|注册API 到apiserver|
|差异化|驱动程序<br>比如文件的Open和Socket Open操作就不同|Controller|

### 分布式系统的标准化交付

[当我们聊 Kubernetes Operator 时，我们在聊些什么](https://www.infoq.cn/article/SJMUvMg_0H7BS5d99euR)

如果说 Docker 是奠定的单实例的标准化交付，那么 Helm 则是集群化多实例、多资源的标准化交付。

**Operator 则在实现自动化的同时实现了智能化**。其主要的工作流程是根据当前的状态，进行智能分析判断，并最终进行创建、恢复、升级等操作。而**位于容器中的脚本，因为缺乏很多全局的信息，仅靠自身是无法无法达实现这些全部的功能的**。而处于第三方视角的 Operator，则可以解决这个问题。他可以通过侧面的观察，获取所有的资源的状态和信息，并且跟预想 / 声明的状态进行比较。通过预置的分析流程进行判断，从而进行相应的操作，并最终达到声明状态的一个目的。这样**所有的运维逻辑就从镜像中抽取出来，集中到 Operator 里去**。层次和逻辑也就更加清楚，容易维护，也更容易交付和传承。

以往的高可用、扩展收缩，以及故障恢复等等运维操作，都通过 Operator 进行沉淀下来。**从长期来看，将会推进 Dev、Ops、DevOps 的深度一体化**。将运维经验、应用的各种方案和功能通过代码的方式进行固化和传承，减少人为故障的概率，提升整个运维的效率。Controller 和 Operator 的关系有点类似于标准库和第三方库的关系

## 原理

[Kubernetes Operator 快速入门教程](https://www.qikqiak.com/post/k8s-operator-101/)以定制化开发一个 AppService 为例，描述了从零创建一个AppService Operator 的过程

    apiVersion: app.example.com/v1
    kind: AppService
    metadata:
    name: nginx-app
    spec:
    size: 2
    image: nginx:1.7.9
    ports:
        - port: 80
        targetPort: 80
        nodePort: 30002

脚手架

1. [Operator SDK](https://github.com/operator-framework/operator-sdk) [如何在Kubernetes中编写自定义控制器](https://mp.weixin.qq.com/s/yTM_l_EHr7TmCMAuwJH0ug)
2. [kubebuilder](https://github.com/kubernetes-sigs/kubebuilder) [Kubernetes operator 模式开发实践](https://mp.weixin.qq.com/s/VJVVB5JJ_RlGnbq0hyFd8A)

## 部署

[Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)The most common way to deploy an Operator is to add the Custom Resource Definition and its associated Controller to your cluster. The Controller will normally run outside of the **control plane**, much as you would run any containerized application. For example, you can run the controller in your cluster as a Deployment. **Operator Controller 物理存在是一个bin文件**，将Operator 以Deployment/容器的方式 运行在Kubernetes 集群中，无需更改Kubernetes 本身。 

我们都知道在 Kubernetes 上安装应用可以使用 Helm 直接安装各种打包成 Chart 形式的 Kubernetes 应用，但随着 Kubernetes Operator 的流行，Kubernetes 社区又推出了 [OperatorHub](https://operatorhub.io/)，你可以在这里分享或安装 Operator：https://www.operatorhub.io。

## 示例

### 在k8s上部署一个etcd

有几种方法

1. 找一个etcd image，组织一个pod，进而写一个 deployment.yaml，kubectl apply 一下
2. 使用helm
3. kubernetes operator [etcd Operator](https://coreos.com/operators/etcd/docs/latest/)。

        apiVersion: etcd.database.coreos.com/v1beta2
        kind: EtcdCluster
        metadata:
        name: example
        spec:
            size: 3
            version: 3.2.13

使用etcd operator 有一个比较好玩的地方，仅需调整 size 和 version 配置，就可以控制etcd cluster 个数和版本，比第一种方法方便的多了。

### redis

我们在看一个redis cluster 的部署过程，可以看到， **operator让你以更贴近redis的特质来部署reids**，而不是在部署deployment和拼装pod。

    apiVersion: app.redislabs.com/v1alpha1
    kind: RedisEnterpriseCluster
    metadata:
    name: redis-enterprise
    spec:
    nodes: 3
    persistentSpec:
        enabled: 'true'
        storageClassName: gp2
    uiServiceType: LoadBalancer
    username: admin@acme.com
    redisEnterpriseNodeResources:
        limits:
        cpu: 400m
        memory: 4 Gi
        requests:
        cpu: 400m
        memory: 4 Gi
    redisEnterpriseImageSpec:
        imagePullPolicy: IfNotPresent
        repository: redislabs/redis
        versionTag: 5.4.0-19

