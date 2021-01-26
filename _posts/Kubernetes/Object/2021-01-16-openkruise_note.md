---

layout: post
title: openkruise学习
category: 架构
tags: Kubernetes
keywords:  kubernetes client-go

---

## 简介（未完成）

## Kubernetes的价值与不足

![](/public/upload/kubernetes/application_delivery.jpg)

docker 让镜像和容器融合在一起，`docker run` 扣动扳机，实现镜像到 容器的转变。但`docker run` 仍然有太多要表述的，比如`docker run`的各种参数： 资源、网络、存储等，“一个容器一个服务”本身也不够反应应用的复杂性（或者 说还需要额外的信息 描述容器之间的关系，比如`--link`）。

我们学习linux的时候，知道linux 提供了一种抽象：一切皆文件。没有人会质疑基本单元为何是文件不是磁盘块？linux 还提供了一种抽象叫“进程”，没有人会好奇为何 linux 不让你直接操纵cpu 和 内存。**一个复杂的系统，最终会在某个层面收敛起来**，就好像一个web系统，搞到最后就是一系列object 的crud。类似的，如果我们要实现一个“集群操作系统”，容器的粒度未免太小。也就是说，镜像和容器之间仍然有鸿沟要去填平？kubernetes 叫pod，marathon叫Application，中文名统一叫“应用”。在这里，应用是一组容器的有机组 合，同时也包括了应用运行所需的网络、存储的需求的描述。而像这样一个“描述” 应用的 YAML 文件，放在 etcd 里存起来，然后通过控制器模型驱动整个基础设施的 状态不断地向用户声明的状态逼近，就是 Kubernetes 的核心工作原理了。“有了 Pod 和容 器设计模式，我们的**应用基础设施才能够与应用(而不是容器)进行交互和响应的能力**，实现了“云”与“应用”的直接对接。而有了声明式 API，我们的应用基础而设施才能真正同下层资源、调度、编排、网络、存储等云的细节与逻辑解耦”。

[解读容器的 2020：寻找云原生的下一站](https://mp.weixin.qq.com/s/_4IuskCv7IsnBg1eEssB2g)面向终态的声明式 API 与其背后“辛勤”工作的控制器，为“构建基础设施层抽象”这个充满了挑战的技术难题，提供了一个能够在复杂度与可用性之间取得平衡的解决方案。**但k8s 也并不完美。在绝大多数情况下，企业基于 Kubernetes 构建上层平台，都会引入各种各样其他的抽象作为补充，甚至取代或者隐藏掉 Kubernetes 的部分内置抽象**：阿里巴巴开源的 CloneSet，腾讯的 GameStatefulSet 实践等扩展型工作负载等都是这个趋势的最好的案例。今天能够基于 Kubernetes 成体系构建出完整上层平台的团队，其实集中在一、二线大型互联网公司当中，并且其实践往往“仅供参考”，鲜有可复制性。

事实上，平台构建者之所以要基于 Kubernetes 进一步构建上层平台，其根本动机无非来自两个诉求：
1. 更高的抽象维度：比如，用户希望操作的概念是“应用”和“灰度发布”，而不是“容器”和“Pod”；大家为 Kubernetes 构建的各种 Dashboard，其实就是一种“抽象”的实现方式：这些 Dashboard 本质上是在 Kubernetes API 对象的基础上暴露出了一组允许用户填写的字段，从而实现了‘’简化用户使用心智、提升用户体验‘’的目的
2. 更多的扩展能力：比如，用户希望的应用灰度发布策略是基于“双 Deployment + Istio” 的金丝雀发布，而不是 Kubernetes 默认的 Pod 线性滚动升级。这些增强或者扩展能力，在 Kubernetes 中一般是以 CRD + Controller 的插件方式来实现的。

在云原生与 Kubernetes 项目极大程度的统一与标准化了基础设施层抽象之后，如何进一步帮助平台团队在此之上快速、轻松、可复制的构建上层平台，正在成为业界开始积极思考的一条关键路径。

[KubeVela：标准化的云原生平台构建引擎](KubeVela：标准化的云原生平台构建引擎)

![](/public/upload/kubernetes/kubevela_application.png)

## 代码结构

```
github.com/openkruise/kruise
    /pkg
        /client
        /controller         // 支持crd 对应实现
            /cloneset
                /cloneset_controller.go 
            /controllers.go     // 将各个Controller/Reconciler 实现注册到 Manager 中
        /webhook
    /main.go 
```

kubebuilder 脚手架生成的项目代码 一般假设只有一个crd 和Controller/Reconciler，因此main.go 的核心逻辑是: Reconciler 注册到Manager； Manager.Start。 对于kruise 来讲因为有多个 Controller，所以每一个controllr pkg 都实现了一个Add 方法负责 将自己注册到 Manager

