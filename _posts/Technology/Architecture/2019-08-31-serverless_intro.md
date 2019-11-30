---

layout: post
title: serverless 泛谈
category: 架构
tags: Architecture
keywords: window

---

## 简介（未完成）

* TOC
{:toc}

[Serverless 的喧哗与骚动（一）：Serverless 行业发展简史](https://www.infoq.cn/article/SXv6xredWW03P7NXaJ4m)

1. 单机时代，操作系统管理了硬件资源，贴着资源层，高级语言让程序员描述业务，贴着业务层，编译器 /VM 把高级语言翻译成机器码，交给操作系统；
2. 今天的云时代，资源的单位不再是 CPU、内存、硬盘了，而是容器、分布式队列、分布式缓存、分布式文件系统。

![](/public/upload/architecture/machine_vs_cloud.png)

今天我们把应用程序往云上搬的时候（a.k.a Cloud Native)，往往都会做两件事情：

1. 把巨型应用拆小，微服务化；
2. 摇身一变成为 yaml 工程师，写很多 yaml 文件来管理云上的资源。

这里存在两个巨大的 gap，这两个 gap 在图中用灰色的框表示了：

1.  编程语言和框架，目前主流的编程语言基本都是假设单机体系架构运行的
2. 编译器，程序员不应该花大量时间去写 yaml 文件，这些面向资源的 yaml 文件应该是由机器生成的，我称之为云编译器，高级编程语言用来表达业务的领域模型和逻辑，云编译器负责将语言编译成资源描述。

## serverless 领域行业划分

![](/public/upload/architecture/serverless_layer.png)

1. 资源层关注的是资源（如容器）的生命周期管理，以及安全隔离。这里是 Kubernetes 的天下，Firecracker，gVisor 等产品在做轻量级安全沙箱。这一层关注的是如何能够更快地生产资源，以及保证好安全性。
2. DevOps 层关注的是变更管理、流量调配以及弹性伸缩，还包括基于事件模型和云生态打通。这一层的核心目标是如何把运维这件事情给做没了（NoOps）。虽然所有云厂商都有自己的产品（各种 FaaS），但是我个人比较看好 Knative 这个开源产品，原因有二：

    1. 其模型非常完备；
    2. 其生态发展非常迅速和健康。很有可能未来所有云厂商都要去兼容 Knative 的标准，就像今天所有云厂商都在兼容 Kubernetes 一样。

3. 框架和运行时层呢，由于个人经验所限，我看的仅仅是 Java 领域，其实核心的还是在解决 Java 应用程序启动慢的问题（GraalVM）。当然框架如何避免 vendor lock-in 也很重要，谁都怕被一家云厂商绑定，怕换个云厂商要改代码，这方面主要是 Spring Cloud Function 在做。



[当我们在聊 Serverless 时你应该知道这些](https://mp.weixin.qq.com/s/Krfhpi7G93el4avhv9UN4g)

![](/public/upload/architecture/ali_cloud_function.png)