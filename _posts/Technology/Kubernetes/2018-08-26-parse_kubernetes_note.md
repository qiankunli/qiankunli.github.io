---

layout: post
title: 《深入剖析kubernetes》笔记
category: 技术
tags: Kubernetes
keywords: kubernetes 

---

## 简介(会不断增补)

本文来自对极客时间《深入剖析kubernetes》的学习，作者本身对k8s 有一定的基础，但认为同样一个事情 听听别人 如何理解、表述 是很有裨益的，尤其是作者 还是k8s 领域的大牛。

作者在开篇中提到的几个问题 ，也是笔者一直的疑惑

1. 容器技术纷繁复杂，“牵一发而动全身”的主线 在哪里
2. Linux 内核、分布式系统、网络、存储等方方面面的知识，并不会在docker 和 k8s 的文档中交代清楚。可偏偏就是它们，才是真正掌握容器技术体系的精髓所在，是我们需要悉心修炼的内功。

## 一个很长但精彩的故事

### 打包发布阶段

在docker 之前有一个 cloud foundry Paas项目，使用`cf push` 将用户的可执行文件和 启动脚本打进一个压缩包内，上传到cloud foundry 的存储中，然后cloud foundry 会通过调度器选择一个可以运行这个应用的虚拟机，然后通知这个机器上的agent 把应用压缩包下载下来启动。由于需要在一个虚拟机中 启动不同用户的应用，cloud foundry为客户的应用单独创建一个称作沙盒的隔离环境，然后在沙盒中启动这些应用进程。

PaaS 主要是提供了一种名叫“应用托管”的能力。虚拟机技术发展 ==> 客户不自己维护物理机、转而购买虚拟机服务，按需使用 ==> 应用需要部署到云端 ==> 部署时云端虚拟机和本地环境不一致。所以产生了两种思路

1. 将云端虚拟机 做的尽量与 本地环境一样
2. 无论本地还是云端，代码都跑在 约定的环境里 ==> docker 镜像的精髓

与《尽在双11》作者提到的 “docker 最重要的特质是docker 镜像” 一致，docker 镜像提供了一种非常便利的打包机制。

### 农村包围城市

为应对docker 一家在容器领域一家独大的情况，google 等制定了一套标准和规范OCI，意在将容器运行时和镜像的实现从Docker 项目中完全剥离出来。然并卵，Docker 是容器领域事实上的标准。为此，google 将战争引向容器之上的平台层（或者说PaaS层），发起了一个名为CNCF的基金会。所谓平台层，就是除容器化、容器编排之外，推动容器监控、存储层、日志手机、微服务（lstio）等项目百花争鸣，与kubernetes 融为一体。

此时，kubernetes 对docker 依赖的只是一个 OCI 接口，docker 仍然是容器化的基础组件，但其 重要性在 平台化的 角度下已经大大下降了。若是kubernetes 说不支持 docker？那么。。。

笔者负责将公司的测试环境docker化，一直为是否使用kubernetes 替换mesos 而纠结，从现在看：

1. 笔者要做的其实是一个PaaS的事情
1. 单从“测试环境docker化” 来看（将PaaS理解为 打包发布），mesos 也是够用的。
2. 打包发布不是全部，加上编排也不是。就好像一个公司的技术组织架构，靠业务组写业务代码赚钱，但光有业务组是不够的。我们在用docker，更是在搭建一套稳定的 运行平台。

这让笔者想到了孟子说的“民为重、社稷次之、君为轻”，docker 和 kubernetes 的博弈历程 充分的体现了：最重要的角色 如何 演变成 一件可以最重要 也可以 不太重要的角色。

java 是一个单机版的业务逻辑实现语言，但在微服务架构成为标配的今天，服务发现、日志监控报警、熔断等也成为必备组件（spring cloud 提供了完整的一套）。如果这些组件 都可以使用协议来定义，那么最后用不用java 来写业务逻辑就不是那么重要了。

借用同事的一句总结：之前在应用程序里做的或者利用第三方工具做的微服务治理的事情，被下放到paas平台来负责。类似很多jvm兼容的脚本语言，groovy，scala等是以jvm为中心。而微服务以paas为中心，不以类似dubbo之类的框架为中心。如果以框架为中心，那么PaaS 只做打包和发布就行了，若是以PaaS 为中心，服务发现、路由、监控这类事PaaS就要做起来。


## docker

[docker中涉及到的一些linux知识](http://qiankunli.github.io/2016/12/02/linux_docker.html)

有哪些容器与虚拟机表现不一致的问题? 本质上还是共享内核带来的问题

1. 很多资源无法隔离，也就是说隔离是不彻底的。比如宿主机的 时间，你设置为0时区，我设置为东八区，肯定乱套了
2. 很多linux 命令依赖 /proc，比如top，而 容器内/proc 反应的是 宿主机的信息

对于大多数开发者而言，他们对应用依赖的理解，一直局限在编程语言层面，比如golang的godeps.json。容器镜像 打包的不只是应用， 还有整个操作系统的文件和目录。这就意味着，应用以及它运行所需要的所有依赖，都被封装在了一起，进而成为“沙盒”的一部分。参见 [linux 文件系统](http://qiankunli.github.io/2018/05/19/linux_file_mount.html)

默认情况下，Docker 会为你提供一个隐含的ENTRYPOINT，即`/bin/sh -c`。所以在不指定ENTRYPOINT时，CMD的内容就是ENTRYPOINT 的参数。因此，一个不成文的约定是称docker 容器的启动进程为ENTRYPOINT 进程。

一个进程的每种Linux namespace 都可以在对应的`/proc/进程号/ns/` 下有一个对应的虚拟文件，并且链接到一个真实的namespace 文件上。通过`/proc/进程号/ns/` 可以感知到进程的所有linux namespace；一个进程 可以选择加入到一个已经存在的namespace 当中；也就是可以加入到一个“namespace” 所在的容器中。这便是`docker exec`、两个容器共享network namespace 的原理。

一个 容器，可以被如下看待

1. 在docker registry 上 由manifest + 一组blob/layer 构成的镜像文件
2. 一组union mount 在`/var/lib/docker/aufs/mnt` 上的rootfs
3. 一个由namespace + cgroup 构成的隔离环境，即container runtime

## kubernetes——从容器到容器云

### k8s 要解决的核心问题

对于大多数用户来说，需求是确定的：提供一个容器镜像，请在一个给定的集群上把这个应用运行起来。

从这个角度看，k8s和 swarm、mesos 等并没有特别的优势，那么**k8s 的核心优势/解决的核心问题是什么**？编排（对应下图的kube-controller-manager）？调度（对应下图的kube-scheduler）？容器云？还是集群管理？实际上这个问题到目前为止都没有固定的答案。因为在不同的阶段，kubernetes 需要着重解决的问题是不同的。

kubernetes脱胎于Borg，Borg在的时候还没docker呢？所以天然的，Borg及其衍生的kubernetes 从未将docker 作为架构的核心，而仅仅将它作为一个container runtime 实现。

![](/public/upload/kubernetes/borg_in_google.PNG)

PS：从Borg 在google 基础设施的定位看，**或许我们学习k8s，不是它有什么功能，所以我用k8s来做什么事儿。而是打算为它赋予什么样的职责，所以需要k8s具备什么样的能力。 k8s 要做的不是dockerize，也不是containerize，而是作为一个集群操作系统，为此重新定义了可执行文件、进程、存储的形态。**比如“进程”的形态，不单是换成了容器，还有附属的服务发现、负载均衡、日志监控等。

容器是容器云的主角，但在很多电影中，主角只是提供了一个视角，很重要，但不是故事的核心。

### 所以k8s要这样设计 

![](/public/upload/kubernetes/k8s_framework.JPG)

PS：笔者一直对CNI、CSI 有点困惑，为什么不将其纳入到container runtime/oci 的范畴。在此猜测一番

1. oci 的两个部分：runtime spec 和 image spec，runtime spec不说了oci就是干这事儿的。对于后者，image spec 若是不统一，runtime spec 也没法统一，更何况docker image才是 docker 的最大创新点，所以干脆放一起了。
2. docker 给人的错觉是 创建容器 与 配置容器volume/network 是一起的，其实呢，完全可以在容器启动之后，通过csi、cni 驱动相关工具（比如容器引擎）去做。
2. 若是oci 纳入了CSI 和 CNI，则oci 就不是一个单机跑的小工具了。csi和cni 都需要对全局状态有所感知，比如提供一个ipam driver 来维护已分配/未分配的ip。当然，将CNI、CSI 从OCI中剥离，由k8s 在编排/调度层直接操纵，使其最大限度了减少了对container runtime 的依赖性。

k8s 要支持这么多应用，将它们容器化，便要对应用的关系、形态等分门别类，在更高的层次将它们统一进来。

1. 任务与任务的几种关系

	1. 没有关系
	2. 访问关系，比如web 与 db 的依赖。对应提出了service 的概念
	2. 紧密关系，比如一个微服务日志采集监控报警 agent 与 一个微服务microservice的关系，agent 读取microservice 产生的文件。对应提出了pod 的概念

2. 任务的形态，有状态、无状态、定时运行等
3. 如何描述上述信息。k8s中的api对象 有两种

	1. 待编排的对象，比如Pod、Job、CronJob 等用来描述你的应用
	2. 服务对象，比如Service、Secret 等，负责具体的平台级功能

	文章中提到这是声明式api，然后跟编程中的声明式事务、命令/编程式事务 等名词可以对比一下。 笔者的一个体会是，假设实现扩容功能，你可以实现一个`xx scale` 指令，也可以复用`xx create xx.json` 指令，只是在json 文件中说明新的实例数量即可。

![](/public/upload/kubernetes/k8s_pod.PNG)

yarn、mesos 以及swarm 锁擅长的，都是把一个容器按照某种规则，放置在某个最佳节点运行起来，我们称之为“调度”。而k8s 锁擅长的，是按照用户的意愿和系统的规则，完全自动化的处理好应用之间的各种关系，称之为“编排”。


PS：从容器到容器云？是量变还是质变？量变的话，是跑更多的容器么？这个swarm就可以解决。是质变的话，变在哪里？kubernetes 真正的价值，在于提供了一套基于容器构建分布式系统的基础依赖。具体的说，不是一个服务有多个实例就是分布式系统，k8s提供了一种宏观抽象，作为一个集群操作系统，运行各种类型的应用。

运维的同学都知道，公司内的服务器会分为 应用集群、数据集群等，分别用来跑业务应用 和 大数据组件（比如hadoop、spark）等。为何呢？一个是物理机配置不一样；一个是依赖服务不一样，比如应用集群每台主机都会配一个日志采集监控agent，数据集群每台主机也有类似的agent，但因为业务属性不一样，监控的侧重点不一样，所以agent 逻辑不一样。进而，应用集群的服务 不可以随便部署在 数据集群的机器上。





![](/public/upload/kubernetes/parse_k8s_ad.JPG)

笔者个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)
