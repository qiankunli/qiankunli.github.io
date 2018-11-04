---

layout: post
title: kubernetes objects再认识
category: 技术
tags: Kubernetes
keywords: kubernetes yaml

---

## 简介（持续更新）

* TOC
{:toc}

本文来自对极客时间《深入剖析kubernetes》的学习

kubernetes objects 配置部分可以参见 [kubernetes yaml配置](http://qiankunli.github.io/2018/11/04/kubernetes_yaml.html)

## Pod

为什么需要pod？

1. 操作系统为什么要有进程组？原因之一是 Linux 操作系统只需要将信号，比如，SIGKILL 信号，发送给一个进程组，那么该进程组中的所有进程就都会收到这个信号而终止运行。
2. 在 Borg 项目的开发和实践过程中，Google 公司的工程师们发现，他们部署的应用，往往都存在着类似于“进程和进程组”的关系。更具体地说，就是这些应用之间有着密切的协作关系，使得它们必须部署在同一台机器上。具有“超亲密关系”容器的典型特征包括但不限于：

	* 互相之间会发生直接的文件交换
	* 使用 localhost 或者 Socket文件进行本地通信
	* 会发生非常频繁的远程调用
	* 需要共享某些 Linux Namespace

3. 亲密关系 ==> 亲密关系为什么不在调度层面解决掉？非得提出pod 的概念？[容器设计模式](https://www.usenix.org/system/files/conference/hotcloud16/hotcloud16_burns.pdf)
4. Pod，其实是一组共享了某些资源的容器。当然，共享Network Namespace和Volume 可以通过`通过docker run --net=B --volumes-from=B --name-=A image-A...`来实现，但这样 容器 B 就必须比容器 A 先启动，这样一个 Pod 里的多个容器就不是对等关系，而是拓扑关系了。
5. **Pod 最重要的一个事实是：它只是一个逻辑概念。有了Pod，我们可以说Network Namespace和Volume 不是container A 的，也不是Container B的，而是Pod 的。哪怕Container A/B 还没有启动，我们也可以 配置Network Namespace和Volume**。以network namespace 为例，为什么需要一个pause 容器参见[《Container-Networking-Docker-Kubernetes》笔记](http://qiankunli.github.io/2018/10/11/docker_to_k8s_network_note.html)
4. Pod 这种“超亲密关系”容器的设计思想，实际上就是希望，当用户想在一个容器里跑多个功能并不相关的应用时，应该优先考虑它们是不是更应该被描述成一个 Pod 里的多个容器。你就可以把整个虚拟机想象成为一个 Pod，把这些进程分别做成分别做成容器镜像，把有顺序关系的容器，定义为 Init Container。 作者提到了tomcat 镜像和war 包（war包单独做一个镜像）的例子，非常精彩，好就好在 分别做镜像 肯定比 镜像坐在一起要方便。



笔者个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)
