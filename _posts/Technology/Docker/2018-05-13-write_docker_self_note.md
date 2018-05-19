---

layout: post
title: 《自己动手写docker》笔记
category: 技术
tags: Docker
keywords: Docker

---


## 简介（未完成）

背景材料

[docker中涉及到的一些linux知识](http://qiankunli.github.io/2016/12/02/linux_docker.html)

## 启动流程

![](/public/upload/docker/docker_note_1.png)

改图是两个角色，并不准确，还未准确理解mydocker，mydocker runCommand进程，mydocker initCommand进程，以及用户自定义command的作用关系。但大体可以分为 mydocker 一方，容器一方。

据猜测应该是，mydocker runCommand 进程 启动 了 initCommand 进程（相当于 `mydocker run` 内部执行了 `mydocker init`），同时为两者沟通建立管道，并为其配置cgroup。
initComand 启动后挂载文件系统，从管道中读取用户输入的command，用户输入的command 顶替掉 initCommand
	

## 其它

[Docker：一场令人追悔莫及的豪赌](https://mp.weixin.qq.com/s?__biz=MzA5OTAyNzQ2OA==&mid=2649697704&idx=1&sn=5f7ef3d2f9d5e2c7b33b1085559fd0f5&chksm=889312cbbfe49bddaa66ba5a6f761531a0baf8f6ee572bd24e2720066987e3dc1ff3fafacc51&mpshare=1&scene=23&srcid=0515E2lfioYH9HzxGsd8SmWZ%23rd) 要点如下：

1. 将所有依赖关系、一切必要配置乃至全部必要的资源都塞进同一个二进制文件，从而简化整体系统资源机制。 如果具备这种能力，docker 便不是必要的。

	* 静态链接库方式。一些语言，比如go，可以将依赖一起打包，生成一个毫无依赖的二进制文件。
	* macos 则从 更广泛的意义上实现了 该效果
2. Docker 的重点 不在于 提供可移植性、安全性以及资源管理与编排能力，而是 标准化。docker 做的事情以前各种运维工程师也在做，只是在A公司的经验无法复制到B公司上。
3. 在我看来，Docker有朝一日将被定性为一个巨大的错误。其中最强有力的论据在于，即使最终成为标准、始终最终发展成熟，Docker也只是为科技行业目前遭遇的种种难题贴上一张“创可贴”



[Is K8s Too Complicated?](http://jmoiron.net/blog/is-k8s-too-complicated/) 未读完

	* Kubernetes is a complex system. It does a lot and brings new abstractions. Those abstractions aren't always justified for all problems. I'm sure that there are plenty of people using Kubernetes that could get by with something simpler.
	* That being said, I think that, as engineers, we tend to discount the complexity we build ourselves vs. complexity we need to learn. 个人也觉得 k8s 管的太宽了，实现基本功能，外围的我们自己做，也比学它的弄法要好。

	
	
[An architecture of small apps](http://www.smashcompany.com/technology/an-architecture-of-small-apps)