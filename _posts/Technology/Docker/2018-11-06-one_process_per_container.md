---

layout: post
title: 一个容器多个进程
category: 技术
tags: Docker
keywords: 持续交付

---

## 简介

* TOC
{:toc}

## 为什么要一个容器一个进程？

stack exchange [Why it is recommended to run only one process in a container?](https://devops.stackexchange.com/questions/447/why-it-is-recommended-to-run-only-one-process-in-a-container)


理由要找的话有很多，比较喜欢一个回答：As in most cases, it's not all-or-nothing. The guidance of "one process per container" stems from the idea that containers should serve a distinct purpose. For example, a container should not be both a web application and a Redis server.

There are cases where it makes sense to run multiple processes in a single container, as long as both processes support a single, modular function.



## 一个容器多个进程有什么风险


[理解Docker容器的进程管理](https://yq.aliyun.com/articles/5545)

1. 当PID1进程结束之后，Docker会销毁对应的PID名空间，并向容器内所有其它的子进程发送SIGKILL。
2. PID1进程对于操作系统而言具有特殊意义

	* 操作系统的PID1进程是init进程，以守护进程方式运行，是所有其他进程的祖先，具有完整的进程生命周期管理能力。
	* 如果它没有提供某个信号的处理逻辑，那么与其在同一个PID名空间下的进程发送给它的该信号都会被屏蔽。这个功能的主要作用是防止init进程被误杀。
	* 当一个子进程终止后，它首先会变成一个“失效(defunct)”的进程，也称为“僵尸（zombie）”进程，等待父进程或系统收回（reap）。在Linux内核中维护了关于“僵尸”进程的一组信息（PID，终止状态，资源使用信息），从而允许父进程能够获取有关子进程的信息。如果不能正确回收“僵尸”进程，那么他们的进程描述符仍然保存在系统中，系统资源会缓慢泄露。
	* 如果父进程已经结束了，那些依然在运行中的子进程会成为“孤儿（orphaned）”进程。在Linux中Init进程(PID1)作为所有进程的父进程，会维护进程树的状态，一旦有某个子进程成为了“孤儿”进程后，init就会负责接管这个子进程。当一个子进程成为“僵尸”进程之后，如果其父进程已经结束，init会收割这些“僵尸”，释放PID资源。
	
	
docker stop  对PID1进程 的要求

1. 容器的PID1进程需要能够正确的处理SIGTERM信号来支持优雅退出
2. 如果容器中包含多个进程，需要PID1进程能够正确的传播SIGTERM信号来结束所有的子进程之后再退出。


综上，如果一个容器有多个进程，可选的实践方式为：

1. 多个进程关系对等，由一个init 进程管理，比如supervisor、systemd
2. 一个进程（A）作为主进程，拉起另一个进程（B）

	* A 先挂，因为容器的生命周期与 主进程一致，则进程B 也会被kill 结束
	* B 先挂，则要看A 是否具备僵尸进程的处理能力（大部分不具备）。若A 不具备，B 成为僵尸进程，容器存续期间，僵尸进程一致存在。
	* A 通常不支持 SIGTERM

所以第二种方案通常不可取，对于第一种方案，则有init 进程的选型问题

||僵尸进程回收|处理SIGTERM信号|
|---|---|---|
|sh/bash|支持|不支持|
|Supervisor|大部分不支持|支持|

## 一个容器多个进程的最佳实践

[CHAPTER 3. USING SYSTEMD WITH CONTAINERS](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux_atomic_host/7/html/managing_containers/using_systemd_with_containers)

[Running Docker Containers with Systemd](https://container-solutions.com/running-docker-containers-with-systemd/)

[Do you need to execute more than one process per container?](https://gomex.me/2018/07/21/do-you-need-to-execute-more-than-one-process-per-container/)

[Run multiple services in a container](https://docs.docker.com/config/containers/multi-service_container/)


其它 [Optimizing Spring Boot apps for Docker](https://openliberty.io/blog/2018/06/29/optimizing-spring-boot-apps-for-docker.html)

个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)