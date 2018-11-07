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

## 背景

对于springboot项目，一开始是用`java -jar `方式容器中启动，并作为容器的主进程。但测试环境，经常代码逻辑可能有问题，导致主进程失败，容器启动失败，进而触发marathon/k8s健康检查失败，进而不断重启容器。开发呢也一直抱怨看不到“事故现场”。所以针对这种情况，直观的想法是 不让`java -jar` 作为容器的主进程，进而产生一个在容器中运行多进程的问题。

但容器中运行多进程，跟 one process per container 的理念相悖，我们就得视图探寻下来龙去脉了。

## 为什么推荐一个容器一个进程？

stack exchange [Why it is recommended to run only one process in a container?](https://devops.stackexchange.com/questions/447/why-it-is-recommended-to-run-only-one-process-in-a-container) 有一系列回答

[Run Multiple Processes in a Container](https://runnable.com/docker/rails/run-multiple-processes-in-a-container) 也提了三个advantages

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

||僵尸进程回收|处理SIGTERM信号|alpine 安装大小|专用镜像|备注|
|---|---|---|---|---|---|
|sh/bash|支持|不支持|0m||脚本中可以使用exec 顶替掉sh/bash 自身|
|Supervisor|待确认|支持|79m||
|runit|待确认|支持|31m| [phusion/baseimage-docker](https://github.com/phusion/baseimage-docker)|
|s6|||33m||

## 一个容器多个进程的可能选择

### 自定义脚本

官方 [Run multiple services in a container](https://docs.docker.com/config/containers/multi-service_container/)

### systemd

[CHAPTER 3. USING SYSTEMD WITH CONTAINERS](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux_atomic_host/7/html/managing_containers/using_systemd_with_containers)

[Running Docker Containers with Systemd](https://container-solutions.com/running-docker-containers-with-systemd/)

[Do you need to execute more than one process per container?](https://gomex.me/2018/07/21/do-you-need-to-execute-more-than-one-process-per-container/)

### supervisor

官方 [Run multiple services in a container](https://docs.docker.com/config/containers/multi-service_container/)

[Admatic Tech Blog: Starting Multiple Services inside a Container with Supervisord](https://medium.com/@SaravSun/admatic-tech-blog-starting-multiple-services-inside-a-container-with-supervisord-16e3beb55916)

使用 

supervisord.conf

	[supervisord]
	nodaemon=true
	logfile=/dev/stdout
	loglevel=debug
	logfile_maxbytes=0
	
	[program:pinggoogle]
	command=ping admatic.in
	autostart=true
	autorestart=true
	startsecs=5
	stdout_logfile=NONE
	stderr_logfile=NONE
	
Dockerfile
	
	FROM ubuntu
	...
	COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
	...
	CMD ["/usr/bin/supervisord"]

### runit

[Run Multiple Processes in a Container](https://runnable.com/docker/rails/run-multiple-processes-in-a-container)

A fully­ powered Linux environment typically includes an ​init​ process that spawns and supervises other processes, such as system daemons. The command defined in the CMD instruction of the Dockerfile is the only process launched inside the Docker container, so ​system daemons do not start automatically, even if properly installed.

[runit - a UNIX init scheme with service supervision](http://smarden.org/runit/)


使用

Dockerfile

	FROM phusion/passenger-­ruby22
	
	...
	
	#install custom bootstrap script as runit service
	COPY myapp/start.sh /etc/service/myapp/run
	
	
在这个Dockerfile 中，CMD 继承自 base image。 将`myapp/start.sh` 拷贝到 容器的 `/etc/service/myapp/run`	文件中即可 被runit 管理，runit 会管理 `/etc/service/` 下的应用（目录可配置），即 Each service is associated with a service directory

### s6

[Managing multiple processes in Docker containers](https://medium.com/@beld_pro/managing-multiple-processes-in-docker-containers-455480f959cc)

## Docker-friendliness image

与其在init 进程工具的选型上挣扎，是否有更有魄力的工具呢？

1. docker 原生支持多进程，比如阿里的 pouch
2. 原生支持多进程的 镜像

github 有一个 [phusion/baseimage-docker](https://github.com/phusion/baseimage-docker) 笔者2018.11.7 看到时，有6848个star。 该镜像有几个优点：

1. Modifications for Docker-friendliness.
2. Administration tools that are especially useful in the context of Docker.
3. Mechanisms for easily running multiple processes, without violating the Docker philosophy.  具体的说，The Docker developers advocate running a single logical service inside a single container. But we are not disputing that. Baseimage-docker advocates running multiple OS processes inside a single container, and a single logical service can consist of multiple OS processes. 


什么叫 ubuntu 对 Docker-friendliness？（待体会）

1. multi-user
2. multi-process

[phusion/baseimage-docker](https://github.com/phusion/baseimage-docker)  的image 是基于ubuntu，笔者试着用alpine + runit + sshd 实现了一个简洁的base image，具体情况待实践一段时间再交流。

个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)