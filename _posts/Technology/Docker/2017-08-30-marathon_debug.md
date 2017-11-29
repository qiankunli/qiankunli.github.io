---

layout: post
title: 从一个marathon的问题开始的
category: 技术
tags: Docker
keywords: Docker

---


## 问题一描述

基础环境：

1. centos7主机
1. marathon + mesos + docker组成的容器集群
2. 通过jenkins向marathon发布创建or更新app指令
3. 通过[portainer](https://github.com/portainer/portainer)查看容器

问题：

一个app的某一个instance，配置了健康检查，但处于unknown状态。然后对该app进行更新或restart操作时，该app一直处于deploying状态，并无法响应后面的命令。直观感觉就是，该app卡住了。

### debug过程

marathon 日志`/var/log/messages`

	 No kill ack received for instance [deploy-to-docker-xima-accounting-order-rpc-test.marathon-a4b86920-8c5f-11e7-9837-f2f3c189fa2c], retrying (205 attempts so far) (mesosphere.marathon.core.task.termination.impl.KillServiceActor:marathon-akka.actor.default-dispatcher-137)
	
可以看到marathon kill请求已经发出205次了，mesos没有ack响应。

找到对应的主机的messos日志`/var/log/messages`

	Failed to get resource statistics for executor 'deploy-to-docker-xima-accounting-order-rpc-test.a4b86920-8c5f-11e7-9837-f2f3c189fa2c' of framework d637e32a-a1df-43eb-adaf-b1d2e3d6235a-0000: Failed to run 'docker -H unix:///var/run/docker.sock inspect mesos-9f8a309f-649d-42c7-a3d5-a2b6ef038af9-S48.930d5968-b10f-426c-a5bb-1a45742e65c6': exited with status 1; stderr='Error: No such object: mesos-9f8a309f-649d-42c7-a3d5-a2b6ef038af9-S48.930d5968-b10f-426c-a5bb-1a45742e65c6
	
可以看到`docker -H unix:///var/run/docker.sock inspect mesos-9f8a309f-649d-42c7-a3d5-a2b6ef038af9-S48.930d5968-b10f-426c-a5bb-1a45742e65c6` 也就是`docker inspect container_id`执行失败，我们来寻找个container id，`docker ps -a | grep container_id`，没有找到。

但是对于一个mesos slave，一个容器的执行往往对应着两个进程

1. mesos-docker-executor，启动docker容器，并监控容器状态，响应来自mesos-slave（或者说之上的mesos framework = marathon）的指令
2. mesos-docker-executor的子进程，其command内容为`docker run xxx image_name`。从mesos代码上看， mesos-docker-executor启动该子进程命令docker启动容器，然后一直存在（cpp代码不是特别懂）。

然后我们发现，docker容器不在，对应的mesos-docker-executor和mesos-docker-executor的子进程还在。

所以，事情就是：marathon向mesos询问task `deploy-to-docker-xima-accounting-order-rpc-test.marathon-a4b86920-8c5f-11e7-9837-f2f3c189fa2c`的状态，mesos slave的mesos executor执行了一下`docker inspect xx`表示拿不到数据，然后陷入循环。

因此，可以判断，有人绕过marathon + mesos 体系干掉了容器，而marathon + mesos 发现拿不到容器信息后，无所作为。

办法：干掉docker容器对应的mesos-docker-executor及其子进程

### 问题2

集群明明有空闲资源，但waiting状态的项目就是不调度到空闲的主机上

1. 分析marathon task的部署有没有

[Marathon/Mesos 集群排错记录](http://www.ituring.com.cn/article/264014)

根据Mesos state API (http://ip:5050/state)得到当前Mesos集群的所有状态信息的Json文件。

## 引用

[Mesos源码分析](http://www.cnblogs.com/popsuper1982/p/5926724.html)






	
	