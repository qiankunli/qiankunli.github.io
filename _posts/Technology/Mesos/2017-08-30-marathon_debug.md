---

layout: post
title: 从一个marathon的问题开始的
category: 技术
tags: Mesos
keywords: Docker

---

## 问题目录

1. app 一直处于unknown 状态，restart/destroy 失败
2. waiting 状态的项目一直无法部署
3. 创建项目时 Invalid JSON
4. CPU Resources in Docker, Mesos and Marathon 指的是什么


## 1. app 一直处于unknown

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

### 类似问题

marathon 部署一个新的task时，会先部署新的实例，然后干掉老的实例，但老的实例一直干不掉，导致task 一直处于deploying状态，并无法继续响应新的deploy请求。

[Docker executor hangs forever if `docker stop` fails.](https://issues.apache.org/jira/browse/MESOS-6743)

但升级mesos slave版本中后，仍然无法解决，一个暂时妥协的方法是：

1. `systemctl stop marathon`
2. 在zookeeper中，`rmr /marathon/state/xx/marathon_app_name`
3. `systemctl stop marathon`

2018.05.22补充， mesos 升级为1.6.0 时，可以强杀 该容器，算是变相解决了该问题。

## 2 waiting 状态的项目一直无法部署

集群明明有空闲资源，但waiting状态的项目就是不调度到空闲的主机上

1. 分析marathon task的部署有没有特殊的限定条件，符合限定条件的主机是否还有资源

2. [Marathon/Mesos 集群排错记录](http://www.ituring.com.cn/article/264014)

根据Mesos state API (http://ip:5050/state)得到当前Mesos集群的所有状态信息的Json文件。

## 3 创建项目时 Invalid JSON


	{"message":"Invalid JSON","details":[{"path":"/id","errors":["error.pattern"]}]}
	
原因：marathon Application name 不允许出现下划线


## 升级带来的api变化

marathon从1.5.x版本开始支持单独的Networking 配置，该配置与原来的docker network配置不能共存。[Networking](https://mesosphere.github.io/marathon/docs/networking.html)

## 通过restful api 控制marathon和messos

[如何用curl 来访问MESOS Scheduler HTTP API](http://geek.csdn.net/news/detail/68985)

teardown framework。该操作非常危险，但执行完毕，将marathon 重新启动后，很多mesos 和marathon 数据不一致的问题，都解决了

	curl -XPOST http://192.168.60.8:5050/master/teardown -d 'frameworkId='$@''
	


mesos kill task（执行失败，还在找原因）

	curl -vv --no-buffer -XPOST -H "Content-Type:application/json" -H "Mesos-Stream-Id:3f055808-1fad-4400-ba9f-9817bfb1df2f-0000" -d '{"framework_id":{"value" : "3f055808-1fad-4400-ba9f-9817bfb1df2f-0000"},"type":"KILL","kill":{"task_id":{"value":"docker-war-demo2.06c168dc-01bf-11e8-bcd1-2a0c412ba8a6"},"agent_id":{"value":"e357a322-224e-4fe3-9a7e-69e9eb5642d1-S14"}}}' http://192.168.60.8:5050/api/v1/scheduler
 
 marathon kill task
 
	curl -XPOST  -H "Content-Type:application/json" -d '{"ids":["docker-war-demo2.06c168dc-01bf-11e8-bcd1-2a0c412ba8a6"]}' http://192.168.60.8:8080/v2/tasks/delete
	
## 反例操作

marathon 显示一个 task 是unhealthy，但对应物理机docker 容器及mesos 进程都找不到，笔者的土方法是：到marathon的zk上删除对应app的数据，然后restart marathon。然后有一次公司停电，unhealthy 的任务太多，笔者干脆zk 操作`rmr /marathon`，然后所有的任务都不见了。后来没办法，根据zk snapshot恢复的zk。

针对集群的么一个节点，恢复zk时，先备份zk data目录，然后查看集群上的所有snapshot，根据snapshot的创建时间，找到合适的文件。删除集群所有节点 data目录中其它的snapshot，启动所有节点zkServer 服务

## 引用

[Mesos源码分析](http://www.cnblogs.com/popsuper1982/p/5926724.html)






	
	