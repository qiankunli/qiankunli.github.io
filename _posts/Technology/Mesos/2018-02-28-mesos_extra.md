---

layout: post
title: mesos 的一些tips
category: 技术
tags: Mesos
keywords: Docker, calico

---


## 简介

17年初的时候，mesos和k8s还难分伯仲，18年的时候，mesos 就已经有点尴尬了。但在具体的环境下，mesos 仍有可取之处

1. 大部分互联网公司的大部分业务是 web 项目/rpc server  + tomcat，具体的说，就是用tomcat 做项目容器。一个tomcat  作为 marathon 的 application，提供healthcheck、实例控制等，就够用了。k8s 的 pod 概念没有必要
2. rpc 服务治理框架普遍自带 服务发现机制，k8s的service 没有必要


[Mesos中文手册](https://mesos-cn.gitbooks.io)

## mesos 的配置

### 配置的分类

[mesos 配置向导](https://mesos-cn.gitbooks.io/mesos-cn/content/document/runing-Mesos/Configuration.html)

1. mesos 配置分为：通用配置、master配置、slave配置
2. 配置的配置方式

	* `/usr/sbin/mesos-slave` 参数
	* 特定目录下的文件 ，文件内容为 配置值。 比如`/etc/mesos-slave/containerizers`，文件内容为 `docker,mesos`

### 配置的位置

[Mesos 安装与使用](https://yeasy.gitbooks.io/docker_practice/content/mesos/installation.html#%E8%BD%AF%E4%BB%B6%E6%BA%90%E5%AE%89%E8%A3%85)

mesos 配置目录有三个

1. `/etc/mesos` 主节点和从节点都会读取的配置文件，其中最关键的就是zk
2. `/etc/mesos-master/`只有主节点会读取的配置，等价于启动 mesos-master 命令时候的默认选项
3. `/etc/mesos-slave/`只有从节点会读取的配置，等价于启动 mesos-slave 命令时候的默认选项

此外，/etc/default/mesos、/etc/default/mesos-master、/etc/default/mesos-slave 这三个文件中可以存放一些环境变量定义，Mesos 服务启动之前，会将这些环境变量导入进来作为启动参数。
	
查看配置位置最稳妥的办法是:查看对应systemd serivce文件中指定的 EnvironmentFile

## marathon 

marathon 配置参数的方式与mesos 基本相同

配置目录为 `/etc/marathon/conf`（需要手动创建），此外默认配置文件在 `/etc/default/marathon`。

主要就是配置master 和 zk 两个参数

###  marathon status

[Marathon Web Interface](https://mesosphere.github.io/marathon/docs/marathon-ui.html#application-status-reference)

Application Status

1. Running
2. Deploying, Whenever a change to the application has been requested by the user. Marathon is performing the required actions, which haven’t completed yet.
3. Suspended, An application with a target instances of 0 and whose running tasks count is 0.
4. Waiting,Marathon is waiting for offers from Mesos. 
5. Delayed,An app is considered delayed whenever too many tasks of the application failed in a short amount of time. Marathon will pause this deployment and retry later. 

Health Status

1. Healthy
2. Unhealthy
3. Staged
4. Unknown
5. Overcapacity
6. Unscheduled

## mesos 常见问题

### 重启一直失败

 可能原因：

1. 磁盘空间不够
2. 将mesos 以前的日志干掉， 重新启动，查看mesos日志可以发现

		Log file created at: 2018/10/09 11:28:23
		Running on machine: xx
		Log line format: [IWEF]mmdd hh:mm:ss.uuuuuu threadid file:line] msg
		E1009 11:28:23.090240  9958 slave.cpp:7290] EXIT with status 1: Failed to perform recovery: Collect failed: Detected duplicate pid 1526 for container ed7f5d07-b6ca-432c-8a4d-05ff5dda9407
		If recovery failed due to a change in configuration and you want to
		keep the current agent id, you might want to change the
		`--reconfiguration_policy` flag to a more permissive value.
		
		To restart this agent with a new agent id instead, do as follows:
		rm -f /var/lib/mesos/meta/slaves/latest
		This ensures that the agent does not recover old live executors.
		
	执行`rm -f /var/lib/mesos/meta/slaves/latest` 后重启，发现成功