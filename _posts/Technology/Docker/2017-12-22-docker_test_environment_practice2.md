---

layout: post
title: 测试环境docker化实践2
category: 技术
tags: Docker
keywords: Docker,macvlan

---


## 简介



将一个web app部署到docker 容器上。除了编排问题之外

1. 网络模型的选择。公司测试环境遍布两三个办公楼，macvlan直接依赖两层，对交换机配置要求较高。
2. docker+os 对swap 分区的支持。现在的表现是，同样规模的物理机，直接部署比通过docker部署支持的服务更多，后者比较吃资源。
3. marathon + mesos 方案通过zk 沟通，mesos-slave 有时无法杀死docker-slave

mesos+marathon方案也取得很多经验：

1. macvlan 直接在ip 上打通 容器与物理机的访问，使用起来比较便捷。
2. marathon app + instance + cpu + memory 对app的抽象与控制比较直观，用户输入与实际marathon.json 映射比较直观。

应用背景：web项目，将war包拷贝到tomcat上，启动tomcat运行


## docker build 比较慢


jenkins流程

1. maven build war
2. 根据war build image
3. push image
4. 调度marathon

问题

docker build 慢，docker push 慢
	
build 慢的原因：

1. level 多
2. war 包大

build 慢的解决办法

1. 在jenkins机器上，docker run -d, docker cp,docker cimmit 的方式替代dockerfile
2. 先调度marathon在目标物理机上启动容器，然后将war包拷到目标物理机，进而拷贝到容器中，启动tomcat

	* 优点：完全规避了docker build
	* 缺点：每个版本的war包没有镜像，容器退化为了一个执行机器， 镜像带来的版本管理、回滚等不再支持

	
[Docker 持续集成过程中的性能问题及解决方法](http://oilbeater.com/docker/2016/01/02/use-docker-performance-issue-and-solution.html)

## 性能不及物理机(未完成)

表现为过快的耗尽物理机资源：

cpu设置问题

[Docker: 限制容器可用的 CPU](https://www.cnblogs.com/sparkdev/p/8052522.html)

[Docker 运行时资源限制](http://blog.csdn.net/candcplusplus/article/details/53728507)

## 网络方案选择与多机房扩展问题（未完成）

