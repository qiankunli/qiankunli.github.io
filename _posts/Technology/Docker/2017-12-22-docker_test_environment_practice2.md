---

layout: post
title: 测试环境docker化实践2
category: 技术
tags: Docker
keywords: Docker,macvlan

---


## 简介

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