---

layout: post
title: 关于docker image的那点事儿
category: 技术
tags: Docker
keywords: Docker image registry

---
## 简介

* TOC
{:toc}

## 多阶段构建

[Use multi-stage builds](https://docs.docker.com/develop/develop-images/multistage-build/) **multi-stage builds 的重点不是multi-stage  而是 builds**

先使用docker 将go文件编译为可执行文件

	FROM golang:1.7.3
	WORKDIR /go/src/github.com/alexellis/href-counter/
	COPY app.go .
	RUN go get -d -v golang.org/x/net/html \
	  && CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o app .
	  
因为笔者一直是直接做镜像，所以没这种感觉。不过这么做倒有一点好处：可执行文件的实际运行环境 可以不用跟 开发机相同。学名叫同构/异构镜像构建。

然后将可执行文件 做成镜像

	FROM alpine:latest  
	RUN apk --no-cache add ca-certificates
	WORKDIR /root/
	COPY app .
	CMD ["./app"]  

## 和ssh的是是非非

2018.12.01 补充 [ssh连接远程主机执行脚本的环境变量问题](http://feihu.me/blog/2014/env-problem-when-ssh-executing-command-on-remote/)

背景：

1. 容器启动时会运行sshd，所以可以ssh 到容器
2. 镜像dockerfile中 包含`ENV PATH=${PATH}:/usr/local/jdk/bin`
2. `docker exec -it container bash` 可以看到 PATH 环境变量中包含 `/usr/local/jdk/bin`
3. `ssh root@xxx` 到容器内，观察 PATH 环境变量，则不包含  `/usr/local/jdk/bin`

这个问题涉及到 bash的四种模式

1. 通过ssh登陆到远程主机  属于bash 模式的一种：login + interactive
2. 不同的模式，启动shell时会去查找并加载 不同而配置文件，比如`/etc/bash.bashrc`、`~/.bashrc` 、`/etc/profile` 等
3. login + interactive 模式启动shell时会 第一加载`/etc/profile`
4. `/etc/profile` 文件内容默认有一句 `export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`

所以 `docker exec `可以看到 正确的PATH 环境变量值，而ssh 到容器不可以，解决方法之一就是 制作镜像时 向`/etc/profile` 追加 一个export 命令

## build 过程

    cid=$(docker run -v /foo/bar debian:jessie) 
    image_id=$(docker commit $cid) 
    cid=$(docker run $image_id touch /foo/bar/baz) 
    docker commit $(cid) my_debian

image的build过程，粗略的说，就是以容器执行命令（`docker run`）和提交更改（`docker commit`）的过程

## COPY VS ADD

将文件添加到镜像中有以下两种方式：

- COPY 方式
     
        COPY resources/jdk-7u79-linux-x64.tar.gz /tmp/
        RUN tar -zxvf /tmp/jdk-7u79-linux-x64.tar.gz -C /usr/local
        RUN rm /tmp/jdk-7u79-linux-x64.tar.gz
 
- ADD 方式 
  
        ADD resources/jdk-7u79-linux-x64.tar.gz /usr/local/
        

两者效果一样，但COPY方式将占用三个layer，并大大增加image的size。一开始用ADD时，我还在奇怪，为什么docker自动将添加到其中的`xxx.tar.gz`解压，现在看来，能省空间喔。

## tag

假设tomcat镜像有两个tag

- tomcat:7
- tomcat:6

当你`docker push tomcat`，不明确指定tag时，docker会将`tomcat:7`和`tomcat:6`都push到registry上。

所以，当你打算让docker image name携带版本信息时，版本信息加在name还是tag上，要慎重。


## local storage

以virtualbox ubuntu 14.04为例

![Alt text](/public/upload/docker/local_storage.png)

repositories-aufs这个文件输出的内容，跟”docker images”输出的是“一样的”。

graph中，有很多文件夹，名字是image/container的id。文件夹包括两个子文件：

- json                该镜像层的描述，有的还有“parent”表明上一层镜像的id
- layersize           该镜像层内部文件的总大小

aufs和vfs，一个是文件系统，一个是文件系统接口，从上图每个文件（夹）的大小可以看到，这两个文件夹是实际存储数据的地方。

## docker镜像与容器存储目录

参见[docker 镜像与容器存储目录结构精讲][]

## docker 镜像下载加速

两种方案

1. 使用private registry
2. 使用registry mirror,以使用daocloud的registry mirror为例，假设你的daocloud的用户名问`lisi`，则`DOCKER_OPTS=--registry-mirror=http://lisi.m.daocloud.io`

    
   
## 引用

[Where are Docker images stored?][]
    
    




[Where are Docker images stored?]: http://blog.thoward37.me/articles/where-are-docker-images-stored/
[docker 镜像与容器存储目录结构精讲]: http://blog.csdn.net/wanglei_storage/article/details/50299491