---

layout: post
title: 关于docker image的那点事儿
category: 技术
tags: Docker
keywords: Docker image registry

---
## 简介

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

当你`docker push tomcat`，docker会将`tomcat:7`和`tomcat:6`都push到registry上。

所以，当你打算让docker image name携带版本信息时，版本信息加在name还是tag上，要慎重。


## local storage

以virtualbox ubuntu 14.04为例

![Alt text](/public/upload/docker/local_storage.png)

repositories-aufs这个文件输出的内容，跟”docker images”输出的是“一样的”。

graph中，有很多文件夹，名字是image/container的id。文件夹包括两个子文件：

- json                该镜像层的描述，有的还有“parent”表明上一层镜像的id
- layersize           该镜像层内部文件的总大小

aufs和vfs，一个是文件系统，一个是文件系统接口，从上图每个文件（夹）的大小可以看到，这两个文件夹是实际存储数据的地方。



## remote storage （未完待续）

## docker registry remote api


## docker 镜像下载加速

两种方案

1. 使用private registry
2. 使用registry mirror,以使用daocloud的registry mirror为例，假设你的daocloud的用户名问`lisi`，则`DOCKER_OPTS=--registry-mirror=http://lisi.m.daocloud.io`

    
   
## 引用

[Where are Docker images stored?][]
    
    




[Where are Docker images stored?]: http://blog.thoward37.me/articles/where-are-docker-images-stored/