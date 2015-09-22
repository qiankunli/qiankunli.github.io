---

layout: post
title: 关于docker image的那点事儿
category: 技术
tags: Docker
keywords: Docker image registry

---
## 简介

## COPY VS ADD

在Dockerfile中，一开始用ADD时，我还在奇怪，为什么docker自动将添加到其中的`xxx.tar.gz`解压，以至于后来开始使用COPY

- COPY 方式
     
        COPY resources/jdk-7u79-linux-x64.tar.gz /tmp/
        tar -zxvf /tmp/jdk-7u79-linux-x64.tar.gz -C /usr/local
        rm /tmp/jdk-7u79-linux-x64.tar.gz
 
- ADD 方式 
  
        ADD resources/jdk-7u79-linux-x64.tar.gz /usr/local/
        

两者效果一样，但COPY方式将占用三个layer，并大大增加image的size，因此要纠正这个陋习。


## image的存储格式

## docker registry

## docker registry remote api
   
    
    


