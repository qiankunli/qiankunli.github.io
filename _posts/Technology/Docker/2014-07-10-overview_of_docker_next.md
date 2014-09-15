---
layout: post
title: docker进一步学习
category: 技术
tags: Docker
keywords: Docker Container Image
---

## 1 前言 ##
下面讲述的是我在使用docker的过程中一些小技巧，不太系统。

## 查看后台运行的container相关情况##

Container后台运行时，都会执行一个启动脚本，我们可以使用`docker logs ContainerID`来跟踪这个脚本的执行情况。当然，这个脚本需要有一些必要的输出，来为我们提供判断。

## 查看container的IP地址 ##

我们可以使用 `docker inspect ContainerID`来观察container的很多信息，其中经常使用的就是`docker inspect ContainerID | grep IPADDRESS`来查看Container的IP地址。

## 为container设置root用户密码 ##

Container后台运行时，我们通常需要“SSH”进去，进行一些必要的操作，而Container root用户的密码实际是随机生成的。So，这就要求我们在制作Image时，加上这么一句`RUN echo 'root:docker' |chpasswd`，这样root用户的密码就固定了。

## 使用fig ##
如果为模拟一个系统，需要启动多个Container，那就得写一个脚本，执行多个`docker run`命令了，然而，这样做或许不太清晰。于是就有了fig工具，我们可以使用`fig.yml`来表示需要执行的image，示例如下:

	postgresql:  
  		image: ImageA
 		ports:
    		- :22
    		- 5432:5432
  		environment:
    		DB: bmc
    		USER: bmc
    		PASS: bmc
  		volumes:
    		- /tmp/postgresql:/postgresql
	karaf:  
  		image: ImageB
  		ports:
    		- :22
    		- 8055:8055
  		links:
    		- ImageA:imagea

然后执行`fig up -d`，这两个container便可以愉快的执行了。

## 清除掉所有正在运行的container ##
使用命令`docker rm -f $(docker ps -aq)`，便可以清除掉所有正在运行的contaienr，当然，这个命令比较暴力，小心“误伤”。

## 待续 ##