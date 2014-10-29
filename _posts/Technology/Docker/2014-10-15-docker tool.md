---
layout: post
title: docker进一步学习
category: 技术
tags: Docker
keywords: Docker Container Image
---

## 前言 ##
下面讲述的是我在使用docker的过程中一些小技巧，不太系统。

## 1 使用fig ##
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

同时，还可以使用`fig [-f xx.yml] ps`来查看当前运行的contaienr，以及`fig [-f xx.yml] logs`来查看后台运行的contaienr的输出。

注意，fig是由python编写完成，所以`fig.yml`文件内容应符合python语法要求，尤其是缩进。

## expect

docker中推荐一个contaienr只运行一个service。这样，对于一个复杂的系统，就需要多个container去协调运行。而我们在使用时，往往希望只进入一个container，便完成对整个系统的操作，所以expect必不可少。

## dockerize ##

http://jasonwilder.com/blog/2014/10/13/a-simple-way-to-dockerize-applications/

## Coreos ##

http://www.blogjava.net/yongboy/archive/2013/08/26/403325.html