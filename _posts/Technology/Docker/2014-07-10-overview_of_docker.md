---
layout: post
title: docker快速入门
category: 技术
tags: Docker
---

## 1 前言 ##
本文是关于我对docker的一些理解，将持续更新，如有错误和建议，请及时反馈到qiankun.li@qq.com。本文更类似于一个知识点的总结，具体的细节请参见官方文档 [https://docs.docker.com](https://docs.docker.com) 。如果你事先对docker并不了解，建议先完成[http://www.slideshare.net/larrycai/learn-docker-in-90-minutes](http://www.slideshare.net/larrycai/learn-docker-in-90-minutes)的内容，通过实际操作对docker有一个感性的认识。
<p>
文中提到的“docker主机”、“docker虚拟机”等可以认为是安装docker服务的linux主机/虚拟机。
</p>


## 2 docker概述 ##
<p>
Docker是一个开源的引擎，可以轻松的为任何应用创建一个轻量级的、可移植的、自给自足的容器。从概念上讲，它有点像个虚拟机，允许多个应用使用单个强劲机器，同时保持每个应用各自不同的具体配置，不会干扰其他应用。与虚拟机不同的是，应用原生地运行在 Linux 内核下，每个应用与其他应用隔离，在操作系统下面也隔离。
</p>
<p>
各位都用过虚拟机软件，想必都对虚拟机的资源占用情况颇有微词。虚拟机（类似于virtualbox之类）提供的功能还是比较全的，基本上整了一台新的主机出来。但如果我们不需要那么多的功能，比如宿主机和虚拟机都是linux系统，那就有机会减少一些设计，最终减少资源占用，提高效率。docker就是这样，它提供一部分虚拟机的功能，没有虚拟机全面，但比虚拟机更高效。
</p>

## 3 docker安装 ##
### 3.1 windows下v1.1.1安装 ###
1. 准备
	* windows7
	* docker-install.exe，下载地址[https://github.com/boot2docker/windows-installer/releases](https://github.com/boot2docker/windows-installer/releases)
2. 安装过程
	1. 运行安装程序
	2. 安装过程中会附带安装virtualbox和git，如果您已安装此软件，可以取消选中
	3. 安装完毕后，可以运行程序。如果成功，可以看到命令行：<br/>
	`docker@bootdocker~$`<br/>
	![Alt text](/public/upload/boot2docker_start.png)<br/>
## 4 docker操作 ##
### 4.1 image和container ###
<p>image和container是docker中很重要的两个概念，docker程序提供的docker命令，主要就是对这两个“实体”进行操作。</p>
<p>还记得上大学的时候，很流行用一键ghost装系统，GHO镜像还原到系统后，除了windows系统本身，还自动装好了许多软件，做了一些配置（比如将my documents设置到D盘），真是非常方便。docker中的image也有这么点意思。首先会有一个base image（类似于windows纯净版），在base image的基础上做一些改动，比如装个软件啥的，就形成了新的有个人特色的image，新的image的可以传给别人使用。</p>
<p>我们运行一个image，就会产生一个container。就像我们用一键ghost还原一个GHO，就会有一个操作系统可以运行一样。container实际上就是docker虚拟出来的linux，操作container就像你操作一般的linux系统一样。</p>

### 4.2 image管理 ###
#### 4.2.1 什么是Dockerfile ####
<p>
“Dockerfile是一个image的表示，可以通过Dockerfile来描述构建镜像的步骤。”说的接地气点，dockerfile类似于数据库的日志，根据日志我们知道数据库从时刻1到时刻2发生了什么，由此可以恢复或到达数据库某个时刻的状态。已知image1，我们在dockerfile中记录对image1的改动，便可以根据dockerfile build出image2。也因为dockerfile，image1和image2便具备了父子关系。有了父子，自然也可以搞出来兄弟关系，我们可以使用`docker images --tree`查看image之间的树形家族结构。
</p>
<p>
dockerfile方便了image的传播，只要有同一个base image，我们下载一个“日志文件”，便可以运行对方制作的image。“一个不包括恶意行为的dockerfile” + “一个可靠地base image” = “一个可靠好用的image”。
</p>

#### 4.2.2 如何制作image ####
制作image有两种方式：

1. 通过Dockerfile
    dockerfile记录了目标image所属的初始image，以及对初始image所做的操作及改动，通过dockerfile文件，我们可以docker build出一个新的image文件。dockerfile的语法可以参见相关文档。
2. 通过commit container
	我们可以先docker run运行一个image，在对应的container中进行个性化更改，然后docker commit该container。contaienr可以跟踪我们对其做的改动，并在原来image的基础上生成新的image。


#### 4.2.3 增删改查image ####

我们本地的image可以由docker命令来管理，下面是相关的一些命令：


1. `docker images`，列出已有的images
    ![Alt text](/public/upload/docker_images.png)<br/>
2. `docker rmi redhat-base:6.4`，删除image，前提所有运行该image的container已被删除掉
3. `docker build -t redhat-base:6.4 /path`，根据path下的Dockerfile创建新的image
4. `docker tag redhat-base:6.4`，更改某个image的tag

#### 4.2.4 image库 ####
docker的开发团队不只是要做一个软件，还想做一个社区。我们可以在github上分享我们的dockfile或image，寻找并pull我们需要的image（这一点跟git很像）。下面是相关的一些操作：


1. `docker search larrycai/postgresql`，查询库中关于imageName的库
2. `docker pull larrycai/postgresql`，从库中拉取iamgeName到本地
3. `docker push larrycai/postgresql`，将imageName上传到库中

### 4.3 container管理 ###
image和container的关系很像程序和进程之间的关系。
#### 4.3.1 运行container ####
* 简单运行，执行完命令后退出<br/>
	`docker run redhat-base:6.4 echo "hello world"`<br/>
* 运行image并进入bash，通过bash控制container<br/>
	`docker run -i -t redhat-base:6.4 /bin/bash`<br/>
* 运行image并对外映射端口<br/>
	`docker run -i -t -p 2022:22 redhat-base:6.4 /bin/bash`<br/>
	由此，即可以在docker本机的2022端口访问container的22端口。<br/>
* 以后台方式运行image<br/>
	`docker run -d -p 41880:80 redhat-base:6.4 apache2ctl start FOREGROUND`<br/>
    这时，container运行后，将不提供tty与用户交互。用户可以通过docker主机的41880端口访问container的apache2服务。

#### 4.3.2 增删改查container ####
如果docker run 算是增加container的话，其他相关命令如下：

1. `docker ps -a`，列出所有container
	![Alt text](/public/upload/docker_ps_a.png)<br/>
2. `docker rm dc126312903f`，删除containerId对应的container
3. `docker start dc126312903f`，启动一个已经exited的container
4. `docker attach dc126312903f`，从docker主机进入一个已exited的container
5. `docker stop dc126312903f`，停止一个正在运行的container

## 5 访问和文件共享 ##
我们知道，传统的虚拟方式整出来一个完整的“计算机”，在一定配置下，虚拟出来的计算机之间以及虚拟机与宿主机之间可以自由的互相访问和文件共享（或传输），那么docker出来的container如何实现这种效果呢？
### 5.1 docker主机和container之间的联系 ###


1. 4.3.1部分已经讲到了docker主机和container之间可以进行端口映射。以container运行apache服务为例，执行命令<br/>
	`docker run -i -t -p 9080:80 imageName /bin/bash`<br/>
	我们便可以通过docker主机上的9080端口来访问container的apache服务。
2. 除端口映射外，docker主机与container还可以进行文件共享，执行命令<br/>
	`docker run -i -t -p 9080:80  -v /home/docker/git:/root/git imageName /bin/bash`<br/>
	对docker主机上/home/docker/git的更改将同步到container的/root/git目录，反之亦然。


### 5.2 container之间的联系 ###


1. docker提供container linking功能。
	* 运行一个db container `docker run -i -t --name db -P imageName /bin/bash`
	* 再运行一个app container `docker run -i -t --name app --link db:db imageName /bin/bash`<br/>
由此，app container将能够获取db container的环境变量值，db container的相关信息也将添加到app contaienr的/etc/hosts文件下，两个container也因此可以协同工作。关于container linking具体信息请参见[https://docs.docker.com/userguide/dockerlinks/](https://docs.docker.com/userguide/dockerlinks/)。
2. contaienr之间的文件共享
	container之间进行文件共享有多种方式，最简单地一种就是，两个contaienr和所在docker主机共享同一个文件。


## 6 一些细节 ##
### 6.1 docker如何和windows宿主机文件共享 ###
virtualbox使用docker自带的iso无法使docker虚拟机与windows主机共享文件，一老外不服，自己维护了一个网站[https://medium.com/boot2docker-lightweight-linux-for-docker/boot2docker-together-with-virtualbox-guest-additions-da1e3ab2465c](https://medium.com/boot2docker-lightweight-linux-for-docker/boot2docker-together-with-virtualbox-guest-additions-da1e3ab2465c)，我们可以从这里下载到docker相应版本的iso，使用这里的iso，docker虚拟机与windows主机可以share folder。前文提到docker虚拟机可以与container文件共享，再结合此处，我们便可以实现windows、docker主机和container之间的文件共享。
### 6.2 container网络访问问题 ###
一个比较郁闷的事是container有时会无法访问网络，这是docker的一个bug。所以，每次出现问题时，就得有劳大家亲自操刀，在docker下执行：<br/>
    `sudo udhcpc`<br/>
	`sudo /etc/init.d/docker restart`<br/>
在大多数情况下可以解决这个问题。如果不愉快还是发生了，亲，重启虚拟机吧！
## 7 我们可以用docker做什么 ##
这是一个很开放的问题，这里我揣测两点：


1. 计算机界的先驱们呕心沥血的解决了程序的可移植性，比如java的“一次编写，处处运行”。但随着系统越来越复杂，节点越来越多，配置越来越多，移植一套系统到新的环境上也慢慢成为一个“很有含量”的工作。举个最简单的例子，笔者为了在github上写这个博客，需要一套装有jekyll环境的系统。jekyll依赖ruby和其它不知道干啥的程序，windows下安装jekyll，那是各种坑。linux下，配repo源，install各种程序，别说不好找这样的网页，就算找到了，各种莫名其妙的错，你懂的呀！最后，我找了一个配好jekyll环境的image，docker run一下，直接ok，我是写博客的，可不是来搭环境的。
2. 启动一个虚拟机需要多长时间？一个真实的linux启动一次需要多长时间？你的笔记本可以同时运行几个虚拟机？运行了虚拟机之后，还能流畅的运行其他程序么？要不要体验一下一两秒中进入“虚拟机”感觉？要不要看看同时运行十几个“虚拟机”是什么样子？你想不想用自己的笔记本搭一个小集群？