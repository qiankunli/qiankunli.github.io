---
layout: post
title: docker快速入门 
category: blog
---
# docker快速入门 #
## 前言 ##
<p>本文是关于我对docker的一些理解，将持续更新，如有错误和建议，请及时反馈到qiankun.li@qq.com。</p>
## docker概述 ##
<p>
Docker是一个开源的引擎，可以轻松的为任何应用创建一个轻量级的、可移植的、自给自足的容器。概念上讲，它有点像个虚拟机，允许多个应用使用单个强劲机器，同时保持每个应用各自不同的具体配置，不会干扰其他应用。与虚拟机不同的是，应用原生地运行在 Linux 内核下，每个应用与其他应用隔离，在操作系统下面也隔离。
</p>
<p>
各位想必都用过虚拟机软件，大概都对虚拟机的资源占用情况颇有微词。虚拟机（类似于virtualbox之类）提供的功能还是比较全的，基本上整了一台新的主机出来。但如果我们不需要那么多的功能，比如宿主机和虚拟机都是linux系统，那就有机会减少一些设计，最终减少资源占用，提高效率。docker就是这样，它提供虚拟机的一些功能，没有虚拟机全面，但比虚拟机更高效。
</p>

## docker安装 ##
### windows下v1.0.1安装 ###
1. 准备
	1. windows7
	2. docker-install.exe，下载地址[https://github.com/boot2docker/windows-installer/releases](https://github.com/boot2docker/windows-installer/releases)
2. 安装过程
	1. 运行安装程序
	2. 安装过程中会附带安装virtualbox和git，如果您已安装此软件，可以取消选中
	3. 安装完毕后，可以运行程序。如果成功，可以看到命令行：`docker@bootdocker~$` <br/>
	![Alt text](/imags/blog/boot2docker_start.png) <br/>
## docker操作 ##
### image和container概念 ###
<p>image和container是docker中很重要的两个概念，docker对应的docker命令，主要就是对这两个“实体”进行操作。</p>
<p>刚上大学的时候，很流行用一键ghost装系统，那个GHO镜像装好后，除了windows系统外，还装了好多软件，真是非常方便。docker中的image也有点这么点意思。首先会有一个base image，在base image的基础上做一些改动，比如装个软件啥的，就形成了新的有个人特色的image，新的image的可以传给别人使用。</p>
<p>我们运行一个image，就会产生一个container。就像我们安装一个iso，就会有一个操作系统可以运行一样。而container实际上就是docker虚拟出来的主机了，然后我们就可以像操作电脑一样操作container。</p>

### image管理 ###

#### 如何制作image ####


1. 通过Dockerfile
	dockerfile记录了该image所属的base image，以及对base image所做的操作及改动，通过dockerfile文件，我们可以docker build出一个新的image文件。dockerfile的语法可以参见相关文档。
2. 通过commit container
	我们可以先docker run运行一个image，在对应的container中进行个性化更改，然后docker commit该container。contaienr可以跟踪我们对其做的改动，并在原来image的基础上生成新的image。


#### 增删改查image ####

我们本地的image可以由docker命令来管理，下面是相关的一些命令：


1. docker images，列出已有的images
2. docker rmi imageName/imageId，删除image，前提所有运行该image的container已被删除掉
3. docker build -t tagName path，根据path下的Dockerfile创建新的image
4. docker tag imageName/imageId，更改某个image的tag

#### image库 ####
docker的开发团队不只是要做一个软件，还想做一个社区。我们可以在github上分享我们的源代码，也可以分享我们的image。下面是相关的一些操作：


1. docker search imageName，查询库中关于imageName的库
2. docker pull imageName，从库中拉取iamgeName到本地
3. docker push imageName，将imageName上传到库中
### container管理 ###
#### 运行container ####
1. 简单运行
	> docker run imageName echo "hello world"
2. 运行image并进入bash
	> docker run -i -t imageName /bin/bash
3. 运行image并对外映射端口
	> docker run -i -t -p 2022:22 imageName /bin/bash
	由此，既可以在docker本机的2022端口访问container的22端口。
4. 以后台方式运行image
	> docker run -d -p 41880:80 imageName apache2ctl start FOREGROUND
    以此方式，container运行后，将不提供tty与用户交互。
#### 增删改查container ####
如果docker run 算是增加container的话，其他相关命令如下：


1. docker ps，列出container
2. docker rm containerId，删除containerId对应的container
3. docker start containerId，启动一个已经exited的container
4. docker attach containerId，进入一个container
5. docker stop containerId，停止一个正在运行的container
## 访问和文件共享 ##
上述内容涉及到的很多的细节，下面我来讲一下在docker中，host主机和container以及container之间的如何进行访问和文件共享。我们知道，传统的虚拟方式虚拟出来一个完整的“计算机”，在一定配置下，虚拟出来的计算机之间可以自由的进行访问和文件传输，那么docker出来的container如何实现这种效果呢？
### host和container之间的联系 ###


1. “运行container”部分已经讲到了host和container之间可以进行端口映射。以container运行apache服务为例，假设执行`docker run -i -t -p 9080:80 imageName /bin/bash`，我们便可以通过host上的9080端口来访问container的apache服务。
2. 除端口映射外，host与container还可以进行文件共享，假设执行`docker run -i -t -p 9080:80  -v /home/docker/git:/root/git imageName /bin/bash`，对host上的/home/docker/git的更改将同步到container的/root/git目录，反之亦然。
### container之间的联系 ###


1. docker提供container linking功能。

	1. 运行一个db container `docker run -i -t --name db -P imageName /bin/bash`
	2. 运行一个app container `docker run -i -t --name app --link db:db imageName /bin/bash`
由此，app container将能够获取db container的环境变量值，db container的相关信息也将添加到app contaienr的/etc/hosts文件下，
两个container也因此可以协同工作。

2.  contaienr之间的文件共享
	container之间进行文件共享有多种方式，最简单地一种就是两个contaienr和所在host共享同一个文件。
### 我们可以用docker做什么 ###
这是一个很开放的问题，
