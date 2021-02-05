---

layout: post
title: 关于docker image的那点事儿
category: 技术
tags: Container
keywords: Docker image 

---
## 简介

* TOC
{:toc}

2013 年，Docker 出现了，工程师可以第一次到软件生产环境中定义，通过 Docker image 完成**单机**软件的交付和分发。

## 打包所有依赖

代码分发：在物理机时代， springboot项目普遍都带了一个run.sh文件，不论项目本身的特点如何，开发和运维约定`run.sh start/stop`来启停应用
1. 这说明只有一个jar是运行不起来的
2. 如果我们不是一个java系为主的公司，这么做够么？
到后面，你就发现，run.sh 里可能什么都有，包括依赖库（比如转码程序会安装ffmpeg）、下载文件等，run.sh做到极致：一个应用一个操作系统环境（依赖库、env等），但整个文件岂不是很大？**Docker最大的贡献就是提出了分层镜像的概念**。

想起了大学宿舍时候的装系统：
1. 当说镜像的时候第一反应是啥？一键ghost ==> 装系统 ==> 装qq  ==>  自动启动项目 ==> 装系统顺带装app 
2. App最完整的依赖的OS ==> 为了启动app干脆配上一个系统 ==> 裸操作系统 + docker即可运行任何项目。


## 镜像与操作系统的关系

[并非每个容器内部都能包含一个操作系统](https://mp.weixin.qq.com/s/ALTxkwAXBdKdQLMYJIMrLw)docker镜像不仅能够打包应用，还能打包整个操作系统的文件和目录，记住是操作系统的文件和目录。通过这种方式docker就把一个应用所有的依赖库包括操作系统中的文件和目录都被打包到镜像中。docker正是通过打包操作系统级别的方式，解决了开发到线上环境的一致性。宿主机操作系统只有一个内核，也就是说，所有的容器都依赖这一个内核了？比如我现在有一个需求，我的两个容器运行在同一台宿主机上，但是依赖的内核版本不一样，或者需要配置的内核参数不一样，怎么解决呢？**解决不了**，这也是容器化技术相比于虚拟机的主要缺陷之一。

内核，操作系统和发行版之间的区别：

1. Linux内核是Linux操作系统的核心部分。这就是Linus最初写的。
2. Linux操作系统是内核和用户域（库，GNU实用程序，配置文件等）的组合。
3. Linux发行版是Linux操作系统的特定版本，例如Debian，CentOS或Alpine。

其实linux操作系统中代码包含两部分，一部分是文件目录和配置，另外一部分是内核，这两部分是分开存放的，**内核模块只有在宿主机开机启动时才会加载**。说白了，即使镜像中包含了内核也不会被加载。说到最后，原来镜像只是包含了操作系统的躯干（文件系统），并没有包含操作系统的灵魂（内核）。
**容器中的根文件系统**，其实就是我们做的镜像。

![](/public/upload/container/container_image_overview.png)

## 本地存储

[Where are Docker images stored?](http://blog.thoward37.me/articles/where-are-docker-images-stored/)

`docker inspect xx` GraphDriver 的部分输出
```
"GraphDriver": {
    "Data": {
        "LowerDir": "/var/lib/docker/overlay2/cad72d6e952bbffb754bf3a13af0c401ae1ab743ef4ed0b9994e57ef127c3d29-init/diff:/var/lib/docker/overlay2/6a190d31ec303cc0a4163c2698a38ce449d660265bbab709503a2ac4dde4aa7f/diff:/var/lib/docker/overlay2/05c46a2829a0b37e856434489ef2684c507276697cc325f98225d1d15c84a9bb/diff:/var/lib/docker/overlay2/77aa0717d4e28977139c25a52afb04de25b8bb478670b24f3c8b409b1f1b9495/diff",
        "MergedDir": "/var/lib/docker/overlay2/cad72d6e952bbffb754bf3a13af0c401ae1ab743ef4ed0b9994e57ef127c3d29/merged",
        "UpperDir": "/var/lib/docker/overlay2/cad72d6e952bbffb754bf3a13af0c401ae1ab743ef4ed0b9994e57ef127c3d29/diff",
        "WorkDir": "/var/lib/docker/overlay2/cad72d6e952bbffb754bf3a13af0c401ae1ab743ef4ed0b9994e57ef127c3d29/work"
    },
    "Name": "overlay2"
},
```

1. lower， 这一层里的文件是不会被修改的，你可以认为它是只读的。
2. uppder， 在 OverlayFS 中，如果有文件的创建，修改，删除操作，那么都会在这一层反映出来，它是可读写的。
3. merged，挂载点（mount point）目录，也是用户看到的目录，用户的实际文件操作在这里进行。
4. work，只是一个存放临时文件的目录，OverlayFS 中如果有文件修改，就会在中间过程中临时存放文件到这里。

![](/public/upload/container/container_aufs.jpg)

镜像要复用，所以只能只读。但程序的运行会写文件，所以要可写。OverlayFS/UnionFS 解决这个折中的问题。

## 制作镜像

```sh
cid=$(docker run -v /foo/bar debian:jessie) 
image_id=$(docker commit $cid) 
cid=$(docker run $image_id touch /foo/bar/baz) 
docker commit $(cid) my_debian
```

image的build过程，粗略的说，就是以容器执行命令（`docker run`）和提交更改（`docker commit`）的过程

开发者可以使用一些工具(如Dockerfile)构建出自己的容器镜像、**签名**并上传到互联网上（分发内容一定会有签名， 更进一步签名还可以作为内容寻址），然后需要运行这些软件的人可以通过指定名称（如_example.com/my-app_）下载、验证和运行这些容器。

### build 时使用http代理

2019.4.3 补充

```
docker build --build-arg HTTPS_PROXY='http://userName:password@proxyAddress:port' \
    --build-arg HTTP_PROXY='http://userName:password@proxyAddress:port' \
    -t $IMAGE_NAGE .
```

当你在公司负责维护docker 镜像时，不同镜像的Dockerfile 为了支持协作及版本跟踪 一般会保存在git 库中。制作镜像 通常需要安装类似jdk/tomcat 等可执行文件，这些文件建议使用远程下载的方式（因为git 保存二进制文件 不太优雅），以安装tomcat 为例

```
RUN \
    DIR=/tmp/tomcat && mkdir -p ${DIR} && cd ${DIR} && \
    curl -sLO http://mirrors.tuna.tsinghua.edu.cn/apache/tomcat/tomcat-8/v8.5.40/bin/apache-tomcat-8.5.40.tar.gz &&\
    tar -zxvf apache-tomcat-8.5.40.tar.gz -C /usr/local/ && \
    mv /usr/local/apache-tomcat-8.5.40 /usr/local/tomcat && \
    rm -rf ${DIR}
```

### 多阶段构建

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


## 镜像下载

镜像一般会包括两部分内容，一个是 manifests 文件，这个文件定义了镜像的 元数据，另一个是镜像层，是实际的镜像分层文件。

### docker login

我们在拉取私有镜像之前，要使用 docker login 命令来登录镜像仓库。登录主要就做了三件 事情:
1. 向用户要账户密码
2. docker 访问镜像仓库的 https 地址，并通过挑战 v2 接口来确 认，接口是否会返回 Docker-Distribution-Api-Version 头字段。它的作用跟 ping 差不多，只是确认下 v2 镜像仓库是否在线，以及版本是否匹配。
3. docker 使用用户提供的账户密码，访问 Www-Authenticate 头字段返回的鉴权服务器的地址 Bearer realm。如果这个访问成功，则鉴权服务器会返回 jwt 格式的 token 给 docker，然后 docker 会把账户密码编码并保存在用户目录的 .docker/docker.json 文件里。这个文件作为 docker 登录仓库的 唯一证据，在后续镜像仓库操作中，会被不断的读取并使用。

![](/public/upload/container/pull_image_security.png)

### docker 镜像下载加速

两种方案

1. 使用private registry
2. 使用registry mirror,以使用daocloud的registry mirror为例，假设你的daocloud的用户名问`lisi`，则`DOCKER_OPTS=--registry-mirror=http://lisi.m.daocloud.io`












    
   



   

