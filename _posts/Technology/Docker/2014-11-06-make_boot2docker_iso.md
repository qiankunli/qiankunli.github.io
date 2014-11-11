---
layout: post
title: 定制自己的boot2docker.iso
category: 技术
tags: Docker
keywords: Docker Container boot2docker
---
## 前言

阅读本文，需要你对docker,Dockerfile,boot2docker-vm,virtualbox以及github等有一定了解。文章中如有错误，欢迎大家批评指正。

## 背景

docker刚开始推出的时候，只支持ubuntu。也难怪，docker只支持虚拟linux container，人家本来就不是给windows玩的。但后来估计是为了推广docker的使用，我们可以下载[boot2docker.exe][]，安装并运行它，会在virtualbox上创建一个boot2docker-vm，一个极精简的可以运行docker service的linux系统。

观察boot2docker-vm的设置，我们可以看到，boot2docker-vm以光盘方式启动，iso来自于你PC上的`/c/Users/yourname/.boot2docker/boot2docker.iso`。那么在实践中，我们可以自己制作iso并替换掉它。

## 为什么要定制

当然，你会怀疑这样做的必要性。确实没必要，如果你只是想玩一下docker的话。但考虑到在公司环境中要跟他人协作，情况也许就有所不同了。

笔者曾将一个系统转移到了docker上，并附上两页精美的guide。manager却问：“可以再精简点么？不是所有人都想知道细节的”。好吧，我最后把这个系统在boot2docker-vm上做成了一键启动。大家只要使用这个服务即可，而不用关心它的安装与执行过程。为达到这个目的，定制自己的boot2docker.iso就很有必要了。下面是一些具体的需求：

### 配置代理

如果你的PC运行在公司的内网中，那么很遗憾，你无法直接pull [docker官网][]上的image。So,你需要通过http_proxy环境变量配置代理。进而，如果贵公司有自己私有的docker-registry，还需要配置no_proxy环境变量，确保pull私有库的image时不必通过代理。

### 更改时区

如果在boot2docker-vm中执行`date`命令，你会发现显示时间比实际少了8个小时。在一些情况下，你需要一个正确的时间，相信我。

### 安装软件

boot2docker-vm是一个非常精简的linux，很多linux命令没有安装。

以fig为例，早期的boot2docker-vm并不支持fig命令。即使现在，也是通过运行一个装有fig的容器的方式“委婉”的使用fig，并且这种方式还有一些使用限制。

再假设一个场景，系统需要运行容器A和容器B。容器B的运行依赖容器A中的服务ServiceA。但容器A启动后，serviceA的启动需要一定的时间。如何确保容器B启动时，serviceA已经启动完毕？一个比较low的方案是（如果有更好的方案，请及时告诉我）：

1. 容器A使用`-v`和boot2docker-vm映射一个目录DirA；
2. boot2docker-vm使用inotifywait监控这个目录；
3. serviceA启动完毕后，向DirA中写入一个文件FileA。
4. inotifywait监控到FileA后，即认为serviceA已启动，于是启动容器B。

看来，boot2docker-vm有必要装一个inotifywait。当然，我们可以增加一个容器C运行inotifywait，并监控DirA。

从实现上讲，你也可以运行一个ubuntu-vm，并在ubuntu-vm上安装docker服务来替代boot2docker-vm。这样，安装fig和inotifywait全不是问题。不过你确定所有同事都乐意在virtualbox或vmware上安装一个ubuntu vm。你会问“不是有vagrant么？”好吧，你确定所有同事都想知道vagrant是什么么？他们只想完成今天的工作，其它的还是由我们代劳吧。

### 添加脚本

我们可以自定义一个脚本，比如叫做start.sh，然后配置其在boot2docker-vm启动时执行，它能做很多事。

1. mount sharedfolder
            
    有时候容器的运行需要操作你PC上的一些文件，一方面virtualbox需要配置sharedfolder，另一方面boot2docker-vm需要mount sharedfolder。而后者，start.sh可以代劳喔。
       
2. 触发容器的运行

    在start.sh中写入启动容器的脚本。这时，只要启动boot2docker-vm，就可以触发容器的执行。这样，我们就可以在PC上使用容器提供的服务，使用者不需要操作boot2docker-vm，不需要对docker有任何了解。
    
## 如何定制

### 基本流程

执行如下命令

    $ docker pull boot2docker/boot2docker
    $ docker run --rm boot2docker/boot2docker > boot2docker.iso
    
你就可以得到一个默认的boot2docker.iso。

一言以蔽之，**制作boot2docker.iso的关键是`boot2docker/boot2docker`**。你可以登陆[https://github.com/][]搜索 `boot2docker/boot2docker`repositories。通过运行`boot2docker/boot2docker` image以及对其Dockerfile的观察我发现：`boot2docker/boot2docker` image `$ROOTFS`目录下的 目录结构跟 boot2docker-vm的目录结构一模一样。So，在`boot2docker/boot2docker` image下对`$ROOTFS`所做的更改，都将通过`makeiso.sh`制作到boot2docker.iso中。

### 配置案例

    FROM boot2docker/boot2docker
    ADD start.sh /start.sh
    # 更改boot2docker-vm的欢迎界面
    RUN echo "" >> $ROOTFS/etc/motd; \
        echo "hello boot2docker vm" >> $ROOTFS/etc/motd; \
        echo "" >> $ROOTFS/etc/motd
    # 配置http_proxy代理
    RUN echo "export http_proxy= your proxy" >> $ROOTFS/etc/profile
    # 配置boot2docker-vm时区
    RUN cp /usr/share/zoneinfo/Asia/Shanghai $ROOTFS/etc/localtime
    # 配置启动脚本
    RUN echo "/start.sh" >> $ROOTFS/etc/profile
    RUN /make_iso.sh
    CMD ["cat", "boot2docker.iso"]

看到这里，我相信你可以根据自己的需求制作自己的boot2docker.iso了。

## 其他

本文已授权在[DockerPool][]发表。

[boot2docker.exe]: https://github.com/boot2docker/windows-installer/releases
[docker官网]: https://hub.docker.com
[https://github.com/]: https://github.com/
[http://qiankunli.github.io/]: http://qiankunli.github.io/
[qiankun.li@qq.com]: qiankun.li@qq.com
[DockerPool]: http://www.dockerpool.com/