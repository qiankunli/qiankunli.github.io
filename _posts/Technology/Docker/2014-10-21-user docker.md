---
layout: post
title: docker使用
category: 技术
tags: Docker
keywords: Docker Container Image
---

## 1 前言 ##
下面讲述的是我在使用docker的过程中一些小技巧，不太系统。

## 2 查看后台运行的container相关情况##

Container后台运行时，都会执行一个启动脚本，我们可以使用`docker logs ContainerID`来跟踪这个脚本的执行情况。当然，这个脚本需要有一些必要的输出，来为我们提供判断。

## 3 查看container的IP地址 ##

我们可以使用 `docker inspect ContainerID`来观察container的很多信息，其中经常使用的就是`docker inspect ContainerID | grep IPADDRESS`来查看Container的IP地址。

## 4 为container设置root用户密码 ##

Container后台运行时，我们通常需要“SSH”进去，进行一些必要的操作，而Container root用户的密码实际是随机生成的。So，这就要求我们在制作Image时，加上这么一句`RUN echo 'root:docker' |chpasswd`，这样root用户的密码就固定了。

## 5 保存image到本地 ##

大多数情况下，我们可以从[docker官方registry](https://registry.hub.docker.com/ "")中下载image。同时，也可以将image保存在tar文件，方便在工作中，进行image的分发（当然，在企业中最好建立自己的docker私有registry）。

1. 将image保存为tar文件:`docker save -o TarFileName ImageID`
2. 将tar文件导入为image:·docker load -i TarFileName·

## 6 清除掉所有正在运行的container ##
使用命令`docker rm -f $(docker ps -aq)`，便可以清除掉所有正在运行的contaienr，当然，这个命令比较暴力，小心“误伤”。

## 7 Dockerfile中的ADD ##
`ADD`可以将文件加入到Image中，但不要直接“ADD” “tar.gz”等文件，我也不知道为什么，但直接“ADD”后，运行image时，会发现“tar.gz”文件已被解压过。解决这个问题有两种办法：

1. 我们可以将“tar.gz”放到一个文件夹下，“ADD”这个文件夹，这就避免了前述情况；
2. 使用COPY命令，其功能跟ADD类似，但不会对压缩文件解压。

读者可以根据自己的喜好自由选择。

## 8 Dockerfile中的RUN ##
如果有一两次编译Dockerfile文件的报错经历的话，你会发现“RUN”后的命令，实际是由“sh -c”负责具体执行的。并且，每一次RUN就会使文件系统有一个新的layer。知道这个有什么用呢？我们在写Dockerfile时，不可避免要对其进行更改，更改完后再重新编译，有时候就会很耗时。这时，可以将不会再修改的RUN操作写在Dockerfile的前面，还需要多次修改的RUN操作写在后面。这样，每次编译时，前面编译的cache会被利用到，节省时间。

## 9 volume的速度问题 ##
在windows中运行boot2docker时，我们有时在container中运行一些软件，这些软件的运行需要windows宿主机提供一些文件(比如在`C:/Users/id/git`下)。此时，我们可以通过virtualbox的sharedfolder功能，将共享文件夹挂载到boot2docker-vm的`/mnt/git`下，然后将该目录映射到container的`/root/git`。比如以下指令:
`docker run -it -v /mnt/git:/root/git centos:centos6 bash`。
此时，contaienr操作`/root/git`就如同操作windows的`C:/Users/id/git` 。

但这样做有一个问题，那就是contaienr中的软件直接操作`/root/git`下的文件时，速度非常慢。据我估计，这应该是中间经过几层映射，同时windows的文件系统和contaienr（也就是linux）的文件系统不同额缘故。

解决方案：将`/root/git`下的文件复制到另一个文件夹下，比如`/root/tmp`,使程序在`/root/tmp`下操作文件。麻烦的是，当windows下的文件发生改动时，需要重新复制。为了提高复制速度，在container中可以使用`rsync`命令，比如`rsync -vzrtopgu --delete zabbix-2.2.6/ /root/zabbix-2.2.6/`。

## 10 Dockerfile和初始化脚本 ##

containre在实际应用中，通常是后台运行，多个container相互配合，模拟一个比较大的应用。这时，Dockerfile和初始化脚本要有一个平衡

未完待续

## 11 尽量缩小image的大小 ##

0. base image应该尽可能的小
1. 如果添加了tar.gz,那么解压缩完毕后，可以删除原来的压缩文件
2. 如果tar.gz是从网络上获取的，则可以直接  `curl xxx.tar.gz | tar -C destinationdir -zxvf`
3. 安装软件尽量通过yum apt-get install等方式  

此段未整理

## 12 更改boot2docker-vm 时区##

`docker pull boot2docker/boot2docker`拿到最新的boot2docker image，自己使用Dockerfile

    FROM boot2docker/boot2docker
    RUN cp /usr/share/zoneinfo/Asia/Shanghai $ROOTFS/etc/localtime
    RUN /make_iso.sh
    CMD ["cat", "boot2docker.iso"]
    
大致是这样。制作出新的image，再执行`docker run --rm ImageId > boot2docker.iso`即可得到新的boot2docker.iso。

以此为引子，运行`boot2docker/boot2docker`image 后，其`$ROOTFS`目录下的内容会被写入boot2docker.iso,而这也是我们定制boot2docker.iso的重要渠道。除了上面提到的更改boot2docker-vm的时区外，还可以写入环境变量和自定义boot2docker-vm后，首次执行的脚本。

    FROM boot2docker/boot2docker
    RUN echo "http_proxy=xxxx" >> $ROOTFS/etc/profile
    RUN echo "/scripts/start.sh" >> $ROOTFS/etc/profile
    RUN /make_iso.sh
    CMD ["cat", "boot2docker.iso"]
    
