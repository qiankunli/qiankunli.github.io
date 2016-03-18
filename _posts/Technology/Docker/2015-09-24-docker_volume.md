---

layout: post
title: docker volume
category: 技术
tags: Docker
keywords: Docker volume

---

## 简介（未完待续）


## 从AUFS说起

以下引用自[深入理解Docker Volume（一）](http://dockone.io/article/128)

先谈下Docker的文件系统是如何工作的。Docker镜像是由多个文件系统（只读层）叠加而成。当我们启动一个容器的时候，Docker会加载只读镜像层并在其上添加一个读写层。如果运行中的容器修改了现有的一个已经存在的文件，那该文件将会从读写层下面的只读层复制到读写层，该文件的只读版本仍然存在，只是已经被读写层中该文件的副本所隐藏。当删除Docker容器，并通过该镜像重新启动时，之前的更改将会丢失。在Docker中，只读层及在顶部的读写层的组合被称为Union File System（联合文件系统）。

那么容器为什么使用AUFS作为文件系统呢？假设容器不使用AUFS作为文件系统，那么根据image创建container时，便类似于Virtualbox根据box文件生成vm实例，将box中的关于Linux文件系统数据整个复制一套，这个过程的耗时还是比较长的。想要复用image中的文件数据，就得使用类似UFS系统。这样看来，docker启动容器速度快是因为实际启动的是进程。能够快速创建新的实例，则是因为所有（基于共同image的）容器共享了image的文件。


	# 假设存在以下目录结构
    root@Standard-PC:/tmp# tree
    .
    ├── aufs
    ├── dir1
    │   └── file1
    └── dir2
        └── file2
    # 将dir1和dir2挂载到aufs目录下，这样aufs目录就包含了dir1和dir2包含的文件总和
    root@Standard-PC:/tmp# sudo mount -t aufs -o br=/tmp/dir1=ro:/tmp/dir2=rw none /tmp/aufs
    mount: warning: /tmp/aufs seems to be mounted read-only.
    # 向file1写入文件
    root@Standard-PC:/tmp/aufs# echo hello > file1
    bash: file1: Read-only file system
    # 向file2写入文件
    root@Standard-PC:/tmp/aufs# echo hello > file2
    root@Standard-PC:/tmp/dir2# cat file2 
    hello

## 为什么要有volumn 

为了能够保存（持久化）数据以及共享容器间的数据，Docker提出了Volume的概念。简单来说，Volume就是目录或者文件，它可以**绕过默认的联合文件系统**，而以正常的文件或者目录的形式存在于宿主机上。换句话说，宿主机和容器建立`/a:/b`的映射，那么对容器`/b`的写入即对宿主机`/a`的写入（反之也可）。

## docker volume


    // 创建一个容器，包含两个数据卷
    $ docker run -v /var/volume1 -v /var/volume2 -name Volume_Container ubuntu14.04 linux_command
    // 创建App_Container容器，挂载Volume_Container容器中的数据卷
    $ docker run -t -i -rm -volumes-from Volume_Container -name App_Container ubuntu14.04  linux_command
    // 这样两个容器就可以共用这个数据卷了
    
    // 最后可以专门安排一个容器，在应用结束时，将数据卷中的内容备份到主机上
    docker run -rm --volumes-from DATA -v $(pwd):/backup busybox tar cvf /backup/backup.tar /data
    
在默认方式下，volume就是在`/var/lib/docker/volumes`目录下创建一个文件夹，并将该文件夹挂载到容器的某个目录下（以UFS文件系统的方式挂载）。当然，我们也可以指定将主机的某个特定目录（该目录要显式指定）挂载到容器的目录中。

## 基于分布式文件系统的volume

