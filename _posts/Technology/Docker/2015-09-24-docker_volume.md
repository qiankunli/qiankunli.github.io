---

layout: post
title: docker volume
category: 技术
tags: Docker
keywords: Docker volume

---

## 简介


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

简单来说，Volume就是目录或者文件，它可以**绕过默认的联合文件系统**，而以正常的文件或者目录的形式存在于宿主机上。换句话说，宿主机和容器建立`/a:/b`的映射，那么对容器`/b`的写入即对宿主机`/a`的写入（反之也可）。

volumn的作用：

- 将容器以及容器产生的数据分离开来

    人们很容易想到volumn是为了持久化数据，其实容器只要你不删除，它就在那里，停掉的容器也可以重新启动运行，所以容器是持久的。
    
    估计也正是因为如此，`docker cp`、`docker commit`和`docker export`还不支持Volume（只是对容器本身的数据做了相应处理）。
    

- 容器间共享数据

## docker volume


    // 创建一个容器，包含两个数据卷
    $ docker run -v /var/volume1 -v /var/volume2 -name Volume_Container ubuntu14.04 linux_command
    // 创建App_Container容器，挂载Volume_Container容器中的数据卷
    $ docker run -t -i -rm -volumes-from Volume_Container -name App_Container ubuntu14.04  linux_command
    // 这样两个容器就可以共用这个数据卷了    
    // 最后可以专门安排一个容器，在应用结束时，将数据卷中的内容备份到主机上
    docker run -rm --volumes-from DATA -v $(pwd):/backup busybox tar cvf /backup/backup.tar /data
    
在默认方式下，volume就是在`/var/lib/docker/volumes`目录下创建一个文件夹，并将该文件夹挂载到容器的某个目录下（以UFS文件系统的方式挂载）。当然，我们也可以指定将主机的某个特定目录（该目录要显式指定）挂载到容器的目录中。

到目前为止，容器的创建/销毁期间来管理Volume（创建/销毁）是唯一的方式。

- 该容器是用`docker rm －v`命令来删除的（-v是必不可少的）。
- `docker run`中使用了`--rm`参数

即使用以上两种命令，也只能删除没有容器连接的Volume。连接到用户指定主机目录的Volume永远不会被docker删除。

## 基于分布式文件系统的volume

与docker容器在网络方面碰到的问题一样，在存储方面docker容器存在着

1. 同一主机两个容器如何共享volume。与网络一样，docker本身就支持
2. 同一个容器跨主机，如何使用先前的volume。新版docker使用overlay网络，可以确保跨主机后，容器的ip不变。

	- volume文件夹同步。比如rsync
	- volume直接使用分布式文件系统，比如glusterfs和ceph。这也可以解决第三个问题
    
3. 跨主机两个容器如何共享volume


在volume使用分布式文件系统，有以下两种方式

1. 如果文件系统支持NFS，则可以将dfs挂载到本地操作系统目录上，docker使用传统方式创建volume
2. 直接使用docker volume plugin,docker通过volumn plugin与dfs交互

	比如`sudo docker run --volume-driver glusterfs --volume datastore:/data alpine touch /data/hello`,具体参见[Docker volume plugin for GlusterFS](https://github.com/calavera/docker-volume-glusterfs)
    
据我估计，假如以前volume直接是通过aufs挂载的方式实现的话，那么现在docker则是通过volume plugin来访问volume数据，只不过通过aufs挂载是volume plugin的一个最简单实现而已。

## 引用

[深入理解Docker Volume（一）][]

[深入理解Docker Volume（二）][]

[Persistence With Docker Containers - Team 1: GlusterFS](http://blog.xebia.com/persistence-with-docker-containers-team-1-glusterfs-2/)

[深入理解Docker Volume（二）]: http://dockone.io/article/129
[深入理解Docker Volume（一）]: http://dockone.io/article/128