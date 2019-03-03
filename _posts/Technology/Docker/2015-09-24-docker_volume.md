---

layout: post
title: docker volume
category: 技术
tags: Docker
keywords: Docker volume

---

## 简介

背景材料 [linux 文件系统](http://qiankunli.github.io/2018/05/19/linux_file_mount.html)

## 从AUFS说起

以下引用自[深入理解Docker Volume（一）](http://dockone.io/article/128)

先谈下Docker的文件系统是如何工作的。Docker镜像是由多个文件系统（只读层）叠加而成。当我们启动一个容器的时候，Docker会加载只读镜像层并在其上添加一个读写层。如果运行中的容器修改了现有的一个已经存在的文件，那该文件将会从读写层下面的只读层复制到读写层，该文件的只读版本仍然存在，只是已经被读写层中该文件的副本所隐藏。当删除Docker容器，并通过该镜像重新启动时，之前的更改将会丢失。在Docker中，只读层及在顶部的读写层的组合被称为Union File System（联合文件系统）。

那么**容器为什么使用AUFS作为文件系统呢？**

假设容器不使用AUFS作为文件系统，那么根据image创建container时，便类似于Virtualbox根据box文件生成vm实例，将box中的关于Linux文件系统数据整个复制一套（要是不复制，a容器更改了fs的内容，就会影响到b容器），这个过程的耗时还是比较长的。想要复用image中的文件数据，就得使用类似UFS系统。**这也是docker启动速度快的一个重要因素（**除了实际启动的是一个进程）。


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

[DockOne技术分享（五十七）：Docker容器对存储的定义（Volume 与 Volume Plugin）](http://dockone.io/article/1257)提到：我们要深刻理解的是**Docker容器是承载应用的，是对应用环境的抽象而不是对OS运行环境的抽象。**Docker容器天生设计就是为了应用的运行环境打包，启动，迁移，弹性拓展，所以Docker容器一个最重要的特性就是disposable，是可以被丢弃处理，稍瞬即逝的。而应用访问的重要数据可不是disposable的，这些重要数据需要持久化的存储保持。Docker提出了Volume数据卷的概念就是来应对数据持久化的。

## docker volume

    // 创建一个容器，包含两个数据卷
    $ docker run -v /var/volume1 -v /var/volume2 -name Volume_Container ubuntu14.04 linux_command
    // 创建App_Container容器，挂载Volume_Container容器中的数据卷
    $ docker run -t -i -rm -volumes-from Volume_Container -name App_Container ubuntu14.04  linux_command
    // 这样两个容器就可以共用这个数据卷了    
    // 最后可以专门安排一个容器，在应用结束时，将数据卷中的内容备份到主机上
    docker run -rm --volumes-from DATA -v $(pwd):/backup busybox tar cvf /backup/backup.tar /data
    
在默认方式下，volume就是在`/var/lib/docker/volumes`目录下创建一个文件夹，并将该文件夹挂载到容器的某个目录下（以UFS文件系统的方式挂载）。当然，我们也可以指定将主机的某个特定目录（该目录要显式指定）挂载到容器的目录中。

    docker run -v /container/dir imagename command
    docker run -v /host/dir:/container/dir imagename command
    docker run -v dir:/container/dir imagename command
    
第三种方式相当于`docker run -v /var/lib/docker/volumes/dir:/container/dir imagename command`

到目前为止，容器的创建/销毁期间来管理Volume（创建/销毁）是唯一的方式。

- 该容器是用`docker rm －v`命令来删除的（-v是必不可少的）。
- `docker run`中使用了`--rm`参数

即使用以上两种命令，也只能删除没有容器连接的Volume。连接到用户指定主机目录的Volume永远不会被docker删除。

## 补充材料

docker关于存储方面的总体的理念是：**分层（因为不想像virtualbox一样copy整个镜像），层之间有父子关系。**

分层内容的“镜像图（image graph）”代表了各种分层之间的关系，用来处理这些分层的驱动就被叫做“图驱动（graphdriver）”。docker1.2.0 源码中，驱动接口如下


	type Driver interface {
		String() string
		Create(id, parent string) error
		Remove(id string) error
		Get(id, mountLabel string) (dir string, err error)
		Put(id string)
		Exists(id string) bool
		Status() [][2]string
		Cleanup() error
	}
	
换句话说，接口表述了docker上层操作的要求。基于docker的分层理念，镜像数据以层为单位来组织，根据文件系统的不同，数据的内容不同，实现Driver interface的算法不同，这就是graphdriver。




## docker volume plugin

[Docker使用OpenStack Cinder持久化volume原理分析及实践](https://zhuanlan.zhihu.com/p/29905177)，几个要点

1. Docker通过volume实现数据的持久化存储以及共享
2. Docker创建的volume只能用于当前宿主机的容器使用，不能挂载到其它宿主机的容器中，这种情况下只能运行些无状态服务，对于需要满足HA的有状态服务，则需要使用分布式共享volume持久化数据，保证宿主机挂了后，容器能够迁移到另一台宿主机中。而Docker本身并没有提供分布式共享存储方案，而是通过插件(plugin)机制实现与第三方存储系统对接集成
3. 最重要的是：If a plugin registers itself as a VolumeDriver when activated, it must provide the Docker Daemon with writeable paths on the host filesystem.Docker不能直接读写外部存储系统，而必须把存储系统挂载到宿主机的本地文件系统中，Docker当作本地目录挂载到容器中，换句话说，只要外部存储设备能够挂载到本地文件系统就可以作为Docker的volume。
4. docker daemon与plugin daemon通信的API ，部分

    * VolumeDriver.Mount : 挂载一个卷到本机，Docker会把卷名称和参数发送给参数。**插件会返回一个本地路径给Docker，这个路径就是卷所在的位置。Docker在创建容器的时候，会将这个路径挂载到容器中**。
    * VolumeDriver.Path : 一个卷创建成功后，Docker会调用Path API来获取这个卷的路径，随后Docker通过调用Mount API，让插件将这个卷挂载到本机。 
    * VolumeDriver.Unmount : 当容器退出时，Docker daemon会发送Umount API给插件，通知插件这个卷不再被使用，插件可以对该卷做些清理工作（比如引用计数减一，不同的插件行为不同）。 
    * VolumeDriver.Remove : 删掉特定的卷时调用，当运行”docker rm -v”命令时，Docker会调用该API发送请求给插件。 

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
    

从另一个角度划分

|类型|单机|集群环境|优缺点|
|---|---|---|---|
|create volume|-v支持；Dockerfile支持|使用docker plugin driver|需要安装插件、但适用范围更大|
|mount host dir|-v支持；Dockerfile不可以|类似于lizardfs，将分布式文件系统作为本机的一个host dir|简单，但docker file的中的volume就无法弄了|


## docker container fs的加载过程

docker cotainer fs的演化过程:rootfs(read only image) ==> read-write filesystem（初始状态下是空的） ==> volume（这就不是一个文件系统，只是部分目录的覆盖了）

## 小结

上述谈的内容比较杂，上述说这么东西，主要是为了三点，这两点之间没什么必然关系

1. 隔离，mount namespace。你在新的namespace中执行mount命令，不会影响其它namespace。上述docker container fs挂载的演化才不会影响其它container。
2. 分层，分层是为了复用
3. 分离容器数据与容器产生的数据，volume。docker doc的说法是bypasses the Union File System, independent of the container’s life cycle.Docker therefore never automatically deletes volumes when you remove a container, nor will it “garbage collect” volumes that are no longer referenced by a container.


## 引用

[深入理解Docker Volume（一）][]

[深入理解Docker Volume（二）][]

[Persistence With Docker Containers - Team 1: GlusterFS](http://blog.xebia.com/persistence-with-docker-containers-team-1-glusterfs-2/)

[深入理解Docker Volume（二）]: http://dockone.io/article/129
[深入理解Docker Volume（一）]: http://dockone.io/article/128