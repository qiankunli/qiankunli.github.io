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


## 一些基础知识（待整理）

linux系统的进程结构体有以下几个字段

    struct task_struct {
        struct m_inode * pwd;
    	struct m_inode * root;
    	struct m_inode * executable;				//进程对应可执行文件的i节点
    }
    

### 传统的linux fs加载过程

参见`https://www.kernel.org/doc/Documentation/filesystems/ramfs-rootfs-initramfs.txt`

What is rootfs?

Rootfs is a special instance of ramfs (or tmpfs, if that's enabled), which is
always present in 2.6 systems.it's smaller and simpler for the kernel
to just make sure certain lists can't become empty.Most systems just mount another filesystem over rootfs and ignore it（一般情况下，通过某种文件系统挂载内容至挂载点的话，挂载点目录中原先的内容将会被隐藏）.


What is initramfs?

All 2.6 Linux kernels contain a gzipped "cpio" format archive, which is
extracted into rootfs when the kernel boots up.  After extracting, the kernel
checks to see if rootfs contains a file "init", and if so it executes it as PID
1.  If found, this init process is responsible for bringing the system the
rest of the way up, including locating and mounting the real root device (if
any).  If rootfs does not contain an init program after the embedded cpio
archive is extracted into it, the kernel will fall through to the older code
to locate and mount a root partition, then exec some variant of /sbin/init
out of that.

所以一个linux的启动过程经历了rootfs ==> 挂载initramfs ==> 挂载磁盘上的真正的fs

为什么要有initrd？

linux系统在启动时，会执行文件系统中的`/sbin/init`进程完成linux系统的初始化，执行`/sbin/init`进程的前提是linux内核已经拿到了存在硬盘上的系统镜像文件（加载设备驱动，挂载文件系统）。linux 发行版必须适应各种不同的硬件架构，将所有的驱动编译进内核是不现实的。Linux发行版在内核中只编译了基本的硬件驱动
，在 linux内核启动前，boot loader会将存储介质中的initrd文件(cpio是其中的一种)加载到内存，内核启动时会在访问真正的根文件系统前先访问该内存中的initrd文件系统，执行initrd文件系统的某个文件（不同的linux版本差异较大），扫描设备，加载驱动。

### docker container fs的加载过程

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