---
layout: post
title: Kubernetes存储
category: 技术
tags: Kubernetes
keywords: Docker Kubernetes Volume
---

## 简介

* TOC
{:toc}

云原生存储的两个关键领域：
1. Docker 存储卷：容器服务在**单节点**的存储组织形式，关注数据存储、容器运行时的相关技术；
2. K8s 存储卷：关注容器集群的**存储编排**，从应用使用存储的角度关注存储服务。

## Volume 背景介绍

### 从AUFS说起

[云原生存储详解：容器存储与 K8s 存储卷](https://mp.weixin.qq.com/s/7rGrXhlc4-9jgSoVHqcs4A)容器服务之所以如此流行，一大优势即来自于运行容器时容器镜像的组织形式。容器通过复用容器镜像的技术，实现多个容器共享一个镜像资源（**更细一点说是共享某一个镜像层**），避免了每次启动容器时都拷贝、加载镜像文件，这种方式既节省了主机的存储空间，又提高了容器启动效率。**为了实现多个容器间共享镜像数据，容器镜像每一层都是只读的**。

以下引用自[深入理解Docker Volume（一）](http://dockone.io/article/128)先谈下Docker的文件系统是如何工作的。Docker镜像是由多个文件系统（只读层）叠加而成。当我们启动一个容器的时候，Docker会加载只读镜像层并在其上添加一个读写层。写时复制：如果运行中的容器修改了现有的一个已经存在的文件，那该文件将会从读写层下面的只读层复制到读写层，该文件的只读版本仍然存在，只是已经被读写层中该文件的副本所隐藏。一旦容器销毁，这个读写层也随之销毁，之前的更改将会丢失。在Docker中，只读层及在顶部的读写层的组合被称为Union File System（联合文件系统）。

存储驱动是指如何对容器的各层数据进行管理，已达到上述需要实现共享、可读写的效果。常见的存储驱动：

1. AUFS
2. OverlayFS
3. Devicemapper
4. Btrfs
5. ZFS


```
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
```
背景材料 [linux 文件系统](http://qiankunli.github.io/2018/05/19/linux_file_mount.html)

### 为什么要有volume 

[云原生存储详解：容器存储与 K8s 存储卷](https://mp.weixin.qq.com/s/7rGrXhlc4-9jgSoVHqcs4A)容器中的应用读写数据都是发生在容器的读写层，**镜像层+读写层映射为容器内部文件系统**、负责容器内部存储的底层架构。当我们需要容器内部应用和外部存储进行交互时，**需要一个类似于计算机 U 盘一样的外置存储**，容器数据卷即提供了这样的功能。 ==> `容器存储组成：只读层（容器镜像） + 读写层 + 外置存储（数据卷）`

[DockOne技术分享（五十七）：Docker容器对存储的定义（Volume 与 Volume Plugin）](http://dockone.io/article/1257)提到：Docker容器天生设计就是为了应用的运行环境打包，启动，迁移，弹性拓展，所以Docker容器一个最重要的特性就是disposable，是可以被丢弃处理，稍瞬即逝的。而应用访问的重要数据可不是disposable的，这些重要数据需要持久化的存储保持。Docker提出了Volume数据卷的概念就是来应对数据持久化的。

简单来说，Volume就是目录或者文件，它可以**绕过默认的UFS**，而以正常的文件或者目录的形式存在于宿主机上。换句话说，宿主机和容器建立`/a:/b`的映射，那么对容器`/b`的写入即对宿主机`/a`的写入（反之也可）。

 the two main reasons to use Volumes are data persistency and shared resources：

- 将容器以及容器产生的数据分离开来。相比通过存储驱动实现的可写层，数据卷读写是直接对外置存储进行读写，效率更高
- 容器间共享数据


## docker volume

Volume 挂载方式语法：`-v: src:dst:opts`

```
// 创建一个容器，包含两个数据卷
$ docker run -v /var/volume1 -v /var/volume2 -name Volume_Container ubuntu14.04 linux_command
// 创建App_Container容器，挂载Volume_Container容器中的数据卷
$ docker run -t -i -rm -volumes-from Volume_Container -name App_Container ubuntu14.04  linux_command
// 这样两个容器就可以共用这个数据卷了    
// 最后可以专门安排一个容器，在应用结束时，将数据卷中的内容备份到主机上
docker run -rm --volumes-from DATA -v $(pwd):/backup busybox tar cvf /backup/backup.tar /data
```

在默认方式下，volume就是在`/var/lib/docker/volumes`目录下创建一个文件夹，并将该文件夹挂载到容器的某个目录下（以UFS文件系统的方式挂载）。当然，我们也可以指定将主机的某个特定目录（该目录要显式指定）挂载到容器的目录中。

```
docker run -v /container/dir imagename command
docker run -v /host/dir:/container/dir imagename command
docker run -v dir:/container/dir imagename command
```  

第三种方式相当于`docker run -v /var/lib/docker/volumes/dir:/container/dir imagename command`

到目前为止，容器的创建/销毁期间来管理Volume（创建/销毁）是唯一的方式。

- 该容器是用`docker rm －v`命令来删除的（-v是必不可少的）。
- `docker run`中使用了`--rm`参数

即使用以上两种命令，也只能删除没有容器连接的Volume。连接到用户指定主机目录的Volume永远不会被docker删除。bypasses the Union File System, independent of the container’s life cycle.Docker therefore never automatically deletes volumes when you remove a container, nor will it “garbage collect” volumes that are no longer referenced by a container. **Docker 有 Volume 的概念，但对它只有少量且松散的管理（没有生命周期的概念），Docker 较新版本才支持对基于本地磁盘的 Volume 的生存期进行管理**。

## kubernetes volume

A Volume is a directory, possibly with some data in it, which is accessible to a Container. Kubernetes Volumes are similar to but not the same as Docker Volumes.

**A process in a Container sees a filesystem view composed from two sources: a single Docker image and zero or more Volumes**（这种表述方式很有意思）. A Docker image is at the root of the file hierarchy. Any Volumes are mounted at points on the Docker image; Volumes do not mount on other Volumes and do not have hard links to other Volumes. Each container in the Pod independently specifies where on its image to mount each Volume. This is specified a VolumeMounts property.

The storage media (Disk, SSD, or memory) of a volume is determined by the media of the filesystem holding the kubelet root dir (typically `/var/lib/kubelet`)(volumn的存储类型（硬盘，固态硬盘等）是由kubelet所在的目录决定的). There is no limit on how much space an EmptyDir or PersistentDir volume can consume（大小也是没有限制的）, and no isolation between containers or between pods.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: test-container
    image: k8s.gcr.io/busybox
    volumeMounts:   ## 这里的volume 配置可以视为单纯的传递给 container runtime的参数
    - name: cache-volume
      mountPath: /cache
    - name: test-volume
      mountPath: /hostpath
    - name: config-volume
      mountPath: /data/configmap
    - name: special-volume
      mountPath: /data/secret
  volumes:  ## 集群范围内的volume配置，在pod 调度到某个机器上时，k8s 要负责将这些volume 在 Node 上准备好
  - name: cache-volume
    emptyDir: {}
  - name: hostpath-volume
    hostPath:
      path: /data/hostpath
      type: Directory
  - name: config-volume
    configMap:
      name: special-config
  - name: secret-volume
    secret:
      secretName: secret-config
```

[Types of Volumes](https://kubernetes.io/docs/concepts/storage/volumes/#types-of-volumes) 支持十几种类型的Volume

1. Volume 与pod 声明周期相同，不是 Kubernetes 对象，主要用于跨节点或者容器对数据进行同步和共享。 EmptyDir、HostPath、ConfigMap 和 Secret
2. PersistentVolume，为集群中资源的一种，它与集群中的节点 Node 有些相似，PV 为 Kubernete 集群提供了一个如何提供并且使用存储的抽象，与它一起被引入的另一个对象就是 PersistentVolumeClaim(PVC)，这两个对象之间的关系与Node和 Pod 之间的关系差不多。**PVC 消耗了持久卷资源**，而 Pod 消耗了节点上的 CPU 和内存等物理资源。PS：当 Kubernetes 创建一个节点时，它其实仅仅创建了一个对象来代表这个节点，并基于 metadata.name 字段执行健康检查，对节点进行验证。如果节点可用，意即所有必要服务都已运行，它就符合了运行一个 pod 的条件；否则它将被所有的集群动作忽略直到变为可用。

[云原生存储详解：容器存储与 K8s 存储卷](https://mp.weixin.qq.com/s/7rGrXhlc4-9jgSoVHqcs4A)另一种划分方式：
1. 本地存储：如 HostPath、emptyDir，这些存储卷的特点是，数据保存在集群的特定节点上，并且不能随着应用漂移，节点宕机时数据即不再可用；
2. 网络存储：Ceph、Glusterfs、NFS、Iscsi 等类型，这些存储卷的特点是数据不在集群的某个节点上，而是在远端的存储服务上，使用存储卷时需要将存储服务挂载到本地使用；
3. Secret/ConfigMap：这些存储卷类型，其数据是集群的一些对象信息，并不属于某个节点，使用时将对象数据以卷的形式挂载到节点上供应用使用；
4. CSI/Flexvolume：这是两种数据卷扩容方式，可以理解为抽象的数据卷类型。每种扩展方式都可再细化成不同的存储类型；
5. 一种数据卷定义方式，将数据卷抽象成一个独立于 pod 的对象，这个对象定义（关联）的存储信息即存储卷对应的真正存储信息，供 K8s 负载（也就是pod）挂载使用。

因为 PVC 允许用户消耗抽象的存储资源，所以用户需要不同类型、属性和性能的 PV 就是一个比较常见的需求了，在这时我们可以通过 StorageClass 来提供不同种类的 PV 资源，上层用户就可以直接使用系统管理员提供好的存储类型。










