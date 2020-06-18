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

## Volume 背景介绍

### ## 从AUFS说起

以下引用自[深入理解Docker Volume（一）](http://dockone.io/article/128)

先谈下Docker的文件系统是如何工作的。Docker镜像是由多个文件系统（只读层）叠加而成。当我们启动一个容器的时候，Docker会加载只读镜像层并在其上添加一个读写层。如果运行中的容器修改了现有的一个已经存在的文件，那该文件将会从读写层下面的只读层复制到读写层，该文件的只读版本仍然存在，只是已经被读写层中该文件的副本所隐藏。当删除Docker容器，并通过该镜像重新启动时，之前的更改将会丢失。在Docker中，只读层及在顶部的读写层的组合被称为Union File System（联合文件系统）。

那么**容器为什么使用AUFS作为文件系统呢？**

假设容器不使用AUFS作为文件系统，那么根据image创建container时，便类似于Virtualbox根据box文件生成vm实例，将box中的关于Linux文件系统数据整个复制一套（要是不复制，a容器更改了fs的内容，就会影响到b容器），这个过程的耗时还是比较长的。想要复用image中的文件数据，就得使用类似UFS系统。**这也是docker启动速度快的一个重要因素（**除了实际启动的是一个进程）。
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

### 为什么要有volumn 

[DockOne技术分享（五十七）：Docker容器对存储的定义（Volume 与 Volume Plugin）](http://dockone.io/article/1257)提到：Docker容器天生设计就是为了应用的运行环境打包，启动，迁移，弹性拓展，所以Docker容器一个最重要的特性就是disposable，是可以被丢弃处理，稍瞬即逝的。而应用访问的重要数据可不是disposable的，这些重要数据需要持久化的存储保持。Docker提出了Volume数据卷的概念就是来应对数据持久化的。

简单来说，Volume就是目录或者文件，它可以**绕过默认的UFS**，而以正常的文件或者目录的形式存在于宿主机上。换句话说，宿主机和容器建立`/a:/b`的映射，那么对容器`/b`的写入即对宿主机`/a`的写入（反之也可）。

 the two main reasons to use Volumes are data persistency and shared resources：

- 将容器以及容器产生的数据分离开来

    人们很容易想到volumn是为了持久化数据，其实容器只要你不删除，它就在那里，停掉的容器也可以重新启动运行，所以容器是持久的。
    
    估计也正是因为如此，`docker cp`、`docker commit`和`docker export`还不支持Volume（只是对容器本身的数据做了相应处理）。
    

- 容器间共享数据

### docker volume

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

### kubernetes volume

A Volume is a directory, possibly with some data in it, which is accessible to a Container. Kubernetes Volumes are similar to but not the same as Docker Volumes.

A Pod specifies which Volumes its containers need in its ContainerManifest property.

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
    volumeMounts:
    - name: cache-volume
      mountPath: /cache
    - name: test-volume
      mountPath: /hostpath
    - name: config-volume
      mountPath: /data/configmap
    - name: special-volume
      mountPath: /data/secret
  volumes:
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

1. Volume 与pod 声明周期相同，不是 Kubernetes 对象，主要用于跨节点或者容器对数据进行同步和共享。 EmptyDir、HostPath、ConfigMap 和 Secret
2. PersistentVolume，为集群中资源的一种，它与集群中的节点 Node 有些相似，PV 为 Kubernete 集群提供了一个如何提供并且使用存储的抽象，与它一起被引入的另一个对象就是 PersistentVolumeClaim(PVC)，这两个对象之间的关系与Node和 Pod 之间的关系差不多。**PVC 消耗了持久卷资源**，而 Pod 消耗了节点上的 CPU 和内存等物理资源。PS：当 Kubernetes 创建一个节点时，它其实仅仅创建了一个对象来代表这个节点，并基于 metadata.name 字段执行健康检查，对节点进行验证。如果节点可用，意即所有必要服务都已运行，它就符合了运行一个 pod 的条件；否则它将被所有的集群动作忽略直到变为可用。

因为 PVC 允许用户消耗抽象的存储资源，所以用户需要不同类型、属性和性能的 PV 就是一个比较常见的需求了，在这时我们可以通过 StorageClass 来提供不同种类的 PV 资源，上层用户就可以直接使用系统管理员提供好的存储类型。

## PV 和 PVC

与CPU 和 Mem 这些资源相比，“存储”对k8s 来说更像是“外设”，k8s 提供统一的“总线”接入。[Kata Containers 创始人带你入门安全容器技术](https://mp.weixin.qq.com/s/w2SkC6TuSBqurvAae0RAUA)OCI规范规定了容器之中应用被放到什么样的环境下、如何运行，比如说容器的根文件系统上哪个可执行文件会被执行，是用什么用户执行，需要什么样的 CPU，有什么样的内存资源、**外置存储**，还有什么样的共享需求等等。

### 为何引入PV、PVC以及StorageClass？

[Kubernetes云原生开源分布式存储介绍](https://mp.weixin.qq.com/s/lHY6cvaag1TdIist-Xg0Bg)早期Pod使用Volume的写法

```yaml
apiVersion: v1
kind: Pod
metadata:
labels:
    role: web-frontend
spec:
containers:
- name: web
    image: nginx
    ports:
    - name: web
        containerPort: 80
    volumeMounts:
        - name: ceph
        mountPath: "/usr/share/nginx/html"
volumes:
- name: ceph
	capacity:
	  storage: 10Gi
    cephfs:
      monitors:
	  - 172.16.0.1:6789
	  - 172.16.0.2:6789
	  - 172.16.0.3:6789
	  path: /ceph
      user: admin
	  secretRef:
	    name: ceph-secret
```

这种方式至少存在两个问题：

1. Pod声明与底层存储耦合在一起，每次声明Volume都需要配置存储类型以及该存储插件的一堆配置，如果是第三方存储，配置会非常复杂。
2. 开发人员的需求可能只是需要一个20GB的卷，这种方式却不得不强制要求开发人员了解底层存储类型和配置。

于是引入了PV（Persistent Volume），PV其实就是把Volume的配置声明部分从Pod中分离出来，PV的spec部分几乎和前面Pod的Volume定义部分是一样的由运维人员事先创建在 Kubernetes 集群里待用

```yaml
apiVersion: v1
kind: PersistentVolume
metadata: 
  name: cephfs-pv
spec:
  capacity:
  	storage: 10Gi
  cephfs:
  	monitors:
  	- 172.16.0.1:6789
  	- 172.16.0.2:6789
  	- 172.16.0.3:6789
  	path: /ceph_storage
  	user: admin
  	secretRef:
  	  name: ceph-secret
```

有了PV，在Pod中就可以不用再定义Volume的配置了，**直接引用**即可。但是这没有解决Volume定义的第二个问题，存储系统通常由运维人员管理，开发人员并不知道底层存储配置，也就很难去定义好PV。为了解决这个问题，引入了PVC（Persistent Volume Claim），声明与消费分离，开发与运维责任分离。

```yaml
kind:PersistentVolumeClaim
apiVersion:v1
metadata:
  name: cephfs-pvc
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
	  storage: 8Gi
```
通过 `kubectl get pv` 命令可看到 PV 和 PVC 的绑定情况

```yaml
apiVersion: v1
kind: Pod
metadata:
labels:
    role: web-frontend
spec:
containers:
- name: web
    image: nginx
    ports:
    - name: web
        containerPort: 80
    volumeMounts:
       - name: cephfs-volume
       mountPath: "/usr/share/nginx/html"
volumes:
- name: cephfs-volume
    persistentVolumeClaim:
        claimName: cephfs-pvc
```

运维人员负责存储管理，可以事先根据存储配置定义好PV，而开发人员无需了解底层存储配置，只需要通过PVC声明需要的存储类型、大小、访问模式等需求即可，然后就可以在Pod中引用PVC，完全不用关心底层存储细节。

PS：感觉上，在Pod的早期，以Pod 为核心，Pod 运行所需的资源都定义在Pod yaml 中，导致Pod 越来越臃肿。后来，Kubernetes 集群中出现了一些 与Pod 生命周期不一致的资源，并单独管理。 Pod 与他们 更多是引用关系， 而不是共生 关系了。 

### Persistent Volume（PV）和 Persistent Volume Claim（PVC）

![](/public/upload/kubernetes/k8s_pvc.jpg)

[一文读懂 K8s 持久化存储流程](https://mp.weixin.qq.com/s/jpopq16BOA_vrnLmejwEdQ)

PVC 和 PV 的设计，其实跟“面向对象”的思想完全一致。PVC 可以理解为持久化存储的“接口”，它提供了对某种持久化存储的描述，但不提供具体的实现；而这个持久化存储的实现部分则由 PV 负责完成。这样做的好处是，作为应用开发者，我们只需要跟 PVC 这个“接口”打交道，而不必关心具体的实现是 NFS 还是 Ceph

||PVC|Pod|
|---|---|---|
|资源|消耗 PV 资源，**PV资源是集群的**|消耗 Node 资源|
||可以请求特定存储卷的大小及访问模式|Pod 可以请求特定级别的资源（CPU 和内存）|
||确定Node后，为Node挂载存储设备 ==> <br>Pod 为Node 带了一份“嫁妆”|能调度到Node上，说明Node本身的CPU和内存够用|
||完全是 Kubernetes 项目自己负责管理的<br>runtime 只知道mount 本地的一个目录| 容器操作基本委托给runtime|

## K8s 持久化存储流程

[详解 Kubernetes Volume 的实现原理](https://draveness.me/kubernetes-volume/)集群中的每一个卷在被 Pod 使用时都会经历四个操作，也就是附着（Attach）、挂载（Mount）、卸载（Unmount）和分离（Detach）。如果 Pod 中使用的是 EmptyDir、HostPath 这种类型的卷，那么这些卷并不会经历附着和分离的操作，它们只会被挂载和卸载到某一个的 Pod 中。

Volume 的创建和管理在 Kubernetes 中主要由卷管理器 VolumeManager 和 AttachDetachController 和 PVController 三个组件负责。
1. VolumeManager 在 Kubernetes 集群中的每一个节点（Node）上的 kubelet 启动时都会运行一个 VolumeManager Goroutine，它会负责在当前节点上的 Pod 和 Volume 发生变动时对 Volume 进行挂载和卸载等操作。
2. AttachDetachController 主要负责对集群中的卷进行 Attach 和 Detach
    1. 让卷的挂载和卸载能够与节点的可用性脱离；一旦节点或者 kubelet 宕机，附着（Attach）在当前节点上的卷应该能够被分离（Detach），分离之后的卷就能够再次附着到其他节点上；
    2. 保证云服务商秘钥的安全；如果每一个 kubelet 都需要触发卷的附着和分离逻辑，那么每一个节点都应该有操作卷的权限，但是这些权限应该只由主节点掌握，这样能够降低秘钥泄露的风险；
    3. 提高卷附着和分离部分代码的稳定性；
3. PVController 负责处理持久卷的变更

[一文读懂 K8s 持久化存储流程](https://mp.weixin.qq.com/s/jpopq16BOA_vrnLmejwEdQ)

![](/public/upload/kubernetes/persistent_process.png)

流程如下：

1. 用户创建了一个包含 PVC 的 Pod，该 PVC 要求使用动态存储卷；
2. Scheduler 根据 Pod 配置、节点状态、PV 配置等信息，把 Pod 调度到一个合适的 Worker 节点上；
3. PV 控制器 watch 到该 Pod 使用的 PVC 处于 Pending 状态，于是调用 Volume Plugin（in-tree）创建存储卷，并创建 PV 对象（out-of-tree 由 External Provisioner 来处理）；
4. AD 控制器发现 Pod 和 PVC 处于待挂接状态，于是调用 Volume Plugin 挂接存储设备到目标 Worker 节点上
5. 在 Worker 节点上，Kubelet 中的 Volume Manager 等待存储设备挂接完成，并通过 Volume Plugin 将设备挂载到全局目录：`/var/lib/kubelet/pods/[pod uid]/volumes/kubernetes.io~iscsi/[PV name]`（以 iscsi 为例）；
6. Kubelet 通过 Docker 启动 Pod 的 Containers，用 bind mount 方式将已挂载到本地全局目录的卷映射到容器中。

在 Kubernetes 中，实际上存在着一个专门处理持久化存储的控制器，叫作 Volume Controller。这个Volume Controller 维护着多个控制循环，其中有一个循环，扮演的就是撮合 PV 和 PVC 的“红娘”的角色。它的名字叫作 PersistentVolumeController

## CSI

Kubernetes存储方案发展过程概述

1. 最开始是通过Volume Plugin实现集成外部存储系统，即不同的存储系统对应不同的Volume Plugin。Volume Plugin实现代码全都放在了Kubernetes主干代码中（in-tree)，也就是说这些插件与核心Kubernetes二进制文件一起链接、编译、构建和发布。
2. 从1.8开始，Kubernetes停止往Kubernetes代码中增加新的存储支持。从1.2开始，推出了一种新的插件形式支持外部存储系统，即FlexVolume。FlexVolume类似于CNI插件，通过外部脚本集成外部存储接口，这些脚本默认放在`/usr/libexec/kubernetes/kubelet-plugins/volume/exec/`，需要安装到所有Node节点上。
3. 从1.9开始又引入了Container Storage Interface（CSI）容器存储接口

![](/public/upload/kubernetes/k8s_csi.png)

CSI 插件体系的设计思想，就是把Dynamic Provision 阶段以及 Kubernetes 里的一部分存储管理功能（比如“Attach 阶段”和“Mount 阶段”，实际上是通过调用 CSI 插件来完成的），从主干代码里剥离出来，做成了几个单独的组件。这些组件会通过 Watch API 监听 Kubernetes 里与存储相关的事件变化，比如 PVC 的创建，来执行具体的存储管理动作。

CSI 的设计思想，把插件的职责从“两阶段处理”，扩展成了Provision、Attach 和 Mount 三个阶段。其中，Provision 等价于“创建磁盘”，Attach 等价于“挂载磁盘到虚拟机”，Mount 等价于“将该磁盘格式化后，挂载在 Volume 的宿主机目录上”。

一个 CSI 插件只有一个二进制文件，但它会以 gRPC 的方式对外提供三个服务（gRPC Service），分别叫作：CSI Identity、CSI Controller 和 CSI Node。


1. Identity Service用于返回一些插件信息；
2. Controller Service实现Volume的CURD操作；
3. Node Service运行在所有的Node节点，用于实现把Volume挂载在当前Node节点的指定目录，该服务会监听一个Socket，controller通过这个Socket进行通信，可以参考官方提供的样例CSI Hostpath driver Sample[2]。


