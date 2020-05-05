---
layout: post
title: Kubernetes持久化存储
category: 技术
tags: Kubernetes
keywords: Docker Kubernetes Volume
---

## 简介

* TOC
{:toc}

与CPU 和 Mem 这些资源相比，“存储”对k8s 来说更像是“外设”，k8s 提供统一的“总线”接入。[Kata Containers 创始人带你入门安全容器技术](https://mp.weixin.qq.com/s/w2SkC6TuSBqurvAae0RAUA)OCI规范规定了容器之中应用被放到什么样的环境下、如何运行，比如说容器的根文件系统上哪个可执行文件会被执行，是用什么用户执行，需要什么样的 CPU，有什么样的内存资源、**外置存储**，还有什么样的共享需求等等。

## Volume 背景介绍

A Volume is a directory, possibly with some data in it, which is accessible to a Container. Kubernetes Volumes are similar to but not the same as Docker Volumes.

A Pod specifies which Volumes its containers need in its ContainerManifest property.

**A process in a Container sees a filesystem view composed from two sources: a single Docker image and zero or more Volumes**（这种表述方式很有意思）. A Docker image is at the root of the file hierarchy. Any Volumes are mounted at points on the Docker image; Volumes do not mount on other Volumes and do not have hard links to other Volumes. Each container in the Pod independently specifies where on its image to mount each Volume. This is specified a VolumeMounts property.

The storage media (Disk, SSD, or memory) of a volume is determined by the media of the filesystem holding the kubelet root dir (typically `/var/lib/kubelet`)(volumn的存储类型（硬盘，固态硬盘等）是由kubelet所在的目录决定的). There is no limit on how much space an EmptyDir or PersistentDir volume can consume（大小也是没有限制的）, and no isolation between containers or between pods.

可以与 [docker volume](http://qiankunli.github.io/2015/09/24/docker_volume.html) 对比下异同

## PV 和 PVC

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
        - name: nfs
        mountPath: "/usr/share/nginx/html"
volumes:
- name: nfs
    nfs:
      server: 10.244.1.4
      path: /
```

这种方式至少存在两个问题：

1. Pod声明与底层存储耦合在一起，每次声明Volume都需要配置存储类型以及该存储插件的一堆配置，如果是第三方存储，配置会非常复杂。
2. 开发人员的需求可能只是需要一个20GB的卷，这种方式却不得不强制要求开发人员了解底层存储类型和配置。

于是引入了PV（Persistent Volume），PV其实就是把Volume的配置声明部分从Pod中分离出来，PV的spec部分几乎和前面Pod的Volume定义部分是一样的由运维人员事先创建在 Kubernetes 集群里待用

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
name: nfs
spec:
storageClassName: manual
capacity:
    storage: 1Gi
accessModes:
    - ReadWriteMany
nfs:
    server: 10.244.1.4
    path: "/"
```

有了PV，在Pod中就可以不用再定义Volume的配置了，**直接引用**即可。但是这没有解决Volume定义的第二个问题，存储系统通常由运维人员管理，开发人员并不知道底层存储配置，也就很难去定义好PV。为了解决这个问题，引入了PVC（Persistent Volume Claim），声明与消费分离，开发与运维责任分离。

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
        - name: nfs
        mountPath: "/usr/share/nginx/html"
volumes:
- name: nfs
    persistentVolumeClaim:
    claimName: nfs
```

运维人员负责存储管理，可以事先根据存储配置定义好PV，而开发人员无需了解底层存储配置，只需要通过PVC声明需要的存储类型、大小、访问模式等需求即可，然后就可以在Pod中引用PVC，完全不用关心底层存储细节。

汇总一下：**不想和Pod定义写在一起**。所以定义一个kind=PV 的Kubernetes Object
    1. Pod 一般有开发编写，而开发通常不懂 存储相关的配置
    2. 每一次 编写Pod 都copy 一份 Volume 配置（对于一些分布式存储方案来说，配置非常复杂）有点浪费。

感觉上，在Pod的早期，以Pod 为核心，Pod 运行所需的资源都定义在Pod yaml 中，导致Pod 越来越臃肿。后来，Kubernetes 集群中出现了一些 与Pod 生命周期不一致的资源，并单独管理。 Pod 与他们 更多是引用关系， 而不是共生 关系了。 

### Persistent Volume（PV）和 Persistent Volume Claim（PVC）

![](/public/upload/kubernetes/k8s_pvc.jpg)


[一文读懂 K8s 持久化存储流程](https://mp.weixin.qq.com/s/jpopq16BOA_vrnLmejwEdQ)

||PV|PVC|
|---|---|---|
|范围|集群级别的资源|命名空间级别的资源|
|创建者|由 集群管理员 or External Provisioner 创建|由 用户 or StatefulSet 控制器（根据VolumeClaimTemplate） 创建|
|生命周期|PV 的生命周期独立于使用 PV 的 Pod||

PVC 和 PV 的设计，其实跟“面向对象”的思想完全一致。PVC 可以理解为持久化存储的“接口”，它提供了对某种持久化存储的描述，但不提供具体的实现；而这个持久化存储的实现部分则由 PV 负责完成。这样做的好处是，作为应用开发者，我们只需要跟 PVC 这个“接口”打交道，而不必关心具体的实现是 NFS 还是 Ceph

||PVC|Pod|
|---|---|---|
|资源|消耗 PV 资源，**PV资源是集群的**|消耗 Node 资源|
||可以请求特定存储卷的大小及访问模式|Pod 可以请求特定级别的资源（CPU 和内存）|
||确定Node后，为Node挂载存储设备 ==> <br>Pod 为Node 带了一份“嫁妆”|能调度到Node上，说明Node本身的CPU和内存够用|
||完全是 Kubernetes 项目自己负责管理的<br>runtime 只知道mount 本地的一个目录| 容器操作基本委托给runtime|

## K8s 持久化存储流程

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


