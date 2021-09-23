---

layout: post
title: controller 组件介绍
category: 架构
tags: Kubernetes
keywords: controller
---

## 简介

* TOC
{:toc}


## Garbage Collection

在 Kubernetes 引入垃圾收集器之前，所有的级联删除逻辑都是在客户端完成的，kubectl 会先删除 ReplicaSet 持有的 Pod 再删除 ReplicaSet，但是**垃圾收集器的引入就让级联删除的实现移到了服务端**。

[Garbage Collection](https://kubernetes.io/docs/concepts/workloads/controllers/garbage-collection/)Some Kubernetes objects are owners of other objects. For example, a ReplicaSet is the owner of a set of Pods. The owned objects are called dependents of the owner object. Every dependent object has a metadata.ownerReferences field that points to the owning object.Kubernetes objects 之间有父子关系，那么当删除owners 节点时，如何处理其dependents呢？

1. cascading deletion

    1. Foreground 先删除dependents再删除owners. In foreground cascading deletion, the root object first enters a “deletion in progress” state.Once the “deletion in progress” state is set, the garbage collector deletes the object’s dependents. Once the garbage collector has deleted all “blocking” dependents (objects with ownerReference.blockOwnerDeletion=true), it deletes the owner object.
    2. background 先删owners 后台再慢慢删dependents. Kubernetes deletes the owner object immediately and the garbage collector then deletes the dependents in the background.
2. 不管，此时the dependents are said to be orphaned.

如何控制Garbage Collection？设置propagationPolicy

    kubectl proxy --port=8080
    curl -X DELETE localhost:8080/apis/apps/v1/namespaces/default/replicasets/my-repset \
    -d '{"kind":"DeleteOptions","apiVersion":"v1","propagationPolicy":"Background"}' \
    -H "Content-Type: application/json"
    ## cascade 默认值是true
    kubectl delete replicaset my-repset --cascade=false

### kubelet Garbage Collection

回收物理机上不用的 容器或镜像。

[Configuring kubelet Garbage Collection](https://kubernetes.io/docs/concepts/cluster-administration/kubelet-garbage-collection/)（未读）

1. Image Collection, Disk usage above the HighThresholdPercent will trigger garbage collection. The garbage collection will delete least recently used images until the LowThresholdPercent has been met. `[LowThresholdPercent,HighThresholdPercent]` 大于HighThresholdPercent 开始回收直到 磁盘占用小于LowThresholdPercent
2. Container Collection 核心就是什么时候开始删除容器，什么样的容器可以被删掉

    1. minimum-container-ttl-duration, 容器dead 之后多久可以被删除
    2. maximum-dead-containers-per-container, 每个pod 最多允许的dead 容器数量，超过的容器会被删掉
    3. maximum-dead-containers, 主机上最多允许的dead 容器数量，超过的容器会被删掉