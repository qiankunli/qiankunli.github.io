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


## Finalizer

[Kubernetes模型设计与控制器模式精要](https://mp.weixin.qq.com/s/Dbf0NSJIX-fz28Heix3EtA)如果只看社区实现，那么该属性毫无存在感，因为在社区代码中，很少有对Finalizer的操作。但在企业化落地过程中，它是一个十分重要，值得重点强调的属性。因为Kubernetes不是一个独立存在的系统，它最终会跟企业资源和系统整合，这意味着Kubernetes会操作这些集群外部资源或系统。试想一个场景，用户创建了一个Kubernetes对象，假设对应的控制器需要从外部系统获取资源，当用户删除该对象时，控制器接收到删除事件后，会尝试释放该资源。可是如果此时外部系统无法连通，并且同时控制器发生重启了会有何后果？该对象永远泄露了。

Finalizer本质上是一个资源锁，Kubernetes在接收到某对象的删除请求，会检查Finalizer是否为空，如果不为空则只对其做逻辑删除，即只会更新对象中metadata.deletionTimestamp字段。**具有Finalizer的对象，不会立刻删除**，需等到Finalizer列表中所有字段被删除后，也就是该对象相关的所有外部资源已被删除，这个对象才会被最终被删除。

因此，如果控制器需要操作集群外部资源，则一定要在操作外部资源之前为对象添加Finalizer，确保资源不会因对象删除而泄露。同时控制器需要监听对象的更新时间，当对象的deletionTimestamp不为空时，则处理对象删除逻辑，回收外部资源，并清空自己之前添加的Finalizer。PS：本质是可以干预 资源的删除逻辑。

[Using Finalizers to Control Deletion](https://kubernetes.io/blog/2021/05/14/using-finalizers-to-control-deletion/)Finalizers are keys on resources that signal pre-delete operations. They control the garbage collection on resources, and are designed to alert controllers what cleanup operations to perform prior to removing a resource. However, they don’t necessarily name code that should be executed; finalizers on resources are basically just lists of keys much like annotations. Like annotations, they can be manipulated.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mymap
  finalizers:
  - kubernetes
```

`kubectl delete configmap/mymap` 只是给 mymap.deletionTimestamp 赋了一个值，当手动移除 finalizers （比如kubectl patch） 之后，才会真正删除mymap。



## Garbage Collection

在 Kubernetes 引入垃圾收集器之前，所有的级联删除逻辑都是在客户端完成的，kubectl 会先删除 ReplicaSet 持有的 Pod 再删除 ReplicaSet，但是**垃圾收集器的引入就让级联删除的实现移到了服务端**。

[Garbage Collection](https://kubernetes.io/docs/concepts/workloads/controllers/garbage-collection/)Some Kubernetes objects are owners of other objects. For example, a ReplicaSet is the owner of a set of Pods. The owned objects are called dependents of the owner object. Every dependent object has a metadata.ownerReferences field that points to the owning object.Kubernetes objects 之间有父子关系，那么当删除owners 节点时，如何处理其dependents呢？

1. cascading deletion

    1. Foreground 先删除dependents再删除owners. In foreground cascading deletion, the root object first enters a “deletion in progress” state.Once the “deletion in progress” state is set, the garbage collector deletes the object’s dependents. Once the garbage collector has deleted all “blocking” dependents (objects with ownerReference.blockOwnerDeletion=true), it deletes the owner object.
    2. background 先删owners 后台再慢慢删dependents. Kubernetes deletes the owner object immediately and the garbage collector then deletes the dependents in the background.
2. 不管，此时the dependents are said to be orphaned.

如何控制Garbage Collection？设置propagationPolicy

```
kubectl proxy --port=8080
curl -X DELETE localhost:8080/apis/apps/v1/namespaces/default/replicasets/my-repset \
-d '{"kind":"DeleteOptions","apiVersion":"v1","propagationPolicy":"Background"}' \
-H "Content-Type: application/json"
## cascade 默认值是true
kubectl delete replicaset my-repset --cascade=false
```

### kubelet Garbage Collection

回收物理机上不用的 容器或镜像。

[Configuring kubelet Garbage Collection](https://kubernetes.io/docs/concepts/cluster-administration/kubelet-garbage-collection/)（未读）

1. Image Collection, Disk usage above the HighThresholdPercent will trigger garbage collection. The garbage collection will delete least recently used images until the LowThresholdPercent has been met. `[LowThresholdPercent,HighThresholdPercent]` 大于HighThresholdPercent 开始回收直到 磁盘占用小于LowThresholdPercent
2. Container Collection 核心就是什么时候开始删除容器，什么样的容器可以被删掉

    1. minimum-container-ttl-duration, 容器dead 之后多久可以被删除
    2. maximum-dead-containers-per-container, 每个pod 最多允许的dead 容器数量，超过的容器会被删掉
    3. maximum-dead-containers, 主机上最多允许的dead 容器数量，超过的容器会被删掉

[kubelet 垃圾回收机制](https://mp.weixin.qq.com/s/GInMyCUdAjaa2hFX3swbNg)