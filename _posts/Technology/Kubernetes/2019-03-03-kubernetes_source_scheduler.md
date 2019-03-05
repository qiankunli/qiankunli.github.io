---

layout: post
title: Kubernetes源码分析——scheduler
category: 技术
tags: Kubernetes
keywords: kubernetes 源码分析

---

## 简介（持续更新）

* TOC
{:toc}

## Kubernetes 资源模型与资源管理

在 Kubernetes 里，Pod 是最小的原子调度单位。这也就意味着，所有跟调度和资源管理相关的属性都应该是属于 Pod 对象的字段。而这其中最重要的部分，就是 Pod 的CPU 和内存配置

在 Kubernetes 中，像 CPU 这样的资源被称作“可压缩资源”（compressible resources）。它的典型特点是，当可压缩资源不足时，Pod 只会“饥饿”，但不会退出。

而像内存这样的资源，则被称作“不可压缩资源（incompressible resources）。当不可压缩资源不足时，Pod 就会因为 OOM（Out-Of-Memory）被内核杀掉。

Kubernetes 里 Pod 的 CPU 和内存资源，实际上还要分为 limits 和 requests 两种情况：在调度的时候，kube-scheduler 只会按照 requests 的值进行计算。而在真正设置 Cgroups 限制的时候，kubelet 则会按照 limits 的值来进行设置。这个理念基于一种假设：容器化作业在提交时所设置的资源边界，并不一定是调度系统所必须严格遵守的，这是因为在实际场景中，大多数作业使用到的资源其实远小于它所请求的资源限额。

QoS 划分的主要应用场景，是当宿主机资源（主要是不可压缩资源）紧张的时候，kubelet 对 Pod 进行 Eviction（即资源回收）时需要用到的。“紧张”程度可以作为kubelet 启动参数配置，默认为

	memory.available<100Mi
	nodefs.available<10%
	nodefs.inodesFree<5%
	imagefs.available<15%

Kubernetes 计算 Eviction 阈值的数据来源，主要依赖于从 Cgroups 读取到的值，以及使用 cAdvisor 监控到的数据。当宿主机的 Eviction 阈值达到后，就会进入 MemoryPressure 或者 DiskPressure 状态，从而避免新的 Pod 被调度到这台宿主机上。

limit 不设定，默认值由 LimitRange object确定

|limits|requests||Qos模型|
|---|---|---|---|
|有|有|两者相等|Guaranteed|
|有|无|默认两者相等|Guaranteed|
|x|有|两者不相等|Burstable|
|无|无||BestEffort|

而当 Eviction 发生的时候，kubelet 具体会挑选哪些 Pod 进行删除操作，就需要参考这些 Pod 的 QoS 类别了。PS：怎么有一种 缓存 evit 的感觉。limit 越“模糊”，物理机MemoryPressure/DiskPressure 时，越容易优先被干掉。

DaemonSet 的 Pod 都设置为 Guaranteed 的 QoS 类型。否则，一旦 DaemonSet 的 PPod 被回收，它又会立即在原宿主机上被重建出来，这就使得前面资源回收的动作，完全没有意义了。

## 源码


在 Kubernetes 项目中，默认调度器的主要职责，就是为一个新创建出来的 Pod，寻找一个最合适的节点（Node）而这里“最合适”的含义，包括两层： 

1. 从集群所有的节点中，根据调度算法挑选出所有可以运行该 Pod 的节点；
2. 从第一步的结果中，再根据调度算法挑选一个最符合条件的节点作为最终结果。

所以在具体的调度流程中，默认调度器会首先调用一组叫作 Predicate 的调度算法，来检查每个 Node。然后，再调用一组叫作 Priority 的调度算法，来给上一步得到的结果里的每个 Node 打分。最终的调度结果，就是得分最高的那个Node。

**调度器对一个 Pod 调度成功，实际上就是将它的 spec.nodeName 字段填上调度结果的节点名字**。 这在k8s 的很多地方都要体现，k8s 不仅将对容器的操作“标准化” ==> “配置化”，一些配置是用户决定的，另一个些是系统决定的

