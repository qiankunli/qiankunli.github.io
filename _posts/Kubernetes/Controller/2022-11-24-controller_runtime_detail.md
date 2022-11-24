---

layout: post
title: controller-runtime细节分析
category: 架构
tags: Kubernetes
keywords: controller-runtime 
---

## 简介

* TOC
{:toc}


以下部分是controller-runtime 组件

1. Cache，Kubebuilder 的核心组件，负责在 Controller 进程里面根据 Scheme 同步 Api Server 中所有该 Controller 关心 GVKs 的 GVRs，其核心是 GVK -> Informer 的映射，Informer 会负责监听对应 GVK 的 GVRs 的创建/删除/更新操作，以触发 Controller 的 Reconcile 逻辑。
2. Controller，Kubebuidler 为我们生成的脚手架文件，我们只需要实现 Reconcile 方法即可。
3. Clients，在实现 Controller 的时候不可避免地需要对某些资源类型进行创建/删除/更新，就是通过该 Clients 实现的，其中查询功能实际查询是本地的 Cache，写操作直接访问 Api Server。
4. Index，由于 Controller 经常要对 Cache 进行查询，Kubebuilder 提供 Index utility 给 Cache 加索引提升查询效率。
5. Finalizer，在一般情况下，如果资源被删除之后，我们虽然能够被触发删除事件，但是这个时候从 Cache 里面无法读取任何被删除对象的信息，这样一来，导致很多垃圾清理工作因为信息不足无法进行，K8s 的 Finalizer 字段用于处理这种情况。在 K8s 中，**只要对象 ObjectMeta 里面的 Finalizers 不为空，对该对象的 delete 操作就会转变为 update 操作**，具体说就是 update deletionTimestamp 字段，其意义就是告诉 K8s 的 GC“在deletionTimestamp 这个时刻之后，只要 Finalizers 为空，就立马删除掉该对象”。所以一般的使用姿势是
    1. 在DeletionTimestamp 为空时， 若对象没有Finalizers 就把 Finalizers 设置好（任意 string），
    2. 在DeletionTimestamp 不为空时， 根据 Finalizers 的值执行完所有的 pre-delete hook（此时可以在 Cache 里面读取到被删除对象的任何信息），之后将 Finalizers 置为空。
    一个使用场景时：正常情况下 A 创建B，则B的 ownerreference 指向A，删除A时会自动删除B。但 ownerreference 不能跨ns，因此在对 跨ns 进行级联删除时，可以使用
6. OwnerReference，K8s GC 在删除一个对象时，任何 ownerReference 是该对象的对象都会被清除，与此同时，Kubebuidler 支持所有对象的变更都会触发 Owner 对象 controller 的 Reconcile 方法。

## client

```
cli, err := client.New(restConf, client.Options{Scheme: scheme.Scheme,})
```