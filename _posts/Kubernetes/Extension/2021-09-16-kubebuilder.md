---

layout: post
title: kubebuilder 学习
category: 架构
tags: Kubernetes
keywords: kubebuilder
---

## 简介

* TOC
{:toc}

Kubernetes 这样的分布式操作系统对外提供服务是通过 API 的形式，**分布式操作系统本身提供的 API 相当于单机操作系统的系统调用**。Kubernetes 本身提供的API，通过名为 Controller 的组件来支持，由开发者为 Kubernetes 提供的新的 API，则通过 Operator 来支持，Operator 本身和 Controller 基于同一套机制开发。Kubernetes 的工作机制和单机操作系统也较为相似，etcd 提供一个 watch 机制，Controller 和 Operator 需要指定自己 watch 哪些内容，并告诉 etcd，这相当于是微内核架构在 IDT 或 SCV 中注册系统调用的过程。

[Kubebuilder：让编写 CRD 变得更简单](https://mp.weixin.qq.com/s/Gzpq71nCfSBc1uJw3dR7xA)K8s 作为一个“容器编排”平台，其核心的功能是编排，Pod 作为 K8s 调度的最小单位,具备很多属性和字段，K8s 的编排正是通过一个个控制器根据被控制对象的属性和字段来实现。PS：再具体点就是 crud pod及其属性字段

对于用户来说，实现 CRD 扩展主要做两件事：

1. 编写 CRD 并将其部署到 K8s 集群里；这一步的作用就是让 K8s 知道有这个资源及其结构属性，在用户提交该自定义资源的定义时（通常是 YAML 文件定义），K8s 能够成功校验该资源并创建出对应的 Go struct 进行持久化，同时触发控制器的调谐逻辑。
2. 编写 Controller 并将其部署到 K8s 集群里。这一步的作用就是实现调谐逻辑。

[面向 K8s 设计误区](https://mp.weixin.qq.com/s/W_UjqI0Rd4AAVcafMiaYGA)

## 从 code-generator开始

client-go 只提供了rest api和 dynamic client来操作第三方资源，需要自己实现反序列化等功能（client-go内置informer 只针对k8s 内置object）。**建立好自己的crd struct后**（在types.go中），code-generator提供了以下工具为kubernetes中的资源生成代码（还有一些未列出）:
1. deepcopy-gen: 生成深度拷贝方法,避免性能开销
2. client-gen: 为资源生成标准的操作方法(get,list,create,update,patch,delete,deleteCollection,watch)
3. informer-gen: 生成informer,提供事件机制来相应kubernetes的event
4. lister-gen: 为get和list方法提供只读缓存层
5. conversion-gen是用于自动生成在内部和外部类型之间转换的函数的工具
6. defaulter-gen 用于生产Defaulter函数
7. openapi-gen  生成openAPI定义
code-generator还专门整合了这些gen，形成了generate-groups.sh和generate-internal-groups.sh这两个脚本。 PS：原来client/informer/lister 这些代码都是自动生成的

一般带有crd 项目会有 hack 目录 包含 update-codegen.sh  脚本文件（执行code-generator） 或包含Makefile 用来支持根据 crd struct 生成相关的deepcopy/defaults/client(clientset/informer/lister)/kustomize(包含crd yaml定义)文件。

```
crd-project
    /mainfests  # 
    /hack
        /update-codegen.sh  
    /pkg
        /apis/xx/v1
            /types.go                       # 自定义crd struct
            /zz_generated.deepcopy.go
            /zz_generated.defaults.go
        /client
            /clientset
            /informers
            /listers
    /Makefile
```


## 和controller-runtime 的关系

对于 CRD Controller 的构建，有几个主流的工具
1.  coreOS 开源的 Operator-SDK（https://github.com/operator-framework/operator-sdk ）
2.  K8s 兴趣小组维护的 Kubebuilder（https://github.com/kubernetes-sigs/kubebuilder ）

当我们需要增加新资源时就需要声明crd资源和构建operator。不同的opertor都需要连接apiserver同步自定义资源，因此会有高度冗余的逻辑。kubebuilder不仅会为我们生成opertor服务框架，还能自动同步自定义资源的变化。用户只需要定义CRD结构，和处理资源变动时的回调即可。

![](/public/upload/kubernetes/kubebuilder_controller.png)

```go
type Reconciler interface {
    // Reconciler performs a full reconciliation for the object referred to by the Request.The Controller will requeue the Request to be processed again if an error is non-nil or Result.Requeue is true, otherwise upon completion it will remove the work from the queue.
    Reconcile(Request) (Result, error)
}
```

[kubebuilder](https://github.com/kubernetes-sigs/kubebuilder) 是一个用来帮助用户快速实现 Kubernetes CRD Operator 的 SDK。当然，kubebuilder 也不是从0 生成所有controller 代码，k8s 提供给一个 [Kubernetes controller-runtime Project](https://github.com/kubernetes-sigs/controller-runtime)  a set of go libraries for building Controllers. controller-runtime 在Operator SDK中也有被用到。


以下部分是kubebuilder的框架性组件

1. Cache，Kubebuilder 的核心组件，负责在 Controller 进程里面根据 Scheme 同步 Api Server 中所有该 Controller 关心 GVKs 的 GVRs，其核心是 GVK -> Informer 的映射，Informer 会负责监听对应 GVK 的 GVRs 的创建/删除/更新操作，以触发 Controller 的 Reconcile 逻辑。
2. Controller，Kubebuidler 为我们生成的脚手架文件，我们只需要实现 Reconcile 方法即可。
3. Clients，在实现 Controller 的时候不可避免地需要对某些资源类型进行创建/删除/更新，就是通过该 Clients 实现的，其中查询功能实际查询是本地的 Cache，写操作直接访问 Api Server。
4. Index，由于 Controller 经常要对 Cache 进行查询，Kubebuilder 提供 Index utility 给 Cache 加索引提升查询效率。
5. Finalizer，在一般情况下，如果资源被删除之后，我们虽然能够被触发删除事件，但是这个时候从 Cache 里面无法读取任何被删除对象的信息，这样一来，导致很多垃圾清理工作因为信息不足无法进行，K8s 的 Finalizer 字段用于处理这种情况。在 K8s 中，只要对象 ObjectMeta 里面的 Finalizers 不为空，对该对象的 delete 操作就会转变为 update 操作，具体说就是 update deletionTimestamp 字段，其意义就是告诉 K8s 的 GC“在deletionTimestamp 这个时刻之后，只要 Finalizers 为空，就立马删除掉该对象”。所以一般的使用姿势就是在创建对象时把 Finalizers 设置好（任意 string），然后处理 DeletionTimestamp 不为空的 update 操作（实际是 delete），根据 Finalizers 的值执行完所有的 pre-delete hook（此时可以在 Cache 里面读取到被删除对象的任何信息）之后将 Finalizers 置为空即可。
6. OwnerReference，K8s GC 在删除一个对象时，任何 ownerReference 是该对象的对象都会被清除，与此同时，Kubebuidler 支持所有对象的变更都会触发 Owner 对象 controller 的 Reconcile 方法。


kubebuilder 依赖于 controller-runtime 实现 controller 整个处理流程，在此工程中，controller 对资源的监听依赖于 Informer 机制，controller-runtime 在此机制上又封装了一层，其整体流程入下图

![](/public/upload/kubernetes/kubebuilder_reconcile.png)

其中 Informer 已经由kubebuilder和contorller-runtime 实现，监听到的资源的事件（创建、删除、更新）都会放在 Informer 中。然后这个事件会经过 predict()方法进行过滤，经过interface enqueue进行处理，最终放入 workqueue中。我们创建的 controller 则会依次从workqueue中拿取事件，并调用我们自己实现的 Recontile() 方法进行业务处理。

## 示例demo

[Kubebuilder中文文档](https://cloudnative.to/kubebuilder/introduction.html) 对理解k8s 上下游知识以及使用kubebuiler 编写控制器很有帮助。

1. 在`GOPATH/src/app`创建脚手架工程 `kubebuilder init --domain example.io`
    ```
    GOPATH/src/app
        /config                 // 跟k8s 集群交互所需的一些yaml配置
            /certmanager
            /default
            /manager
            /prometheus
            /rbac
            /webhook
        main.go                 // 创建并启动 Manager，容器的entrypoint
        Dockerfile              // 制作Controller 镜像
        go.mod                   
            module app
            go 1.13
            require (
                k8s.io/apimachinery v0.17.2
                k8s.io/client-go v0.17.2
                sigs.k8s.io/controller-runtime v0.5.0
            )
    ```
2.  创建 API `kubebuilder create api --group apps --version v1alpha1 --kind Application` 后文件变化
    ```
    GOPATH/src/app
        /api/v1alpha1
            /application_types.go      // 新增 Application/ApplicationSpec/ApplicationStatus struct; 将类型注册到 scheme 辅助接口 
            /zz_generated.deepcopy.go
        /config
            /crd                        // Application CustomResourceDefinition。提交后apiserver 可crudw该crd
            /...
        /controllers
            /application_controller.go  // 定义 ApplicationReconciler ，核心逻辑就在这里实现
        main.go                         // ApplicationReconciler 添加到 Manager，Manager.Start(stopCh)
        go.mod     
        Makefile                     
    ```
执行 `make install` 实质是执行 `kustomize build config/crd | kubectl apply -f -` 将cr yaml 提交到apiserver上。之后就可以 提交Application yaml 到 k8s 了。将crd struct 注册到 schema，则client-go 可以支持对crd的 crudw 等操作。

## 其它

[kubebuilder2.0学习笔记——搭建和使用](https://segmentfault.com/a/1190000020338350)
[kubebuilder2.0学习笔记——进阶使用](https://segmentfault.com/a/1190000020359577) 
go build  之后，可执行文件即可 监听k8s（由`--kubeconfig` 参数指定 ），执行Reconcile 逻辑了

如果我们需要对 用户录入的 Application 进行合法性检查，可以开发一个webhook
`kubebuilder create webhook --group apps --version v1alpha1 --kind Application --programmatic-validation --defaulting`

可以在crd types.go 加一些kubebuilder 标记，用于codegen 工具辅助生成 yaml文件、go代码等。比如：令crd支持kubectl scale，对crd实例进行基础的值校验，允许在kubectl get命令中显示crd的更多字段，等等。更多注解见[Markers for Config/Code Generation​book.kubebuilder.io](https://book.kubebuilder.io/reference/markers.html)



