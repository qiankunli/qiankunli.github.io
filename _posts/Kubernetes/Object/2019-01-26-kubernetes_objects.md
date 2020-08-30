---

layout: post
title: Kubernetes objects 
category: 技术
tags: Kubernetes
keywords: kubernetes stateset

---

## 简介（未完成）

* TOC
{:toc}

Kubernetes 对象是系统中的持久实体，描述集群的期望状态

![](/public/upload/kubernetes/kubernetes_object.png)

你一定有方法在不使用 Kubernetes、甚至不使用容器的情况下，自己 DIY 一个类似的方案出来。但是，一旦涉及到升级、版本管理等更工程化的能力，Kubernetes 的好处，才会更加凸现。

**Kubernetes 的各种object，就是常规的各个项目组件在 kubernetes 上的表示** [深入理解StatefulSet（三）：有状态应用实践](https://time.geekbang.org/column/article/41217) 充分体现了在我们把服务 迁移到Kubernetes 的过程中，要做多少概念上的映射。

## Kubernetes 基础类型系统

k8s.io/client-go, k8s.io/api, k8s.io/apimachinery 是基于Golang的 Kubernetes 编程的核心。api machinery 代码库实现了 Kubernetes 基础类型系统（实际指的是kinds）

![](/public/upload/kubernetes/kubernetes_type.png)
kinds被分为 group 和verison，因此api machinery 代码中的核心术语是 GroupVersionKind，简称GVK。 与kinds 同级概念的是 resource，也按group 和version 划分，因此有术语GroupVersionResource 简称GVR，每个GVR 对应一个http 路径（kind 不会），用于标识 Kubernetes API的REST 接口

scheme struct 将golang object 映射为可能的GVK。一个GVK 到一个GVR 的映射被称为 REST mapping,  RESTMapper interface/ RESTMapping struct 来完成转换。

```go
// k8s.io/apimachinery/pkg/api/meta/interface.go
type RESTMapper interface {
	KindFor(resource schema.GroupVersionResource) (schema.GroupVersionKind, error)
    ResourceFor(input schema.GroupVersionResource) (schema.GroupVersionResource, error)
    ...
}
// k8s.io/apimachinery/pkg/runtime/scheme.go
type Scheme struct {
	gvkToType map[schema.GroupVersionKind]reflect.Type
    typeToGVK map[reflect.Type][]schema.GroupVersionKind
}
func (s *Scheme) ObjectKinds(obj Object) ([]schema.GroupVersionKind, bool, error) {...}
```
为了使 scheme正常工作，必须将golang 类型注册到 scheme 中。对于Kubernetes 核心类型，在`k8s.io/client-go/kubernetes/scheme` 包中 均已预先注册

```go
// k8s.io/client-go/kubernetes/scheme/register.go
var Scheme = runtime.NewScheme()
var AddToScheme = localSchemeBuilder.AddToScheme
func init(){
    v1.AddToGroupVersion(Scheme, schema.GroupVersion{Version: "v1"})
    utilruntime.Must(AddToScheme(Scheme))
}
var localSchemeBuilder = runtime.SchemeBuilder{
    corev1.AddToScheme,
    appsv1.AddToScheme,
}
// k8s.io/api/core/v1/register.go
var (
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)
	AddToScheme   = SchemeBuilder.AddToScheme
)
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Pod{},
        &PodList{},
        &Service{},
    )
    ...
}
```
kubernetes object 在go 中是struct（`k8s.io/api/core/v1/types.go`），struct 的filed 当然不同， 但也共用一些结构 runtime.Object。用来约定：可以set/get GroupVersionKind 和 deepCopy，即**k8s object 存储其类型并允许克隆**。

```go
// k8s.io/apimachinery/pkg/runtime/interface.go
type Object interface{
    GetObjectKind() schema.ObjectKind
    DeepCopyObject() Object
}
type ObjectKind interface{
    SetGroupVersionKind(kind GroupVersionKind)
    GroupVersionKind() GroupVersionKind
}
// k8s.io/apimachinery/pkg/apis/meta/v1/types.go
// 实现 ObjectKind
type TypeMeta struct{
    Kind string             `json:"kind"`
    APIVersion string       `json:"apiVersion"`
}
type ObjectMeta struct{
    Name string
    Namespace string
    UID types.UID
    ResourceVersion string
    CreationTimestamp Time
    DeletionTimestamp Time
    Labels map[string]string
    Annotations map[string]string
}
```
go 中的pod 声明 如下所示

```go
// k8s.io/api/core/v1/types.go
type Pod struct{
    metav1.TypeMeta 
    metav1.ObjectMeta   `json:"metadata"`
    Spec PodSpec        `json:"spec"`
    Status PodStatus    `json:"status"`
}
```

每一个对象都包含两个嵌套对象来描述规格（Spec）和状态（Status），对象的规格其实就是我们期望的目标状态。而Status描述了对象的当前状态（或者说愿望的结果），是我们观察集群本身的一个接口。

```go
type Deployment struct { 
    metav1.TypeMeta `json:",inline"` 
    metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"` 
    Spec DeploymentSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"` 
    Status DeploymentStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"` 
} 
```

## 集大成者——StatefulSet

StatefulSet 的设计其实非常容易理解。它把真实世界的应用状态，抽象为了两种情况：
1. 拓扑状态，比如应用的主节点 A 要先于从节点 B 启动
2. 存储状态，应用的多个实例分别绑定了不同的存储数据

StatefulSet 的核心功能，就是通过某种方式记录这些状态，然后在 Pod 被重新创建时，能够为新 Pod 恢复这些状态。程序 = 数据结构 + 算法。**新增了一个功能，一定在数据表示上有体现（对应数据结构），一定在原来的工作流程中有体现或者改了工作流程（对应算法）**


StatefulSet 这个控制器的主要作用之一，就是使用Pod 模板创建 Pod 的时候，对它们进行编号，并且按照编号顺序逐一完成创建工作。而当 StatefulSet 的“控制循环”发现 Pod 的“实际状态”与“期望状态”不一致，需要新建或者删除 Pod 进行“调谐”的时候，它会严格按照这些Pod 编号的顺序，逐一完成这些操作。**所以，StatefulSet 其实可以认为是对 Deployment 的改良。**

StatefulSet 里的不同 Pod 实例，不再像 ReplicaSet 中那样都是完全一样的，而是有了细微区别的。比如，每个 Pod 的 hostname、名字等都是不同的、携带了编号的。Kubernetes 通过 Headless Service，为这些有编号的 Pod，在 DNS 服务器中生成带有同样编号的 DNS 记录。StatefulSet 还为每一个 Pod 分配并创建一个同样编号的 PVC。


 [DNS for Services and Pods](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/) “Normal” (not headless) Services are assigned a DNS A record for a name of the form `my-svc.my-namespace.svc.cluster.local`. Headless Service 所代理的所有 Pod 的 IP 地址，都会被绑定一个这样格式的 DNS 记录 `<pod-name>.<svc-name>.<namespace>.svc.cluster.local`

通过 Headless Service 的方式，StatefulSet 为每个 Pod 创建了一个固定并且稳定的 DNS记录，来作为它的访问入口。在部署“有状态应用”的时候，应用的每个实例拥有唯一并且稳定的“网络标识”，是一个非常重要的假设。

Persistent Volume Claim 和 PV 的关系。运维人员创建PV，告知有多少volume。开发人员创建Persistent Volume Claim 告知需要多少大小的volume。创建一个 PVC，Kubernetes 就会自动为它绑定一个符合条件的Volume。即使 Pod 被删除，它所对应的 PVC 和 PV 依然会保留下来。所以当这个 Pod 被重新创建出来之后，Kubernetes 会为它找到同样编号的 PVC，挂载这个 PVC 对应的 Volume，从而获取到以前保存在 Volume 里的数据。

## ConfigMap

## DaemonSet

## Job/CronJob

## 体会

学习rc、deployment、service、pod 这些Kubernetes object 时，因为功能和yaml 有直接的一对一关系，所以体会不深。在学习StatefulSet 和 DaemonSet 时，有几个感觉

1. Kubernetes object 是分层次的，pod 是很基础的层次，然后rc、deployment、StatefulSet 等用来描述如何管理它。

    * 换句话说，pod 的配置更多是给docker看的，deployment 和StatefulSet 等配置更多是给 Kubernetes Controller 看的
    * pod 其实有一份儿配置的全集， DaemonSet 的生效 是背后偷偷改 pod 配置 加上 恰当的时机操作pod api
2. Kubernetes objects是否可以笼统的划分一下，编排对象架构在调度对象之上？

    1. 调度对象pod、service、volume
    2. 编排对象StatefulSet、DaemonSet 和Job/CronJob 等
