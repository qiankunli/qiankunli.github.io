---

layout: post
title: kubebuilder 及controller-runtime学习
category: 架构
tags: Kubernetes
keywords: Kubernetes crd kubebuilder
---

## 简介

* TOC
{:toc}

[Kubebuilder：让编写 CRD 变得更简单](https://mp.weixin.qq.com/s/Gzpq71nCfSBc1uJw3dR7xA)K8s 作为一个“容器编排”平台，其核心的功能是编排，Pod 作为 K8s 调度的最小单位,具备很多属性和字段，K8s 的编排正是通过一个个控制器根据被控制对象的属性和字段来实现。PS：再具体点就是 crud pod及其属性字段

对于用户来说，实现 CRD 扩展主要做两件事：

1. 编写 CRD 并将其部署到 K8s 集群里；这一步的作用就是让 K8s 知道有这个资源及其结构属性，在用户提交该自定义资源的定义时（通常是 YAML 文件定义），K8s 能够成功校验该资源并创建出对应的 Go struct 进行持久化，同时触发控制器的调谐逻辑。
2. 编写 Controller 并将其部署到 K8s 集群里。这一步的作用就是实现调谐逻辑。

## kubebuilder 

### 和controller-runtime 的关系

对于 CRD Controller 的构建，有几个主流的工具
1.  coreOS 开源的 Operator-SDK（https://github.com/operator-framework/operator-sdk ）
2.  K8s 兴趣小组维护的 Kubebuilder（https://github.com/kubernetes-sigs/kubebuilder ）

[kubebuilder](https://github.com/kubernetes-sigs/kubebuilder) 是一个用来帮助用户快速实现 Kubernetes CRD Operator 的 SDK。当然，kubebuilder 也不是从0 生成所有controller 代码，k8s 提供给一个 [Kubernetes controller-runtime Project](https://github.com/kubernetes-sigs/controller-runtime)  a set of go libraries for building Controllers. controller-runtime 在Operator SDK中也有被用到。

有点类似于spring/controller-runtime提供核心抽象,springboot/kubebuilder 将一切集成起来，**我们只需要实现 Reconcile 方法即可**。

```go
type Reconciler interface {
    // Reconciler performs a full reconciliation for the object referred to by the Request.The Controller will requeue the Request to be processed again if an error is non-nil or Result.Requeue is true, otherwise upon completion it will remove the work from the queue.
    Reconcile(Request) (Result, error)
}
```

### 示例demo

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
            /crd                        // 新增Application CustomResourceDefinition
            /...
        /controllers
            /application_controller.go  // 定义 ApplicationReconciler ，核心逻辑就在这里实现
        main.go                         // ApplicationReconciler 添加到 Manager，Manager.Start(stopCh)
        go.mod                          
    ```

Kubernetes 有类型系统，cr 要符合相关的规范（一部分由开发人员手动编写，另一些由代码生成器生成），包括

1. struct 定义放在 项目 `pkg/apis/$group/$version` 包下
2. struct 嵌入 TypeMeta struct  ObjectMeta，定义spec 和 status

## controller-runtime 整体设计

[controller-runtime 之控制器实现](https://mp.weixin.qq.com/s/m-eNII-h-Gq74bMZ3fQLKg)
[controller-runtime 之 manager 实现](https://mp.weixin.qq.com/s/3i3t-PBP3UN8W9quEhAQDQ)

当我们谈到 k8s 的控制器模型时，其伪代码如下

```go
for {
    actualState := GetResourceActualState(rsvc)
    expectState := GetResourceExpectState(rsvc) // 来自yaml 文件
    if actualState == expectState {
        // do nothing
    } else {
        Reconcile(rsvc) // 编排逻辑，调谐的最终结果一般是对被控制对象的某种写操作，比如增/删/改 Pod
    }
}
```
controller-runtime 代码结构
```
/sigs.k8s.io/controller-runtime/pkg
    /manager
        /manager.go     // 定义了 Manager interface
        /internal.go    // 定义了Manager 实现类 controllerManager  
    /controller
        /controller.go  // 定义了 Controller interface
    /reconcile
        /reconcile.go   // 定义了 Reconciler interface
```
controller-runtime 的核心是Manager 驱动 Controller 进而驱动 Reconciler。kubebuiler 用Manager.start 作为驱动入口， Reconciler 作为自定义入口（变的部分），Controller 是不变的部分。

Manager 管理多个Controller 的运行，并提供 数据读（cache）写（client）等crudw基础能力。

![](/public/upload/kubernetes/controller_runtime.png)

1. Cache, 负责在 Controller 进程里面根据 Scheme 同步 Api Server 中所有该 Controller 关心 的资源对象，其核心是 相关Resource的 Informer,Informer 会负责监听对应 Resource的创建/删除/更新操作，以触发 Controller 的 Reconcile 逻辑。
2. Clients,  Reconciler不可避免地需要对某些资源类型进行crud，就是通过该 Clients 实现的，其中查询功能实际查询是本地的 Cache，写操作直接访问 Api Server。

![](/public/upload/kubernetes/controller_runtime_logic.png)

一个controller中只有一个Reconciler。A Controller manages a work queue fed reconcile.Requests from source.Sources.  Work is performed through the reconcile.Reconciler for each enqueued item. Work typically is reads and writes Kubernetes objects to make the system state match the state specified in the object Spec. 

## 源码分析

### 启动流程

启动逻辑比较简单：
1. 初始化Manager；初始化流程主要是创建cache 与 Clients。 
    1. 创建 Cache，可以看到 Cache 主要就是创建了 InformersMap，Scheme 里面的每个 GVK 都创建了对应的 Informer，通过 informersByGVK 这个 map 做 GVK 到 Informer 的映射，每个 Informer 会根据 ListWatch 函数对对应的 GVK 进行 List 和 Watch。
    2. 创建 Clients，读操作使用上面创建的 Cache，写操作使用 K8s go-client 直连。
2. 将 Manager 的 Client 传给 Controller，并且调用 SetupWithManager 方法传入 Manager 进行 Controller 的初始化；
3. 启动 Manager。即 `Manager.Start(stopCh)`

```go
func main() {
	...
    // 1. init Manager 
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:             scheme,
		Port:               9443,})
    // 2. init Reconciler（Controller）
	if err = (&controllers.ApplicationReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {...}
    // 3. start Manager
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {...}
}
```
ApplicationReconciler 是我们定义的Application object 对应的Reconciler 实现。ApplicationReconciler.SetupWithManager 有点绕，把构造Controller 以及Reconciler/Controller/Manager 一步做掉了。

```go
// kubebuilder 生成
func (r *ApplicationReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).    // 生成Controller Builder 对象
		For(&appsv1alpha1.Application{}).       // 为Controller指定cr
        Complete(r)                             // 为Controller指定Reconciler
}
// sigs.k8s.io/controller-runtime/pkg/controller/controller.go
type Controller struct {
	Name string // Name is used to uniquely identify a Controller in tracing, logging and monitoring. Name is required.
	Do reconcile.Reconciler
	Client client.Client // Client is a lazily initialized Client.  
	Scheme *runtime.Scheme
	Cache cache.Cache
	Config *rest.Config // Config is the rest.Config used to talk to the apiserver.  
	// Queue is an listeningQueue that listens for events from Informers and adds object keys to the Queue for processing
	Queue workqueue.RateLimitingInterface
    ...
}
// Complete ==> Build
func (blder *Builder) Build(r reconcile.Reconciler) (controller.Controller, error) {
	// Set the Config
	blder.loadRestConfig()
	// Set the ControllerManagedBy
	if err := blder.doController(r); err != nil {return nil, err}
	// Set the Watch
	if err := blder.doWatch(); err != nil {return nil, err}
	return blder.ctrl, nil
}
func New(name string, mgr manager.Manager, options Options) (Controller, error) {
	...
	// Inject dependencies into Reconciler
	if err := mgr.SetFields(options.Reconciler); err != nil {...}
	c := &controller.Controller{    // Create controller with dependencies set
		Do:       options.Reconciler,
		Cache:    mgr.GetCache(),
		Config:   mgr.GetConfig(),
		Scheme:   mgr.GetScheme(),
		Client:   mgr.GetClient(),
        Name:     name,
	}
	return c, mgr.Add(c)    // Add the controller as a Manager components
}
```

### Manager 启动

Manager 可以管理 Runnable/Controller 的生命周期（添加/启动），持有Controller共同的依赖：client、cache、scheme 等。提供了object getter(例如GetClient())，还有一个简单的依赖注入机制(runtime/inject)，还支持领导人选举，提供了一个用于优雅关闭的信号处理程序。

```go
func (cm *controllerManager) Start(stop <-chan struct{}) error {
    // 启动metric 组件供Prometheus 拉取数据
    go cm.serveMetrics(cm.internalStop)
    // 启动健康检查探针
	go cm.serveHealthProbes(cm.internalStop)
	go cm.startNonLeaderElectionRunnables()
	if cm.resourceLock != nil {
		if err := cm.startLeaderElection(); err != nil{...}
	} else {
		go cm.startLeaderElectionRunnables()
	}
    ...
}
type controllerManager struct {
    ...
	// leaderElectionRunnables is the set of Controllers that the controllerManager injects deps into and Starts.
	// These Runnables are managed by lead election.
	leaderElectionRunnables []Runnable
}
// 启动cache/informer 及 Controller
func (cm *controllerManager) startLeaderElectionRunnables() {
    // 核心是启动Informer, waitForCache ==> cm.startCache = cm.cache.Start ==> InformersMap.Start ==> 
    // InformersMap.structured/unstructured.Start ==> Informer.Run
	cm.waitForCache()
	for _, c := range cm.leaderElectionRunnables {
		ctrl := c
		go func() {
            // 启动Controller
			if err := ctrl.Start(cm.internalStop); err != nil {...}
		}()
	}
}
```

### Controller 逻辑

Controller 逻辑主要有两个（任何Controller 都是如此）
1. 监听 object 事件并加入到 queue 中。为提高扩展性 Controller 将这个职责独立出来交给了 Source 组件，不只是监听apiserver，任何外界资源变动 都可以通过 Source 接口加入 到Reconcile 逻辑中。
2. 从queue 中取出object event 执行Reconcile 逻辑。 

![](/public/upload/kubernetes/controller_runtime_object.png)

```go
func (c *Controller) Start(stop <-chan struct{}) error {
	err := func() error {
        ...
		// Launch workers to process resources
		for i := 0; i < c.MaxConcurrentReconciles; i++ {
			// Process work items
			go wait.Until(c.worker, c.JitterPeriod, stop)
		}
		return nil
	}()
	return nil
}
func (c *Controller) worker() {
	for c.processNextWorkItem() {
	}
}
// 从队列中取出 变更的对象（也就是需要处理的对象），包括队列操作相关的线速、重试等，并触发Reconcile 逻辑
func (c *Controller) processNextWorkItem() bool {
	obj, shutdown := c.Queue.Get()
	if shutdown {
		return false// Stop working
	}
	defer c.Queue.Done(obj)
	return c.reconcileHandler(obj)
}
func (c *Controller) reconcileHandler(obj interface{}) bool {
	if req, ok = obj.(reconcile.Request); !ok {...}
	// RunInformersAndControllers the syncHandler, passing it the namespace/Name string of the resource to be synced.
	if result, err := c.Do.Reconcile(req); err != nil {
		c.Queue.AddRateLimited(req)
		return false
	} else if result.RequeueAfter > 0 {
		c.Queue.Forget(obj)
		c.Queue.AddAfter(req, result.RequeueAfter)
		return true
	} else if result.Requeue {
		c.Queue.AddRateLimited(req)
		return true
	}
	c.Queue.Forget(obj)
	return true
}
```

## 其它
### 依赖注入

controllerManager  持有了Config/Client/APIReader/Scheme/Cache/Injector/StopChannel/Mapper 实例，将这些数据通过 SetFields 注入到Controller 中。Controller 再转手 将部分实例注入到 Source 中（Source 需要监听apiserver）

```go
func (cm *controllerManager) SetFields(i interface{}) error {
	if _, err := inject.ConfigInto(cm.config, i); err != nil {return err}
	if _, err := inject.ClientInto(cm.client, i); err != nil {return err}
	if _, err := inject.APIReaderInto(cm.apiReader, i); err != nil {return err}
	if _, err := inject.SchemeInto(cm.scheme, i); err != nil {return err}
	if _, err := inject.CacheInto(cm.cache, i); err != nil {return err}
	if _, err := inject.InjectorInto(cm.SetFields, i); err != nil {return err}
	if _, err := inject.StopChannelInto(cm.internalStop, i); err != nil {return err}
	if _, err := inject.MapperInto(cm.mapper, i); err != nil {return err}
	return nil
}
func (c *Controller) Watch(src source.Source, evthdler handler.EventHandler, prct ...predicate.Predicate) error {
	// Inject Cache into arguments   
	if err := c.SetFields(src); err != nil {...}
	if err := c.SetFields(evthdler); err != nil {...}
	for _, pr := range prct {
		if err := c.SetFields(pr); err != nil {...}
    }
    ...
}
```
### kubectl 感知 crd
《programming Kubernetes》 假设存在 一个CR

```yaml
apiVersion: cnat.programming-kubernetes.info/v1alpha1
kind: At
metadata:
  name: example-at
spec:
  schedule: ...
status:
  phase: "pending"
```

整个感知过程由 RESTMapper实现，kubectl 在`~/.kubectl` 中缓存了资源类型，由此它不必每次访问重新检索感知信息，缓存每隔10分钟失效。

1. 最初，kubectl 并不知道 ats 是什么？
2. kubectl 使用`/apis` 感知endpoint 的方式，向api server 查询所有的api groups  
    ```sh
    $ http://localhost:8080/apis
    {
        "groups":[
            {
                "name":"at.cnat.programming-kubernetes.info/v1alpha1",
                "versions":[{
                    "groupVersion":"cnat.programming-kubernetes.info/v1alpha1",
                    "version":"v1alpha1"
                }]
            },
            ...
        ]
    }
    ```
3. 接着，kubectl 使用`/apis/$groupVersion` 感知endpoint 的方式，向apiserver 查询所有API groups 中的资源
    ```sh
    $ http://localhost:8080/apis/cnat.programming-kubernetes.info/v1alpha1
    {
        "apiVersion":"v1",
        "groupVersion":"cnat.programming-kubernetes.info/v1alpha1",
        "kind": "APIResourceList",
        "resource":[{
            "kind": "At",
            "name": "ats",
            "namespaced": true,
            "verbs":["create","delete","get","list","update","watch",...]
        },...]

    }
    ```
4. 然后kubectl 将 给定的类型ats 转换为一下3 种类型：Group(cnat.programming-kubernetes.info) Version (v1alpha1)  Resource (ats)


