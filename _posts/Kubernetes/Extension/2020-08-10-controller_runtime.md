---

layout: post
title: controller-runtime源码分析
category: 架构
tags: Kubernetes
keywords: controller-runtime 
---

## 简介

* TOC
{:toc}

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
**Control Loop通过code-generator生成**，开发者编写差异处理逻辑Reconcile即可。controller-runtime 代码结构
```
/sigs.k8s.io/controller-runtime/pkg
    /manager
        /manager.go     // 定义了 Manager interface
        /internal.go    // 定义了Manager 实现类 controllerManager ，因为可能有多个controller
    /controller
        /controller.go  // 定义了 Controller interface
    /reconcile
        /reconcile.go   // 定义了 Reconciler interface
    /handler            // 事件处理器/入队器，负责将informer 的cud event 转换为reconcile.Request加入到queue中
```
controller-runtime 的核心是Manager 驱动 Controller 进而驱动 Reconciler。kubebuiler 用Manager.start 作为驱动入口， Reconciler 作为自定义入口（变的部分），Controller 是不变的部分。

## 整体设计

Manager 管理多个Controller 的运行，并提供 数据读（cache）写（client）等crudw基础能力。

![](/public/upload/kubernetes/controller_runtime.png)

1. Cache, 负责在 Controller 进程里面根据 Scheme 同步 Api Server 中所有该 Controller 关心 的资源对象，其核心是 相关Resource的 Informer,Informer 会负责监听对应 Resource的创建/删除/更新操作，以触发 Controller 的 Reconcile 逻辑。
2. Clients,  Reconciler不可避免地需要对某些资源类型进行crud，就是通过该 Clients 实现的，其中查询功能实际查询是本地的 Cache，写操作直接访问 Api Server。

![](/public/upload/kubernetes/controller_runtime_logic.png)

A Controller manages a work queue fed reconcile.Requests from source.Sources.  Work is performed through the reconcile.Reconciler for each enqueued item. Work typically is reads and writes Kubernetes objects to make the system state match the state specified in the object Spec. 

当我们观察Manager/Contronller/Reconciler Interface 的时候，接口定义是非常清晰的，但是为了实现接口定义的 能力（方法）要聚合很多struct，初始化时要为它们赋值，这个部分代码实现的比较负责，可能是go 缺少类似ioc 工具带来的问题。 

## 启动流程

kubebuilder 生成的 controller-runtime 代码相对晦涩一些，很多时候crd 已经在其它项目中定义完成，我们需要监听一个或多个crd 完成一些工作，这种场景下对crd 的处理逻辑可以参考 kubeflow/tf-operator 等项目。

启动逻辑比较简单：创建数据结构，并建立数据结构之间的关系。
1. 初始化Manager；初始化流程主要是创建cache 与 Clients。 
    1. 创建 Cache，可以看到 Cache 主要就是创建了 InformersMap，Scheme 里面的每个 GVK 都创建了对应的 Informer，通过 informersByGVK 这个 map 做 GVK 到 Informer 的映射，每个 Informer 会根据 ListWatch 函数对对应的 GVK 进行 List 和 Watch。PS： 肯定不能一个Controller 一个informer，但controller 之间会共用informer，所以informer 要找一个地方集中持有。 
    2. 创建 Clients，读操作使用上面创建的 Cache，写操作使用 K8s go-client 直连。
2. 将 Manager 的 Client 传给 Controller，并且调用 SetupWithManager 方法传入 Manager 进行 Controller 的初始化；
3. 启动 Manager。即 `Manager.Start(stopCh)`

因为Controller 的逻辑相对固定， 所以main 入口只有Reconciler 和Manager，Controller 被隐藏了。 

```go
func main() {
	...
    // 1. init Manager 
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:             scheme,		// 要将你监听的crd 加入到scheme 中
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
ApplicationReconciler 是我们定义的Application object 对应的Reconciler 实现。ApplicationReconciler.SetupWithManager 有点绕
1. 创建Reconciler，将Reconciler 加入到controller
1. 创建controller，将controller 加入到manager
2. 创建时 会执行`Controller.watch` 配置controller 监听哪些gvk 或者说将哪些gvk 加入到cache

```go
// kubebuilder 生成
func (r *ApplicationReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).    // 生成Controller Builder 对象
		For(&appsv1alpha1.Application{}).       // 为Controller指定cr
        Complete(r)                             // 为Controller指定Reconciler
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


## Controller

[controller-runtime 之控制器实现](https://mp.weixin.qq.com/s/m-eNII-h-Gq74bMZ3fQLKg)

```go
type Controller interface {
	// Reconciler is called to reconcile an object by Namespace/Name
	reconcile.Reconciler
	// Watch takes events provided by a Source and uses the EventHandler to
	// enqueue reconcile.Requests in response to the events.
	Watch(src source.Source, eventhandler handler.EventHandler, predicates ...predicate.Predicate) error
	// Start starts the controller.  Start blocks until the context is closed or a controller has an error starting.
	Start(ctx context.Context) error
	...
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
func New(name string, mgr manager.Manager, options Options) (Controller, error) {
	c, err := NewUnmanaged(name, mgr, options)
	// Add the controller as a Manager components
	return c, mgr.Add(c)
}
```
Controller 逻辑主要有两个（任何Controller 都是如此），对应两个函数是 Watch 与 Start
1. 监听 object 事件并加入到 queue 中。为提高扩展性 Controller 将这个职责独立出来交给了 Source 组件，不只是监听apiserver，任何外界资源变动 都可以通过 Source 接口加入 到Reconcile 逻辑中。
2. 从queue 中取出object event 执行Reconcile 逻辑。 

### watch



![](/public/upload/kubernetes/controller_watch.png)

```go
// 这里明确了 Controller 监听哪些Type， 或者说哪些 event 会触发Controller
func (r *TFJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	c, err := controller.New(r.ControllerName(), mgr, controller.Options{Reconciler: r,})
	// using onOwnerCreateFunc is easier to set defaults
	c.Watch(&source.Kind{Type: &tfv1.TFJob{}}, &handler.EnqueueRequestForObject{},predicate.Funcs{...}) 
	// inject watching for job related pod
	c.Watch(&source.Kind{Type: &corev1.Pod{}}, &handler.EnqueueRequestForOwner{...}, predicate.Funcs{...)
	// inject watching for job related service
	...
	return nil
}
// watch 就是启动 source.start
func (c *Controller) Watch(src source.Source, evthdler handler.EventHandler, prct ...predicate.Predicate) error {
	// Inject Cache into arguments
	c.SetFields(src)
	c.SetFields(evthdler)
	for _, pr := range prct {
		c.SetFields(pr)
	}
	c.Log.Info("Starting EventSource", "source", src)
	return src.Start(c.ctx, evthdler, c.Queue, prct...)
}
```
Source 是事件的源，使用 Kind 来处理来自集群的事件（如 Pod 创建、Pod 更新、Deployment 更新）；使用 Channel 来处理来自集群外部的事件（如 GitHub Webhook 回调、轮询外部 URL）。
```go
// Kind 用于提供来自集群内部的事件源，这些事件来自于 Watches（例如 Pod Create 事件）
type Kind struct {
 	// Type 是 watch 对象的类型，比如 &v1.Pod{}
 	Type runtime.Object
 	// cache 用于 watch 的 APIs 接口
 	cache cache.Cache
}
func (ks *Kind) Start(handler handler.EventHandler, queue workqueue.RateLimitingInterface,prct ...predicate.Predicate) error {
 	// 从 Cache 中获取 Informer 并添加一个事件处理程序来添加队列
 	i, err := ks.cache.GetInformer(context.TODO(), ks.Type)
	i.AddEventHandler(internal.EventHandler{Queue: queue, EventHandler: handler, Predicates: prct})
 	return nil
}
```

watch 在 controller 初始化时调用，明确了 Controller 监听哪些Type，订阅这些Type的变化（入队逻辑挂到informer 上）。Controller.Watch ==> Source.Start 也就是 Kind.Start 就是从cache 中获取资源对象的 Informer 并注册事件监听函数。 对 事件监听函数进行了封装，放入到工作队列中的元素不是以前默认的元素唯一的 KEY，而是经过封装的 reconcile.Request 对象，当然通过这个对象也可以很方便获取对象的唯一标识 KEY。

### start

start 由manager.Start 触发，消费workqueue，和 一般控制器中启动控制循环比较类似

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

## Manager 启动

[controller-runtime 之 manager 实现](https://mp.weixin.qq.com/s/3i3t-PBP3UN8W9quEhAQDQ)
Manager interface 充分体现了它的作用：添加Controller 并Start 它们。 

```go
// Manager 初始化共享的依赖关系，比如 Caches 和 Client，并将他们提供给 Runnables
type Manager interface {
 	// Add 将在组件上设置所需的依赖关系，并在调用 Start 时启动组件
  	// Add 将注入接口的依赖关系 - 比如 注入 inject.Client
  	// 根据 Runnable 是否实现了 LeaderElectionRunnable 接口判断Runnable 可以在非 LeaderElection 模式（始终运行）或 LeaderElection 模式（如果启用了 LeaderElection，则由 LeaderElection 管理）下运行
  	Add(Runnable) error
 	// SetFields 设置对象上的所有依赖关系，而该对象已经实现了 inject 接口
  	// 比如 inject.Client
 	SetFields(interface{}) error
 	// Start 启动所有已注册的控制器，并一直运行，直到停止通道关闭
  	// 如果使用了 LeaderElection，则必须在此返回后立即退出二进制，否则需要 Leader 选举的组件可能会在 Leader 锁丢失后继续运行
 	Start(<-chan struct{}) error
 	...
}
```

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
## Reconciler 开发模式

```go
type ApplicationReconciler struct {
	client.Client
	Scheme *runtime.Scheme
    Log    logr.Logger
}
type Reconciler interface {
	Reconcile(Request) (Result, error)
}
type Request struct {
	types.NamespacedName
}
type NamespacedName struct {
	Namespace string
	Name      string
}
type Result struct {
	Requeue bool
	RequeueAfter time.Duration
}
```

ApplicationReconciler 持有Client，便有能力对 相关资源进行 crud
1. 从request 中资源name ，进而通过client 获取资源obj
2. 处理完毕后，通过Result 告知是Requeue 下次重新处理，还是处理成功开始下一个

## 其它

### 依赖注入

controllerManager  持有了Config/Client/APIReader/Scheme/Cache/Injector/StopChannel/Mapper 实例，将这些数据通过 SetFields 注入到Controller 中。Controller 再转手 将部分实例注入到 Source 中（Source 需要监听apiserver）

```go
func (c *Controller) Watch(src source.Source, evthdler handler.EventHandler, prct ...predicate.Predicate) error {
	// Inject Cache into arguments   
	if err := c.SetFields(src); err != nil {...}
	if err := c.SetFields(evthdler); err != nil {...}
	for _, pr := range prct {
		if err := c.SetFields(pr); err != nil {...}
    }
    ...
}
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
```
以注入client 为例，Client interface 很像spring 中的XXAware ，受体声明和实现自己可以/需要注入啥，注入方根据受体的type 进行注入

```go
// sigs.k8s.io/controller-runtime/pkg/runtime/inject/inject.go
func ClientInto(client client.Client, i interface{}) (bool, error) {
	if s, ok := i.(Client); ok {
		return true, s.InjectClient(client)
	}
	return false, nil
}
type Client interface {
	InjectClient(client.Client) error
}
// ReplicaSetReconciler 实现了 Client 接口
type ReplicaSetReconciler struct {
	client.Client
}
func (a *ReplicaSetReconciler) InjectClient(c client.Client) error {
	a.Client = c
	return nil
}
```

### 入队器

一般一个crd 对应一个 reconciler。以Application crd为例，可能需要根据pod 变化采取动作，因此Application Controller 需要监听  Application及关联的pod的变更，有两类方法

1. Reconcile 方法可以收到 Application 和 Pod object 的变更。此时实现`Reconciler.Reconcile(Request) (Result, error)` 要注意区分 Request 中的object 类型。
    1. 构建Application Controller时，Watch Pod
    2. 用更hack的手段去构建，可能对源码造成入侵。
2. 添加自定义的入队器。比如 当pod 变更时，则找到与pod 相关的Application  加入队列 。这样pod 和Application 变更均可以触发 Application 的Reconciler.Reconcile 逻辑。

controller-runtime 暴露 handler.EventHandler接口，EventHandlers map an Event for one object to trigger Reconciles for either the same object or different objects。这个接口实现了Create,Update,Delete,Generic方法，用来在资源实例的不同生命阶段，进行判断与入队。 

### 其它



![](/public/upload/kubernetes/controller_runtime_object.png)


