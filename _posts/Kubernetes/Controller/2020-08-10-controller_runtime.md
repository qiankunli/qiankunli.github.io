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

## 用法

单纯使用 client-go informer 机制 监听 某个object的写法。

```go
kubeClient = kubernetes.NewForConfigOrDie(opt.Config)
// 基于GVK 操作资源，假设需要操作数十种不同资源时，我们需要为每一种资源实现各自的函数
podInformer = informers.NewSharedInformerFactory(pod.kubeClient, 0).Core().V1().Pods()
podLister = pod.podInformer.Lister()
podSynced = pod.podInformer.Informer().HasSynced
podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
	AddFunc:    AddPod,
	DeleteFunc: DeletePod,
	UpdateFunc: UpdatePod,
})

// 启动informer
podInformer.Informer().Run(stopCh)
cache.WaitForCacheSync(stopCh, podSynced)
// 此处没有使用workqueue，但一般都是会用workqueue 增强处理逻辑的
```

单纯基于 client-go informer 可以监听 object 变化并做出处理，但仍然有很多问题，还需进一步的封装，于是引出了controller-runtime。
1. 多个object 的informer 与 多个worklaod 的reconcile 可能具有多对多关系
	1. 一个workload 可能需要针对多个 object 的event 进行reconcile
	2. 多个workload，每一个workload 都持有 podInformer 会有重复的问题，为解决这个问题，需要一个独立的对象（比如叫cache）持有所有用到的informer（一个object 一个informer），**向informer 注册eventhandler 也应改为向 cache 注册eventhandler**。
2. reconcile 时失败、延迟重试等功能


[Controller Runtime 的四种使用姿势](https://mp.weixin.qq.com/s/zWvxbO1C2QrZY7iqvGtMGA)
```go
func start() {
  	scheme := runtime.NewScheme()
  	_ = corev1.AddToScheme(scheme)
  	// 1. init Manager	初始化 Manager，同时生成一个默认配置的 Cache
  	mgr, _ := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
    	Scheme: scheme,
    	Port:   9443,
  	})
  	// 2. init Reconciler（Controller）
  	_ = ctrl.NewControllerManagedBy(mgr).	// 使用了builder 模式
    	For(&corev1.Pod{}).				// 指定 watch 的资源类型
		// .Owns()						// 表示某资源是我关心资源的从属，其 event 也会进去 Controller 的队列中；
    	Complete(&ApplicationReconciler{})	// 将用户的 Reconciler 注册进 Controller，并生成 watch 资源的默认 eventHandler，同时执行 Controller 的 watch 函数；
  	// 3. start Manager
  	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
  	}
}
type ApplicationReconciler struct {
}
func (a ApplicationReconciler) Reconcile(ctx context.Context, request reconcile.Request) (reconcile.Result, error) {
  	return reconcile.Result{}, nil
}
```
上述代码经过了builder 模式的封装，相对底层的 样子如下
```go
func start() {
  scheme := runtime.NewScheme()
  _ = corev1.AddToScheme(scheme)
  // 1. init Manager
  mgr, _ := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
    Scheme: scheme,
    Port:   9443,
  })
  // 2. init Reconciler（Controller）
  c, _ := controller.New("app", mgr, controller.Options{
	Reconciler: &ApplicationReconciler{},
  })
  // 监控Pod变动并将key写入workqueue
  _ = c.Watch(&source.Kind{Type: &corev1.Pod{}}, &handler.EnqueueRequestForObject{}, predicate.Funcs{
    CreateFunc: func(event event.CreateEvent) bool {...},
    UpdateFunc: func(updateEvent event.UpdateEvent) bool {...},
    DeleteFunc: func(deleteEvent event.DeleteEvent) bool {...},
  })
  // 3. start Manager
  if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
  }
}
type ApplicationReconciler struct {
	...
}
func (a ApplicationReconciler) Reconcile(ctx context.Context, request reconcile.Request) (reconcile.Result, error) {
  	// 初始化log
	log := log.FromContext(ctx)
	// 从缓存中获取Pod
	pod := &corev1.Pod{}
	err := r.client.Get(ctx, request.NamespacedName, rs)	// 使用controller-runtime的Client时，我们只需要提供资源名（Namespace/Name）、资源类型（结构体指针），即可开始CRUD操作。
	if errors.IsNotFound(err) {
		log.Error(nil, "Could not find Pod")
		return reconcile.Result{}, nil
	}
	if err != nil {
		return reconcile.Result{}, fmt.Errorf("could not fetch Pod: %+v", err)
	}
	// 打印Pod
	log.Info("Reconciling Pod", "pod name", pod.Name)
	return reconcile.Result{}, nil
}
```

我们可以看到，原先复杂的准备工作现在已经简化为几个步骤：

1. 创建manager
2. 创建controller，添加需要监控的资源
1. 实现Reconcile方法，处理资源变动

## 整体设计

![](/public/upload/kubernetes/controller_runtime_overview.jpg)

Manager 管理多个Controller 的运行，并提供 数据读（cache）写（client）等crudw基础能力，或者说 Manager 负责初始化cache、clients 等公共依赖，并提供个runnbale 使用。

![](/public/upload/kubernetes/controller_runtime.png)

1. Cache, 负责在 Controller 进程里面根据 Scheme 同步 Api Server 中所有该 Controller 关心 的资源对象，其核心是 相关Resource的 Informer,Informer 会负责监听对应 Resource的创建/删除/更新操作，以触发 Controller 的 Reconcile 逻辑。
2. Clients,  Reconciler不可避免地需要对某些资源类型进行crud，就是通过该 Clients 实现的，其中查询功能实际查询是本地的 Cache，写操作直接访问 Api Server。

![](/public/upload/kubernetes/controller_runtime_logic.png)

Cache 顾名思义就是缓存，用于建立 Informer 对 ApiServer 进行连接 watch 资源，并将 watch 到的 object 推入队列；Controller 一方面会向 Informer 注册 eventHandler，另一方面会从队列中拿数据并执行用户侧 Reconciler 的函数。A Controller manages a work queue fed reconcile.Requests from source.Sources.  Work is performed through the reconcile.Reconciler for each enqueued item. Work typically is reads and writes Kubernetes objects to make the system state match the state specified in the object Spec. 

当我们观察Manager/Contronller/Reconciler Interface 的时候，接口定义是非常清晰的，但是为了实现接口定义的 能力（方法）要聚合很多struct，初始化时要为它们赋值，这个部分代码实现的比较复杂，可能是go 缺少类似ioc 工具带来的问题。 

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
	// manager -> controller -> reconciler 的对象层级结构
	// 主体分为两部分，一是配置manager，二是启动manager
    // 1. init Manager 
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:             scheme,		// 要将你监听的crd 加入到scheme 中
		Port:               9443,})
    // 2. init Reconciler（Controller）
	c := &controllers.ApplicationReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}
	if err = c.SetupWithManager(mgr); err != nil {...}
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
	c, err := controller.New(r.ControllerName(), mgr, controller.Options{
		Reconciler: r,
	})
	// 为Controller指定cr
	c.Watch(&source.Kind{Type: &appsv1alpha1.Application{}}, &handler.EnqueueRequestForObject{},
		predicate.Funcs{CreateFunc: r.onOwnerCreateFunc()},
	)
	return nil
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

## Manager 启动

[controller-runtime 之 manager 实现](https://mp.weixin.qq.com/s/3i3t-PBP3UN8W9quEhAQDQ)
Manager interface 充分体现了它的作用：添加Controller 并Start 它们。 

![](/public/upload/kubernetes/manager_controller.png)

manager中可以设置多个controller，但是一个controller中只有一个Reconciler。

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

Manager 可以管理 Runnable的生命周期（添加/启动），**Controller  只是 Runnable 的一个特例**。
1. 持有Runnable共同的依赖：client、cache、scheme 等。
2. 提供了object getter(例如GetClient())，还有一个简单的依赖注入机制(runtime/inject)，
3. 支持领导人选举，提供了一个用于优雅关闭的信号处理程序。PS：所以哪怕 不是处理crd，普通的一个 服务端程序如果需要选主，也可以使用Manager

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

## Controller


Controller 管理一个工作队列，并从 source.Sources 中获取 reconcile.Requests 加入队列， 通过执行 reconcile.Reconciler 来处理队列中的每项 reconcile.Requests。Controller 逻辑主要有两个，对应两个函数是 Watch 与 Start
1. 监听 object 事件并加入到 queue 中。
	1. Controller 会先向 Informer 注册特定资源的 eventHandler；然后 Cache 会启动 Informer，Informer 向 ApiServer 发出请求，建立连接；当 Informer 检测到有资源变动后，使用 Controller 注册进来的 eventHandler 判断是否推入队列中；
	1. 为提高扩展性 Controller 将这个职责独立出来交给了 Source 组件，不只是监听apiserver，任何外界资源变动 都可以通过 Source 接口加入 到Reconcile 逻辑中。
2. 从queue 中取出object event 执行Reconcile 逻辑。 PS：**一个controller 持有了一个queue，一手包办了queue的生产和消费**。

![](/public/upload/kubernetes/controller_runtime_controller.png)

[controller-runtime 之控制器实现](https://mp.weixin.qq.com/s/m-eNII-h-Gq74bMZ3fQLKg)

```go
type Controller interface {
	// Reconciler is called to reconcile an object by Namespace/Name
	reconcile.Reconciler
	// Watch takes events provided by a Source and uses the EventHandler to enqueue reconcile.Requests in response to the events.
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


### watch

![](/public/upload/kubernetes/controller_watch.png)

以 tf-job的 TFJobReconciler 为例
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
Source 抽象了事件源，event 由 具体实现提供，将其加入到  workqueue。
1.  Kind 来处理来自集群的事件（如 Pod 创建、Pod 更新、Deployment 更新）；
2.  Channel 来处理来自集群外部的事件（如 GitHub Webhook 回调、轮询外部 URL）。
```go
// pkg/source/source.go
type Source interface {  
   // Start() 是 Controller-runtime 的内部⽅法，应该仅由 Controller 调⽤
   Start(context.Context, handler.EventHandler, workqueue.RateLimitingInterface, ...predicate.Predicate) error  
}
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
// sigs.k8s.io/controller-runtime/pkg/internal/controller/controller.go
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

类比spring ioc，“受体” 想要自己的某个field 被赋值，应该支持 setXX 方法。controller-runtimey类似，“受体”想要cache 成员被赋值，即需要实现 InjectCache 方法（也就实现了 Cache interface）。

```go
// controller-runtime/pkg/runtime/inject/inject.go
type Cache interface {	// 很像spring 中的XXAware
	InjectCache(cache cache.Cache) error
}
func CacheInto(c cache.Cache, i interface{}) (bool, error) {
	if s, ok := i.(Cache); ok {
		return true, s.InjectCache(c)
	}
	return false, nil
}
type Config interface {
	InjectConfig(*rest.Config) error
}
type Client interface {
	InjectClient(client.Client) error
}
type Scheme interface {
	InjectScheme(scheme *runtime.Scheme) error
}
...
```
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

一般一个crd 对应一个 reconciler。以Application crd为例，可能需要根据pod 变化采取动作，因此Application Controller 需要监听  Application及关联的pod的变更。

Watch函数支持三种资源监听类型，通过定义 eventhandler 实现：

1. EnqueueRequestForObject：资源变动时将资源key加入workqueue，例如直接监听Pod变动
2. EnqueueRequestForOwner：资源变动时将资源owner的key加入workqueue，例如在Pod变动时，若Pod的Owner为ReplicaSet，则通知ReplicaSet发生了资源变动
3. EnqueueRequestsFromMapFunc：定义一个关联函数，资源变动时生成一组reconcile.Request，例如在集群扩容时添加了Node，通知一组对象发生了资源变动

此外还可以自定义 predicates，用于过滤资源，例如只监听指定命名空间、包含指定注解或标签的资源。




