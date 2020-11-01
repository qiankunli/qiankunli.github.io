---

layout: post
title: kubernetes crd 及kubebuilder学习
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

对于 CRD Controller 的构建，有几个主流的工具
1.  coreOS 开源的 Operator-SDK（https://github.com/operator-framework/operator-sdk ）
2.  K8s 兴趣小组维护的 Kubebuilder（https://github.com/kubernetes-sigs/kubebuilder ）

[kubebuilder](https://github.com/kubernetes-sigs/kubebuilder) 是一个用来帮助用户快速实现 Kubernetes CRD Operator 的 SDK。当然，kubebuilder 也不是从0 生成所有controller 代码，k8s 提供给一个 [Kubernetes controller-runtime Project](https://github.com/kubernetes-sigs/controller-runtime)  a set of go libraries for building Controllers. controller-runtime 在Operator SDK中也有被用到。

[controller-runtime 之控制器实现](https://mp.weixin.qq.com/s/m-eNII-h-Gq74bMZ3fQLKg)
[controller-runtime 之 manager 实现](https://mp.weixin.qq.com/s/3i3t-PBP3UN8W9quEhAQDQ)

### 整体设计

Kubebuilder 包含以下核心组件
1. Manager
    1. 负责运行所有的 Controllers；
    2. 初始化共享 caches，包含 listAndWatch 功能；
    3. 初始化 clients 用于与 Api Server 通信。
2. Cache, 负责在 Controller 进程里面根据 Scheme 同步 Api Server 中所有该 Controller 关心 所有资源对象，其核心是 相关Resource的 Informer,Informer 会负责监听对应 Resource的创建/删除/更新操作，以触发 Controller 的 Reconcile 逻辑。
3. Clients, 在实现 Controller 的时候不可避免地需要对某些资源类型进行创建/删除/更新，就是通过该 Clients 实现的，其中查询功能实际查询是本地的 Cache，写操作直接访问 Api Server。
4. Controller, Kubebuidler 为我们生成的脚手架文件，**我们只需要实现 Reconcile 方法即可**。

![](/public/upload/kubernetes/kubebuilder_overview.png)

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

[Kubernetes之controller-runtime事件再处理](https://mp.weixin.qq.com/s/NTRog9zrSv3en9MV5_nJuQ)在controller-runtime中，Event的处理逻辑是Reconciler对象，Reconciler被controller引用，这里的controller便是控制器。在controller之上，还有一个更高层的管理者manager。manager中可以设置多个controller，但是一个controller中只有一个Reconciler。

![](/public/upload/kubernetes/controller_runtime_overview.png)



### 目录结构

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
2.  创建 API `kubebuilder create api --group apps --version v1alpha1 --kind Application`
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
        main.go                         // ApplicationReconciler 添加到 Manager
        go.mod                          
    ```

Kubernetes 有类型系统，cr 要符合相关的规范（一部分由开发人员手动编写，另一些由代码生成器生成），包括

1. struct 定义放在 项目 `pkg/apis/$group/$version` 包下
2. struct 嵌入 TypeMeta struct  ObjectMeta，定义spec 和 status

## 启动流程

启动逻辑比较简单：
1. 初始化Manager；初始化流程主要是创建cache 与 Clients。 
    1. 创建 Cache，可以看到 Cache 主要就是创建了 InformersMap，Scheme 里面的每个 GVK 都创建了对应的 Informer，通过 informersByGVK 这个 map 做 GVK 到 Informer 的映射，每个 Informer 会根据 ListWatch 函数对对应的 GVK 进行 List 和 Watch。
    2. 创建 Clients，读操作使用上面创建的 Cache，写操作使用 K8s go-client 直连。
2. 将 Manager 的 Client 传给 Controller，并且调用 SetupWithManager 方法传入 Manager 进行 Controller 的初始化；
3. 启动 Manager。

```go
func main() {
	...
    // 1. init Manager
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:             scheme,
		MetricsBindAddress: metricsAddr,
		Port:               9443,
		LeaderElection:     enableLeaderElection,
		LeaderElectionID:   "1f6a832c.example.io",
	})
    // 2. init Reconciler（Controller）
	if err = (&controllers.ApplicationReconciler{
		Client: mgr.GetClient(),
		Log:    ctrl.Log.WithName("controllers").WithName("Application"),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {...}
    // 3. start Manager
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {...}
}
```

### Controller 初始化

```go
type ApplicationReconciler struct {
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
}
func (r *ApplicationReconciler) Reconcile(req ctrl.Request) (ctrl.Result, error) {
	_ = context.Background()
	_ = r.Log.WithValues("application", req.NamespacedName)
	// your logic here  需要扩充的业务逻辑
	return ctrl.Result{}, nil
}
func (r *ApplicationReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&appsv1alpha1.Application{}).
		Complete(r)
}
type Controller struct {
	// Name is used to uniquely identify a Controller in tracing, logging and monitoring.  Name is required.
	Name string
	Do reconcile.Reconciler
	// Client is a lazily initialized Client.  
	Client client.Client
	Scheme *runtime.Scheme
	Cache cache.Cache
	// Config is the rest.Config used to talk to the apiserver.  
	Config *rest.Config
	// Queue is an listeningQueue that listens for events from Informers and adds object keys to
	// the Queue for processing
	Queue workqueue.RateLimitingInterface
    ...
}
```

ctrl.NewControllerManagedBy = builder.ControllerManagedBy 返回一个  Builder struct，即使用 Builder 模式构建Controller 

```go
func (blder *Builder) Build(r reconcile.Reconciler) (controller.Controller, error) {
	// Set the Config
	blder.loadRestConfig()
	// Set the ControllerManagedBy   
	if err := blder.doController(r); err != nil {...}
	// Set the Watch
	if err := blder.doWatch(); err != nil {...}
	return blder.ctrl, nil
}
// Builder.doController ==> newController == controller.New 创建一个Controller struct 并将其 加入到 Manager 中
func (blder *Builder) doController(r reconcile.Reconciler) error {
	name, err := blder.getControllerName()
	ctrlOptions := blder.ctrlOptions
	ctrlOptions.Reconciler = r
	blder.ctrl, err = newController(name, blder.mgr, ctrlOptions)
	return err
}
// 对Reconciler client 和 cache 进行了初始化
func New(name string, mgr manager.Manager, options Options) (Controller, error) {
	...
	// Inject dependencies into Reconciler
	if err := mgr.SetFields(options.Reconciler); err != nil {...}
	// Create controller with dependencies set
	c := &controller.Controller{
		Do:       options.Reconciler,
		Cache:    mgr.GetCache(),
		Config:   mgr.GetConfig(),
		Scheme:   mgr.GetScheme(),
		Client:   mgr.GetClient(),
        Name:                    name,
        ...
	}
	// Add the controller as a Manager components
	return c, mgr.Add(c)
}
```
**Informer 触发eventHandler  ==> 变更resource 加入队列**：通过 Cache 我们创建了所有 Scheme 里面 GVKs 的 Informers，然后对应 GVK 的 Controller 注册了 Watch Handler 到对应的 Informer，这样一来对应的 GVK 里面的资源有变更都会触发 Handler，将变更事件写到 Controller 的事件队列中

![](/public/upload/kubernetes/kubebuilder_logic.png)
```go
func (blder *Builder) doWatch() error {
	// watch 本 Controller 负责的 CRD 
	src := &source.Kind{Type: blder.apiType}
	hdler := &handler.EnqueueRequestForObject{} //  Handler 就是将发生变更的对象的 NamespacedName 加入队列，实际注册到 Informer 上面
	err := blder.ctrl.Watch(src, hdler, blder.predicates...)
	// watch 本 CRD 管理的其他资源/managedObjects
	for _, obj := range blder.managedObjects {
		src := &source.Kind{Type: obj}
		hdler := &handler.EnqueueRequestForOwner{
			OwnerType:    blder.apiType,
			IsController: true,
		}
		if err := blder.ctrl.Watch(src, hdler, blder.predicates...); err != nil {...}
	}
	// Do the watch requests
	for _, w := range blder.watchRequest {
		if err := blder.ctrl.Watch(w.src, w.eventhandler, blder.predicates...); err != nil {...}
	}
	return nil
}
func (c *Controller) Watch(src source.Source, evthdler handler.EventHandler, prct ...predicate.Predicate) error {
	// Inject Cache into arguments   即给Source 成员赋值
	if err := c.SetFields(src); err != nil {...}
	if err := c.SetFields(evthdler); err != nil {...}
	for _, pr := range prct {
		if err := c.SetFields(pr); err != nil {...}
	}
	c.watches = append(c.watches, watchDescription{src: src, handler: evthdler, predicates: prct})
	if c.Started {
		log.Info("Starting EventSource", "controller", c.Name, "source", src)
		return src.Start(evthdler, c.Queue, prct...)
	}
	return nil
}
```
Controller.setFields = controllerManager.SetFields 有点依赖注入的意思，**controllerManager 类似ioc 容器**， 持有了Config/Client/APIReader/Scheme/Cache/Injector/StopChannel/Mapper 实例，对于传入的interface（对于上例是Source），如果有这些类型的field的话，就给interface 的field 赋值。

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
```

### Manager 启动

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
**controller 是变更事件的消费者**：Controller 的初始化是启动 goroutine 不断地查询队列，如果有变更消息则触发到我们自定义的 Reconcile 逻辑。
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

## kubectl 感知 crd

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


