---

layout: post
title: Kubernetes 控制器模型
category: 技术
tags: Kubernetes
keywords: kubernetes scheduler

---

## 简介

* TOC
{:toc}

《阿里巴巴云原生实践15讲》 K8S 的关键词就是最终一致性，所有的 Controller 都会朝着最终一致 性不断 sync。PS：文章里经常出现一个词：面向终态。

## 一种新的模型

[Kubernetes: Controllers, Informers, Reflectors and Stores](http://borismattijssen.github.io/articles/kubernetes-informers-controllers-reflectors-stores)

**We really like the Kubernetes ideology of seeing the entire system as a control system. That is, the system constantly tries to move its current state to a desired state**.The worker units that guarantee the desired state are called controllers. 控制器就是保证系统按 desired state运行。

## 控制器模型

[kube-controller-manager](https://kubernetes.io/docs/reference/command-line-tools-reference/kube-controller-manager/) **In applications of robotics and automation, a control loop is a non-terminating loop that regulates the state of the system**（在自动化行业是常见方式）. In Kubernetes, a controller is a control loop that watches the shared state of the cluster through the API server and makes changes attempting to move the current state towards the desired state. Examples of controllers that ship with Kubernetes today are the replication controller, endpoints controller, namespace controller, and serviceaccounts controller.

docker是单机版的，当我们接触k8s时，天然的认为这是一个集群版的docker，再具体的说，就在在集群里给镜像找一个主机来运行容器。但实际上比调度更重要的是编排，那么编排如何实现呢？控制器

声明式API对象与控制器模型相辅相成，声明式API对象定义出期望的资源状态，控制器模型则通过控制循环（Control Loop）将Kubernetes内部的资源调整为声明式API对象期望的样子。因此可以认为声明式API对象和控制器模型，才是Kubernetes项目编排能力“赖以生存”的核心所在。

### 有什么

controller是一系列控制器的集合，不单指RC。

	$ cd kubernetes/pkg/controller/
	$ ls -d */              
	deployment/             job/                    podautoscaler/          
	cloud/                  disruption/             namespace/              
	replicaset/             serviceaccount/         volume/
	cronjob/                garbagecollector/       nodelifecycle/          replication/            statefulset/            daemon/
	...

### 整体架构

这些控制器之所以被统一放在 pkg/controller 目录下，就是因为它们都遵循 Kubernetes 项目中的一个通用编排模式，即：控制循环（control loop）。 

	for {
	  实际状态 := 获取集群中对象 X 的实际状态（Actual State）
	  期望状态 := 获取集群中对象 X 的期望状态（Desired State）
	  if 实际状态 == 期望状态{
	    什么都不做
	  } else {
	    执行编排动作，将实际状态调整为期望状态
	  }
	}

实际状态往往来自于 Kubernetes 集群本身。 比如，**kubelet 通过心跳汇报的容器状态和节点状态**，或者监控系统中保存的应用监控数据，或者控制器主动收集的它自己感兴趣的信息。而期望状态，一般来自于用户提交的 YAML 文件。 比如，Deployment 对象中 Replicas 字段的值，这些信息往往都保存在 Etcd 中。

![](/public/upload/kubernetes/k8s_controller_definition.PNG)

## 整体架构

[通过自定义资源扩展Kubernetes](https://blog.gmem.cc/extend-kubernetes-with-custom-resources)

[A Deep Dive Into Kubernetes Controllers](https://engineering.bitnami.com/articles/a-deep-dive-into-kubernetes-controllers.html) 

![](/public/upload/kubernetes/k8s_custom_controller.png)

### 控制器与Informer——如何高效监听一个http server

控制器与api server的关系——从拉取到监听：In order to retrieve an object's information, the controller sends a request to Kubernetes API server.However, repeatedly retrieving information from the API server can become expensive. Thus, in order to get and list objects multiple times in code, Kubernetes developers end up using cache which has already been provided by the client-go library. Additionally, the controller doesn't really want to send requests continuously. It only cares about events when the object has been created, modified or deleted. 

[从 Kubernetes 资源控制到开放应用模型，控制器的进化之旅](https://mp.weixin.qq.com/s/AZhyux2PMYpNmWGhZnmI1g)

1. Controller 一直访问API Server 导致API Server 压力太大，于是有了Informer
2. 由 Informer 代替Controller去访问 API Server，而Controller不管是查状态还是对资源进行伸缩都和 Informer 进行交接。而且 Informer 不需要每次都去访问 API Server，它只要在初始化的时候通过 LIST API 获取所有资源的最新状态，然后再通过 WATCH API 去监听这些资源状态的变化，整个过程被称作 ListAndWatch。
3. Informer 也有一个助手叫 Reflector，上面所说的 ListAndWatch 事实上是由 Reflector 一手操办的。这使 API Server 的压力大大减少。
4. 后来，WATCH 数据的太多了，Informer/Reflector去 API Server 那里 WATCH 状态的时候，只 WATCH 特定资源的状态，不要一股脑儿全 WATCH。
5. 一个controller 一个informer 还是压力大，于是针对每个（受多个控制器管理的）资源弄一个 Informer。比如 Pod 同时受 Deployment 和 StatefulSet 管理。这样当多个控制器同时想查 Pod 的状态时，只需要访问一个 Informer 就行了。
6. 但这又引来了新的问题，SharedInformer 无法同时给多个控制器提供信息，这就需要每个控制器自己排队和重试。为了配合控制器更好地实现排队和重试，SharedInformer  搞了一个 Delta FIFO Queue（增量先进先出队列），每当资源被修改时，它的助手 Reflector 就会收到事件通知，并将对应的事件放入 Delta FIFO Queue 中。与此同时，SharedInformer 会不断从 Delta FIFO Queue 中读取事件，然后更新本地缓存的状态。
7. 这还不行，SharedInformer 除了更新本地缓存之外，还要想办法将数据同步给各个控制器，为了解决这个问题，它又搞了个工作队列（Workqueue），一旦有资源被添加、修改或删除，就会将相应的事件加入到工作队列中。所有的控制器排队进行读取，一旦某个控制器发现这个事件与自己相关，就执行相应的操作。如果操作失败，就将该事件放回队列，等下次排到自己再试一次。如果操作成功，就将该事件从队列中删除。

**Informer 约等于apiserver client sdk**：Informer其实就是一个带有本地缓存和索引机制的、可以注册EventHandler( AddFunc、UpdateFunc 和 DeleteFunc)的 数据+事件总线(event bus)。通过监听etcd数据变化，Informer 可以实时地更新本地缓存，并且调用这些事件对应的 EventHandler。在istio pilot 以及 各种配置中心client sdk 中都有类似逻辑，与远程数据保持同步 并在数据变化时触发业务代码注册的回调函数。 **从这个视角看，Informer 直接使用go etcd client 监听etcd的话就不用这么费事了，当然可能在安全机制上或许有漏洞**。

![](/public/upload/kubernetes/control_loop.png)

从代码上看 informer 由两个部分组成

1. Listwatcher is a combination of a list function and a watch function for a specific resource in a specific namespace. 
2. Resource Event Handler is where the controller handles notifications for changes on a particular resource

		type ResourceEventHandlerFuncs struct {
			AddFunc    func(obj interface{})
			UpdateFunc func(oldObj, newObj interface{})
			DeleteFunc func(obj interface{})
		}


## 单个Controller的工作原理

controller 与上下游的边界/接口如下图

![](/public/upload/kubernetes/controller_overview.png)

事实上，一个controller 通常依赖多个Informer

![](/public/upload/kubernetes/controller_context.png)

从DeploymentController 及 ReplicaSetController 观察到的共同点

1. struct 中都包含获取 决策所以依赖 资源的lister，对于DeploymentController 是DeploymentLister/ReplicaSetLister/PodLister ，对于ReplicaSetController 是ReplicaSetLister和 PodLister
2. struct 中都包含 workqueue， workqueue 数据生产者是 Controller 注册到所以依赖的 Informer 的AddFunc/updateFunc/DeleteFunc，workqueue 数据消费者是  control loop ，每次循环都是 从workqueue.Get 数据开始的。 
3. struct 都包含 kubeClient 类型为 clientset.Interface，controller 比对新老数据 将决策 具体为“指令”使用kubeClient写入 apiserver ，然后 scheduler 和 kubelet 负责干活儿。
4. 相同的执行链条：`Run ==> go worker ==> for processNextWorkItem ==> syncHandler`。Run 方法作为 Controller 逻辑的统一入口，启动指定数量个协程，协程的逻辑为：`wait.Until(dc.worker, time.Second, stopCh)` ，control loop 具体为Controller 的worker 方法，for 循环具体为 `for processNextWorkItem(){}`，两个Controller 的processNextWorkItem 逻辑相似度 90%： 从queue 中get一个key，使用syncHandler 处理，处理成功就标记成功，处理失败就看情况将key 重新放入queue。

    ```go
    func processNextWorkItem() bool {
        key, quit := queue.Get()
        if quit { return false}
        defer queue.Done(key)
        err := syncHandler(key.(string))
        handleErr(err, key)
        return true
    }
    ```

Controller Mananger 的主要逻辑便是 先初始化 资源（重点就是Informer） 并启动Controller。可以将Controller 和 Informer 视为两个组件，也可以认为只有Controller 一个，比如[kubectl 创建 Pod 背后到底发生了什么？](https://mp.weixin.qq.com/s/ctdvbasKE-vpLRxDJjwVMw)将 Deployment 记录存储到 etcd 并初始化后，就可以通过 kube-apiserver 使其可见，DeploymentController工作就是负责监听 Deployment 记录的更改——控制器通过 Informer 注册cud事件的回调函数。

![](/public/upload/kubernetes/controller_object.png)



### 外围——循环及数据获取

	// Run begins watching and syncing.
	func (dc *DeploymentController) Run(workers int, stopCh <-chan struct{}) {
		defer utilruntime.HandleCrash()
		defer dc.queue.ShutDown()
		if !controller.WaitForCacheSync("deployment", stopCh, dc.dListerSynced, dc.rsListerSynced, dc.podListerSynced) {
			return
		}
		for i := 0; i < workers; i++ {
			go wait.Until(dc.worker, time.Second, stopCh)
		}
		<-stopCh
	}

重点就是 `go wait.Until(dc.worker, time.Second, stopCh)`。for 循环隐藏在 `k8s.io/apimachinery/pkg/util/wait/wait.go` 工具方法中，`func Until(f func(), period time.Duration, stopCh <-chan struct{}) {...}` 方法的作用是  Until loops until stop channel is closed, running f every period. 即在stopCh 标记停止之前，每隔 period 执行 一个func，对应到DeploymentController 就是 worker 方法

	// worker runs a worker thread that just dequeues items, processes them, and marks them done.
	// It enforces that the syncHandler is never invoked concurrently with the same key.
	func (dc *DeploymentController) worker() {
		for dc.processNextWorkItem() {
		}
	}
	
	func (dc *DeploymentController) processNextWorkItem() bool {
		// 取元素
		key, quit := dc.queue.Get()
		if quit {
			return false
		}
		// 结束前标记元素被处理过
		defer dc.queue.Done(key)
		// 处理元素
		err := dc.syncHandler(key.(string))
		dc.handleErr(err, key)
		return true
	}

`dc.syncHandler` 实际为 DeploymentController  的syncDeployment方法

### 一次调协（Reconcile）

syncDeployment 包含 扩容、rollback、rolloutRecreate、rolloutRolling 我们裁剪部分代码，以最简单的 扩容为例

	// syncDeployment will sync the deployment with the given key.
	func (dc *DeploymentController) syncDeployment(key string) error {
		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		deployment, err := dc.dLister.Deployments(namespace).Get(name)
		// List ReplicaSets owned by this Deployment, while reconciling ControllerRef through adoption/orphaning.
		rsList, err := dc.getReplicaSetsForDeployment(d)
		scalingEvent, err := dc.isScalingEvent(d, rsList)
		if scalingEvent {
			return dc.sync(d, rsList)
		}
		...
	}

	// sync is responsible for reconciling deployments on scaling events or when they are paused.
	func (dc *DeploymentController) sync(d *apps.Deployment, rsList []*apps.ReplicaSet) error {
		newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(d, rsList, false)
		...
		dc.scale(d, newRS, oldRSs);
		...
		allRSs := append(oldRSs, newRS)
		return dc.syncDeploymentStatus(allRSs, newRS, d)
	}

scale要处理 扩容或 RollingUpdate  各种情况，此处只保留扩容逻辑。 

	func (dc *DeploymentController) scale(deployment *apps.Deployment, newRS *apps.ReplicaSet, oldRSs []*apps.ReplicaSet) error {
		// If there is only one active replica set then we should scale that up to the full count of the
		// deployment. If there is no active replica set, then we should scale up the newest replica set.
		if activeOrLatest := deploymentutil.FindActiveOrLatest(newRS, oldRSs); activeOrLatest != nil {
			if *(activeOrLatest.Spec.Replicas) == *(deployment.Spec.Replicas) {
				return nil
			}
			_, _, err := dc.scaleReplicaSetAndRecordEvent(activeOrLatest, *(deployment.Spec.Replicas), deployment)
			return err
		}
		...
	}

	func (dc *DeploymentController) scaleReplicaSetAndRecordEvent(rs *apps.ReplicaSet, newScale int32, deployment *apps.Deployment) (bool, *apps.ReplicaSet, error) {
		// No need to scale
		if *(rs.Spec.Replicas) == newScale {
			return false, rs, nil
		}
		var scalingOperation string
		if *(rs.Spec.Replicas) < newScale {
			scalingOperation = "up"
		} else {
			scalingOperation = "down"
		}
		scaled, newRS, err := dc.scaleReplicaSet(rs, newScale, deployment, scalingOperation)
		return scaled, newRS, err
	}

	func (dc *DeploymentController) scaleReplicaSet(rs *apps.ReplicaSet, newScale int32, deployment *apps.Deployment, scalingOperation string) (bool, *apps.ReplicaSet, error) {
		sizeNeedsUpdate := *(rs.Spec.Replicas) != newScale
		annotationsNeedUpdate := ...
		scaled := false
		var err error
		if sizeNeedsUpdate || annotationsNeedUpdate {
			rsCopy := rs.DeepCopy()
			*(rsCopy.Spec.Replicas) = newScale
			deploymentutil.SetReplicasAnnotations...
			// 调用api 接口更新 对应ReplicaSet 的数据
			rs, err = dc.client.AppsV1().ReplicaSets(rsCopy.Namespace).Update(rsCopy)
			...
		}
		return scaled, rs, err
	}

调用api 接口更新Deployment 对象本身的数据

	// syncDeploymentStatus checks if the status is up-to-date and sync it if necessary
	func (dc *DeploymentController) syncDeploymentStatus(allRSs []*apps.ReplicaSet, newRS *apps.ReplicaSet, d *apps.Deployment) error {
		newStatus := calculateStatus(allRSs, newRS, d)
		if reflect.DeepEqual(d.Status, newStatus) {
			return nil
		}
		newDeployment := d
		newDeployment.Status = newStatus
		_, err := dc.client.AppsV1().Deployments(newDeployment.Namespace).UpdateStatus(newDeployment)
		return err
	}








