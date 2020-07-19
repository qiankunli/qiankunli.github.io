---

layout: post
title: Kubernetes源码分析——controller mananger
category: 技术
tags: Kubernetes
keywords: kubernetes 源码分析 controller mananger

---

## 简介

* TOC
{:toc}

建议先看下 Kubernetes 控制器实现，摸清楚 Controller 实现需要哪些资源，Controller Mananger 的主要逻辑便是 先初始化这些资源 并启动控制器

![](/public/upload/kubernetes/controller_manager.png)

来自入口 `cmd/kube-controller-manager/controller-manager.go` 的概括

The Kubernetes controller manager is a daemon that embeds the core control loops shipped with Kubernetes. In applications of robotics and
automation, a control loop is a non-terminating loop that regulates the state of the system. In Kubernetes, a controller is a control loop that watches the shared state of the cluster through the apiserver and makes changes attempting to move the current state towards the desired state.

![](/public/upload/kubernetes/controller_view.png)

## ControllerContext

Informer 是 Client-go 中的一个核心工具包。如需要 List/Get Kubernetes 中的 Object（包括pod，service等等），可以直接使用 Informer 实例中的 Lister()方法（包含 Get 和 List 方法）。Informer 最基本的功能就是 List/Get Kubernetes 中的 Object，还可以监听事件并触发回调函数等。

![](/public/upload/kubernetes/controller_context.png)

```go
type ControllerContext struct {
	// ClientBuilder will provide a client for this controller to use ClientBuilder controller.ControllerClientBuilder
	// InformerFactory gives access to informers for the controller.
	InformerFactory informers.SharedInformerFactory

	// ComponentConfig provides access to init options for a given controller
	ComponentConfig kubectrlmgrconfig.KubeControllerManagerConfiguration
	// DeferredDiscoveryRESTMapper is a RESTMapper that will defer
	// initialization of the RESTMapper until the first mapping is
	// requested.
	RESTMapper *restmapper.DeferredDiscoveryRESTMapper
	// AvailableResources is a map listing currently available resources
	AvailableResources map[schema.GroupVersionResource]bool
	// Cloud is the cloud provider interface for the controllers to use.
	// It must be initialized and ready to use.
	Cloud cloudprovider.Interface
	// Control for which control loops to be run
	// IncludeCloudLoops is for a kube-controller-manager running all loops
	// ExternalLoops is for a kube-controller-manager running with a cloud-controller-manager
	LoopMode ControllerLoopMode
	// Stop is the stop channel
	Stop <-chan struct{}
	// InformersStarted is closed after all of the controllers have been initialized and are running.  After this point it is safe,
	// for an individual controller to start the shared informers. Before it is closed, they should not.
	InformersStarted chan struct{}
	// ResyncPeriod generates a duration each time it is invoked; this is so that
	// multiple controllers don't get into lock-step and all hammer the apiserver
	// with list requests simultaneously.
	ResyncPeriod func() time.Duration
}
```

以DeploymentController 为例， 看下启动一个Controller 需要预备哪些资源

```go
func startDeploymentController(ctx ControllerContext) (http.Handler, bool, error) {
	if !ctx.AvailableResources[schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}] {
		return nil, false, nil
	}
	dc, err := deployment.NewDeploymentController(
		ctx.InformerFactory.Apps().V1().Deployments(),
		ctx.InformerFactory.Apps().V1().ReplicaSets(),
		ctx.InformerFactory.Core().V1().Pods(),
		ctx.ClientBuilder.ClientOrDie("deployment-controller"),
	)
	if err != nil {
		return nil, true, fmt.Errorf("error creating Deployment controller: %v", err)
	}
	go dc.Run(int(ctx.ComponentConfig.DeploymentController.ConcurrentDeploymentSyncs), ctx.Stop)
	return nil, true, nil
}
```

## Controller的容器——Controller Mananger

`cmd/kube-controller-manager/controller-manager.go`

controller-manager 根据用户配置启动所有controller，启动单个Controller的过程以DeploymentController为例

	Run
		CreateControllerContext
		StartControllers
			startDeploymentController
				dc = deployment.NewDeploymentController
					注册Deployment Informer 到InformerFactory
				dc.Run
					启动一个goroutine 运行 Run 方法，Run begins watching and syncing.
		ctx.InformerFactory.Start   //启动所有注册的Informer和监听资源的事件

![](/public/upload/kubernetes/controller_manager_init.png)


	k8s.io/kubernetes/cmd/kube-controller-manager/app/controllermanager.go
	// Run runs the KubeControllerManagerOptions.  This should never exit.
	func Run(c *config.CompletedConfig) error {
        //1:拿到对kube-APIserver中资源的操作句柄,创建控制器上下文 
        ctx, err := CreateControllerContext(c, rootClientBuilder, clientBuilder, stop)
        //2:初始化的所有控制器（包括apiserver的客户端，informer的回调函数等等）
        if err := StartControllers(ctx, saTokenControllerInitFunc, NewControllerInitializers(ctx.LoopMode)); err != nil {
            glog.Fatalf("error starting controllers: %v", err)
        }
        //3:启动Informer,并完成Controller最终的启动以及资源监听机制
        ctx.InformerFactory.Start(ctx.Stop)
        close(ctx.InformersStarted)
	}


	func StartControllers(ctx ControllerContext, startSATokenController InitFunc, controllers map[string]InitFunc) error {
	    ···
	    for controllerName, initFn := range controllers {
	        if !ctx.IsControllerEnabled(controllerName) {
	            glog.Warningf("%q is disabled", controllerName)
	            continue
	        }
	        time.Sleep(wait.Jitter(ctx.ComponentConfig.GenericComponent.ControllerStartInterval.Duration, ControllerStartJitter))
	        glog.V(1).Infof("Starting %q", controllerName)
	        //note : initFn为初始化controller是创建的初始化函数
	        started, err := initFn(ctx)
	        ···
	    }
	    return nil
	}
	
initFn 就是一个大而全的map[string]InitFunc 其中之一的函数

	func NewControllerInitializers(loopMode ControllerLoopMode) map[string]InitFunc {
		controllers := map[string]InitFunc{}
		controllers["endpoint"] = startEndpointController
		controllers["replicationcontroller"] = startReplicationController
		controllers["daemonset"] = startDaemonSetController
		controllers["job"] = startJobController
		controllers["deployment"] = startDeploymentController
		controllers["replicaset"] = startReplicaSetController
		...
		if loopMode == IncludeCloudLoops {
			controllers["cloud-node-lifecycle"] = startCloudNodeLifecycleController
			..
		}
		...
		return controllers
	}


