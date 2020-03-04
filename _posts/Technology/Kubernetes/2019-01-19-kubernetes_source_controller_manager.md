---

layout: post
title: Kubernetes源码分析——controller mananger
category: 技术
tags: Kubernetes
keywords: kubernetes 源码分析

---

## 简介

* TOC
{:toc}

![](/public/upload/kubernetes/controller_manager.png)

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


