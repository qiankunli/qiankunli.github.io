---

layout: post
title: kubernetes自动扩容缩容
category: 技术
tags: Kubernetes
keywords: kubernetes yaml

---

## 简介

* TOC
{:toc}

开源框架[kubernetes/autoscaler](https://github.com/kubernetes/autoscaler)

[kubernetes 资源管理概述](https://cizixs.com/2018/06/25/kubernetes-resource-management/)

![](/public/upload/kubernetes/kubernetes_resource_manager.png)


[Kubernetes Autoscaling 101: Cluster Autoscaler, Horizontal Pod Autoscaler, and Vertical Pod Autoscaler](https://medium.com/magalix/kubernetes-autoscaling-101-cluster-autoscaler-horizontal-pod-autoscaler-and-vertical-pod-2a441d9ad231) 

**Kubernetes at its core is a resources management and orchestration tool**. It is ok to focus day-1 operations to explore and play around with its cool features to deploy, monitor and control your pods. However, you need to think of day-2 operations as well. You need to focus on questions like:

1. How am I going to scale pods and applications?
2. How can I keep containers running in a healthy state and running efficiently?
3. With the on-going changes in my code and my users’ workloads, how can I keep up with such changes?

## Cluster Auto Scaler 

[kubernetes 资源管理概述](https://cizixs.com/2018/06/25/kubernetes-resource-management/)

随着业务的发展，应用会逐渐增多，每个应用使用的资源也会增加，总会出现集群资源不足的情况。为了动态地应对这一状况，我们还需要 CLuster Auto Scaler，能够根据整个集群的资源使用情况来增减节点。

对于公有云来说，Cluster Auto Scaler 就是监控这个集群因为资源不足而 pending 的 pod，根据用户配置的阈值调用公有云的接口来申请创建机器或者销毁机器。对于私有云，则需要对接内部的管理平台。

## HPA和VPA工作原理——CRD的典型应用

![](/public/upload/kubernetes/auto_scaler.png)


1. hpa 和 vpa 做出决策依赖 metric server 提供的metric 数据

	![](/public/upload/kubernetes/kubernetes_metric_server.png)
2. Kubernetes 本身“安装” hpa 和 vpa 的CRD，以支持vpa or hpa Kubernetes object 
3. 对于每个应用，创建一个对象的vpa or hpa对象
4. hpa or vpa CRD 不停的拉取metric 数据，根据hpa or vpa 对象配置的策略，计算出pod 的最佳replica（hpa）或resource（vpa），更改deployment 配置，重启deployment

## Horizontal Pod Autoscaler 

配置示例`kubectl apply sample-metrics-app.yaml` 该示例根据一个custom metric 来决定是否进行横向扩容。 

	kind: HorizontalPodAutoscaler
	apiVersion: autoscaling/v2beta1
	metadata:
	  name: sample-metrics-app-hpa
	spec:
	  scaleTargetRef:
	    apiVersion: apps/v1
	    kind: Deployment
	    name: sample-metrics-app
	  minReplicas: 2
	  maxReplicas: 10
	  metrics:
	  - type: Object
	    object:
	      target:
	        kind: Service
	        name: sample-metrics-app
	      metricName: http_requests
	      targetValue: 100


1. scaleTargetRef,指定了被监控的对象是名叫 sample-metrics-app的Deployment
2. 最小的实例数目是 2，最大是 10
3. 在 metrics 字段，我们指定了这个 HPA 进行 Scale 的依据，是名叫 http_requests 的 Metrics。而获取这个 Metrics 的途径，则是访问名叫 sample-metrics-app 的 Service。
4. 有了上述约定，hpa 就可以向请求`https://<apiserver_ip>/apis/custom-metrics.metrics.k8s.io/v1beta1/namespaces/default/services/sample-metrics-app/http_requests` 来获取custome metric 的值了。




## Vertical Pod Autoscaler

[Vertical Pod Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler)


配置示例

	apiVersion: autoscaling.k8s.io/v1beta2
	kind: VerticalPodAutoscaler
	metadata:
	  name: my-rec-vpa
	spec:
	  targetRef:
	    apiVersion: "extensions/v1beta1"
	    kind:       Deployment
	    name:       my-rec-deployment
	  updatePolicy:
	    updateMode: "Off"
	    
1. targetRef 指定了被监控的对象是名叫 my-rec-deployment的Deployment
2. `kubectl create -f my-rec-vpa.yaml` 稍等片刻，然后查看 VerticalPodAutoscaler：`kubectl get vpa my-rec-vpa --output yaml`

输出结果显示推荐的 CPU 和内存请求：

	  recommendation:
    	containerRecommendations:
	    - containerName: my-rec-container
	      lowerBound:
	        cpu: 25m
	        memory: 262144k
	      target:
	        cpu: 25m
	        memory: 262144k
	      upperBound:
	        cpu: 7931m
	        memory: 8291500k
	        
target 推荐显示，容器请求25 milliCPU 和 262144 千字节的内存时将以最佳状态运行。


配置示例

	apiVersion: autoscaling.k8s.io/v1beta2
	kind: VerticalPodAutoscaler
	metadata:
	  name: my-vpa
	spec:
	  targetRef:
	    apiVersion: "extensions/v1beta1"
	    kind:       Deployment
	    name:       my-deployment
	  updatePolicy:
	    updateMode: "Auto"


1. targetRef 指定了被监控的对象是名叫 my-deployment的Deployment
2. updateMode 字段的值为 Auto，意味着VerticalPodAutoscaler 可以删除 Pod，调整 CPU 和内存请求，然后启动一个新 Pod。
3. VerticalPodAutoscaler 使用 lowerBound 和 upperBound 推荐值来决定是否删除 Pod 并将其替换为新 Pod。如果 Pod 的请求小于下限或大于上限，则 VerticalPodAutoscaler 将删除 Pod 并将其替换为具有目标推荐值的 Pod。


## 其它

目前 HPA 和 VPA 不兼容，只能选择一个使用，否则两者会相互干扰。而且 VPA 的调整需要重启 pod，这是因为 pod 资源的修改是比较大的变化，需要重新走一下 apiserver、调度的流程，保证整个系统没有问题。目前社区也有计划在做原地升级，也就是说不通过杀死 pod 再调度新 pod 的方式，而是直接修改原有 pod 来更新。

理论上 HPA 和 VPA 是可以共同工作的，HPA 负责瓶颈资源，VPA 负责其他资源。比如对于 CPU 密集型的应用，使用 HPA 监听 CPU 使用率来调整 pods 个数，然后用 VPA 监听其他资源（memory、IO）来动态扩展这些资源的 request 大小即可。当然这只是理想情况



