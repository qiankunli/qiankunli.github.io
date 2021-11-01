---

layout: post
title: kubernetes扩缩容
category: 技术
tags: Kubernetes
keywords: kubernetes autoscaler

---

## 简介（未完成）

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



## HPA和VPA工作原理——CRD的典型应用

![](/public/upload/kubernetes/auto_scaler.png)


1. hpa 和 vpa 做出决策依赖 metric server 提供的metric 数据

	![](/public/upload/kubernetes/kubernetes_metric_server.png)
2. Kubernetes 本身“安装” hpa 和 vpa 的CRD，以支持vpa or hpa Kubernetes object 
3. 对于每个应用，创建一个对象的vpa or hpa对象
4. hpa or vpa CRD 不停的拉取metric 数据，根据hpa or vpa 对象配置的策略，计算出pod 的最佳replica（hpa）或resource（vpa），更改deployment 配置，重启deployment



## Vertical Pod Autoscaler

[Kubernetes 垂直自动伸缩走向何方?](https://mp.weixin.qq.com/s/ykWgx1WJxBFSPidD1To53Q)垂直自动伸缩（VPA，Vertical Pod Autoscaler） 是一个基于历史数据、集群可使用资源数量和实时的事件（如 OMM， 即 out of memory）来自动设置Pod所需资源并且能够在运行时自动调整资源基础服务。

### 示例

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

### 原理

综述：

1. 提出新的API资源: VerticalPodAutoscaler 。它包括一个标签识别器 label selector（匹配Pod）、资源策略 resources policy（控制VPA如何计算资源）、更新策略 update policy（控制资源变化应用到Pod）和推荐资源信息。
2. VPA Recommender 是一个新的组件，它考虑集群中来自 Metrics Server 的所有 Pod 的资源利用率信号和内存溢出事件。
3. VPA Recommender 会监控所有 Pod，为每个 Pod 持续计算新的推荐资源，并将它们存储到 VPA Object 中。
4. VPA Recommender 会暴露一个同步 API 获取 Pod 详细信息并返回推荐信息。
5. 所有的 Pod 创建请求都会通过 VPA Admission Controller。如果 Pod 与任何一个 VPA 对象匹配，那么 Admission controller 会依据 VPA Recommender 推荐的值重写容器的资源。如果 Recommender 连接不上，它将会返回 VPA Object 中缓存的推荐信息。
6. VPA Updater 是负责实时更新 Pod 的组件。如果一个 Pod 使用 VPA 的自动模式，那么 Updater 会依据推荐资源来决定如何更新。在 MVP 模式中，这需要通过删除 Pod 然后依据新的资源重建 Pod 来实现，这种方法需要 Pod 属于一个 Replica Set（或者其他能够重新创建它的组件）。在未来，Updater 会利用原地升级，因为重新创建或者重新分配Pod对服务是很有破坏性的，必须尽量减少这种操作。
7. VPA 仅仅控制容器的资源请求,它把资源限制设置为无限,资源请求的计算基于对当前和过去运行状况的分析。
8. History Storage 是从 API Server 中获取资源利用率信号和内存溢出并将它们永久保存的组件。Recommender 在一开始用这些历史数据来初始化状态。History Storage 基础的实现是使用 Prometheus。

updatePolicy

1. Intitial: VPA 只在创建 Pod 时分配资源，在 Pod 的其他生命周期不改变Pod的资源。
2. Auto(默认)：VPA 在 Pod 创建时分配资源，并且能够在 Pod 的其他生命周期更新它们，包括淘汰和重新调度 Pod。
3. Off：VPA 从不改变Pod资源。Recommender 而依旧会在VPA对象中生成推荐信息，他们可以被用在演习中。



## Cluster Auto Scaler 

[kubernetes 资源管理概述](https://cizixs.com/2018/06/25/kubernetes-resource-management/)

随着业务的发展，应用会逐渐增多，每个应用使用的资源也会增加，总会出现集群资源不足的情况。为了动态地应对这一状况，我们还需要 CLuster Auto Scaler，能够根据整个集群的资源使用情况来增减节点。CA 是面向事件工作的，并每 10 秒检查一次是否存在不可调度（Pending）的 Pod(当调度器无法找到可以容纳 Pod 的节点时，这个 Pod 是不可调度的）。此时，CA 开始创建新节点。

对于公有云来说，Cluster Auto Scaler 就是监控这个集群因为资源不足而 pending 的 pod，根据用户配置的阈值调用公有云的接口来申请创建机器或者销毁机器。对于私有云，则需要对接内部的管理平台。

[Kubernetes 的自动伸缩你用对了吗？](https://mp.weixin.qq.com/s/GKS3DJHm4p0Tjtj8nJRGmA)


## 自定义crd 支持autoscaler

[Scale subresource](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/#scale-subresource) CustomResourceDefinition 需要在定义中支持 scale subresource

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: crontabs.stable.example.com
spec:
  group: stable.example.com
  versions:
    - name: v1
      served: true
      storage: true
      schema: ...
      subresources:
        # status enables the status subresource.
        status: {}
        # scale enables the scale subresource.
        scale:
          # specReplicasPath defines the JSONPath inside of a custom resource that corresponds to Scale.Spec.Replicas.
          specReplicasPath: .spec.replicas
          # statusReplicasPath defines the JSONPath inside of a custom resource that corresponds to Scale.Status.Replicas.
          statusReplicasPath: .status.replicas
          # labelSelectorPath defines the JSONPath inside of a custom resource that corresponds to Scale.Status.Selector.
          labelSelectorPath: .status.labelSelector
  scope: Namespaced
  names:
    plural: crontabs
    singular: crontab
    kind: CronTab
    shortNames:
    - ct
```
apply crd 之后 apiserver 提供api `/apis/stable.example.com/v1/namespaces/*/crontabs/scale`，也可以 通过  `kubectl scale --replicas=5 crontabs/my-new-cron-object` 命令来修改 `CronTab.spec.replicas`。 可以直接修改 yaml  `CronTab.spec.replicas` 来控制crd 的副本数，但无法与 hpa 等组件集成（hpa 使用rest api）。
```
apiVersion: "stable.example.com/v1"
kind: CronTab
metadata:
  name: my-new-cron-object
spec:
  cronSpec: "* * * * */5"
  image: my-awesome-cron-image
  replicas: 3
```


## 其它


[Kubernetes 垂直自动伸缩走向何方?](https://mp.weixin.qq.com/s/ykWgx1WJxBFSPidD1To53Q)**通常用户在无状态的工作负载时选用 HPA，在有状态的工作负载时选用 VPA**。

理论上 HPA 和 VPA 是可以共同工作的，**HPA 负责瓶颈资源，VPA 负责其他资源**。比如对于 CPU 密集型的应用，使用 HPA 监听 CPU 使用率来调整 pods 个数，然后用 VPA 监听其他资源（memory、IO）来动态扩展这些资源的 request 大小即可。当然这只是理想情况

[Pod 垂直自动伸缩的使用](https://mp.weixin.qq.com/s/R56Ls6eiuSyFWUki2EfhvA) vpa局限性
1. 不能和 HPA 一起使用
2. 需要至少两个健康的 Pod 才能工作。由于 VPA 会破坏一个 Pod，并重新创建一个 Pod 来进行垂直自动伸缩，因此它需要至少两个监控的 Pod 副本来确保不会出现服务中断。
3. 默认最小内存分配为250MiB
4. 不能用于单个独立的 Pod。VPA 只适用于 Deployments、StatefulSets、DaemonSets、ReplicaSets 等控制器

目前 VPA 在生产中的最佳方式是在**推荐模式**下使用，这有助于我们了解最佳的资源请求值是多少，以及随着时间推移它们是如何变化的。
一旦配置了，我们就可以通过获取这些 metrics 指标，并将其发送到监控工具中去，比如 Prometheus 和 Grafana 或者 ELK 技术栈。然后可以利用这些数据来调整 Pods 的大小。




