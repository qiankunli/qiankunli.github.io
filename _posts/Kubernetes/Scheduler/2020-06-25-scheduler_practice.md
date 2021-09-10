---

layout: post
title: 调度实践
category: 架构
tags: Kubernetes
keywords:  Scheduler

---

## 简介（持续更新）

* TOC
{:toc}

[一篇文章搞定大规模容器平台生产落地十大实践](https://mp.weixin.qq.com/s/Cv4i5bxseMEwx1C_Annqig) 为了实现应用的高可用，需要容器在集群中合理分布

1. **拓扑约束依赖** Pod Topology Spread Constraints, 拓扑约束依赖于Node标签来标识每个Node所在的拓扑域。
2. nodeSelector & node Affinity and anti-affinity   Node affinity：指定一些Pod在Node间调度的约束。支持两种形式：              

    1. requiredDuringSchedulingIgnoredDuringExecution
    2. preferredDuringSchedulingIgnoredDuringExecution

    IgnoreDuringExecution表示如果在Pod运行期间Node的标签发生变化，导致亲和性策略不能满足，则继续运行当前的Pod。
3. Inter-pod affinity and anti-affinity, 允许用户通过**已经运行的Pod上的标签**来决定调度策略，如果Node X上运行了一个或多个满足Y条件的Pod，那么这个Pod在Node应该运行在Pod X。有两种类型

    1. requiredDuringSchedulingIgnoredDuringExecution，刚性要求，必须精确匹配
    2. preferredDuringSchedulingIgnoredDuringExecution，软性要求
4. Taints and Tolerations,  Taints和tolerations是避免Pods部署到Node，以及从Node中驱离Pod的灵活方法


## 让pod 的不同实例别再一个机器或机架上

假设给 每个node 打上机架的标签  `kubectl label node 192.168.xx.xx rack=xxx`

```yaml
spec:
  topologySpreadConstraints:
  - maxSkew: 5                              # 机架最最差的情况下，允许存在一个机架上面有5个pod
    topologyKey: rack
    whenUnsatisfiable: ScheduleAnyway       # 实在不满足，允许放宽条件
    labelSelector:
      matchLabels:    # pod选择器
        foo: bar
  - maxSkew: 1                              # 实在不满足允许节点上面跑的pod数量
    topologyKey: kubernetes.io/hostname     # 不同的主机
    whenUnsatisfiable: ScheduleAnyway       # 硬性要求不要调度到同一台机器上面
    labelSelector:
      matchLabels:
        foo: bar
```

## 亲和性与调度
[云原生高可用与容灾系列(一): Pod 打散调度](https://mp.weixin.qq.com/s/Nh4kwSy54rfe4X7zQXeh6Q)

1. 对于Pod yaml进行配置，约束一个 Pod 只能在特定的 Node(s) 上运行，或者优先运行在特定的节点上
    1. nodeSelector
    2. nodeAffinity，相对nodeSelector 更专业和灵活一点
    3. podAffinity，根据 POD 之间的关系进行调度
2. 对Node 进行配置。如果一个节点标记为 Taints ，除非 POD 也被标识为可以容忍污点节点，否则该 Taints 节点不会被调度pod。

[Assigning Pods to Nodes](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#node-affinity-beta-feature)

Node affinity, described [here](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#node-affinity-beta-feature), is a property of pods that attracts them to a set of nodes (either as a preference or a hard requirement). Taints are the opposite – they allow a node to repel（击退） a set of pods.

[Taints and Tolerations](https://kubernetes.io/docs/concepts/configuration/taint-and-toleration/)

给节点加Taints`kubectl taint nodes node1 key=value:NoSchedule`

给pod 加加tolerations
```yaml
tolerations:
- key: "key"
operator: "Equal"
value: "value"
effect: "NoSchedule"
```
NoSchedule 是一个effect. This means that no pod will be able to schedule onto node1 unless it has a matching toleration.


## 集群节点负载不均衡的问题

[被集群节点负载不均所困扰？TKE 重磅推出全链路调度解决方案](https://mp.weixin.qq.com/s/-USAfoI-8SDoR-LpFIrGCQ)

[大型Kubernetes集群的资源编排优化](https://mp.weixin.qq.com/s/lYAWxv_4emKv6uRP9eCvag)Kubernetes原生的调度器多是基于Pod Request的资源来进行调度的，没有根据Node当前和过去一段时间的真实负载情况进行相关调度的决策。这样就会导致一个问题在集群内有些节点的剩余可调度资源比较多但是真实负载却比较高，而另一些节点的剩余可调度资源比较少但是真实负载却比较低, 但是这时候Kube-scheduler会优先将Pod调度到剩余资源比较多的节点上（根据LeastRequestedPriority策略）。

![](/public/upload/kubernetes/dynamic_scheduler_node_annotator.png)

为了将Node的真实负载情况加到调度策略里，避免将Pod调度到高负载的Node上，同时保障集群中各Node的真实负载尽量均衡，腾讯扩展了Kube-scheduler实现了一个基于Node真实负载进行预选和优选的动态调度器（Dynamic-scheduler)。Dynamic-scheduler在调度的时候需要各Node上的负载数据，为了不阻塞动态调度器的调度这些负载数据，需要有模块定期去收集和记录。如下图所示node-annotator会定期收集各节点过去5分钟，1小时，24小时等相关负载数据并记录到Node的annotation里，这样Dynamic-scheduler在调度的时候只需要查看Node的annotation，便能很快获取该节点的历史负载数据。

![](/public/upload/kubernetes/dynamic_scheduler.png)

为了避免Pod调度到高负载的Node上，需要先通过预选把一些高负载的Node过滤掉，同时为了使集群各节点的负载尽量均衡，Dynamic-scheduler会根据Node负载数据进行打分, 负载越低打分越高。Dynamic-scheduler只能保证在调度的那个时刻会将Pod调度到低负载的Node上，但是随着业务的高峰期不同Pod在调度之后，这个Node可能会出现高负载。为了避免由于Node的高负载对业务产生影响，我们在Dynamic-scheduler之外还实现了一个Descheduler，它会监控Node的高负载情况，将一些配置了高负载迁移的Pod迁移到负载比较低的Node上。

![](/public/upload/kubernetes/descheduler.png)

[Kubernetes如何改变美团的云基础设施？](https://mp.weixin.qq.com/s/Df9fjmfTSD8MKg69LaySkQ)

[zhangshunping/dynamicScheduler](https://github.com/zhangshunping/dynamicScheduler)通过监控获取到各个k8s节点的使用率（cpu，mem等），当超过人为设定的阈值后，给对应节点打上status=presure，根据k8s软亲和性策略（nodeAffinity + preferredDuringSchedulingIgnoredDuringExecution），让pod尽量不调度到打上的presure标签的节点。

[kubernetes-sigs/descheduler](https://github.com/kubernetes-sigs/descheduler) 会监控Node的高负载情况，将一些配置了高负载迁移的Pod迁移到负载比较低的Node上。 PS：未细读

## 资源预留

[Kubernetes 资源预留配置](https://mp.weixin.qq.com/s/tirMYoC_o9ahRjiErc99AQ)考虑一个场景，由于某个应用无限制的使用节点的 CPU 资源，导致节点上 CPU 使用持续100%运行，而且压榨到了 kubelet 组件的 CPU 使用，这样就会导致 kubelet 和 apiserver 的心跳出问题，节点就会出现 Not Ready 状况了。默认情况下节点 Not Ready 过后，5分钟后会驱逐应用到其他节点，当这个应用跑到其他节点上的时候同样100%的使用 CPU，是不是也会把这个节点搞挂掉，同样的情况继续下去，也就导致了整个集群的雪崩。

```
$ kubectl describe node ydzs-node4
......
Capacity:
  cpu:                4
  ephemeral-storage:  17921Mi
  hugepages-2Mi:      0
  memory:             8008820Ki
  pods:               110
Allocatable:
  cpu:                4
  ephemeral-storage:  16912377419
  hugepages-2Mi:      0
  memory:             7906420Ki
  pods:               110
......
```

`allocatale = capacity - kube_reserved - system_reserved - eviction_hard`Node Capacity 是节点的所有硬件资源，kube-reserved 是给 kube 组件预留的资源，system-reserved 是给系统进程预留的资源，eviction-threshold 是 kubelet 驱逐的阈值设定，allocatable 才是真正调度器调度 Pod 时的参考值（保证节点上所有 Pods 的 request 资源不超过 Allocatable）。 

