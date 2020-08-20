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

[大型Kubernetes集群的资源编排优化](https://mp.weixin.qq.com/s/lYAWxv_4emKv6uRP9eCvag)Kubernetes原生的调度器多是基于Pod Request的资源来进行调度的，没有根据Node当前和过去一段时间的真实负载情况进行相关调度的决策。这样就会导致一个问题在集群内有些节点的剩余可调度资源比较多但是真实负载却比较高，而另一些节点的剩余可调度资源比较少但是真实负载却比较低, 但是这时候Kube-scheduler会优先将Pod调度到剩余资源比较多的节点上（根据LeastRequestedPriority策略）。

![](/public/upload/kubernetes/dynamic_scheduler_node_annotator.png)

为了将Node的真实负载情况加到调度策略里，避免将Pod调度到高负载的Node上，同时保障集群中各Node的真实负载尽量均衡，腾讯扩展了Kube-scheduler实现了一个基于Node真实负载进行预选和优选的动态调度器（Dynamic-scheduler)。Dynamic-scheduler在调度的时候需要各Node上的负载数据，为了不阻塞动态调度器的调度这些负载数据，需要有模块定期去收集和记录。如下图所示node-annotator会定期收集各节点过去5分钟，1小时，24小时等相关负载数据并记录到Node的annotation里，这样Dynamic-scheduler在调度的时候只需要查看Node的annotation，便能很快获取该节点的历史负载数据。

![](/public/upload/kubernetes/dynamic_scheduler.png)

为了避免Pod调度到高负载的Node上，需要先通过预选把一些高负载的Node过滤掉，同时为了使集群各节点的负载尽量均衡，Dynamic-scheduler会根据Node负载数据进行打分, 负载越低打分越高。Dynamic-scheduler只能保证在调度的那个时刻会将Pod调度到低负载的Node上，但是随着业务的高峰期不同Pod在调度之后，这个Node可能会出现高负载。为了避免由于Node的高负载对业务产生影响，我们在Dynamic-scheduler之外还实现了一个Descheduler，它会监控Node的高负载情况，将一些配置了高负载迁移的Pod迁移到负载比较低的Node上。

![](/public/upload/kubernetes/descheduler.png)