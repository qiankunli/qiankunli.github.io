---

layout: post
title: Kubernetes资源调度——scheduler
category: 技术
tags: Kubernetes
keywords: kubernetes scheduler

---

## 简介

* TOC
{:toc}


几个主要点

2. Kubernetes 调度本身的演化 [从 Kubernetes 资源控制到开放应用模型，控制器的进化之旅](https://mp.weixin.qq.com/s/AZhyux2PMYpNmWGhZnmI1g)
3. 调度器的原理， [How Kubernetes Initializers work](https://ahmet.im/blog/initializers/)
the scheduler is yet another controller, watching for Pods to show up in the API server and assigns each of them to a Node [How does the Kubernetes scheduler work?](https://jvns.ca/blog/2017/07/27/how-does-the-kubernetes-scheduler-work/).

    while True:
        pods = queue.getPod()
        assignNode(pod)

**scheduler is not responsible for actually running the pod – that’s the kubelet’s job. So it basically just needs to make sure every pod has a node assigned to it.**

## Kubernetes 资源模型与资源管理

在 Kubernetes 里，Pod 是最小的原子调度单位。这也就意味着，所有跟调度和资源管理相关的属性都应该是属于 Pod 对象的字段。而这其中最重要的部分，就是 Pod 的CPU 和内存配置

在 Kubernetes 中，像 CPU 这样的资源被称作“可压缩资源”（compressible resources）。它的典型特点是，当可压缩资源不足时，Pod 只会“饥饿”，但不会退出。

而像内存这样的资源，则被称作“不可压缩资源（incompressible resources）。当不可压缩资源不足时，Pod 就会因为 OOM（Out-Of-Memory）被内核杀掉。

Kubernetes 里 Pod 的 CPU 和内存资源，实际上还要分为 limits 和 requests 两种情况：在调度的时候，kube-scheduler 只会按照 requests 的值进行计算。而在真正设置 Cgroups 限制的时候，kubelet 则会按照 limits 的值来进行设置。这个理念基于一种假设：容器化作业在提交时所设置的资源边界，并不一定是调度系统所必须严格遵守的，这是因为在实际场景中，大多数作业使用到的资源其实远小于它所请求的资源限额。

QoS 划分的主要应用场景，是当宿主机资源（主要是不可压缩资源）紧张的时候，kubelet 对 Pod 进行 Eviction（即资源回收）时需要用到的。“紧张”程度可以作为kubelet 启动参数配置，默认为

	memory.available<100Mi
	nodefs.available<10%
	nodefs.inodesFree<5%
	imagefs.available<15%

Kubernetes 计算 Eviction 阈值的数据来源，主要依赖于从 Cgroups 读取到的值，以及使用 cAdvisor 监控到的数据。当宿主机的 Eviction 阈值达到后，就会进入 MemoryPressure 或者 DiskPressure 状态，从而避免新的 Pod 被调度到这台宿主机上。

limit 不设定，默认值由 LimitRange object确定

|limits|requests||Qos模型|
|---|---|---|---|
|有|有|两者相等|Guaranteed|
|有|无|默认两者相等|Guaranteed|
|x|有|两者不相等|Burstable|
|无|无||BestEffort|

而当 Eviction 发生的时候，kubelet 具体会挑选哪些 Pod 进行删除操作，就需要参考这些 Pod 的 QoS 类别了。PS：怎么有一种 缓存 evit 的感觉。limit 越“模糊”，物理机MemoryPressure/DiskPressure 时，越容易优先被干掉。

DaemonSet 的 Pod 都设置为 Guaranteed 的 QoS 类型。否则，一旦 DaemonSet 的 Pod 被回收，它又会立即在原宿主机上被重建出来，这就使得前面资源回收的动作，完全没有意义了。

## Kubernetes 基于资源的调度

在 Kubernetes 项目中，默认调度器的主要职责，就是为一个新创建出来的 Pod，寻找一个最合适的节点（Node）而这里“最合适”的含义，包括两层： 

1. 从集群所有的节点中，根据调度算法挑选出所有可以运行该 Pod 的节点；
2. 从第一步的结果中，再根据调度算法挑选一个最符合条件的节点作为最终结果。

所以在具体的调度流程中，默认调度器会首先调用一组叫作 Predicate 的调度算法，来检查每个 Node。然后，再调用一组叫作 Priority 的调度算法，来给上一步得到的结果里的每个 Node 打分。最终的调度结果，就是得分最高的那个Node。

除了Pod，Scheduler 需要调度其它对象么？不需要。因为Kubernetes 对象虽多，**但只有Pod 是调度对象**，其它对象包括数据对象（比如PVC等）、编排对象（Deployment）、Pod辅助对象（NetworkPolicy等） 都只是影响Pod的调度，本身不直接消耗计算和内存资源。

**调度器对一个 Pod 调度成功，实际上就是将它的 spec.nodeName 字段填上调度结果的节点名字**。 这在k8s 的很多地方都要体现，k8s 不仅将对容器的操作“标准化” ==> “配置化”，一些配置是用户决定的，另一个些是系统决定的

调度主要包括两个部分

1. 组件交互，包括如何与api server交互感知pod 变化，如何感知node 节点的cpu、内存等参数。PS：任何调度系统都有这个问题。
2. 调度算法，上文的Predicate和Priority 算法

调度这个事情，在不同的公司和团队里的实际需求一定是大相径庭的。上游社区不可能提供一个大而全的方案出来。所以，将默认调度器插件化是 kube-scheduler 的演进方向。

### 谓词和优先级算法

[调度系统设计精要](https://mp.weixin.qq.com/s/R3BZpYJrBPBI0DwbJYB0YA)

![](/public/upload/kubernetes/predicates_priorities_schedule.png)

我们假设调度器中存在一个谓词算法和一个 Map-Reduce 优先级算法，当我们为一个 Pod 在 6 个节点中选择最合适的一个时，6 个节点会先经过谓词的筛选，图中的谓词算法会过滤掉一半的节点，剩余的 3 个节点经过 Map 和 Reduce 两个过程分别得到了 5、10 和 5 分，最终调度器就会选择分数最高的 4 号节点。

### Predicate

1. GeneralPredicates
	1. PodFitsResources，检查的只是 Pod 的 requests 字段
	2. PodFitsHost，宿主机的名字是否跟 Pod 的 spec.nodeName 一致。
	3. PodFitsHostPorts，Pod 申请的宿主机端口（spec.nodePort）是不是跟已经被使用的端口有冲突。
	4. PodMatchNodeSelector，Pod 的 nodeSelector 或者 nodeAffinity 指定的节点，是否与待考察节点匹配

2. 与 Volume 相关的过滤规则
3. 是宿主机相关的过滤规则
4. Pod 相关的过滤规则。比较特殊的，是 PodAffinityPredicate。这个规则的作用，是检查待调度 Pod 与 Node 上的已有 Pod 之间的亲密（affinity）和反亲密（anti-affinity）关系

在具体执行的时候， 当开始调度一个 Pod 时，Kubernetes 调度器会同时启动 16 个 Goroutine，来并发地为集群里的所有 Node 计算 Predicates，最后返回可以运行这个 Pod 的宿主机列表。

### Priorities

在 Predicates 阶段完成了节点的“过滤”之后，Priorities 阶段的工作就是为这些节点打分。这里打分的范围是 0-10 分，得分最高的节点就是最后被 Pod 绑定的最佳节点。

1. LeastRequestedPriority + BalancedResourceAllocation

	1. LeastRequestedPriority计算方法`score = (cpu((capacity-sum(requested))10/capacity) + memory((capacity-sum(requested))10/capacity))/2` 实际上就是在选择空闲资源（CPU 和 Memory）最多的物理机
	2. BalancedResourceAllocation，计算方法`score = 10 - variance(cpuFraction,memoryFraction,volumeFraction)*10` 每种资源的 Fraction 的定义是 ：Pod 请求的资源/ 节点上的可用资源。而 variance 算法的作用，则是资源 Fraction 差距最小的节点。BalancedResourceAllocation 选择的，其实是调度完成后，所有节点里各种资源分配最均衡的那个节点，从而避免一个节点上 CPU 被大量分配、而 Memory 大量剩余的情况。
2. NodeAffinityPriority
2. TaintTolerationPriority
3. InterPodAffinityPriority
4. ImageLocalityPriority

### 优先级和抢占

优先级和抢占机制，解决的是 （高优先级的）Pod 调度失败时该怎么办的问题

	apiVersion: v1
	kind: Pod
	metadata:
	name: nginx
	labels:
		env: test
	spec:
	containers:
	- name: nginx
		image: nginx
		imagePullPolicy: IfNotPresent
	priorityClassName: high-priority


Pod 通过 priorityClassName 字段，声明了要使用名叫 high-priority 的 PriorityClass。当这个 Pod 被提交给 Kubernetes 之后，Kubernetes 的 PriorityAdmissionController 就会自动将这个 Pod 的 spec.priority 字段设置为 PriorityClass 对应的value 值。

如果确定抢占可以发生，那么调度器就会把自己缓存的所有节点信息复制一份，然后使用这个副本来模拟抢占过程。

1. 找到牺牲者，判断抢占者是否可以部署在牺牲者所在的Node上
2. 真正开始抢占

	1. 调度器会检查牺牲者列表，清理这些 Pod 所携带的 nominatedNodeName 字段。
	2. 调度器会把抢占者的 nominatedNodeName，设置为被抢占的 Node 的名字。
	3. 调度器会开启一个 Goroutine，同步地删除牺牲者。
3. 调度器就会通过正常的调度流程把抢占者调度成功

## 基于调度框架

明确了 Kubernetes 中的各个调度阶段，提供了设计良好的基于插件的接口。调度框架认为 Kubernetes 中目前存在调度（Scheduling）和绑定（Binding）两个循环：

1. 调度循环在多个 Node 中为 Pod 选择最合适的 Node；
2. 绑定循环将调度决策应用到集群中，包括绑定 Pod 和 Node、绑定持久存储等工作；

除了两个大循环之外，调度框架中还包含 QueueSort、PreFilter、Filter、PostFilter、Score、Reserve、Permit、PreBind、Bind、PostBind 和 Unreserve 11 个扩展点（Extension Point），这些扩展点会在调度的过程中触发，它们的运行顺序如下：

![](/public/upload/kubernetes/framework_schedule.png)

插件规范定义在 `$GOPATH/src/k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1/interface.go` 中，各类插件继承 Plugin

```go
type Plugin interface {
	Name() string
}
type PreFilterPlugin interface {
	Plugin
	// PreFilter is called at the beginning of the scheduling cycle. All PreFilter plugins must return success or the pod will be rejected.
	PreFilter(ctx context.Context, state *CycleState, p *v1.Pod) *Status
	// PreFilterExtensions returns a PreFilterExtensions interface if the plugin implements one,or nil if it does not. A Pre-filter plugin can provide extensions to incrementally modify its pre-processed info. The framework guarantees that the extensions
	// AddPod/RemovePod will only be called after PreFilter, possibly on a cloned CycleState, and may call those functions more than once before calling Filter again on a specific node.
	PreFilterExtensions() PreFilterExtensions
}
```

## 代码分析

![](/public/upload/kubernetes/scheduler_overview.png)

[How does the Kubernetes scheduler work?](https://jvns.ca/blog/2017/07/27/how-does-the-kubernetes-scheduler-work/) 

![](/public/upload/kubernetes/scheduler_object.png)

![](/public/upload/kubernetes/scheduler_sequence.png)

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


