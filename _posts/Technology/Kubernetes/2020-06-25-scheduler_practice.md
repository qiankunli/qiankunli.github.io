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


