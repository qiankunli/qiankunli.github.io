---

layout: post
title: 多集群及clusternet学习
category: 架构
tags: Kubernetes
keywords:  集群

---

## 简介

* TOC
{:toc}


## 多集群

[浅析 Kubernetes 多集群方案](https://mp.weixin.qq.com/s/1ZvqFRYd7cl8-lE_wVDuNA)为什么要有多集群调度？
1. 单集群的容量限制：单集群的最大节点数不是一个确定值，其受到集群的部署方法和业务使用集群资源的方式的影响。在官方文档中的集群注意事项里提到单集群 5000 个节点上限，我们可以理解为推荐最大节点数。
2. 多租户：因为容器没法做到完美的隔离，不同租户可以通过不同宿主机或者不同集群分割开。对企业内部也是如此，业务线就是租户。不同业务线对于资源的敏感程度也不同，企业也会将机器资源划分为不同集群给不同业务线用。
3. 云爆发：云爆发是一种部署模式，通过公有云资源来满足应用高峰时段的资源需求。正常流量时期，业务应用部署在有限资源的集群里。当资源需求量增加，通过扩容到公有云来消减高峰压力。
4. 高可用：单集群能够做到容器级别的容错，当容器异常或者无响应时，应用的副本能够较快地另一节点上重建。但是单集群无法应对网络故障或者数据中心故障导致的服务的不可用。跨地域的多集群架构能够达到跨机房、跨区域的容灾。
5. 地域亲和性：尽管国内互联网一直在提速，但是处于带宽成本的考量，同一调用链的服务网络距离越近越好。服务的主调和被调部署在同一个地域内能够有效减少带宽成本；并且分而治之的方式让应用服务本区域的业务，也能有效缓解应用服务的压力。

多集群的主要攻克的难点就是跨集群的信息同步和跨集群网络连通方式。
1. 跨集群网络连通方式一般的处理方式就是确保不同机房的网络相互可达，这也是最简单的方式。
2. 跨集群的信息同步。多集群的服务实例调度，需要保证在多集群的资源同步的实时，将 pod 调度不同的集群中不会 pod pending 的情况。
  1. 定义专属的 API server：通过一套统一的中心化 API 来管理多集群以及机器资源。KubeFed 就是采用的这种方法，通过扩展 k8s API 对象来管理应用在跨集群的分布。
  2. 基于 Virtual Kubelet：Virtual Kubelet 本质上是允许我们冒充 Kubelet 的行为来管理 virtual node 的机制。这个 virtual node 的背后可以是任何物件，只要 virtual  node 能够做到上报 node 状态、和 pod 的生命周期管理。

多集群的服务实例调度，需要保证在多集群的资源同步的实时，将 pod 调度不同的集群中不会 pod pending 的情况。控制平面的跨集群同步主要有两类方式：
1. 定义专属的 API server：通过一套统一的中心化 API 来管理多集群以及机器资源。
2. 基于 Virtual Kubelet：Virtual Kubelet 本质上是允许我们冒充 Kubelet 的行为来管理 virtual node 的机制。这个 virtual node 的背后可以是任何物件，只要 virtual  node 能够做到上报 node 状态、和 pod 的生命周期管理。PS: 这导致后续干预多集群调度比较难，因为进行任何改动都不能超越kubelet 所提供的能力。

要解决的几个问题（未完成）
1. 应用分发模型。即用户创建的Deployment 等object最终落在哪个集群中，如何表达这个诉求？是优先落在一个集群，还是各个集群都摊一点。

## clusternet 

以创建Deployment 为例，请求url 前缀改为clusternet 的shadow api， 由clusternet 的agg apiserver 处理，agg 拿到deployment 后保存在了自定义 Manifest下（类似下图将 Namespace 对象挂在 Manifest 下），之后调度器根据 用户创建的Subscription 配置策略 决定应用分发（比如一个deployment 6个replicas，是spead 到2个集群，即按集群的空闲资源占比分配，还是binpack 到一个集群上）。hub 根据调度结果，拿到deployment 数据（底层涉及到 base 和 Description 等obj），通过目标cluster 对应的dynamic client在子集群真正的创建 deployment。

![](/public/upload/kubernetes/clusternet_object.png)


### 源码分析

```
clusternet
  /cmd
    /clusternet-agent
    /clusternet-hub
    /clusternet-scheduler
  /pkg
    /agent       
    /apis        
    /controllers    // 不是单独运行的 k8s Controller ，只是封装了 infromer 各种crd 的逻辑，真正处理crd的处理逻辑 在hub 和agent 中
      /apps
        /feedinventory

    /hub         
    /predictor   
    /scheduler   
```

1. hub 负责提供agg server 并维护 用户提交的obj 与内部的对象的转换，最终将用户提交的obj 分发到各个cluster 上。 
   1. 自定义 REST 实现了 apiserver的 Creater 等接口 `clusternet/pkg/registry/shadow/template/rest.go`，由 InstallShadowAPIGroups 注入到agg 处理逻辑中。将 shadow api  传来的 deployment 改为创建Manifest。 
   2. feedinventoryController.handleSubscription 将 deployment 需要的资源计算出来 写入到FeedInventory
2. Scheduler 监听Subscription
   ```go
   clusternet/pkg/scheduler/algorithm/generic.go
   Scheduler.Run ==> scheduleOne ==> scheduleAlgorithm.Schedule
    // Step 1: Filter clusters.
    feasibleClusters, diagnosis, err := g.findClustersThatFitSubscription(ctx, fwk, state, sub)
    // Step 2: Predict max available replicas if necessary.
    availableList, err := predictReplicas(ctx, fwk, state, sub, finv, feasibleClusters)
    // Step 3: Prioritize clusters.
    priorityList, err := prioritizeClusters(ctx, fwk, state, sub, feasibleClusters, availableList)
    // selectClusters takes a prioritized list of clusters and then picks a fraction of clusters in a reservoir sampling manner from the clusters that had the highest score.
    clusters, err := g.selectClusters(ctx, state, priorityList, fwk, sub, finv)
    最终将 targetClusters 写入到 subscription.status.bindingClusters 中
   ```
2. Hub， clusternet 会给每个cluster 建一个namespace，分配到 该cluster 下的obj 都会有对应的 base 和 Description 对象，也在该namespace 下。cluster-xx namespace 对象本身label 会携带一些cluster的信息。
   1. Deployer.handleSubscription（此时调度结果 已经写入了subscription.status.bindingClusters） ==> populateBasesAndLocalizations 。为每一个cluster 创建 Subscription 对应的 Base 和 Localization
   2. Deployer.handleManifest ==> resyncBase ==>  populateDescriptions ==> syncDescriptions。clusterId 会从 Subscription 一路传到 Description 中
   3. Deployer.handleDescription ==> createOrUpdateDescription ==> deployer.applyResourceWithRetry。拿到对应cluster的dynamicClient （根据clusterId） 在cluster 中创建deployment 对象。 


```yaml
# examples/dynamic-dividing-scheduling/subscription.yaml
apiVersion: apps.clusternet.io/v1alpha1
kind: Subscription
metadata:
  name: dynamic-dividing-scheduling-demo
  namespace: default
spec:
  subscribers: # filter out a set of desired clusters
    - clusterAffinity:
        matchExpressions:
          - key: clusters.clusternet.io/cluster-id
            operator: Exists
  schedulingStrategy: Dividing
  dividingScheduling:
    type: Dynamic
    dynamicDividing:
      strategy: Spread # currently we only support Spread dividing strategy
  feeds: # defines all the resources to be deployed with
    - apiVersion: v1
      kind: Namespace
      name: qux
    - apiVersion: v1
      kind: Service
      name: my-nginx-svc
      namespace: qux
    - apiVersion: apps/v1 # with a total of 6 replicas
      kind: Deployment
      name: my-nginx
      namespace: qux
status：
  bindingClusters:
  - clusternet-v7wzq/clusternet-cluster-bb2xp 
  - clusternet-wlf5b/clusternet-cluster-skxd4
  - clusternet-bbf20/clusternet-cluster-aqx3b
  desiredReleases: 6
  replicas:
    apps/v1/Deployment/qux/my-nginx:    # 这里会生成一个 Localization  在cluster 范围更新deployment 的replicas
    - 1
    - 2
    - 3
    v1/Namespace/qux: []
    v1/Service/qux/my-nginx-svc: []
```

scheduler 如何为 deployment 选择cluster ？ 
1. hub，  clusternet  里有一个  PluginFactory 定义了如何从 deployment 中parse deployment 需要的资源，feedinventoryController.handleSubscription 将 deployment 需要的资源计算出来 写入到FeedInventory
2. scheduler 拿到 deployment name 对应的 FeedInventory，在predict 时 向agent 发出http 请求 `/replicas/predict`，agent 返回自己cluster 针对deployment 最多可以运行几个replicas
  ```
  predictReplicas(ctx, fwk, state, sub, finv, feasibleClusters)
    prePredictStatus := fwk.RunPrePredictPlugins(ctx, state, sub, finv, clusters)
      func (pl *Predictor) Predict(_ context.Context, ...)
        replica, err2 := predictMaxAcceptableReplicas(httpClient, predictorAddress, feedOrder.ReplicaRequirements)
          发出http 请求 /replicas/predict
    availableList, predictStatus := fwk.RunPredictPlugins(ctx, state, sub, finv, clusters, availableList)
  ```

## 文章汇总

[Kubernetes 多集群项目介绍](https://mp.weixin.qq.com/s/laMfFgre8PrbC2SayxBFRQ)
阿里：
[还在为多集群管理烦恼吗？OCM来啦！](https://mp.weixin.qq.com/s/t1AGv3E7Q00N7LmHLbdZyA)
[CNCF 沙箱项目 OCM Placement 多集群调度指南](https://mp.weixin.qq.com/s/_k2MV4b3hfTrLUCCOKOG8g)
腾讯：
[Clusternet - 新一代开源多集群管理与应用治理项目](https://mp.weixin.qq.com/s/4kBmo9v35pXz9ooixNrXdQ)
[Clusternet v0.5.0 重磅发布： 全面解决多集群应用分发的差异化配置难题](https://mp.weixin.qq.com/s/fcLN4w_Qu8IAm2unk4B_rg)
其它：
[关于多集群Kubernetes的一些思考](https://mp.weixin.qq.com/s/haBM1BSDWLhRYBJH4cJHvA)
[多云环境下的资源调度：Karmada scheduler的框架和实现](https://mp.weixin.qq.com/s/RvnEMpK7l9bqbQCrbPqBPQ)

[vivo大规模 Kubernetes 集群自动化运维实践](https://mp.weixin.qq.com/s/L9z1xLXUnz52etw2jDkDkw) 未读。
