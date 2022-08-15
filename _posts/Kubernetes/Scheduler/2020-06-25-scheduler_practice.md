---

layout: post
title: 调度实践
category: 架构
tags: Kubernetes
keywords:  Scheduler

---

## 简介

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

[Taints and Tolerations](https://kubernetes.io/docs/concepts/configuration/taint-and-toleration/) [Kubernetes污点和容忍度](https://mp.weixin.qq.com/s/XoBeoqJHZMrhCQN-JyOYMA)

给节点加Taints`kubectl taint nodes node1 key=value:NoSchedule`

给pod 加加tolerations
```yaml
tolerations:
- key: "key"
operator: "Equal"
value: "value"
effect: "NoSchedule"
```

effect 类型
2. NoSchedule, This means that no pod will be able to schedule onto node1 unless it has a matching toleration.
1. PreferNoSchedule, 软性 NoSchedule
3. NoExecute, 除了NoSchedule 的效果外，已经运行的pod 也会被驱逐。 

## 集群节点负载不均衡的问题

[被集群节点负载不均所困扰？TKE 重磅推出全链路调度解决方案](https://mp.weixin.qq.com/s/-USAfoI-8SDoR-LpFIrGCQ)

[大型Kubernetes集群的资源编排优化](https://mp.weixin.qq.com/s/lYAWxv_4emKv6uRP9eCvag)Kubernetes原生的调度器多是基于Pod Request的资源来进行调度的，没有根据Node当前和过去一段时间的真实负载情况进行相关调度的决策。这样就会导致一个问题在集群内有些节点的剩余可调度资源比较多但是真实负载却比较高，而另一些节点的剩余可调度资源比较少但是真实负载却比较低, 但是这时候Kube-scheduler会优先将Pod调度到剩余资源比较多的节点上（根据LeastRequestedPriority策略）。

![](/public/upload/kubernetes/dynamic_scheduler_node_annotator.png)

为了将Node的真实负载情况加到调度策略里，避免将Pod调度到高负载的Node上，同时保障集群中各Node的真实负载尽量均衡，腾讯扩展了Kube-scheduler实现了一个基于Node真实负载进行预选和优选的动态调度器（Dynamic-scheduler)。Dynamic-scheduler在调度的时候需要各Node上的负载数据，为了不阻塞动态调度器的调度这些负载数据，需要有模块定期去收集和记录。如下图所示node-annotator会定期收集各节点过去5分钟，1小时，24小时等相关负载数据并记录到Node的annotation里，这样Dynamic-scheduler在调度的时候只需要查看Node的annotation，便能很快获取该节点的历史负载数据。

![](/public/upload/kubernetes/dynamic_scheduler.png)

为了避免Pod调度到高负载的Node上，需要先通过预选把一些高负载的Node过滤掉，同时为了使集群各节点的负载尽量均衡，Dynamic-scheduler会根据Node负载数据进行打分, 负载越低打分越高。Dynamic-scheduler只能保证在调度的那个时刻会将Pod调度到低负载的Node上，但是随着业务的高峰期不同Pod在调度之后，这个Node可能会出现高负载。为了避免由于Node的高负载对业务产生影响，我们在Dynamic-scheduler之外还实现了一个Descheduler，它会监控Node的高负载情况，将一些配置了高负载迁移的Pod迁移到负载比较低的Node上。[K8s 中 Descheduler 默认策略不够用？扩展之](https://mp.weixin.qq.com/s/pkDxexvrzmtuLMWwzi0p_g)

![](/public/upload/kubernetes/descheduler.png)

[Kubernetes如何改变美团的云基础设施？](https://mp.weixin.qq.com/s/Df9fjmfTSD8MKg69LaySkQ)

[zhangshunping/dynamicScheduler](https://github.com/zhangshunping/dynamicScheduler)通过监控获取到各个k8s节点的使用率（cpu，mem等），当超过人为设定的阈值后，给对应节点打上status=presure，根据k8s软亲和性策略（nodeAffinity + preferredDuringSchedulingIgnoredDuringExecution），让pod尽量不调度到打上的presure标签的节点。

[kubernetes-sigs/descheduler](https://github.com/kubernetes-sigs/descheduler) 会监控Node的高负载情况，将一些配置了高负载迁移的Pod迁移到负载比较低的Node上。 PS：未细读


[作业帮 Kubernetes 原生调度器优化实践](https://mp.weixin.qq.com/s/ULFfmxFH_Y7QOCsySaNUQw)实时调度器，在调度的时候获取各节点实时数据来参与节点打分，但是实际上实时调度在很多场景并不适用，尤其是对于具备明显规律性的业务来说，比如我们大部分服务晚高峰流量是平时流量的几十倍，高低峰资源使用差距巨大，而业务发版一般选择低峰发版，采用实时调度器，往往发版的时候比较均衡，到晚高峰就出现节点间巨大差异，很多实时调度器往往在出现巨大差异的时候会使用再平衡策略来重新调度，高峰时段对服务 POD 进行迁移，服务高可用角度来考虑是不现实的。显然，实时调度是远远无法满足业务场景的。针对这种情况，需要预测性调度方案，根据以往高峰时候 CPU、IO、网络、日志等资源的使用量，通过对服务在节点上进行最优排列组合回归测算，得到各个服务和资源的权重系数，基于资源的权重打分扩展，也就是使用过去高峰数据来预测未来高峰节点服务使用量，从而干预调度节点打分结果。

[如何提高 Kubernetes 集群资源利用率？](https://mp.weixin.qq.com/s/xJTmnba0Ac14p4V4xXbQQw)原生调度器并不感知真实资源的使用情况，需要引入动态资源视图。但会产生一个借用的代价：不稳定的生命周期。因此有两个核心问题要解决：第一，动态资源视图要如何做；第二个单机资源的调配如何保证供给。

## 节点资源预留

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


## “贪心”的开发者 和超卖

《大数据经典论文解读》大部分情况下，开发者并不会去仔细测试自己的程序到底会使用多少资源，很容易作出“拍脑袋”的决策。而且，一般来说，开发者都会偏向于高估自己所需要使用的资源，这样至少不会出现程序运行出问题的情况。但是，我们使用 Borg 的目的，就是尽量让机器的使用率高一点。每个开发者都给自己留点 Buffer，那我们集群的利用率怎么高得起来呢？所以，面对贪心的都会多给自己申请一点资源的开发者，Borg 是通过这样两个方式，来提升机器的使用率。
1. 对资源进行“超卖”。也就是我明明只有 64GB 的内存，但是我允许同时有申明了 80GB 的任务在 Borg 里运行。当然，为了保障所有生产类型的任务一定能够正常运行，Borg 并不会对它们进行超卖。但是，对于非生产类型的任务，比如离线的数据批处理任务，超卖是没有问题的。大不了，其中有些任务在资源不足的时候，会被挂起，或者调度到其他机器上重新运行，任务完成的时间需要久一点而已。
2. 对资源进行动态的“回收”。我们所有的生产的 Task，肯定也没有利用满它们所申请的资源。所以，Borg 实际不会为这些 Task 始终预留这么多资源。Borg 会在 Task 开始的时候，先为它分配它所申请的所有资源。然后，在 Task 成功启动 5 分钟之后，它会慢慢减少给 Task 分配的资源，直到最后变成 Task 当前实际使用的资源，以及 Borg 预留的一些 Buffer（削减的资源称为**回收资源**）。当然，Task 使用的资源可能是动态变化的。比如一个服务是用来处理图片的，平时都是处理的小图片，内存使用很小，忽然来了一张大图片，那么它使用的内存一下子需要大增。这个时候，Borg 会迅速把 Task 分配的资源，增加到它所申请的资源数量。**对于回收资源，Borg 只会分配给非生产类型的任务**。因为，这部分资源的使用是没有保障的，随时可能因为被回收了资源的生产类型 Task，忽然需要资源，被动态地抢回去。如果我们把这部分资源分配给其他生产类型的 Task，那么就会面临两个生产类型的 Task 抢占资源的问题。

[Crane-scheduler：基于真实负载进行调度](https://mp.weixin.qq.com/s/s0usEAA3pFemER97HS5G-Q)Crane-scheduler是一个基于scheduler framework 实现的完整调度器。

[Trimaran: 基于实际负载的K8s调度插件](https://mp.weixin.qq.com/s/I1aJfHIt_frZE9xS59QuPw)

## 拓扑感知

[Kubernetes 资源拓扑感知调度优化](https://mp.weixin.qq.com/s/CgW1zqfQBdUQo8qDtV-57Q)

## 美团

[提升资源利用率与保障服务质量，鱼与熊掌不可兼得？](https://mp.weixin.qq.com/s/hQKM9beWcx7CKMvpJxznfQ)LAR全称是集群负载自动均衡管理系统（LAR，Load Auto-Regulator）

按照很多同学的理解，通过非常简单的操作即可达成这个目标——提高单机的服务部署密度。但如此简单的操作，为何全球数据中心资源利用率仅为10%~20%呢？利用率如此之低，这里最为关键的因素有三个：
1. 部署到同一台物理机的服务在资源使用上存在相互干扰。
2. 服务在流量上存在高低峰，反映在资源使用上也有高低峰。
3. 关键核心在线服务的服务质量下降无法接受。

传统的方案通过节点资源超售来解决资源申请和实际资源使用之间存在的Gap，并引入根据负载的动态调度策略。
1. 调整节点资源超售，虽然能在一定程度上缓解资源申请和使用的Gap问题，但由于Gap在不同的服务间并不相同，加上服务资源使用的波峰波谷分布集中的情况（美团在线业务的典型特征），此方法在整体上过于粗放，会导致节点间的负载分布不均衡，部分节点负载很高，影响服务质量；另一部分节点负载极低，实际上形成资源浪费。
2. 而根据负载直接进行资源调度，由于负载是动态变化的，在调度算法设计及计算框架实现上会非常复杂，且效果一般。

提升资源利用率的本质是提升资源共享复用水平，而保障服务质量则需要通过资源隔离能力，保障服务的性能稳定。针对上述两个根本点，LAR在Kubernetes上提出两个核心创新点：
1. 资源池化分级
  1. 通过将单机资源划分到不同的资源池，提升资源在池内的共享复用水平。
  2. 不同的资源池之间有不同的优先级，并提供不同的资源隔离水平（资源隔离水平越高，资源共享复用水平越低）。
  3. 资源在不同优先级的资源池之间根据优先级和资源池的资源负载水平流动，优先保障高优资源池服务的资源使用，从而保障其服务质量。

2. 动态负载和静态资源映射
  1. 资源的分配，本质上是负载空间的分配。假设单机整体CPU利用率小于50%的情况下，运营在其上的服务的服务质量不会有影响，那么这个机器的静态资源其实对应的就是节点50% CPU利用率的负载空间。换个角度看，就是无论如何调度分配资源，只要这个节点的负载不超过50%即可。
  2. 业务静态的资源申请，根据服务的特征经过调度计算后，服务被放入对应的资源池，而资源池的资源配置则根据池内所有服务的实际负载进行资源配置，并可以实时地根据负载调整资源配置，实现静态资源分配和动态负载的映射管理。

通过池间资源隔离达到池间服务的干扰隔离。资源池内资源的配置**依据服务的负载进行动态调整**，并通过资源配置的调整，控制资源池内部的资源负载维系在相对稳定的范围内，从而保证服务质量。

以3级资源池为例，节点资源被划分为0、1、2三类资源池，优先级依次降低。初始整个机器无服务调度其上，资源全部集中在Pool2。随着服务的调度，Pool1先调度了服务1，这时会根据上述的资源计算方式，LAR将Pool2的对应的资源调整至Poo1，Pool2资源减少。随着Pool1中服务增多，配置的资源随之增多，Pool2相应资源减少。优先级最高的Pool0调入服务后，同样的资源从Pool2调整至Pool0；Pool2调度入服务时，Pool2资源不变。
3个资源池配置不同的资源配置管理策略，0号池优先级最高，池内目标CPU负载控制在30%～50%之间；1号池优先级次之，池内目标CPU负载控制在45%～60%之间；2号池优先级最低，池内目标CPU负载控制在50%～80%。已分配的资源由资源池内服务共享，在池间相互隔离。在负载低时，不同资源池根据资源池管理策略，自动调整各资源池的资源配置，保证资源池内负载稳定；出现资源紧张时，高优资源池可以从低优资源池抢占资源，优先保障高优服务的资源需求。

池内分配资源会随着负载进行变化，引起池间的资源流动。池间资源流动遵循以下规则：
1. 所有资源池的池内分配资源之和为节点可分配的资源总量。
2. 当池内负载降低，释放资源到最低等级的资源池，复用闲时资源。
3. 当池内负载升高，向等级低于自身的资源池，根据从低到高的顺序进行资源请求，根据优先级满足服务资源需求。
4. 池内的资源最多不会超过用户申请的量。

以3级资源池为例：
1. 当Pool1负载升高时，从等级更低的Pool2抢占资源，优先保障自身的服务资源需求，Pool1负载降低时，将冗余的资源释放回Pool2。
2. 当Pool0负载升高时，优先从Pool2抢占资源，当Pool2资源不足时，从Pool1抢占资源，保证更高等级的服务资源需求，当Pool0负载降低时，冗余的资源被释放回Pool2，此时3. 若Pool1存在负载压力，则会重新从Pool2抢占资源。

QoS服务质量保障机制，为提升资源利用率会导致资源竞争，LAR通过池间、池内两层QoS服务质量保障机制，分级保证服务的隔离性和稳定性。
1. 池间多维度资源隔离，LAR对资源池进行了多维度的资源隔离与限制。除了基础资源（CPU、Memory），还对磁盘I/O、CPU调度、Memory Cache、内存带宽、L3 Cache、OOM Score、网络带宽等更细粒度的资源进行了隔离，进一步提升不同等级服务间的隔离性，保证服务不会受到其他资源池的影响。PS：由MTOS 的相关特性支持
2. 池内多层级保障策略，当资源池内负载出现不符合预期的情况时（如容器负载异常），由于资源池内资源共享，整个资源池的服务都可能受到影响。LAR基于资源池内不同的负载等级，制定了多级保障策略。QoSAdaptor周期性（秒级）地获取节点负载的数据，并计算资源池的负载等级。当负载达到一定的资源等级时，执行对应的负载策略。通过CPU降级、驱逐等行为，根据优先级对部分容器进行资源降级，保障池内绝大多数容器的稳定性。
  1. 容器驱逐：当池内Memory使用接近Cgroup限制，避免整个资源池出现OOM，影响所有容器的正常运行，会结合优先级筛选Memory使用较多的容器进行驱逐操作。PS：Kubernetes原生的驱逐策略基于整个节点的负载，LAR中将策略缩小到了资源池维度
  2. CPU降级：池内CPU负载超过一定负载等级，避免高负载导致的容器间互相影响，LAR会结合优先级筛选CPU使用较多的容器，对其CPU使用进行单独的限制。降级操作存在定时检查机制，当负载恢复正常，或有资源可以抢占的情况下，会将CPU限制进行恢复。
  3. 强制抢占：从更低等级的资源池抢占资源，与普通资源抢占的区别为，即使资源已经被其他池使用，强制抢占会优先满足高等级资源池的需求。

LAR基于资源池的历史负载与历史分配情况，对池内高峰资源使用情况进行预测，为节点资源调整提供指导。由于资源池负载变化比较频繁，同时受到池内服务变更、资源总量、高低峰时间区间等因素的影响，节点基于实时负载进行池内资源的变更较不稳定。Recommender周期性地根据各节点资源池的历史负载与分配情况进行高峰资源预测，并下发到节点，提供高峰负载控制指导，提升资源池资源保障的稳定性。同时通过RCF完成动态负载和静态资源的转换，在调度层屏蔽了动态负载变化，减少负载频繁变化对调度准确性的影响。

LAR的设计目标是在保障服务质量的同时提升整体资源的利用率，在资源分池分级的设计上，针对通用的在线服务进行服务分级，对接不同资源池，提供不同的服务质量保障，从而提升资源的利用率。而对于离线服务，本身相对于在线服务的服务质量要求低，故而LAR天然地适用于混部场景。PS：**就是抛开在离线的概念，独立搞出一个带有优先级的资源池的概念**
1. 对于在线服务，通过对服务进行分级，并通过服务画像对服务进行细致刻画，将资源敏感型服务和关键核心服务部署到LAR优先级最高的资源池中
2. 而对于一般的在线服务，部署在次优先级资源池。
3. 在混部场景中，假设将资源池分为0、1、2三个级别，优先级依次由高到低。0和1号池分别对应核心关键在线服务和一般的在线服务，而2号池对应离线服务使用的资源池。

一方面我们对高优资源池配置更强的资源隔离策略（比如CPU绑核、进程优先调度等），另一方面高优池资源利用率控制在一个安全较低的水位；而低优池，则相对在一个更高的水平。LAR的资源动态调整保障负载能力，会自动将0号池与1号池在业务低峰期（负载低）的闲置资源回收，提供给2号池的离线服务使用。并且QoS服务质量保障机制，可以确保在业务高峰来临时，秒级抢占2号池资源（对于内存等非复用型资源，通过驱逐方式强制回收），从而保障在线服务的资源使用。