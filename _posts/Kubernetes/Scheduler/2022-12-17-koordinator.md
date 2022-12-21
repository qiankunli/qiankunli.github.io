---

layout: post
title: 多类型负载协调员Koordinator
category: 架构
tags: Kubernetes
keywords:  Kubernetes 混部

---

## 简介

* TOC
{:toc}

问题：是不是可以认为，用户设置的request 和limit ，除了最开始作为一个参考值之外，其它时候是没啥用的，全靠pod 真实的资源使用和 node 真实负载 根据priority 和 qos 进行调度了。 若不是为了最开始 有个参考作用，是不是 连request和limit 都不用设置，仅需设置下priority 和 qos  即可。

大佬解答：资源规格智能托管是调度技术里面非常重要的一个课题。

大体思路：先分级，再给低优原本高优的资源，尽量确保高优QoS，保证不了了把低优任务重调度，重调度用资源预留确保迁移成功。

## 使用

假如创建一个pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    koordinator.sh/enable-colocation: "true"  
  name: test-pod
spec:
  containers:
  - name: app
    image: nginx:1.15.1
    resources:
        limits:
          cpu: "1"
          memory: "3456Mi"
        requests:
          cpu: "1"
          memory: "3456Mi"
```

Koordinator 提供了一个 ClusterColocationProfile CRD 和 对应的 Webhook ，修改和验证新创建的 Pod，注入 ClusterColocationProfile 中描述的字段。

```yaml
apiVersion: config.koordinator.sh/v1alpha1
kind: ClusterColocationProfile
metadata:
  name: colocation-profile-example
spec:
  namespaceSelector:
    matchLabels:
      koordinator.sh/enable-colocation: "true"
  selector:
    matchLabels:
      koordinator.sh/enable-colocation: "true"
  qosClass: BE
  priorityClassName: koord-batch
  koordinatorPriority: 1000
  schedulerName: koord-scheduler
  labels:
    koordinator.sh/mutated: "true"
  annotations: 
    koordinator.sh/intercepted: "true"
  patch:
    spec:
      terminationGracePeriodSeconds: 30
```
上面的 profile.yaml 文件描述了对所有含有标签 koordinator.sh/enable-colocation=true 的 Namespace 下的所有含有标签 koordinator.sh/enable-colocation=true 的 Pod 进行修改，注入 Koordinator QoSClass、Koordinator Priority 等。

pod 创建完成后 真实的样子
```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations: 
    koordinator.sh/intercepted: true
  labels:
    koordinator.sh/qosClass: BE
    koordinator.sh/priority: 1000
    koordinator.sh/mutated: true
  ...
spec:
  terminationGracePeriodSeconds: 30
  priority: 5000
  priorityClassName: koord-batch
  schedulerName: koord-scheduler      # schedulerName 换成了 koord-scheduler
  containers:
  - name: app
    image: nginx:1.15.1
    resources:
        limits:
          kubernetes.io/batch-cpu: "1000"
          kubernetes.io/batch-memory: 3456Mi
        requests:
          kubernetes.io/batch-cpu: "1000"
          kubernetes.io/batch-memory: 3456Mi
```


## 设计思路

打造现代化的云原生调度系统，Koordinator 坚持了如下的设计思路：

1. 拥抱 Kubernetes 上游标准，基于 Scheduler-Framework 来构建调度能力，而不是实现一个全新的调度器。构建标准形成共识是困难的，但破坏是容易的，Koordinator 社区与 Kubernetes sig-scheduling 社区相向而行。
2. **QoS 是系统的一等公民**，与业界大多数调度器更多的关注编排结果（静态）不同，Koordinator **更关注 Pod 运行时质量（QoS）**，因为对于调度系统的用户而言，运行时稳定性是其业务成功的关键。
3. 状态自闭环，Koordinator 认为调度必须是一个完整的闭环系统，才能满足企业级应用要求。因此，我们在第一个版本就引入了状态反馈回路，节点会根据运行时状态调优容器资源，中心会根据节点运行时状态反馈做调度决策。
    1. 中心调度 + 单机调度联合决策，中心调度看到全局视角，其决策可以找到集群中最适合应用需求的节点，而单机调度可以在节点侧做一定的灵活度，以应对应用突发的流量颠簸。
    2. 调度 + 重调度密切配合，调度解决 Pod 一次性放置的问题，而重调度才是驱动集群资源编排长期保持最优化的关键。Koordinator 将建设面向 SLO 的重调度能力，持续的驱动应用的编排符合预定义的 SLO。
4. 智能化、简单化，Koordinator 并不是就所有的选择暴露把问题留给客户，而是根据应用特征智能的为用户提供优化配置建议，简化用户使用 Kubernetes 的成本。   
    1. 调度系统不止于解决调度、重调度、弹性伸缩等领域问题，一个完整的调度系统，需要具备**基于历史数据驱动的自我迭代演进的能力**。为工作负载的运行历史状态建立数据仓库，基于这些运行历史的大数据分析，持续的改进应用间在各个维度的的亲和、互斥关系，才能在用户运行时体验、集群资源利用效率同时达到最佳状态。
 
Koordinator 针对智能化调度的设计思路如下：
1. 智能资源超卖，**超卖的基本思想是去利用那些已分配但未使用的资源来运行低优先级的任务**。Koordinator 首先解决的是节点资源充分利用的问题，通过分析节点容器的运行状态计算可超卖的资源量，并结合 QoS 的差异化诉求将超卖的资源分配给不同类型的任务，大幅提高集群的资源利用率。 PS：已分配但未使用的资源，对应下图灰色和深蓝色之间的部分。
  ![](/public/upload/kubernetes/koordinator_resource_model.jpg)
2. QoS 感知的重调度，当节点中 Pod 的运行时 QoS 不符合预期时，Koordinator 将智能决策抑制更低优先级的任务亦或是迁移当前受到干扰的容器，从而解决应用 QoS 不满足导致的问题。干扰检测和优化的过程可以分为以下几个过程：
  1. 干扰指标的采集和分析：选取干扰检测使用的指标需要考虑通用性和相关性，并尽量避免引入过多额外的资源开销。
  2. 干扰识别模型及算法：分为横向和纵向两个维度，横向是指分析同一应用中不同容器副本的指标表现，纵向是指分析在时间跨度上的数据表现，识别异常并确定“受害者”和“干扰源”。
  3. 干扰优化策略：充分评估策略成本，通过精准的压制或驱逐策略，控制“干扰源”引入的资源竞争，或将“受害者”进行迁移。同时建设相关模型，用于指导应用后续的调度，提前规避应用干扰的发生。
  4. 问题：重调度的细节问题很多，Pod驱逐后 集群是否有资源保证可以运行，涉及到资源预留。 

[Koordinator v0.7: 为任务调度领域注入新活力](https://mp.weixin.qq.com/s/oOjg8j9tDBs5jOm30XjCMA)

QoS 表示应用在节点上运行时得到的物理资源质量，包含了 LSR（Latency-Sensitive Reserved）、LS（Latency-Sensitive）和 BE（Best Effort），保障策略
1. CPU 方面，通过内核自研的 Group Identity 机制，针对不同 QoS 等级设置内核调度的优先级，优先保障 LSR/LS 的 cpu 调度，允许抢占 BE 的 CPU 使用，以达到最小化在线应用调度延迟的效果；对于 LS 应用的突发流量，提供了 CPU Burst 策略以规避非预期的 CPU 限流。
2. 内存方面，由于容器 cgroup 级别的直接内存回收会带来一定延时，LS 应用普遍开启了容器内存异步回收能力，规避同步回收带来的响应延迟抖动。除此之外，针对末级缓存（Last-Level Cache，LLC）这种共享资源，为了避免大数据等 BE 应用大量刷 Cache 导致 LS/LSR 应用的 Cache Miss Rate 异常增大，降低流水线执行效率，引入了 RDT 技术来限制 BE 应用可分配的 Cache 比例，缩小其争抢范围。

[阿里云容器服务差异化 SLO 混部技术实践](https://mp.weixin.qq.com/s/fkX_lStva96HEbmPbR6iZw)
1. CPU 资源质量
    1. [添加 K8S CPU limit 会降低服务性能？](https://mp.weixin.qq.com/s/cR6MpQu-n1cwMbXmVaXqzQ) 可以查看container_cpu_cfs_throttled_periods_total 指标
    1. CPU Burst，例如对于 CPU Limit = 2 的容器，操作系统内核会限制容器在每 100 ms 周期内最多使用 200 ms 的 CPU 时间片，进而导致请求的响应时延（RT）变大。当容器真实 CPU 资源使用小于 cfs_quota 时，内核会将多余的 CPU 时间“存入”到 cfs_burst 中；当容器有突发的 CPU 资源需求，需要使用超出 cfs_quota 的资源时，内核的 CFS 带宽控制器（CFS Bandwidth Controller，简称 BWC） 会允许其消费其之前存到 cfs_burst 的时间片。最终达到的效果是将容器**更长时间的平均 CPU 消耗限制在 quota 范围内，允许短时间内的 CPU 使用超过其 quota**。
    2. CPU 拓扑感知调度。在多核节点下，进程在运行过程中经常会被迁移到其不同的核心，考虑到有些应用的性能对 CPU 上下文切换比较敏感，kubelet 提供了 static 策略，允许 Guarantee 类型 Pod 独占 CPU 核心。CPU 绑核并不是“银弹”，若同一节点内大量 Burstable  类型 Pod 同时开启了拓扑感知调度，CPU 绑核可能会产生重叠，在个别场景下反而会加剧应用间的干扰。因此，拓扑感知调度更适合针对性的开启。
    3. 针对低优先级离线容器的 CPU 资源压制能力：内核Group Identity，Group Identity 功能可以对每一个容器设置身份标识，以区分容器中的任务优先级，系统内核在调度包含具有身份标识的任务时，会根据不同的优先级做相应处理。比如高优先级任务 有更多资源抢占机会。 
    4. 在 CPU 被持续压制的情况下，BE 任务自身的性能也会受到影响，将其驱逐重调度到其他空闲节点反而可以使任务更快完成。
2. 内存资源质量
    1. 时延敏感型业务（LS）和资源消耗型（BE）任务共同部署时，资源消耗型任务时常会瞬间申请大量的内存，使得系统的空闲内存触及全局最低水位线（global wmark_min），引发系统所有任务进入直接内存回收的慢速路径，进而导致延敏感型业务的性能抖动。
    2. 后台异步回收，当容器内存使用超过 memory.wmark_ratio 时，内核将自动启用异步内存回收机制，提前于直接内存回收，改善服务的运行质量。

![](/public/upload/kubernetes/koordinator_qos.jpg)

## 源码分析

```
koordinator
  /cmd
    /koord-descheduler   
    /koord-manager       
    /koord-runtime-proxy #  充当 Kubelet 和 Containerd 之间的代理，它用于拦截 CRI 请求，并应用一些资源管理策略， 如混合工作负载编排场景下按实例优先级设置不同的 cgroup 参数，针对最新的 Linux 内核、CPU 架构应用新的隔离策略等。
    /koord-scheduler     
    /koordlet
  /pkg
```

### 调度器

```
koordinator
  /cmd
    /koord-scheduler     
  /pkg
    /scheduler
      /apis
      /eventhandlers
      /frameworkext
      /plugins
        /batchresource
        /elasticquota
        /loadaware
        /nodenumaresource
        /reservation
```

1. k8s scheduler本身代码抽象比较好，对外提供`	"k8s.io/kubernetes/cmd/kube-scheduler/app"` 包，一两行就可以启动一个调度器，koord-scheduler  主要提供了 plugin 实现，注册到 plugin  regisry 中即可。
  ```
  cmd := app.NewSchedulerCommand(
    ... // 添加plugin
  );
  cmd.Execute()
  ``` 
2. 各个plugin 代码按需求分开看即可，后续的有 看到感兴趣的技术点持续补充

### koordlet

![](/public/upload/kubernetes/koordlet_overview.jpg)

```
koordinator
  /cmd
    /koordlet
  /pkg
    /koordlet
      /executor
      /metrics
      /pleg
      /qosmanager
      /resmanager
        /cpu_burst.go
        /cpu_evict.go
        /memory_evict.go
      /koordlet.go
```
koordlet 依次启动各个组件，各个组件又由多个功能模块/plugin 组成，每个功能模块运行一个小型control loop。比如 resManager 包含 cpu burst 模块，大致实现是check 所在node 上的所有pod，有需要 cpu burst的pod，就向pod 及container 对应cgroup 目录写入一个文件/配置。

```go
// koordlet 依次启动各个组件
daemon, err := agent.NewDaemon(cfg)
daemon.Run(stopCtx.Done())
	go func() {
		metricCache.Run
	}()
	go func() {
		resManager.Run
	}()
	go func() {
		qosManager.Run
	}()
  ...
```

以resManager 组件为例

```go
// resManager 启动各个功能模块
func (r *resmanager) Run(stopCh <-chan struct{}) error {
	defer utilruntime.HandleCrash()
	klog.Info("Starting resmanager")

	r.podsEvicted.Run(stopCh)
	go configextensions.RunQOSGreyCtrlPlugins(r.kubeClient, stopCh)

	util.RunFeature(r.reconcileBECgroup, ...)

	cgroupResourceReconcile := NewCgroupResourcesReconcile(r)
	util.RunFeatureWithInit(..., cgroupResourceReconcile.reconcile,...)

	cpuSuppress := NewCPUSuppress(r)
	util.RunFeature(cpuSuppress.suppressBECPU, ...)

	cpuBurst := NewCPUBurst(r)
	util.RunFeatureWithInit(..., cpuBurst.start,...)

	cpuEvictor := NewCPUEvictor(r)
	util.RunFeature(cpuEvictor.cpuEvict, ...)

	memoryEvictor := NewMemoryEvictor(r)
	util.RunFeature(memoryEvictor.memoryEvict, ...)

	rdtResCtrl := NewResctrlReconcile(r)
	util.RunFeatureWithInit(...,rdtResCtrl.reconcile,...)

	klog.Infof("start resmanager extensions")
	plugins.SetupPlugins(r.kubeClient, r.metricCache, r.statesInformer)
	utilruntime.Must(plugins.StartPlugins(r.config.QOSExtensionCfg, stopCh))

	klog.Info("Starting resmanager successfully")
	<-stopCh
	return nil
}
```
以cpu burst 为例，找到 需要 burst 的pod，更新其cgroup 配置。

```go
func (b *CPUBurst) start() {
	// sync config from node slo
	nodeSLO := b.resmanager.getNodeSLOCopy()
	b.nodeCPUBurstStrategy = nodeSLO.Spec.CPUBurstStrategy
	podsMeta := b.resmanager.statesInformer.GetAllPods()

	// get node state by node share pool usage
	nodeState := b.getNodeStateForBurst(*b.nodeCPUBurstStrategy.SharePoolThresholdPercent, podsMeta)
	for _, podMeta := range podsMeta {
		// ignore non-burstable pod, e.g. LSR, BE pods
		// merge burst config from pod and node
		cpuBurstCfg := genPodBurstConfig(podMeta.Pod, &b.nodeCPUBurstStrategy.CPUBurstConfig)
		b.applyCPUBurst(cpuBurstCfg, podMeta) // set cpu.cfs_burst_us for containers
		b.applyCFSQuotaBurst(cpuBurstCfg, podMeta, nodeState) // scale cpu.cfs_quota_us for pod and containers
	}
	b.Recycle()
}

// set cpu.cfs_burst_us for containers
func (b *CPUBurst) applyCPUBurst(burstCfg *slov1alpha1.CPUBurstConfig, podMeta *statesinformer.PodMeta) {
	pod := podMeta.Pod
	containerMap := make(map[string]*corev1.Container)
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		containerMap[container.Name] = container
	}

	podCFSBurstVal := int64(0)
	for i := range pod.Status.ContainerStatuses {
		containerStat := &pod.Status.ContainerStatuses[i]
		container, exist := containerMap[containerStat.Name]
		containerCFSBurstVal := calcStaticCPUBurstVal(container, burstCfg)
		containerDir, burstPathErr := koordletutil.GetContainerCgroupPathWithKube(podMeta.CgroupDir, containerStat)
		
		if system.ValidateResourceValue(&containerCFSBurstVal, containerDir, system.CPUBurst) {
			podCFSBurstVal += containerCFSBurstVal
			ownerRef := executor.ContainerOwnerRef(pod.Namespace, pod.Name, container.Name)
			containerCFSBurstValStr := strconv.FormatInt(containerCFSBurstVal, 10)
			updater := executor.NewCommonCgroupResourceUpdater(ownerRef, containerDir, system.CPUBurst, containerCFSBurstValStr)
			updated, err := b.executor.UpdateByCache(updater)
		}
	} // end for containers
f
	podDir := koordletutil.GetPodCgroupDirWithKube(podMeta.CgroupDir)
	if system.ValidateResourceValue(&podCFSBurstVal, podDir, system.CPUBurst) {
		ownerRef := executor.PodOwnerRef(pod.Namespace, pod.Name)
		podCFSBurstValStr := strconv.FormatInt(podCFSBurstVal, 10)
		updater := executor.NewCommonCgroupResourceUpdater(ownerRef, podDir, system.CPUBurst, podCFSBurstValStr)
		updated, err := b.executor.UpdateByCache(updater)
	}
}
```
executor.UpdateByCache 实质是 特定cgroup 目录写入一个文件。