---

layout: post
title: Kubernetes源码分析——kubelet
category: 技术
tags: Kubernetes
keywords: kubernetes 源码分析 kubelet

---

## 简介

* TOC
{:toc}

go 程序中大量使用channel
1. 一个是消灭了观察者模式
2. 很多功能组件得以独立。以前对外提供接口，等着上游组件函数调用。现在改成了消息传递，主流程/入口直接go start启动组件，然后在for 循环里 等着channel 来消息就行。channel 作为独立组件的输入，输出则为io操作 或向另一个channel 发出事件。

反应在源码分析上
1. 之前，功能分解由接口/类体现，类图有很多接口、实现类（因为要用接口界定组件间的关系）。序列图有较深的 函数调用（从左到右很长）。
2. 现在，功能分解由协程体现，一个组件一个协程，大家都是main函数/入口对象的“亲儿子”，各干各的活儿，通过channel 协同

kubelet 源码虽然庞大，但并不复杂，基本适用于上述规律，在1.13 版本中，kubelet 大约有13个mannager 保证pod 正常运行。

## 整体结构

**Kubelet 作为 Kubernetes 集群中的 node agent**，一方面，kubelet 扮演的是集群控制器的角色，它定期从 API Server 获取 Pod 等相关资源的信息，并依照这些信息，控制运行在节点上 Pod 的执行;另外一方面， kubelet 作为节点状况的监视器，它获取节点信息，并以集群客户端的角色，把这些 状况同步到 API Server。

### 节点状况的监视器

Kubelet 会使用上图中的 NodeStatus 机制，定期检查集群节点状况，并把节点 状况同步到 API Server。而 **NodeStatus 判断节点就绪状况的一个主要依据，就是 PLEG**。

PLEG 是 Pod Lifecycle Events Generator 的缩写，基本上它的执行逻辑，是 定期检查节点上 Pod 运行情况，如果发现感兴趣的变化，PLEG 就会把这种变化包 装成 Event 发送给 Kubelet 的主同步机制 syncLoop 去处理。

![](/public/upload/kubernetes/kubelet_pleg.png)

### 集群控制器

[kubectl 创建 Pod 背后到底发生了什么？](https://mp.weixin.qq.com/s/ctdvbasKE-vpLRxDJjwVMw)从kubectl 命令开始，kubectl ==> apiserver ==> controller ==> scheduler 所有的状态变化仅仅只是针对保存在 etcd 中的资源记录。到Kubelet 才开始来真的。如果换一种思维模式，可以把 Kubelet 当成一种特殊的 Controller，它每隔 20 秒（可以自定义）向 kube-apiserver 通过 NodeName 获取自身 Node 上所要运行的 Pod 清单。一旦获取到了这个清单，它就会通过与自己的内部缓存进行比较来检测新增加的 Pod，如果有差异，就开始同步 Pod 列表。

![](/public/upload/kubernetes/kubelet_overview.png)

kubelet 从PodManager 中拿到 Pod数据，判断是否需要操作，SyncPod 到 kubeGenericRuntimeManager 中。除了取Pod操作Pod外，还做一些eviction 逻辑的处理。

![](/public/upload/kubernetes/kubelet_object.png)

## 启动流程

[Kubelet 源码剖析](https://toutiao.io/posts/z2e88b/preview) 有一个启动的序列图

![](/public/upload/kubernetes/kubelet_init_sequence.png)

比较有意思的是 Bootstap interface 的描述：Bootstrap is a bootstrapping interface for kubelet, targets the initialization protocol. 也就是 `cmd/kubelet` 和 `pkg/kubelet` 的边界是 Bootstap interface

```go
// Run starts the kubelet reacting to config updates
func (kl *Kubelet) Run(updates <-chan kubetypes.PodUpdate) {
    ...
    // Start the cloud provider sync manager
    go kl.cloudResourceSyncManager.Run(wait.NeverStop)
    // Start volume manager
    go kl.volumeManager.Run(kl.sourcesReady, wait.NeverStop)
    // Start syncing node status immediately, this may set up things the runtime needs to run.
    go wait.Until(kl.syncNodeStatus, kl.nodeStatusUpdateFrequency, wait.NeverStop)
    go kl.fastStatusUpdateOnce()
    // start syncing lease
    go kl.nodeLeaseController.Run(wait.NeverStop)
    go wait.Until(kl.updateRuntimeUp, 5*time.Second, wait.NeverStop)
    // Start loop to sync iptables util rules
    go wait.Until(kl.syncNetworkUtil, 1*time.Minute, wait.NeverStop)
    // Start a goroutine responsible for killing pods (that are not properly handled by pod workers).
    go wait.Until(kl.podKiller, 1*time.Second, wait.NeverStop)
    // Start component sync loops.
    kl.statusManager.Start()
    kl.probeManager.Start()
    // Start syncing RuntimeClasses if enabled.
    go kl.runtimeClassManager.Run(wait.NeverStop)
    // Start the pod lifecycle event generator.
    kl.pleg.Start()
    kl.syncLoop(updates, kl)
}
```

kubelet 源码包结构

```
k8s.io/kubernetes
    /cmd/kubelet
        /app
        /kubelet.go
    /pkg/kubelet
        /cadvisor
        /configmap
        /prober
        /status
        ...
        /kubelet.go  定义了kubelet struct
```

pkg 下几乎每一个文件夹对应了 kubelet 的一个功能组件，定义了一个manager 协程，负责具体的功能实现，启动时只需 `go manager.start`。此外有一个syncLoop 负责kubelet 主功能的实现。


## syncLoop

syncLoop is the main loop for processing changes. It watches for changes from three channels (**file, apiserver, and http***) and creates a union of them. For any new change seen, will run a sync against desired state and running state. If no changes are seen to the configuration, will synchronize the last known desired
state every sync-frequency seconds. **Never returns**.

```go
func (kl *Kubelet) syncLoop(updates <-chan kubetypes.PodUpdate, handler SyncHandler) {
    // 准备工作
    for{
        time.Sleep(duration)
        if !kl.syncLoopIteration(...) {
			break
		}
        ...
    }
}
```
	
syncLoopIteration 接收来自多个方向的消息(**file, apiserver, and http***)，run a sync against desired state and running state

```go
func (kl *Kubelet) syncLoopIteration(configCh <-chan kubetypes.PodUpdate, handler SyncHandler,
    syncCh <-chan time.Time, housekeepingCh <-chan time.Time, plegCh <-chan *pleg.PodLifecycleEvent) bool {
    select {
    case u, open := <-configCh:
        switch case...
    case e := <-plegCh:
        ...
    case <-syncCh:
        ...
    case update := <-kl.livenessManager.Updates():
        ...
    case <-housekeepingCh:
        ...
    }
    return true
}
```

syncLoopIteration reads from various channels and dispatches pods to the given handler. 以configCh 为例

```go
switch u.Op {
case kubetypes.ADD:
    handler.HandlePodAdditions(u.Pods)
case kubetypes.UPDATE:
    handler.HandlePodUpdates(u.Pods)
case kubetypes.REMOVE:
    handler.HandlePodRemoves(u.Pods)
case kubetypes.RECONCILE:
    handler.HandlePodReconcile(u.Pods)
case kubetypes.DELETE:
    // DELETE is treated as a UPDATE because of graceful deletion.
    handler.HandlePodUpdates(u.Pods)
case kubetypes.RESTORE:
    // These are pods restored from the checkpoint. Treat them as new pods.
    handler.HandlePodAdditions(u.Pods)
}
```
	
最终的立足点还是 syncHandler（还是Kubelet 自己实现的），下面分析下 HandlePodAdditions
	
### 新建 pod

代码中去掉了跟创建 无关的部分，删减了日志、错误校验等

```go
func (kl *Kubelet) HandlePodAdditions(pods []*v1.Pod) {
    sort.Sort(sliceutils.PodsByCreationTime(pods))
    for _, pod := range pods {
        ...
        // Always add the pod to the pod manager. Kubelet relies on the pod manager as the source of truth for the desired state. If a pod does not exist in the pod manager, it means that it has been deleted in the apiserver and no action (other than cleanup) is required.
        kl.podManager.AddPod(pod)
        ...
        mirrorPod, _ := kl.podManager.GetMirrorPodByPod(pod)
        kl.dispatchWork(pod, kubetypes.SyncPodCreate, mirrorPod, start)
        kl.probeManager.AddPod(pod)
    }
}
```
	
`kl.podManager.AddPod` 和 `kl.probeManager.AddPod(pod)` 都只是将pod 纳入跟踪，真正创建pod的是dispatchWork，然后又转回 kl.syncPod

```go
func (kl *Kubelet) syncPod(o syncPodOptions) error {
    ...
    // Generate final API pod status with pod and status manager status
    apiPodStatus := kl.generateAPIPodStatus(pod, podStatus)
    existingStatus, ok := kl.statusManager.GetPodStatus(pod.UID)
    if runnable := kl.canRunPod(pod); !runnable.Admit {...}
    // Update status in the status manager
    kl.statusManager.SetPodStatus(pod, apiPodStatus)
    // Create Cgroups for the pod and apply resource parameters to them if cgroups-per-qos flag is enabled.
    pcm := kl.containerManager.NewPodContainerManager()
    // Make data directories for the pod
    kl.makePodDataDirs(pod);
    // Fetch the pull secrets for the pod
    pullSecrets := kl.getPullSecretsForPod(pod)
    // Call the container runtime's SyncPod callback
    result := kl.containerRuntime.SyncPod(pod, apiPodStatus, podStatus, pullSecrets, kl.backOff)
    ...
}
```

kubeGenericRuntimeManager.syncPod

```go
func (m *kubeGenericRuntimeManager) SyncPod(pod *v1.Pod, _ v1.PodStatus, podStatus *kubecontainer.PodStatus, pullSecrets []v1.Secret, backOff *flowcontrol.Backoff) (result kubecontainer.PodSyncResult) {
    // Step 1: Compute sandbox and container changes.
    podContainerChanges := m.computePodActions(pod, podStatus)
    ...
    // Step 4: Create a sandbox for the pod if necessary.
    podSandboxID, msg, err = m.createPodSandbox(pod, podContainerChanges.Attempt)		
    // Get podSandboxConfig for containers to start.
    podSandboxConfig, err := m.generatePodSandboxConfig(pod, podContainerChanges.Attempt)
    // Step 5: start the init container.
    if container := podContainerChanges.NextInitContainerToStart; container != nil {
        // Start the next init container.
        msg, err := m.startContainer(podSandboxID, podSandboxConfig, container, pod, podStatus, pullSecrets, podIP, kubecontainer.ContainerTypeInit); 
    }
    // Step 6: start containers in podContainerChanges.ContainersToStart.
    for _, idx := range podContainerChanges.ContainersToStart {
        container := &pod.Spec.Containers[idx]
        msg, err := m.startContainer(podSandboxID, podSandboxConfig, container, pod, podStatus, pullSecrets, podIP, kubecontainer.ContainerTypeRegular); 
    }
    ...
}
```
	
m.createPodSandbox 和 startContainer

`pkg/kubelet/kuberuntime/`包中，kuberuntime_manager.go 定义了  kubeGenericRuntimeManager struct 及其接口方法实现，但接口方法的内部依赖方法 分散在 package 下的其它go文件中。多个文件合起来 组成了kubeGenericRuntimeManager 类实现。

|文件|方法|
|---|---|
|kuberuntime_manager.go|NewKubeGenericRuntimeManager<br>GetPods<br>SyncPod<br>KillPod<br>GetPodStatus|
|kuberuntime_sandbox.go|createPodSandbox|
|kuberuntime_container.go|startContainer|
|kuberuntime_image.go|PullImage|

```go
func (m *kubeGenericRuntimeManager) startContainer(podSandboxID string, podSandboxConfig *runtimeapi.PodSandboxConfig, container *v1.Container, pod *v1.Pod, podStatus *kubecontainer.PodStatus, pullSecrets []v1.Secret, podIP string, containerType kubecontainer.ContainerType) (string, error) {
    // Step 1: pull the image.
    imageRef, msg, err := m.imagePuller.EnsureImageExists(pod, container, pullSecrets)
    // Step 2: create the container.
    ref, err := kubecontainer.GenerateContainerRef(pod, container)
    containerConfig, cleanupAction, err := m.generateContainerConfig(container, pod, restartCount, podIP, imageRef, containerType)
    containerID, err := m.runtimeService.CreateContainer(podSandboxID, containerConfig, podSandboxConfig)
    err = m.internalLifecycle.PreStartContainer(pod, container, containerID)
    // Step 3: start the container.
    err = m.runtimeService.StartContainer(containerID)
    // Step 4: execute the post start hook.
    msg, handlerErr := m.runner.Run(kubeContainerID, pod, container, container.Lifecycle.PostStart)
}
```

![](/public/upload/kubernetes/kubelet_create_pod_sequence.png)

从图中可以看到，蓝色区域 grpc 调用 dockershim等cri shim 完成。


## 其它 

![](/public/upload/kubernetes/kubelet_intro.png)

[kubelet 源码分析：Garbage Collect](https://cizixs.com/2017/06/09/kubelet-source-code-analysis-part-3/) gc 机制后面由  eviction 代替

[kubelet 源码分析：statusManager 和 probeManager](https://cizixs.com/2017/06/12/kubelet-source-code-analysis-part4-status-manager/)


kubelet像极了spring mvc的controller-service-rpc，一层一层的 将高层概念/动作 分解为 cri 提供的基本概念/底层操作。

|spring mvc|kubelet|kubelet 所在包|概念|
|----|---|---|---|
|controller|kubelet struct|`pkg/kubelet/kubelet.go`||
|service|Runtime interface|`pkg/kubelet/container`|Pod/PodStatus/Container/ContainerStatus/Image<br/>Mount/PortMapping/VolumeInfo/RunContainerOptions|
|service.impl|kubeGenericRuntimeManager struct|`pkg/kubelet/kuberuntime`|
|rpc|RuntimeService interface/ImageManagerService interface|`pkg/kubelet/apis/cri`|Container/PodSandbox/Image/AuthConfig|
|rpc.impl|RemoteRuntimeService struct|`pkg/kubelet/apis/remote`||