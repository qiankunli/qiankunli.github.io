---

layout: post
title: Kubernetes源码分析——kubelet
category: 技术
tags: Kubernetes
keywords: kubernetes 源码分析

---

## 简介

* TOC
{:toc}


建议先看下前文 [Kubernetes源码分析——从kubectl开始](http://qiankunli.github.io/2018/12/23/kubernetes_source_kubectl.html)

笔者不太喜欢抠细节，加上k8s 没有使用 [uber-go/dig](https://github.com/uber-go/dig) 之类的依赖注入库（很多代码 搞到最后就是 spring ioc干的活儿，但一换go的马甲，经常一下子没看出来），struct 组合代码 和 struct 行为代码耦合在一起，本文主要关注 行为代码。笔者认为，**关键不是一些结构体定义， 而是业务逻辑：怎么启动了一个pod？怎么管理pod 的等？与cni、csi 插件怎么结合？如何与docker 协同？**

背景知识

1. 一个grpc client 和 server 如何实现
2. [spf13/cobra](https://github.com/spf13/cobra)
3. go 运行可执行文件

##  整体结构

《深入剖析Kubernetes》：kubelet 调用下层容器运行时的执行过程，并不会直接调用Docker 的 API，而是通过一组叫作 CRI（Container Runtime Interface，容器运行时接口）的 gRPC 接口来间接执行的。Kubernetes 项目之所以要在 kubelet 中引入这样一层单独的抽象，当然是为了对 Kubernetes 屏蔽下层容器运行时的差异。实际上，对于 1.6 版本之前的 Kubernetes 来说，它就是直接调用 Docker 的 API 来创建和管理容器的。

![](/public/upload/kubernetes/cri_shim.png)

除了 dockershim 之外，其他容器运行时的 CRI shim，都是需要额外部署在宿主机上的。

go 里面没有 extends、implements 这些关键字，我们要把struct、interface重建起来

![](/public/upload/kubernetes/kubelet_object.png)

上图左中右，像极了spring mvc的controller-service-rpc，一层一层的 将高层概念/动作 分解为 cri 提供的基本概念/底层操作。



|spring mvc|kubelet|kubelet 所在包|概念|
|----|---|---|---|
|controller|kubelet struct|`pkg/kubelet/kubelet.go`||
|service|Runtime interface|`pkg/kubelet/container`|Pod/PodStatus/Container/ContainerStatus/Image<br/>Mount/PortMapping/VolumeInfo/RunContainerOptions|
|service.impl|kubeGenericRuntimeManager struct|`pkg/kubelet/kuberuntime`|
|rpc|RuntimeService interface/ImageManagerService interface|`pkg/kubelet/apis/cri`|Container/PodSandbox/Image/AuthConfig|
|rpc.impl|RemoteRuntimeService struct|`pkg/kubelet/apis/remote`||

## 启动流程


[kubelet 源码分析：启动流程](https://cizixs.com/2017/06/06/kubelet-source-code-analysis-part-1/)

[kubernetes源码阅读 kubelet初探](https://zhuanlan.zhihu.com/p/35710779) 

几个感觉：

1. 配置 [spf13/cobra](https://github.com/spf13/cobra)，并从命令行中获取各种参数，封装为KubeletServer（就是个 配置参数的struct） 和kubelet.Dependencies 然后开始run
2. cmd/kubelet 和 pkg/kubelet 边界在哪？
3. kubelet struct 就是一个大controller，依赖各种对象，有一个对它进行配置 和初始化的过程，
4. kubelet 是一个命令行方式启动的 http server，内有有一些“线程” 

	* 监听pod/接收指令，然后做出反应
	* 向api server 汇报数据

[Kubelet 源码剖析](https://toutiao.io/posts/z2e88b/preview) 有一个启动的序列图

![](/public/upload/kubernetes/kubelet_init_sequence.png)

比较有意思的是 Bootstap interface 的描述：Bootstrap is a bootstrapping interface for kubelet, targets the initialization protocol. 也就是 `cmd/kubelet` 和 `pkg/kubelet` 的边界是 Bootstap interface

kubelet 启动逻辑， 启动一堆线程，然后开启一个syncLoop

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

复杂的程序，一定使用面向对象思想写的（函数式编程简化了一部分逻辑）

## syncLoop

syncLoop is the main loop for processing changes. It watches for changes from three channels (file, apiserver, and http) and creates a union of them. For any new change seen, will run a sync against desired state and running state. If no changes are seen to the configuration, will synchronize the last known desired
state every sync-frequency seconds. **Never returns**.

	func (kl *Kubelet) syncLoop(updates <-chan kubetypes.PodUpdate, handler SyncHandler) {
		// 准备工作
		for{
			time.Sleep(duration)
			kl.syncLoopIteration(...)
			...
		}
	}
	
syncLoopIteration 接收来自多个方向的消息，run a sync against desired state and running state

	func (kl *Kubelet) syncLoopIteration(configCh <-chan kubetypes.PodUpdate, handler SyncHandler,
		syncCh <-chan time.Time, housekeepingCh <-chan time.Time, plegCh <-chan *pleg.PodLifecycleEvent) bool {
		select {
		case u, open := <-configCh:
		case e := <-plegCh:...
		case <-syncCh:...
		case update := <-kl.livenessManager.Updates():...
		case <-housekeepingCh:...
		}
		return true
	}

syncLoopIteration reads from various channels and dispatches pods to the given handler. 以configCh 为例

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
	
最终的立足点还是 syncHandler（还是Kubelet 自己实现的），下面分析下 HandlePodAdditions
	
## 新建 pod

代码中去掉了跟创建 无关的部分，删减了日志、错误校验等

	func (kl *Kubelet) HandlePodAdditions(pods []*v1.Pod) {
		sort.Sort(sliceutils.PodsByCreationTime(pods))
		for _, pod := range pods {
			...
			// Always add the pod to the pod manager. Kubelet relies on the pod manager as the source of truth for the desired state. If a pod does
			// not exist in the pod manager, it means that it has been deleted in the apiserver and no action (other than cleanup) is required.
			kl.podManager.AddPod(pod)
			...
			mirrorPod, _ := kl.podManager.GetMirrorPodByPod(pod)
			kl.dispatchWork(pod, kubetypes.SyncPodCreate, mirrorPod, start)
			kl.probeManager.AddPod(pod)
		}
	}
	
`kl.podManager.AddPod` 和 `kl.probeManager.AddPod(pod)` 都只是将pod 纳入跟踪，真正创建pod的是dispatchWork，然后又转回 kl.syncPod

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

kubeGenericRuntimeManager.syncPod

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
	
m.createPodSandbox 和 startContainer

`pkg/kubelet/kuberuntime/`包中，kuberuntime_manager.go 定义了  kubeGenericRuntimeManager struct 及其接口方法实现，但接口方法的内部依赖方法 分散在 package 下的其它go文件中。其本质是将 一个“类方法”分散在了多个go 文件中，多个文件合起来 组成了kubeGenericRuntimeManager 类实现。

|文件|方法|备注|
|---|---|---|
|kuberuntime_manager.go|NewKubeGenericRuntimeManager/GetPods/SyncPod/KillPod/GetPodStatus etc|
|kuberuntime_sandbox.go|createPodSandbox|
|kuberuntime_container.go|startContainer|
|kuberuntime_image.go|PullImage|

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

![](/public/upload/kubernetes/kubelet_create_pod_sequence.png)

从图中可以看到，蓝色区域 grpc 调用 dockershim等cri shim 完成。笔者java 开发出身，上诉代码 换成spring mvc 就很好理解：**从请求到实际的底层接口，将抽象的概念展开，中间经常涉及到model的转换**

其它材料 [kubelet 源码分析：pod 新建流程](https://cizixs.com/2017/06/07/kubelet-source-code-analysis-part-2/)

## dockershim

在 Kubernetes 中，处理容器网络相关的逻辑并不会在kubelet 主干代码里执行，而是会在具体的 CRI（Container Runtime Interface，容器运行时接口）实现里完成。对于 Docker 项目来说，它的 CRI 实现叫作 dockershim，相关代码在 `pkg/kubelet/dockershim` 下

CRI 设计的一个重要原则，**就是确保这个接口本身只关注容器， 不关注Pod**。但CRI 里有一个PodSandbox，抽取了Pod里的一部分与容器运行时相关的字段，比如Hostname、DnsConfig等。作为具体的容器项目，自己决定如何使用这些字段来实现一个k8s期望的Pod模型。

![](/public/upload/kubernetes/k8s_cri_docker.png)

kubelet中调用CRI shim提供的imageService,ContainerService接口，作为gRPC client，dockershim实现了CRI gRPC Server服务端的服务实现，但是dockershim仍然整合到了kubelet中，作为kubelet默认的CRI　shim实现．**所以说，要了解这块得先熟悉 一个grpc client 和 server 如何实现**

dockershim 封了一个`pkg/kubelet/dockershim/libdocker`  会使用docker提供的client来调用cli接口，没错！就是`github.com/docker/docker/client`

顺着上文思路，当 kubelet 组件需要创建 Pod 的时候，它第一个创建的一定是 Infra 容器，这体现在上图的 RunPodSandbox 中


![](/public/upload/kubernetes/dockershim_sequence.png)

RunPodSandbox creates and starts a pod-level sandbox. Runtimes should ensure the sandbox is in ready state.For docker, PodSandbox is implemented by a container holding the network namespace for the pod.Note: docker doesn't use LogDirectory (yet).

	func (ds *dockerService) RunPodSandbox(ctx context.Context, r *runtimeapi.RunPodSandboxRequest) (*runtimeapi.RunPodSandboxResponse, error) {
		config := r.GetConfig()
		// Step 1: Pull the image for the sandbox.
		err := ensureSandboxImageExists(ds.client, defaultSandboxImage);
		// Step 2: Create the sandbox container.
		createConfig, err := ds.makeSandboxDockerConfig(config, image)
		createResp, err := ds.client.CreateContainer(*createConfig)
		ds.setNetworkReady(createResp.ID, false)
		defer func(e *error) {
			// Set networking ready depending on the error return of the parent function
			if *e == nil {
				ds.setNetworkReady(createResp.ID, true)
			}
		}(&err)
		// Step 3: Create Sandbox Checkpoint.
		ds.checkpointManager.CreateCheckpoint(createResp.ID, constructPodSandboxCheckpoint(config)); 
		// Step 4: Start the sandbox container. Assume kubelet's garbage collector would remove the sandbox later, if startContainer failed.
		err = ds.client.StartContainer(createResp.ID)
		// Rewrite resolv.conf file generated by docker.
		containerInfo, err := ds.client.InspectContainer(createResp.ID)
		err := rewriteResolvFile(containerInfo.ResolvConfPath, dnsConfig.Servers, dnsConfig.Searches, dnsConfig.Options);
		// Do not invoke network plugins if in hostNetwork mode.
		if config.GetLinux().GetSecurityContext().GetNamespaceOptions().GetNetwork() == runtimeapi.NamespaceMode_NODE {
			return resp, nil
		}
		// Step 5: Setup networking for the sandbox.
		// All pod networking is setup by a CNI plugin discovered at startup time. This plugin assigns the pod ip, sets up routes inside the sandbox,
		// creates interfaces etc. In theory, its jurisdiction ends with pod sandbox networking, but it might insert iptables rules or open ports
		// on the host as well, to satisfy parts of the pod spec that aren't recognized by the CNI standard yet.
		err = ds.network.SetUpPod(config.GetMetadata().Namespace, config.GetMetadata().Name, cID, config.Annotations, networkOptions)
		return resp, nil
	}


与 kubeGenericRuntimeManager 类似，dockerService 方法分散在各个文件中

|go文件|包含方法|
|---|---|
|docker_service.go|dockerService struct 定义 以及GetNetNS/Start/Status等|
|docker_sandbox.go|RunPodSandbox等|
|docker_container.go|CreateContainer/StartContainer等|
|docker_image.go|PullImage等|

![](/public/upload/kubernetes/dockershim_object.png)

从左到右可以看到用户请求 怎么跟cni plugin(binary file) 产生关联的

golang中一个接口可以包含一个或多个其他的接口，这相当于直接将这些内嵌接口的方法列举在外层接口中一样。

## 加载 CNI plugin

建议参看[《Container-Networking-Docker-Kubernetes》笔记](http://qiankunli.github.io/2018/10/11/docker_to_k8s_network_note.html)了解下CNI 的相关概念及使用。

![](/public/upload/kubernetes/kubelet_cni_init.png)

cniNetworkPlugin.Init 方法逻辑如下

	func (plugin *cniNetworkPlugin) Init(host network.Host, hairpinMode kubeletconfig.HairpinMode, nonMasqueradeCIDR string, mtu int) error {
		err := plugin.platformInit()
		...
		plugin.host = host
		plugin.syncNetworkConfig()
		return nil
	}

	func (plugin *cniNetworkPlugin) syncNetworkConfig() {
		network, err := getDefaultCNINetwork(plugin.confDir, plugin.binDirs)
		...
		plugin.setDefaultNetwork(network)
	}

从confDir 加载xx.conflist，结合binDirs 构造defaultNetwork

	func getDefaultCNINetwork(confDir string, binDirs []string) (*cniNetwork, error) {
		files, err := libcni.ConfFiles(confDir, []string{".conf", ".conflist", ".json"})
		sort.Strings(files)
		for _, confFile := range files {
			var confList *libcni.NetworkConfigList
			if strings.HasSuffix(confFile, ".conflist") {
				confList, err = libcni.ConfListFromFile(confFile)
				...
			} 
			network := &cniNetwork{
				name:          confList.Name,
				NetworkConfig: confList,
				CNIConfig:     &libcni.CNIConfig{Path: binDirs},
			}
			return network, nil
		}
		return nil, fmt.Errorf("No valid networks found in %s", confDir)
	}

docker service 作为grpc server 实现，最终还是操作了 CNI，CNIConfig接收到指令后， 拼凑“shell指令及参数” 执行 cni binary文件。CNI 插件的初始化就是 根据binary path 初始化CNIConfig，进而初始化NetworkPlugin。**至于cni binary 本身只需要执行时运行即可，就像go 运行一般的可执行文件一样**。

`github.com/containernetworking/cni/pkg/invoke/raw_exec.go`

	func (e *RawExec) ExecPlugin(ctx context.Context, pluginPath string, stdinData []byte, environ []string) ([]byte, error) {
		stdout := &bytes.Buffer{}
		c := exec.CommandContext(ctx, pluginPath)
		c.Env = environ
		c.Stdin = bytes.NewBuffer(stdinData)
		c.Stdout = stdout
		c.Stderr = e.Stderr
		if err := c.Run(); err != nil {
			return nil, pluginErr(err, stdout.Bytes())
		}
		return stdout.Bytes(), nil
	}

## 其它 

|k8s涉及的组件|功能交付方式|
|---|---|
|kubectl|命令行，用户直接使用|
|kubelet|命令行，提供http服务|
|cri-shim|grpc server|
|cni plugin|命令行，程序直接使用|

关于k8s 插件，可以回顾下

![](/public/upload/kubernetes/parse_k8s_1.png)

[kubelet 源码分析：Garbage Collect](https://cizixs.com/2017/06/09/kubelet-source-code-analysis-part-3/) gc 机制后面由  eviction 代替

[kubelet 源码分析：statusManager 和 probeManager](https://cizixs.com/2017/06/12/kubelet-source-code-analysis-part4-status-manager/)
