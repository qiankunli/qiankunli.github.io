---
layout: post
title: Kubernetes pod 组件
category: 技术
tags: Kubernetes
keywords: Kubernetes Pod

---


## 简介

* TOC
{:toc}

## What is a pod?

A pod models an application-specific "logical host(逻辑节点)" in a containerized environment. It may contain one or more containers which are relatively tightly coupled—in a pre-container world（在 pre-container 时代紧密联系的进程 ，在container 时代放在一个pod里）, they would have executed on the same physical or virtual host.a pod has a single IP address.  Multiple containers that run in a pod all share that common network name space。

Like running containers, pods are considered to be relatively ephemeral rather than durable entities. Pods are scheduled to nodes and remain there until termination (according to restart policy) or deletion. When a node dies, the pods scheduled to that node are deleted. Specific pods are never rescheduled to new nodes; instead, they must be replaced.

重点不是pod 是什么，而是什么情况下， 我们要将多个容器放在pod 里。 "为什么需要一个pod?" 详细论述 [kubernetes objects再认识](http://qiankunli.github.io/2018/11/04/kubernetes_objects.html)


### Uses of pods（应用场景）

Pods can be used to host vertically integrated application stacks, but their primary motivation is to support co-located, co-managed （这两个形容词绝了）helper programs, such as:

1. Content management systems, file and data loaders, local cache managers, etc.
2. Log and checkpoint backup, compression, rotation, snapshotting, etc.
3. Data-change watchers, log tailers, logging and monitoring adapters, event publishers, etc.
4. Proxies, bridges, and adapters.
5. Controllers, managers, configurators, and updaters.

**Individual pods are not intended to run multiple instances of the same application**, in general.

### 小结

A pod is a relatively tightly coupled group of containers that are scheduled onto the same host. 

1. It models an application-specific(面向应用) "virtual host" in a containerized environment. 
2. Pods serve as units of scheduling, deployment, and horizontal scaling/replication. 
3. Pods share fate（命运）, and share some resources, such as storage volumes and IP addresses.(网络通信和数据交互就非常方便且高效)

## 为什么需要pod？

本小节大部分来自对极客时间《深入剖析kubernetes》的学习

1. 操作系统为什么要有进程组？原因之一是 Linux 操作系统只需要将信号，比如，SIGKILL 信号，发送给一个进程组，那么该进程组中的所有进程就都会收到这个信号而终止运行。
2. 在 Borg 项目的开发和实践过程中，Google 公司的工程师们发现，他们部署的应用，往往都存在着类似于“进程和进程组”的关系。更具体地说，就是这些应用之间有着密切的协作关系，使得它们必须部署在同一台机器上。具有“超亲密关系”容器的典型特征包括但不限于：

	* 互相之间会发生直接的文件交换
	* 使用 localhost 或者 Socket文件进行本地通信
	* 会发生非常频繁的远程调用
	* 需要共享某些 Linux Namespace

3. 亲密关系 ==> 亲密关系为什么不在调度层面解决掉？非得提出pod 的概念？[容器设计模式](https://www.usenix.org/system/files/conference/hotcloud16/hotcloud16_burns.pdf)
4. Pod，其实是一组共享了某些资源的容器。当然，共享Network Namespace和Volume 可以通过`通过docker run --net=B --volumes-from=B --name-=A image-A...`来实现，但这样 容器 B 就必须比容器 A 先启动，这样一个 Pod 里的多个容器就不是对等关系，而是拓扑关系了。
5. **Pod 最重要的一个事实是：它只是一个逻辑概念。有了Pod，我们可以说Network Namespace和Volume 不是container A 的，也不是Container B的，而是Pod 的。哪怕Container A/B 还没有启动，我们也可以 配置Network Namespace和Volume**。以network namespace 为例，为什么需要一个pause 容器参见[《Container-Networking-Docker-Kubernetes》笔记](http://qiankunli.github.io/2018/10/11/docker_to_k8s_network_note.html)
4. Pod 这种“超亲密关系”容器的设计思想，实际上就是希望，当用户想在一个容器里跑多个功能并不相关的应用时，应该优先考虑它们是不是更应该被描述成一个 Pod 里的多个容器。你就可以把整个虚拟机想象成为一个 Pod，把这些进程分别做成分别做成容器镜像，把有顺序关系的容器，定义为 Init Container。 作者提到了tomcat 镜像和war 包（war包单独做一个镜像）的例子，非常精彩，好就好在 分别做镜像 肯定比 镜像做在一起要方便。**重点不是pod 是什么，而是什么情况下， 我们要将多个容器放在pod 里。 **
5. [https://cloud.google.com/container-engine/docs](https://cloud.google.com/container-engine/docs) Pods also simplify application deployment and management by providing a higher-level abstraction than the raw, low-level container interface. **Pods serve as units of deployment and horizontal scaling/replication. Co-location, fate sharing, coordinated replication, resource sharing, and dependency management are handled automatically.**


综上，我们可以有一个不太成熟的理解：pod本质是一个场景的最佳实践的普适化。

“容器”镜像虽然好用，但是容器这样一个“沙盒”的概念，对于描述应用来说，还是太过简单了。Pod 对象，其实就是容器的升级版。它对容器进行了组合，添加了更多的属性和字段。
Pod 这个看似复杂的 API 对象，实际上就是对容器的进一步抽象和封装而已。

所以，将pod 单纯的理解为 多个容器 数字上的叠加 是不科学的。

## Pod Operations

### Creating a pod

Pod，而不是容器，才是 Kubernetes 项目中的最小编排单位。将这个设计落实到 API 对象上，容器（Container）就成了 Pod 属性里的一个普通的字段。那么，一个很自然的问题就是：到底哪些属性属于 Pod 对象，而又有哪些属性属于 Container 呢？

Pod 扮演的是传统部署环境里“虚拟机”的角色。这样的设计，是为了使用户从传统环境（虚拟机环境）向 Kubernetes（容器环境）的迁移，更加平滑。而如果你能把 Pod 看成传统环境里的“机器”、把容器看作是运行在这个“机器”里的“用户程序”，那么很多关于 Pod 对象的设计就非常容易理解了。 比如，**凡是调度、网络、存储，以及安全相关的属性，基本上是 Pod 级别的**。这些属性的共同特征是，它们描述的是“机器”这个整体，而不是“机器”里的“用户程序”。


	apiVersion: v1
	kind: Pod...
	spec: 
		nodeSelector:
		hostAliases:
		containers:
			- name:
			  image:
			  lifecycle: 
			  	postStart: 
			  		exec: 
			  			command: ["/bin/sh","-c","echo hello world"]
			  	preStop:
			  		...

		
可以观察这些配置的位置，Pod的归Pod，容器的归容器。	
#### Pod configuration file

A pod configuration file specifies required information about the pod/ It can be formatted as YAML or as JSON, and supports the following fields:

    {
      "id": string,
      "kind": "Pod",
      "apiVersion": "v1beta1",
      "desiredState": {
        "manifest": {
          manifest object
        }
      },
      "labels": { string: string }
    }
    
Required fields are:

- id: The name of this pod. It must be an RFC1035 compatible value and be unique on this container cluster.
- kind: Always Pod.
- apiVersion: Currently v1beta1.
- desiredState: The configuration for this pod. It must contain a child manifest object.

Optional fields are:

- labels are arbitrary key:value pairs that **can be used by replication controllers and services for grouping and targeting pods**.

#### Manifest

Manifest部分的内容不再赘述（所包含字段，是否必须，以及其意义），可以参见文档

#### Sample file

    {
      "id": "redis-controller",
      "kind": "Pod",
      "apiVersion": "v1beta1",
      "desiredState": {
        "manifest": {
          "version": "v1beta1",
          "containers": [{
            "name": "redis",
            "image": "dockerfile/redis",
            "ports": [{
              "containerPort": 6379,
              "hostPort": 6379
            }]
          }]
        }
      },
      "labels": {
        "name": "redis-controller"
      }
    }

### Viewing a pod

    kubectl get pod xxx
    ## list pod
    kubectl get pods

### Deleting a pod

    kubectl delete pod xxx
    
## Pod 的生命周期

1. pod的生命周期
2. container的生命周期
3. pod restartPolicy
4. pod livenessProbe


restartPolicy 和 Pod 里容器的状态，以及Pod 状态的对应关系（最终体现在`kube get pod pod_name` 时 status 的状态） [有一系列复杂的情况](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#example-states) ，可以概括为两条基本原则：

1. 只要 Pod 的 restartPolicy 指定的策略允许重启异常的容器（比如：Always），那么这个 Pod 就会保持 Running 状态，并进行容器重启。否则，Pod 就会进入 Failed 状态 。
2. 对于包含多个容器的 Pod，只有它里面所有的容器都进入异常状态后，Pod 才会进入 Failed 状态。在此之前，Pod都是 Running 状态。此时，Pod 的 READY 字段会显示正常容器的个数



