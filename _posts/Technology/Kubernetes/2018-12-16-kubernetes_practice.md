---

layout: post
title: kubernetes实践
category: 技术
tags: Kubernetes
keywords: kubernetes yaml

---

## 简介（持续更新）

* TOC
{:toc}

[容器化在一下科技的落地实践](http://www.10tiao.com/html/217/201811/2649699541/1.html)

[荔枝运维平台容器化实践](https://mp.weixin.qq.com/s/Q4t5IptqQmQZ6z4vOIhcjQ) 从打包、监控、日志、网络、存储各方面阐述了一下，还比较全面

## 总纲

两个基本工作

1. 应用容器化

    1. 
    2. 
    3. 
    4. 
2. 编排自动化

    1. 
    2. 
    3. 
    4. 

## 美团实践

[美团点评Kubernetes集群管理实践](https://mp.weixin.qq.com/s/lYDYzEUlvXQhCO1xCJ7HAg) 笔者从中得到一个启发就是，整个kubernetes 的实践是分层次的。

![](/public/upload/kubernetes/meituan_kubernetes_practice.png)

## helm

[Helm安装使用](https://www.qikqiak.com/k8s-book/docs/42.Helm%E5%AE%89%E8%A3%85.html)

## “划一片地”来测试

假设集群一共有10个节点，我们希望一般任务只调度到其中的8多个节点，另外2个节点用来试验一些新特性。

此时便可以采用Taints and Tolerations 特性

### 默认的都匹配 + 特征匹配 node 加label，pod 加nodeSelector/nodeAffinity

[Assigning Pods to Nodes](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#node-affinity-beta-feature)

### 默认的都不匹配 + “特征”匹配 node 加 Taints，pod 加tolerations

Node affinity, described [here](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#node-affinity-beta-feature), is a property of pods that attracts them to a set of nodes (either as a preference or a hard requirement). Taints are the opposite – they allow a node to repel（击退） a set of pods.

[Taints and Tolerations](https://kubernetes.io/docs/concepts/configuration/taint-and-toleration/)

给节点加Taints

    kubectl taint nodes node1 key=value:NoSchedule

 NoSchedule 是一个effect. This means that no pod will be able to schedule onto node1 unless it has a matching toleration.

给pod 加加tolerations

    tolerations:
    - key: "key"
    operator: "Equal"
    value: "value"
    effect: "NoSchedule"

## 另类“Service”——给服务一个稳定的主机名

容器的ip 总是经常变，针对这个问题， k8s早已有一系列解决方案。

1. k8s 套件内

    1. k8s集群内，Service等
    2. k8s集群外，Nodeport，ingress等
2. 提供一个dns服务器，维护`<容器名,ip>` 映射。 实现： codedns + etcd，ip 的变化写入etcd 即可
3. 改写ipam插件，支持静态ip

## 内存突然爆满了

项目日志写满磁盘，k8s 会将pod 从该机器上驱逐出去。k8s驱逐机制（未学习）


## web界面管理

[Qihoo360/wayne](https://github.com/Qihoo360/wayne) Wayne 是一个通用的、基于 Web 的 Kubernetes 多集群管理平台。通过可视化 Kubernetes 对象模板编辑的方式，降低业务接入成本， 拥有完整的权限管理系统，适应多租户场景，是一款适合企业级集群使用的发布平台。

## 容器安全

[绝不避谈 Docker 安全](https://mp.weixin.qq.com/s/IN_JJhg_oG7ILVjNj-UexA?)

![](/public/upload/kubernetes/container_security.png)

[Kubernetic](https://kubernetic.com/)一款kubenretes桌面客户端, Kubernetic uses `~/.kube/config` file to find existing cluster contexts and handle authentication. This means that as soon as you have a kubectl client configured to your machine Kubernetic will be able to login to your configured clusters.

## Garbage Collection

在 Kubernetes 引入垃圾收集器之前，所有的级联删除逻辑都是在客户端完成的，kubectl 会先删除 ReplicaSet 持有的 Pod 再删除 ReplicaSet，但是**垃圾收集器的引入就让级联删除的实现移到了服务端**。

[Garbage Collection](https://kubernetes.io/docs/concepts/workloads/controllers/garbage-collection/)Some Kubernetes objects are owners of other objects. For example, a ReplicaSet is the owner of a set of Pods. The owned objects are called dependents of the owner object. Every dependent object has a metadata.ownerReferences field that points to the owning object.Kubernetes objects 之间有父子关系，那么当删除owners 节点时，如何处理其dependents呢？

1. cascading deletion

    1. Foreground 先删除dependents再删除owners. In foreground cascading deletion, the root object first enters a “deletion in progress” state.Once the “deletion in progress” state is set, the garbage collector deletes the object’s dependents. Once the garbage collector has deleted all “blocking” dependents (objects with ownerReference.blockOwnerDeletion=true), it deletes the owner object.
    2. background 先删owners 后台再慢慢删dependents. Kubernetes deletes the owner object immediately and the garbage collector then deletes the dependents in the background.
2. 不管，此时the dependents are said to be orphaned.

如何控制Garbage Collection？设置propagationPolicy

    kubectl proxy --port=8080
    curl -X DELETE localhost:8080/apis/apps/v1/namespaces/default/replicasets/my-repset \
    -d '{"kind":"DeleteOptions","apiVersion":"v1","propagationPolicy":"Background"}' \
    -H "Content-Type: application/json"
    ## cascade 默认值是true
    kubectl delete replicaset my-repset --cascade=false

### kubelet Garbage Collection

回收物理机上不用的 容器或镜像。

[Configuring kubelet Garbage Collection](https://kubernetes.io/docs/concepts/cluster-administration/kubelet-garbage-collection/)（未读）

1. Image Collection, Disk usage above the HighThresholdPercent will trigger garbage collection. The garbage collection will delete least recently used images until the LowThresholdPercent has been met. `[LowThresholdPercent,HighThresholdPercent]` 大于HighThresholdPercent 开始回收直到 磁盘占用小于LowThresholdPercent
2. Container Collection 核心就是什么时候开始删除容器，什么样的容器可以被删掉

    1. minimum-container-ttl-duration, 容器dead 之后多久可以被删除
    2. maximum-dead-containers-per-container, 每个pod 最多允许的dead 容器数量，超过的容器会被删掉
    3. maximum-dead-containers, 主机上最多允许的dead 容器数量，超过的容器会被删掉

## 工作流

线上环境上线的镜像是已经上线到测试环境的相同镜像。

笔者个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)

