---

layout: post
title: k8s设备管理
category: 架构
tags: Kubernetes
keywords:  kubelet 组件分析

---

## 简介（未完成）

* TOC
{:toc}

[Koordinator 异构资源/任务调度实践](https://mp.weixin.qq.com/s/qcJeFqiUs1QxrKETpoMe8Q)K8s 是通过 kubelet 负责设备管理和分配，并和 device plugin 交互实现整套机制，这套机制在 K8s 早期还是够用的，但也有局限性
1. K8s 只允许通过 kubelet 来分配设备，这样就导致无法获得全局最优的资源编排，也就是从根本上无法发挥资源效能。比如一个集群内有两台节点，都有相同的设备，剩余的可分配设备数量相等，但是实际上两个节点上的设备所在的硬件拓扑会导致 Pod 的运行时效果差异较大，没有调度器介入的情况下，是可能无法抵消这种差异的。
2. 不支持类似 GPU 和 RDMA 联合分配的能力。大模型训练依赖高性能网络，而高性能网络的节点间通信需要用到 RDMA 协议和支持 RDMA 协议的网络设备，而这些设备又和 GPU 在节点上的系统拓扑层面是紧密协作的，这就要求**在分配 GPU 和 RDMA 时需要感知硬件拓扑**，尽可能就近分配这种设备。尝试按照同 PCIe，同 NUMA Node，同 NUMA Socket 和 跨 NUMA 的顺序分配，延迟依次升高。
3.  kubelet 不支持设备的初始化和清理功能，更不支持设备的共享机制，后者在训练场景一般用不到，但在线推理服务会用到。在线推理服务本身也有很明显的峰谷特征，很多时刻并不需要占用完整的 GPU 资源。

K8s 这种节点的设备管理能力一定程度上已经落后时代了，虽然现在最新的版本里支持了 DRA 分配机制（类似于已有的 PVC 调度机制），但是这个机制首先只在最新版本的 K8s 才支持。

koord-scheduler 调度器根据 koordlet 上报的 Device CRD 分配设备（Koordinator Device CRD 用来描述节点的设备信息，包括 Device 的拓扑信息，这些信息可以指导调度器实现精细化的分配逻辑），并写入到 Pod Annotation 中，，再经 kubelet 拉起 Sandbox 和 Container，这中间 kubelet 会发起 CRI 请求到 containerd/docker。在 Koordinator 方案中，CRI 请求会被 koord-runtime-proxy 拦截并转发到 koordlet 内的 GPU 插件，感知 Pod Annotation 上的设备分配结果并生成必要的设备环境变量等信息返回给 koord-runtime-proxy，再最终把修改后的 CRI 请求转给 containerd/docker，最终再返回给 kubelet。这样就可以无缝的介入整个容器的生命周期实现自定义的逻辑。