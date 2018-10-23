---

layout: post
title: 《Container-Networking-Docker-Kubernetes》笔记
category: 技术
tags: Docker
keywords: Container-Networking-Docker-Kubernetes

---


## 简介（未完成）

Nginx 公司的 Michael Hausenblas 发布了一本关于 docker 和 kubernetes 中的容器网络的小册子

容器网络仍然非常年轻，年轻就意味着多变，笔者到目前为止总结几套方案都落伍了， 这更加需要我们对容器网络有一个梳理。

service discovery and container orchestration are two sides of the same idea.


## Pets vs Cattle

[DevOps Concepts: Pets vs Cattle](https://medium.com/@Joachim8675309/devops-concepts-pets-vs-cattle-2380b5aab313)


想让服务可靠，有两种方式：把机器搞可靠；部署多个实例。

||特点|详情| Examples |
|---|---|---|---|
|Pets|scale up|you trait the machines as individuals,you gave each (virtual)machine a name. when a machine gets ill you nurse it back to health and manually redeploy the app. | mainframes, solitary servers, load balancers and firewalls, database systems, and so on.|
|Cattle|scale out|your machines are anonymous;they are all identical,they have numbers rather than names, and apps are automatically deployed onto any and each of the machines. when one of the machines gets ill, you don't worry about it immediately|web server arrays, no-sql clusters, queuing cluster, search cluster, caching reverse proxy cluster, multi-master datastores like Cassandra, big-data cluster solutions, and so on.|


PS：the Cloud Age, virtualized servers that are programmable through a web interface。

一个服务部署多个（几十个/上百个）实例带来许多挑战：如何部署？如何发现？挂了怎么办（总不能还靠人工）？通常依靠一个资源管理和调度平台辅助管理，如何选用和部署这个调度平台？从 "Evolution of Cattle" 的视角来看待 运维技术的演进。

||描述| technologies |部署cattle service需要什么|备注|
|---|---|---|---|---|
|Iron Age|物理机||Robust change configuration tools like Puppet (2005), CFEngine 3 (2008), and Chef|
|The First Cloud Age|IaaS that virtualized the entire infrastructure (networks, storage, memory, cpu) into programmable resources. |Amazon Web Services (2006), Microsoft Azure (2010), Google Cloud Platform |push-based orchestration tools like Salt Stack (2011), Ansible (2012), and Terraform (2014). |
|The Second Cloud Age|virtualize aspects of the infrastructure,This allows applications to be segregated into their own isolated environment without the need to virtualize hardware, which in turn duplicates the operating system per application. |OpenVZ (2005), Linux Containers or LXC (2008), and Docker (2015).|A new set of technologies evolved to allocate resources for containers and schedule these containers across a cluster of servers:Apache Mesos (2009), Kubernetes (2014), Nomad (2015), Swarm | Immutable Production(应用的每一次更改都是重新部署，所以本身是Immutable),disposable containers are configured at deployment. 容器在部署的时候被配置|

## container networking stack


|分层|包括哪些|作用|
|---|---|---|
|the low-level networking layer|networking gear(网络设备),iptables,routing,ipvlan,linux namespaces|这些技术已经存在很多年，我们只是对它们的使用|
|the container networking layer|single-host bridge mode,multi-host,ip-per-container|对底层技术provide some abstractions|
|the container orchestration layer|service discovery,loadbalance,cni,kubernetes networking|marrying the container scheduler's decisions on where to place a container with the primitives provided by lower layers. 重要的事情读三遍：根据调度系统的决定，使用lower layer提供的操作place a container|


## 单机

除了四种网络模型之外，还有administrative point of view

1. allocating ip addresses, 手动分配是不现实的，此外，**to prevent arp collisions on a local network, the docker daemon generates a mac address from the allocated ip address** 
2. managing ports

	* fixed allocation
	* dynamic allocation
	* ip-per-container，就没有端口分配的问题了

3. network security

ip-per-container 是网络方案中的一种，不要用习惯了，就以为只有这一种方式。

## 多机

VLAN 技术主要就是在二层数据包的包头加上tag 标签，表示当前数据包归属的vlan 号

1. 广播域被限制在一个VLAN内,节省了带宽,提高了网络处理能力
2. 增强局域网的安全性:VLAN间不能直接通信,即一个VLAN内的用户不能和其它VLAN内的用户直接通信,而需要通过路由器或三层交换机等三层设备。
3. 灵活构建虚拟工作组:用VLAN可以划分不同的用户到不同的工作组,同一工作组的用户也不必局限于某一固定的物理范围

[Macvlan and IPvlan basics](https://sreeninet.wordpress.com/2016/05/29/macvlan-and-ipvlan/)

1. Macvlan and ipvlan are Linux network drivers that exposes underlay or host interfaces directly to VMs or Containers running in the host. 

there are two ways for Containers or VMs to communicate to each other. 

1. In Underlay network approach, VMs or Containers are directly exposed to host network. Bridge, macvlan and ipvlan network drivers are examples of this approach. 
2. In Overlay network approach, there is an additional level of encapsulation like VXLAN, NVGRE


||特点||
|---|---|---|
|vlan|each sub-interface belongs to a different L2 domain using vlan and all sub-interfaces have same mac address.||
|Macvlan|Macvlan allows a single physical interface to have multiple mac and ip addresses using macvlan sub-interfaces. <br>Containers will directly get exposed in underlay network using Macvlan sub-interfaces.<br> Macvlan has 4 types(Private, VEPA, Bridge, Passthru)<br> 可以在vlan sub-interface 上创建 macvlan subinterface||
|ipvlan|||
|vxlan|||

容器的网卡 来自哪里？
真正与外界通信的网卡是哪个？


![](/public/upload/docker/container_framework.png)

除了各种方案，还有administrative point of view

1. ipvlan
