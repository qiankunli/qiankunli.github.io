---

layout: post
title: 《Container-Networking-Docker-Kubernetes》笔记
category: 技术
tags: Docker
keywords: Container-Networking-Docker-Kubernetes

---


## 简介（持续更新）

Nginx 公司的 Michael Hausenblas 发布了一本关于 docker 和 kubernetes 中的容器网络的小册子

容器网络仍然非常年轻，年轻就意味着多变，笔者之前博客总结几套方案都落伍了， 这更加需要我们对容器网络有一个梳理。

service discovery and container orchestration are two sides of the same idea.


## Pets vs Cattle

[DevOps Concepts: Pets vs Cattle](https://medium.com/@Joachim8675309/devops-concepts-pets-vs-cattle-2380b5aab313)


想让服务可靠，有两种方式：把机器搞可靠；部署多个实例。

||特点|详情| Examples |
|---|---|---|---|
|Pets|scale up|you trait the machines as individuals,you gave each (virtual)machine a name. when a machine gets ill you nurse it back to health and manually redeploy the app. | mainframes, solitary servers, load balancers and firewalls, database systems, and so on.|
|Cattle|scale out|your machines are anonymous;they are all identical,they have numbers rather than names, and apps are automatically deployed onto any and each of the machines. when one of the machines gets ill, you don't worry about it immediately|web server arrays, no-sql clusters, queuing cluster, search cluster, caching reverse proxy cluster, multi-master datastores like Cassandra, big-data cluster solutions, and so on.|


PS：the Cloud Age, virtualized servers that are programmable through a web interface。

with the cattle approach to managing infrastructure,you don't manually allocate certain machines for running an application.Instead,you leave it up to an orchestrator to manage the life cycle of your containers. 一个服务部署多个（几十个/上百个）实例带来许多挑战：如何部署？如何发现？挂了怎么办（总不能还靠人工）？通常依靠一个资源管理和调度平台辅助管理，如何选用和部署这个调度平台？从 "Evolution of Cattle" 的视角来看待 运维技术的演进。

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

[知乎：VXLAN vs VLAN](https://zhuanlan.zhihu.com/p/36165475)

[Virtual Extensible LAN](https://en.wikipedia.org/wiki/Virtual_Extensible_LAN)


1. 容器的一个诉求：To make a VM mobile you want to be able to move it's physical location without changing it's apparent network location.
2. vlan 和vxlan 都是 virtual lan(局域网)，但vlan 是隔离出来的，借助了交换机的支持（或者说推出太久了，以至于交换机普遍支持），vxlan 是虚拟出来的，交换机无感知。这种视角，有点像docker 与传统虚拟机的区别，隔离的好处是轻量也受交换机相关特性的限制（比如mac地址表上限）。虚拟的好处是灵活度高，但需要专门的中间组件。
3. VXLAN与VLAN的最大区别在于，VLAN只是修改了原始的Ethernet Header，但是整个网络数据包还是原来那个数据包，而VXLAN是将原始的Ethernet Frame隐藏在UDP数据里面。经过VTEP封装之后，在网络线路上看起来只有VTEP之间的UDP数据传递，原始的网络数据包被掩盖了。
4. 为什么构建数据中心用VXLAN？

	* VXLAN evolved as a Data Center technology，所以分析vxlan 优势时一切以 数据中心的需求为出发点。一个超大型数据中心，交换机怎么联都是有技术含量的 [What is a Networking Switch Fabric](https://www.sdxcentral.com/sdn/definitions/what-is-networking-switch-fabric/)
	* vlan 4096 数量限制 不是问题
	* TOR（Top Of Rack）交换机MAC地址表限制。数据中心的虚拟化给网络设备带来的最直接影响就是：之前TOR（Top Of Rack）交换机的一个端口连接一个物理主机对应一个MAC地址，但现在交换机的一个端口虽然还是连接一个物理主机但是可能进而连接几十个甚至上百个虚拟机和相应数量的MAC地址。
	* 待补充
	* VTEP 在微服务领域有点像现在的service mesh，一个vm/container 是一个微服务，微服务只需和sevice mesh sidecar 沟通


[Macvlan and IPvlan basics](https://sreeninet.wordpress.com/2016/05/29/macvlan-and-ipvlan/)

 Macvlan and ipvlan are Linux network drivers that exposes underlay or host interfaces directly to VMs or Containers running in the host. 


||特点|ip/mac address|从交换机的视角看vlan方案|
|---|---|---|---|
|vlan|A virtual LAN (VLAN) is any broadcast domain that is partitioned and isolated in a computer network at the data link layer (OSI layer 2).<br>each sub-interface belongs to a different L2 domain using vlan |all sub-interfaces have same mac address.|交换机要支持 vlan tag|
|Macvlan|Containers will directly get exposed in underlay network using Macvlan sub-interfaces.<br> Macvlan has 4 types(Private, VEPA, Bridge, Passthru)<br> 可以在vlan sub-interface 上创建 macvlan subinterface|Macvlan allows a single physical interface to have multiple mac and ip addresses using macvlan sub-interfaces. <br>|交换机的port一般只与一个mac绑定，使用macvlan 后必须支持绑定多个 且 无数量限制|
|ipvlan|  ipvlan supports L2 and L3 mode.|the endpoints have the same mac address|省mac地址|
|vxlan|Virtual Extensible LAN (VXLAN) is a network virtualization technology that attempts to address the scalability problems associated with large cloud computing deployments. <br>VXLAN endpoints, which terminate VXLAN tunnels and may be either virtual or physical switch ports, are known as VXLAN tunnel endpoints (VTEPs)||交换机无感知|


## 给所有网络方案归个类

there are two ways for Containers or VMs to communicate to each other. 

1. In Underlay network approach, VMs or Containers are directly exposed to host network. Bridge, macvlan and ipvlan network drivers are examples of this approach. 
2. In Overlay network approach, there is an additional level of encapsulation like VXLAN, NVGRE

||容器的网卡 来自哪里？|真正与外界通信的网卡是哪个？ external connectivity|容器与物理机网卡的关系及数据连通|
|---|---|---|---|
|bridge|veth|物理机网卡|veth pair 挂在bridge上，NAT 连通 物理机网卡|
|macvlan|macvlan sub-interfaces|macvlan sub-interfaces|
|ipvlan|ipvlan sub-interfaces|ipvlan sub-interfaces|
|calico|veth|物理机网卡|host 侧的veth 与 host eth建立路由|

该表格持续更新中

![](/public/upload/docker/container_networking.png)

bridge 方案加上隧道 就是 vxlan，加上路由方案就是 calico

[容器网络：盘点，解释与分析](http://www.dockerinfo.net/4289.html)

这个归类不好做，有好几个方面：

1. 支持量级的大小
2. 拆封包还是路由
3. 交换机是否有感知
4. 容器所在物理机/宿主机处于一个什么样的角色
5. 是一个L3方案（只要物理机来通就行）还是L2方案，L3网络扩展和提供在过滤和隔离网络流量方面的细粒度控制。
6. 选择网络时，IP地址管理IPAM，组播，广播，IPv6，负载均衡，服务发现，策略，服务质量，高级过滤和性能都是需要额外考虑的。问题是这些能力是否受到支持。即使您的runtime，编排引擎或插件支持容器网络功能，您的基础架构也可能不支持该功能

## container orchestrator

一般orchestrator 包括但不限于以下功能：

1. Organizational primitives，比如k8s的label
2. Scheduling of containers to run on a ost
3. Automated health checks to determine if a container is alive and ready to serve traffic and to relaunch it if necessary
4. autoscaling 
5. upgrading strategies,from rolling updates to more sophisticated techniques such as A/B and canary deployments.
6. service discovery to determine which host a scheduled container ended upon,usually including DNS support.

## CNI

The cni specification is lightweight; it only deals with the network connectivity of containers,as well as the garbage collection of resources once containers are deleted.



cni 接口规范，不是很长[Container Network Interface Specification](https://github.com/containernetworking/cni/blob/master/SPEC.md)，原来技术的世界里很多规范用Specification 来描述。

![](/public/upload/docker/cni_2.png)


![](/public/upload/docker/cni_3.png)

对 CNI SPEC 的解读 [Understanding CNI (Container Networking Interface)](http://www.dasblinkenlichten.com/understanding-cni-container-networking-interface/)

1. If you’re used to dealing with Docker this doesn’t quite seem to fit the mold. 习惯了docker 之后， 再看cni 有点别扭。原因就在于，docker 类似于操作系统领域的windows，把很多事情都固化、隐藏掉了，以至于认为docker 才是标准。
2. The CNI plugin is responsible wiring up the container.  That is – it needs to do all the work to get the container on the network.  In Docker, this would include connecting the container network namespace back to the host somehow. 在cni 的世界里，container刚开始时没有网络的，是container runtime 操作cni plugin 将container add 到 network 中。


### "裸机" 使用cni

[Understanding CNI (Container Networking Interface)](http://www.dasblinkenlichten.com/understanding-cni-container-networking-interface/)

	mkdir cni
	user@ubuntu-1:~$ cd cni
	user@ubuntu-1:~/cni$ curl -O -L https://github.com/containernetworking/cni/releases/download/v0.4.0/cni-amd64-v0.4.0.tgz
	user@ubuntu-1:~/cni$ tar -xzvf cni-amd64-v0.4.0.tgz
	user@ubuntu-1:~/cni$ ls
	bridge  cni-amd64-v0.4.0.tgz  cnitool  dhcp  flannel  host-local  ipvlan  loopback  macvlan  noop  ptp  tuning

创建一个命名空间

	sudo ip netns add 1234567890

调用cni plugin将 container（也就是network namespace） ADD 到 network 上

	sudo CNI_COMMAND=ADD CNI_CONTAINERID=1234567890 CNI_NETNS=/var/run/netns/1234567890 CNI_IFNAME=eth12 CNI_PATH=`pwd` ./bridge <mybridge.conf

**从这也可以看到，前文画了图，整理了脑图，但资料看再多，都不如实操案例来的深刻。才能不仅让你“理性”懂了，也能让你“感性”懂了**

### 在container runtime 下使用cni


[Using CNI with Docker](http://www.dasblinkenlichten.com/using-cni-docker/) 未读