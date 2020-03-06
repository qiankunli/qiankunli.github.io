---

layout: post
title: 跨主机容器通信
category: 技术
tags: Network
keywords: container network

---

## 简介

* TOC
{:toc}

容器的一个诉求：To make a VM mobile you want to be able to move it's physical location without changing it's apparent network location.

## docker 单机

[Kubernetes 网络模型进阶](https://mp.weixin.qq.com/s/QZYpV7ks2xxNvtqWUiS7QQ)容器网络发端于 Docker 的网络。Docker 使用了一个比较简单的网络模型，即内部的网桥加内部的保留 IP。这种设计的好处在于容器的网络和外部世界是解耦的，无需占用宿主机的 IP 或者宿主机的资源，完全是虚拟的。

它的设计初衷是：当需要访问外部世界时，会采用 SNAT 这种方法来借用 Node 的 IP 去访问外面的服务。比如容器需要对外提供服务的时候，所用的是 DNAT 技术，也就是在 Node 上开一个端口，然后通过 iptable 或者别的某些机制，把流导入到容器的进程上以达到目的。简称：出宿主机采用SNAT借IP，进宿主机用DNAT借端口。

该模型的问题在于：**网络中一堆的NAT 包，外部网络无法区分哪些是容器的网络与流量、哪些是宿主机的网络与流量**。

## Kubernetes IP-per-pod model

针对docker 跨主机通信时网络中一堆的NAT包，Kubernetes 提出IP-per-pod model ，这个 IP 是真正属于该 Pod 的，对这个 Pod IP 的访问就是真正对它的服务的访问，中间拒绝任何的变造。比如以 10.1.1.1 的 IP 去访问 10.1.2.1 的 Pod，结果到了 10.1.2.1 上发现，它实际上借用的是宿主机的 IP，而不是源 IP，这样是不被允许的。

Kubernetes 对怎么实现这个模型其实是没有什么限制的，用 underlay 网络来**控制外部路由器进行导流**是可以的；如果希望解耦，用 overlay 网络在底层网络之上再加一层叠加网，这样也是可以的。总之，只要达到模型所要求的目的即可。

Rather than prescribing a certain networking solution, Kubernetes only states three fundamental requirements:

* Containers can communicate with all other containers without NAT.
* Nodes can communicate with all containers (and vice versa) without NAT.
* The IP a container sees itself is the same IP as others see it. each pod has its own IP address that other pods can find and use. 很多业务启动时会将自己的ip 发出去（比如注册到配置中心），这个ip必须是外界可访问的。 学名叫：flat address space across the cluster.


Kubernetes requires each pod to have an IP in a flat networking namespace with full connectivity to other nodes and pods across the network. This IP-per-pod model yields a backward-compatible way for you to treat a pod almost identically to a VM or a physical host（**ip-per-pod 的优势**）, in the context of naming, service discovery, or port allocations. The model allows for a smoother transition from non–cloud native apps and environments.  这样就 no need to manage port allocation

## 跨主机通信

Network是一组可以相互通信的Endpoints，网络提供connectivity and discoverability.

there are two ways for Containers or VMs to communicate to each other. 

1. In Underlay network approach, VMs or Containers are directly exposed to host network. Bridge, macvlan and ipvlan network drivers are examples of this approach. 
2. In Overlay network approach, there is an additional level of encapsulation like VXLAN, NVGRE between the Container/VM network and the underlay network

一个容器的包所要解决的问题分为两步：第一步，如何从容器的空间 (c1) 跳到宿主机的空间 (infra)；第二步，如何从宿主机空间到达远端。容器网络的方案可以通过接入、流控、通道这三个层面来考虑。

1. 第一个是接入，就是说我们的容器和宿主机之间是使用哪一种机制做连接，比如 Veth + bridge、Veth + pair 这样的经典方式，也有利用高版本内核的新机制等其他方式（如 mac/IPvlan 等），来把包送入到宿主机空间；
2. 第二个是流控，就是说我的这个方案要不要支持 Network Policy，如果支持的话又要用何种方式去实现。这里需要注意的是，我们的实现方式一定需要在数据路径必经的一个关节点上。如果数据路径不通过该 Hook 点，那就不会起作用；
3. 第三个是通道，即两个主机之间通过什么方式完成包的传输。我们有很多种方式，比如以路由的方式，具体又可分为 BGP 路由或者直接路由。还有各种各样的隧道技术等等。最终我们实现的目的就是一个容器内的包通过容器，经过接入层传到宿主机，再穿越宿主机的流控模块（如果有）到达通道送到对端。


### overlay 网络

[kubectl 创建 Pod 背后到底发生了什么？](https://mp.weixin.qq.com/s/ctdvbasKE-vpLRxDJjwVMw)**overlay 网络是一种动态同步多个主机间路由的方法**。

我们找一下覆盖网络的感觉

1. 容器网卡不能直接发送/接收数据，而要通过宿主机网卡发送/接收数据
1. 根据第一点，无论overlay 网络的所有方案，交换机都无法感知到容器的mac地址

基于上述两点，容器内数据包从发送方的角度看，无法直接用目标容器mac发送，便需要先发到目标容器所在的宿主机上。于是：

1. overlay 网络主要有隧道 和 路由两种方式，无论哪种方式，“容器在哪个主机上“ 这个信息都必须专门维护
2. 容器内数据包必须先发送目标容器所在的宿主机上，那么容器内原生的数据包便要进行改造（解封包或根据路由更改目标mac）
3. 数据包到达目标宿主机上之后，目标宿主机要进行一定的操作转发到目标容器。

![](/public/upload/docker/overlay_network_1.png)

覆盖网络如何解决connectivity and discoverability？connectivity由物理机之间解决，discoverability由veth 与 宿主机eth 之间解决，将上图细化一下

![](/public/upload/docker/overlay_network_2.png)

#### 隧道

隧道一般用到了解封包，那么问题来了，谁来解封包？怎么解封包？

|overlay network|包格式|解封包设备|要求|
|---|---|---|---|---|
|flannel + udp|ip数据包 package|flanneld 更新路由 + tun 解封包|宿主机三层连通|
|flannel + vxlan|二层数据帧 Ethernet frame|flanneld 更新路由 + VTEP解封包|宿主机二层互通|
|calico + ipip|ip数据包 package|bgp 更新路由 + tunl0 解封包|宿主机三层连通|

flannel + udp 和flannel + vxlan 有一个共性，那就是用户的容器都连接在 docker0网桥上。而网络插件则在宿主机上创建了一个特殊的设备（UDP 模式创建的是 TUN 设备，VXLAN 模式创建的则是 VTEP 设备），docker0 与这个设备之间，通过 IP 转发（路由表）进行协作。然后，**网络插件真正要做的事情，则是通过某种方法，把不同宿主机上的特殊设备连通，从而达到容器跨主机通信的目的。**

Flannel 支持三种后端实现，分别是： VXLAN；host-gw； UDP。而 UDP 模式，是 Flannel 项目最早支持的一种方式，也是性能最差的一种方式。所以，这个模式目前已经被弃用。我们在进行系统级编程的时候，有一个非常重要的优化原则，**就是要减少用户态到内核态的切换次数，并且把核心的处理逻辑都放在内核态进行**。这也是为什么，Flannel 后来支持的VXLAN 模式，逐渐成为了主流的容器网络方案的原因。

#### 路由

路由方案的关键是谁来路由？路由信息怎么感知？

|overlay network|路由设备|路由更新|要求|
|---|---|---|---|
|flannel + host-gw| 宿主机|flanneld|宿主机二层连通|
|calico + Node-to-Node Mesh |宿主机|bgp 更新路由 |宿主机二层连通|
|calico + 网关bgp |网关|bgp 更新路由 |宿主机三层连通|

1. flannel + udp/flannel + vxlan（tcp数据包），udp 和tcp 数据包首部大致相同


	<table>
	<tr>
		<td colspan="2">frame header</td>
		<td colspan="5">frame body</td>
	</tr>
	<tr>
		<td>host1 mac</td>
		<td>host2 mac</td>
		<td bgcolor="green">container1 mac</td>
		<td bgcolor="green">container2 mac</td>
		<td bgcolor="green">container1 ip</td>
		<td bgcolor="green">container1 ip</td>
		<td bgcolor="green">body</td>
	</tr>
	</table>


2. flannel + host-gw/calico
	
	<table>
	<tr>
		<td colspan="2">frame header</td>
		<td colspan="3">frame body</td>
	</tr>
	<tr>
		<td>container1 mac</td>
		<td>host2 mac</td>
		<td>container1 ip</td>
		<td>container2 ip</td>
		<td>body</td>
	</tr>
	</table>

[A container networking overview](https://jvns.ca/blog/2016/12/22/container-networking/) **How do routes get distributed**?Every container networking thing to runs some kind of **daemon program** on every box which is in charge of adding routes to the route table.

There are two main ways they do it:

1. the routes are in an etcd cluster, and the program talks to the etcd cluster to figure out which routes to set
2. use the BGP protocol to gossip to each other about routes, and a daemon (BIRD) listens for BGP messages on every box

### underlay/physical 网络


||容器的网卡 来自哪里？|真正与外界通信的网卡是哪个？ external connectivity|容器与物理机网卡的关系及数据连通|
|---|---|---|---|
|bridge|veth|物理机网卡|veth pair 挂在bridge上，NAT 连通 物理机网卡|
|macvlan|macvlan sub-interfaces|macvlan sub-interfaces|
|ipvlan|ipvlan sub-interfaces|ipvlan sub-interfaces|

[容器网络：盘点，解释与分析](http://www.dockerinfo.net/4289.html)

## 网络隔离

Kubernetes 对 Pod 进行“隔离”的手段，即：NetworkPolicy，NetworkPolicy 实际上只是宿主机上的一系列 iptables 规则。在具体实现上，凡是支持 NetworkPolicy 的 CNI 网络插件，都维护着一个 NetworkPolicy Controller，通过控制循环的方式对 NetworkPolicy 对象的增删改查做出响应，然后在宿主机上完成 iptables 规则的配置工作。

