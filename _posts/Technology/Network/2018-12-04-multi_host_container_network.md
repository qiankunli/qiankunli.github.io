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

|网络方案|网络上跑的什么包|辅助机制|优缺点|
|---|---|---|---|
|基于docker单机|NAT包/目的ip变为物理机|
|overlay+隧道|封装后的udp包|解封包进程<br>就像运行vpn要启动一个进程才能登陆公司内网一样|不依赖底层网络<br>就像你的电脑连上网就可以连公司的vpn|
|overlay+路由|主机路由后的包/目的mac变为物理机<br>容器所在物理机上一堆路由表（主要是直接路由）让物理机变为了一个路由器|直接路由机制需要二层连通|
|underlay|原始数据包|linux相关网络驱动<br>网络设备支持|性能最好<br>强依赖底层网络|

笔者最惊喜的就是以“网络上跑的什么包” 来切入点来梳理 容器网络模型。

## docker 单机

![](/public/upload/network/container_network_host.png)

[Kubernetes 网络模型进阶](https://mp.weixin.qq.com/s/QZYpV7ks2xxNvtqWUiS7QQ)容器网络发端于 Docker 的网络。Docker 使用了一个比较简单的网络模型，即内部的网桥加内部的保留 IP。这种设计的好处在于容器的网络和外部世界是解耦的，无需占用宿主机的 IP 或者宿主机的资源，完全是虚拟的。

它的设计初衷是：当需要访问外部世界时，会采用 SNAT 这种方法来借用 Node 的 IP 去访问外面的服务。比如容器需要对外提供服务的时候，所用的是 DNAT 技术，也就是在 Node 上开一个端口，然后通过 iptable 或者别的某些机制，把流导入到容器的进程上以达到目的。简称：出宿主机采用SNAT借IP，进宿主机用DNAT借端口。

该模型的问题在于：**网络中一堆的NAT 包，外部网络无法区分哪些是容器的网络与流量、哪些是宿主机的网络与流量**。

由于网桥是虚拟的二层设备，同节点的 Pod 之间通信直接走二层转发，跨节点通信才会经过宿主机 eth0

## Kubernetes IP-per-pod model

Kubernetes 网络大致分为两大类，使用不同的技术
1. 一类是 Cluster IP，是一层反向代理的虚拟ip；service/ingress，早期 kube-proxy 是采用 iptables，后来引入 IPVS 也解决了大规模容器集群的网络编排的性能问题。
2. 一类是 Pod IP，容器间交互数据

[深入理解 Kubernetes 网络模型 - 自己实现 kube-proxy 的功能](https://mp.weixin.qq.com/s/zWH5gAWpeAGie9hMrGscEg)主机A上的实例(容器、VM等)如何与主机B上的另一个实例通信?有很多解决方案:

1. 直接路由: BGP等
2. 隧道: VxLAN, IPIP, GRE等
3. NAT: 例如docker的桥接网络模式
4. 其它方式

针对docker 跨主机通信时网络中一堆的NAT包，Kubernetes 提出IP-per-pod model ，这个 IP 是真正属于该 Pod 的，对这个 Pod IP 的访问就是真正对它的服务的访问，中间拒绝任何的变造。比如以 10.1.1.1 的 IP 去访问 10.1.2.1 的 Pod，结果到了 10.1.2.1 上发现，它实际上借用的是宿主机的 IP，而不是源 IP，这样是不被允许的。**在通信的两端Pod看来，以及整个通信链路中`<source ip,source port,dest ip,dest port>` 是不能改变的**。设计这个原则的原因是，用户不需要额外考虑如何建立Pod之间的连接，也不需要考虑如何将容器端口映射到主机端口等问题。

Kubernetes 对怎么实现这个模型其实是没有什么限制的，用 underlay 网络来**控制外部路由器进行导流**是可以的；如果希望解耦，用 overlay 网络在底层网络之上再加一层叠加网，这样也是可以的。总之，只要达到模型所要求的目的即可。**因为`<source ip,source port,dest ip,dest port>`不能变，排除NAT/DAT，其实也就只剩下路由和解封包两个办法了**。

Rather than prescribing a certain networking solution, Kubernetes only states three fundamental requirements:

* Containers can communicate with all other containers without NAT.
* Nodes can communicate with all containers (and vice versa) without NAT.
* The IP a container sees itself is the same IP as others see it. each pod has its own IP address that other pods can find and use. 很多业务启动时会将自己的ip 发出去（比如注册到配置中心），这个ip必须是外界可访问的。 学名叫：flat address space across the cluster.


Kubernetes requires each pod to have an IP in a flat networking namespace with full connectivity to other nodes and pods across the network. This IP-per-pod model yields a backward-compatible way for you to treat a pod almost identically to a VM or a physical host（**ip-per-pod 的优势**）, in the context of naming, service discovery, or port allocations. The model allows for a smoother transition from non–cloud native apps and environments.  这样就 no need to manage port allocation

## 跨主机通信

[CNI 网络方案优缺点及最终选择](https://mp.weixin.qq.com/s/pPrA_5BaYG9AwYNy4n_gKg)

2020.4.18补充：很多文章都是从跨主机容器如何通信 的来阐述网络方案，这或许是一个很不好的理解曲线，从实际来说，一定是先有网络，再为Pod “连上网”。

Network是一组可以相互通信的Endpoints，网络提供connectivity and discoverability.

there are two ways for Containers or VMs to communicate to each other. 

1. In Underlay network approach, VMs or Containers are directly exposed to host network. Bridge, macvlan and ipvlan network drivers are examples of this approach. 
2. In Overlay network approach, there is an additional level of encapsulation like VXLAN, NVGRE between the Container/VM network and the underlay network

一个容器的包所要解决的问题分为两步：第一步，如何从容器的空间 (c1) 跳到宿主机的空间 (infra)；第二步，如何从宿主机空间到达远端。容器网络的方案可以通过接入、流控、通道这三个层面来考虑。

### 从容器的空间 跳到宿主机的空间

![](/public/upload/container/container_host.jpeg)

接入，就是说我们的容器和宿主机之间是使用哪一种机制做连
1. Veth + bridge、Veth + pair 这样的经典方式
2. 利用高版本内核的新机制等其他方式（如 mac/IPvlan 等），来把包送入到宿主机空间；

数据包到了 Host Network Namespace 之后呢，怎么把它从宿主机上的 eth0 发送出去?
1. nat
2. 建立 Overlay 网络发送
3. 通过配置 proxy arp 加路由的方法来实现。

### 宿主机之间

1. 流控，就是说我的这个方案要不要支持 Network Policy，如果支持的话又要用何种方式去实现。这里需要注意的是，我们的实现方式一定需要在数据路径必经的一个关节点上。如果数据路径不通过该 Hook 点，那就不会起作用；
2. 通道，即两个主机之间通过什么方式完成包的传输。我们有很多种方式，比如以路由的方式，具体又可分为 **BGP 路由**或者**直接路由**。还有各种各样的**隧道技术**等等。最终我们实现的目的就是一个容器内的包通过容器，经过接入层传到宿主机，再穿越宿主机的流控模块（如果有）到达通道送到对端。

## overlay / underlay

[理解 CNI 和 CNI 插件](https://mp.weixin.qq.com/s/g3QECjZOgbEZ8FG9R3b9iw)

![](/public/upload/network/container_network.png)

### overlay网络

overlay 网络主要有隧道 和 路由两种方式

1. 容器网卡不能直接发送/接收数据，而要通过双方容器所在宿主机网卡发送/接收数据
2. “容器在哪个主机上“ 这个信息都必须专门维护。[kubectl 创建 Pod 背后到底发生了什么？](https://mp.weixin.qq.com/s/ctdvbasKE-vpLRxDJjwVMw)**overlay 网络是一种动态同步多个主机间路由的方法**。
3. 容器内数据包必须先发送目标容器所在的宿主机上，那么容器内原生的数据包便要进行改造（解封包或根据路由更改目标mac）
4. 数据包到达目标宿主机上之后，目标宿主机要进行一定的操作转发到目标容器。

覆盖网络如何解决connectivity and discoverability？connectivity由物理机之间解决，discoverability由**容器在物理机侧的veth** 与 宿主机eth 之间解决，一般由主机上网络协议栈具体负责（**一般网络组件除解封包外，不参与通信过程，只是负责向网络协议栈写入routes和iptables**）。

![](/public/upload/docker/overlay_network_2.png)

### 封包方式

![](/public/upload/network/container_network_vpn.png)

隧道一般用到了解封包，那么问题来了，谁来解封包？怎么解封包？

|overlay network|包格式|解封包设备|要求|
|---|---|---|---|---|
|flannel + udp|ip数据包 package|flanneld 更新路由 + tun 解封包|宿主机三层连通|
|flannel + vxlan|二层数据帧 Ethernet frame|flanneld 更新路由 + VTEP解封包|宿主机二层互通|
|calico + ipip|ip数据包 package|bgp 更新路由 + tunl0 解封包|宿主机三层连通|

flannel + udp 和flannel + vxlan 有一个共性，那就是用户的容器都连接在 docker0网桥上。而网络插件则在宿主机上创建了一个特殊的设备（UDP 模式创建的是 TUN 设备，VXLAN 模式创建的则是 VTEP 设备），docker0 与这个设备之间，通过 IP 转发（路由表）进行协作。然后，**网络插件真正要做的事情，则是通过某种方法，把不同宿主机上的特殊设备连通，从而达到容器跨主机通信的目的。**

Flannel 支持三种后端实现，分别是： VXLAN；host-gw； UDP。而 UDP 模式，是 Flannel 项目最早支持的一种方式，也是性能最差的一种方式。所以，这个模式目前已经被弃用。我们在进行系统级编程的时候，有一个非常重要的优化原则，**就是要减少用户态到内核态的切换次数，并且把核心的处理逻辑都放在内核态进行**。这也是为什么，Flannel 后来支持的VXLAN 模式，逐渐成为了主流的容器网络方案的原因。

### 路由方式

![](/public/upload/network/container_network_route_2.png)

容器互通主要基于路由表打通，**一般配套设计是 一个物理机对应一个网段**，路由方案的关键是谁来路由？路由信息怎么感知？路由信息存哪？Kubernetes/etcd/每个主机bgp分发都来一份。calico 容器在**主机内外**都通过 路由规则连通（主机内不会创建网桥设备）；flannel host-gw 主机外靠路由连通，主机内靠网桥连通。

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

容器所在主机的路由表 让linux 主机变成了一个路由器，路由表主要由**直接路由**构成，将数据包中目的mac 改为直接路由中下一跳主机 的mac 地址。 

![](/public/upload/network/container_network_route.png)

### underlay/physical 网络

[容器网络：盘点，解释与分析](http://www.dockerinfo.net/4289.html)

||容器的网卡 来自哪里？|真正与外界通信的网卡是哪个？ external connectivity|容器与物理机网卡的关系及数据连通|
|---|---|---|---|
|bridge|veth|物理机网卡|veth pair 挂在bridge上，NAT 连通 物理机网卡|
|macvlan|macvlan sub-interfaces|macvlan sub-interfaces|
|ipvlan|ipvlan sub-interfaces|ipvlan sub-interfaces|

[阿里云如何构建高性能云原生容器网络？](https://mp.weixin.qq.com/s/tAlEtCap6bvv6-N96sKJjw)云原生容器网络是直接使用云上原生云资源配置容器网络：

1. 容器和节点同等网络平面，同等网络地位；
2. Pod 网络可以和云产品无缝整合；
3. 不需要封包和路由，网络性能和虚机几乎一致。

![](/public/upload/network/container_network_cloud.png)

## 网络隔离

Kubernetes 对 Pod 进行“隔离”的手段，即：NetworkPolicy，NetworkPolicy 实际上只是宿主机上的一系列 iptables 规则。在具体实现上，凡是支持 NetworkPolicy 的 CNI 网络插件，都维护着一个 NetworkPolicy Controller，通过控制循环的方式对 NetworkPolicy 对象的增删改查做出响应，然后在宿主机上完成 iptables 规则的配置工作。

## Cilium 

[Cilium 容器网络的落地实践](https://mp.weixin.qq.com/s/3B1JZVpS8NI1ESkTp-PHKg)  服务网格和无服务器等新技术对 Kubernetes 底层提出了更多的定制化要求。这些新需求都有一些共同点：它们需要一个更可编程的数据平面（也就是agent），能够在不牺牲性能的情况下执行 Kubernetes 感知的网络数据操作。Cilium 项目通过引入扩展的伯克利数据包过滤器（eBPF）技术，在 Linux 内核内向网络栈暴露了可编程的钩子。使得网格数据包不需要在用户和内核空间之间来回切换就可以通过上下文快速进行数据交换操作。PS： envoy 运行在用户态，Cilium 能将所有的逻辑下沉到内核。这是一种新型的网络范式，它也是 Cilium 容器网络项目的核心思想。

一种技术满足所有的 网络需求
1. service/ingress 提供一个虚拟ip，负载均衡，访问多个pod
2. 容器间网络通信
2. 网络隔离
3. metric 监控，数据可视化
4. trace 跟踪

![](/public/upload/network/cilium_network.png)
