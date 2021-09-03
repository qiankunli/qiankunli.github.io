---

layout: post
title: 虚拟网络
category: 技术
tags: Network
keywords: Docker

---

## 前言


* TOC
{:toc}

建议看下前文 [程序猿视角看网络](http://qiankunli.github.io/2018/03/08/network.html)

不仅数据协议上要有相关字段的体现，操作系统（具体说是网络驱动）和交换机也都要一定的配合。

[Linux虚拟网络技术学习](https://mp.weixin.qq.com/s/2PYds2LDie7W5sXYi1qwew)在Linux虚拟化技术中，网络层面，通常重要的三个技术分别是Network Namespace、veth pair、以及网桥或虚拟交换机技术。

1. 对于每个 Network Namespace 来说，它会有自己独立的网卡、路由表、ARP 表、iptables 等和网络相关的资源。ip命令提供了`ip netns exec`子命令可以在对应的 Network Namespace 中执行命令。PS： 也就是`ip netns exec` 操作网卡、路由表、ARP表、iptables 等。也可以打开一个shell ，后面所有的命令都在这个Network Namespace中执行，好处是不用每次执行命令时都要带上ip netns exec ，缺点是我们无法清楚知道自己当前所在的shell，容易混淆。
2. 默认情况下，network namespace 是不能和主机网络，或者其他 network namespace 通信的。可以使用 Linux 提供的veth pair来完成通信，veth pair你可以理解为使用网线连接好的两个接口，把两个端口放到两个namespace中，那么这两个namespace就能打通。
3. 虽然veth pair可以实现两个 Network Namespace 之间的通信，但 veth pair 有一个明显的缺陷，就是只能实现两个网络接口之间的通信。如果多个network namespace需要进行通信，则需要借助bridge。

将vlan 理解为网络协议的多路复用，vxlan 理解为mesh，网络路由、交换设备理解为支持某种协议的进程，feel 会很不一样。

## vlan

[VLAN是二层技术还是三层技术-车小胖的回答知乎](https://www.zhihu.com/question/52278720/answer/140914508)先用生活里的例子来比喻一下什么是VLAN，以及VLAN解决哪些问题。在魔都中心城区，经常有一些大房子被用来群租，有时大客厅也会放几张床用于出租，睡在客厅里的人肯定不爽的，因为有打呼噜的、有磨牙的、有梦游的、有说梦话的，一地鸡毛。为了克服以上的互相干扰，房东将客厅改造成若干个小房间，小房间还有门锁，这样每个房间就有了自己的私密空间，又可以有一定的安全性，至少贵重的物品可以放在房间里不怕别人拿错了。

**隔断间**形象的解释了划分vlan的理由，一个vlan一个广播域。我们将大广播域所对应的网段10.1.0.0/16分割成255个，那它们所对应的网段为10.1.1.0/24、10.1.2.0/24、10.1.3.0/24…10.1.255.0/24，它们对应的VLAN ID为1、2、3…255，这样每个广播域理论可以容纳255个终端，广播冲突的影响指数下降。

以太网/Ethernet 对vlan 的支持

![](/public/upload/network/vlan_vlanId.png)

对于支持 VLAN 的交换机，当这个交换机把二层的头取下来的时候，就能够识别这个 VLAN ID。这样只有相同 VLAN 的包，才会互相转发，不同 VLAN 的包，是看不到的。有一种口叫作 Trunk 口，它可以转发属于任何 VLAN 的口。交换机之间可以通过这种口相互连接。PS：说白了就是 在**网络层支持了 “多路复用”**，类似于http2 的StreamId，rpc 框架中的requestId

### vlan 划分

1. 常用的 VLAN 划分方式是通过端口进行划分，虽然这种划分 VLAN 的方式设置比较很简单， 但仅适用于终端设备物理位置比较固定的组网环境。随着移动办公的普及，终端设备可能不 再通过固定端口接入交换机，这就会增加网络管理的工作量。比如，一个用户可能本次接入 交换机的端口 1，而下一次接入交换机的端口 2，由于端口 1 和端口 2 属于不同的 VLAN，若 用户想要接入原来的 VLAN 中，网管就必须重新对交换机进行配置。显然，这种划分方式不 适合那些需要频繁改变拓扑结构的网络。
2. 而 MAC VLAN 则可以有效解决这个问题，它根据 终端设备的 MAC 地址来划分 VLAN。这样，即使用户改变了接入端口，也仍然处在原 VLAN 中。**注意，这种称为mac based vlan，跟macvlan还不是一个意思**

在交换机上配置了IMP(ip-mac-port映射)功能以后，交换机会检查每个数据包的源IP地址和MAC，对于没有在交换机内记录的IP和MAC地址的计算机所发出的数据包都会被交换机所阻止。ip-mac-port映射静态设置比较麻烦，可以开启交换机上面的DHCP SNOOPING功能， DHCP Snooping可以自动的学习IP和MAC以及端口的配对，并将学习到的对应关系保存到交换机的本地数据库中。

默认情况下，交换机上每个端口只允许绑定一个IP-MAC条目，所以在使用docker macvlan时要打开这样的限制。

### 为何要一个VLAN一个网段？

先来看看若是同一个网段会有什么问题？

**计算机发送数据包的基本过程：**

1. 如果要访问的目标IP跟自己是一个网段的（根据CIDR就可以判断出目标端IP和自己是否在一个网段内了），就不用经过网关了，先通过ARP协议获取目标端的MAC地址，源IP直接发送数据给目标端IP即可。
2. 如果访问的不是跟自己一个网段的，就会先发给网关，然后再由网关发送出去，网关就是路由器的一个网口，网关一般跟自己是在一个网段内的，通过ARP获得网关的mac地址，就可以发送出去了

从中可以看到，两个vlan 若是同一个网段，数据包会不经网关 直接发往目的ip，也就是第一种方式，会先通过ARP广播获取目的ip的MAC地址，而arp 广播只会在源ip本身的vlan 中广播（vlan限定了广播域），因此永远也无法拿到目的ip的mac地址。

若是一个vlan 一个网段，按照上述基本过程

1. 数据包会先发往网关（你必须为pc配置一个网关）
2. 网关进行数据转发

这里要注意：

2. vlan是实现在二层交换机上的，二层交换机没有网段的概念，只有路由器/三层交换机才有网段的概念。
3. vlan网段的配置是配置在路由器的子接口上，每个子接口<b>（Sub-Interface）</b>采用<b>802.1Q VLAN ID</b>封装，这样就把网段和对应的VLAN联系了起来，子接口通常作为此网段的缺省网关。如果采用三层交换机来替代路由器，则使用 <b>Interface VLAN ID </b>来将VLAN与网段联系起来，和使用路由器类似。

### 为啥vlan 要有个sub interface？

基于IEEE 802.1Q附加的VLAN信息，就像在传递物品时附加的标签。因此，它也被称作“标签型VLAN（Tagging VLAN）”。 [IEEE 802.1Q](https://en.wikipedia.org/wiki/IEEE_802.1Q)IEEE 802.1Q, often referred to as Dot1q, is the networking standard that supports virtual LANs (VLANs) on an IEEE 802.3 Ethernet network. The standard defines a system of VLAN tagging for Ethernet frames and the accompanying procedures to be used by bridges and switches in handling such frames. 一种说法是vlan 的划分方式太多（基于交换机port、基于mac地址、基于ip地址等），各家交换机各搞各的也不统一，不能互通，干脆弄一个802.1Q 协议统一下，让数据包带标签吧。交换机发现数据帧里有vlan tag（物理机接口 要能给 数据帧打tag，所以需要sub-interface）就按vlan tag 来，没有vlan tag 就按自家支持的 vlan 划分方法来。

## macvlan 和 ipvlan

无论是 macvlan 还是 ipvlan，它们都是在一个物理的网络接口上再配置几个虚拟的网络接口。在这些虚拟的网络接口上，都可以配置独立的 IP，并且这些 IP 可以属于不同的 Namespace。对于 macvlan，每个虚拟网络接口都有自己独立的 mac 地址；而 ipvlan 的虚拟网络接口是和物理网络接口共享同一个 mac 地址。而且它们都有自己的 L2/L3 的配置方式.

[一文读懂容器网络发展](https://mp.weixin.qq.com/s/fAThT7hKxDYXvFGJvqO42g)Macvlan 是 linux kernel 比较新的特性，允许在主机的一个网络接口上配置多个虚拟的网络接口，这些网络 interface 有自己独立的 mac 地址，也可以配置上 ip 地址进行通信。macvlan 下的虚拟机或者容器网络和主机在同一个网段中，共享同一个广播域。除此之外，macvlan 自身也完美支持 VLAN。

[Macvlan and IPvlan basics](https://sreeninet.wordpress.com/2016/05/29/macvlan-and-ipvlan/)Macvlan and ipvlan are Linux network drivers that exposes underlay or host interfaces directly to VMs or Containers running in the host. 

要跟mac based vlan 有所区分，参见[虚拟网络](http://qiankunli.github.io/2015/04/24/virtual_network.html)。

Macvlan, MACVLAN or MAC-VLAN allows you to configure multiple Layer 2 (i.e. Ethernet MAC) addresses **on a single physical interface**. Macvlan allows you to configure sub-interfaces (also termed slave devices) of a parent, physical Ethernet interface (also termed upper device), each with its own unique (randomly generated) MAC address, and consequently its own IP address. Applications, VMs and containers can then bind to a specific sub-interface to connect directly to the physical network, using their own MAC and IP address. 基于物理机网卡 physical interface 创建多个 sub-interface，拥有自己的MAC and IP ，直接接入physical network。

[Kubernetes在信也科技的落地实战](https://mp.weixin.qq.com/s/OBxnAitZaoI0lbP219Fvwg)Macvlan是一种直连网络，数据包通过宿主机网卡传输到硬件链路上，通过交换机路由器等设备最终到达目的地。Macvlan网络是Linux内核原生支持，不需要部署额外的组件。本身包含VLAN特性，一台宿主机上面可以虚拟出多块VLAN网卡，支持多个C类地址的IP分配。此时宿主机和交换机的链路必须是trunk模式，同一链路根据不同报文内的vlanID如（10、20、30）组成逻辑信道，互不干扰。

![](/public/upload/network/macvlan_network.png)

采用Macvlan网络模式之后，容器里面的网络协议栈和宿主机是完全独立。这就导致容器不能使用宿主机的iptables规则从而容器里面无法通过ClusterIP去访问Service。可以 采用Multus-CNI网络插件在Pod里面使用Bridge模式添加了第二块网卡，并设置路由如果访问ClusterIP就走Bridge模式的网卡。

### sub-interface

[Macvlan and IPvlan basics](https://sreeninet.wordpress.com/2016/05/29/macvlan-and-ipvlan/) 讲清了macvlan sub-interface 和 vlan-sub-interface 的异同

||物理网卡|vlan sub-interface|macvlan sub-interface|
|---|---|---|---|
|mac/ip||all sub-interfaces have same mac address（ip 手动/自动配）|each sub-interface will get unique mac and ip address|
|||each sub-interface belongs to a different L2 domain using vlan（发的包自带vlan id，是数据帧的一部分）| exposed directly in underlay network|
|配置文件|独立的网络接口配置文件|保存在临时文件`/proc/net/vlan/config`，重启会丢失|独立的网络接口配置文件|
|一般格式|eth0|eth0.1|eth0:1|

vlan sub-interface他们没有自己的配置文件，他们只是通过将物理网加入不同的VLAN而生成的VLAN虚拟网卡。如果将一个物理网卡添加到多个VLAN当中去的话，就会有多个VLAN虚拟网卡出现。vlan sub-interface 的mac 地址都一样

macvlan 本身跟vlan 没啥关系，如果不考虑虚拟化的概念，甚至可以理解为一个物理机插了多个网卡。但在容器里面通常跟vlan 结合使用（因为一个宿主机的上百个容器可能属于不同的vlan）。 Following picture shows an example where macvlan sub-interface works together with vlan sub-interface. Containers c1, c2 are connected to underlay interface ethx.1 and Containers c3, c4 are connected to underlay interface ethx.2.

![](/public/upload/network/macvlan_and_vlan.png)

[Docker Networking: macvlans with VLANs](https://hicu.be/docker-networking-macvlan-vlan-configuration) 

One macvlan, one Layer 2 domain and one subnet per physical interface, however, is a rather serious limitation in a modern virtualization solution. 这个说的是物理机时代，一个host 两个网卡，每个网卡属于不同的vlan（属于同一个vlan的话还整两个网卡干啥），而两个vlan 不可以是同一个网段。

a Docker host sub-interface can serve as a parent interface for the macvlan network. This aligns perfectly with the Linux implementation of VLANs, where each VLAN on a 802.1Q trunk connection is terminated on a sub-interface of the physical interface. You can map each Docker host interface to a macvlan network, thus extending the Layer 2 domain from the VLAN into the macvlan network.

Docker macvlan driver automagically creates host sub interfaces when you create a new macvlan network with sub interface as a parent。vlan sub interface 创建完毕后，以其为parent 创建macvlan sub interface 由 macvlan driver 自动完成。

### ipvlan

为容器手动配置上 ipvlan 的网络接口
```
docker run --init --name lat-test-1 --network none -d registry/latency-test:v1 sleep 36000
pid1=$(docker inspect lat-test-1 | grep -i Pid | head -n 1 | awk '{print $2}' | awk -F "," '{print $1}')
echo $pid1
ln -s /proc/$pid1/ns/net /var/run/netns/$pid1
ip link add link eth0 ipvt1 type ipvlan mode l2
ip link set dev ipvt1 netns $pid1
ip netns exec $pid1 ip link set ipvt1 name eth0
ip netns exec $pid1 ip addr add 172.17.3.2/16 dev eth0
ip netns exec $pid1 ip link set eth0 up
```

容器的虚拟网络接口，直接连接在了宿主机的物理网络接口上了，直接形成了一个网络二层的连接。如果从容器里向宿主机外发送数据，通过的接口要比 veth 少了，没有内部额外的 softirq 处理开销。。

```c
static int ipvlan_xmit_mode_l2(struct sk_buff *skb, struct net_device *dev){
    …
    /* 拿到ipvlan对应的物理网路接口设备， 然后直接从这个设备发送数据。*/ 
    skb->dev = ipvlan->phy_dev;
    return dev_queue_xmit(skb);
}
```

## vxlan

对于云平台中的**隔离**问题，前面咱们用的策略一直都是 VLAN，但是我们也说过这种策略的问题，VLAN 只有 12 位，共 4096 个。当时设计的时候，看起来是够了，但是现在绝对不够用，怎么办呢？

1. 一种方式是修改这个协议。这种方法往往不可行
2. 另一种方式就是扩展，在原来包的格式的基础上扩展出一个头，里面包含足够用于区分租户的 ID。一旦遇到需要区分用户的地方，我们就用**一个特殊的程序**，来处理这个特殊的包的格式。

![](/public/upload/network/vxlan_frame.jpg)

[知乎：VXLAN vs VLAN](https://zhuanlan.zhihu.com/p/36165475)

[Virtual Extensible LAN](https://en.wikipedia.org/wiki/Virtual_Extensible_LAN)

2. vlan 和vxlan 都是 virtual lan(局域网)，但vlan 是隔离出来的，借助了交换机的支持（或者说推出太久了，以至于交换机普遍支持），vxlan 是虚拟出来的，交换机无感知。
3. VXLAN与VLAN的最大区别在于，VLAN只是修改了原始的Ethernet Header，但是整个网络数据包还是原来那个数据包，而VXLAN是将原始的Ethernet Frame隐藏在UDP数据里面。经过VTEP封装之后，在网络线路上看起来只有VTEP之间的UDP数据传递，原始的网络数据包被掩盖了。
4. 为什么构建数据中心用VXLAN？

	* VXLAN evolved as a Data Center technology，所以分析vxlan 优势时一切以 数据中心的需求为出发点。一个超大型数据中心，交换机怎么联都是有技术含量的 [What is a Networking Switch Fabric](https://www.sdxcentral.com/sdn/definitions/what-is-networking-switch-fabric/)
	* vlan 4096 数量限制 不是问题
	* TOR（Top Of Rack）交换机MAC地址表限制。数据中心的虚拟化给网络设备带来的最直接影响就是：之前TOR（Top Of Rack）交换机的一个端口连接一个物理主机对应一个MAC地址，但现在交换机的一个端口虽然还是连接一个物理主机但是可能进而连接几十个甚至上百个虚拟机和相应数量的MAC地址。
	* **VTEP 在微服务领域有点像现在的service mesh**，一个vm/container 是一个微服务，微服务只需和sevice mesh sidecar 沟通


**使用报文解耦二三层**

![](/public/upload/network/vxlan_vtep.png)

[为什么集群需要 Overlay 网络](https://mp.weixin.qq.com/s/x7jLgThS2uwoPJcqsJE29w)Overlay 网络其实与软件定义网络（Software-defined networking、SDN）密切相关，而 SDN 引入了数据平面和控制平面，其中**数据平面负责转发数据，而控制平面负责计算并分发转发表**。VxLAN 的 RFC7348 中只定义了数据平面的内容，由该技术组成的网络可以通过传统的自学习模式学习网络中的 MAC 与 ARP 表项，但是在大规模的集群中，我们仍然需要引入控制平面分发路由转发表。

## 虚拟设备

[Linux 虚拟网络设备之 tun/tap](https://mp.weixin.qq.com/s/VgC7DiZEyCRamFs_Fix1lw)Linux内核中有一个网络设备管理层，处于网络设备驱动和协议栈之间，负责衔接它们之间的数据交互。驱动不需要了解协议栈的细节，协议栈也不需要了解设备驱动的细节。对于一个网络设备来说，就像一个管道（pipe）一样，有两端，从其中任意一端收到的数据将从另一端发送出去。对于Linux内核网络设备管理模块来说，**虚拟设备和物理设备没有区别**，都是网络设备，都能配置IP，从网络设备来的数据，都会转发给协议栈，协议栈过来的数据，也会交由网络设备发送出去，至于是怎么发送出去的，发到哪里去，那是设备驱动的事情，跟Linux内核就没关系了，所以说虚拟网络设备的一端也是协议栈，而**另一端是什么取决于虚拟网络设备的驱动实现**。

常见 PCIe 设备中，最适合 虚拟化 的就是网卡了: 一或多对 TX/RX queue + 一或多个中断，结合上一个 Routing ID，就可以抽象为一个 VF。而且它是近乎无状态的。

### tun/tap

tun0是一个Tun/Tap虚拟设备，从上图中可以看出它和物理设备eth0的差别，它们的一端虽然都连着协议栈，但另一端不一样，eth0的另一端是物理网络，这个物理网络可能就是一个交换机，而tun0的另一端是一个用户层的程序，协议栈发给tun0的数据包能被这个应用程序读取到，并且应用程序能直接向tun0写数据。

tun/tap设备的用处是将协议栈中的部分数据包转发给用户空间的应用程序，**给用户空间的程序一个处理数据包的机会**。于是比较常用的数据压缩、加密等功能就可以在应用程序B里面做进去，tun/tap设备最常用的场景是VPN。

### veth

veth和其它的网络设备都一样，一端连接的是内核协议栈。eth0的另一端是物理网络，veth设备是成对出现的，另一端两个设备彼此相连，一个设备收到协议栈的数据发送请求后，会将数据发送到另一个设备上去。

[手把手带你搞定4大容器网络问题](https://mp.weixin.qq.com/s/2PakMU3NR_tkly6K0HsAlQ)如果我们不能与一个专用的网络堆栈通信，那么它就没那么有用了。veth 设备是虚拟以太网设备。它们可以作为网络命名空间之间的隧道（也可以作为独立的网络设备使用）。虚拟以太网设备总是成对出现：`sudo ip link add veth0 type veth peer name ceth0`。创建后，veth0和ceth0都驻留在主机的网络堆栈（也称为根网络命名空间）上。为了连接根命名空间和netns0命名空间，我们需要将一个设备保留在根命名空间中，并将另一个设备移到netns0中：`sudo ip link set ceth0 netns netns0`。一旦我们打开设备并分配了正确的 IP 地址，任何出现在其中一台设备上的数据包都会立即出现在连接两个命名空间的对端设备上。

![](/public/upload/network/linux_veth.png)

veth 发送数据的函数是 veth_xmit()，它里面的主要操作就是找到 veth peer 设备，然后触发 peer 设备去接收数据包。
```c
static netdev_tx_t veth_xmit(struct sk_buff *skb, struct net_device *dev){
    ...
    /* 拿到veth peer设备的net_device */
    rcv = rcu_dereference(priv->peer);
    ...
    /* 将数据送到veth peer设备 */
    if (likely(veth_forward_skb(rcv, skb, rq, rcv_xdp) == NET_RX_SUCCESS)) {
    ...
}
static int veth_forward_skb(struct net_device *dev, struct sk_buff *skb,
                            struct veth_rq *rq, bool xdp){
    /* 这里最后调用了 netif_rx()， 是一个网络设备驱动里面标准的接收数据包的函数，netif_rx() 里面会为这个数据包 raise 一个 softirq。*/
    return __dev_forward_skb(dev, skb) ?: xdp ?
        veth_xdp_rx(rq, skb) :
        netif_rx(skb);
}
```
容器网络延时要比宿主机上的高吗?veth 发送数据的函数是 `veth_xmit()`，它里面的主要操作就是找到 veth peer 设备，然后触发 peer 设备去接收数据包。虽然 veth 是一个虚拟的网络接口，但是在接收数据包的操作上，除了没有硬件中断的处理，虚拟接口和真实的网路接口并没有太大的区别，特别是软中断（softirq）的处理部分和真实的网络接口是一样的。即使 softirq 的执行速度很快，还是会带来额外的开销。如果要减小容器网络延时，可以给容器配置 ipvlan/macvlan 的网络接口来替代 veth 网络接口。Ipvlan/macvlan 直接在物理网络接口上虚拟出接口，在发送对外数据包的时候可以直接通过物理接口完成，可以非常接近物理网络接口的延时。不过，由于 ipvlan/macvlan 网络接口直接挂载在物理网络接口上，对于需要使用 iptables 规则的容器，比如 Kubernetes 里使用 service 的容器，就不能工作了。



### 网桥

如果对网络不太熟悉，对于网桥的概念是很困惑的，下面试着简单解释一下。

1. 如果两台计算机想要互联？这种方式就是一根网线，有两个头。一头插在一台电脑的网卡上，另一头插在 另一台电脑的网卡上。但是在当时，普通的网线这样是通不了的，所以水晶头要做交叉线，用的就是所 谓的1-3、2-6 交叉接法。水晶头的第 1、2 和第 3、6 脚，它们分别起着收、发信号的作用。将一端的 1 号和 3 号线、2 号和 6 号线互换一下位置，就能够在物理层实现一端发送的信号，另一端能收到。

2. 三台计算机互联的方法

    1. 两两连接，那得需要多少网线，每个电脑得两个“插槽”，线路也比较乱。
    
    2. 使用集线器。

    3. 某个主机使用网桥。可以使用独立设备，也可以在计算机内模拟。

        host A ： 网卡1，网卡2，eth0（eth0连通外网）
    
        host B ： 网卡3（连接网卡1）
    
        host C ： 网卡4（连接网卡2）

        此时hosta分别和hostb、hostc彼此互访，因为网卡1和网卡2之间没有形成通路（在一个主机上，你是不是觉得默认应该是连通的？），hostb和hostc不能互相访问，所以弄一个网桥，将网卡1和网卡2“连通”。
        
使用集线器连接局域网中的pc时，一个重要缺点是：任何一个pc发数据，其它pc都会收到，无用不说，还导致物理介质争用。网桥与交换机类似，会学习mac地址与端口（串口）的映射。使用交换机替换集线器后，pc1发给pc2的数据只有pc2才会接收到。

[Bridge vs Macvlan](https://hicu.be/bridge-vs-macvlan)

**Switching was just a fancy name for bridging**, and that was a 1980s technology – or so the thinking went.A bridge can be a physical device or implemented entirely in software. Linux kernel is able to perform bridging since 1999. Switches have meanwhile became specialized physical devices and software bridging had almost lost its place. However, with the advent of virtualization, virtual machines running on physical hosts required Layer 2 connection to the physical network and other VMs. Linux bridging provided a well proven technology and entered it’s Renaissance（文艺复兴）. 最开始bridge是一个硬件， 也叫swtich，后来软件也可以实现bridge了，swtich就专门称呼硬件交换机了，再后来虚拟化时代到来，bridge 迎来了第二春。

[Linux 虚拟网络设备之 bridge](https://mp.weixin.qq.com/s/BWyO9zb4I2lMyjVBAoVCiA)bridge 常用场景

![](/public/upload/network/bridge_tun.png)

![](/public/upload/network/bridge_veth.png)


[Macvlan and IPvlan basics](https://sreeninet.wordpress.com/2016/05/29/macvlan-and-ipvlan/)In linux bridge implementation, VMs or Containers will connect to bridge and bridge will connect to outside world. For external connectivity, we would need to use NAT. container 光靠 bridge 无法直接访问外网。

A bridge transparently relays traffic between multiple network interfaces. **In plain English this means that a bridge connects two or more physical Ethernets together to form one bigger (logical) Ethernet** 


<table>
	<tr>
		<td>network layer</td>
		<td colspan="3">iptables rules</td>
	</tr>
	<tr>
		<td>func</td>
		<td>netif_receive_skb/dev_queue_xmit</td>
		<td colspan=2>netif_receive_skb/dev_queue_xmit</td>
	</tr>
	<tr>
		<td rowspan="2">data link layer</td>
		<td rowspan="2">eth0</td>
		<td colspan="2">br0</td>
	</tr>
	<tr>
		<td>eth1</td>
		<td>eth2</td>
	</tr>
	<tr>
		<td>func</td>
		<td>rx_handler/hard_start_xmit</td>
		<td>rx_handler/hard_start_xmit</td>
		<td>rx_handler/hard_start_xmit</td>
	</tr>
	<tr>
		<td>phsical layer</td>
		<td>device driver</td>
		<td>device driver</td>
		<td>device driver</td>
	</tr>
</table>

通俗的说，网桥屏蔽了eth1和eth2的存在。正常情况下，每一个linux 网卡都有一个device or net_device struct.这个struct有一个rx_handler。

eth0驱动程序收到数据后，会执行rx_handler。rx_handler会把数据包一包，交给network layer。从源码实现就是，接入网桥的eth1，在其绑定br0时，其rx_handler会换成br0的rx_handler。等于是eth1网卡的驱动程序拿到数据后，直接执行br0的rx_handler往下走了。所以，eth1本身的ip和mac，network layer已经不知道了，只知道br0。

br0的rx_handler会决定将收到的报文转发、丢弃或提交到协议栈上层。如果是转发，br0的报文转发在数据链路层，但也会执行一些本来属于network layer的钩子函数。也有一种说法是，网桥处于forwarding状态时，报文必须经过layer3转发。这些细节的确定要通过学习源码来达到，此处先不纠结。

读了上文，应该能明白以下几点。

1. 为什么要给网桥配置ip，或者说创建br0 bridge的同时，还会创建一个br0 iface。
2. 为什么eth0和eth1在l2,连上br0后，eth1和eth0的连通还要受到iptables rule的控制。
3. 网桥首先是为了屏蔽eth0和eth1的，其次是才是连通了eth0和eth1。

2018.12.3 补充：一旦一张虚拟网卡被“插”在网桥上，它就会变成该网桥的“从设备”。从设备会被“剥夺”调用网络协议栈处理数据包的资格，从而“降级”成为网桥上的一个端口。而这个端口唯一的作用，就是接收流入的数据包，然后把这些数据包的“生杀大权”（比如转发或者丢弃），全部交给对应的网桥。

## 虚拟设备 ==> 虚拟网络

Linux 用户想要使用网络功能，不能通过直接操作硬件完成，而需要直接或间接的操作一个Linux 为我们抽象出来的设备，即通用的 Linux 网络设备来完成。“eth0”并不是网卡，而是Linux为我们抽象（或模拟）出来的“网卡”。除了网卡，现实世界中存在的网络元素Linux都可以模拟出来，包括但不限于：电脑终端、二层交换机、路由器、网关、支持 802.1Q VLAN 的交换机、三层交换机、物理网卡、支持 Hairpin 模式的交换机。同时，既然linux可以模拟网络设备，自然提供了操作这些虚拟的网络设备的命令或interface。

什么是network driver?

A network device driver is a device driver that enables a network device to communicate between the computer and operating system as well as with other network computers and network devices.

[Device driver](https://en.wikipedia.org/wiki/Device_driver)In computing, a device driver is a computer program that operates or controls a particular type of device that is attached to a computer. **A driver provides a software interface to hardware devices**, enabling operating systems and other computer programs to access hardware functions without needing to know precise details about the hardware being used. 驱动就是对硬件提供软件接口，屏蔽硬件细节。

A driver communicates with the device through the computer bus or communications subsystem to which the hardware connects. When a calling program invokes a routine in the driver, the driver issues commands to the device. Once the device sends data back to the driver, the driver may invoke routines in the original calling program. Drivers are hardware dependent and operating-system-specific. They usually provide the interrupt handling required for any necessary asynchronous time-dependent hardware interface.


网卡 ==> computer bus ==> network driver ==> Subroutine/子程序 ==> calling program。也就是network driver 在网卡 与操作系统之间，从这个角度看，跟磁盘驱动、鼠标驱动类似了。


## macvlan 实操

以下实现基于docker1.13，物理机使用`192.168.0.0/16`网段，容器使用`172.31.0.0/16`网段。

1. docker host，自定义ipam plugin负责ip地址管理，每个docker host运行一个ipam plugin，并根据ipam plugin创建local scope的macvlan network。
2. 创建容器时使用macvlan网络
3. 外置交换机负责容器之间、host之间、容器与host之间的连通性。

MACVLAN可以从一个主机接口虚拟出多个macvtap，且每个macvtap设备都拥有不同的mac地址（对应不同的linux字符设备）。

docker macvlan 用802.1q模式，对于一个交换机端口来说：

1. 物理机和容器的数据包属于不同的vlan，so， 交换机端口设置为trunk；
2. 物理机和容器的数据包属于不同的网段，so，在交换机的三层加一层路由，打通物理机和容器的两个网段。

### 设置路由器或交换机

[Docker Networking: macvlans with VLANs](https://hicu.be/docker-networking-macvlan-vlan-configuration) 

本小节是2018.12.17补充，所以网段部分对不上

if you happen to have a Cisco IOS router

```
router(config)# interface fastEthernet 0/0
router(config-if)# no shutdown

router(config)# interface fastEthernet 0/0.10
router(config-subif)# encapsulation dot1Q 10
router(config-subif)# ip address 10.0.10.1 255.255.255.0
router(config-subif)# ipv6 address 2001:db8:babe:10::1/64

router(config)# interface fastEthernet 0/0.20
router(config-subif)# encapsulation dot1Q 20
router(config-subif)# ip address 10.0.20.1 255.255.255.0
router(config-subif)# ipv6 address 2001:db8:babe:20::1/64

…or Cisco Layer 3 Switch…

switch# configure terminal
switch(config)# vlan 10
switch(config)# vlan 20

switch(config)# interface fastEthernet0/0
switch(config-if)# switchport mode trunk
switch(config-if)# switchport trunk native vlan 1

switch(config)# interface vlan 10
switch(config-if)# ip address 10.0.10.1 255.255.255.0
switch(config-if)# ipv6 address 2001:db8:babe:10::1/64

switch(config)# interface vlan 20
switch(config-if)# ip address 10.0.20.1 255.255.255.0
switch(config-if)# ipv6 address 2001:db8:babe:20::1/64
```
	
可以看到，从交换机的角度看，也是与linux 类似的ip命令，配置ip、网段等。

### 物理机创建vlan的sub interface

使用802.1q vlan时，我们发出去的数据包，要有802.1q中的vlan tag。为了不影响物理网卡的正常使用，就是只有基于sub interface（eth1.10）来发送802.1q package。

1. Load the 802.1q module into the kernel.`sudo modprobe 8021q`
2. **Create a new interface that is a member of a specific VLAN**, 
VLAN id 10 is used in this example. Keep in mind you can only use physical interfaces as a base, creating VLAN's on virtual interfaces (i.e. eth0:1) will not work. We use the physical interface eth1 in this example. This command will add an additional interface next to the interfaces which have been configured already, so your existing configuration of eth1 will not be affected. `sudo vconfig add eth1 10`
3. Assign an address to the new interface. `sudo ip addr add 10.0.0.1/24 dev eth0.10`
4. Starting the new interface. `sudo ip link set up eth0.10`
	
基于sub interface创建docker macvlan 网络

```sh
docker network  create  -d macvlan \
    --subnet=172.31.0.0/16 \
    --gateway=172.31.0.1 \
    -o parent=eth0.10 macvlan10
```
创建容器，指定使用macvlan网络

```sh
docker run --net=macvlan10 -it --name macvlan_test5 --rm alpine /bin/sh
```	
## 小结

||特点|ip/mac address|从交换机的视角看vlan方案|
|---|---|---|---|
|vlan|A virtual LAN (VLAN) is any broadcast domain that is partitioned and isolated in a computer network at the data link layer (OSI layer 2).<br>each sub-interface belongs to a different L2 domain using vlan |all sub-interfaces have same mac address.|交换机要支持 vlan tag|
|Macvlan|Containers will directly get exposed in underlay network using Macvlan sub-interfaces.<br> Macvlan has 4 types(Private, VEPA, Bridge, Passthru)<br> 可以在vlan sub-interface 上创建 macvlan subinterface|Macvlan allows a single physical interface to have multiple mac and ip addresses using macvlan sub-interfaces. <br>|交换机的port一般只与一个mac绑定，使用macvlan 后必须支持绑定多个 且 无数量限制|
|ipvlan|  ipvlan supports L2 and L3 mode.|the endpoints have the same mac address|省mac地址|
|vxlan|Virtual Extensible LAN (VXLAN) is a network virtualization technology that attempts to address the scalability problems associated with large cloud computing deployments. <br>VXLAN endpoints, which terminate VXLAN tunnels and may be either virtual or physical switch ports, are known as VXLAN tunnel endpoints (VTEPs)||交换机无感知|


    
