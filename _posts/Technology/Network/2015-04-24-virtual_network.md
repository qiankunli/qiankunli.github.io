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

相对于物理网络，虚拟化有两个方面：

1. 虚拟设备  一般伴随网络驱动
2. 虚拟网路

## vlan

[VLAN是二层技术还是三层技术-车小胖的回答知乎](https://www.zhihu.com/question/52278720/answer/140914508)先用生活里的例子来比喻一下什么是VLAN，以及VLAN解决哪些问题。在魔都中心城区，经常有一些大房子被用来群租，有时大客厅也会放几张床用于出租，睡在客厅里的人肯定不爽的，因为有打呼噜的、有磨牙的、有梦游的、有说梦话的，一地鸡毛。为了克服以上的互相干扰，房东将客厅改造成若干个小房间，小房间还有门锁，这样每个房间就有了自己的私密空间，又可以有一定的安全性，至少贵重的物品可以放在房间里不怕别人拿错了。

**隔断间**形象的解释了划分vlan的理由，一个vlan一个广播域。我们将大广播域所对应的网段10.1.0.0/16分割成255个，那它们所对应的网段为10.1.1.0/24、10.1.2.0/24、10.1.3.0/24…10.1.255.0/24，它们对应的VLAN ID为1、2、3…255，这样每个广播域理论可以容纳255个终端，广播冲突的影响指数下降。

以太网/Ethernet 对vlan 的支持

![](/public/upload/network/vlan_vlanId.png)

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


## vxlan

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

## macvlan 和 ipvlan

[Macvlan and IPvlan basics](https://sreeninet.wordpress.com/2016/05/29/macvlan-and-ipvlan/)

Macvlan and ipvlan are Linux network drivers that exposes underlay or host interfaces directly to VMs or Containers running in the host. 

## 虚拟设备

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


[Macvlan and IPvlan basics](https://sreeninet.wordpress.com/2016/05/29/macvlan-and-ipvlan/)In linux bridge implementation, VMs or Containers will connect to bridge and bridge will connect to outside world. For external connectivity, we would need to use NAT. container 光靠 bridge 无法直接访问外网。

建议看下 [docker中涉及到的一些linux知识](http://qiankunli.github.io/2016/12/02/linux_docker.html) 对网桥源码的分析。

## 虚拟设备 ==> 虚拟网络

Linux 用户想要使用网络功能，不能通过直接操作硬件完成，而需要直接或间接的操作一个Linux 为我们抽象出来的设备，即通用的 Linux 网络设备来完成。“eth0”并不是网卡，而是Linux为我们抽象（或模拟）出来的“网卡”。除了网卡，现实世界中存在的网络元素Linux都可以模拟出来，包括但不限于：电脑终端、二层交换机、路由器、网关、支持 802.1Q VLAN 的交换机、三层交换机、物理网卡、支持 Hairpin 模式的交换机。同时，既然linux可以模拟网络设备，自然提供了操作这些虚拟的网络设备的命令或interface。

## 小结

||特点|ip/mac address|从交换机的视角看vlan方案|
|---|---|---|---|
|vlan|A virtual LAN (VLAN) is any broadcast domain that is partitioned and isolated in a computer network at the data link layer (OSI layer 2).<br>each sub-interface belongs to a different L2 domain using vlan |all sub-interfaces have same mac address.|交换机要支持 vlan tag|
|Macvlan|Containers will directly get exposed in underlay network using Macvlan sub-interfaces.<br> Macvlan has 4 types(Private, VEPA, Bridge, Passthru)<br> 可以在vlan sub-interface 上创建 macvlan subinterface|Macvlan allows a single physical interface to have multiple mac and ip addresses using macvlan sub-interfaces. <br>|交换机的port一般只与一个mac绑定，使用macvlan 后必须支持绑定多个 且 无数量限制|
|ipvlan|  ipvlan supports L2 and L3 mode.|the endpoints have the same mac address|省mac地址|
|vxlan|Virtual Extensible LAN (VXLAN) is a network virtualization technology that attempts to address the scalability problems associated with large cloud computing deployments. <br>VXLAN endpoints, which terminate VXLAN tunnels and may be either virtual or physical switch ports, are known as VXLAN tunnel endpoints (VTEPs)||交换机无感知|


    
