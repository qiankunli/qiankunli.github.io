---

layout: post
title: 虚拟网络
category: 技术
tags: Network
keywords: virtual network

---

## 前言


* TOC
{:toc}

网络虚拟化作为 SDN（Software Defined Network，软件定义网络）的一种实现，无非就是虚拟出 vNIC（虚拟网卡）、vSwitch（虚拟交换机）、vRouter（虚拟路由器）等设备，配置相应的数据包流转规则而已。

[Macvlan和IPvlan基础知识](https://mp.weixin.qq.com/s/r_CuqjypaaMRDZfW-RHjxw)运行裸机服务器时，主机网络可以很简单，只需很少的以太网接口和提供外部连接的默认网关。但当我们在一个主机中运行多个虚拟机时，需要在主机内和跨主机之间提供虚拟机之间的连接。一般，单个主机中的VM数量不超过15-20个。但在一台主机上运行Containers时，单个主机上的Containers数量很容易超过100个，需要有成熟的机制来实现Containers之间的网络互联。概括地说，容器或虚拟机之间有两种通信方式。在底层网络方法中，虚拟机或容器直接暴露给主机网络，Bridge、macvlan和ipvlan网络驱动程序都可以做到。在Overlay网络方法中，容器或VM网络和底层网络之间存在额外的封装形式，如VXLAN、NVGRE等。

![](/public/upload/network/network_device.png)

**我们为什么要虚拟网络呢？**这个问题有很多答案，但是和云计算的价值一样，虚拟网络作为云计算的一项基础技术，它们最终都是为了资源利用效率的提升。MACVlan 和 IPVlan 就是服务了这个最终目的，从虚拟化到容器化，时代对**网络密度**提出了越来越高的要求，伴随着容器技术的诞生，首先是 veth 走上舞台，但是密度够了，**还要性能的高效**，MACVlan 和 IPVlan 通过子设备提升密度并保证高效的方式应运而生。

[Linux虚拟网络技术学习](https://mp.weixin.qq.com/s/2PYds2LDie7W5sXYi1qwew)在Linux虚拟化技术中，网络层面，通常重要的三个技术分别是Network Namespace、veth pair、以及网桥或虚拟交换机技术。

1. 对于每个 Network Namespace 来说，它会有自己独立的网卡、路由表、ARP 表、iptables 等和网络相关的资源。ip命令提供了`ip netns exec`子命令可以在对应的 Network Namespace 中执行命令。PS： 也就是`ip netns exec` 操作网卡、路由表、ARP表、iptables 等。也可以打开一个shell ，后面所有的命令都在这个Network Namespace中执行，好处是不用每次执行命令时都要带上ip netns exec ，缺点是我们无法清楚知道自己当前所在的shell，容易混淆。
2. 默认情况下，network namespace 是不能和主机网络，或者其他 network namespace 通信的。可以使用 Linux 提供的veth pair来完成通信，veth pair你可以理解为使用网线连接好的两个接口，把两个端口放到两个namespace中，那么这两个namespace就能打通。
3. 虽然veth pair可以实现两个 Network Namespace 之间的通信，但 veth pair 有一个明显的缺陷，就是只能实现两个网络接口之间的通信。如果多个network namespace需要进行通信，则需要借助bridge。

将vlan 理解为网络协议的多路复用，vxlan 理解为mesh，网络路由、交换设备理解为支持某种协议的进程，feel 会很不一样。

## 虚拟设备

[Linux 虚拟网络设备之 tun/tap](https://mp.weixin.qq.com/s/VgC7DiZEyCRamFs_Fix1lw)Linux内核中有一个网络设备管理层，处于网络设备驱动和协议栈之间，负责衔接它们之间的数据交互。驱动不需要了解协议栈的细节，协议栈也不需要了解设备驱动的细节。对于一个网络设备来说，就像一个管道（pipe）一样，有两端，从其中任意一端收到的数据将从另一端发送出去。对于Linux内核网络设备管理模块来说，**虚拟设备和物理设备没有区别**，都是网络设备，都能配置IP，从网络设备来的数据，都会转发给协议栈，协议栈过来的数据，也会交由网络设备发送出去，至于是怎么发送出去的，发到哪里去，那是设备驱动的事情，跟Linux内核就没关系了，所以说虚拟网络设备的一端也是协议栈，而**另一端是什么取决于虚拟网络设备的驱动实现**。

常见 PCIe 设备中，最适合 虚拟化 的就是网卡了: 一或多对 TX/RX queue + 一或多个中断，结合上一个 Routing ID，就可以抽象为一个 VF。而且它是近乎无状态的。目前主流的虚拟网卡方案有 tun/tap 和 veth 两种，容器一般是使用 veth 来实现网络通信的。 

### tun/tap

[一文读懂网络虚拟化之 tun/tap 网络设备](https://mp.weixin.qq.com/s/bGY7BJdIz3SE491SclKRMQ)tun 和 tap 是一组通用的虚拟驱动程序包，是两个相对独立的虚拟网络设备
1. tun 模式 与 tap 模式。其中 tap 模拟了以太网设备，操作二层数据包（以太帧），tun 则是模拟了网络层设备，操作三层数据包（IP 报文）。
2. 它和物理设备eth0的差别，它们的一端虽然都连着协议栈，但另一端不一样，eth0的另一端是物理网络，这个物理网络可能就是一个交换机，**而tun0的另一端是一个用户层的程序**。
3. 使用 tun/tap 设备的目的，其实是为了把来自协议栈的数据包，先交给某个打开了`/dev/net/tun`字符设备的用户进程处理后，再把数据包重新发回到链路中。如此一来，只要**协议栈中的数据包能被用户态程序截获并加工处理**，程序员就有足够的舞台空间去玩出各种花样，比如数据压缩、流量加密、透明代理等功能，都能够在此基础上实现。PS：有点类似于FUSE 用户态实现文件系统

![](/public/upload/network/tun_tap.png)

以最典型的 VPN 应用程序为例，应用程序通过 tun 设备对外发送数据包后，tun 设备如果发现另一端的字符设备已经被 VPN 程序打开（这就是一端连接着网络协议栈，另一端连接着用户态程序），就会把数据包通过字符设备发送给 VPN 程序，VPN 收到数据包，会修改后再重新封装成新报文，比如数据包原本是发送给 A 地址的，VPN 把整个包进行加密，然后作为报文体，封装到另一个发送给 B 地址的新数据包当中。

![](/public/upload/network/tun_vpn.png)

不过，使用 tun/tap 设备来传输数据需要经过两次协议栈，所以会不可避免地产生一定的性能损耗，因而如果条件允许，容器对容器的直接通信并不会把 tun/tap 作为首选方案，而是一般基于 veth 来实现的。但 tun/tap 并没有像 veth 那样，有要求设备成对出现、数据要原样传输的限制，数据包到了用户态程序后，我们就有完全掌控的权力，要进行哪些修改、要发送到什么地方，都可以通过编写代码去实现，所以 tun/tap 方案比起 veth 方案有更广泛的适用范围。

### veth

veth和其它的网络设备都一样，一端连接的是内核协议栈。eth0的另一端是物理网络，veth设备是成对出现的，**veth另一端两个设备彼此相连**（内部通过指针相互关联），一个设备收到协议栈的数据发送请求后，会将数据发送到另一个设备上去。Linux 开始支持网络名空间隔离的同时，也提供了veth，让两个隔离的网络名称空间之间可以互相通信。

![](/public/upload/network/virtual_network_dev_veth.jpg)

[手把手带你搞定4大容器网络问题](https://mp.weixin.qq.com/s/2PakMU3NR_tkly6K0HsAlQ)如果我们不能与一个专用的网络堆栈通信，那么它就没那么有用了。veth 设备是虚拟以太网设备。它们可以作为**网络命名空间之间的隧道**（也可以作为独立的网络设备使用）。虚拟以太网设备总是成对出现：`sudo ip link add veth0 type veth peer name ceth0`。创建后，veth0和ceth0都驻留在主机的网络堆栈（也称为根网络命名空间）上。为了连接根命名空间和netns0命名空间，我们需要将一个设备保留在根命名空间中，并将另一个设备移到netns0中：`sudo ip link set ceth0 netns netns0`。一旦我们打开设备并分配了正确的 IP 地址，任何出现在其中一台设备上的数据包都会立即出现在连接两个命名空间的对端设备上。

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
    2. 使用集线器。使用集线器连接局域网中的pc时，一个重要缺点是：任何一个pc发数据，其它pc都会收到，无用不说，还导致物理介质争用。
	3. 使用交换机，使用交换机替换集线器后，会学习mac地址与端口（串口）的映射，pc1发给pc2的数据只有pc2才会接收到。
    
        
虽然 veth 以模拟网卡直连的方式，很好地解决了两个容器之间的通信问题，然而对多个容器间通信，如果仍然单纯只用 veth pair 的话，事情就会变得非常麻烦，毕竟，让每个容器都为与它通信的其他容器建立一对专用的 veth pair，根本就不实际，真正做起来成本会很高。因此这时，**就迫切需要有一台虚拟化的交换机，来解决多容器之间的通信问题了**。

![](/public/upload/network/bridge_veth.jpg)

Linux Bridge 是在 Linux Kernel 2.2 版本开始提供的二层转发工具，由brctl命令创建和管理。Linux Bridge 创建以后，就能够接入任何位于二层的网络设备，无论是真实的物理设备（比如 eth0），还是虚拟的设备（比如 veth 或者 tap），都能与 Linux Bridge 配合工作。当有二层数据包（以太帧）从网卡进入 Linux Bridge，它就会根据数据包的类型和目标 MAC 地址，按照如下规则转发处理：
1. 如果数据包是广播帧，转发给所有接入网桥的设备。
2. 如果数据包是单播帧，且 MAC 地址在地址转发表中不存在（网桥与交换机类似，会学习mac地址与端口（串口）的映射），则洪泛（Flooding）给所有接入网桥的设备，并把响应设备的接口与 MAC 地址学习（MAC Learning）到自己的 MAC 地址转发表中。
3. 如果数据包是单播帧，且 MAC 地址在地址转发表中已存在，则直接转发到地址表中指定的设备。
4. 如果数据包是此前转发过的，又重新发回到此 Bridge，说明冗余链路产生了环路。由于以太帧不像 IP 报文那样有 TTL 来约束，所以一旦出现环路，如果没有额外措施来处理的话，就会永不停歇地转发下去。那么对于这种数据包，就需要交换机实现生成树协议（Spanning Tree Protocol，STP）来交换拓扑信息，生成唯一拓扑链路以切断环路。

Linux Bridge 不仅用起来像交换机，实现起来也像交换机。对于通过brctl命令显式接入网桥的设备，Linux Bridge 与物理交换机的转发行为是完全一致的，它也不允许给接入的设备设置 IP 地址，**因为网桥是根据 MAC 地址做二层转发的，就算设置了三层的 IP 地址也没有意义**。不过，它与普通的物理交换机也还是有一点差别的，普通交换机只会单纯地做二层转发，**Linux Bridge 却还支持把发给它自身的数据包，接入到主机的三层协议栈中**。除了显式接入的设备外，它自己也无可分割地连接着一台有着完整网络协议栈的 Linux 主机，因为 Linux Bridge 本身肯定是在某台 Linux 主机上创建的，我们可以看作是 Linux Bridge 有一个与自己名字相同的隐藏端口，隐式地连接了创建它的那台 Linux 主机。因此，Linux Bridge 允许给自己设置 IP 地址，这样就比普通交换机多出了一种特殊的转发情况：如果数据包的目的 MAC 地址为网桥本身，并且网桥设置了 IP 地址的话，那该数据包就会被认为是收到发往创建网桥那台主机的数据包，这个数据包将不会转发到任何设备，而是直接交给上层（三层）协议栈去处理。这时，网桥就取代了物理网卡 eth0 设备来对接协议栈，进行三层协议的处理。


[Bridge vs Macvlan](https://hicu.be/bridge-vs-macvlan)

**Switching was just a fancy name for bridging**, and that was a 1980s technology – or so the thinking went.A bridge can be a physical device or implemented entirely in software. Linux kernel is able to perform bridging since 1999. Switches have meanwhile became specialized physical devices and software bridging had almost lost its place. However, with the advent of virtualization, virtual machines running on physical hosts required Layer 2 connection to the physical network and other VMs. Linux bridging provided a well proven technology and entered it’s Renaissance（文艺复兴）. 最开始bridge是一个硬件， 也叫swtich，后来软件也可以实现bridge了，swtich就专门称呼硬件交换机了，再后来虚拟化时代到来，bridge 迎来了第二春。

[Linux 虚拟网络设备之 bridge](https://mp.weixin.qq.com/s/BWyO9zb4I2lMyjVBAoVCiA)bridge 常用场景

![](/public/upload/network/bridge_veth.png)

[Macvlan and IPvlan basics](https://sreeninet.wordpress.com/2016/05/29/macvlan-and-ipvlan/)In linux bridge implementation, VMs or Containers will connect to bridge and bridge will connect to outside world. For external connectivity, we would need to use NAT. container 光靠 bridge 无法直接访问外网。


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

1. 正常情况下，每一个linux 网卡都有一个device or net_device struct，这个struct有一个rx_handler。eth0驱动程序收到数据后，会执行rx_handler，rx_handler会把数据包一包，交给network layer。
2. 而接入网桥的eth1，在其绑定br0时，其rx_handler会换成br0的rx_handler（不同版本实现不同，总之是bridge会接管该veth的数据接收过程）。等于是eth1网卡的驱动程序拿到数据后，直接执行br0的rx_handler往下走了。所以，eth1本身的ip和mac，network layer已经不知道了，只知道br0。br0的rx_handler会决定将收到的报文转发、丢弃或提交到协议栈上层。如果是转发，从自己连接的所有设备中查找目的设备，调用该设备的发送函数将数据发送出去。

![](/public/upload/network/bridge_veth_communication.png)

2018.12.3 补充：一旦一张虚拟网卡被“插”在网桥上，它就会变成该网桥的“从设备”。从设备会被“剥夺”调用网络协议栈处理数据包的资格，从而“降级”成为网桥上的一个端口。而这个端口唯一的作用，就是接收流入的数据包，然后把这些数据包的“生杀大权”（比如转发或者丢弃），全部交给对应的网桥。

## （交换机的）vlan



[VLAN是二层技术还是三层技术-车小胖的回答知乎](https://www.zhihu.com/question/52278720/answer/140914508)先用生活里的例子来比喻一下什么是VLAN，以及VLAN解决哪些问题。在魔都中心城区，经常有一些大房子被用来群租，有时大客厅也会放几张床用于出租，睡在客厅里的人肯定不爽的，因为有打呼噜的、有磨牙的、有梦游的、有说梦话的，一地鸡毛。为了克服以上的互相干扰，房东将客厅改造成若干个小房间，小房间还有门锁，这样每个房间就有了自己的私密空间，又可以有一定的安全性，至少贵重的物品可以放在房间里不怕别人拿错了。

**隔断间**形象的解释了划分vlan的理由，随着计算机的增多，如果仅有一个广播域，会有大量的广播帧（如 ARP 请求、DHCP、RIP 都会产生广播帧）转发到同一网络中的所有客户机上。这样造成了没有必要的浪费，一方面广播信息消耗了网络整体的带宽，另一方面，收到广播信息的计算机还要消耗一部分CPU时间来对它进行处理。造成了网络带宽和CPU运算能力的大量无谓消耗。在这种情况下出现了 VLAN 技术。**VLAN 的首要职责就是划分广播域**，一个vlan一个广播域，把连接在同一个物理网络上的设备区分开来，VLAN 间不能直接互通，广播报文就被限制在一个 VLAN 内。我们将大广播域所对应的网段10.1.0.0/16分割成255个，那它们所对应的网段为10.1.1.0/24、10.1.2.0/24、10.1.3.0/24…10.1.255.0/24，它们对应的VLAN ID为1、2、3…255，这样每个广播域理论可以容纳255个终端，广播冲突的影响指数下降。

划分的具体方法是在以太帧的报文头中加入 VLAN Tag，让所有广播只针对具有相同 VLAN Tag 的设备生效。

![](/public/upload/network/vlan_vlanId.png)

对于支持 VLAN 的交换机，当这个交换机把二层的头取下来的时候，就能够识别这个 VLAN ID。这样只有相同 VLAN 的包，才会互相转发，不同 VLAN 的包，是看不到的。有一种口叫作 Trunk 口，它可以转发属于任何 VLAN 的口。交换机之间可以通过这种口相互连接。PS：说白了就是 在**网络层支持了 “多路复用”**，类似于http2 的StreamId，rpc 框架中的requestId

![](/public/upload/network/vlan_sub_interface.png)

### vlan 划分

交换机端口类型
1. Access，只能属于1个VLAN， 比如`port default vlan vlan-id`，将Access端口加入到指定的VLAN中。它缺省VLAN ID就是它所在VLAN。
	1. Acess端口收报文，收到报文判断是否有VLAN信息，如果没有则打上PVID并进行交换转发；如果有，则丢弃。
	2. Acess端口发报文，报文VLAN信息被剥离直接发送出去。
2. Trunk，可以允许多个VLAN通过，可以接收和发送多个VLAN 报文，所以需要设置缺省VLAN ID/PVID。
	1. Trunk端口接收报文，如果没有则打上端口的PVID，并进行交换转发。如果有则判断该Trunk端口是否允许该VLAN的数据进入：如果允许则转发，如果不允许则丢弃
	2. Trunk端口发送报文，比较端口的PVID和将要发送报文的VLAN信息，如果两者相等则剥离VLAN信息，再发送；如果不相等则直接发送
3. Hybrid，可以允许多个VLAN通过，可以接收和发送多个VLAN 报文。

交换机内部在处理数据包时，所有的数据包一定是打上VLAN tag的，交换机在接收到帧后，会根据对应端口类型采取相应的数据收、发处理。如果帧需要通过另一台交换机转发，则该帧必须通过干道链路透传到对端交换设备上。为了保证其它交换设备能够正确处理帧中的VLAN信息，在干道链路上传输的帧必须都打上了VLAN标签。当交换机最终确定帧出端口后，在将帧发送给主机前需要将VLAN标签从帧中删除，这样主机接收到的帧都是不带VLAN标签的以太网帧，也只有这样主机才可能识别。所以一般情况下，**干道链路上传输的都是带VLAN标签的帧，接入链路上传送到的都是不带VLAN标签帧**。这样处理的好处是：网络中配置的VLAN信息可以被所有交换设备正确处理，而主机不需要了解VLAN信息。

1. 常用的 VLAN 划分方式是通过端口进行划分，虽然这种划分 VLAN 的方式设置比较很简单， 但仅适用于终端设备物理位置比较固定的组网环境。随着移动办公的普及，终端设备可能不再通过固定端口接入交换机，这就会增加网络管理的工作量。比如，一个用户可能本次接入 交换机的端口 1，而下一次接入交换机的端口 2，由于端口 1 和端口 2 属于不同的 VLAN，若 用户想要接入原来的 VLAN 中，网管就必须重新对交换机进行配置。显然，这种划分方式不 适合那些需要频繁改变拓扑结构的网络。
2. 而 MAC VLAN 则可以有效解决这个问题，它根据 终端设备的 MAC 地址来划分 VLAN。这样，即使用户改变了接入端口，也仍然处在原 VLAN 中。**注意，这种称为mac based vlan，跟macvlan还不是一个意思**
3. 基于ip地址 划分vlan

在交换机上配置了IMP(ip-mac-port映射)功能以后，交换机会检查每个数据包的源IP地址和MAC，对于没有在交换机内记录的IP和MAC地址的计算机所发出的数据包都会被交换机所阻止。ip-mac-port映射静态设置比较麻烦，可以开启交换机上面的DHCP SNOOPING功能， DHCP Snooping可以自动的学习IP和MAC以及端口的配对，并将学习到的对应关系保存到交换机的本地数据库中。默认情况下，交换机上每个端口只允许绑定一个IP-MAC条目，所以在使用docker macvlan时要打开这样的限制。

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

两个 VLAN 之间位于独立的广播域，是完全二层隔离的，要通信就只能通过三层设备。假设位于 VLAN-A 中的主机 A1，希望把数据包发送给 VLAN-B 中的主机 B2，由于 A、B 两个 VLAN 之间二层链路不通，因此引入了单臂路由。单臂路由不属于任何 VLAN，它与交换机之间的链路允许任何 VLAN ID 的数据包通过，这种接口被称为 TRUNK。这样，A1 要和 B2 通信，A1 就把数据包先发送给路由（只需把路由设置为网关即可做到），然后路由根据数据包上的 IP 地址得知 B2 的位置，去掉 VLAN-A 的 VLAN Tag，改用 VLAN-B 的 VLAN Tag 重新封装数据包后，发回给交换机，交换机收到后就可以顺利转发给 B2 了。由于 A1、B2 各自处于独立的网段上，它们又各自要把同一个路由作为网关使用，这就要求路由器必须同时具备 192.168.1.0/24 和 192.168.2.0/24 的 IP 地址。当然，如果真的就只有 VLAN-A、VLAN-B 两个 VLAN，那把路由器上的两个接口分别设置不同的 IP 地址，然后用两条网线分别连接到交换机上，也勉强算是一个解决办法。但要知道，VLAN 最多可以支持 4096 个 VLAN，那如果要接四千多条网线就太离谱了。因此为了解决这个问题，802.1Q 规范中专门定义了子接口（Sub-Interface）的概念，它的作用是允许在同一张物理网卡上，针对不同的 VLAN 绑定不同的 IP 地址。

## vxlan

### 与vlan 对比

[云原生虚拟网络之 VXLAN 协议](https://mp.weixin.qq.com/s/z3W_RGoTTkbwuzxVqJ6-9g)

VLAN 有两个明显的缺陷
1. 第一个缺陷在于 VLAN Tag 的设计。VLAN 只有 12 位，共 4096 个。
2. 第二个缺陷：跨数据中心传递。VLAN 是一个二层网络技术，但是在两个独立数据中心之间信息只能够通过三层网络传递，云计算的发展普及很多业务有跨数据中心运作的需求，所以数据中心间传递 VLAN Tag 又是一件比较麻烦的事情；并且在虚拟网络中，一台物理机会有多个容器，容器与 VM 相比也是呈数量级增长，每个虚拟机都有独立的 IP 地址和 MAC 地址，这样带给交换机的压力也是成倍增加。

![](/public/upload/network/vxlan_frame.jpg)

[知乎：VXLAN vs VLAN](https://zhuanlan.zhihu.com/p/36165475)

[Virtual Extensible LAN](https://en.wikipedia.org/wiki/Virtual_Extensible_LAN)

1. vlan 和vxlan 都是 virtual lan(局域网)，但vlan 是隔离出来的，借助了交换机的支持（或者说推出太久了，以至于交换机普遍支持），vxlan 是虚拟出来的，交换机无感知。
2. VXLAN 对网络基础设施的要求很低，**不需要专门的硬件提供特别支持**，只要三层可达的网络就能部署 VXLAN。
3. **VLAN修改了原始的Ethernet Header，而VXLAN是将原始的Ethernet Frame隐藏在UDP数据里面**。VxLAN将 L2 的以太网帧（Ethernet frames）封装成 L4 的 UDP 数据包，然后在 L3 的网络中传输，效果就像 L2 的以太网帧在一个广播域中传输一样，实际上是跨越了 L3 网络，但却感知不到 L3 网络的存在。
4. VXLAN 网络的每个边缘入口上，布置有一个 VTEP（VXLAN Tunnel Endpoints）设备，它既可以是物理设备，也可以是虚拟化设备，主要负责 VXLAN 协议报文的封包和解包。经过VTEP封装之后，在网络线路上看起来只有VTEP之间的UDP数据传递，原始的网络数据包被掩盖了。

PS：应用进程本来发出 L2 包，通过路由规则发给 VTEP设备，VTEP 负责转为 UDP 重新发给 物理网卡。有点类似于 文件系统fuse的感觉。

**使用报文解耦二三层**

![](/public/upload/network/vxlan_vtep.png)

不过，VXLAN也带来了额外的复杂度和性能开销，具体表现为以下两点：
1. 传输效率的下降，经过 VXLAN 封装后的报文，新增加的报文头部分就整整占了 50 Bytes（VXLAN 报文头占 8 Bytes，UDP 报文头占 8 Bytes，IP 报文头占 20 Bytes，以太帧的 MAC 头占 14 Bytes），而原本只需要 14 Bytes 而已，而且现在这 14 Bytes 的消耗也还在，只是被封到了最里面的以太帧中。以太网的MTU是 1500 Bytes，如果是传输大量数据，额外损耗 50 Bytes 并不算很高的成本，但如果传输的数据本来就只有几个 Bytes 的话，那传输消耗在报文头上的成本就很高昂了。
2. 传输性能的下降，每个 VXLAN 报文的封包和解包操作都属于额外的处理过程，尤其是用软件来实现的 VTEP，要知道额外的运算资源消耗，有时候会成为不可忽略的性能影响因素。

### 一种新的隧道技术

[Kubernetes 网络插件学习之网络基础](https://mp.weixin.qq.com/s/yAkxR7YZY3uX5wow51Psig)隧道（Tunneling）是一种网络数据通信技术，主要解决网络协议不支持，数据传输不安全等网络通信问题。将不支持的协议数据包打包成支持的协议数据包之后进行传输，或在不安全网络上提供一个安全路径。通过网络隧道技术，可以使隧道两端的网络组成一个更大的内部网络。 隧道协议有二层隧道协议与三层隧道协议两类，二层隧道协议对应OSI模型中数据链路层，使用帧作为数据交换单位，将数据封装在点对点协议的帧中通过互联网络发送，协议包含PPTP、L2TP、L2F等。三层隧道协议对应OSI模型中网络层，使用包作为数据交换单位，将数据包封装在附加的IP包头中通过IP网络传送，协议包含GRE、IPSec、GRE等。Linux原生支持多种三层隧道，**其底层实现原理都是基于tun设备**。

Linux内核自3.7版本开始支持VXLAN隧道技术，将二层以太网帧封装在四层UDP报文中，通过三层网络传输，组成一个虚拟大二层网络。VXLAN使用VTEP（VXLAN Tunnel Endpoint）来进行封包和解包：在发送端，源VTEP将原始报文封装成VXLAN报文，通过UDP发送到对端VTEP；在接收端，VTEP将解开VXLAN报文，将原始的二层数据帧转发给目的的接收方。VTEP，VXLAN网络的边缘设备，用来进行VXLAN报文的封包与解包，VTEP可以是独立的网络设备，例如交换机，也可以是部署在服务器上的虚拟设备。例如使用置顶交换机（TOR）作为VTEP时，VXLAN的网络模型如下图：

![](/public/upload/network/vxlan_tor.png)

VXLAN的报文就是MAC in UDP，即在三层网络的基础上构建一个虚拟的二层网络。VXLAN的封包格式显示原来的二层以太网帧（包含MAC头部、IP头部和传输层头部的报文），被放在**VXLAN包头里**进行封装，再套到标准的UDP头部（UDP头部、IP头部和MAC头部），用来在底层网络上传输报文。VXLAN报文比原始报文多出50个字节，这降低了网络链路传输的有效数据比例，尤其是小包。**UDP目的端口是接收方VTEP设备使用的端口**（PS：感觉这是称之为隧道的关键），IANA分配了4789作为VXLAN的目的UDP端口（flannel　VXLAN模式是8472）。PS：正常是二层header + 三层header + 四层header + 四层数据，vxlan 下，四层数据变成了vxlan header + 二层数据包 ==> 二层header + 三层header + 四层header + vxlan header + 二层header + 三层header + 四层header + 四层数据。 

## macvlan 

macvlan模式是从一个物理网卡虚拟出多个虚拟网络接口，每个虚拟的接口都有单独的mac地址，可以给这些虚拟接口配置IP地址，在bridge模式下，父接口作为交换机来完成子接口间的通信，子接口可以通过父接口访问外网；

MACVLAN 借用了 VLAN 子接口的思路，并且在这个基础上更进一步，不仅允许对同一个网卡设置多个 IP 地址，还允许对同一张网卡上设置多个 MAC 地址，这也是 MACVLAN 名字的由来。原本 MAC 地址是网卡接口的“身份证”，应该是严格的一对一关系，而 MACVLAN 打破了这层关系。方法就是在物理设备之上、网络栈之下生成多个虚拟的 Device，每个 Device 都有一个 MAC 地址，新增 Device 的操作本质上相当于在系统内核中，注册了一个收发特定数据包的回调函数，每个回调函数都能对一个 MAC 地址的数据包进行响应，当物理设备收到数据包时，会先根据 MAC 地址进行一次判断，确定交给哪个 Device 来处理，如下图所示。

![](/public/upload/network/macvlan.png)

用 MACVLAN 技术虚拟出来的副本网卡，在功能上和真实的网卡是完全对等的，此时真正的物理网卡实际上也确实承担着类似交换机的职责。在收到数据包后，物理网卡会根据目标 MAC 地址，判断这个包应该转发给哪块副本网卡处理，由同一块物理网卡虚拟出来的副本网卡，天然处于同一个 VLAN 之中，因此可以直接二层通信，不需要将流量转发到外部网络。而除了模拟交换机的 Bridge 模式外，MACVLAN 还支持虚拟以太网端口聚合模式（Virtual Ethernet Port Aggregator，VEPA）、Private 模式、Passthru 模式、Source 模式等另外几种工作模式。

[Macvlan and IPvlan basics](https://sreeninet.wordpress.com/2016/05/29/macvlan-and-ipvlan/)Macvlan and ipvlan are Linux network drivers that exposes underlay or host interfaces directly to VMs or Containers running in the host. 

[Kubernetes在信也科技的落地实战](https://mp.weixin.qq.com/s/OBxnAitZaoI0lbP219Fvwg)Macvlan是一种直连网络，数据包通过宿主机网卡传输到硬件链路上，通过交换机路由器等设备最终到达目的地。Macvlan网络是Linux内核原生支持，不需要部署额外的组件。本身包含VLAN特性，一台宿主机上面可以虚拟出多块VLAN网卡，支持多个C类地址的IP分配。此时宿主机和交换机的链路必须是trunk模式，同一链路根据不同报文内的vlanID如（10、20、30）组成逻辑信道，互不干扰。

![](/public/upload/network/macvlan_network.png)

采用Macvlan网络模式之后，容器里面的网络协议栈和宿主机是完全独立。这就导致容器不能使用宿主机的iptables规则从而容器里面无法通过ClusterIP去访问Service。

### 与 mac based vlan 区别

要跟mac based vlan 有所区分，参见[虚拟网络](http://qiankunli.github.io/2015/04/24/virtual_network.html)。

Macvlan, MACVLAN or MAC-VLAN allows you to configure multiple Layer 2 (i.e. Ethernet MAC) addresses **on a single physical interface**. Macvlan allows you to configure sub-interfaces (also termed slave devices) of a parent, physical Ethernet interface (also termed upper device), each with its own unique (randomly generated) MAC address, and consequently its own IP address. Applications, VMs and containers can then bind to a specific sub-interface to connect directly to the physical network, using their own MAC and IP address. 基于物理机网卡 physical interface 创建多个 sub-interface，拥有自己的MAC and IP ，直接接入physical network。

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

## ipvlan

与macvlan类似，ipvlan也是在一个物理网卡上虚拟出多个子接口，与macvlan不同的是，ipvlan的每一个子接口的mac地址是一样的，IP地址不同；ipvlan支持L2和L3模式。
1. ipvlan L2 模式和 macvlan bridge 模式工作原理很相似，父接口作为交换机来转发子接口的数据。同一个网络的子接口可以通过父接口来转发数据，而如果想发送到其他网络，报文则会通过父接口的路由转发出去。
2. L3 模式下，ipvlan 有点像路由器的功能，它在各个虚拟网络和主机网络之间进行不同网络报文的路由转发工作。只要父接口相同，即使虚拟机/容器不在同一个网络，也可以互相 ping 通对方，因为 ipvlan 会在中间做报文的转发工作。L3 模式下的虚拟接口不会接收到多播或者广播的报文，为什么呢？这个模式下，所有的网络都会发送给父接口，所有的 ARP 过程或者其他多播报文都是在底层的父接口完成的。

需要注意的是：外部网络默认情况下是不知道 ipvlan 虚拟出来的网络的，如果不在外部路由器上配置好对应的路由规则，ipvlan 的网络是不能被外部直接访问的。


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

## vlan/macvlan/ipvlan 底层实现

[从 VLAN 到 IPVLAN: 聊聊虚拟网络设备及其在云原生中的应用](https://mp.weixin.qq.com/s/eGLvJHW5AbQx7KqynSIpLw)
1. VLAN全称是 Virtual Local Area Network，用于在以太网中隔离不同的广播域。它诞生的时间很早，1995 年，IEEE 就发表了 802.1Q 标准定义了在以太网数据帧中 VLAN 的格式，并且沿用至今。VLAN 原本和 bridge 一样是一个交换机上的概念，不过 Linux 将它们都进行了软件的实现。Linux 在每个以太网数据帧中使用一个 16bit 的 vlan_proto 字段和 16bit 的 vlan_tci 字段实现 802.1q 协议，同时对于每一个 VLAN，都会虚拟出一个子设备来处理去除 VLAN 之后的报文，没错 VLAN 也有属于自己的子设备，即 VLAN sub-interface，不同的 VLAN 子设备通过一个主设备进行物理上的报文收发。
1. 内核对网络设备的抽象（无论是真实的还是虚拟的）
	1. 内核会对每一类网络设备抽象出一个专门响应 netlink 消息的结构体，它们都按照 rtnl_link_ops 结构来实现，用于响应对网络设备的创建，销毁和修改。例如比较直观的 Veth 设备：
		```C
		static struct rtnl_link_ops veth_link_ops = {
			.kind   = DRV_NAME,
			.priv_size  = sizeof(struct veth_priv),
			.setup    = veth_setup,
			.validate = veth_validate,
			.newlink  = veth_newlink,
			.dellink  = veth_dellink,
			.policy   = veth_policy,
			.maxtype  = VETH_INFO_MAX,
			.get_link_net = veth_get_link_net,
			.get_num_tx_queues  = veth_get_num_queues,
			.get_num_rx_queues  = veth_get_num_queues,
		};
		```

	2. 对于一个网络设备来说，Linux 的操作和硬件设备的响应本身也需要一套规范，Linux 将其抽象为 net_device_ops 这个结构体。依然以 Veth 设备为例：
		```C
		static const struct net_device_ops veth_netdev_ops = {
			.ndo_init            = veth_dev_init,
			.ndo_open            = veth_open,
			.ndo_stop            = veth_close,
			.ndo_start_xmit      = veth_xmit,		// 发送数据包
			.ndo_get_stats64     = veth_get_stats64,
			.ndo_set_rx_mode     = veth_set_multicast_list,
			.ndo_set_mac_address = eth_mac_addr,
			#ifdef CONFIG_NET_POLL_CONTROLLER
			.ndo_poll_controller  = veth_poll_controller,
			#endif
			.ndo_get_iflink   = veth_get_iflink,
			.ndo_fix_features = veth_fix_features,
			.ndo_set_features = veth_set_features,
			.ndo_features_check = passthru_features_check,
			.ndo_set_rx_headroom  = veth_set_rx_headroom,
			.ndo_bpf    = veth_xdp,
			.ndo_xdp_xmit   = veth_ndo_xdp_xmit,
			.ndo_get_peer_dev = veth_peer_dev,
		};
		```

1. 对于接收数据包，Linux 的收包动作并不是由各个进程自己去完成的，而是由 ksoftirqd 内核线程负责了从驱动接收、网络层（ip，iptables）、传输层（tcp，udp）的处理，最终放到用户进程持有的 Socket 的 recv 缓冲区中，然后由内核 inotify 用户进程处理。对于虚拟设备来说，所有的差异集中于网络层之前，在这里有一个统一的入口，即__netif_receive_skb_core。__netif_receive_skb_core ==> xx(将 VLAN 信息从数据包中提取等) ==> vlan_do_receive ==> another_round 重新按照正常数据包的流程执行一次__netif_receive_skb_core，按照正常包的处理逻辑进入，进入了 rx_handler 的处理，就像一个正常的数据包一样，在子设备上通过与主设备相同的 rx_handler 进入到网络层。

	![](/public/upload/network/vlan_sub_interface_receive.png)
2. VLAN 子设备的数据发送的入口是 vlan_dev_hard_start_xmit，在硬件发送时，VLAN 子设备会进入 vlan_dev_hard_start_xmit 方法，这个方法实现了 ndo_start_xmit 接口，它通过__vlan_hwaccel_put_tag 方法填充 VLAN 相关的以太网信息到报文中，然后**修改了报文的设备为主设备**，调用主设备的 dev_queue_xmit 方法重新进入主设备的发送队列进行发送

	![](/public/upload/network/vlan_sub_interface_send.png)
3. **macvlan 和 ipvlan 也有上述的类似的结构体和方法**。VLAN 子设备和 MACVlan，IPVlan 的核心逻辑很相似：
	1. 主设备负责物理上的收发包。
	2. 主设备将子设备管理为多个 port，然后根据一定的规则找到 port，比如 VLAN 信息，mac 地址以及 ip 地址（macvlan_hash_lookup，vlan_find_dev，ipvlan_addr_lookup）。
	3. 主设备收包后，都需要经过在__netif_receive_skb_core 中走一段“回头路”.
	4. 子设备**发包**最终都是直接通过修改报文的 dev，然后让主设备去操作。


[Macvlan和IPvlan基础知识](https://mp.weixin.qq.com/s/r_CuqjypaaMRDZfW-RHjxw)
1. 使用Bridge时，必须需要使用NAT进行外部连接。
1. 对于vlan子接口，每个子接口使用vlan属于不同的L2域，并且所有子接口具有相同的MAC地址。
2. 使用macvlan，每个子接口都会获得唯一的mac和ip地址，并**直接暴露在底层网络中**。Macvlan接口通常用于虚拟化应用程序，每个macvlan接口都连接到一个Container或VM。每个容器或VM都可以像主机一样直接从公共服务器获取dhcp地址。这将有助于已经拥有IP寻址方案的Containers成为其传统网络的一部分的客户。
4. ipvlan与macvlan类似，不同之处在于端点具有相同的mac地址。ipvlan支持L2和L3模式。在ipvlan l2模式下，每个端点获取相同的MAC地址但不同的IP地址。在ipvlan l3模式下，数据包可在端点之间路由，因此提供了更好的可扩展性。
3. 在某些交换机，由于端口安全的原因而限制了每个物理端口的最大MAC地址数，在这种情况下，应使用ipvlan。在使用普通dhcp服务器的情况下则需要使用macvlan，因为dhcp服务器要求每个子接口必须要有唯一的mac地址，这种情况下显然ipvlan不满足要求。

## 其它

什么是network driver?

A network device driver is a device driver that enables a network device to communicate between the computer and operating system as well as with other network computers and network devices.

[Device driver](https://en.wikipedia.org/wiki/Device_driver)In computing, a device driver is a computer program that operates or controls a particular type of device that is attached to a computer. **A driver provides a software interface to hardware devices**, enabling operating systems and other computer programs to access hardware functions without needing to know precise details about the hardware being used. 驱动就是对硬件提供软件接口，屏蔽硬件细节。

A driver communicates with the device through the computer bus or communications subsystem to which the hardware connects. When a calling program invokes a routine in the driver, the driver issues commands to the device. Once the device sends data back to the driver, the driver may invoke routines in the original calling program. Drivers are hardware dependent and operating-system-specific. They usually provide the interrupt handling required for any necessary asynchronous time-dependent hardware interface.


网卡 ==> computer bus ==> network driver ==> Subroutine/子程序 ==> calling program。也就是network driver 在网卡 与操作系统之间，从这个角度看，跟磁盘驱动、鼠标驱动类似了。
