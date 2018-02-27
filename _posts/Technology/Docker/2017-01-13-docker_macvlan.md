---

layout: post
title: docker macvlan实践
category: 技术
tags: Docker
keywords: Docker,macvlan

---


## 简介

## macvlan基础

如果采用硬件支持的方式来设置vlan，交换机是划分局域网的关键设备，所以本文说xx vlan，主要是针对交换机说的。

交换机，维基百科解释：是一个扩大网络的器材，能为子网络中提供更多的port，以便连接更多的电脑。

### macvlan

常用的 VLAN 划分方式是通过端口进行划分，虽然这种划分 VLAN 的方式设置比较很简单， 但仅适用于终端设备物理位置比较固定的组网环境。随着移动办公的普及，终端设备可能不 再通过固定端口接入交换机，这就会增加网络管理的工作量。比如，一个用户可能本次接入 交换机的端口 1，而下一次接入交换机的端口 2，由于端口 1 和端口 2 属于不同的 VLAN，若 用户想要接入原来的 VLAN 中，网管就必须重新对交换机进行配置。显然，这种划分方式不 适合那些需要频繁改变拓扑结构的网络。而 MAC VLAN 则可以有效解决这个问题，它根据 终端设备的 MAC 地址来划分 VLAN。这样，即使用户改变了接入端口，也仍然处在原 VLAN 中。

mac vlan不是以交换机端口来划分vlan，因此，一个交换机端口可以接受来自多个mac地址的数据。一个交换机端口要处理多个vlan的数据，则要设置trunk模式。

### 交换机IP-MAC-PORT

网络中实际传输的是“帧”，帧里面是有目标主机的MAC地址的。在以太网中，一个主机要和另一个主机进行直接通信，必须要知道目标主机的MAC地址。但这个目标MAC地址是如何获得的呢？它就是通过地址解析协议获得的。所谓“地址解析”就是主机在发送帧前将目标IP地址转换成目标MAC地址的过程。ARP协议的基本功能就是通过目标设备的IP地址，查询目标设备的MAC地址，以保证通信的顺利进行。

ARP欺骗，当计算机接收到ARP应答数据包的时候，就会对本地的ARP缓存进行更新，将应答中的IP和MAC地址存储在ARP缓存中。但是，**ARP协议并不只在发送ARP请求才接收ARP应答。**ARP应答可以不请自来，有人发送一个自己伪造的ARP应答(比如错误的ip-mac映射)，网络可能就会出现问题，这是协议的设计者当初没考虑到的。

在交换机上配置了IMP(ip-mac-port映射)功能以后，交换机会检查每个数据包的源IP地址和MAC，对于没有在交换机内记录的IP和MAC地址的计算机所发出的数据包都会被交换机所阻止。ip-mac-port映射静态设置比较麻烦，可以开启交换机上面的DHCP SNOOPING功能， DHCP Snooping可以自动的学习IP和MAC以及端口的配对，并将学习到的对应关系保存到交换机的本地数据库中。

默认情况下，交换机上每个端口只允许绑定一个IP-MAC条目，所以在使用docker macvlan时要打开这样的限制。


## 整体思路

以下实现基于docker1.13，物理机使用`192.168.0.0/16`网段，容器使用`172.31.0.0/16`网段。

1. docker host，自定义ipam plugin负责ip地址管理，每个docker host运行一个ipam plugin，并根据ipam plugin创建local scope的macvlan network。
2. 创建容器时使用macvlan网络
3. 外置交换机负责容器之间、host之间、容器与host之间的连通性。

MACVLAN可以从一个主机接口虚拟出多个macvtap，且每个macvtap设备都拥有不同的mac地址（对应不同的linux字符设备）。

docker macvlan 用802.1q模式，对于一个交换机端口来说：

1. 物理机和容器的数据包属于不同的vlan，so， 交换机端口设置为trunk；
2. 物理机和容器的数据包属于不同的网段，so，在交换机的三层加一层路由，打通物理机和容器的两个网段。


## macvlan网络

### 物理机创建vlan的sub interface

使用802.1q vlan时，我们发出去的数据包，要有802.1q中的vlan tag。为了不影响物理网卡的正常使用，就是只有基于sub interface（eth1.10）来发送802.1q package。

1. Load the 802.1q module into the kernel.

	`sudo modprobe 8021q`

2. **Create a new interface that is a member of a specific VLAN**, 
VLAN id 10 is used in this example. Keep in mind you can only use physical interfaces as a base, creating VLAN's on virtual interfaces (i.e. eth0:1) will not work. We use the physical interface eth1 in this example. This command will add an additional interface next to the interfaces which have been configured already, so your existing configuration of eth1 will not be affected.
	
	`sudo vconfig add eth1 10`

3. Assign an address to the new interface.

	`sudo ip addr add 10.0.0.1/24 dev eth0.10`

4. Starting the new interface.

	`sudo ip link set up eth0.10`
	

	
### 基于sub interface创建docker macvlan 网络

	docker network  create  -d macvlan \
	    --subnet=172.31.0.0/16 \
	    --gateway=172.31.0.1 \
	    -o parent=eth0.10 macvlan10

### 创建容器，指定使用macvlan网络

	docker run --net=macvlan10 -it --name macvlan_test5 --rm alpine /bin/sh
	
	
## 引用

[MAC VLAN 配置简介 ](http://service.tp-link.com.cn/download/20155/MAC%20VLAN%E9%85%8D%E7%BD%AE%E6%8C%87%E5%8D%971.0.0.pdf)

[交换机IP-MAC-PORT绑定和DHCP Snooping的应用](http://yonggang.blog.51cto.com/94083/109150/)