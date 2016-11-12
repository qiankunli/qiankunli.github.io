---

layout: post
title: Linux网络源代码学习
category: 技术
tags: Linux
keywords: network 

---

## 简介

以下来自linux1.2.13源码，算是参见[Linux1.0](https://github.com/wanggx/Linux1.0.git)的学习笔记。

linux的网络部分由网卡的驱动程序和kernel的网络协议栈部分组成，它们相互交互，完成数据的接收和发送。

首先，我们从device struct开始。struct反映了很多东西，比如看一下linux的进程struct，就很容易理解进程为什么能干那么多事情。

linux会维护一个device struct list，通过它能找到所有的网络设备。device struct 和设备不是一对一关系。

## 网络驱动程序
	
	include/linux/netdevice.h
	struct device{
		/*
		* This is the first field of the "visible" part of this structure
		* (i.e. as seen by users in the "Space.c" file). It is the name
		* the interface.
		*/
		char *name;
		/* I/O specific fields - FIXME: Merge these and struct ifmap into one */
		unsigned long rmem_end; /* shmem "recv" end */
		unsigned long rmem_start; /* shmem "recv" start */
		unsigned long mem_end; /* shared mem end */
		unsigned long mem_start; /* shared mem start */
		// device 只是一个struct，可能几个struct共用一个物理网卡
		unsigned long base_addr; /* device I/O address */
		// 赋给中断号
		unsigned char irq; /* device IRQ number */
		/* Low-level status flags. */
		volatile unsigned char start, /* start an operation */
			tbusy, /* transmitter busy */
			interrupt; /* interrupt arrived */
	
		struct device *next;
		/* The device initialization function. Called only once. */
		// 初始化函数
		int (*init)(struct device *dev);
		/* Some hardware also needs these fields, but they are not part of the
		usual set specified in Space.c. */
		unsigned char if_port; /* Selectable AUI, TP,..*/
		unsigned char dma; /* DMA channel */
		struct enet_statistics* (*get_stats)(struct device *dev);
		/*
		* This marks the end of the "visible" part of the structure. All
		* fields hereafter are internal to the system, and may change at
		* will (read: may be cleaned up at will).
		*/
		/* These may be needed for future network-power-down code. */
		unsigned long trans_start; /* Time (in jiffies) of last Tx */
		unsigned long last_rx; /* Time of last Rx */
		unsigned short flags; /* interface flags (a la BSD) */
		unsigned short family; /* address family ID (AF_INET) */
		unsigned short metric; /* routing metric (not used) */
		unsigned short mtu; /* interface MTU value */
		unsigned short type; /* interface hardware type */
		unsigned short hard_header_len; /* hardware hdr length */
		void *priv; /* pointer to private data */
		/* Interface address info. */
		unsigned char broadcast[MAX_ADDR_LEN]; /* hw bcast add */
		unsigned char dev_addr[MAX_ADDR_LEN]; /* hw address */
		unsigned char addr_len; /* hardware address length */
		unsigned long pa_addr; /* protocol address */
		unsigned long pa_brdaddr; /* protocol broadcast addr */
		unsigned long pa_dstaddr; /* protocol P-P other side addr */
		unsigned long pa_mask; /* protocol netmask */
		unsigned short pa_alen; /* protocol address length */
		struct dev_mc_list *mc_list; /* Multicast mac addresses */
		int mc_count; /* Number of installed mcasts*/
		struct ip_mc_list *ip_mc_list; /* IP multicast filter chain */
		/* For load balancing driver pair support */
		unsigned long pkt_queue; /* Packets queued */
		struct device *slave; /* Slave device */
		// device的数据缓冲区
		/* Pointer to the interface buffers. */
		struct sk_buff_head buffs[DEV_NUMBUFFS];
		/* Pointers to interface service routines. */
		// 打开设备
		int (*open)(struct device *dev);
		// 关闭设备
		int (*stop)(struct device *dev);
		// 调用具体的硬件将数据发到物理介质上，网络栈最终调用它发数据
		int (*hard_start_xmit) (struct sk_buff *skb, struct device *dev);
		int (*hard_header) (unsigned char *buff,struct device *dev,unsigned short type,void *daddr,void *saddr,unsigned len,struct sk_buff *skb);
		int (*rebuild_header)(void *eth, struct device *dev,unsigned long raddr, struct sk_buff *skb);
		unsigned short (*type_trans) (struct sk_buff *skb, struct device *dev);
		#define HAVE_MULTICAST
		void (*set_multicast_list)(struct device *dev, int num_addrs, void *addrs);
		#define HAVE_SET_MAC_ADDR
		int (*set_mac_address)(struct device *dev, void *addr);
		#define HAVE_PRIVATE_IOCTL
		int (*do_ioctl)(struct device *dev, struct ifreq *ifr, int cmd);
		#define HAVE_SET_CONFIG
		int (*set_config)(struct device *dev, struct ifmap *map);
	};

耐心的看完这个结构体，网络部分的初始化就是围绕device struct的创建及其中字段（和函数）的初始化.

linux内核与网络驱动程序的边界：

linux内核准备好device struct和dev_base指针(这句不准确，或许是ethdev_index[])，kernel启动时，执行驱动程序事先挂好的init函数，init函数初始化device struct并挂到dev_base上(或ethdev_index上)。

ei开头的都是驱动程序自己的函数。

1. 接收数据，device struct 初始化时，会为这个设备生成一个irq(中断号)，为irq其绑定ei_interrutp（网卡的中断处理函数），同时会建立一个irq与device的映射。接收到数据后，触发ei_interrutp, ei_interrutp根据中断号得到device,执行ei_receive（device), ei_receive 将数据拷贝到某个位置，执行内核的netif_rx,netif_rx执行net_bh，net_bh将数据包传递给网络层协议接收函数比如arp_rcv,ip_rcv.

2. 发送数据，由网络协议栈调用hard_start_xmit(初始化时，驱动程序将ei_start_xmit函数挂到其上)

总的来说，kernel有几个extern的struct、pointer和func，驱动程序初始化完毕后，为linux内核准备了一个device struct list（驱动程序自己有一些功能函数，挂到device struct的函数成员上）。收到数据时，**kernel的extern func(比如netif_rx)在中断环境下被驱动程序调用**。发送数据时，则由内核网络协议栈调用device.hard_start_xmit，进而执行驱动程序函数。

## 网络协议栈

socket分为多种，除了inet还有unix。反应在代码结构上，就是net包下只有net/unix,net/inet两个文件夹。之所以叫unix域，可能跟描述其地址时，使用`unix://xxx`有关

The difference is that an INET socket is bound to an IP address-port tuple, while a UNIX socket is "bound" to a special file on your filesystem. Generally, only processes running on the same machine can communicate through the latter.

本文重点是inet,net/inet下有以下几个比较重要的文件，这跟网络书上的知识就对上了。

	arp.c
	eth.c
	ip.c
	route.c
	tcp.c
	udp.c
	datalink.h		// 应该是数据链路层

|各层之间的桥梁|备注|
|---|---|
|应用层||
|struct socket||
|BSD socket|socket函数集，比如socket、bind、accept|
|struct net_proto|inet,unix,ipx等|
|INET||
|struct proto|tcp_proto,udp_proto,raw_proto|
|传输层||
|struct inet_protocol,header为inet_protocol_base|tcp_protocol,udp_protocol ,icmp_protocol,igmp_protocol|
|网络层||
|struct packet_type，header为ptype_base|ip_packet_type,arp_packet_type|
|链路层||
|struct device|loopback_dev等|
|驱动层|  |

怎么理解整个表格呢？以ip.c为例，在该文件中定义了ip_rcv、ip_queue_xmit(用于写数据)，链路层收到数据后，通过ptype_base找到ip_packet_type,进而执行ip_rcv。tcp发送数据时，通过tcp_proto找到ip_queue_xmit并执行。

tcp_protocol是为了从下到上的数据接收，tcp_proto是为了从上到下的数据发送。为什么卡在传输层呢，因为在传输层在开始真正进行用户数据的处理（归功于tcp的复杂和重要，比如拥塞控制等）

主要要搞清楚三个问题，具体可以参见相关的书籍，此处不详述。参见[Linux1.0](https://github.com/wanggx/Linux1.0.git)中的Linux1.2.13内核网络栈源码分析的第四章节。

1. 这些结构如何初始化。有的结构直接就是定义好的，比如tcp_protocol等
2. 如何接收数据。由中断程序触发。接收数据的时候，可以得到device，从数据中可以取得协议数据，进而从ptype_base及inet_protocol_base执行相应的rcv
3. 如何发送数据。通常不直接发送，先发到queue里。可以从socket初始化时拿到protocol类型（比如tcp）、目的ip，通过route等决定device，于是一路向下执行xx_write方法

## 小结

**重要的不是细节**，这个过程让我想到了web编程中的controller,service,dao。都是分层，区别是web请求要立即返回，网络通信则不用。

1. mac ==> device  ==> ip_rcv ==> tcp_rcv ==> 上层
2. url ==》 controller ==> service ==> dao ==> 数据库

想一想，整个网络协议栈，其实就是一群loopbackController、eth0Controller、ipService、TcpDao组成，该是一件多么有意思的事。

|类别|依赖关系的存储或表示|如何找依赖|依赖关系建立的时机是集中的|
|---|---|---|---|
|web|由spring管理，springmvc建立`<url,beanid>`,ioc建立`<beanId,bean>`|根据request信息及自身逻辑决定一步步如何往下走。|依赖关系建立的代码是集中的|
|linux|所谓的“依赖关系”是通过一个个struct及其数组（或链表）header，下层持有上层的struct header以完成接收，发送时则直接指定下层函数|接收时根据packet的一些字段，发送时根据socket参数及路由|依赖关系建立的代码是分散的，就好比有个全局的map，所有service(或者dao)自己向map注入自己的信息|