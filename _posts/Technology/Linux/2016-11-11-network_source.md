---

layout: post
title: Linux网络源代码学习
category: 技术
tags: Linux
keywords: network 

---

## 简介（未完待续）

以下来自linux1.2.13源码

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