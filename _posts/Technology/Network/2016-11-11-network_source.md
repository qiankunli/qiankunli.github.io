---

layout: post
title: Linux网络源代码学习
category: 技术
tags: Network
keywords: network 

---

## 简介

以下来自linux1.2.13源码，算是参见[Linux1.0](https://github.com/wanggx/Linux1.0.git)的学习笔记。

linux的网络部分由网卡的驱动程序和kernel的网络协议栈部分组成，它们相互交互，完成数据的接收和发送。


## 网络操作的开始

	linux-1.2.13
	|
	|---net
		  |
		  |---protocols.c
		  |---socket.c
		  |---unix
		  |	     |
		  |	     |---proc.c
		  |	     |---sock.c
		  |	     |---unix.h
		  |---inet
		  		  |
		  		  |---af_inet.c
		  		  |---arp.h,arp.c
		  		  |---...
		  		  |---udp.c,utils.c
		  		 

其中 unix 子文件夹中三个文件是有关 UNIX 域代码， UNIX 域是模拟网络传输方式在本机范围内用于进程间数据传输的一种机制。

系统调用通过 INT $0x80 进入内核执行函数，该函数根据 AX 寄存器中的系统调用号，进一步调用内核网络栈相应的实现函数。

file_operations 结构定义了普通文件操作函数集。系统中每个文件对应一个 file 结构， file 结构中有一个 file_operations 变量，当使用 write，read 函数对某个文件描述符进行读写操作时，系统首先根据文件描述符索引到其对应的 file 结构，然后调用其成员变量 file_operations 中对应函数完成请求。

	// 参见socket.c
	static struct file_operations socket_file_ops = {
		sock_lseek,		// Î´ÊµÏÖ
		sock_read,
		sock_write,
		sock_readdir,	// Î´ÊµÏÖ
		sock_select,
		sock_ioctl,
		NULL,			/* mmap */
		NULL,			/* no special open code... */
		sock_close,
		NULL,			/* no fsync */
		sock_fasync
	};
	
以上 socket_file_ops 变量中声明的函数即是**网络协议对应的普通文件操作函数集合**。从而使得read， write， ioctl 等这些常见普通文件操作函数也可以被使用在网络接口的处理上。kernel维护一个struct file list，通过fd ==> struct file ==> file->ops ==> socket_file_ops,便可以以文件接口的方式进行网络操作。同时，每个 file 结构都需要有一个 inode 结构对应。用于存储struct file的元信息

	struct inode{
		...
		union {
		   ...
			struct ext_inode_info ext_i;
			struct nfs_inode_info nfs_i;
			struct socket socket_i;
		}u
	}

也就是说，对linux系统，一切皆文件，由struct file描述，通过file->ops指向具体操作，由file->inode 存储一些元信息。对于ext文件系统，是载入内存的超级块、磁盘块等数据。对于网络通信，则是待发送和接收的数据块、网络设备等信息。从这个角度看，**struct socket和struct ext_inode_info 等是类似的。**


![](/public/upload/linux/linux_network.png)

从这个图中，可以看到，到传输层时，横生枝节，**代码不再针对任何数据包都通用**。从上到下，数据包的发送使用什么传输层协议，由socket初始化时确定。从下到上，收到的数据包由哪个传输层协议处理，根据从数据包传输层header中解析的数据确定。


## 网络驱动程序

首先，我们从device struct开始。struct反映了很多东西，比如看一下linux的进程struct，就很容易理解进程为什么能干那么多事情。

linux会维护一个device struct list，通过它能找到所有的网络设备。device struct 和设备不是一对一关系。
	
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

1. 接收数据，device struct 初始化时，会为这个设备生成一个irq(中断号)，为irq其绑定ei_interrutp（网卡的中断处理函数），同时会建立一个irq与device的映射。接收到数据后，触发ei_interrutp, ei_interrutp根据中断号得到device,执行`ei_receive（device)`, ei_receive 将数据拷贝到 数据接收队列（元素为 sk_buff，具有prev和next指针，struct device 维护了 sk_buff_head），执行内核的netif_rx,netif_rx 触发软中断 执行net_bh，net_bh 遍历 packet_type list 查看数据 符合哪个协议（不是每次都遍历），执行`packet_type.func`将数据包传递给网络层协议接收函数，`packet_type.func` 的可选值 arp_rcv,ip_rcv. ip_rcv中带有device 参数，用于校验数据包的mac 地址是否在 device.mc_list 之内，及检查是否开启IP_FORWARD等。

2. 发送数据，由网络协议栈调用hard_start_xmit(初始化时，驱动程序将ei_start_xmit函数挂到其上)

总的来说，kernel有几个extern的struct、pointer和func，驱动程序初始化完毕后，为linux内核准备了一个device struct list（驱动程序自己有一些功能函数，挂到device struct的函数成员上）。收到数据时，**kernel的extern func(比如netif_rx)在中断环境下被驱动程序调用**。发送数据时，则由内核网络协议栈调用device.hard_start_xmit，进而执行驱动程序函数。

[Linux TCP/IP 协议栈源码分析](https://www.cnblogs.com/my_life/articles/4691254.html) 接收发送过程的详图

![](/public/upload/linux/network_source_send.gif)

![](/public/upload/linux/network_source_recv.gif)



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


||数据struct|数据struct 例子|协议struct|协议struct例子|
|---|---|---|---|---|
|应用层|struct file||struct file_operations|struct socket_file_ops|
|bsd socket 层|struct socket,struct sk_buff||struct proto_ops,struct net_proto(for init)|inet,unix,ipx等||
|inet socket 层|struct sock||struct prot|tcp_prot,udp_prot,raw_prot|
|传输层|||struct inet_protocol|tcp_protocol,udp_protocol ,icmp_protocol,igmp_protocol|
|网络层|||struct packet_type |ip_packet_type,arp_packet_type|
|链路层| device | loopback_dev |||
|硬件层|||

怎么理解整个表格呢？协议struct和数据struct有何异同？

1. struct一般由一个数组或链表组织，数组用index，链表用header(比如packet_type_base、inet_protocol_base)指针查找数据。
2. **协议struct是怎么回事呢？通常是一个函数操作集，类似于controller-server-dao之间的interface定义**，类似于本文开头的file_operations，有open、close、read等方法，但对ext是一回事，对socket操作是另一回事。
3. 数据struct实例可以有很多，比如一个主机有多少个连接就有多少个struct sock，而协议struct个数由协议类型个数决定，具体的协议struct比如tcp_prot就只有一个。比较特别的是，通过tcp_prot就可以找到所有的struct sock实例。

以ip.c为例，在该文件中定义了ip_rcv（读数据）、ip_queue_xmit(用于写数据)，链路层收到数据后，通过ptype_base找到ip_packet_type,进而执行ip_rcv。tcp发送数据时，通过tcp_proto找到ip_queue_xmit并执行。

tcp_protocol是为了从下到上的数据接收，其函数集主要是handler、frag_handler和err_handler，对应数据接收后的处理。tcp_prot是为了从上到下的数据发送(所以struct proto没有icmp对应的结构)，其函数集connect、read等主要对应上层接口方法。

到bsd socket层，相关的结构在`/include/linux`下定义，而不是在net包下。这就对上了，bsd socket是一层接口规范，而net包下的相关struct则是linux自己的抽象了。

主要要搞清楚三个问题，具体可以参见相关的书籍，此处不详述。参见[Linux1.0](https://github.com/wanggx/Linux1.0.git)中的Linux1.2.13内核网络栈源码分析的第四章节。

1. 这些结构如何初始化。有的结构直接就是定义好的，比如tcp_protocol等
2. 如何接收数据。由中断程序触发。接收数据的时候，可以得到device，从数据中可以取得协议数据，进而从ptype_base及inet_protocol_base执行相应的rcv
3. 如何发送数据。通常不直接发送，先发到queue里。可以从socket初始化时拿到protocol类型（比如tcp）、目的ip，通过route等决定device，于是一路向下执行xx_write方法


## 面向过程/对象/ioc

**重要的不是细节**，这个过程让我想到了web编程中的controller,service,dao。都是分层，区别是web请求要立即返回，网络通信则不用。

1. mac ==> device  ==> ip_rcv ==> tcp_rcv ==> 上层
2. url ==》 controller ==> service ==> dao ==> 数据库

想一想，整个网络协议栈，其实就是一群loopbackController、eth0Controller、ipService、TcpDao组成，该是一件多么有意思的事。

|类别|依赖关系的存储或表示|如何找依赖|依赖关系建立的时机是集中的|
|---|---|---|---|
|web|由spring管理，springmvc建立`<url,beanid>`,ioc建立`<beanId,bean>`|根据request信息及自身逻辑决定一步步如何往下走。|依赖关系建立的代码是集中的|
|linux|所谓的“依赖关系”是通过一个个struct及其数组（或链表）header，下层持有上层的struct header以完成接收，发送时则直接指定下层函数|接收时根据packet的一些字段，发送时根据socket参数及路由|依赖关系建立的代码是分散的，就好比有个全局的map，所有service(或者dao)自己向map注入自己的信息|

当然，c 毕竟 不是java，也不面向对象。比如struct device。

1. 在接收过程中，更多的是作为一个数据载体存在的，在网卡驱动层 将device 包到 sk_buff->device 数据中传入 ip 层，在ip 层（ip_rcv） 被引用来 比对数据包的mac 地址是否 跟device 匹配。当然，毕竟struct device的类型不同，在接收数据的过程中，会调用device 的某些函数，以适配不同类型设备的处理逻辑。
2. 在数据发送阶段，就有点向“对象”, device. hard_start_xmit 调用 驱动程序发送数据。

在数据接收阶段，主要是面向过程的思路， 网卡驱动 ==> 接收缓冲区 ==> 软中断 ==> 网络层及 更上层校验处理 ==> socket 缓冲区。

1. 所以，从数据的接收看，一个linux 有多少个 struct device 没关系，可以分成两类：有 没有 irq 跟 struct device 关联。没有的话，就只能通过网络层 转发 来触发 该device 相关的数据 接收过程。
2. 网络接收 分成两个部分：驱动程序 驱动 代码将数据 写到 socket 缓冲区。上层调用 驱动代码 从socket 缓冲区 读取数据。