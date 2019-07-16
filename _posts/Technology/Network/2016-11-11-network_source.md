---

layout: post
title: Linux网络源代码学习——整体介绍
category: 技术
tags: Network
keywords: network 

---

## 简介

* TOC
{:toc}

以下来自linux1.2.13源码，算是参见[Linux1.0](https://github.com/wanggx/Linux1.0.git)的学习笔记。

linux的网络部分由网卡的驱动程序和kernel的网络协议栈部分组成，它们相互交互，完成数据的接收和发送。

## 源码目录

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

## 网络与文件操作

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

## 网络分层

![](/public/upload/linux/linux_network.png)

从这个图中，可以看到，到传输层时，横生枝节，**代码不再针对任何数据包都通用**。从上到下，数据包的发送使用什么传输层协议，由socket初始化时确定。从下到上，收到的数据包由哪个传输层协议处理，根据从数据包传输层header中解析的数据确定。


1. vfs层
1. socket 是用于负责对上给用户提供接口，并且和文件系统关联。
2. sock，负责向下对接内核网络协议栈
3. tcp层 和 ip 层， linux 1.2.13相关方法都在 tcp_prot中。在高版本linux 中，sock 负责tcp 层， ip层另由struct inet_connection_sock 和 icsk_af_ops 负责。分层之后，诸如拥塞控制和滑动窗口的 字段和方法就只体现在struct sock和tcp_prot中，代码实现与tcp规范设计是一致的
4. ip层 负责路由等逻辑，并执行nf_hook，也就是netfilter，netfilter一个著名的实现，就是内核模块 ip_tables。在用户态，还有一个客户端程序 iptables，用命令行来干预内核的规则

    ![](/public/upload/network/linux_netfilter.png)

5. link 层，先寻找下一跳（ip ==> mac），有了 MAC 地址，就可以调用 dev_queue_xmit发送二层网络包了，它会调用 __dev_xmit_skb 会将请求放入块设备的队列。同时还会处理一些vlan 的逻辑
6. 设备层：网络包的发送会触发一个软中断 NET_TX_SOFTIRQ 来处理队列中的数据。这个软中断的处理函数是 net_tx_action。在软中断处理函数中，会将网络包从队列上拿下来，调用网络设备的传输函数 ixgb_xmit_frame，将网络包发的设备的队列上去。


## 网络协议栈实现——数据struct 和 协议struct

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

怎么理解整个表格呢？**协议struct**和**数据struct**有何异同？

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

![](/public/upload/network/tcp_object.png)

![](/public/upload/network/ip_link_object.png)

**重要的不是细节**，这个过程让我想到了web编程中的controller,service,dao。都是分层，区别是web请求要立即返回，网络通信则不用。

1. mac ==> device  ==> ip_rcv ==> tcp_rcv ==> 上层
2. url ==》 controller ==> service ==> dao ==> 数据库

想一想，整个网络协议栈，其实就是一群loopbackController、eth0Controller、ipService、TcpDao组成，该是一件多么有意思的事。

|类别|依赖关系的存储或表示|如何找依赖|依赖关系建立的时机是集中的|
|---|---|---|---|
|web|由spring管理，springmvc建立`<url,beanid>`,ioc建立`<beanId,bean>`|根据request信息及自身逻辑决定一步步如何往下走。|依赖关系建立的代码是集中的|
|linux|所谓的“依赖关系”是通过一个个struct及其数组（或链表）header，下层持有上层的struct header以完成接收，发送时则直接指定下层函数|接收时根据packet的一些字段，发送时根据socket参数及路由|依赖关系建立的代码是分散的，就好比有个全局的map，所有service(或者dao)自己向map注入自己的信息|

