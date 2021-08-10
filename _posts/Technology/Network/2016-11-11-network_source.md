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

## 源码目录

```
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
```	  		 

其中 unix 子文件夹中三个文件是有关 UNIX 域代码， UNIX 域是模拟网络传输方式在本机范围内用于进程间数据传输的一种机制。

系统调用通过 INT $0x80 进入内核执行函数，该函数根据 AX 寄存器中的系统调用号，进一步调用内核网络栈相应的实现函数。

## 网络分层

![](/public/upload/linux/linux_network.png)

从这个图中，可以看到，到传输层时，横生枝节，**代码不再针对任何数据包都通用**。从下到上，收到的数据包由哪个传输层协议处理，根据从数据包传输层header中解析的数据确定。从上到下，数据包的发送使用什么传输层协议，由socket初始化时确定。

1. vfs层
1. socket 是用于负责对上给用户提供接口，并且和文件系统关联。
2. sock，负责向下对接内核网络协议栈
3. tcp层 和 ip 层， linux 1.2.13相关方法都在 tcp_prot中。在高版本linux 中，sock 负责tcp 层， ip层另由struct inet_connection_sock 和 icsk_af_ops 负责。分层之后，诸如拥塞控制和滑动窗口的 字段和方法就只体现在struct sock和tcp_prot中，代码实现与tcp规范设计是一致的
4. ip层 负责路由等逻辑，并执行nf_hook，也就是netfilter。netfilter是工作于内核空间当中的一系列网络（TCP/IP）协议栈的钩子（hook），为内核模块在网络协议栈中的不同位置注册回调函数（callback）。也就是说，**在数据包经过网络协议栈的不同位置时做相应的由iptables配置好的处理逻辑**。
5. link 层，先寻找下一跳（ip ==> mac），有了 MAC 地址，就可以调用 dev_queue_xmit发送二层网络包了，它会调用 __dev_xmit_skb 会将请求放入块设备的队列。同时还会处理一些vlan 的逻辑
6. 设备层：网卡是发送和接收网络包的基本设备。在系统启动过程中，网卡通过内核中的网卡驱动程序注册到系统中。而在网络收发过程中，**内核通过中断跟网卡进行交互**。网络包的发送会触发一个软中断 NET_TX_SOFTIRQ 来处理队列中的数据。这个软中断的处理函数是 net_tx_action。在软中断处理函数中，会将网络包从队列上拿下来，调用网络设备的传输函数 ixgb_xmit_frame，将网络包发的设备的队列上去。

![](/public/upload/network/linux_recv_send.png)

网卡中断处理程序为网络帧分配的，内核数据结构 sk_buff 缓冲区；是一个维护网络帧结构的双向链表，链表中的每一个元素都是一个网络帧（Packet）。**虽然 TCP/IP 协议栈分了好几层，但上下不同层之间的传递，实际上只需要操作这个数据结构中的指针，而无需进行数据复制**。

## 数据结构

### 套接字

每个程序使用的套接字都有一个 struct socket 数据结构与 struct sock 数据结构的实例，socket 与sock 一一对应。
```c
struct socket { 
    socket_state            state;  // 套接字的状态
    unsigned long           flags;  // 套接字的设置标志。存放套接字等待缓冲区的状态信息，其值的形式如SOCK_ASYNC_NOSPACE等
    struct fasync_struct    *fasync_list;  // 等待被唤醒的套接字列表，该链表用于异步文件调用
    struct file             *file;  // 套接字所属的文件描述符
    struct sock             *sk;  // 指向存放套接字属性的结构指针
    wait_queue_head_t       wait;  //套接字的等待队列
    short                   type;  // 套接字的类型。其取值为SOCK_XXXX形式
    const struct proto_ops *ops;  // 套接字层的操作函数块
}
struct sock {
    ...
    struct sk_buff_head	write_queue,	receive_queue;
    ...	
}
```

套接字的连接建立起来后，用户进程就可以使用常规文件操作访问套接字了。每个套接字都分配了一个该类型的 inode，inode 和 socket 的链接是通过直接分配一个辅助数据结构来socket_slloc实现的
```c
struct socket_slloc {
  struct socket socket;
  struct inode vfs_inode;
}
```

VFS为文件系统抽象了一套API，实现了该系列API就可以把对应的资源当作文件使用，当调用socket函数的时候，我们拿到的不是socket本身，而是一个文件描述符fd。

![](/public/upload/linux/vfs_socket.jpeg)

[从linux5.9看网络层的设计](https://zhuanlan.zhihu.com/p/369460776)整个网络层的实际中，主要分为socket层、af_inet层和具体协议层（TCP、UDP等）。当使用网络编程的时候，首先会创建一个socket结构体（socket层），socket结构体是最上层的抽象，然后通过协议簇类型创建一个对应的sock结构体，sock是协议簇抽象（af_inet层），同一个协议簇下又分为不同的协议类型，比如TCP、UDP（具体协议层），然后根据socket的类型（流式、数据包）找到对应的操作函数集并赋值到socket和sock结构体中，**后续的操作就调用对应的函数就行**，调用某个网络函数的时候，会从socket层到af_inet层，af_inet做了一些封装，必要的时候调用底层协议（TCP、UDP）对应的函数。而不同的协议只需要实现自己的逻辑就能加入到网络协议中。

file_operations 结构定义了普通文件操作函数集。系统中每个文件对应一个 file 结构， file 结构中有一个 file_operations 变量，当使用 write，read 函数对某个文件描述符进行读写操作时，系统首先根据文件描述符索引到其对应的 file 结构，然后调用其成员变量 file_operations 中对应函数完成请求。

```c
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
```
	
以上 socket_file_ops 变量中声明的函数即是**网络协议对应的普通文件操作函数集合**。从而使得read， write， ioctl 等这些常见普通文件操作函数也可以被使用在网络接口的处理上。kernel维护一个struct file list，通过fd ==> struct file ==> file->ops ==> socket_file_ops,便可以以文件接口的方式进行网络操作。同时，每个 file 结构都需要有一个 inode 结构对应。用于存储struct file的元信息

```c
struct inode{
    ...
    union {
        ...
        struct ext_inode_info ext_i;
        struct nfs_inode_info nfs_i;
        struct socket socket_i;
    }u
}
```

也就是说，对linux系统，一切皆文件，由struct file描述，通过file->ops指向具体操作，由file->inode 存储一些元信息。对于ext文件系统，是载入内存的超级块、磁盘块等数据。对于网络通信，则是待发送和接收的数据块、网络设备等信息。从这个角度看，**struct socket和struct ext_inode_info 等是类似的。**

### sk_buff结构

当在内核中对数据包进行时，**内核还需要一些其他的数据来管理数据包和操作数据包**（就像加入jvm堆的数据必须有mark word一样），例如协议之间的交换信息，数据的状态，时间等。在发送数据时，在套接字层创建了 Socket Buffer 缓冲区与管理数据结构，存放来自应用程序的数据。在接收数据包时，Socket Buffer 则在网络设备的驱动程序中创建，存放来自网络的数据。在发送和接受数据的过程中，各层协议的头信息会不断从数据包中插入和去掉，sk_buff 结构中描述协议头信息的地址指针也会被不断地赋值和复位。

sk_buff部分字段如下，


```c
struct sk_buff {  
    /* These two members must be first. */  
    struct sk_buff      *next;  
    struct sk_buff      *prev;  
    
    struct sk_buff_head *list;  
    struct sock     *sk;  
    struct timeval      stamp;  
    struct net_device   *dev;  
    struct net_device   *real_dev;  
    union {  
        struct tcphdr   *th;  
        struct udphdr   *uh;  
        struct icmphdr  *icmph;  
        struct igmphdr  *igmph;  
        struct iphdr    *ipiph;  
        unsigned char   *raw;  
    } h;  // Transport layer header 
    union {  
        struct iphdr    *iph;  
        struct ipv6hdr  *ipv6h;  
        struct arphdr   *arph;  
        unsigned char   *raw;  
    } nh;  // Network layer header 
    
    union {  
        struct ethhdr   *ethernet;  
        unsigned char   *raw;  
    } mac;  // Link layer header 
    
    struct  dst_entry   *dst;  
    struct  sec_path    *sp;  
    
    void            (*destructor)(struct sk_buff *skb);  

    /* These elements must be at the end, see alloc_skb() for details.  */  
    unsigned int        truesize;  
    atomic_t        users;  
    unsigned char       *head,  
                *data,  
                *tail,  
                *end;  
}; 
```
	
head和end字段指向了buf的起始位置和终止位置。然后使用header指针指像各种协议填值。然后data就是实际数据。tail记录了数据的偏移值。

sk_buff 是各层通用的，在应用层数据包叫 data，在 TCP 层我们称为 segment，在 IP 层我们叫 packet，在数据链路层称为 frame。下层协议将上层协议数据作为data部分，并加上自己的header。这也是为什么代码注释中说，哪些字段必须在最前，哪些必须在最后， 这个其中的妙处可以自己体会。

sk_buff由sk_buff_head组织

```c
struct sk_buff_head {
    struct sk_buff		* volatile next;
    struct sk_buff		* volatile prev;
    #if CONFIG_SKB_CHECK
    int				magic_debug_cookie;
    #endif
};
```

TCP/IP 协议栈处理完输入数据包后，将数据包交给套接字层，放在套接字的接收缓冲区队列（sk_rcv_queue）。然后数据包从套接字层离开内核，送给应用层等待数据包的用户程序。用户程序向外发送的数据包缓存在套接字的传送缓冲区队列（sk_write_queue），从套接字层进入内核地址空间。



### 网络协议栈实现——数据struct 和 协议struct

socket分为多种，除了inet还有unix。反应在代码结构上，就是net包下只有net/unix,net/inet两个文件夹。之所以叫unix域，可能跟描述其地址时，使用`unix://xxx`有关

The difference is that an INET socket is bound to an IP address-port tuple, while a UNIX socket is "bound" to a special file on your filesystem. Generally, only processes running on the same machine can communicate through the latter.

本文重点是inet,net/inet下有以下几个比较重要的文件，这跟网络书上的知识就对上了。

```
arp.c
eth.c
ip.c
route.c
tcp.c
udp.c
datalink.h		// 应该是数据链路层
```

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
4. socket、sock、device等数据struct经常被作为分析的重点，**其实各种协议struct 才是流程的关键，并且契合了网络协议分层的理念**。 

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
2. url ==> controller ==> service ==> dao ==> 数据库

想一想，整个网络协议栈，其实就是一群loopbackController、eth0Controller、ipService、TcpDao组成，该是一件多么有意思的事。

|类别|依赖关系的存储或表示|如何找依赖|依赖关系建立的时机是集中的|
|---|---|---|---|
|web|由spring管理，springmvc建立`<url,beanid>`,ioc建立`<beanId,bean>`|根据request信息及自身逻辑决定一步步如何往下走。|依赖关系建立的代码是集中的|
|linux|所谓的“依赖关系”是通过一个个struct及其数组（或链表）header，下层持有上层的struct header以完成接收，发送时则直接指定下层函数|接收时根据packet的一些字段，发送时根据socket参数及路由|依赖关系建立的代码是分散的，就好比有个全局的map，所有service(或者dao)自己向map注入自己的信息|

## 其它

无论 TCP 还是 UDP，端口号都只占 16 位，也就说其最大值也只有 65535。那是不是说，如果使用 TCP 协议，在单台机器、单个 IP 地址时，并发连接数最大也只有 65535 呢？对于这个问题，首先你要知道，Linux 协议栈，通过五元组来标志一个连接（即协议，源 IP、源端口、目的 IP、目的端口)。对客户端来说，每次发起 TCP 连接请求时，都需要分配一个空闲的本地端口，去连接远端的服务器。由于这个本地端口是独占的，所以客户端最多只能发起 65535 个连接。对服务器端来说，其通常监听在固定端口上（比如 80 端口），等待客户端的连接。根据五元组结构，我们知道，客户端的 IP 和端口都是可变的。如果不考虑 IP 地址分类以及资源限制，服务器端的理论最大连接数，可以达到 2 的 48 次方（IP 为 32 位，端口号为 16 位），远大于 65535。服务器端可支持的连接数是海量的，当然，由于 Linux 协议栈本身的性能，以及各种物理和软件的资源限制等，这么大的连接数，还是远远达不到的（实际上，C10M 就已经很难了）。

软中断有专门的内核线程 ksoftirqd处理。每个 CPU 都会绑定一个 ksoftirqd 内核线程，比如， 2 个 CPU 时，就会有 ksoftirqd/0 和 ksoftirqd/1 这两个内核线程。