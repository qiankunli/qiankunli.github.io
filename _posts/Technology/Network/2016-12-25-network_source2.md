---

layout: post
title: Linux网络源代码学习2
category: 技术
tags: Network
keywords: network 

---

## 简介

linux网络编程中，各层有各层的struct，但有一个struct是各层通用的，这就是描述接收和发送数据的struct sk_buff.

## sk_buff结构

sk_buff部分字段如下，


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
	
head和end字段指向了buf的起始位置和终止位置。然后使用header指针指像各种协议填值。然后data就是实际数据。tail记录了数据的偏移值。

sk_buff 是各层通用的，下层协议将上层协议数据作为data部分，并加上自己的header。这也是为什么代码注释中说，哪些字段必须在最前，哪些必须在最后， 这个其中的妙处可以自己体会。

sk_buff由sk_buff_head组织

	struct sk_buff_head {
	  	struct sk_buff		* volatile next;
	  	struct sk_buff		* volatile prev;
		#if CONFIG_SKB_CHECK
	  	int				magic_debug_cookie;
		#endif
	};

## 数据接收过程



	struct sock {
		...
		struct sk_buff_head	write_queue,	receive_queue;
		...	
	}
	
硬件监听物理介质，进行数据的接收，当接收的数据填满了缓冲区，硬件就会产生中断，中断产生后，系统会转向中断服务子程序。在中断服务子程序中，数据会从硬件的缓冲区复制到内核的空间缓冲区，并包装成一个数据结构（sk_buff），然后调用对驱动层的接口函数netif_rx()将数据包发送给链路层。从链路层向网络层传递时将调用ip_rcv函数。该函数完成本层的处理后会根据IP首部中使用的传输层协议来调用相应协议的处理函数（比如UDP对应udp_rcv、TCP对应tcp_rcv）。

如果在IP数据报的首部标明的是使用TCP传输数据，则在上述函数中会调用tcp_rcv函数。该函数的大体处理流程为：

所有使用TCP 协议的套接字对应sock 结构都被挂入tcp_prot（proto 结构）之sock_array 数组中，采用以本地端口号为索引的插入方式，所以当tcp_rcv 函数接收到一个数据包，在完成必要的检查和处理后，其将以TCP 协议首部中目的端口号（对于一个接收的数据包而言，其目的端口号就是本地所使用的端口号）为索引，在tcp_prot 对应sock 结构之sock_array 数组中得到正确的sock 结构队列，再辅之以其他条件遍历该队列进行对应sock 结构的查询，在得到匹配的sock 结构后，将数据包挂入该sock 结构中的缓存队列中（由sock 结构中receive_queue 字段指向），从而完成数据包的最终接收。

	struct prot{
		...
		struct sock *	sock_array[SOCK_ARRAY_SIZE]; // 注意是一个数组指针，一个主机有多个端口，一个端口可以建立多个连接，因而有多个struct sock
	}

总结来看，数据接收的流程是  sk_buff ==> prot 某个端口的sock_array中的某个sock 的sk_buff 队列。

上文中还有一个比较重要的点是，根据接收的数据，如何找到其对应的struct sock。从中也应该可以了解到，端口的本质含义。

当用户需要接收数据时，首先根据文件描述符inode得到socket结构和sock结构（**socket结构在用户层，sock在内核层，两者涉及到数据在内核态和用户态的拷贝**），然后从sock结构中指向的队列recieve_queue中读取数据包，将数据包COPY到用户空间缓冲区。数据就完整的从硬件中传输到用户空间。这样也完成了一次完整的从下到上的传输。

## 数据的发送

用户在初始化socket之后，会得到一个fd，socket.write ==> sock.write ==> inet.write ==> tcp.write ==> ip_queue_xmit ==> dev_queue_xmit ==> ei_start_xmit.

传输层将用户的数据包装成sk_buff 下放到ip层，在ip层，函数ip_queue_xmit()的功能是将数据包进行一系列复杂的操作，比如是检查数据包是否需要分片。同时，根据目的ip、iptables等规则，选取一个dev发送数据。

## 不同的缓存方式

io数据的读写，不会一个字节一个字节的来，所以数据会缓存（缓冲）。

||管理程序|备注|
|---|---|---|
|硬件/驱动程序缓存区|由驱动程序管理|粗略的说，所谓发送，就是将数据拷贝到相关的硬件寄存器，并触发驱动程序发送。|
|内核空间缓存区|由内核管理||
|用户空间缓存区|由用户程序管理||

针对内核空间缓存区

linux文件操作中，会专门开辟一段内存，用来存储磁盘文件系统的超级块和部分磁盘块，通过文件path ==> fd ==> inode ==> ... ==> 块索引 找到块的数据并读写。磁盘块的管理是公共的、独立的，fd记录自己相关的磁盘块索引即可。fd不用这个磁盘块了，除非空间不够，这个磁盘块数据不会被回收，下次用到时还能用。即便fd只要一部分数据，系统也会自动加载相关的多个磁盘块进来。

linux网络操作中，通过socket fd ==> sock ==> write_queue, receive_queue 在缓存读写的数据。sk_buff 的管理是sock各自为政。

这其中的不同，值得品味。一个重要原因是，因为linux事先加载了超级块数据，可以根据需要，精确的指定加载和写入多少数据。而对于网络编程来说，一次能读取和写出多少，都是未知的，每个sock都不同。


## 引用

[linux内核学习之网络篇——套接字缓冲区](http://blog.csdn.net/wallwind/article/details/8030306)

[Linux内核--网络栈实现分析（二）--数据包的传递过程--转](http://www.cnblogs.com/davidwang456/p/3604089.html)