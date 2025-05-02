---

layout: post
title: Linux网络源代码学习——数据包的发送与接收
category: 技术
tags: Network
keywords: network 

---

## 简介

* TOC
{:toc}

linux网络编程中，各层有各层的struct，但有一个struct是各层通用的，这就是描述接收和发送数据的struct sk_buff.

## 网络分层与内核态

2019.7.5补充：应用层和内核互通的机制是通过 Socket 系统调用，经常有人会问，**Socket 属于哪一层？其实它哪一层都不属于**，它属于操作系统的概念，而非网络协议分层的概念。只不过操作系统选择对于网络协议的实现模式是，**二到四层的处理代码在内核里面**，七层的处理代码让应用自己去做，两者需要跨内核态和用户态通信，就需要一个系统调用完成这个衔接，这就是 Socket。

从 TCP/IP 协议栈的角度来看，传输层以上的都是应用程序的一部分，Linux 与传统的 UNIX 类似，TCP/IP 协议栈驻留在内核中，与内核的其他组件共享内存。传输层以上执行的网络功能，都是在用户地址空间完成的。

![](/public/upload/network/tcp_and_kernel.png)

TCP 层会根据 TCP 头中的序列号等信息，发现它是一个正确的网络包，就会将网络包缓存起来，等待应用层的读取。应用层通过 Socket 监听某个端口，因而读取的时候，内核会根据 TCP 头中的端口号，将网络包发给相应的Socket。

## 宏观

系统调用 与TCP 标志位的状态转换关系

![](/public/upload/network/tcp_state_transition.png)

TCP包含SYN、ACK、FIN、PSH、RST 等标志位，有些是**应用程序触发**（比如connect和close），有些是**操作系统触发**（比如当客户端尝试连接到一个未监听的端口时，服务器会发送 RST 包。再如果 TCP 收到一个不属于任何已建立连接的数据包（例如，序列号不匹配或连接已被关闭），它会发送 RST 包。当系统资源不足（例如 TCP 内存不足或孤儿连接过多）时，系统可能会主动发送 RST 包来终止连接），**有些是中间设备触发**（比如连接长时间无数据交互，中间设备（如负载均衡器）可能会认为连接已超时并发送 RST 包）。

![](/public/upload/network/tcp_send_recv.jpg)

![](/public/upload/network/linux_tcp_function.png)

创建socket

![](/public/upload/linux/socket_create.png)

sock_create 函数完成通用套接字创建、初始化任务后，再调用特定协议族(上图`net_families[family]`)的套接字创建函数，对于 TCP/IP 协议栈，协议族将设置为 AF_INET。

## 建立连接

[那些你不知道的TCP冷门知识！](https://mp.weixin.qq.com/s/6lop61UtnQ-vfWJy17V87w)在Linux中，一般情况下都是**内核代理三次握手**的
1. 当client端调用 `connect()` 之后内核负责发送SYN，接收SYN-ACK，发送ACK。然后 `connect()` 系统调用才会返回，客户端侧握手成功。
2. 服务端的Linux内核会在收到SYN之后负责回复SYN-ACK再等待ACK之后才会让 `accept()` 返回，从而完成服务端侧握手。于是Linux内核就需要引入半连接队列（用于存放收到SYN，但还没收到ACK的连接）和全连接队列（用于存放已经完成3次握手，但是应用层代码还没有完成 accept() 的连接）两个概念，用于存放在握手中的连接。

系统调用跟 TCP 状态关系的示意图如下

![](/public/upload/network/tcp_handshake.png)

sync 和accept 队列的长度，sync 的重试次数都可以设置。如果应用程序处理较慢，会导致accept 队列。sync 攻击（伪造很多客户端ip 发送sync请求，但不发送ack）会导致sync 队列满。

## 数据接收过程

《深入理解Linux网络》

![](/public/upload/network/network_receive.png)

[容器网络一直在颤抖，罪魁祸首竟然是 ipvs 定时器](https://mp.weixin.qq.com/s/pY4ZKkzgfTmoxsAjr5ckbQ)在内核中，网络设备驱动是通过中断的方式来接受和处理数据包。当网卡设备上有数据到达的时候，会触发一个硬件中断来通知 CPU 来处理数据，此类处理中断的程序一般称作 ISR (Interrupt Service Routines)。ISR 程序不宜处理过多逻辑，否则会让设备的中断处理无法及时响应。因此 Linux 中将中断处理函数分为上半部和下半部。上半部是只进行最简单的工作，快速处理然后释放 CPU。剩下将绝大部分的工作都放到下半部中，下半部中逻辑由内核线程选择合适时机进行处理。
Linux 2.4 以后内核版本采用的下半部实现方式是软中断，由 ksoftirqd 内核线程全权处理， 正常情况下每个 CPU 核上都有自己的软中断处理数队列和 ksoftirqd 内核线程。

![](/public/upload/network/linux_network_package_receive.png)

网络相关的中断程序在网络子系统初始化的时候进行注册， NET_RX_SOFTIRQ 的对应函数为 `net_rx_action()` ，在 `net_rx_action()` 函数中会调用网卡设备设置的 poll 函数，批量收取网络数据包并调用上层注册的协议函数进行处理，如果是为 ip 协议，则会调用 ip_rcv，上层协议为 icmp 的话，继续调用 icmp_rcv 函数进行后续的处理。PS：参考后续的ksoftirqd

```c
struct sock {
    ...
    struct sk_buff_head	write_queue,	receive_queue;
    ...	
}
```
	
硬件监听物理介质，进行数据的接收，当接收的数据填满了缓冲区，硬件就会产生中断，中断产生后，系统会转向中断服务子程序。在中断服务子程序中，数据会从硬件的缓冲区复制到内核的空间缓冲区，并包装成一个数据结构（sk_buff），然后调用对驱动层的接口函数netif_rx()将数据包发送给链路层。从链路层向网络层传递时将调用ip_rcv函数。该函数完成本层的处理后会根据IP首部中使用的传输层协议来调用相应协议的处理函数（比如UDP对应udp_rcv、TCP对应tcp_rcv）。

如果在IP数据报的首部标明的是使用TCP传输数据，则在上述函数中会调用tcp_rcv函数。该函数的大体处理流程为：

所有使用TCP 协议的套接字对应sock 结构都被挂入tcp_prot（proto 结构）之sock_array 数组中，采用以本地端口号为索引的插入方式，所以当tcp_rcv 函数接收到一个数据包，在完成必要的检查和处理后，其将以TCP 协议首部中目的端口号（对于一个接收的数据包而言，其目的端口号就是本地所使用的端口号）为索引，在tcp_prot 对应sock 结构之sock_array 数组中得到正确的sock 结构队列，再辅之以其他条件遍历该队列进行对应sock 结构的查询，在得到匹配的sock 结构后，将数据包挂入该sock 结构中的缓存队列中（由sock 结构中receive_queue 字段指向），从而完成数据包的最终接收。

```c
struct prot{
    ...
    struct sock *	sock_array[SOCK_ARRAY_SIZE]; // 注意是一个数组指针，一个主机有多个端口，一个端口可以建立多个连接，因而有多个struct sock
}
```

总结来看，数据接收的流程是  sk_buff ==> prot 某个端口的sock_array中的某个sock 的sk_buff 队列。

上文中还有一个比较重要的点是，根据接收的数据，如何找到其对应的struct sock。从中也应该可以了解到，端口的本质含义。

当用户需要接收数据时，首先根据文件描述符inode得到socket结构和sock结构（**socket结构在用户层，sock在内核层，两者涉及到数据在内核态和用户态的拷贝**），然后从sock结构中指向的队列recieve_queue中读取数据包，将数据包COPY到用户空间缓冲区。数据就完整的从硬件中传输到用户空间。这样也完成了一次完整的从下到上的传输。

![](/public/upload/network/linux_package_receive.png)

从 tcp 协议栈的处理入口函数 tcp_v4_rcv 开始说：在 tcp_v4_rcv 中首先根据收到的网络包的 header 里的 source 和 dest 信息来在本机上查询对应的 socket。 tcp_v4_do_rcv ==> tcp_rcv_established ==> tcp_queue_rcv 将接收数据放到 socket 的接收队列上。接着再调用 sk_data_ready 来唤醒在 socket上等待的用户进程。sk_data_ready 是一个函数指针，执行sock 初始化时设定的函数。

![](/public/upload/linux/tcp_rcv.png)

[深入理解高性能网络开发路上的绊脚石 - 同步阻塞网络 IO](https://mp.weixin.qq.com/s/cIcw0S-Q8pBl1-WYN0UwnA)汇总一下

![](/public/upload/linux/network_recv.png)

## 数据的发送

send 系统调用：在内核中找到 socket（记录着各种协议栈的函数地址），函数调用由系统调用进入协议栈，inet_sendmsg 是AF_INET 协议族提供的通用发送函数，经过协议栈处理后进入RingBuffer（存储网络数据包的元数据，接收时还未经过协议栈的处理，不能直接用），随后网卡驱动真正将数据发送出去，当发送完成的时候，通过硬中断来通知CPU，然后清理RingBuffer。

![](/public/upload/linux/tcp_send.png)

```
inet_sendmsg ==>
tcp_sendmsg # 申请一个 内核态的skb 内存并挂到 socket 发送队列sk_write_queue，将用户待发送的数据拷贝到skb，同时进行一些判断，如果条件不满足（比如未发送的数据是否已经超过最大窗口的一半了），这次用户要发送的数据只是拷贝到内核就完事了。假设发射条件已经满足了
  ==> tcp_write_xmit # 滑动窗口、拥塞控制
    ==> tcp_transmit_skb # 克隆一个新的skb，实际传输出去的是skb_copy，网卡发送完成的时候，skb_copy 会释放掉。tcp 支持丢失重传，skb 要等到收到ack 再真正删除
      ==> ip_queue_xmit # 路由项查找、IP头设置、netfilter过滤、skb切分（大于MTU的话）
        ==> 邻居子系统 # 有可能发出arp请求，然后封装mac头，将skb再传递到更下层的设备子系统
          ==> dev_queue_xmit # 网卡有多个发送队列（一个队列由一个RingBuffer表示），选择发送队列，while循环不断的从队列中取出skb并进行发送
            ==> dev_hard_start_xmit # 将skb 挂到RingBuffer上，将skb所有数据都映射到DMA地址，触发真实的发送。发送完毕后，网卡触发硬中断NET_RX_SOFTIRQ 来释放 清理skb，解除DMA 映射。 
```

![](/public/upload/network/tcp_buffer.png)

## 真正干活的ksoftirqd

软中断有专门的内核线程 ksoftirqd处理/内核线程ksoftirqd包含了所有的软中断处理逻辑。每个 CPU 都会绑定一个 ksoftirqd 内核线程，比如， 2 个 CPU 时，就会有 ksoftirqd/0 和 ksoftirqd/1 这两个内核线程。

[聊聊 veth 数据流](https://mp.weixin.qq.com/s/3aoQCJywV00berRwbH0ocQ)数据包（data package）穿过TCP/IP不同层时叫法不同。在应用层叫做message，到了TCP层叫做segment、UDP层叫datagram，流到了IP层叫做datagram，而在链路层则称为frame，到了物理层就变成bitstream（比特流）

![](/public/upload/linux/ksoftirqd.png)

1. RingBuffer是内存中一块特殊区域，网卡在收到数据的时候以DMA的方式将包写到RingBuffer中。
1. ksoftirqd首先是一个死循环。如果有网络设备挂在poll_list上面，只要满足条件，它就会从poll_list上面将其取下来，执行该设备驱动程序所注册的poll()。poll()不断地从net_device的RingBuffer里面取出数据包，转成skb格式，并沿着网络设备子系统 -> IP协议层 -> TCP层一路调用内核里面的函数来分析和处理这个skb。从上图可以看到**skb从RingBuffer被取出来，到最后落到位于TCP层的socket接收队列里，都是在ksoftirqd这个内核线程里完成的**。这个处理过程还包括iptables的处理，路由的查询等各种费时费力的工作。所以如果iptables设置得非常多的话，会导致ksoftirqd处理每一个skb的时间变长，进而导致消费RingBuffer的速度变慢，对外的表现就是机器的吞吐量降低。
2. 在网络包的发送过程中，**用户进程（在内核态）完成了绝大部分工作**，甚至连调用驱动的工作都干了。如果发送网络包的时候 进程内核态 cpu quota用尽 或者其它进程需要cpu的时候，触发软中断 NET_TX_SOFTIRQ ，由ksoftirqd 执行net_tx_action 函数，找到发送队列，最终调用驱动程序的入口函数 dev_hard_start_xmit 

![](/public/upload/network/linux_package_send.png)

在linux内核中，对于接收过程来讲，网卡硬件将数据通过DMA的方式放入到内存中，发起硬中断来通知CPU处理，耗时会体现在top命令的hi时间上。不过硬中断的优先级很高，为了避免对系统任务产生影响，硬中断要执行的工作很简单，主要工作都丢给软中断来处理了，所以一般我们看到hi 不会太高。内核协议栈的主要工作都是在软中断中完成的。在发送过程中，优先用用户进程的内核态时间来处理发送（体现在top命令sy这个指标），但当用户进程处理发送的时间用光了的时候，会启用软中断ksoftirqd 这个线程来处理，体现在top命令si 上。对于hi、si的开销来说，不仅要关注总开销，还要看它分布在哪几个核上。因为不是所有的核都会参与硬中断、软中断的处理。 对于老式的单队列网卡、或者在没有开启多队列的虚拟机中，可能只有一个队列，软、硬中断的处理都集中在一个核上。现在主流网卡都支持多队列，linux上有一个irqbalance服务，它可以根据当前系统的负载情况自动优化中断分配，把各个中断号（每个可中断到cpu的设备都有一个中断号，一个网卡的不同队列对应不同的中断号）分到不同的cpu核上处理。

## 不同的缓存方式 and 处理网络包的三个主体

io数据的读写，不会一个字节一个字节的来，所以数据会缓存（缓冲）。

||管理程序|备注|
|---|---|---|
|硬件/驱动程序缓存区|由驱动程序管理|粗略的说，所谓发送，就是将数据拷贝到相关的硬件寄存器，并触发驱动程序发送。|
|内核空间缓存区|由内核管理||
|用户空间缓存区|由用户程序管理||

针对内核空间缓存区

1. linux文件操作中，会专门开辟一段内存，用来存储磁盘文件系统的超级块和部分磁盘块，通过文件path ==> fd ==> inode ==> ... ==> 块索引 找到块的数据并读写。磁盘块的管理是公共的、独立的，fd记录自己相关的磁盘块索引即可。fd不用这个磁盘块了，除非系统剩余内存不够，这个磁盘块数据不会被释放，下次用到时还能用。即便fd只要一部分数据，系统也会自动加载相关的多个磁盘块到内存。
2. linux网络操作中，通过socket fd ==> sock ==> write_queue, receive_queue 在缓存读写的数据。sk_buff 的管理是sock各自为政。

这其中的不同，值得品味。一个重要原因是，因为linux事先加载了超级块数据，可以根据需要，精确的指定加载和写入多少数据。而对于网络编程来说，一次能读取和写出多少，都是未知的，每个sock都不同。

（较高版本linux）网络包的接收过程，这里面涉及三个队列：

1. backlog 队列
2. prequeue 队列
3. sk_receive_queue 队列

为什么接收网络包的过程，需要在这三个队列里面倒腾过来、倒腾过去呢？这是因为，同样一个网络包要在三个主体之间交接。

1. 软中断的处理过程。我们在执行tcp_v4_rcv 函数的时候，依然处于软中断的处理逻辑里，所以必然会占用这个软中断。
2. 用户态进程。如果用户态触发系统调用 read 读取网络包，也要从队列里面找。
3. 内核协议栈。哪怕用户进程没有调用 read读取网络包，当网络包来的时候，也得有一个地方收着。


## 发送与接收成本 和 优化

[浅谈Service Mesh体系中的Envoy](https://yq.aliyun.com/articles/606655)

linux 一次发送或接收处理链路还是比较长的，且需要在内核态与用户态之间做内存拷贝、上下文切换、软硬件中断等。以对我们最直观的内存拷贝为例，正常情况下，一个网络数据包从网卡到应用程序需要经过如下的过程：数据从网卡通过 DMA 等方式传到内核开辟的缓冲区，然后从内核空间拷贝到用户态空间，在 Linux 内核协议栈中，这个耗时操作甚至占到了数据包整个处理流程的 57.1%

虽然Linux设计初衷是以通用性为目的的，但随着Linux在服务器市场的广泛应用，其原有的网络数据包处理方式已很难跟上人们对高性能网络数据处理能力的诉求。在这种背景下DPDK应运而生，其利用UIO技术，在Driver层直接将数据包导入到用户态进程，绕过了Linux协议栈，接下来由用户进程完成所有后续处理，再通过Driver将数据发送出去。原有内核态与用户态之间的内存拷贝采用mmap将用户内存映射到内核，如此就规避了内存拷贝、上下文切换、系统调用等问题，然后再利用大页内存、CPU亲和性、无锁队列、基于轮询的驱动模式、多核调度充分压榨机器性能，从而实现高效率的数据包处理。

DPDK全称Intel Data Plane Development Kit，是Intel提供的数据平面开发工具集，为Intel Architecture（IA）处理器架构下用户空间高效的数据包处理提供库函数和驱动的支持。VPP是the vector packet processor的简称，是一套基于DPDK的网络帧处理解决方案，是一个可扩展框架，提供开箱即用的交换机/路由器功能。是Linux基金会下开源项目FD.io的一个子项目，由思科贡献的开源版本，目前是FD.io的最核心的项目。

![](/public/upload/network/dpdk_package_send.png)

![](/public/upload/network/dpdk_package_receive.png)

通过对比得知，DPDK拦截中断，不触发后续中断流程，并绕过内核协议栈，通过UIO（Userspace I/O）技术将网卡收到的报文拷贝到应用层处理，报文不再经过内核协议栈。减少了中断，DPDK的包全部在用户空间使用内存池管理，内核空间与用户空间的内存交互不用进行拷贝，只做控制权转移，减少报文拷贝过程，提高报文的转发效率。

DPDK能够绕过内核协议栈，本质上是得益于 UIO 技术，UIO技术也不是DPDK创立的，是内核提供的一种运行在用户空间的I/O技术，Linux系统中一般的驱动设备都是运行在内核空间，在用户空间用的程序调用即可，UIO则是将驱动的很少一部分运行在内核空间，绝大多数功能在用户空间实现，通过 UIO 能够拦截中断，并重设中断回调行为，从而绕过内核协议栈后续的处理流程。

![](/public/upload/network/tcp.png)

## 其它

无论 TCP 还是 UDP，端口号都只占 16 位，也就说其最大值也只有 65535。那是不是说，如果使用 TCP 协议，在单台机器、单个 IP 地址时，并发连接数最大也只有 65535 呢？对于这个问题，首先你要知道，Linux 协议栈，通过五元组来标志一个连接（即协议，源 IP、源端口、目的 IP、目的端口)。
1. 对客户端来说，每次发起 TCP 连接请求时，都需要分配一个空闲的本地端口，去连接远端的服务器。由于这个本地端口是独占的，所以客户端最多只能发起 65535 个连接。
2. 对服务器端来说，其通常监听在固定端口上（比如 80 端口），等待客户端的连接。根据五元组结构，我们知道，客户端的 IP 和端口都是可变的。如果不考虑 IP 地址分类以及资源限制，服务器端的理论最大连接数，可以达到 2 的 48 次方（IP 为 32 位，端口号为 16 位），远大于 65535。服务器端可支持的连接数是海量的，当然，由于 Linux 协议栈本身的性能，以及各种物理和软件的资源限制等，这么大的连接数，还是远远达不到的（实际上，C10M 就已经很难了）。

## 引用

[linux内核学习之网络篇——套接字缓冲区](http://blog.csdn.net/wallwind/article/details/8030306)

[Linux内核--网络栈实现分析（二）--数据包的传递过程--转](http://www.cnblogs.com/davidwang456/p/3604089.html)