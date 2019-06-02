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

本文介绍下Docker网络的相关知识

建议看下前文 [程序猿视角看网络](http://qiankunli.github.io/2018/03/08/network.html)

相对于物理网络，虚拟化有两个方面：

1. 网络本身的虚拟，比如vlan等
2. container 等带来的主机内网络虚拟 和跨主机网络虚拟

## 传统以太网络



## 交换机IP-MAC-PORT


在交换机上配置了IMP(ip-mac-port映射)功能以后，交换机会检查每个数据包的源IP地址和MAC，对于没有在交换机内记录的IP和MAC地址的计算机所发出的数据包都会被交换机所阻止。ip-mac-port映射静态设置比较麻烦，可以开启交换机上面的DHCP SNOOPING功能， DHCP Snooping可以自动的学习IP和MAC以及端口的配对，并将学习到的对应关系保存到交换机的本地数据库中。

默认情况下，交换机上每个端口只允许绑定一个IP-MAC条目，所以在使用docker macvlan时要打开这样的限制。

## vlan 划分

1. 常用的 VLAN 划分方式是通过端口进行划分，虽然这种划分 VLAN 的方式设置比较很简单， 但仅适用于终端设备物理位置比较固定的组网环境。随着移动办公的普及，终端设备可能不 再通过固定端口接入交换机，这就会增加网络管理的工作量。比如，一个用户可能本次接入 交换机的端口 1，而下一次接入交换机的端口 2，由于端口 1 和端口 2 属于不同的 VLAN，若 用户想要接入原来的 VLAN 中，网管就必须重新对交换机进行配置。显然，这种划分方式不 适合那些需要频繁改变拓扑结构的网络。
2. 而 MAC VLAN 则可以有效解决这个问题，它根据 终端设备的 MAC 地址来划分 VLAN。这样，即使用户改变了接入端口，也仍然处在原 VLAN 中。**注意，这种称为mac based vlan，跟macvlan还不是一个意思**

如果采用硬件支持的方式来设置vlan，交换机是划分局域网的关键设备，所以本文说xx vlan，主要是针对交换机说的。

## 网桥

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
        
使用集线器连接局域网中的pc时，一个重要缺点是：任何一个pc发数据，其它pc都会收到，无用不说，还导致物理介质争用。网桥与交换机类似（其不同不足以影响对docker网络的理解），会学习mac地址与端口（串口）的映射。使用交换机替换集线器后，pc1发给pc2的数据只有pc2才会接收到。

[Bridge vs Macvlan](https://hicu.be/bridge-vs-macvlan)

Switching was just a fancy name for bridging, and that was a 1980s technology – or so the thinking went.A bridge can be a physical device or implemented entirely in software. Linux kernel is able to perform bridging since 1999. Switches have meanwhile became specialized physical devices and software bridging had almost lost its place. However, with the advent of virtualization, virtual machines running on physical hosts required Layer 2 connection to the physical network and other VMs. Linux bridging provided a well proven technology and entered it’s Renaissance（文艺复兴）. 最开始bridge是一个硬件， 也叫swtich，后来软件也可以实现bridge了，swtich就专门称呼硬件交换机了，再后来虚拟化时代到来，bridge 迎来了第二春。


[Macvlan and IPvlan basics](https://sreeninet.wordpress.com/2016/05/29/macvlan-and-ipvlan/)In linux bridge implementation, VMs or Containers will connect to bridge and bridge will connect to outside world. For external connectivity, we would need to use NAT. container 光靠 bridge 无法直接访问外网。

建议看下 [docker中涉及到的一些linux知识](http://qiankunli.github.io/2016/12/02/linux_docker.html) 对网桥源码的分析。

## 虚拟设备 ==> 虚拟网络

Linux 用户想要使用网络功能，不能通过直接操作硬件完成，而需要直接或间接的操作一个Linux 为我们抽象出来的设备，即通用的 Linux 网络设备来完成。“eth0”并不是网卡，而是Linux为我们抽象（或模拟）出来的“网卡”。除了网卡，现实世界中存在的网络元素Linux都可以模拟出来，包括但不限于：电脑终端、二层交换机、路由器、网关、支持 802.1Q VLAN 的交换机、三层交换机、物理网卡、支持 Hairpin 模式的交换机。同时，既然linux可以模拟网络设备，自然提供了操作这些虚拟的网络设备的命令或interface。

既然**Linux可以模拟网络设备**，那么现实世界中的网络拓扑结构，Linux自然也可以在一台（或多台）主机中模拟出来。用虚拟网络来模拟现实网络，这是虚拟化技术的重要一环。







    
## 参考文献

[Docker 网络配置][]

[Linux 上虚拟网络与真实网络的映射][]

[Docker 网络配置]: http://www.oschina.net/translate/docker-network-configuration
[Linux 上的基础网络设备详解]: https://www.ibm.com/developerworks/cn/linux/1310_xiawc_networkdevice/
[Linux 上虚拟网络与真实网络的映射]: https://www.ibm.com/developerworks/cn/linux/1312_xiawc_linuxvirtnet/