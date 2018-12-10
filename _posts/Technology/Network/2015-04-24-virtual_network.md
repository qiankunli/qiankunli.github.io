---

layout: post
title: 虚拟网络
category: 技术
tags: Network
keywords: Docker

---

## 前言

本文介绍下Docker网络的相关知识

建议看下前文 [程序猿视角看网络](http://qiankunli.github.io/2018/03/08/network.html)

* TOC
{:toc}


相对于物理网络，虚拟化有两个方面：

1. 网络本身的虚拟，比如vlan等
2. container 等带来的主机内网络虚拟 和跨主机网络虚拟

## 传统以太网络

现实世界中一个常见的网络环境，小公司的局域网经常这么干（蓝色图标表示交换机），以下称图1。

![Alt text](/public/upload/docker/traditional_lan_architecture.jpg)

## 交换机IP-MAC-PORT

网络中实际传输的是“帧”，帧里面是有目标主机的MAC地址的。在以太网中，一个主机要和另一个主机进行直接通信，必须要知道目标主机的MAC地址。但这个目标MAC地址是如何获得的呢？它就是通过地址解析协议获得的。所谓“地址解析”就是主机在发送帧前将目标IP地址转换成目标MAC地址的过程。ARP协议的基本功能就是通过目标设备的IP地址，查询目标设备的MAC地址，以保证通信的顺利进行。

ARP欺骗，当计算机接收到ARP应答数据包的时候，就会对本地的ARP缓存进行更新，将应答中的IP和MAC地址存储在ARP缓存中。但是，**ARP协议并不只在发送ARP请求才接收ARP应答。**ARP应答可以不请自来，有人发送一个自己伪造的ARP应答(比如错误的ip-mac映射)，网络可能就会出现问题，这是协议的设计者当初没考虑到的。

在交换机上配置了IMP(ip-mac-port映射)功能以后，交换机会检查每个数据包的源IP地址和MAC，对于没有在交换机内记录的IP和MAC地址的计算机所发出的数据包都会被交换机所阻止。ip-mac-port映射静态设置比较麻烦，可以开启交换机上面的DHCP SNOOPING功能， DHCP Snooping可以自动的学习IP和MAC以及端口的配对，并将学习到的对应关系保存到交换机的本地数据库中。

默认情况下，交换机上每个端口只允许绑定一个IP-MAC条目，所以在使用docker macvlan时要打开这样的限制。

## vlan 划分

1. 常用的 VLAN 划分方式是通过端口进行划分，虽然这种划分 VLAN 的方式设置比较很简单， 但仅适用于终端设备物理位置比较固定的组网环境。随着移动办公的普及，终端设备可能不 再通过固定端口接入交换机，这就会增加网络管理的工作量。比如，一个用户可能本次接入 交换机的端口 1，而下一次接入交换机的端口 2，由于端口 1 和端口 2 属于不同的 VLAN，若 用户想要接入原来的 VLAN 中，网管就必须重新对交换机进行配置。显然，这种划分方式不 适合那些需要频繁改变拓扑结构的网络。
2. 而 MAC VLAN 则可以有效解决这个问题，它根据 终端设备的 MAC 地址来划分 VLAN。这样，即使用户改变了接入端口，也仍然处在原 VLAN 中。**注意，这种称为mac based vlan，跟macvlan还不是一个意思**


## 802.1Q VLAN 以太网

使用802.1QVLAN 技术，可以把逻辑上的子网和物理上的子网分割开来，即物理上连接在同一交换机上的终端可以属于不同逻辑子网（这个子网，跟同属于一个网段“子网”（比如`192.168.3.0/24`）不是一回事），处于不同逻辑子网的终端相互隔离，从而解决广播域混乱问题。以下称图3.

![Alt text](/public/upload/docker/traditional_vlan_architecture.jpg)

图3所示为一个现实世界中的 802.1Q VLAN 网络。六台电脑终端通过一级交换机接入网络，分属 VLAN 10、VLAN 20、VLAN 30。做为例子，图中左侧的交换机不支持 802.1Q VLAN，导致其连接的两台终端处于一个广播域中，尽管它们属于不同子网。作为对比，图中右侧的交换机支持 802.1Q VLAN，通过正确配置正确切割了子网的广播域，从而隔离了分属不同网段的终端。在连接外网之间，需要一个支持 802.1Q VLAN 的三层交换机，在进行数据外发时剥离 VLAN Tag，收到数据时根据IP信息转发到正确的VLAN子网。路由器根据IP信息进行NAT转换最终连接外网。



如果采用硬件支持的方式来设置vlan，交换机是划分局域网的关键设备，所以本文说xx vlan，主要是针对交换机说的。

交换机，维基百科解释：是一个扩大网络的器材，能为子网络中提供更多的port，以便连接更多的电脑。

[VLAN原理详解](https://blog.csdn.net/phunxm/article/details/9498829)

## 网桥

如果对网络不太熟悉，对于网桥的概念是很困惑的，下面试着简单解释一下。

1. 如果两台计算机想要互联

    1. 利用计算机的串口连接

    2. 利用两台计算机的网卡互联（网线的两端插到网卡的插槽上么？）

    3. 利用电话线联网

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


Switching was just a fancy name for bridging, and that was a 1980s technology – or so the thinking went.A bridge can be a physical device or implemented entirely in software. Linux kernel is able to perform bridging since 1999. Switches have meanwhile became specialized physical devices and software bridging had almost lost its place. However, with the advent of virtualization, virtual machines running on physical hosts required Layer 2 connection to the physical network and other VMs. Linux bridging provided a well proven technology and entered it’s Renaissance. 

最开始bridge是一个硬件， 也叫swtich，后来软件也可以实现bridge了，swtich就专门称呼硬件交换机了，再后来虚拟化时代到来，bridge 迎来了第二春。


[Macvlan and IPvlan basics](https://sreeninet.wordpress.com/2016/05/29/macvlan-and-ipvlan/)In linux bridge implementation, VMs or Containers will connect to bridge and bridge will connect to outside world. For external connectivity, we would need to use NAT. container 光靠 bridge 无法直接访问外网。

建议看下 [docker中涉及到的一些linux知识](http://qiankunli.github.io/2016/12/02/linux_docker.html) 对网桥源码的分析。

## 虚拟设备 ==> 虚拟网络

Linux 用户想要使用网络功能，不能通过直接操作硬件完成，而需要直接或间接的操作一个Linux 为我们抽象出来的设备，即通用的 Linux 网络设备来完成。“eth0”并不是网卡，而是Linux为我们抽象（或模拟）出来的“网卡”。除了网卡，现实世界中存在的网络元素Linux都可以模拟出来，包括但不限于：电脑终端、二层交换机、路由器、网关、支持 802.1Q VLAN 的交换机、三层交换机、物理网卡、支持 Hairpin 模式的交换机。同时，既然linux可以模拟网络设备，自然提供了操作这些虚拟的网络设备的命令或接口。

既然**Linux可以模拟网络设备**，那么现实世界中的网络拓扑结构，Linux自然也可以在一台（或多台）主机中模拟出来。用虚拟网络来模拟现实网络，这是虚拟化技术的重要一环。

传统以太网路与docker网桥及其网络模型，vlan网络与docker容器跨主机通信网络模型，其实有很大的相关性。本文举一个较为简单的例子。内容摘自[Linux 上虚拟网络与真实网络的映射][]



在一台Linux主机上进行虚拟化模拟，以下称图2.

![Alt text](/public/upload/docker/virtual_lan_architecture.jpg)

四台虚拟机通过 TAP 设备连接到接入层 Bridge 设备，接入层 Bridge 设备通过一对 VETH 设备连接到二级 Bridge 设备，主机通过一对 VETH 设备接入二级 Bridge 设备。二级 Bridge 设备进一步通过 IP Tables 、Linux 路由表与物理网卡形成数据转发关系，最终和外部物理网络连接。或者说，物理网卡接着网桥，某个容器发送数据，bridge收到，物理网卡便接收到了数据并发出。其中，物理网卡接收数据时，数据经过网络协议栈，数据包内容被修改。具体的讲，是在**网络协议栈的传输层与链路层之间**，linux根据iptables和route tables改变了数据包的相关内容，比如将数据包中的源ip由虚拟机的ip改为物理网卡的ip。


### docker单机网络模型

使用docker后，容器之间、容器与host之间的网络拓扑模型就毋庸赘言了。

- 容器之间通信。通过网桥连接。
- 容器与外界的通信。容器发包时，数据从容器出发，经过docker0，“iptables规则”实现类似图1的路由器的功能，**根据ip信息**对数据包进行调整，经eth0发出。收包过程类似。

docker容器和外界的交互，主要包含以下情况：

1. 同一主机的容器之间（两个network namespace之间如何通信 ）
2. 容器与宿主机之间（root network namespace与某个network namespace通信，本质上还是两个namespace通信）
3. 容器与宿主机所在网络的其它主机之间（实际通过端口转发，通过向iptables添加nat规则实现）
4. 容器与宿主机所在网络的其它主机的容器之间（使用覆盖网络，或OpenVswitch和pipework等）

下面主要介绍第三种情况的实现原理：NAT（通过添加iptables规则实现网络地址转换）


    root@ubuntu1:~# docker run -d -P imageid
    
    root@ubuntu1:~# docker ps
    CONTAINER ID        IMAGE                             COMMAND                CREATED             STATUS              PORTS                     NAMES
    0afe24ab86b8        docker-registry.sh/myapp:v0.0.1   "/bin/sh -c 'service   2 seconds ago       Up 1 seconds        0.0.0.0:49153->8080/tcp   stupefied_lovelace
    
    root@ubuntu1:~# docker inspect 0afe24ab86b8 | grep IP
        "IPAddress": "172.17.0.2",

    root@ubuntu1:~# sudo iptables -nvL -t nat
    
    Chain POSTROUTING (policy ACCEPT 3 packets, 432 bytes)
     pkts bytes target     prot opt in     out     source               destination
    0     0 MASQUERADE  all  --  *      !docker0  172.17.0.0/16        0.0.0.0/0

    Chain DOCKER (2 references)
     pkts bytes target     prot opt in     out     source               destination
        2   104 DNAT       tcp  --  !docker0 *       0.0.0.0/0            0.0.0.0/0            tcp dpt:49153 to:172.17.0.2:8080

Chain POSTROUTING规则会将源地址为172.17.0.0/16的包（也就是从Docker容器产生的包），并且目的地不是docker0的包，进行源地址转换，容器ip转换成主机网卡的ip。

Chain DOCKER规则就是对主机eth0收到的目的端口为80的tcp流量（不是来自docker0的包）进行DNAT转换，数据包目的ip由主机ip转换为容器ip，将流量发往172.17.0.5:80，也就是我们上面创建的docker容器。

a “routing process” must be running in the global network namespace to receive traffic from the physical interface, and route it through the appropriate virtual interfaces to to the correct child network namespaces. 

这样就实现了容器与外界的相互访问。其本质是，发包进行数据包源ip转换，接包通过（端口与容器ip的）映射关系进行目的ip转换。

在docker的网络世界里，与其说docker“容器”是如何与外界（容器，物理机，网络上的其它主机）交流的，不如说linux的 内部的network namespace是如何与外界（其它network namespace，根network namespace）交流的。




    
## 参考文献

[Docker 网络配置][]

[Linux 上虚拟网络与真实网络的映射][]

[Docker 网络配置]: http://www.oschina.net/translate/docker-network-configuration
[Linux 上的基础网络设备详解]: https://www.ibm.com/developerworks/cn/linux/1310_xiawc_networkdevice/
[Linux 上虚拟网络与真实网络的映射]: https://www.ibm.com/developerworks/cn/linux/1312_xiawc_linuxvirtnet/