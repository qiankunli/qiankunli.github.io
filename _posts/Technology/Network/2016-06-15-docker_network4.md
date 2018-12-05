---

layout: post
title: Docker网络四,基于Centos搭建Docker跨主机网络
category: 技术
tags: Network
keywords: Docker,OVS

---

## 前言(就目前docker network发展看，已有些过时)

docker 1.10以上版本的内置overlay网络挺好用的，奈何公司的线上环境必须是centos6.6，内核版本不够，只能基于ovs另搭建一套网络环境，使容器能够跨主机互通。

本文基于的操作环境，centos6.6

因为要操作network namespace，所以要升级下iproute，参见`http://www.xiaomastack.com/2015/04/05/centos6-ip-netns/`

本文从docker容器使用ovs网桥开始，最终搭建一个跨主机容器通信网络。

## Docker 使用 OpenvSwitch 网桥

主要步骤可以参见 [Docker 使用 OpenvSwitch 网桥][]

这里边有两个改进的地方（其实是一个）

1. 无需到Container内部设置容器ip

    设置容器的ip时，可以使用ovs-docker。`https://github.com/openvswitch/ovs/raw/master/utilities/ovs-docker`。

    比如`ovs-docker add-port vxbr eth1 fe8d1b02dc90 --ipaddress=10.1.2.3/24`
    
    或者
    
        ovs-docker add-port vxbr eth1 fe8d1b02dc90
         // 获取容器进程id
        docker inspect --format "{{ \.State.Pid }}" fe8d1b02dc90
        1915
        ln -s /proc/1915/ns/net /var/run/netns/1915
        ip netns exec 1935 ifconfig eth1 10.1.2.3/24
    
    
2. 容器的ip从dhcp服务器获取，而非静态设置

## 从dhcp服务器获取容器ip

首先，需要搭建一个dhcp服务器，参见[详解如何搭建DHCP服务器][]，修改配置文件。

`cp  /usr/share/doc/dhcp-3.0.5/dhcpd.conf.sample /etc/dhcp/dhcpd.conf`，添加如下内容：

    subnet 10.1.2.0 netmask 255.255.255.0 {
        range 10.1.2.100 10.1.2.200;
        option subnet-mask 255.255.255.0;
        # option routers 10.1.2.1;
    }

在这里，**我们只需要根据dhcp获取动态ip，至于routers等信息则没有必要**。配置routers会更改容器的路由表，跟我们的下列设计不符。

安装后，打开防火墙，允许其它主机访问67端口。

    iptables -A INPUT -m state --state NEW -p udp --dport 67 -j ACCEPT


创建容器并设置动态ip，（**参考pipework源码**）

    // 创建容器，此处只是用ovs网桥，所有使用none网络模式
    docker run -d -P --net=none 192.168.3.56:5000/tomcat7 bash /start.sh
    fe8d1b02dc90
    // 为容器添加网卡
    ovs-docker add-port vxbr eth0 fe8d1b02dc90
    // 获取容器进程id
    docker inspect --format "{{ \.State.Pid }}" fe8d1b02dc90
    3599
    // ip套件默认操作的是/var/run/netns下的数据
    ln -s /proc/3599/ns/net /var/run/netns/3599
    // 设定网卡ip
    ip netns exec 3599 dhclient eth0 -pf /var/run/dhclient-fe8d1b02dc90-eth0.pid \
    -lf /var/lib/dhclient/dhclient-fe8d1b02dc90-eth0.lease
    

使用dhclient（一种dhcp客户端）时，常见用法是`dhcpclient iface`，如果想要设置多个iface。（可以参见`man dhclient`）

1. `dhcpclient iface1 iface2`
2. 指定pid file和lease file

笔者在实践时，有一个疑问，pipework可以分开做以下两件事

1.  `pipework vxbr -i eth0 $CONTAINERID 10.1.2.3/24`，为容器增加eth0网卡并设置ip
2.  `ovs-docker add-port vxbr eth0 $CONTAINERID &&`
    `pipework eth0 $CONTAINERID dhclient`

合起来干，笔者没走通，最后只好直接用pipework dhcp特定的底层实现。

## 跨主机容器互通

上文解决了容器使用ovs网桥并动态设置ip的问题，再将不同主机的ovs网桥“连起来”，就可以解决容器互通问题。参见[Docker+OpenvSwitch搭建VxLAN实验环境][]

假设已在`192.168.56.105`和`192.168.56.106`上完成上述操作，则

在`192.168.56.105`上

    ovs-vsctl add-port vxbr vxlan -- set interface vxlan type=vxlan options:remote_ip=192.168.56.106
    
在`192.168.56.106`上

    ovs-vsctl add-port vxbr vxlan -- set interface vxlan type=vxlan options:remote_ip=192.168.56.105
    
即可实现`192.168.56.105`和`192.168.56.106`上容器的跨主机通信。


## 一个容器两个网桥

每个主机安装一个ovs，通过上文的操作可以为容器提供一个overlay网络，保证容器间互通。那么容器与外界的交互呢？办法之一是，一个容器两个网卡（假设叫eth0和eth1），eth0连接docker0（即保持容器原有的docker0网络），负责与外界网络通信。eth1连接ovs网桥，负责容器的跨主机访问。

另一个办法是，容器只有一个网卡，连接ovs网桥，自己设置iptables规则（还未实验）

## 其它要做的

1. 移除docker的时候，记得用ovs-docker删除一下port
2. 部分操作脚本化
3. 系统重启后，docker容器重启需要做什么样的工作
4. 这套办法想办法直接做成docker的network plugin
5. 清除`/var/run/netns/`下的软链接


[Docker 使用 OpenvSwitch 网桥]: http://blog.csdn.net/yeasy/article/details/42555431
[详解如何搭建DHCP服务器]: http://www.ahlinux.com/server/dhcp/17429.html
[Docker+OpenvSwitch搭建VxLAN实验环境]: http://www.cnblogs.com/yuuyuu/p/5180827.html#commentform