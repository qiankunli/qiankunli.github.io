---

layout: post
title: Linux网络命令操作
category: 技术
tags: Linux
keywords: network ovs

---

## 简介（未完待续）

http://fishcried.com/ 有一个linux 网络的基础知识系列，要研读下

## ip命令

我们知道经典的OSI七层网络模型，学要致用，要将其掺入到对linux网络命令的理解中。

我们知道网卡有mac地址和ip地址，分别用在链路层和网络层。打个比方，MAC地址像是我们的身份证，到哪都是那个样子；IP像是居住证，换了地方信息就要变了。政府机构同时给公民发身份证和居住证以便管理动态的社会，网络管理机构则通过给所有的上网设备同时分配MAC和IP达到这个目的。（mac地址是和位置无关的，所以是不可路由的）（这个比方来自知乎）是不是有点动静结合的意思。

一开始mac地址都是烧在网卡中的，后来则是可以动态设置（虚拟的网卡mac地址就更可以改变了）。网卡的ip地址和mac地址可变也没什么关系，因为交换机和路由器可以学习所连接网络pc的ip、mac和swtich/router的port的对应关系。网卡可以设置为混杂模式，这样一个网卡可以接受所有收到的数据包（即便数据包的目的ip地址不是该网卡的ip地址）。

iproute2是一个套件，包含的是一套命令，类似于docker，所有的docker操作命令以docker开头，就可以完成关于docker的所有操作。具体的子操作，则类似于”docker network xx”。

既然网络是分层的，那么理论上讲不同的ip命令负责不同层的事情。比如`ip link` （Data Link layer，所以叫ip link）负责第二层，`ip address`负责第三层。所以，当你想设置mtu时（肯定是第二层的事），你得找`ip link`。

较高版本的linux内核支持namespace，因此ip命令还可以设置某个namespace的网卡（实际上，我们通常在root namespace执行ip命令，root namespace可以“看见”所有的子namespace）。

通过`man ip`我们可以看到

    ip link add link DEVICE [ name ] NAME
                   [ txqueuelen PACKETS ]
                   [ address LLADDR ] [ broadcast LLADDR ]
                   [ mtu MTU ]
                   type TYPE [ ARGS ]
           TYPE := [ vlan | veth | vcan | dummy | ifb | macvlan | can | bridge]
           
这说明，ip命令不仅可以添加网卡，还可以添加网桥等网络设备。

## brctl 

## pipework

## ovs

## ovs-docker


## 引用

http://fishcried.com/2016-02-09/openvswitch-ops-guide/