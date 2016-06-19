---

layout: post
title: Docker网络四,基于Centos搭建Docker跨主机网络
category: 技术
tags: Docker
keywords: Docker,OVS

---

## 前言（待整理）


http://www.cnblogs.com/yuuyuu/p/5180827.html#commentform 最后两步可以换成

pipework vxbr -i  eth0 157e10effa9c 10.1.2.3/24


pipework eth1 $CONTAINERID dhclient 能做大自动获取ip，但是他妈的和ovs一结合就做不到，看来得看ovs原理。

pipework eth0 157e10effa9c dhclient。ip 从10.1.2.16开始，这是virtualbox的，字节搭建的dhcp服务就没有起来。



环境，centos6.6



iproute安装

http://www.xiaomastack.com/2015/04/05/centos6-ip-netns/



dhcp 客户端会发广播包询问，http://www.ahlinux.com/server/dhcp/17429.html


搭建dhcp

http://www.linuxde.net/2011/07/324.html

dhcp还可以为镜像指定dns地址


https://opsbot.com/advanced-docker-networking-pipework/


通过dhcp指定ip地址，pipework eth0 77c767808af0 dhclient

dhclient 要监听一个网卡（这才是正确的使用姿势），dhclient执行完就算了，好像要周期性的给dhcp server汇报的。


http://www.cnblogs.com/CasonChan/p/4604871.html（可以借鉴下，容器局域网，提供一个网卡，这个网卡没有绑定容器，是一个inernal类型的网卡，绑在ovs网桥上，一个ip工具监听这个网卡。这样容器启动后，如果先使用dhcp的话，这个internal网卡收到广播请求，就可以得到ip了）



## 通过dhcp获取ip

https://goldmann.pl/blog/2014/01/30/assigning-ip-addresses-to-docker-containers-via-dhcp/


## 这个是系列，直接齐了

https://goldmann.pl/blog/2014/01/21/connecting-docker-containers-on-multiple-hosts/



## 移除docker的时候，记得用ovs删除一下port



## 保留原来的docker0网桥，只是将容器添加到新的网桥

为了端口映射的需要

http://www.sdnlab.com/13111.html


# 工作

1. 重新学习dhcp原理和使用
2. pipework dhcp翻译一遍，最好搞清楚原理
3. ovs-docker搞清楚原理

