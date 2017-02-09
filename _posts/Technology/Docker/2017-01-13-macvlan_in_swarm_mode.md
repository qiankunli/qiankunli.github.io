---

layout: post
title: macvlan in swarm mode
category: 技术
tags: Docker
keywords: Docker,macvlan

---


## 简介（待整理）

以下实现基于docker1.13，物理机使用`192.168.0.0/16`网段，容器使用`172.31.0.0/16`网段。

整体思路：每个docker host创建local scope的macvlan network，自定义ipam plugin负责ip地址管理。容器使用macvlan网络，由外置交换机负责容器之间、host之间、容器与host之间的连通性。

docker macvlan 用802.1q模式，对于一个交换机端口来说，物理机和容器的数据包属于不同的vlan，so 交换机端口设置为trunk；也属于不同的网段，在交换机的三层加一层路由，打通物理机和容器的两个网段。


## macvlan网络

### 物理机创建vlan的sub interface


1. Load the 8021q module into the kernel.

	`sudo modprobe 8021q`

2. **Create a new interface that is a member of a specific VLAN**, 
VLAN id 10 is used in this example. Keep in mind you can only use physical interfaces as a base, creating VLAN's on virtual interfaces (i.e. eth0:1) will not work. We use the physical interface eth1 in this example. This command will add an additional interface next to the interfaces which have been configured already, so your existing configuration of eth1 will not be affected.
	
	`sudo vconfig add eth1 10`

3. Assign an address to the new interface.

	`sudo ip addr add 10.0.0.1/24 dev eth0.10`

4. Starting the new interface.

	`sudo ip link set up eth0.10`
	
	
简单说，就是只有基于sub interface（eth1.10），你发出去的数据包，才会有802.1q中的vlan tag。
	
### 基于sub interface创建docker macvlan 网络

	docker network  create  -d macvlan \
	    --subnet=172.31.0.0/16 \
	    --gateway=172.31.0.1 \
	    -o parent=eth0.10 macvlan10

### 创建容器，指定使用macvlan网络

	docker run --net=macvlan10 -it --name macvlan_test5 --rm alpine /bin/sh