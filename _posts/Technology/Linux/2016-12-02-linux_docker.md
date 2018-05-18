---

layout: post
title: docker中涉及到的一些linux知识
category: 技术
tags: Linux
keywords: network 

---

## 简介

## 镜像文件

An archive file is a file that is composed of one or more computer files **along with metadata**. Archive files are used to collect multiple data files together into a single file for easier portability and storage, or simply to compress files to use less storage space. Archive files often store directory structures, error detection and correction information, arbitrary comments, and sometimes use built-in encryption.

文件系统 是解决 根据 file name 找 file data的问题，从这个角度看，文件系统跟dns 有点异曲同工的意思。

rootfs是基于内存的文件系统，所有操作都在内存中完成；也没有实际的存储设备，所以不需要设备驱动程序的参与。基于以上原因，Linux在启动阶段使用rootfs文件系统，当磁盘驱动程序和磁盘文件系统成功加载后，linux系统会将系统根目录从rootfs切换到磁盘文件系统（这句表述不准确）。

所以呢，文件系统有内存文件系统，磁盘文件系统，还有基于磁盘文件系统之上的联合文件系统。

参见[linux文件系统初始化过程(2)---挂载rootfs文件系统
](http://blog.csdn.net/luomoweilan/article/details/17894473),linux文件系统中重要的数据结构有：文件、挂载点、超级块、目录项、索引节点等。图中含有两个文件系统（红色和绿色表示的部分），并且绿色文件系统挂载在红色文件系统tmp目录下。一般来说，每个文件系统在VFS层都是由挂载点、超级块、目录和索引节点组成；当挂载一个文件系统时，实际也就是创建这四个数据结构的过程，因此这四个数据结构的地位很重要，关系也很紧密。**由于VFS要求实际的文件系统必须提供以上数据结构，所以不同的文件系统在VFS层可以互相访问。**
    如果进程打开了某个文件，还会创建file(文件)数据结构，这样进程就可以通过file来访问VFS的文件系统了。

![](/public/upload/linux/linux_fs.png)

这个图从上往下看，可以知道，各个数据结构通过数组等自己组织在一起，又通过引用上下关联。

划重点：

1. 文件系统有事实上的规范——VFS，定义了挂载点、超级块、目录和索引节点等基本数据结构，定义了open/close/write/read 等基本接口
2. 因为VFS的存在，一个linux 实际上可以运转多个文件系统


## linux网桥

本文所说的网桥，主要指的是linux 虚拟网桥。

A bridge transparently relays traffic between multiple network interfaces. **In plain English this means that a bridge connects two or more physical Ethernets together to form one bigger (logical) Ethernet** 


<table>
	<tr>
		<td>network layer</td>
		<td colspan="3">iptables rules</td>
	</tr>
	<tr>
		<td>func</td>
		<td>netif_receive_skb/dev_queue_xmit</td>
		<td colspan=2>netif_receive_skb/dev_queue_xmit</td>
	</tr>
	<tr>
		<td rowspan="2">data link layer</td>
		<td rowspan="2">eth0</td>
		<td colspan="2">br0</td>
	</tr>
	<tr>
		<td>eth1</td>
		<td>eth2</td>
	</tr>
	<tr>
		<td>func</td>
		<td>rx_handler/hard_start_xmit</td>
		<td>rx_handler/hard_start_xmit</td>
		<td>rx_handler/hard_start_xmit</td>
	</tr>
	<tr>
		<td>phsical layer</td>
		<td>device driver</td>
		<td>device driver</td>
		<td>device driver</td>
	</tr>
</table>

通俗的说，网桥屏蔽了eth1和eth2的存在。正常情况下，每一个linux 网卡都有一个device or net_device struct.这个struct有一个rx_handler。

eth0驱动程序收到数据后，会执行rx_handler。rx_handler会把数据包一包，交给network layer。从源码实现就是，接入网桥的eth1，在其绑定br0时，其rx_handler会换成br0的rx_handler。等于是eth1网卡的驱动程序拿到数据后，直接执行br0的rx_handler往下走了。所以，eth1本身的ip和mac，network layer已经不知道了，只知道br0。

br0的rx_handler会决定将收到的报文转发、丢弃或提交到协议栈上层。如果是转发，br0的报文转发在数据链路层，但也会执行一些本来属于network layer的钩子函数。也有一种说法是，网桥处于forwarding状态时，报文必须经过layer3转发。这些细节的确定要通过学习源码来达到，此处先不纠结。

读了上文，应该能明白以下几点。

1. 为什么要给网桥配置ip，或者说创建br0 bridge的同时，还会创建一个br0 iface。
2. 为什么eth0和eth1在l2,连上br0后，eth1和eth0的连通还要受到iptables rule的控制。
3. 网桥首先是为了屏蔽eth0和eth1的，其次是才是连通了eth0和eth1。