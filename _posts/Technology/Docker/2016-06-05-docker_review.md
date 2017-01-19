---

layout: post
title: Docker回顾
category: 技术
tags: Docker
keywords: Docker

---

## 前言

对于一个比较熟悉的知识，回味回味，经常能整出点新的想法来。

## Docker VS vm

两者都算是一种虚拟化技术，那么一个虚拟机需要什么？内存，cpu，网卡等

docker和vm的最主要区别就是

1. vm是虚拟出这些设备。假设vm中的进程申请内存，是经过Guest OS，hypervisior，Host os最终得到分配。
2. docker是隔离出这些设备（当然，网卡啥的也是虚拟的）。container中的进程直接向Host Os申请即可。


## cgroup和namespace

介绍Docker的文章都会说：cgroup负责资源限制，namepace负责资源隔离。

我们知道，Host OS中的进程，申请内存往往是没有限制的，cgroup则提供了解决了这个问题。

对于隔离，什么叫隔离，隔离了什么呢？隔离就是不能相互访问，甚至感知不到彼此的存在。对于Host OS中的进程，**OS只保证不能访问彼此的地址空间（存储程序和程序）**，但更大的资源范围，比如内存、cpu、文件系统，则是共享的，有时也能感知到对方的存在，否则也没办法rpc。**namespace能做到在更大的资源范围内隔离进程**。

从表现上看，举个例子，不同namespace的进程可以具备相同的进程id。又比如network namespace

    #ip netns add ns1
    #ip netns list
        ns1
    #ip link list
    #ip link add veth0 type veth peer name veth1
    #ip link list
        1: lo:  mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT 
            link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
        2: eth0:  mtu 1500 qdisc pfifo_fast state UNKNOWN mode DEFAULT qlen 1000
            link/ether 00:0c:29:65:25:9e brd ff:ff:ff:ff:ff:ff
        3: veth1:  mtu 1500 qdisc noop state DOWN mode DEFAULT qlen 1000
            link/ether d2:e9:52:18:19:ab brd ff:ff:ff:ff:ff:ff
        4: veth0:  mtu 1500 qdisc noop state DOWN mode DEFAULT qlen 1000
            link/ether f2:f7:5e:e2:22:ac brd ff:ff:ff:ff:ff:ff
    #ip link set veth1 netns ns1
    #ip netns exec ns1 ip link list
        1: lo:  mtu 65536 qdisc noop state DOWN mode DEFAULT 
        link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
        3: veth1:  mtu 1500 qdisc noop state DOWN mode DEFAULT qlen 1000
        link/ether d2:e9:52:18:19:ab brd ff:ff:ff:ff:ff:ff
        
这个与`docker exec container cmd`异曲同工，不过一个只限于网络的隔离，一个是全方位的隔离。

从实现上看，以mount namespace为例：

linux 进程结构体 task_struct 中保有了一个进程用到的各种资源的指针。引入了nsproxy概念后，task_struct 有一个nsproxy 成员。



	struct nsproxy {
		atomic_t count;               /* 被引用的次数 */
		struct uts_namespace *uts_ns;
		struct ipc_namespace *ipc_ns;
		struct mnt_namespace *mnt_ns;
		struct pid_namespace *pid_ns;
		struct net 	     *net_ns;
	};


对于mnt namespace，

	struct mnt_namespace {
		atomic_t		count;
		struct vfsmount *	root;///当前namespace下的根文件系统
		struct list_head	list; ///当前namespace下的文件系统链表（vfsmount list）
		wait_queue_head_t poll;
		int event;
	};
	
原来task_struct 会有一个 fs_struct成员（1.x内核），现在呢，则是task_struct ==> nsproxy ==> mnt_ns ==> vsfmount.

我们说namespace实现资源隔离，为什么可以隔离，因为数据多了一层namespace的组织，只不过不同的资源，隔离形式不同，组织形式不同，namespace介入的形式不同。

## Docker的三大概念

镜像、容器和仓库。其实跟VirtualBox对比就是：box文件，vm运行实例和`http://www.vagrantbox.es/`。docker在这里的作用就类似于vagrant。

## docker实现架构

参见孙宏亮童鞋的docker源码分析

docker架构采用C/S模式：docker client和docker daemon。和其它C/S程序没什么两样

1. 解决通信问题。约定协议，client将协议转换为二进制流，server/daemon将二进制流转换成协议。每个协议约定了几个动作/命令（参看ftp和redis），会有一个dispatcher将动作调度到不同的handler中。handler可以是一个task，然后由类似redis的单线程模型执行。也可以是一个线程，有一个线程池执行（类似tomcat）。


2. 解决业务问题。不同的软件业务不同。从代码管理的角度，docker daemon将动作/命令分为三块，每一块称为一个driver。

    a. image的下载、存储、管理和查询，因为image之间是多对多关系（也或者是layer与image之间，这个描述待确认）就是一个或多个图节点的下载、存储和查询，所以叫GraphDriver。
    
    b. NetworkDriver，网络设备的创建和配置。在docker网络中，网络设备主要用的linux虚拟设备，倒不是像内存一样靠隔离出来的。
    
    c. ExecDriver，负责创建容器的命名空间、容器资源使用的限制与统计和内部进程的真正运行。
    
三个driver（除GraphDriver外）也只是调用libcontainer的接口，libcontainer是一个go语言库，符合一套接口标准（有点类似jvm规范）。我们知道，windows和mac也支持go语言，如果它们也提供对这套标准接口的支持，那么上述的docker client/daemon也可以完美运行。**所以从某种角度看，docker不是一个具体的软件，而是一种理念，通过资源限制和隔离（扩大一个线程（组）独有的资源范围）来使用计算机资源。
**
## 使用Docker要解决的几个基本问题

完全的隔离并不是好事，免不了要通信和共享

### Docker的网络模型

现实生活中，以太网的拓扑结构在不断的发展（简单以太网、vlan和vxlan等），那么如何在单机环境或多机环境中用虚拟网络模拟物理网络，是一个重要问题。换个描述方式：

1. 如何实现容器互通
2. 如何实现容器与宿主机互通
3. 如何实现容器与宿主机所在网络的其它物理主机互通
4. 如何实现跨主机容器互通

### Docker文件系统

类似的

1. 容器之间如何共享文件
2. 容器与宿主机之间如何共享文件
3. 跨主机容器之间如何共享文件

## Docker实践

为什么要用到docker，这就要谈到其优势，它的优势分为两部分

1. 虚拟化技术本身的优势

    资源可以更细粒度的分配和释放（以前只能是物理机维度，并且不能够软件定义）。或者说，先有“资源可以更细粒度的分配和释放”的需求，然后催生了虚拟化技术，然后出现了虚拟网络模拟现实网络之类的问题。

2. docker相对于其它虚拟化技术的优势（其实，OpenStack也可以做到以下几点，但OpenStack（作为IaaS）侧重点不在这里，一个Openstack VM通常也不会只跑一个应用）

    - 隔离性。一个应用程序的可靠运行，除了代码的正确性，一个可靠地env是必不可少的一部分，有了docker，这个env可以归这个应用程序专有。
    
    - 可移植性。env可以像配置文件一样保存和转移。
    
    - 进程级的虚拟化，非常有利于扩容和缩容时的响应速度。

## docker学习路线图

[Docker学习路线图 (持续更新中)][]

[Docker学习路线图 (持续更新中)]: https://yq.aliyun.com/articles/40494