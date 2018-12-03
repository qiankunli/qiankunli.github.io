---

layout: post
title: docker中涉及到的一些linux知识
category: 技术
tags: Linux
keywords: network 

---

## 简介

## namespace

### 来源

命名空间最初是用来解决命名唯一性问题的，即解决不同编码人员编写的代码模块在合并时可能出现的重名问题。

传统上，在Linux以及其他衍生的UNIX变体中，许多资源是全局管理的。这意味着进程之间彼此可能相互影响。偏偏有这样一些场景，比如一场“黑客马拉松”的比赛，组织者需要运行参赛者提供的代码，为了防止一些恶意的程序，必然要提供一套隔离的环境，一些提供在线持续集成服务的网站也有类似的需求。

我们不想让进程之间相互影响，就必须将它们隔离起来，最好都不知道对方的存在。而所谓的隔离，便是隔离他们使用的资源（比如），进而资源的管理也不在是全局的了。

### 原理

[Namespaces in operation, part 1: namespaces overview](https://lwn.net/Articles/531114/) 是一个介绍 namespace 的系列文章，要点如下：

1.  The purpose of each namespace is to wrap a particular global system resource in an abstraction that makes it appear to the processes within the namespace that they have their own isolated instance of the global resource. 对global system resource的封装
2. there will probably be further extensions to existing namespaces, such as the addition of namespace isolation for the kernel log. 将会有越来越多的namespace


namespace 简单说，就是进程的task_struct 以前都直接 引用资源id（各种资源或者是index，或者 是一个地址），现在是进程  task struct   ==> nsproxy ==> 资源表(操作系统就是提供抽象，并将各种抽象封装为数据结构，外界可以引用)

[Linux内核的namespace机制分析](https://blog.csdn.net/xinxing__8185/article/details/51868685)

	struct task_struct {	
		……..		
		/* namespaces */		
		struct nsproxy *nsproxy;	
		…….
	}
	struct nsproxy {
         atomic_t count;	// nsproxy可以共享使用，count字段是该结构的引用计数
         struct uts_namespace *uts_ns;
         struct ipc_namespace *ipc_ns;
         struct mnt_namespace *mnt_ns;
         struct pid_namespace *pid_ns_for_children;
         struct net             *net_ns;
	};


[What is the relation between `task_struct` and `pid_namespace`?](https://stackoverflow.com/questions/26779416/what-is-the-relation-between-task-struct-and-pid-namespace)


[Separation Anxiety: A Tutorial for Isolating Your System with Linux Namespaces](https://www.toptal.com/linux/separation-anxiety-isolating-your-system-with-linux-namespaces) 该文章 用图的方式，解释了各个namespace 生效的机理，值得一读。其实要理解的比较通透，首先就得对 linux 进程、文件、网络这块了解的比较通透。**此外，虽说都是隔离，但他们隔离的方式不一样，比如root namespace是否可见，隔离的资源多少（比如pid只隔离了pid，mnt则隔离了root directory 和 挂载点，network 则隔离网卡、路由表、端口等所有网络资源），隔离后跨namespace如何交互**

1. 进程和 namespace 通常是多对多关系
2. 进程是树结构的，每个namespace 理解的 根不一样，pid root namespace  最终提供完整视图

	![](/public/upload/linux/pid_namespace.png)

3. mount 也是有树的，每个namespace 理解的根 不一样, 挂载点目录彼此看不到. task_struct  ==> nsproxy 包括 mnt_namespace。

		struct mnt_namespace {
			atomic_t		count;
			struct vfsmount *	root;///当前namespace下的根文件系统
			struct list_head	list; ///当前namespace下的文件系统链表（vfsmount list）
			wait_queue_head_t poll;
			int event;
		};
		struct vfsmount {
			...
			struct dentry *mnt_mountpoint;	/* dentry of mountpoint,挂载点目录 */
			struct dentry *mnt_root;	/* root of the mounted tree,文件系统根目录 */
			...
		}
		
	[Mount Point Definition](http://www.linfo.org/mount_point.html)A mount point is a directory in the currently accessible filesystem on which an additional filesystem is mounted, 对于一个linux 来说，一般顶层rootfs，然后加载`/etc/fstab` 加载那些默认的挂载点。
	
	只是单纯一个隔离的 mnt namespace 环境是不够的，还要"change root"，参见《自己动手写docker》P45

4. network namespace 倒是没有根， 但docker 创建 veth pair，root namespace 一个，child namespace 一个。此外 为 root namespace 额外加 iptables 和 路由规则，为 各个ethxx 提供路由和数据转发，并提供跨network namesapce 通信。

[Mount Point Definition](http://www.linfo.org/mount_point.html)A mount point is a directory in the currently accessible filesystem on which an additional filesystem is mounted. 对于一个linux 来说，一般顶层rootfs，然后加载`/etc/fstab` 加载那些默认的挂载点

从mnt 和 network namespace 可以看到， 一个可用的 容器主要 是一个隔离的 环境，其次还需要 docker 进行 各种微操以补充。 

《深入剖析kubernetes》：用户运行在容器里的应用进程，跟宿主机上的其他进程一样，都由宿主机操作系统统一管理，只不过这些被隔离的进程拥有额外设置过的Namespace 参数。而docker 在这里扮演的角色，更多的是旁路式的辅助和管理工作。 

## cgroups

[使用cgroups控制进程cpu配额](http://www.pchou.info/linux/2017/06/24/cgroups-cpu-quota.html)

cgroups Control Group，原来叫process group，是分配资源的基本单位。cgroup 具备继承关系，因此可以组成 hierarchy。子系统（subsystem），一个子系统就是一个（只是一个）资源控制器，子系统必须附加（attach）到一个hierarchy上才能起作用

从操作上看：

1. 可以创建一个目录（比如叫cgroup-test）， `mount -t cgroup -o none  cgroup-test ./cgroup-test` cgroup-test 便是一个hierarchy了，一个hierarchy 默认自动创建很多文件

		- cgroup.clone_children
		- cgroup.procs
		- notify_on_release
		- tasks

你为其创建一个子文件`cgroup-test/	cgroup-1`，则目录变成

		- cgroup.clone_children
		- cgroup.procs
		- notify_on_release
		- tasks
		- cgroup-1
			- cgroup.clone_children
			- cgroup.procs
			- notify_on_release
			- tasks

往task 中写进程号，则标记该进程 属于某个cgroup。

注意，mount时，`-o none` 为none。 若是  `mount -t cgroup -o cpu cgroup-test ./cgroup-test` 则表示为cgroup-test  hierarchy 挂载 cpu 子系统

	- cgroup.event_control
	- notify_on_release
	- cgroup.procs
	- tasks
	
	- cpu.cfs_period_us
	- cpu.rt_period_us
	- cpu.shares
	- cpu.cfs_quota_us
	- cpu.rt_runtime_us
	- cpu.stat
	
cpu 开头的都跟cpu 子系统有关。可以一次挂载多个子系统，比如`-o cpu,mem`


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

2018.12.3 补充：一旦一张虚拟网卡被“插”在网桥上，它就会变成该网桥的“从设备”。从设备会被“剥夺”调用网络协议栈处理数据包的资格，从而“降级”成为网桥上的一个端口。而这个端口唯一的作用，就是接收流入的数据包，然后把这些数据包的“生杀大权”（比如转发或者丢弃），全部交给对应的网桥。