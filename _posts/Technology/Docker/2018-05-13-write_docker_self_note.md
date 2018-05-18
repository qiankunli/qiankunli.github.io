---

layout: post
title: 《自己动手写docker》笔记
category: 技术
tags: Docker
keywords: Docker

---


## 简介（未完成）

## namespace

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

	![](/public/upload/docker/pid_namespace.png)

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

## 启动流程

![](/public/upload/docker/docker_note_1.png)

改图是两个角色，并不准确，还未准确理解mydocker，mydocker runCommand进程，mydocker initCommand进程，以及用户自定义command的作用关系。但大体可以分为 mydocker 一方，容器一方。

据猜测应该是，mydocker runCommand 进程 启动 了 initCommand 进程（相当于 `mydocker run` 内部执行了 `mydocker init`），同时为两者沟通建立管道，并为其配置cgroup。
initComand 启动后挂载文件系统，从管道中读取用户输入的command，用户输入的command 顶替掉 initCommand
	

## 其它

[Docker：一场令人追悔莫及的豪赌](https://mp.weixin.qq.com/s?__biz=MzA5OTAyNzQ2OA==&mid=2649697704&idx=1&sn=5f7ef3d2f9d5e2c7b33b1085559fd0f5&chksm=889312cbbfe49bddaa66ba5a6f761531a0baf8f6ee572bd24e2720066987e3dc1ff3fafacc51&mpshare=1&scene=23&srcid=0515E2lfioYH9HzxGsd8SmWZ%23rd) 要点如下：

1. 将所有依赖关系、一切必要配置乃至全部必要的资源都塞进同一个二进制文件，从而简化整体系统资源机制。 如果具备这种能力，docker 便不是必要的。

	* 静态链接库方式。一些语言，比如go，可以将依赖一起打包，生成一个毫无依赖的二进制文件。
	* macos 则从 更广泛的意义上实现了 该效果
2. Docker 的重点 不在于 提供可移植性、安全性以及资源管理与编排能力，而是 标准化。docker 做的事情以前各种运维工程师也在做，只是在A公司的经验无法复制到B公司上。
3. 在我看来，Docker有朝一日将被定性为一个巨大的错误。其中最强有力的论据在于，即使最终成为标准、始终最终发展成熟，Docker也只是为科技行业目前遭遇的种种难题贴上一张“创可贴”



[Is K8s Too Complicated?](http://jmoiron.net/blog/is-k8s-too-complicated/) 未读完

	* Kubernetes is a complex system. It does a lot and brings new abstractions. Those abstractions aren't always justified for all problems. I'm sure that there are plenty of people using Kubernetes that could get by with something simpler.
	* That being said, I think that, as engineers, we tend to discount the complexity we build ourselves vs. complexity we need to learn. 个人也觉得 k8s 管的太宽了，实现基本功能，外围的我们自己做，也比学它的弄法要好。

	
	
[An architecture of small apps](http://www.smashcompany.com/technology/an-architecture-of-small-apps)