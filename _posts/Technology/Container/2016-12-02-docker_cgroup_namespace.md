---

layout: post
title: docker namespace和cgroup
category: 技术
tags: Container
keywords: network 

---

## 简介

* TOC
{:toc}

刘超《趣谈linux操作系统》

从一台物理机虚拟化出很多的虚拟机这种方式，一定程度上实现了资源创建的灵活性。但虚拟化的方式还是非常复杂的，CPU、内存、网络、硬盘全部需要虚拟化，一个都不能偷懒。有没有一种更加灵活的方式，专门用于某个进程，又不需要费劲周折的虚拟化这么多的硬件呢？毕竟最终我只想跑一个程序，而不是要一整个 Linux 系统。就像在一家大公司搞创新，如果每一个创新项目都要成立一家子公司的话，那简直太麻烦了。一般方式是在公司内部成立一个独立的组织，分配独立的资源和人力，先做一段时间的内部创业。如果真的做成功了，再成立子公司也不迟。

容器的英文叫 Container，Container 的另一个意思是“集装箱”。其实容器就像船上的不同的集装箱装着不同的货物，有一定的隔离，但是隔离性又没有那么好，仅仅做简单的封装。当然封装也带来了好处，一个是打包，二是标准。有了集装箱还不行，大家的高长宽形状不一样也不方便，还要通过镜像将这些集装箱标准化，使其**在哪艘船上都能运输**，在哪个码头都能装卸（在哪个物理机上都能跑），就好像集装箱在开发、测试、生产这三个码头非常顺利地整体迁移，这样产品的发布和上线速度就加快了。[什么是标准容器（2021 版）](https://mp.weixin.qq.com/s/73qpg0RTr3gd79D5Bz3BQQ)

除了可以如此简单地创建一个操作系统环境，容器还有一个很酷的功能，就是镜像里面带应用。这样的话，应用就可以像集装箱一样，到处迁移，启动即可提供服务。而不用像虚拟机那样，要先有一个操作系统的环境，然后再在里面安装应用。

![](/public/upload/linux/docker_theory.jpg)

两个基本点

1. 数据结构：namespace 和 cgroups 数据在内核中如何组织
2. 算法：内核如何应用namespace 和 cgroups 数据

## namespace

《深入剖析kubernetes》：用户运行在容器里的应用进程，跟宿主机上的其他进程一样，都由宿主机操作系统统一管理，只不过这些被隔离的进程拥有额外设置过的Namespace 参数。而docker 在这里扮演的角色，更多的是旁路式的辅助和管理工作。 

linux内核对命名空间的支持完全隔离了工作环境中应用程序的视野。

### 来源

命名空间最初是用来解决命名唯一性问题的，即解决不同编码人员编写的代码模块在合并时可能出现的重名问题。

传统上，在Linux以及其他衍生的UNIX变体中，许多资源是全局管理的。这意味着进程之间彼此可能相互影响。偏偏有这样一些场景，比如一场“黑客马拉松”的比赛，组织者需要运行参赛者提供的代码，为了防止一些恶意的程序，必然要提供一套隔离的环境，一些提供在线持续集成服务的网站也有类似的需求。

我们不想让进程之间相互影响，就必须将它们隔离起来，最好都不知道对方的存在。而所谓的隔离，便是隔离他们使用的资源（比如），进而资源的管理也不在是全局的了。

### namespace 内核数据结构

[Namespaces in operation, part 1: namespaces overview](https://lwn.net/Articles/531114/) 是一个介绍 namespace 的系列文章，要点如下：

1.  The purpose of each namespace is to wrap a particular global system resource in an abstraction that makes it appear to the processes within the namespace that they have their own isolated instance of the global resource. 对global system resource的封装
2. there will probably be further extensions to existing namespaces, such as the addition of namespace isolation for the kernel log. 将会有越来越多的namespace


namespace 简单说，就是进程的task_struct 以前都直接 引用资源id（各种资源或者是index，或者 是一个地址），现在是进程  task struct   ==> nsproxy ==> 资源表(操作系统就是提供抽象，并将各种抽象封装为数据结构，外界可以引用)

[Linux内核的namespace机制分析](https://blog.csdn.net/xinxing__8185/article/details/51868685)

```c
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
```


[What is the relation between `task_struct` and `pid_namespace`?](https://stackoverflow.com/questions/26779416/what-is-the-relation-between-task-struct-and-pid-namespace)


[Separation Anxiety: A Tutorial for Isolating Your System with Linux Namespaces](https://www.toptal.com/linux/separation-anxiety-isolating-your-system-with-linux-namespaces) 该文章 用图的方式，解释了各个namespace 生效的机理，值得一读。其实要理解的比较通透，首先就得对 linux 进程、文件、网络这块了解的比较通透。**此外，虽说都是隔离，但他们隔离的方式不一样，比如root namespace是否可见，隔离的资源多少（比如pid只隔离了pid，mnt则隔离了root directory 和 挂载点，network 则隔离网卡、路由表、端口等所有网络资源），隔离后跨namespace如何交互**

### namespace 生效机制

![](/public/upload/linux/linux_namespace_object.png)

### pid namespace

进程是树结构的，每个namespace 理解的 根不一样，pid root namespace  最终提供完整视图

![](/public/upload/linux/pid_namespace.png)

### mount namespace

mount 也是有树的，每个namespace 理解的根 不一样, 挂载点目录彼此看不到. task_struct  ==> nsproxy 包括 mnt_namespace。

```c
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
```
		
只是单纯一个隔离的 mnt namespace 环境是不够的，还要"change root"，参见《自己动手写docker》P45

《阿里巴巴云原生实践15讲》chroot 的作用是“重定向进程及其子进程的根目录到一个文件系统 上的新位置”，使得该进程再也**看不到也没法接触到这个位置上层的“世界”**。所以这 个被隔离出来的新环境就有一个非常形象的名字，叫做 Chroot Jail。

### network namespace

Linux 目前提供的八种名称空间里，网络名称空间无疑是隔离内容最多的一种，它为命名空间内的所有进程提供了全套的网络设施，包括独立的设备界面、路由表、ARP 表，IP 地址表、iptables/ebtables 规则、协议栈，等等。

## cgroups

[彻底搞懂容器技术的基石： cgroup](https://mp.weixin.qq.com/s/6Ts6-aZDr8qOdnaNUqwTFQ)

### V1 和 V2

![](/public/upload/container/cgroup_v1.jpeg)

Cgroup v1 的一个整体结构，每一个子系统都是独立的，资源的限制只能在子系统中发生。比如pid可以分别属于 memory Cgroup 和 blkio Cgroup。但是在 blkio Cgroup 对进程 pid 做磁盘 I/O 做限制的时候，blkio 子系统是不会去关心 pid 用了哪些内存，这些内存是不是属于 Page Cache，而这些 Page Cache 的页面在刷入磁盘的时候，产生的 I/O 也不会被计算到进程 pid 上面。**Cgroup v2 相比 Cgroup v1 做的最大的变动就是一个进程属于一个控制组，而每个控制组里可以定义自己需要的多个子系统**。Cgroup v2 对进程 pid 的磁盘 I/O 做限制的时候，就可以考虑到进程 pid 写入到 Page Cache 内存的页面了，这样 buffered I/O 的磁盘限速就实现了。

![](/public/upload/container/cgroup_v2.jpeg)

### 整体实现（可能过时了）

[cgroup 原理与实现](https://mp.weixin.qq.com/s/yXJxTR_sPdEMt56cf7JPhQ) 写的很清晰了

![](/public/upload/linux/cgroup_struct.png)

1. 包含哪些数据结构
2. CGroup 的挂载
3. 向 CGroup 添加要进行资源控制的进程（写文件 ==> 数据结构间建立关联关系）
4. 如何 限制 CGroup 的资源使用

![](/public/upload/linux/linux_cgroup_object.png)

### 从右向左 ==> 和docker run放在一起看 

![](/public/upload/linux/linux_cgroup_docker.png)

### 从左向右 ==> 从 task 结构开始找到 cgroup 结构

[Docker 背后的内核知识——cgroups 资源限制](https://www.infoq.cn/article/docker-kernel-knowledge-cgroups-resource-isolation/)

在图中使用的回环箭头，均表示可以通过该字段找到所有同类结构

![](/public/upload/linux/linux_task_cgroup.png)

### 从右向左 ==> 查看一个cgroup 有哪些task

![]()(/public/upload/linux/linux_task_cgroup.png)

为什么要使用cg_cgroup_link结构体呢？因为 task 与 cgroup 之间是多对多的关系。熟悉数据库的读者很容易理解，在数据库中，如果两张表是多对多的关系，那么如果不加入第三张关系表，就必须为一个字段的不同添加许多行记录，导致大量冗余。通过从主表和副表各拿一个主键新建一张关系表，可以提高数据查询的灵活性和效率。

### 使cgroups 数据生效

我们知道， 任何内存申请都是从缺页中断开始的，`handle_pte_fault ==> do_anonymous_page ==> mem_cgroup_newpage_charge（不同linux版本方法名不同） ==> mem_cgroup_charge_common ==> __mem_cgroup_try_charge`

```c
static int __mem_cgroup_try_charge(struct mm_struct *mm,
    gfp_t gfp_mask,
    unsigned int nr_pages,
    struct mem_cgroup **ptr,
    bool oom){
    ...
    struct mem_cgroup *memcg = NULL;
    ...
    memcg = mem_cgroup_from_task(p);
    ...
}
```

`mem_cgroup_from_task ==> mem_cgroup_from_css` 

```c
struct mem_cgroup *mem_cgroup_from_task(struct task_struct *p){
    ...
    return mem_cgroup_from_css(task_subsys_state(p, mem_cgroup_subsys_id));
}
```








