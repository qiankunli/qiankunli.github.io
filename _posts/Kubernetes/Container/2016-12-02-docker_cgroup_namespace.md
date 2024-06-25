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

[使用 Go 和 Linux Kernel 技术探究容器化原理](https://mp.weixin.qq.com/s/Z5j0LPYQE5dCR0LOzbUlrQ) 写的非常好，与go 调用有结合，建议细读。

## namespace

《深入剖析kubernetes》：用户运行在容器里的应用进程，跟宿主机上的其他进程一样，都由宿主机操作系统统一管理，只不过这些被隔离的进程拥有额外设置过的Namespace 参数。而docker 在这里扮演的角色，更多的是旁路式的辅助和管理工作。 

linux内核对命名空间的支持完全隔离了工作环境中应用程序的视野。

一个进程的每种Linux namespace 都可以在对应的`/proc/进程号/ns/` 下有一个对应的虚拟文件，并且链接到一个真实的namespace 文件上。通过`/proc/进程号/ns/` 可以感知到进程的所有linux namespace；一个进程 可以选择加入到一个已经存在的namespace 当中；也就是可以加入到一个“namespace” 所在的容器中。这便是`docker exec`、两个容器共享network namespace 的原理。

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

[Docker容器里进程的 pid 是如何申请出来的？](https://mp.weixin.qq.com/s/LDu6s1eZw6_xEwfa6pMM-A) 宜细读

```c
struct pid_namespace{
    pidmap  // 是一个 bitmap，一个 bit 如果为 1，就表示当前序号的 pid 已经分配出去了
    int level   // 默认命名空间的 level 初始化是 0，如果有多个命名空间创建出来，它们之间会组成一棵树。level 表示树在第几层。根节点的 level 是 0。
}
```
在 create_pid_namespace 真正申请了新的 pid 命名空间，为它的 pidmap 申请了内存（在 create_pid_cachep 中申请的），也进行了初始化。另外还有一点比较重要的是新命名空间和旧命名空间通过 parent、level 等字段组成了一棵树。


```c
static struct task_struct *copy_process(...){
    ...
    //2.1 拷贝进程的命名空间 nsproxy
    retval = copy_namespaces(clone_flags, p);
    //2.2 申请 pid 
    pid = alloc_pid(p->nsproxy->pid_ns);
    //2.3 记录 pid 
    p->pid = pid_nr(pid);
    p->tgid = p->pid;
    attach_pid(p, PIDTYPE_PID, pid);
    ...
}
```
1. 支持namespace 之前，很多数据结构比如pidmap，都是进程全局共享的（或者说就是 全局变量），支持了namespace之后，都变成了 per-namespace的，每个task_struct 都有个ns_proxy 去引用它们。pidmap（或者说包裹它的pid_namespace）自己也组成了树状结构。
2. 创建进程的核心是在于 copy_process 函数。是先 copy_namespaces，把新的namespace struct 都创建好之后，再从这些数据结构里 走申请 资源（pid号等）逻辑

### mount namespace

[linux内核中根文件系统的初始化及init程序的运行](https://mp.weixin.qq.com/s/adoQPvWNR1sBwOytjtzoFg)我们在和vfs打交道时，为其提供的最主要的信息就是文件路径，而文件路径有两种格式，一种是绝对路径，即从根目录开始查找文件，一种是相对路径，即从当前目录查找文件。所以，vfs如果想要解析我们提供的路径，就必须还要知道一个信息，即我们当前的根目录和当前目录指向哪里。那这些信息存放在哪里了呢？我们和操作系统交互的方式是通过程序，或者更确切的说，是通过进程，所以当我们要求vfs帮我们解析一个路径时，其实是这个进程在要求vfs解析这个路径，所以vfs获取的根目录或当前目录，其实就是这个进程所属的根目录或当前目录。所以，存放根目录及当前目录的最好位置，就是在进程里。而且，在进程内，我们是可以通过 chdir 这个系统调用来修改当前目录的，也就是说，每个进程都有自己独有的当前目录，这就更说明，根目录和当前目录信息，只能存放在进程里。到这里有些同学可能会有疑问，当前目录存放在进程里比较好理解，但**根目录应该是所有进程共用的吧，为什么也要存放在进程里呢？**这是因为，不仅进程所属的当前目录是可以修改的，进程所属的根目录也是可以修改的，修改的方式就是通过 chroot 这个系统调用。根目录和当前目录，存放在进程的具体位置为：

![](/public/upload/kubernetes/linux_task_fs.jpg)

current指向的是当前进程，进程对应的结构体是struct task_struct，在这个结构体里，有一个fs字段，它又指向struct fs_struct结构体，而在struct fs_struct结构体里面，则存放了当前进程所属的根目录及当前目录，即root和pwd字段。那有了这两个字段，vfs就可以解析我们提供的文件路径，进而找到对应的文件数据了。

![](/public/upload/linux/pid_namespace.png)

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

Linux 中每个进程（线程）都是用 task_struct 来表示的。每个 task_struct 都要关联到一个 namespace 对象 nsproxy，而 nsproxy 又包含了 netns。对于网卡设备和 socket 来说，通过自己的成员来直接表明自己的归属。

![](/public/upload/container/netns_struct.png)

网络 namespace 的主要数据结构 struct net 的定义，每一个 netns 中都有一个 loopback_dev，最核心的数据结构是 `struct netns_ipv4 ipv4`，每个 net 下都包含了自己的路由表、iptable 以及内核参数配置等等。

![](/public/upload/container/netns_struct_detail.png)

## cgroups

[彻底搞懂容器技术的基石： cgroup](https://mp.weixin.qq.com/s/6Ts6-aZDr8qOdnaNUqwTFQ)

[一文带你搞懂 Docker 容器的核心基石 Cgroups](https://mp.weixin.qq.com/s/uvtuaXIDoCyy7-JO4qYY8Q) 未细读。在 Linux 里，一直以来就有对进程进行分组的概念和需求，并在分组的基础上对进程进行监控和资源控制管理等。

[彻底搞懂 Kubernetes 中的 Cgroup](https://mp.weixin.qq.com/s/bgoFj-aZo-RMh2hR5h0zrA) 几个概念
1. task（任务），系统中的进程
3. cgroup(控制组)，cgroups 中的资源控制都以 cgroup 为单位实现。cgroup 表示按某种资源控制标准划分而成的任务组，包含一个或多个子系统。一个任务可以加入某个 cgroup，也可以从某个 cgroup 迁移到另外一个 cgroup
3. subsystem(子系统)，cgroups 中的 subsystem 就是一个资源调度控制器（Resource Controller）。比如 CPU 子系统可以控制 CPU 时间分配，内存子系统可以限制 cgroup 内存使用量。每个 cgroup subsystem代表一种资源，**每一个子系统都需要与内核的其他模块配合来完成资源的控制**，比如对 cpu 资源的限制是通过进程调度模块根据 cpu 子系统的配置来完成的；对内存资源的限制则是内存模块根据 memory 子系统的配置来完成的，而对网络数据包的控制则需要 Traffic Control 子系统来配合完成。PS：**cgroup 更像是提供了一个配置内核的入口**，gpu 因为nvidia 一家独大，搞的有点黑盒，隔离效果就不如cpu做的好
4. hierarchy（层级树），hierarchy 由一系列 cgroup 以一个树状结构排列而成，每个 hierarchy 通过绑定对应的 subsystem 进行资源调度。hierarchy 中的 cgroup 节点可以包含零或多个子节点，子节点继承父节点的属性。整个系统可以有多个 hierarchy。

Linux 通过文件的方式，将 cgroups 的功能和配置暴露给用户，这得益于 Linux 的虚拟文件系统（VFS）。VFS 将具体文件系统的细节隐藏起来，给用户态提供一个统一的文件系统 API 接口，cgroups 和 VFS 之间的链接部分，称之为 cgroups 文件系统。比如挂在 cpu、cpuset、memory 三个子系统到 /cgroups/cpu_mem 目录下：`mount -t cgroup -o cpu,cpuset,memory cpu_mem /cgroups/cpu_mem`

runtime 有两种 cgroup 驱动：一种是 systemd，另外一种是 cgroupfs：
1. cgroupfs 比较好理解，比如说要限制内存是多少、要用 CPU share 为多少，其实直接把 pid 写入到对应 cgroup task 文件中，然后把对应需要限制的资源也写入相应的 memory cgroup 文件和 CPU 的 cgroup 文件就可以了；
2. 另外一个是 systemd 的 cgroup 驱动，这个驱动是因为 systemd 本身可以提供一个 cgroup 管理方式。所以如果用 systemd 做 cgroup 驱动的话，所有的写 cgroup 操作都必须通过 systemd 的接口来完成，不能手动更改 cgroup 的文件；

### V1 和 V2

![](/public/upload/container/cgroup_v1.jpeg)

Cgroup v1 的一个整体结构，每一个子系统都是独立的，资源的限制只能在子系统中发生。比如pid可以分别属于 memory Cgroup 和 blkio Cgroup。但是在 blkio Cgroup 对进程 pid 做磁盘 I/O 做限制的时候，blkio 子系统是不会去关心 pid 用了哪些内存，这些内存是不是属于 Page Cache，而这些 Page Cache 的页面在刷入磁盘的时候，产生的 I/O 也不会被计算到进程 pid 上面。**Cgroup v2 相比 Cgroup v1 做的最大的变动就是一个进程属于一个控制组，而每个控制组里可以定义自己需要的多个子系统**。Cgroup v2 对进程 pid 的磁盘 I/O 做限制的时候，就可以考虑到进程 pid 写入到 Page Cache 内存的页面了，这样 buffered I/O 的磁盘限速就实现了。

![](/public/upload/container/cgroup_v2.jpeg)

## cgroups 整体实现

[内核是如何给容器中的进程分配CPU资源的？](https://mp.weixin.qq.com/s/rUQLM8WfjMqa__Nvhjhmxw)

![](/public/upload/linux/cgroup_struct.jpg)

1. 包含哪些数据结构/内核对象
    1. 一个 cgroup 对象中可以指定对 cpu、cpuset、memory 等一种或多种资源的限制。每个 cgroup 都有一个 cgroup_subsys_state 类型的数组 subsys，其中的每一个元素代表的是一种资源控制，如 cpu、cpuset、memory 等等。
    2. 其实 cgroup_subsys_state 并不是真实的资源控制统计信息结构，对于 CPU 子系统真正的资源控制结构是 task_group，对于内存子系统控制统计信息结构是 mem_cgroup，其它子系统也类似。它是 cgroup_subsys_state 结构的扩展，类似父类和子类的概念，当 task_group 需要被当成cgroup_subsys_state 类型使用的时候，只需要强制类型转换就可以。
    3. 和 cgroup 和多个子系统关联定义类似，task_struct 中也定义了一个 cgroup_subsys_state 类型的数组 subsys，来表达这种一对多的关系。
    4. 无论是进程、还是 cgroup 对象，最后都能找到和其关联的具体的 cpu、内存等资源控制子系统的对象。
2. 通过创建目录来创建 cgroup 对象。在 `/sys/fs/cgroup/cpu,cpuacct` 创建一个目录 test，实际上内核是创建了 cgroup、task_group 等内核对象。
3. 在目录中设置 cpu 的限制情况。在 task_group 下有个核心的 cfs_bandwidth 对象，用户所设置的 cfs_quota_us 和 cfs_period_us 的值最后都存到它下面了。
4. 将进程添加到 cgroup 中进行资源管控。当在 cgroup 的 cgroup.proc 下添加进程 pid 时，实际上是将该进程加入到了这个新的 task_group 调度组了。

### cpu cgroups生效

```c
//file:kernel/sched/core.c
struct task_group root_task_group;  // 所有的 task_group 都是以 root_task_group 为根节点组成了一棵树。
//file:kernel/sched/sched.h
struct task_group {
 struct cgroup_subsys_state css;
 ...

 // task_group 树结构
 struct task_group   *parent;
 struct list_head    siblings;
 struct list_head    children;

 //task_group 持有的 N 个调度实体(N = CPU 核数)
 struct sched_entity **se;
 //task_group 自己的 N 个公平调度队列(N = CPU 核数)
 struct cfs_rq       **cfs_rq;

 //公平调度带宽限制
 struct cfs_bandwidth    cfs_bandwidth;
 ...
}
```

假如当前系统有两个逻辑核，那么一个 task_group 树和 cfs_rq 的简单示意图大概是下面这个样子。

![](/public/upload/linux/linux_cgroup_schedule.jpg)

Linux 中的进程调度是一个层级的结构，进程创建后的一件重要的事情，就是调用sched_class的enqueue_task方法，将这个进程放进某个CPU的队列上来。默认有一个根group，也就是，创建普通进程后（没有配置cgroup）加入percpu.rq 实质是加入到了一个默认的task_group中。对于容器来讲，**宿主机中进行进程调度的时候，先调度到的实际上不是容器中的具体某个进程，而是一个 task_group**。然后接下来再进入容器 task_group 的调度队列 cfs_rq 中进行调度，才能最终确定具体的进程 pid。

系统会定期在每个 cpu 核上发起 timer interrupt，在时钟中断中处理的事情包括 cpu 利用率的统计，以及周期性的进程调度等。当cgroup cpu 子系统创建完成后，内核的 period_timer 会根据 task_group->cfs_bandwidth 下用户设置的 period 定时给可执行时间 runtime 上加上 quota 这么多的时间（相当于按月发工资），以供 task_group 下的进程执行（消费）的时候使用。

在完全公平器调度的时候，每次 pick_next_task_fair 时会做两件事情
1. 将从 cpu 上拿下来的进程所在的运行队列进行执行时间的更新与申请。会将 cfs_rq 的 runtime_remaining 减去已经执行了的时间。如果减为了负数，则从 cfs_rq 所在的 task_group 下的 cfs_bandwidth 去申请一些。
2. 判断 cfs_rq 上是否申请到了可执行时间，如果没有申请到，需要将这个队列上的所有进程都从完全公平调度器的红黑树上取下。这样再次调度的时候，这些进程就不会被调度了。
当 period_timer 再次给 task_group 分配时间的时候，或者是自己有申请时间没用完回收后触发 slack_timer 的时候，被限制调度的进程会被解除调度限制，重新正常参与运行。这里要注意的是，一般 period_timer 分配时间的周期都是 100 ms 左右。假如说你的进程前 50 ms 就把 cpu 给用光了，那你收到的请求可能在后面的 50 ms 都没有办法处理，对请求处理耗时会有影响。这也是为啥在关注 CPU 性能的时候要关注对容器 throttle 次数和时间的原因了。

### 内存 cgroups生效

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
### 查看配置

从右向左 ==> 和docker run放在一起看 

[cgroup 原理与实现](https://mp.weixin.qq.com/s/yXJxTR_sPdEMt56cf7JPhQ) 写的很清晰了

![](/public/upload/linux/linux_cgroup_docker.png)

从左向右 ==> 从 task 结构开始找到 cgroup 结构

[Docker 背后的内核知识——cgroups 资源限制](https://www.infoq.cn/article/docker-kernel-knowledge-cgroups-resource-isolation/)

在图中使用的回环箭头，均表示可以通过该字段找到所有同类结构

![](/public/upload/linux/linux_task_cgroup.png)

从右向左 ==> 查看一个cgroup 有哪些task

![]()(/public/upload/linux/linux_task_cgroup.png)

为什么要使用cg_cgroup_link结构体呢？因为 task 与 cgroup 之间是多对多的关系。熟悉数据库的读者很容易理解，在数据库中，如果两张表是多对多的关系，那么如果不加入第三张关系表，就必须为一个字段的不同添加许多行记录，导致大量冗余。通过从主表和副表各拿一个主键新建一张关系表，可以提高数据查询的灵活性和效率。









