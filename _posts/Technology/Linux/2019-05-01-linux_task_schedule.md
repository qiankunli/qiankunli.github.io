---

layout: post
title: Linux进程调度
category: 技术
tags: Linux
keywords: linux 进程调度

---

## 简介

* TOC
{:toc}

作为初学者，了解数据结构之间的组织关系，这远比了解一个数据结构所有字段的作用和细节重要得多。彭东《操作系统实战》中自己实现的os的进程调度部分提供了一个很好的认知思路（程序=数据结构+算法）
1. 进程数据结构 thread_t，管理结构thrdlst_t/schdata_t/schedclass_t
2. 初始化：初始化schedclass_t 变量（全局的）。至于初始化thread_t 内部包括分配进程的内核栈与应用程序栈，并对进程的内核栈进行初始化，最后将进程加入调度系统
3. 实现进程调度器，入口函数krlschedul，获取当前运行的进程krlsched_retn_currthread，选择进程函数krlsched_select_thread，最后是进程切换函数，进程切换过程：我们把当前进程的通用寄存器保存到当前进程的内核栈中；然后，保存 CPU 的 RSP 寄存器到当前进程的机器上下文结构中，并且读取保存在下一个进程机器上下文结构中的 RSP 的值，把它存到 CPU 的 RSP 寄存器中；接着，调用一个函数切换 MMU 页表；最后，从下一个进程的内核栈中恢复下一个进程的通用寄存器。
3. 进程等待函数 krlsched_wait，设置进程状态为等待状态，让进程从调度系统数据结构中脱离，最后让进程加入到 kwlst_t 等待结构中。
4. 进程唤醒函数 krlsched_up
5. 提供一个空转进程

[万字详解Linux内核调度器及其妙用](https://mp.weixin.qq.com/s/gkZ0kve8wOrV5a8Q2YeYPQ)可以试着切换到硬件CPU的视角，这样应用程序和内核，就都变成客体研究对象了，很多逻辑也就清晰了。

## 数据结构

我们知道应用程序由源代码编译而成，**没有运行之前它是一个文件**，直到它被装入内存中运行、操作数据执行相应的计算，完成相应的输入输出。在计算机中，CPU、内存、网络、各种输入、输出设备甚至文件数据都可以看成是资源，操作系统就是这些资源的管理者。应用程序要想使用这些“资本”，就要向操作系统申请。比方说，应用程序占用了多少内存，使用了多少网络链接和通信端口，打开多少文件等，**这些使用的资源通通要记录在案**。记录在哪里比较合适呢？当然是代表一个应用程序的活动实体——进程之中最为稳妥。

![](/public/upload/linux/task_struct_part.png)

结合上图我们发现，进程可以看作操作系统用来管理应用程序资源的容器，通过进程就能控制和保护应用程序。操作系统各个模块的运作原理，就是不断和这些数据结构（公共资源比如内存等一般是都放在一个全局的list、dict里，附带一些操作函数，进程task_struct 负责引用其中的数据一部分数据单元）打交道而已。

### 进程部分

![](/public/upload/linux/linux_task_struct_data.png)

一个进程的运行竟然要保存这么多信息，这些信息都可以通过命令行取出来。fork 进程时，核心是一个copy_process函数，它以复制父进程的方式来生成一个新的task_struct，然后调用wake_up_new_task 将新进程添加到就绪队列中，等待调度器调度执行。 

```c
long _do_fork(unsigned long clone_flags,unsigned long stack_start,unsigned long stack_size,int __user *parent_tidptr,int __user *child_tidptr,unsigned long tls){
    struct task_struct *p;
    int trace = 0;
    long nr;
    ......
    // 复制结构
    p = copy_process(clone_flags, stack_start, stack_size,child_tidptr, NULL, trace, tls, NUMA_NO_NODE);
    ......
    if (!IS_ERR(p)) {
        struct pid *pid;
        pid = get_task_pid(p, PIDTYPE_PID);
        nr = pid_vnr(pid);
        if (clone_flags & CLONE_PARENT_SETTID)
            put_user(nr, parent_tidptr);
        ......
        // 唤醒新进程
        wake_up_new_task(p);
        ......
        put_pid(pid);
    }
    ...
} 
static struct task_struct *copy_process(...){
    // 复制进程task_struct结构体
    struct task_struct *p;
    p = dup_task_struct(current, ...);
    // 复制files_struct
    retval = copy_files(clone_flags,p)
    // 复制fs_struct
    retval = copy_fs(clone_flags,p)
    // 复制mm_struct
    retval = copy_mm(clone_flags,p)
    // 复制进程的命名空间 nsproxy
    retval = copy_namespace(clone_flags,p)
    // 申请pid并设置进程号
    pid = alloc_pid(p->nsproxy->pid_ns_for_children,...);
    p->pid = pid_nr(pid);
    if (clone_flags & CLONE_THREAD) {
        p->tgid = current->tgid;
    }else{
        p->tgid = p->pid;
    }
    ...
}
```

### 进程切换

在 x86 体系结构中，提供了一种以硬件的方式进行进程切换的模式，对于每个进程，x86 希望在内存里面维护一个 TSS（Task State Segment，任务状态段）结构。这里面有所有的寄存器。另外，还有一个特殊的寄存器 TR（Task Register，任务寄存器），指向某个进程的 TSS。
1. 更改 TR 的值，将会触发硬件保存 CPU 所有寄存器的值到当前进程的 TSS 中
2. 然后从新进程的 TSS 中读出所有寄存器值，加载到 CPU 对应的寄存器中。

但是这样有个缺点。我们做进程切换的时候，没必要每个寄存器都切换，这样每个进程一个 TSS，就需要全量保存，全量切换，动作太大了。于是，Linux 操作系统想了一个办法。还记得在系统初始化的时候，会调用 cpu_init 吗？这里面会给每一个CPU 关联一个 TSS，然后将 TR 指向这个 TSS，然后在操作系统的运行过程中，TR 就不切换了，永远指向这个TSS

在 Linux 中，真的参与进程切换的寄存器很少，主要的就是栈顶寄存器。所谓的进程切换，就是将某个进程的 thread_struct里面的寄存器的值，写入到 CPU 的 TR 指向的 tss_struct，对于 CPU 来讲，这就算是完成了切换。

![](/public/upload/linux/cpu_rq.png)

[CPU明明8个核，网卡为啥拼命折腾一号核？](https://mp.weixin.qq.com/s/LFv3VYC1DIKtcEjjTfYoTQ)

## 进程创建

shell 实现的功能有别于其它应用，它的功能是接受用户输入，然后查找用户输入的应用程序，最后加载运行这个应用程序。shell 应用首先调用了 fork，通过写时复制创建了一个自己的副本，我们暂且称为 shell 子应用。然后，shell 子应用中调用了 execl，该函数会通过文件内容重载应用的地址空间，它会读取应用程序文件中的代码段、数据段、bss 段和调用进程的栈，覆盖掉原有的应用程序地址空间中的对应部分。而且 execl 函数执行成功后不会返回，而是会调整 CPU 的 PC 寄存器，从而执行新的 init 段和 text 段。从此，一个新的应用就产生并开始运行了。

[万字详解Linux内核调度器及其妙用](https://mp.weixin.qq.com/s/gkZ0kve8wOrV5a8Q2YeYPQ) 整个Linux系统的第一个用户态进程就是这样运行起来的。Linux系统启动的时候，先初始化的肯定是内核，当内核初始化结束了，会创建第一个用户态进程，1号进程。创建的方式是在内核态运行do_execve，来运行"/sbin/init"，"/etc/init"，"/bin/init"，"/bin/sh"中的一个，不同的Linux版本不同。

写过Linux程序的我们都知道，execve是一个系统调用，它的作用是运行一个执行文件。加一个do_的往往是内核系统调用的实现。在do_execve中，会有一步是设置struct pt_regs，主要设置的是ip和sp，指向第一个进程的起始位置，这样调用iret就可以从系统调用中返回。这个时候会从pt_regs恢复寄存器。指令指针寄存器IP恢复了，指向用户态下一个要执行的语句。函数栈指针SP也被恢复了，指向用户态函数栈的栈顶。所以，下一条指令，就从用户态开始运行了。

接下来所有的用户进程都是这个1号进程的徒子徒孙了。如果要创建新进程，是某个用户态进程调用fork，fork是系统调用会调用到内核，在内核中子进程会复制父进程的几乎一切，包括task_struct，内存空间等。这里注意的是，fork作为一个系统调用，是将用户态的当前运行状态放在pt_regs里面了，IP和SP指向的就是fork这个函数，然后就进内核了。

子进程创建完了，如果像前面讲过的check_preempt_curr里面成功抢占了父进程，则父进程会被标记TIF_NEED_RESCHED，则在fork返回的时候，会检查这个标记，会将CPU主动让给子进程，然后让子进程从内核返回用户态，因为子进程是完全复制的父进程，因而返回用户态的时候，仍然在fork的位置，当然父进程将来返回的时候，也会在fork的位置，只不过返回值不一样，等于0说明是子进程。

![](/public/upload/linux/task_fork.jpeg)

||创建进程|创建线程|
|---|---|---|
|系统调用|fork|clone|
|copy_process逻辑|会将五大结构 files_struct、fs_struct、sighand_struct、signal_struct、mm_struct 都复制一遍<br>从此父进程和子进程各用各的数据结构|五大结构仅仅是引用计数加一<br>也即线程共享进程的数据结构|
||完全由内核实现|由内核态和用户态合作完成<br>相当一部分逻辑由glibc库函数pthread_create来做|
|数据结构||内核态struct task_struct <br>用户态 struct pthread|


[聊聊Linux中线程和进程的联系与区别](https://mp.weixin.qq.com/s/--S94B3RswMdBKBh6uxt0w)**Linux进程和线程的相同点要远远大于不同点，本质上是同一个东西**，都是一个 task_struct。每一个 task_struct 都需要被唯一的标识，它的 pid 就是唯一标识号。对于进程来说，这个 pid 就是我们平时常说的进程 pid。对于线程来说，我们假如一个进程下创建了多个线程出来。那么每个线程的 pid 都是不同的。但是我们一般又需要记录线程是属于哪个进程的，通过 tgid 字段来表示自己所归属的进程 ID。
1. 进程创建 fork ==> fork ==> do_fork ==> copy_process
2. 线程创建 pthread_create ==> do_clone ==> clone ==> do_fork ==> copy_process
可见和创建进程时使用的 fork 系统调用相比，创建线程的 clone 系统调用几乎和 fork 差不多，也一样使用的是内核里的 do_fork 函数，最后走到 copy_process 来完整创建。不过创建过程的区别是二者在调用 do_fork 时传入的 clone_flags 里的标记不一样！。
1. 创建进程时的 flag：仅有一个 SIGCHLD
2. 创建线程时的 flag：包括 CLONE_VM(新 task 和父进程共享地址空间)、CLONE_FS(新 task 和父进程共享文件系统信息)、CLONE_FILES(新 task 和父进程共享文件描述符表)、CLONE_SIGNAL、CLONE_SETTLS、CLONE_PARENT_SETTID、CLONE_CHILD_CLEARTID、CLONE_SYSVSEM。PS：带了就会复用、共享对应xx_struct。代码段、数据段、堆内存共享，各个线程的栈区独立（不复用）。进程在创建的时候，启动调用exec加载可执行文件的过程中，os 会为其分配一个栈内存供进程运行使用。**线程没有办法使用os 默认给进程分配的栈内存**，linux中glibc库的做法是自己(用户态)申请内存来当线程栈用。

![](/public/upload/linux/process_vs_thread.png)

对于线程来讲，其地址空间 mm_struct、目录信息 fs_struct、打开文件列表 files_struct 都是和创建它的任务共享的。但是对于进程来讲，地址空间 mm_struct、挂载点 fs_struct、打开文件列表 files_struct 都要是独立拥有的，都需要去申请内存并初始化它们。

## 进程调度

两个核心问题
1. cpu如何选择下面让哪一个任务运行
2. 允许选中的进程运行多长时间

**进程调度第一定律**：所有进程的调度最终是通过正在运行的进程调用`__schedule` 函数实现

![](/public/upload/linux/process_schedule.png)

### 基于虚拟运行时间的调度

到linux2.4（2001年），整个系统有一个调度队列，当有新任务到达时，先设置一个静态优先级，调度器选择进程执行的时候，选择的办法是遍历整个任务队列，从中挑选优先级最高的。但是优先级是在静态优先级的基础上动态变化的，如果获得了cpu，那动态优先级就会变低。如果一直未获得cpu，动态优先级就会变高。这样既照顾了对实时性要求高的高优先级进程，也避免了把低优先级的进程饿死。随后cpu开始朝多核发展了，所有cpu 访问一个任务队列，锁竞争的开销越来越高，随着服务器上跑的进程越来越多，O(n)遍历也显得有一点低效。在linux2.5中，为每个cpu 逻辑核都准备一个runqueue，减少了锁的开销。采用多优先级队列，每一个优先级都有一个链表，在查找的时候引入bitmap（通过一个bit 来表示相对应的优先级上是否有任务存在，顺带定位指定优先级上对应的任务链表） 辅助结构实现O(1)查找（PS：跟go gmp的演进一致）。但linux2.5 存在一点瑕疵，就是按优先级固定计算时间片，优先级高的进程被分配了更多的时间片，优先级低的进程被分配的时间片较少，一个进程的运行时间片长度是10~200ms之间，最大的问题是调度延迟不可控。背景是一个**调度周期**（volcano 也有调度周期的概念，cfs代码上不涉及调度周期）就是当前周期内需要执行的所有任务时间片之和，新来一个任务或本轮时间片用完的话，在下一个周期才能调度到。linux 2.6.23 采用cfs 作为用户进程的调度算法，在一个调度周期内所有进程的运行时间片大小相等（时间周期T/进程数量N），通过引入一个虚拟运行时间vruntime 的概念来极大维护了调度算法的简洁性。如果等待运行的进程数量不多，那就使用一个固定的调度周期（sysctl_sched_latency默认24ms），如果等待运行的进程数量过多，有N进程等待，那就N*sysctl_sched_min_granularity(默认3ms)为一个调度周期。保证所有进程都能快速得到一小段处理时间，进程的调度延迟最大不会超过一个调度周期。

||linux2.5|linux 2.6.23|
|---|---|---|
|cpu如何选择下面让哪一个任务运行|动态优先级|vruntime 最小|
|允许选中的进程运行多长时间|跟静态优先级挂钩，10~200ms|在一个调度周期内，时间周期T/进程数量N|

```c
// file: kernel/sched/sched.h
struct rq {
    struct rt_rq rt;
    struct cfs_rq cfs;
    struct dl_rq dl;
}
struct task_struct{
    ...
    unsigned int policy;    // 调度策略
    ...
    int prio, static_prio, normal_prio;
    unsigned int rt_priority;
    ...
    const struct sched_class *sched_class; // 调度策略的执行逻辑，实现了调取器类中要求的添加任务队列、删除任务队列、从队列中选择进程等方法。
}
```

CPU 会提供一个时钟，过一段时间就触发一个时钟中断Tick，task_struct.sched_entity里面有一个重要的变量vruntime，来记录一个进程的虚拟运行时间。如果一个进程在运行，随着时间的增长，也就是一个个 tick 的到来，进程的 vruntime 将不断增大。没有得到执行的进程 vruntime 不变，**最后尽量保证所有进程的vruntime相等**。为什么是 虚拟运行时间呢？`虚拟运行时间 vruntime += 实际运行时间 delta_exec * NICE_0_LOAD/ 权重`。就好比可以把你安排进“尖子班”变相走后门，但高考都是按分数（vruntime）统一考核的。完全公平调度器维持的是所有sched_entity 的vruntime的公平，但是vruntime会根据weight 来进行缩放。vruntime的计算是在calc_delta_fair函数中实现的。

```c
/*
 * Update the current task's runtime statistics.
 */
static void update_curr(struct cfs_rq *cfs_rq){
    struct sched_entity *curr = cfs_rq->curr;
    u64 now = rq_clock_task(rq_of(cfs_rq));
    u64 delta_exec;
    ......
    delta_exec = now - curr->exec_start;
    ......
    curr->exec_start = now;
    ......
    curr->sum_exec_runtime += delta_exec;
    ......
    curr->vruntime += calc_delta_fair(delta_exec, curr);
    update_min_vruntime(cfs_rq);
    ......
}

/*
 * delta /= w
 */
static inline u64 calc_delta_fair(u64 delta, struct sched_entity *se){
    if (unlikely(se->load.weight != NICE_0_LOAD))
        /* delta_exec * weight / lw.weight */
        delta = __calc_delta(delta, NICE_0_LOAD, &se->load);
    return delta;
}
```

NICE_0_LOAD宏对应的是1024，如果权重是1024，那么vruntime 就正好等于实际运行时间，否则会进入__calc_delta 来根据权重和实际运行时间折算一个vruntime增量。如果weight 较高，则同样实际运行时间算出来的vruntime 就会偏小，就会在调度中获取更多的cpu，cfs 就是这样实现了cpu资源按权重分配。

调度需要一个数据结构来对 vruntime 进行排序，因为任何一个策略做调度的时候，都是要区分谁先运行谁后运行。这个能够排序的数据结构不但需要查询的时候，能够快速找到最小的，更新的时候也需要能够快速的调整排序，毕竟每一个tick vruntime都会增长。能够平衡查询和更新速度的是树，在这里使用的是红黑树，key=vruntime。红黑树的一个node由sched_entity 表示（数据结构中很少有一个Tree 存在，都是根节点`Node* root`就表示tree了），可能是一个真正的进程，也可能是一个进程组。

```c
// file: kernel/sched/sched.h
struct cfs_rq {
    u64 min_vruntime;   // 当前队列中所有进程vruntime中的最小值
    struct rb_root_cached tasks_timeline;   // 保存就绪任务的红黑树（根节点）
}
struct task_struct{
    ...
    const struct sched_class *sched_class; // 调度策略的执行逻辑
    struct sched_entity se;     // 对应完全公平算法调度
    struct sched_rt_entity rt;  // 对应实时调度
    struct sched_dl_entity dl;  // 对应deadline 调度
    ...
}
struct sched_entity {
    struct load_weight load; // 当前进程权重，对应cgroup cpu.shares
    struct rb_node run_node; // 指向自己在红黑树上的节点位置
    u64 exec_start;         // 进程开始运行的时间
    u64 sum_exec_runtime;   // 总的运行时间
    u64 vruntime;           // 进程虚拟运行时间
}
```
fork 创建进程时，copy_process对新进程的task_struct 进行各种初始化，其中会调用sched_fork 来完成调度相关的初始化。
```c
static struct task_struct *copy_process(...){
    ...
    retval = sched_fork(clone_flags, p);
}
// file:kernel/sched/core.c
int sched_fork(unsigned long clone_flags, struct task_struct *p){
    __schedd_fork(clone_flags, p);
    p->__state = TASK_NEW;
    if (rt_prio(p->prio))
        p->sched_class = &rt_sched_class;
    else
        p->sched_class = &fair_sched_class; // fair_sched_class 是一个全局对象
}
// file:kernel/sched/core.c
static void __sched_fork(struct task_struct *p){
    p ->on_rq = 0;
    ...
    p->se.nr_migrations = 0;
    p->se.vruntime = 0; // 新进程是0对老进程不公平，在新进程真正被加入运行队列时，会将其值设置为cfs_rq->min_vruntime
}
// file:kernel/sched/core.c
void wake_up_new_task(struct task_struct *p){
    // 为进程选择一个合适的cpu
    cpu = select_task_rq(p, task_cpu(p)
    // 为进程指定运行队列
    __set_task_cpu(p,cpu, WF_FORK));
    // 将进程添加到运行队列红黑树
    rq = __task_rq_lock(p);
    activate_task(rq, p, 0);
}
```
select_task_rq ==> task_struct->sched_class->select_task_rq在缓存性能和空闲核两个点做权衡，同等条件会尽量优先考虑缓存命中率，选择同L1/L2的核，其次会选择同一个物理cpu上的（共享L3），最坏情况下去选择另一个（负载最小的）物理cpu上的核，称之为漂移。现在互联网公司都流行在离线混部，宿主机的cpu利用率被打到很高的水平，比如70%，进程在不同的核上运行概率增加，缓存中的数据都是“凉的”，穿透到内存的访问次数增加，进程的运行性能就会下降很多。

**每个逻辑核都有自己的 struct rq 结构**，其用于描述在此 CPU 上所运行的所有进程，其包括一个实时进程队列rt_rq 和一个 CFS 运行队列 cfs_rq。在调度时，调度器首先会先去实时进程队列找是否有实时进程需要运行，如果没有才会去 CFS 运行队列找是否有进行需要运行。**这样保证了实时任务的优先级永远大于普通任务**。PS： [离线调度算法(BT)](https://github.com/Tencent/TencentOS-kernel) 为了支持在混部，支持了bt_rq。


```c
// Pick up the highest-prio task:
static inline struct task_struct *pick_next_task(struct rq *rq, struct task_struct *prev, struct rq_flags *rf){
    const struct sched_class *class;
    struct task_struct *p;
    ......
    for_each_class(class) {
        p = class->pick_next_task(rq, prev, rf);
        if (p) {
            if (unlikely(p == RETRY_TASK))
                goto again;
            return p;
        }
    }
}
```

CFS 的队列是一棵红黑树（所以叫“队列”很误导人），树的每一个节点都是一个 sched_entity（说白了每个节点是一个进/线程），每个 sched_entity 都属于一个 task_struct，task_struct 里面有指针指向这个进程属于哪个调度类。如何获取下一个待执行任务的呢？**其实就是从当前任务队列的红黑树节点将运行虚拟时间最小的节点（最左侧的节点）选出来而已**。

<div class="class=width:100%;height:auto;">
    <img src="/public/upload/linux/process_schedule_impl.jpeg"/>
</div>

基于进程调度第一定律，上图就是一个很完整的循环，**cpu的执行一直是方法调方法**（process1.func1 ==> process1.schedule ==> process2.func2 ==> process2.schedule ==> process3.func3），只不过是跨了进程

### 调度类

如果将task_struct 视为一个对象，在很多场景下 主动调用`schedule()` 让出cpu，那么如何选取下一个task 就是其应该具备的能力，sched_class 作为其成员就顺理成章了。

```c
DEFINE_SCHED_CLASS(fair) = {
    ...
    .select_task_rq = cfs_select_task_rq, // 选择运行队列
}
```

![](/public/upload/linux/schedule_class.png)

sched_class结构体类似面向对象中的基类，通过函数指针类型的成员指向不同的函数，实现了多态。

### 调度时机

调度时机包括：定时调度节拍和其它进程阻塞时主动让出两种。

调度节拍 的入口是scheduler_tick ==> curr->sched_class->task_tick：时钟节拍最终会调用调度类task_tick方法完成调度相关的工作，会在这里判断是否需要调度下一个任务来抢占当前cpu核。也会触发多核之间任务队列的负载均衡，保证不让忙的核忙死，闲的核闲死。在调度节拍中会定时将每个进程所执行过的时间都换算成vruntime，并累计起来，也会定时判断当前进程是否已经执行了足够长的时间，如果是的话，需要再选择另一个vruntime较小的任务来运行。

主动调度，就是进程运行到一半，因为等待 I/O 等操作而主动调用 schedule() 函数让出 CPU。在 Linux 内核中有数百处**调用点**，它们会把进程设置为 D 状态（TASK_UNINTERRUPTIBLE），主要集中在 disk I/O 的访问和信号量（Semaphore）锁的访问上，因此 D 状态的进程在 Linux 里是很常见的。

写入块设备的一个典型场景。写入需要一段时间，这段时间用不上CPU

```c
static void btrfs_wait_for_no_snapshoting_writes(struct btrfs_root *root){
    ......
    do {
        prepare_to_wait(&root->subv_writers->wait, &wait,
                TASK_UNINTERRUPTIBLE);
        writers = percpu_counter_sum(&root->subv_writers->counter);
        if (writers)
            schedule();
        finish_wait(&root->subv_writers->wait, &wait);
    } while (writers);
}
```
从 Tap 网络设备等待一个读取
```c
static ssize_t tap_do_read(struct tap_queue *q,
            struct iov_iter *to,
            int noblock, struct sk_buff *skb){
    ......
    while (1) {
        if (!noblock)
            prepare_to_wait(sk_sleep(&q->sk), &wait,
                    TASK_INTERRUPTIBLE);
    ......
        /* Nothing to read, let's sleep */
        schedule();
    }
    ......
}
```

**这段跟golang协程的读写过程 是一样一样的**，内核机制上层化（内存管理、线程调度放到语言层/框架层来解决）是一个普遍趋势。

### 抢占式调度

所谓的抢占调度，就是A进程运行的时间太长了，会被其他进程抢占。还有一种情况是，有一个进程B原来等待某个I/O事件，等待到了被唤醒，发现比当前正在CPU上运行的进行优先级高，于是进行抢占。

vruntime就是一个数据，如果没有任何机制对它进行更新，就会导致一个进程永远运行下去，因为那个进程的虚拟时间没有更新，虚拟时间永远最小，这当然不行。在计算机里面有一个时钟，会过一段时间触发一次时钟中断，时钟中断处理函数会调用 scheduler_tick()，代码如下

```c
void scheduler_tick(void){
    // 取出当前的cpu及其任务队列
    int cpu = smp_processor_id();
    struct rq *rq = cpu_rq(cpu);
    struct task_struct *curr = rq->curr;
    ......
    curr->sched_class->task_tick(rq, curr, 0);
    cpu_load_update_active(rq);
    calc_global_load_tick(rq);
    ......
}
```

对于普通进程 scheduler_tick ==> fair_sched_class.task_tick_fair ==> entity_tick ==> update_curr 更新当前进程的 vruntime ==> check_preempt_tick 检查是否是时候被抢占了

```c
static void check_preempt_tick(struct cfs_rq *cfs_rq, struct sched_entity *curr){
    unsigned long ideal_runtime, delta_exec;
    struct sched_entity *se;
    s64 delta;
    //计算当前进程在本次调度中分配的运行时间
    ideal_runtime = sched_slice(cfs_rq, curr);
    //当前进程已经运行的实际时间
    delta_exec = curr->sum_exec_runtime - curr->prev_sum_exec_runtime;
    //如果实际运行时间已经超过分配给进程的运行时间，就需要抢占当前进程。设置进程的TIF_NEED_RESCHED抢占标志。
    if (delta_exec > ideal_runtime) {
        resched_curr(rq_of(cfs_rq));
        return;
    }
    //因此如果进程运行时间小于最小调度粒度时间，不应该抢占
    if (delta_exec < sysctl_sched_min_granularity)
        return;
    //从红黑树中找到虚拟时间最小的调度实体
    se = __pick_first_entity(cfs_rq);
    delta = curr->vruntime - se->vruntime;
    //如果当前进程的虚拟时间仍然比红黑树中最左边调度实体虚拟时间小，也不应该发生调度
    if (delta < 0)
        return;
}
```

当发现当前进程应该被抢占，不能直接把它踢下来，而是把它标记为应该被抢占。为什么呢？因为进程调度第一定律呀，一定要等待正在运行的进程调用 __schedule 才行

1. 用户态的抢占时机
    1. 从系统调用中返回的那个时刻
    2. 从中断中返回的那个时刻
2. 内核态的抢占时机
    1. 一般发生在 preempt_enable()。在内核态的执行中，有的操作是不能被中断的，所以在进行这些操作之前，总是先调用 preempt_disable() 关闭抢占，当再次打开的时候，就是一次内核态代码被抢占的机会。
    2. 在内核态也会遇到中断的情况，当中断返回的时候，返回的仍然是内核态。这个时候也是一个执行抢占的时机


### Schedule

[调度系统设计精要](https://mp.weixin.qq.com/s/R3BZpYJrBPBI0DwbJYB0YA)CFS 的调度过程还是由 schedule 函数完成的，该函数的执行过程可以分成以下几个步骤：

1. 关闭当前 CPU 的抢占功能；
2. 如果当前 CPU 的运行队列中不存在任务，调用 idle_balance 从其他 CPU 的运行队列中取一部分执行；
3. 调用 pick_next_task 选择红黑树中优先级最高的任务；
4. 调用 context_switch 切换运行的上下文，包括寄存器的状态和堆栈；
5. 重新开启当前 CPU 的抢占功能；

```c
// schedule 方法入口
asmlinkage __visible void __sched schedule(void){
    struct task_struct *tsk = current;
    sched_submit_work(tsk);
    do {
        preempt_disable();
        __schedule(false);
        sched_preempt_enable_no_resched();
    } while (need_resched());
}
// 主要逻辑是在 __schedule 函数中实现的
static void __sched notrace __schedule(bool preempt){
    struct task_struct *prev, *next;
    unsigned long *switch_count;
    struct rq_flags rf;
    struct rq *rq;
    int cpu;
    // 在当前cpu 上取出任务队列rq（其实是红黑树）
    cpu = smp_processor_id();
    rq = cpu_rq(cpu);   
    prev = rq->curr;
    // 获取下一个待执行任务，其实就是从当前rq 的红黑树节点中选择vruntime最小的节点
    next = pick_next_task(rq, prev, &rf);
    clear_tsk_need_resched(prev);
    clear_preempt_need_resched();
    // 当选出的继任者和前任不同，就要进行上下文切换，继任者进程正式进入运行
    rq = context_switch(rq, prev, next, &rf);
```

上下文切换主要干两件事情，一是切换进程空间，也即虚拟内存；二是切换寄存器和栈等。

```c
// context_switch - switch to the new MM and the new thread's register state.
static __always_inline struct rq *context_switch(struct rq *rq, struct task_struct *prev,struct task_struct *next, struct rq_flags *rf){
    struct mm_struct *mm, *oldmm;
    ......
    // 切换虚拟地址空间
    mm = next->mm;
    oldmm = prev->active_mm;
    ......
    switch_mm_irqs_off(oldmm, mm, next);
    ......
    /* Here we just switch the register state and the stack. */
    // 切换栈和寄存器
    switch_to(prev, next, prev);
    barrier();
    return finish_task_switch(prev);
}
```
假如进程AB switch_to：在task_struct里面，还有一个成员变量struct thread_struct thread，里面保存了进程切换时的寄存器的值。在switch_to里面，A进程的SP就保存进A进程的task_struct的thread结构中，而B进程的SP就从他的task_struct的thread结构中取出来，加载到CPU的SP栈顶寄存器里面。在switch_to里面，还将这个CPU的current_task指向B进程的task_struct。当switch_to干完了上面的事情，返回的时候，IP就指向了下一行代码finish_task_switch。下面请问，这里的finish_task_switch是B进程的finish_task_switch，还是A进程的finish_task_switch呢？其实是B进程的finish_task_switch，因为当年B进程就是运行到switch_to被切走的，所以现在回来，运行的是B进程的finish_task_switch。其实从CPU的角度来看，这个时候还不区分A还是B的finish_task_switch，在CPU眼里，这就是内核的一行代码。但是代码再往下运行，就能区分出来了，因为finish_task_switch结束，要从schedule函数返回了，那应该是返回到A还是B对应的内核态代码呢？根据函数调用栈的原理，栈顶指针指向哪行，就返回哪行，别忘了前面SP已经切换成为B进程的了，已经指向B的内核栈了，所以返回的是B内核态代码。

虽然指令指针寄存器IP还是一行一行代码的执行下去，但由于所有的调度都会走schedule函数，IP没变，但是SP变了，进程切换就完成了。

## 其它

cpu 就是不断从pc 找活儿（指令）干，加上scheduler + task list之后就是不断从task list找个活儿（task_struct）干，跟java executor 不断从队列找活儿（runnable）干是一样的。又比如go中，用户逻辑包在goroutine中，goroutine 放在P中，M 不断从P 中取出G 来干活儿。 就好像网络包，一层套一层，符合一定的格式才可以收发和识别。

调度周期的另一块素材：什么是调度延迟？其实就是保证每一个可运行的进程，都至少运行一次的时间间隔。假设系统中有 3 个可运行进程，每个进程都运行 10ms，那么调度延迟就是 30ms；随着进程的增加，每个进程分配的时间在减少，进程调度次数会增加，调度器占用的时间就会增加。因此，CFS 调度器的调度延迟时间的设定并不是固定的。当运行进程少于 8 个的时候，调度延迟是固定的 6ms 不变。当运行进程个数超过 8 个时，就要保证每个进程至少运行一段时间，才被调度。这个“至少一段时间”叫作最小调度粒度时间。在 CFS 默认设置中，最小调度粒度时间是 0.75ms

### cpu 如何访问task_struct

**进程及调度数据结构的组织**：在 task_struct 结构中，会包含至少一个 sched_entity 结构的变量。它其实是 Linux 进程调度系统的一部分，被嵌入到了 Linux 进程数据结构中，与调度器进行关联，能间接地访问进程。我们只要通过 sched_entity 结构变量的地址，减去它在 task_struct 结构中的偏移（由编译器自动计算），就能获取到 task_struct 结构的地址。这样就能达到通过 sched_entity 结构，访问 task_struct 结构的目的了。sched_entity 结构是通过红黑树组织起来的，红黑树的根在 cfs_rq 结构中，cfs_rq 结构又被包含在 rq 结构，每个 CPU 对应一个 rq 结构。

![](/public/upload/linux/thread_sturct_relation.png)

从CPU的视角，是如何访问task_struct结构的？进程的地址空间分为用户态和内核态，无论从哪个进程进入的内核态，进来后访问的是同一份，对应的也是物理内存中同一块空间。内核态 有一个数据结构 struct list_head tasks 维护了task_struct 列表。在处理器内部，有一个控制寄存器叫 CR3，存放着页目录的物理地址，故 CR3 又叫做页目录基址寄存器。
1. 每个进程都有自己的用户态虚拟地址空间，因而每个进程都有自己的页表，每个进程的页表的根，放在task_struct结构中。进程运行在用户态，则从task_struct里面找到页表顶级目录，加载到CR3里面去，则程序里面访问的虚拟地址就通过CPU指向的页表转换成物理地址进行访问。
2. 内核有统一的一个内核虚拟地址空间，因而内核也应该有个页表，内核页表的根是内存初始化的时候预设在一个虚拟地址和物理地址。进程进入内核后，CR3要变成指向内核页表的顶级目录。

### Per CPU的struct

SMP 系统的出现，对应用软件没有任何影响，因为应用软件始终看到是一颗 CPU，然而这却给操作系统带来了麻烦，操作系统必须使每个 CPU 都正确地执行进程。
1. 操作系统要开发更先进的同步机制，解决数据竞争问题。比如原子变量、自旋锁、信号量等高级的同步机制。
2. 进程调度问题，需要使得多个 CPU 尽量忙起来，否则多核还是等同于单核。为此，操作系统需要对进程调度模块进行改造。
    1. 单核 CPU 一般使用全局进程队列，系统所有进程都挂载到这个队列上，进程调度器每次从该队列中获取进程让 CPU 执行。**多核心系统下，每个 CPU 一个进程队列**，虽然提升了进程调度的性能，但同时又引发了另一个问题——每个 CPU 的压力各不相同。这是因为进程暂停或者退出，会导致各队列上进程数量不均衡，有的队列上很少或者没有进程，有的队列上进程数量则很多，间接地出现一部分 CPU 太忙吃不消，而其他 CPU 太闲（处于饥饿空闲状态）的情况。
    2. 怎么解决呢？这就需要操作系统时时查看各 CPU 进程队列上的进程数量，做出动态调整，把进程多的队列上的进程迁移到较少进程的队列上，使各大进程队列上的进程数量尽量相等，使 CPU 之间能为彼此分担压力。这就叫负载均衡，这种机制能提升系统的整体性能。这里有一个调度域的概念，从根级调度域、二级、三级到基本单位逻辑核构成一棵树，load_balance 的时候从下到上，先判断当前cpu是否有余力多处理一些任务，如果有则看下兄弟cpu谁最忙，**从其rq 拉一些任务放到自己的rq上就可以了**，当然也不是随便一个任务都可以拉出来，比如绑核或涉及到调度亲和性的就不可以。PS：这不就是go gmp work steal嘛。也侧面说明了golang 调度的必要性，linux调度线程要考虑的工作很多。

linux 内有很多 struct 是Per CPU的，估计是都在内核空间特定的部分。**有点线程本地变量的意思**

1. 结构体 tss， 所有寄存器切换 ==> 内存拷贝/拷贝到特定tss_struct
2. struct rq，为每一个CPU都创建一个队列来保存可以在这个CPU上运行的任务，这样调度的时候，只需要看当前的 CPU 上的资源就行，把（竞争全局线程队列）锁的开销就砍掉了。这里面包括一个实时进程队列rt_rq和一个CFS运行队列cfs_rq ，task_struct就是用sched_entity这个成员变量将自己挂载到某个CPU的队列上的。PS： Go中GMP 模型中的P抄的也是这个。

    ![](/public/upload/linux/cpu_runqueue.jpg)

进程创建后的一件重要的事情，就是调用sched_class的enqueue_task方法，**将这个进程放进某个CPU的队列上来**。选择CPU，CPU 调度是在缓存性能和空闲核心两个点之间做权衡，同等条件下会尽量优先考虑缓存命中率，选择同 L1/L2 的核，其次会选择同一个物理 CPU 上的（共享 L3），最坏情况下去选择另外一个物理 CPU 上的核心。

### 彭东《操作系统实战》

```c
struct task_struct {
    struct thread_info thread_info;//处理器特有数据 
    volatile long   state;       //进程状态 
    void            *stack;      //进程内核栈地址 
    refcount_t      usage;       //进程使用计数
    int             on_rq;       //进程是否在运行队列上
    // 进程调度优先级
    int             prio;        //动态优先级
    int             static_prio; //静态优先级
    int             normal_prio; //取决于静态优先级和调度策略
    unsigned int    rt_priority; //实时优先级
    const struct sched_class    *sched_class;//指向其所在的调度类
    struct sched_entity         se;//普通进程的调度实体
    struct sched_rt_entity      rt;//实时进程的调度实体
    struct sched_dl_entity      dl;//采用EDF算法调度实时进程的调度实体
    struct sched_info       sched_info;//用于调度器统计进程的运行信息 
    struct list_head        tasks;//所有进程的链表
    // 进程地址空间，在用户态所需要的内存数据，如代码、全局变量数据以及mmap 内存映射等都是通过mm_struct 进行内存查找和寻址的
    struct mm_struct        *mm;  //指向进程内存结构
    struct mm_struct        *active_mm;
    pid_t               pid;            //进程id
    // 进程树关系
    struct task_struct __rcu    *parent;//指向其父进程
    struct list_head        children; //链表中的所有元素都是它的子进程
    struct list_head        sibling;  //用于把当前进程插入到兄弟链表中
    struct task_struct      *group_leader;//指向其所在进程组的领头进程
    u64             utime;   //用于记录进程在用户态下所经过的节拍数
    u64             stime;   //用于记录进程在内核态下所经过的节拍数
    u64             gtime;   //用于记录作为虚拟机进程所经过的节拍数
    unsigned long           min_flt;//缺页统计 
    unsigned long           maj_flt;
    struct fs_struct        *fs;    //进程相关的文件系统信息
    struct files_struct     *files;//进程打开的所有文件
    struct vm_struct        *stack_vm_area;//内核栈的内存区
  };
```






