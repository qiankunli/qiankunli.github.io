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

之所以抽象一个执行体（进程/线程/协程）的概念，是要分时使用硬件（cpu和寄存器等）。 调度器就好比教练，线程就好比球员，给线程调度cpu就好比安排哪个球员上球场。

[万字详解Linux内核调度器及其妙用](https://mp.weixin.qq.com/s/gkZ0kve8wOrV5a8Q2YeYPQ)可以试着切换到硬件CPU的视角，这样应用程序和内核，就都变成客体研究对象了，很多逻辑也就清晰了。

## 数据结构

彭东《操作系统实战》
```c
struct task_struct {
    struct thread_info thread_info;//处理器特有数据 
    volatile long   state;       //进程状态 
    void            *stack;      //进程内核栈地址 
    refcount_t      usage;       //进程使用计数
    int             on_rq;       //进程是否在运行队列上
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
    struct mm_struct        *mm;  //指向进程内存结构
    struct mm_struct        *active_mm;
    pid_t               pid;            //进程id
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
**进程及调度数据结构的组织**：在 task_struct 结构中，会包含至少一个 sched_entity 结构的变量。它其实是 Linux 进程调度系统的一部分，被嵌入到了 Linux 进程数据结构中，与调度器进行关联，能间接地访问进程。我们只要通过 sched_entity 结构变量的地址，减去它在 task_struct 结构中的偏移（由编译器自动计算），就能获取到 task_struct 结构的地址。这样就能达到通过 sched_entity 结构，访问 task_struct 结构的目的了。sched_entity 结构是通过红黑树组织起来的，红黑树的根在 cfs_rq 结构中，cfs_rq 结构又被包含在 rq 结构，每个 CPU 对应一个 rq 结构。

![](/public/upload/linux/thread_sturct_relation.png)

### 进程部分

![](/public/upload/linux/linux_task_struct_data.png)

一个进程的运行竟然要保存这么多信息，这些信息都可以通过命令行取出来。fork 进程时， 创建一个空的task_struct 结构之后，这些信息也将被一一复制。

```c
long _do_fork(unsigned long clone_flags,
        unsigned long stack_start,
        unsigned long stack_size,
        int __user *parent_tidptr,
        int __user *child_tidptr,
        unsigned long tls){
    struct task_struct *p;
    int trace = 0;
    long nr;
    ......
    // 复制结构
    p = copy_process(clone_flags, stack_start, stack_size,
            child_tidptr, NULL, trace, tls, NUMA_NO_NODE);
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
```
### Per CPU的struct

linux 内有很多 struct 是Per CPU的，估计是都在内核空间特定的部分。**有点线程本地变量的意思**

1. 结构体 tss， 所有寄存器切换 ==> 内存拷贝/拷贝到特定tss_struct
2. struct rq，为每一个CPU都创建一个队列来保存可以在这个CPU上运行的任务，这里面包括一个实时进程队列rt_rq和一个CFS运行队列cfs_rq ，task_struct就是用sched_entity这个成员变量将自己挂载到某个CPU的队列上的。
    ![](/public/upload/linux/cpu_runqueue.jpg)

进程创建后的一件重要的事情，就是调用sched_class的enqueue_task方法，将这个进程放进某个CPU的队列上来，虽然不一定马上运行，但是说明可以在这个CPU上被调度上去运行了。

在 x86 体系结构中，提供了一种以硬件的方式进行进程切换的模式，对于每个进程，x86 希望在内存里面维护一个 TSS（Task State Segment，任务状态段）结构。这里面有所有的寄存器。另外，还有一个特殊的寄存器 TR（Task Register，任务寄存器），指向某个进程的 TSS。更改 TR 的值，将会触发硬件保存 CPU 所有寄存器的值到当前进程的 TSS 中，然后从新进程的 TSS 中读出所有寄存器值，加载到 CPU 对应的寄存器中。

但是这样有个缺点。我们做进程切换的时候，没必要每个寄存器都切换，这样每个进程一个 TSS，就需要全量保存，全量切换，动作太大了。于是，Linux 操作系统想了一个办法。还记得在系统初始化的时候，会调用 cpu_init 吗？这里面会给每一个CPU 关联一个 TSS，然后将 TR 指向这个 TSS，然后在操作系统的运行过程中，TR 就不切换了，永远指向这个TSS

在 Linux 中，真的参与进程切换的寄存器很少，主要的就是栈顶寄存器

所谓的进程切换，就是将某个进程的 thread_struct里面的寄存器的值，写入到 CPU 的 TR 指向的 tss_struct，对于 CPU 来讲，这就算是完成了切换。

![](/public/upload/linux/cpu_rq.png)

### cpu 如何访问task_struct

从CPU的视角，是如何访问task_struct结构的？进程的地址空间分为用户态和内核态，无论从哪个进程进入的内核态，进来后访问的是同一份，对应的也是物理内存中同一块空间。内核态 有一个数据结构 struct list_head tasks 维护了task_struct 列表。在处理器内部，有一个控制寄存器叫 CR3，存放着页目录的物理地址，故 CR3 又叫做页目录基址寄存器。
1. 每个进程都有自己的用户态虚拟地址空间，因而每个进程都有自己的页表，每个进程的页表的根，放在task_struct结构中。进程运行在用户态，则从task_struct里面找到页表顶级目录，加载到CR3里面去，则程序里面访问的虚拟地址就通过CPU指向的页表转换成物理地址进行访问。
2. 内核有统一的一个内核虚拟地址空间，因而内核也应该有个页表，内核页表的根是内存初始化的时候预设在一个虚拟地址和物理地址。进程进入内核后，CR3要变成指向内核页表的顶级目录。

## 进程创建

[万字详解Linux内核调度器及其妙用](https://mp.weixin.qq.com/s/gkZ0kve8wOrV5a8Q2YeYPQ) 整个Linux系统的第一个用户态进程就是这样运行起来的。Linux系统启动的时候，先初始化的肯定是内核，当内核初始化结束了，会创建第一个用户态进程，1号进程。创建的方式是在内核态运行do_execve，来运行"/sbin/init"，"/etc/init"，"/bin/init"，"/bin/sh"中的一个，不同的Linux版本不同。

写过Linux程序的我们都知道，execve是一个系统调用，它的作用是运行一个执行文件。加一个do_的往往是内核系统调用的实现。

在do_execve中，会有一步是设置struct pt_regs，主要设置的是ip和sp，指向第一个进程的起始位置，这样调用iret就可以从系统调用中返回。这个时候会从pt_regs恢复寄存器。指令指针寄存器IP恢复了，指向用户态下一个要执行的语句。函数栈指针SP也被恢复了，指向用户态函数栈的栈顶。所以，下一条指令，就从用户态开始运行了。

接下来所有的用户进程都是这个1号进程的徒子徒孙了。如果要创建新进程，是某个用户态进程调用fork，fork是系统调用会调用到内核，在内核中子进程会复制父进程的几乎一切，包括task_struct，内存空间等。这里注意的是，fork作为一个系统调用，是将用户态的当前运行状态放在pt_regs里面了，IP和SP指向的就是fork这个函数，然后就进内核了。

子进程创建完了，如果像前面讲过的check_preempt_curr里面成功抢占了父进程，则父进程会被标记TIF_NEED_RESCHED，则在fork返回的时候，会检查这个标记，会将CPU主动让给子进程，然后让子进程从内核返回用户态，因为子进程是完全复制的父进程，因而返回用户态的时候，仍然在fork的位置，当然父进程将来返回的时候，也会在fork的位置，只不过返回值不一样，等于0说明是子进程。

![](/public/upload/linux/task_fork.jpeg)

||创建进程|创建线程|
|---|---|---|
|系统调用|fork|clone|
|copy_process逻辑|会将五大结构 files_struct、fs_struct、sighand_struct、signal_struct、mm_struct 都复制一遍<br>从此父进程和子进程各用各的数据结构|五大结构仅仅是引用计数加一<br>也即线程共享进程的数据结构|
||完全由内核实现|由内核态和用户态合作完成<br>相当一部分逻辑由glibc库函数pthread_create来做|
|数据结构||内核态struct task_struct <br>用户态 struct pthread|

## 进程调度

**进程调度第一定律**：所有进程的调度最终是通过正在运行的进程调用__schedule 函数实现

![](/public/upload/linux/process_schedule.png)

### 基于虚拟运行时间的调度

```c
struct task_struct{
    ...
    unsigned int policy;    // 调度策略
    ...
    int prio, static_prio, normal_prio;
    unsigned int rt_priority;
    ...
    const struct sched_class *sched_class; // 调度策略的执行逻辑
}
```

CPU 会提供一个时钟，过一段时间就触发一个时钟中断Tick，task_struct.sched_entity里面有一个重要的变量vruntime，来记录一个进程的虚拟运行时间。如果一个进程在运行，随着时间的增长，也就是一个个 tick 的到来，进程的 vruntime 将不断增大。没有得到执行的进程 vruntime 不变。为什么是 虚拟运行时间呢？`虚拟运行时间 vruntime += 实际运行时间 delta_exec * NICE_0_LOAD/ 权重`。就好比可以把你安排进“尖子班”变相走后门，但高考都是按分数（vruntime）统一考核的。PS， vruntime 正是理解 docker --cpu-shares 的钥匙。

```c
/*
 * Update the current task's runtime statistics.
 */
static void update_curr(struct cfs_rq *cfs_rq)
{
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
static inline u64 calc_delta_fair(u64 delta, struct sched_entity *se)
{
  if (unlikely(se->load.weight != NICE_0_LOAD))
        /* delta_exec * weight / lw.weight */
    delta = __calc_delta(delta, NICE_0_LOAD, &se->load);
  return delta;
}
```

调度需要一个数据结构来对 vruntime 进行排序，因为任何一个策略做调度的时候，都是要区分谁先运行谁后运行。这个能够排序的数据结构不但需要查询的时候，能够快速找到最小的，更新的时候也需要能够快速的调整排序，毕竟每一个tick vruntime都会增长。能够平衡查询和更新速度的是树，在这里使用的是红黑树。sched_entity 表示红黑树的一个node（数据结构中很少有一个Tree 存在，都是根节点`Node* root`就表示tree了）。

```c
struct task_struct{
    ...
    struct sched_entity se;     // 对应完全公平算法调度
    struct sched_rt_entity rt;  // 对应实时调度
    struct sched_dl_entity dl;  // 对应deadline 调度
    ...
}
```

每个 CPU 都有自己的 struct rq 结构，其用于描述在此 CPU 上所运行的所有进程，其包括一个实时进程队列rt_rq 和一个 CFS 运行队列 cfs_rq。在调度时，调度器首先会先去实时进程队列找是否有实时进程需要运行，如果没有才会去 CFS 运行队列找是否有进行需要运行。这样保证了实时任务的优先级永远大于普通任务。

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

CFS 的队列是一棵红黑树（所以叫“队列”很误导人），树的每一个节点都是一个 sched_entity（说白了每个节点是一个进/线程），每个 sched_entity 都属于一个 task_struct，task_struct 里面有指针指向这个进程属于哪个调度类。

<div class="class=width:100%;height:auto;">
    <img src="/public/upload/linux/process_schedule_impl.jpeg"/>
</div>

基于进程调度第一定律，上图就是一个很完整的循环，**cpu的执行一直是方法调方法**（process1.func1 ==> process1.schedule ==> process2.func2 ==> process2.schedule ==> process3.func3），只不过是跨了进程

### 调度类

如果将task_struct 视为一个对象，在很多场景下 主动调用`schedule()` 让出cpu，那么如何选取下一个task 就是其应该具备的能力，sched_class 作为其成员就顺理成章了。

```c
struct task_struct{
    const struct sched_class *sched_class; // 调度策略的执行逻辑
}
```

![](/public/upload/linux/schedule_class.png)

sched_class结构体类似面向对象中的基类啊,通过函数指针类型的成员指向不同的函数，实现了多态。

### 主动调度

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
    // 获取下一个任务
    next = pick_next_task(rq, prev, &rf);
    clear_tsk_need_resched(prev);
    clear_preempt_need_resched();
    // 当选出的继任者和前任不同，就要进行上下文切换，继任者进程正式进入运行
    if (likely(prev != next)) {
    rq->nr_switches++;
    rq->curr = next;
    ++*switch_count;
    ......
    rq = context_switch(rq, prev, next, &rf);
}
```

上下文切换主要干两件事情，一是切换进程空间，也即虚拟内存；二是切换寄存器和 CPU 上下文。

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
    // 切换寄存器
    switch_to(prev, next, prev);
    barrier();
    return finish_task_switch(prev);
}
```
假如进程AB switch_to：在task_struct里面，还有一个成员变量struct thread_struct thread，里面保存了进程切换时的寄存器的值。在switch_to里面，A进程的SP就保存进A进程的task_struct的thread结构中，而B进程的SP就从他的task_struct的thread结构中取出来，加载到CPU的SP栈顶寄存器里面。在switch_to里面，还将这个CPU的current_task指向B进程的task_struct。当switch_to干完了上面的事情，返回的时候，IP就指向了下一行代码finish_task_switch。下面请问，这里的finish_task_switch是B进程的finish_task_switch，还是A进程的finish_task_switch呢？其实是B进程的finish_task_switch，因为当年B进程就是运行到switch_to被切走的，所以现在回来，运行的是B进程的finish_task_switch。其实从CPU的角度来看，这个时候还不区分A还是B的finish_task_switch，在CPU眼里，这就是内核的一行代码。但是代码再往下运行，就能区分出来了，因为finish_task_switch结束，要从schedule函数返回了，那应该是返回到A还是B对应的内核态代码呢？根据函数调用栈的原理，栈顶指针指向哪行，就返回哪行，别忘了前面SP已经切换成为B进程的了，已经指向B的内核栈了，所以返回的是B内核态代码。

虽然指令指针寄存器IP还是一行一行代码的执行下去，但由于所有的调度都会走schedule函数，IP没变，但是SP变了，进程切换就完成了。

什么是调度延迟？其实就是保证每一个可运行的进程，都至少运行一次的时间间隔。假设系统中有 3 个可运行进程，每个进程都运行 10ms，那么调度延迟就是 30ms；随着进程的增加，每个进程分配的时间在减少，进程调度次数会增加，调度器占用的时间就会增加。因此，CFS 调度器的调度延迟时间的设定并不是固定的。当运行进程少于 8 个的时候，调度延迟是固定的 6ms 不变。当运行进程个数超过 8 个时，就要保证每个进程至少运行一段时间，才被调度。这个“至少一段时间”叫作最小调度粒度时间。在 CFS 默认设置中，最小调度粒度时间是 0.75ms

## 其它

应用：CFS的抢占，其实是不那么暴力的抢占。目前互联网企业都拥有海量的服务器，其中大部分只运行交互类延时敏感的在线业务，使CPU利用率非常低，造成资源的浪费（据统计全球服务器CPU利用率不足20%）。为提高服务器CPU的利用率，需要在运行在线业务的服务器上，混合部署一些高CPU消耗且延时不敏感的离线业务。为了使得在离线互相不影响，需要在在线业务CPU使用率低的时候，可以运行一些离线业务，而当在线业务CPU利用率高的时候，可以对离线业务快速抢占。如果在离线业务都使用不暴力的CFS调度器，现有的混部方案没办法做到及时抢占的。在同调度类优先级的进程，互相抢占的时候，需要满足两个条件。第一个是抢占进程的虚拟时间要小于被抢占进程，第二是被抢占进程已经运行的实际要大于最小运行时间。如果两个条件不能同时满足，就会导致无法抢占。基于这种考虑，开发了针对离线业务的新调度算法[bt](https://github.com/Tencent/TencentOS-kernel)，该算法可以保证在线业务优先运行。

cpu 就是不端从pc 找活儿（指令）干，加上scheduler + task list之后就是不断从task list找个活儿（task_struct）干，跟java executor 不断从队列找活儿（runnable）干是一样的。又比如go中，用户逻辑包在goroutine中，goroutine 放在P中，M 不断从P 中取出G 来干活儿。 就好像网络包，一层套一层，符合一定的格式才可以收发和识别。

![](/public/upload/basic/scheduler_design.png)
