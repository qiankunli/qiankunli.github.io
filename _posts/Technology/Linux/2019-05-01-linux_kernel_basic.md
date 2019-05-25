---

layout: post
title: Linux内核基础知识
category: 技术
tags: Linux
keywords: linux 内核

---

## 简介

* TOC
{:toc}

## 虚拟地址空间：用户空间和内核空间

### 进程“独占”虚拟内存及虚拟内存划分

为了保证操作系统的稳定性和安全性。用户程序不可以直接访问硬件资源，如果用户程序需要访问硬件资源，必须调用操作系统提供的接口，这个调用接口的过程也就是系统调用。每一次系统调用都会存在两个内存空间之间的相互切换，通常的网络传输也是一次系统调用，通过网络传输的数据先是从内核空间接收到远程主机的数据，然后再**从内核空间复制到用户空间**，供用户程序使用。这种从内核空间到用户空间的数据复制很费时，虽然保住了程序运行的安全性和稳定性，但是牺牲了一部分的效率。

如何分配用户空间和内核空间的比例也是一个问题，是更多地分配给用户空间供用户程序使用，还是首先保住内核有足够的空间来运行。在当前的Windows 32位操作系统中，默认用户空间：内核空间的比例是1:1，而在32位Linux系统中的默认比例是3:1（3GB用户空间、1GB内核空间）（这里只是地址空间，映射到物理地址，可没有某个物理地址的内存只能存储内核态数据或用户态数据的说法）。

||用户地址空间|内核地址空间|备注|
|---|---|---|---|
|地址类型|虚拟地址|虚拟地址|都要经过 MMU 的翻译，变成物理地址|
|生存期|随进程创建产生|持续存在|
|共享|进程独占|所有进程共享|
|地址映射方式|走四级页表来翻译|线性映射|内核空间也必须有一部分是非线性映射，比如下图的vmalloc|
|地址映射/页表创建|随进程创建产生|内核在初始化时，就创建内核空间的映射，所有进程共享|
|对应物理空间|分散且不固定|提前固定下来一片连续的物理地址空间，所有进程共享|

“进程独占内存”的内涵：每个进程有自己独立的页表映射，当然内核地址空间是一样的，页表映射也是共享的，在创建进程时候，就可以直接“拷贝”内核的页表，作为该进程 的页表的一部分。进程用户地址空间部分另外映射。

![](/public/upload/linux/virtual_memory_space.jpg)

左右两侧均表示虚拟地址空间，左侧以描述内核空间为主，右侧以描述用户空间为主。

### 在代码上的体现

    // 持有task_struct 便可以访问进程在内存中的所有数据
    struct task_struct {
        ...
        struct mm_struct                *mm;
        struct mm_struct                *active_mm;
        ...
        void  *stack;   // 指向内核栈的指针
    }

内核使用内存描述符mm_struct来表示进程的地址空间，该描述符表示着进程所有地址空间的信息

![](/public/upload/linux/linux_virtual_address.png)

![](/public/upload/linux/mm_struct.png)

## 地址空间内的栈

[Linux虚拟地址空间布局以及进程栈和线程栈总结](https://www.cnblogs.com/sky-heaven/p/7112006.html)

栈是主要用途就是支持函数调用。

大多数的处理器架构，都有实现**硬件栈**。有专门的栈指针寄存器，以及特定的硬件指令来完成 入栈/出栈 的操作。

### 用户栈和内核栈的切换

删改自[进程内核栈、用户栈](http://www.cnblogs.com/shengge/articles/2158748.html)

内核在创建进程的时候，在创建task_struct的同时，会为进程创建相应的堆栈。每个进程会有两个栈，一个用户栈，存在于用户空间，一个内核栈，存在于内核空间。**当进程在用户空间运行时，cpu堆栈指针寄存器里面的内容是用户堆栈地址，使用用户栈；当进程在内核空间时，cpu堆栈指针寄存器里面的内容是内核栈空间地址，使用内核栈**。

当进程因为中断或者系统调用而陷入内核态之行时，进程所使用的堆栈也要从用户栈转到内核栈。

如何相互切换呢？

进程陷入内核态后，先把用户态堆栈的地址保存在内核栈之中，然后设置堆栈指针寄存器的内容为内核栈的地址，这样就完成了用户栈向内核栈的转换；当进程从内核态恢复到用户态执行时，在内核态执行的最后，将保存在内核栈里面的用户栈的地址恢复到堆栈指针寄存器即可。这样就实现了内核栈和用户栈的互转。

那么，我们知道从内核转到用户态时用户栈的地址是在陷入内核的时候保存在内核栈里面的，但是在陷入内核的时候，我们是如何知道内核栈的地址的呢？

**关键在进程从用户态转到内核态的时候，进程的内核栈总是空的**。这是因为，一旦进程从内核态返回到用户态后，内核栈中保存的信息无效，会全部恢复。因此，每次进程从用户态陷入内核的时候得到的内核栈都是空的，直接把内核栈的栈顶地址给堆栈指针寄存器就可以了。

### 为什么需要单独的进程内核栈？

内核地址空间所有进程空闲，但内核栈却不共享。为什么需要单独的进程内核栈？**因为同时可能会有多个进程在内核运行**。

所有进程运行的时候，都可能通过系统调用陷入内核态继续执行。假设第一个进程 A 陷入内核态执行的时候，需要等待读取网卡的数据，主动调用 `schedule()` 让出 CPU；此时调度器唤醒了另一个进程 B，碰巧进程 B 也需要系统调用进入内核态。那问题就来了，如果内核栈只有一个，那进程 B 进入内核态的时候产生的压栈操作，必然会破坏掉进程 A 已有的内核栈数据；一但进程 A 的内核栈数据被破坏，很可能导致进程 A 的内核态无法正确返回到对应的用户态了。

进程内核栈在**进程创建的时候**，通过 slab 分配器从 thread_info_cache 缓存池中分配出来，其大小为 THREAD_SIZE，一般来说是一个页大小 4K；

### 进程切换带来的用户栈切换和内核栈切换

    // 持有task_struct 便可以访问进程在内存中的所有数据
    struct task_struct {
        ...
        struct mm_struct                *mm;
        struct mm_struct                *active_mm;
        ...
        void  *stack;   // 指向内核栈的指针
    }

从进程 A 切换到进程 B，用户栈要不要切换呢？当然要，在切换内存空间的时候就切换了，每个进程的用户栈都是独立的，都在内存空间里面。

那内核栈呢？已经在 __switch_to 里面切换了，也就是将 current_task 指向当前的 task_struct。里面的 void *stack 指针，指向的就是当前的内核栈。

内核栈的栈顶指针呢？在 __switch_to_asm 里面已经切换了栈顶指针，并且将栈顶指针在 __switch_to加载到了 TSS 里面。

用户栈的栈顶指针呢？如果当前在内核里面的话，它当然是在内核栈顶部的 pt_regs 结构里面呀。当从内核返回用户态运行的时候，pt_regs 里面有所有当时在用户态的时候运行的上下文信息，就可以开始运行了。

## 中断栈

[Java和操作系统交互细节](https://mp.weixin.qq.com/s/fmS7FtVyd7KReebKzxzKvQ)中断有点类似于我们经常说的事件驱动编程，而这个事件通知机制是怎么实现的呢，硬件中断的实现通过一个导线和 CPU 相连来传输中断信号，软件上会有特定的指令，例如执行系统调用创建线程的指令，而 CPU 每执行完一个指令，就会检查中断寄存器中是否有中断，如果有就取出然后执行该中断对应的处理程序。

当系统收到中断事件后，进行中断处理的时候，也需要中断栈来支持函数调用。由于系统中断的时候，系统当然是处于内核态的，所以中断栈是可以和内核栈共享的。但是具体是否共享，这和具体处理架构密切相关。ARM 架构就没有独立的中断栈。

## 进程调度

**进程调度第一定律**：所有进程的调度最终是通过正在运行的进程调用__schedule 函数实现

![](/public/upload/linux/process_schedule.png)

### 基于虚拟运行时间的调度

    struct task_struct{
        ...
        unsigned int policy;    // 调度策略
        ...
        int prio, static_prio, normal_prio;
        unsigned int rt_priority;
        ...
        const struct sched_class *sched_class; // 调度策略的执行逻辑
    }

CPU 会提供一个时钟，过一段时间就触发一个时钟中断Tick，定义一个vruntime来记录一个进程的虚拟运行时间。如果一个进程在运行，随着时间的增长，也就是一个个 tick 的到来，进程的 vruntime 将不断增大。没有得到执行的进程 vruntime 不变。为什么是 虚拟运行时间呢？`虚拟运行时间 vruntime += delta_exec * NICE_0_LOAD/ 权重`。就好比可以把你安排进“尖子班”变相走后门，但高考都是按分数（vruntime）统一考核的

调度需要一个数据结构来对 vruntime 进行排序，因为任何一个策略做调度的时候，都是要区分谁先运行谁后运行。这个能够排序的数据结构不但需要查询的时候，能够快速找到最小的，更新的时候也需要能够快速的调整排序，毕竟每一个tick vruntime都会增长。能够平衡查询和更新速度的是树，在这里使用的是红黑树。sched_entity 表示红黑树的一个node（数据结构中很少有一个Tree 存在，都是根节点`Node* root`就表示tree了）。

    struct task_struct{
        ...
        struct sched_entity se;     // 对应完全公平算法调度
        struct sched_rt_entity rt;  // 对应实时调度
        struct sched_dl_entity dl;  // 对应deadline 调度
        ...
    }

每个 CPU 都有自己的 struct rq 结构，其用于描述在此 CPU 上所运行的所有进程，其包括一个实时进程队列rt_rq 和一个 CFS 运行队列 cfs_rq。在调度时，调度器首先会先去实时进程队列找是否有实时进程需要运行，如果没有才会去 CFS 运行队列找是否有进行需要运行。这样保证了实时任务的优先级永远大于普通任务。

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

CFS 的队列是一棵红黑树（所以叫“队列”很误导人），树的每一个节点都是一个 sched_entity（说白了每个节点是一个进/线程），每个 sched_entity 都属于一个 task_struct，task_struct 里面有指针指向这个进程属于哪个调度类。

<div class="class=width:100%;height:auto;">
    <img src="/public/upload/linux/process_schedule_impl.jpeg"/>
</div>

基于进程调度第一定律，上图就是一个很完整的循环，cpu的执行一直是方法调方法（process1.func1 ==> process1.schedule ==> process2.func2 ==> process2.schedule ==> process3.func3），只不过是跨了进程

### 调度类

如果将task_struct 视为一个对象，在很多场景下 主动调用`schedule()` 让出cpu，那么如何选取下一个task 就是其应该具备的能力，sched_class 作为其成员就顺理成章了。

    struct task_struct{
        const struct sched_class *sched_class; // 调度策略的执行逻辑
    }

![](/public/upload/linux/schedule_class.png)

sched_class结构体类似面向对象中的基类啊,通过函数指针类型的成员指向不同的函数，实现了多态。

### 主动调度

主动调度，就是进程运行到一半，因为等待 I/O 等操作而主动调用 schedule() 函数让出 CPU。

写入块设备的一个典型场景。写入需要一段时间，这段时间用不上CPU

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

从 Tap 网络设备等待一个读取

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

**这段跟golang协程的读写过程 是一样一样的**，内核机制上层化（内存管理、线程调度放到语言层/框架层来解决）是一个普遍趋势。

### 抢占式调度

在计算机里面有一个时钟，会过一段时间触发一次时钟中断，时钟中断处理函数会调用 scheduler_tick()，代码如下

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

对于普通进程 scheduler_tick ==> fair_sched_class.task_tick_fair ==> entity_tick ==> update_curr 更新当前进程的 vruntime ==> check_preempt_tick 检查是否是时候被抢占了

当发现当前进程应该被抢占，不能直接把它踢下来，而是把它标记为应该被抢占。为什么呢？因为进程调度第一定律呀，一定要等待正在运行的进程调用 __schedule 才行


### Schedule

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

上下文切换主要干两件事情，一是切换进程空间，也即虚拟内存；二是切换寄存器和 CPU 上下文。

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

## Per CPU的struct

linux 内有很多 struct 是Per CPU的，估计是都在内核空间特定的部分。**有点线程本地变量的意思**

1. struct rq，描述在此 CPU 上所运行的所有进程
2. 结构体 tss， 所有寄存器切换 ==> 内存拷贝/拷贝到特定tss_struct

在 x86 体系结构中，提供了一种以硬件的方式进行进程切换的模式，对于每个进程，x86 希望在内存里面维护一个 TSS（Task State Segment，任务状态段）结构。这里面有所有的寄存器。另外，还有一个特殊的寄存器 TR（Task Register，任务寄存器），指向某个进程的 TSS。更改 TR 的值，将会触发硬件保存 CPU 所有寄存器的值到当前进程的 TSS 中，然后从新进程的 TSS 中读出所有寄存器值，加载到 CPU 对应的寄存器中。

但是这样有个缺点。我们做进程切换的时候，没必要每个寄存器都切换，这样每个进程一个 TSS，就需要全量保存，全量切换，动作太大了。于是，Linux 操作系统想了一个办法。还记得在系统初始化的时候，会调用 cpu_init 吗？这里面会给每一个CPU 关联一个 TSS，然后将 TR 指向这个 TSS，然后在操作系统的运行过程中，TR 就不切换了，永远指向这个TSS

在 Linux 中，真的参与进程切换的寄存器很少，主要的就是栈顶寄存器

所谓的进程切换，就是将某个进程的 thread_struct里面的寄存器的值，写入到 CPU 的 TR 指向的 tss_struct，对于 CPU 来讲，这就算是完成了切换。


