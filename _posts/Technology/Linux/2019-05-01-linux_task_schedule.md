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

## 进程数据结构

![](/public/upload/linux/linux_task_struct_data.png)

一个进程的运行竟然要保存这么多信息，这些信息都可以通过命令行取出来。fork 进程时， 创建一个空的task_struct 结构之后，这些信息也将被一一复制。

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


![](/public/upload/linux/task_fork.jpeg)


||创建进程|创建线程|
|---|---|---|
|系统调用|fork|clone|
|copy_process逻辑|会将五大结构 files_struct、fs_struct、sighand_struct、signal_struct、mm_struct 都复制一遍<br>从此父进程和子进程各用各的数据结构|五大结构仅仅是引用计数加一<br>也即线程共享进程的数据结构|
||完全由内核实现|由内核态和用户态合作完成<br>相当一部分逻辑由glibc库函数pthread_create来做|
|数据结构||内核态struct task_struct <br>用户态 struct pthread|

[线程创建的成本](http://qiankunli.github.io/2014/10/09/Threads.html)

[线程切换的成本](http://qiankunli.github.io/2018/01/07/hardware_software.html)



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


