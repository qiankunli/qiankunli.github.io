---

layout: post
title: 一次docker debug过程
category: 技术
tags: Docker
keywords: Docker

---


## 问题描述

环境：marathon + mesos + docker 集群

现象：容器有时会莫名其妙重启


## debug过程

![](/public/upload/docker/mesos_debug_tab.png)

1. marathon debug tab页发现：Container exited with status 137

2. 根据mesos link 查看error日志，无收获

2. 找到最后一次失败所在的物理机`test-a3-60-17`，并登陆

3. `docker ps -a`找到那个失败的容器id`812c82c5a7a5`，并未发现异常日志，应该不是容器业务本身的原因。

4. 复制容器id，`cat /var/log/messages | grep 812c82c5a7a5 -n`确定相关日志的大致行号范围

5. `sed -n 'start_line,end_line p' /var/log/messages`看看docker最后发出该容器的信息之前，都发生了什么。

日志内容为

	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff816861cc>] dump_stack+0x19/0x1b
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff81681177>] dump_header+0x8e/0x225
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff811842b6>] ? find_lock_task_mm+0x56/0xc0
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff8118476e>] oom_kill_process+0x24e/0x3c0
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff8118420d>] ? oom_unkillable_task+0xcd/0x120
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff810937ee>] ? has_capability_noaudit+0x1e/0x30
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff811f3131>] mem_cgroup_oom_synchronize+0x551/0x580
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff811f2580>] ? mem_cgroup_charge_common+0xc0/0xc0
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff81184ff4>] pagefault_out_of_memory+0x14/0x90
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff8167ef67>] mm_fault_error+0x68/0x12b
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff81691ed5>] __do_page_fault+0x395/0x450
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff81691fc5>] do_page_fault+0x35/0x90
	Aug 25 18:21:35 test-a3-60-17 kernel: [<ffffffff8168e288>] page_fault+0x28/0x30
	Aug 25 18:21:35 test-a3-60-17 kernel: Task in /docker/812c82c5a7a5ae054e6f242215bdc4415a2d9b690768962a7bcd8d4676b16f60 killed as a result of limit of /docker/812c82c5a7a5ae054e6f242215bdc4415a2d9b690768962a7bcd8d4676b16f60
	Aug 25 18:21:35 test-a3-60-17 kernel: memory: usage 4194304kB, limit 4194304kB, failcnt 25581
	Aug 25 18:21:35 test-a3-60-17 kernel: memory+swap: usage 4194304kB, limit 8388608kB, failcnt 0
	Aug 25 18:21:35 test-a3-60-17 kernel: kmem: usage 0kB, limit 9007199254740988kB, failcnt 0
	Aug 25 18:21:35 test-a3-60-17 kernel: Memory cgroup stats for /docker/812c82c5a7a5ae054e6f242215bdc4415a2d9b690768962a7bcd8d4676b16f60: cache:52KB rss:4194252KB rss_huge:1277952KB mapped_file:0KB swap:0KB inactive_anon:939800KB active_anon:3254256KB inactive_file:8KB active_file:0KB unevictable:0KB
	Aug 25 18:21:35 test-a3-60-17 kernel: [ pid ]   uid  tgid total_vm      rss nr_ptes swapents oom_score_adj name
	Aug 25 18:21:35 test-a3-60-17 kernel: [76513]     0 76513  6553193  1047384    2363        0             0 java
	Aug 25 18:21:35 test-a3-60-17 kernel: Memory cgroup out of memory: Kill process 76756 (java) score 1001 or sacrifice child
	Aug 25 18:21:35 test-a3-60-17 kernel: Killed process 76513 (java) total-vm:26212772kB, anon-rss:4189536kB, file-rss:0kB, shmem-rss:0kB
	Aug 25 18:21:35 test-a3-60-17 dockerd: time="2017-08-25T18:21:35.949063698+08:00" level=error msg="containerd: deleting container" error="exit status 1: \"container 812c82c5a7a5ae054e6f242215bdc4415a2d9b690768962a7bcd8d4676b16f60 does not exist\\none or more of the container deletions failed\\n\""
	Aug 25 18:21:35 test-a3-60-17 mesos-slave[2905]: I0825 18:21:35.965833  2952 slave.cpp:3634] Handling status update TASK_FAILED (UUID: 031c0691-0dd0-4030-872e-d7d210c554c4) for task deploy-to-docker-business-product-rpc-test.a22e5797-897e-11e7-9837-f2f3c189fa2c of framework d637e32a-a1df-43eb-adaf-b1d2e3d6235a-0000 from executor(1)@192.168.60.17:38672
	
可以看到，在mesos和docker containerd对812c82c5a7a5做出反应之前，kerner因为内存限制的缘故，kill掉了一个进程。

如果有机会找到76513和812c82c5a7a5的对应关系，问题基本就可以确认了。

解决方法：增加内存







	
	