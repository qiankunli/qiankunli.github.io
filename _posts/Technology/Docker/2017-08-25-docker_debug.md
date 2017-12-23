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

但是!

### java项目增大内存是不解决问题的

参见

1. [在 Docker 里跑 Java，趟坑总结](http://blog.tenxcloud.com/?p=1894)， jvm无法感知到自己在容器中进行，默认，堆的上限是物理机内存的四分之一，当容器的jvm没有设置xmx，即便容器内存设置的很大，也没有解决问题，导致容器会周期性重启（没有gc，逐渐累积到容器内存的限制值）。结论：要管控jvm 堆大小等参数，或使用特殊镜像。

2. [在docker中使用java的内存情况](http://www.jianshu.com/p/1bf938fd8d70)提到了容器内存与jvm堆内存的基本关系。`Max memory = [-Xmx] + [-XX:MaxPermSize] + number_of_threads * [-Xss]`.在设置jvm启动参数的时候 -Xmx的这个值一般要小于docker限制内存数，个人觉得  -Xmx:docker的比例为 4/5 - 3/4

## Container stuck, can't be stopped or killed, can't exec into it either

jdk6 编译的项目运行在jdk8上

1. 代码本身经由jdk6编译，运行在jdk8上
2. 代码依赖的jar由jdk6编译

在docker-ce 1.3 以下会出现`docker ps`可以看到，但容器内jvm进程已经退出的情况。升级到docker-ce 1.7 则貌似解决了该问题。

2017.12.05 更新

现象描述：

1. marathon ==> mesos 尝试杀死容器，但杀死失败
1. `docker ps`可以看到容器
2. `docker exec`无法进入容器
3. `docker rm`可以移除容器
3. `journalctl -xe -u docker.service` 可以看到：

		time="2017-12-05T10:14:01.066063275+08:00" level=info msg="Container 8f8020e79b13bcbf40298d3c9680cd1a838e16dc810ef3b1357eb3e75f78ef99 failed to exit within 120 seconds
		time="2017-12-05T10:14:01.066755930+08:00" level=warning msg="container kill failed because of 'container not found' or 'no such process': Cannot kill container 8f8020
		time="2017-12-05T10:14:11.067151142+08:00" level=info msg="Container 8f8020e79b13 failed to exit within 10 seconds of kill - trying direct SIGKILL"
		time="2017-12-05T12:52:58.017517865+08:00" level=error msg="Error running exec in container: containerd: container not found"
		
通过观察容器中运行的tomcat日志，可以看到，10:12，docker向容器发送 term signal，tomcat 开始stop，`docker logs --tail 200  8f8020e79b13`日志如下

	[album-facade]10:12:02 420 INFO  org.springframework.beans.factory.support.DefaultListableBeanFactory:444 - Destroying singletons in org.springframework.beans.factory.support.DefaultListableBeanFactory@b342fcf: defining beans [mvcContentNegotiationManager,org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerMapping#0,org.springframework.format.support.FormattingConversionServiceFactoryBean#0,org.springframework.web.servlet.mvc.method.annotation.RequestM...
	Dec 5, 2017 10:12:05 AM org.apache.catalina.loader.WebappClassLoader checkThreadLocalMapForLeaks
	SEVERE: The web application [/album-facade] created a ThreadLocal with key of type [java.lang.ThreadLocal] (value [java.lang.ThreadLocal@64de4a8c]) and a value of type [com.ibatis.sqlmap.engine.mapping.result.ResultObjectFactoryUtil.FactorySettings] (value [com.ibatis.sqlmap.engine.mapping.result.ResultObjectFactoryUtil$FactorySettings@70748134]) but failed to remove it when the web application was stopped. This is very likely to create a memory leak.
	Dec 5, 2017 10:12:15 AM org.apache.coyote.http11.Http11Protocol destroy
	INFO: Stopping Coyote HTTP/1.1 on http-8080
	
	
tomcat花了12秒停掉，但是docker认为容器还未停掉，等了120s到10:14，尝试kill，10s后仍然失败，然后任何对容器的操作就`container not found`。可能原因：移除容器的操作确实开始执行，但到了某一个步骤卡住或失败，导致一部分数据结构显示docker还在，但另一个部分数据结构已经删掉了。

docker认为容器一直“活着”，但主进程已经退出了。所以，容器退出不等于主进程退出。主进程退出后，docker还要回收各种资源，比如volume等等，耗时太长，或者干脆操作失败。

仍需解决！

## docker 停住
环境：

1. ubuntu 16.04
2. kernel 4.4.0
3. docker 17.09-ce


12月13日下午排查时发现

1. docker 日志停留在 12.12 18.52
2. docker rm 容器无法remove掉
3. docker 无法停掉，`systemctl stop docker`卡住
4. kill docker 所有进程，`rm /var/lib/docker` 时，部分文件无法删除，`deivce or resource is busy`
5. `/var/log/kern.log` 大量的

		[1055520.745390] unregister_netdevice: waiting for veth54a2de6 to become free. Usage count = 1
		Dec 13 15:53:58 test-a1-60-36 kernel: [1055530.985735] unregister_netdevice: waiting for veth54a2de6 to become free. Usage count = 1
		Dec 13 15:54:09 test-a1-60-36 kernel: [1055541.226086] unregister_netdevice: waiting for veth54a2de6 to become free. Usage count = 1
		Dec 13 15:54:19 test-a1-60-36 kernel: [1055551.466446] unregister_netdevice: waiting for veth54a2de6 to become free. Usage count = 1


60.36 ad 项目日志打了4t

解决 [Swarm Kernel Panic after "unregister_netdevice"](https://github.com/moby/moby/issues/35068)，升级了内核版本到4.14.5


## 发现与预防

如何评估docker集群的健康状态？具体的，对于笔者实践中应用的mesos集群？


![](/public/upload/docker/docker_debug_completed_tasks.png)

对于`http://mesos:5050/`的completed tasks列表，正常的开发同学更新代码，老的task会是killed状态。而非正常退出，比如上文说的内存不够问题，则是failed。因此当系统中发现大量的failed task时，即需要警惕并排查原因了。










	
	