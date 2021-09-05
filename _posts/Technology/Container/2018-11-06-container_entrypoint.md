---

layout: post
title: 为容器选择一个合适的entrypoint
category: 技术
tags: Container
keywords: container entrypoint

---

## 简介

* TOC
{:toc}

在容器化早期，因为需要ssh访问容器等原因（尽量为开发提供与物理机一致的体验），需要一个容器内运行业务进程与ssh等进程。随着k8s pod 多容器的推进， 一个容器内多进程的需求减弱，但业务进程直接作为 entrypoint 仍有很多问题，因此会寻求一些进程管理工具作为pid=1进程。

对于springboot项目，一开始是用`java -jar `方式容器中启动，并作为容器的主进程。但在测试环境，经常代码逻辑可能有问题，java启动失败，进而触发k8s健康检查失败，进而不断重启容器。开发一直抱怨看不到“事故现场”。所以针对这种情况，直观的想法是 不让`java -jar` 作为容器的主进程。

## 一个容器一个进程？

容器中运行多进程，跟 one process per container 的理念相悖，我们就得探寻下来龙去脉了。从业界来说，虽然一个容器一个进程是官方推荐，但好像并不被大厂所遵守，以至于阿里甚至专门搞了一个PouchContainer 出来，[美团容器平台架构及容器技术实践](https://mp.weixin.qq.com/s/jojKe2MAe5btJCJ5013bgw)

stack exchange [Why it is recommended to run only one process in a container?](https://devops.stackexchange.com/questions/447/why-it-is-recommended-to-run-only-one-process-in-a-container) 有一系列回答

[Run Multiple Processes in a Container](https://runnable.com/docker/rails/run-multiple-processes-in-a-container) 也提了三个advantages

理由要找的话有很多，比较喜欢一个回答：As in most cases, it's not all-or-nothing. The guidance of "one process per container" stems from the idea that containers should serve a distinct purpose. For example, a container should not be both a web application and a Redis server.

There are cases where it makes sense to run multiple processes in a single container, as long as both processes support a single, modular function.

从笔者的实践感受来说

1. 除了业务进程，**你有没有比较刚的需求运行其他进程？**比如ssh 等。笔者在实践中，每个容器内还跑了一个监控进程，用来跟踪容器内的进程数据、以及执行一些异常诊断指令。
2. entrypoint 启动失败的可能性有多高，entrypoint 挂了会不停地重启，对集群带来的不良影响是否可控？

2020.11.1补充： [并非每个容器内部都能包含一个操作系统](https://mp.weixin.qq.com/s/ALTxkwAXBdKdQLMYJIMrLw)容器单进程并不是指容器里只能运行"一个"进程而是指容器没有管理多进程的能力。这是因为容器里PID=1的进程就是应用本身，其他的进程都是PID=1进程的子进程。

## init 进程

Linux 内核执行文件一般会放在 /boot 目录下，文件名类似 vmlinuz*。在内核完成了操作系统的各种初始化之后，这个程序需要执行的第一个**用户态**进程就是 init 进程。它直接或者间接创建了 Namespace 中的其他进程。


```c
init/main.c
/*
* We try each of these until one succeeds.
*
* The Bourne shell can be used instead of init if we are
* trying to recover a really broken machine.
*/
if (execute_command) {
        ret = run_init_process(execute_command);
        if (!ret)
                return 0;
        panic("Requested init %s failed (error %d).",
                execute_command, ret);
}
if (!try_to_run_init_process("/sbin/init") ||
    !try_to_run_init_process("/etc/init") ||
    !try_to_run_init_process("/bin/init") ||
    !try_to_run_init_process("/bin/sh"))
        return 0;
panic("No working init found.  Try passing init= option to kernel. "
        "See Linux Documentation/admin-guide/init.rst for guidance.");
```

管理孤儿进程。当一个子进程终止后，它首先会变成一个“失效(defunct)”的进程，也称为“僵尸（zombie）”进程，等待父进程或系统收回（通过wait/waitpid 函数）。如果父进程已经结束了，那些依然在运行中的子进程会成为“孤儿（orphaned）”进程。在Linux中Init进程(PID1)作为所有进程的父进程，会维护进程树的状态，一旦有某个子进程成为了“孤儿”进程后，init就会负责接管这个子进程。当一个子进程成为“僵尸”进程之后，如果其父进程已经结束，init会收割这些“僵尸”，释放PID资源。

僵尸进程（内存文件等都已释放，只留了一个stask_struct instance）如果不清理，就会消耗系统中的进程号资源，最坏会导致创建新进程。

社区 有一个容器init 项目tini

```c
int reap_zombies(const pid_t child_pid, int* const child_exitcode_ptr) {
        pid_t current_pid;
        int current_status;
        while (1) {
                current_pid = waitpid(-1, &current_status, WNOHANG);

                switch (current_pid) {
                        case -1:
                                if (errno == ECHILD) {
                                        PRINT_TRACE("No child to wait");
                                        break;
                                }

…
```

linux 信号机制

1. 进程在收到信号后，可以选择
    1. 忽略，对这个信号不做任何处理，但对特权信号 SIGKILL 和 SIGSTOP 例外，不能忽略和捕获（注册signal handler 也会报错），只能采取默认行为——终止。
    2. 捕获，用户进程可以注册自己针对这个信号的 handler
    3. Default，，Linux 为每个信号都定义了一个默认的行为，包含终止、忽略等，SIGKILL 和 SIGSTOP 的默认行为都是终止。
2. SIGTERM 是kill 默认发出的  `kill pid` = `kill -SIGTERM pid`
3. init 进程的特殊性，在每个 Namespace 的 init 进程建立的时候，就会打上 SIGNAL_UNKILLABLE 这个标签，1 号进程永远不会响应 SIGKILL 和 SIGSTOP 这两个特权信号。对于其他的信号，如果用户自己注册了 handler，1 号进程可以响应。
3. 容器的一般使用专用的init 进程（注册并实现非SIGKILL 和 SIGSTOP  信号的处理函数）， 负责转发信号到所有的子进程，并且回收僵尸进程。 docker原生提供的init进程为tini 

```sh
docker run -d --init ubuntu:14.04 bash -c "cd /home/ && sleep 100
oot@24cc26039c4d:/# ps -ef
UID PID PPID C STIME TTY TIME CMD
root 1 0 2 14:50 ? 00:00:00 /sbin/docker-init -- bash -c cd /home/ && sleep 100
root 6 1 0 14:50 ? 00:00:00 bash -c cd /home/ && sleep 100
root 7 6 0 14:50 ? 00:00:00 sleep 100
```

## 以进程管理工具作为entrypoint

[理解Docker容器的进程管理](https://yq.aliyun.com/articles/5545)docker stop  对PID1进程 的要求

1. 支持管理运行过程中可能产生的僵尸/孤儿进程
2. 容器的PID1进程需要能够正确的处理SIGTERM信号来支持优雅退出，如果容器中包含多个进程，需要PID1进程能够正确的传播SIGTERM信号来结束所有的子进程之后再退出。


综上，如果一个容器有多个进程，可选的实践方式为：

1. 多个进程关系对等，由一个init 进程管理，比如supervisor、systemd
2. 一个进程（A）作为主进程，拉起另一个进程（B）

	* A 先挂，因为容器的生命周期与 主进程一致，则进程B 也会被kill 结束
	* B 先挂，则要看A 是否具备僵尸进程的处理能力（大部分不具备）。若A 不具备，B 成为僵尸进程，容器存续期间，僵尸进程一致存在。
	* A 通常不支持 SIGTERM

所以第二种方案通常不可取，对于第一种方案，则有init 进程的选型问题

||僵尸进程回收|处理SIGTERM信号|alpine 安装大小|专用镜像|备注|
|---|---|---|---|---|---|
|sh/bash|支持|不支持|0m||脚本中可以使用exec 顶替掉sh/bash 自身|
|Supervisor|待确认|支持|79m|要求管理的进程为前台进程，后台进程管不了|
|runit|待确认|支持|31m| [phusion/baseimage-docker](https://github.com/phusion/baseimage-docker)|要求管理的进程为前台进程，后台进程管不了|
|s6|||33m||
|Systemd|支持|支持|alpine没有Systemd||systemd跑不跑前台只是说的systemctl能不能控制，那些不被控制的，关闭过程systemd也会负责回收的<br>对系统特权有要求<br>不会透传环境变量|

## 进程管理工具的选择

### 自定义脚本

官方 [Run multiple services in a container](https://docs.docker.com/config/containers/multi-service_container/)

### runit

[Run Multiple Processes in a Container](https://runnable.com/docker/rails/run-multiple-processes-in-a-container)

A fully­ powered Linux environment typically includes an ​init​ process that spawns and supervises other processes, such as system daemons. The command defined in the CMD instruction of the Dockerfile is the only process launched inside the Docker container, so ​system daemons do not start automatically, even if properly installed.

[runit - a UNIX init scheme with service supervision](http://smarden.org/runit/)



Dockerfile

```Dockerfile
FROM phusion/passenger-­ruby22
...
#install custom bootstrap script as runit service
COPY myapp/start.sh /etc/service/myapp/run
```
	
```sh	
#// myapp/start.sh
#!/bin/sh
exec command
```

	
在这个Dockerfile 中，CMD 继承自 base image。 将`myapp/start.sh` 拷贝到 容器的 `/etc/service/myapp/run`	文件中即可 被runit 管理，runit 会管理 `/etc/service/` 下的应用（目录可配置），即 Each service is associated with a service directory

这里要注意：记得通过exec 方式执行 command，这涉及到 [shell，exec，source执行脚本的区别](https://www.jianshu.com/p/dd7956aec097)


[Using runit for maintaining services](https://debian-administration.org/article/697/Using_runit_for_maintaining_services) 

[runit：进程管理工具runit](https://www.lijiaocn.com/%E6%8A%80%E5%B7%A7/2017/08/08/linux-tool-runit.html)

||作用|备注|
|---|---|---|
|runit-init|runit-init is the first process the kernel starts. If runit-init is started as process no 1, it runs and replaces itself with runit|
|runsvdir| starts and monitors a collection of runsv processes|当runsvdir检查到`/etc/service`目录下包含一个新的目录时，runsvdir会启动一个runsv进程来执行和监控run脚本。|
|runsvchdir|change services directory of runsvdir|
|sv|control and manage services monitored by runsv|`sv status /etc/service/test`<br>`sv stop /etc/service/test`<br>`sv restart /etc/service/test`|
|runsv| starts and monitors a service and optionally an appendant log service|
|chpst|runs a program with a changed process state|run脚本默认被root用户执行，通过chpst可以将run配置为普通用户来执行。|
|utmpset| logout a line from utmp and wtmp file|
|svlogd|runit’s service logging daemon|

runit 工作原理

![](/public/upload/docker/runit.png)

### systemd

[CHAPTER 3. USING SYSTEMD WITH CONTAINERS](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux_atomic_host/7/html/managing_containers/using_systemd_with_containers)

[Running Docker Containers with Systemd](https://container-solutions.com/running-docker-containers-with-systemd/)

[Do you need to execute more than one process per container?](https://gomex.me/2018/07/21/do-you-need-to-execute-more-than-one-process-per-container/)

### supervisor

官方 [Run multiple services in a container](https://docs.docker.com/config/containers/multi-service_container/)

[Admatic Tech Blog: Starting Multiple Services inside a Container with Supervisord](https://medium.com/@SaravSun/admatic-tech-blog-starting-multiple-services-inside-a-container-with-supervisord-16e3beb55916)

使用 

supervisord.conf

	[supervisord]
	nodaemon=true
	logfile=/dev/stdout
	loglevel=debug
	logfile_maxbytes=0
	
	[program:pinggoogle]
	command=ping admatic.in
	autostart=true
	autorestart=true
	startsecs=5
	stdout_logfile=NONE
	stderr_logfile=NONE
	
Dockerfile
	
	FROM ubuntu
	...
	COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
	...
	CMD ["/usr/bin/supervisord"]
## 其它

### Docker-friendliness image

与其在init 进程工具的选型上挣扎，是否有更有魄力的工具呢？

1. docker 原生支持多进程，比如阿里的 pouch
2. 原生支持多进程的 镜像

github 有一个 [phusion/baseimage-docker](https://github.com/phusion/baseimage-docker) 笔者2018.11.7 看到时，有6848个star。 该镜像有几个优点：

1. Modifications for Docker-friendliness.
2. Administration tools that are especially useful in the context of Docker.
3. Mechanisms for easily running multiple processes, without violating the Docker philosophy.  具体的说，The Docker developers advocate running a single logical service inside a single container. But we are not disputing that. Baseimage-docker advocates running multiple OS processes inside a single container, and a single logical service can consist of multiple OS processes. 


什么叫 ubuntu 对 Docker-friendliness？（待体会）

1. multi-user
2. multi-process

### 和ssh的是是非非

2020.07.17 补充：随着web console 工具（底层由kubectl exec支持）不及，ssh 渐渐没有必要了。

2018.12.01 补充： [ssh连接远程主机执行脚本的环境变量问题](http://feihu.me/blog/2014/env-problem-when-ssh-executing-command-on-remote/)

背景：

1. 容器启动时会运行sshd，所以可以ssh 到容器
2. 镜像dockerfile中 包含`ENV PATH=${PATH}:/usr/local/jdk/bin`
2. `docker exec -it container bash` 可以看到 PATH 环境变量中包含 `/usr/local/jdk/bin`
3. `ssh root@xxx` 到容器内，观察 PATH 环境变量，则不包含  `/usr/local/jdk/bin`

这个问题涉及到 bash的四种模式

1. 通过ssh登陆到远程主机  属于bash 模式的一种：login + interactive
2. 不同的模式，启动shell时会去查找并加载 不同而配置文件，比如`/etc/bash.bashrc`、`~/.bashrc` 、`/etc/profile` 等
3. login + interactive 模式启动shell时会 第一加载`/etc/profile`
4. `/etc/profile` 文件内容默认有一句 `export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`

所以 `docker exec `可以看到 正确的PATH 环境变量值，而ssh 到容器不可以，解决方法之一就是 制作镜像时 向`/etc/profile` 追加 一个export 命令


