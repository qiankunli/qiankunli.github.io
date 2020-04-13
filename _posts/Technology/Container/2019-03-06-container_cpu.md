---

layout: post
title: 容器狂占cpu怎么办？
category: 技术
tags: Container
keywords: container cpu

---

## 简介（未完成）

* TOC
{:toc}

笔者曾经碰到一个现象， 物理机load average 达到120，第一号进程的%CPU 达到了 1091，比第二名大了50倍，导致整个服务器非常卡，本文尝试分析和解决这个问题

[Docker: 限制容器可用的 CPU](https://www.cnblogs.com/sparkdev/p/8052522.html)

## 如何感知某个项目占用了过多的cpu



### docker

[docker stats](https://docs.docker.com/engine/reference/commandline/stats/) returns a live data stream for running containers.

docker stats 命令输出

    CONTAINER ID        NAME                                         CPU %               MEM USAGE / LIMIT     MEM %               NET I/O             BLOCK I/O           PIDS
    4aeb15578094        mesos-19ba2ecd-7a98-4e92-beed-c132b063578d   0.21%               376.4MiB / 1GiB       36.75%              3.65MB / 1.68MB     119kB / 90.1kB      174
    77747f26dff4        mesos-e3d34892-8af6-4ab7-a649-0a4b424ccd04   0.31%               752.5MiB / 800MiB     94.06%              11.9MB / 3.06MB     86.1MB / 47MB       132
    d64e482d2843        mesos-705b5dc6-7169-42e8-a143-6a7dc2e32600   0.18%               680.5MiB / 800MiB     85.06%              43.1MB / 17.1MB     194MB / 228MB       196
    808a4bd888fb        mesos-65c9d5a6-3967-4a4a-9834-b83ae8c033be   1.81%               1.45GiB / 2GiB        72.50%              1.32GB / 1.83GB     8.36MB / 19.8MB     2392


1. CPU % 体现了 quota/period 值
2. MEM USAGE / LIMIT 反映了内存占用
3. NET I/O 反映了进出带宽
4. BLOCK I/O 反映了磁盘带宽，The amount of data the container has read to and written from block devices on the host ，貌似是一个累计值，但可以部分反映 项目对磁盘的写程度，有助于解决[容器狂打日志怎么办？](http://qiankunli.github.io/2019/03/05/container_log.html)
5. PID 反映了对应的进程号，也列出了进程id 与容器id的关系。根据pid 查询容器 id `docker stats --no-stream | grep 1169`

## cpu 隔离在不同系统的不同概念

||示例|备注|
|---|---|---|
|--cpu-period|和cpu-quota结合使用|
|--cpu-quota|和cpu-period结合使用|
|--cpus|1|表示quota/period=1|貌似是docker 提出的概念|
|--cpuset-cpus|`--cpuset-cpus="1,3"`|
|--cpu-share|默认为1024|绝对值没有意义，只有和另一个进程对比起来才有意义|

现在的多核系统中每个核心都有自己的缓存，如果频繁的调度进程在不同的核心上执行势必会带来缓存失效等开销。--cpuset-cpus 可以让容器始终在一个或某几个 CPU 上运行。--cpuset-cpus 选项的一个缺点是必须指定 CPU 在操作系统中的编号，这对于动态调度的环境(无法预测容器会在哪些主机上运行，只能通过程序动态的检测系统中的 CPU 编号，并生成 docker run 命令)会带来一些不便。

### CPU Resources in Docker, Mesos and Marathon

[CPU Resources in Docker, Mesos and Marathon](https://zcox.wordpress.com/2014/09/17/cpu-resources-in-docker-mesos-and-marathon/) 几个要点（假设物理机有8个核心）

1. 即便marathon cpus 配置为0.1 ，容器依然可以访问物理机所有的8个核心，即cpus和物理机核心没有对应关系
2. mesos convert the cpus value into a value for Docker’s `--cpu-shares` setting,which according to the Docker documentation is just a **priority weight** for that process relative to all others on the machine. An application run with cpus=2 should receive twice the priority as one using cpus=1.
3.  This is another effect that the cpus parameter has: it specifies the CPU capacity used up by the application.Maybe cpu-capacity or cpu-weight would be more descriptive
4. 因为cpus 还作为CPU capacity 的描述载体，它以cpu 核心数作为上限。你如何配置marathon app 的cpus，完全取决于你打算让一个机器跑多少app，以及它相对其它app的重要性。 


## 如何避免系统被应用拖垮

1. 监控报警，先感知到再说
2. 当发现cpu-share 不足以约束应用时，改用 docker 本身的cpus 概念进一步限制




