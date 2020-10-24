---

layout: post
title: 容器狂占cpu怎么办？
category: 技术
tags: Container
keywords: container cpu

---

## 简介

* TOC
{:toc}

笔者曾经碰到一个现象， 物理机load average 达到120，第一号进程的%CPU 达到了 1091，比第二名大了50倍，导致整个服务器非常卡，本文尝试分析和解决这个问题

[Docker: 限制容器可用的 CPU](https://www.cnblogs.com/sparkdev/p/8052522.html)

## 如何感知某个项目占用了过多的cpu

[docker stats](https://docs.docker.com/engine/reference/commandline/stats/) returns a live data stream for running containers.

`docker stats` 命令输出

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

## CFS 的发展

在 Linux 里面，进程大概可以分成两种：实时进程和 普通进程。每个 CPU 都有自己的 struct rq 结构，其用于描述在此 CPU 上所运行的所有进程，其包括一个实时进程队列 rt_rq 和一个 CFS 运行队列 cfs_rq，在调度时，调度器首先会先去实时进程队列找是否有实时进程需要运行，如果没有才会去 CFS 运行队列找是否有进程需要运行。

cgroup 是 调度器 暴露给外界操作 的接口，对于 进程cpu 相关的资源配置 RT（realtime调度器） 和CFS 均有实现。本文主要讲  CFS，CFS 也是在不断发展的。

### CFS 基于虚拟运行时间的调度

[What is the concept of vruntime in CFS](https://stackoverflow.com/questions/19181834/what-is-the-concept-of-vruntime-in-cfs/19193619)vruntime is a measure of the "runtime" of the thread - the amount of time it has spent on the processor. The whole point of CFS is to be fair to all; hence, the algo kind of boils down to a simple thing: (among the tasks on a given runqueue) the task with the lowest vruntime is the task that most deserves to run, hence select it as 'next'. CFS（完全公平调度器）是Linux内核2.6.23版本开始采用的进程调度器，具体的细节蛮复杂的，整体来说是 每次选择vruntime 较少的进程来执行。

vruntime就是根据权重、优先级（留给上层介入的配置）等将实际运行时间标准化。在内核中通过prio_to_weight数组进行nice值和权重的转换。

```c
static const int prio_to_weight[40] = {
 /* -20 */     88761,     71755,     56483,     46273,     36291,
 /* -15 */     29154,     23254,     18705,     14949,     11916,
 /* -10 */      9548,      7620,      6100,      4904,      3906,
 /*  -5 */      3121,      2501,      1991,      1586,      1277,
 /*   0 */      1024,       820,       655,       526,       423,
 /*   5 */       335,       272,       215,       172,       137,
 /*  10 */       110,        87,        70,        56,        45,
 /*  15 */        36,        29,        23,        18,        15,
};
```
NICE_0_LOAD = 1024

`虚拟运行时间 vruntime += 实际运行时间 delta_exec * NICE_0_LOAD/ 权重`

### CFS hard limits

对一个进程配置 cpu-shares 并不能绝对限制 进程对cpu 的使用（如果机器很闲，进程的vruntime 虽然很大，但依然可以一直调度执行），也不能预估 进程使用了多少cpu 资源。这对于 桌面用户无所谓，但对于企业用户或者云厂商来说就很关键了。所以CFS 后续加入了 bandwith control。 将 进程的 cpu-period,cpu-quota 数据纳入到调度逻辑中。

具体逻辑比较复杂，有兴趣可以看[CPU bandwidth control for CFS](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/36669.pdf)

## 容器 与 CFS

[Kubernetes中的CPU节流：事后分析](https://mp.weixin.qq.com/s/2lcX3-QBYFP5UnPw-b-Rvg)几乎所有的容器编排器都依赖于内核控制组（cgroup）机制来管理资源约束。在容器编排器中设置硬CPU限制后，内核将使用完全公平调度器（CFS）Cgroup带宽控制来实施这些限制。CFS-Cgroup带宽控制机制使用两个设置来管理CPU分配：配额（quota）和周期（period）。当应用程序在给定时间段内使用完其分配的CPU配额时，此时它将受到CPU节流，直到下一个周期才能被调度。

[docker CPU文档](https://docs.docker.com/config/containers/resource_constraints/)By default, each container’s access to the host machine’s CPU cycles is unlimited. You can set various constraints to limit a given container’s access to the host machine’s CPU cycles. Most users use and configure the default CFS scheduler. In Docker 1.13 and higher, you can also configure the realtime scheduler.The CFS is the Linux kernel CPU scheduler for normal Linux processes. **Several runtime flags allow you to configure the amount of access to CPU resources your container has**. When you use these settings, Docker modifies the settings for the container’s cgroup on the host machine.  默认情况，容器可以任意使用 host 上的cpu cycles，**配的不是容器，配的是 CFS**，dockers仅仅是帮你设置了下参数而已。

||示例|默认值|
|---|---|---|
|--cpuset-cpus|`--cpuset-cpus="1,3"`|
|--cpu-shares|默认为1024|绝对值没有意义，只有和另一个进程对比起来才有意义|
|--cpu-period|100ms|Specify the CPU CFS scheduler period，和cpu-quota结合使用|
|--cpu-quota||Impose a CPU CFS quota on the container，和cpu-period结合使用|
|--cpus|1|表示quota/period=1<br>docker 1.13支持支持，替换cpu-period和cpu-quota|

现在的多核系统中每个核心都有自己的缓存，如果频繁的调度进程在不同的核心上执行势必会带来缓存失效等开销。--cpuset-cpus 可以让容器始终在一个或某几个 CPU 上运行。--cpuset-cpus 选项的一个缺点是必须指定 CPU 在操作系统中的编号，这对于动态调度的环境(无法预测容器会在哪些主机上运行，只能通过程序动态的检测系统中的 CPU 编号，并生成 docker run 命令)会带来一些不便。

If you have 1 CPU, each of the following commands guarantees the container at most 50% of the CPU every second.
1. Docker 1.13 and higher:`docker run -it --cpus=".5" ubuntu /bin/bash`
2. Docker 1.12 and lower: `docker run -it --cpu-period=100000 --cpu-quota=50000 ubuntu /bin/bash`

shares值即CFS中每个进程的(准确的说是调度实体)权重(weight/load)，shares的大小决定了在一个CFS调度周期中，进程占用的比例，比如进程A的shares是1024，B的shares是512，假设调度周期为3秒，那么A将只用2秒，B将使用1秒

## Kubernetes

[Understanding resource limits in kubernetes: cpu time](https://medium.com/@betz.mark/understanding-resource-limits-in-kubernetes-cpu-time-9eff74d3161b)

```yaml
resources:
  requests:
    memory: 50Mi
    cpu: 50m
  limits:
    memory: 100Mi
    cpu: 100m
```
单位后缀 m 表示千分之一核，也就是说 1 Core = 1000m。so this resources object specifies that the container process needs 50/1000 of a core (5%) and is allowed to use at most 100/1000 of a core (10%).同样，2000m 表示两个完整的 CPU 核心，你也可以写成 2 或者 2.0。

cpu requests and cpu limits are implemented using two separate control systems.

|Kubernetes|docker|cpu,cpuacct cgroup|
|---|---|---|
|request=50m|cpu-shares=51|cpu.shares|
|limit=100m|cpu-period=100000,cpu-quota=10000|cpu.cfs_period_us,cpu.cfs_quota_us|

request并不能限定 容器使用cpu 的上界

request 和 limit 

1. if you set a limit but don’t set a request kubernetes will default the request to the limit. This can be fine if you have very good knowledge of how much cpu time your workload requires. 
2. How about setting a request with no limit? In this case kubernetes is able to accurately schedule your pod, and the kernel will make sure it gets at least the number of shares asked for, but your process will not be prevented from using more than the amount of cpu requested, which will be stolen from other process’s cpu shares when available. 
3. Setting neither a request nor a limit is the worst case scenario: the scheduler has no idea what the container needs, and the process’s use of cpu shares is unbounded, which may affect the node adversely. 

### request

```
$ kubectl run limit-test --image=busybox --requests "cpu=50m" --command -- /bin/sh -c "while true; do sleep 2; done"
deployment.apps "limit-test" created
$ docker ps | grep busy | cut -d' ' -f1
f2321226620e
$ docker inspect f2321226620e --format '{{.HostConfig.CpuShares}}'
51
```

Why 51, and not 50? The cpu control group and docker both divide a core into 1024 shares, whereas kubernetes divides it into 1000.

### limit
```
$ kubectl run limit-test --image=busybox --requests "cpu=50m" --limits "cpu=100m" --command -- /bin/sh -c "while true; do
sleep 2; done"
deployment.apps "limit-test" created
$ kubectl get pods limit-test-5b4fb64549-qpd4n -o=jsonpath='{.spec.containers[0].resources}'
map[limits:map[cpu:100m] requests:map[cpu:50m]]
$ docker ps | grep busy | cut -d' ' -f1
f2321226620e
$ docker inspect 472abbce32a5 --format '{{.HostConfig.CpuShares}} {{.HostConfig.CpuQuota}} {{.HostConfig.CpuPeriod}}'
51 10000 100000
```

## 如何避免系统被应用拖垮

使用 k8s 的cpu limit 机制，在实际落地过程中，容器平台提供几种固定的规格，单个实例的资源配置 能够支持正常跑即可，项目的服务能力通过横向扩展来解决。






