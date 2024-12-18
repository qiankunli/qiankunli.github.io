---

layout: post
title: 容器狂占资源怎么办？
category: 技术
tags: Container
keywords: container cpu

---

## 简介

* TOC
{:toc}

笔者曾经碰到一个现象， 物理机load average 达到120，第一号进程的%CPU 达到了 1091，比第二名大了50倍，导致整个服务器非常卡，本文尝试分析和解决这个问题

[Docker: 限制容器可用的 CPU](https://www.cnblogs.com/sparkdev/p/8052522.html)

[为什么不建议把数据库部署在Docker容器内](https://mp.weixin.qq.com/s/WetiMHwBEHmGzvXY6Pb-jQ)资源隔离方面，Docker确实不如虚拟机KVM，Docker是利用Cgroup实现资源限制的，**只能限制资源消耗的最大值，而不能隔绝其他程序占用自己的资源**。如果其他应用过渡占用物理机资源，将会影响容器里MySQL的读写效率。

## 内存（待补充）

[K8s里我的容器到底用了多少内存？](https://mp.weixin.qq.com/s/iN3tMmJ2y_nUa6ATInyP1A)

隔离性一般没啥问题，但为了提高资源利用率，一般会对节点进行超卖，此时任务之间就可能相互影响。 

[Koordinator 1.0 正式发布：业界首个生产可用、面向规模场景的开源混部系统](https://mp.weixin.qq.com/s/wox_TMvB4caziOVU5xSTKg)Koordinator 基于 Resource Director Technology (RDT, 资源导向技术) ，控制由不同优先级的工作负载可以使用的末级缓存（服务器上通常为 L3 缓存）。RDT 还使用内存带宽分配 (MBA) 功能来控制工作负载可以使用的内存带宽。这样可以隔离工作负载使用的 L3 缓存和内存带宽，确保高优先级工作负载的服务质量，并提高整体资源利用率。

## 磁盘

`/sys/fs/cgroup/blkio/`
1. blkio.throttle.read_iops_device, 磁盘读取 IOPS 限制
2. blkio.throttle.read_bps_device, 磁盘读取吞吐量限制
3. blkio.throttle.write_iops_device, 磁盘写入 IOPS 限制
4. blkio.throttle.write_bps_device, 磁盘写入吞吐量限制

OverlayFS 自己没有专门的特性，可以限制文件数据写入量。可以依靠底层文件系统的 Quota 特性（XFS Quota 的 Project 模式）来限制 OverlayFS 的 upperdir 目录的大小，这样就能实现限制容器写磁盘的目的。

在 Cgroup v1 中有 blkio 子系统，它可以来限制磁盘的 I/O。
1. 衡量磁盘性能的两个常见的指标 IOPS 和吞吐量（Throughput）。IOPS 是 Input/Output Operations Per Second 的简称，也就是每秒钟磁盘读写的次数，这个数值越大表示性能越好。吞吐量（Throughput）是指每秒钟磁盘中数据的读取量，一般以 MB/s 为单位。这个读取量可以叫作吞吐量，有时候也被称为带宽（Bandwidth）。它们的关系大概是：吞吐量 = 数据块大小 *IOPS。
2. Linux 文件 I/O 模式
    1. Direct I/O 
    2. Buffered I/O, blkio 在 Cgroups v1 里不能限制 Buffered I/O

Cgroup v1 的一个整体结构，每一个子系统都是独立的，资源的限制只能在子系统中发生。比如pid可以分别属于 memory Cgroup 和 blkio Cgroup。但是在 blkio Cgroup 对进程 pid 做磁盘 I/O 做限制的时候，blkio 子系统是不会去关心 pid 用了哪些内存，这些内存是不是属于 Page Cache，而这些 Page Cache 的页面在刷入磁盘的时候，产生的 I/O 也不会被计算到进程 pid 上面。**Cgroup v2 相比 Cgroup v1 做的最大的变动就是一个进程属于一个控制组，而每个控制组里可以定义自己需要的多个子系统**。Cgroup v2 对进程 pid 的磁盘 I/O 做限制的时候，就可以考虑到进程 pid 写入到 Page Cache 内存的页面了，这样 buffered I/O 的磁盘限速就实现了。但带来的问题是：在对容器做 Memory Cgroup 限制内存大小的时候，如果仅考虑容器中进程实际使用的内存量，未考虑容器中程序 I/O 的量，则写数据到 Page Cache 的时候，需要不断地去释放原有的页面，造成容器中 Buffered I/O write() 不稳定。

[Linux Disk Quota实践](https://mp.weixin.qq.com/s/c5tbrSEinYVNl9anIvyMTg)

[云原生环境中的磁盘IO性能问题排查与解决](https://mp.weixin.qq.com/s/RrTjhSJOviiINsy-DURV2A)

## 网络

Network Namespace 隔离了哪些资源

1. 网络设备，比如 lo，eth0 等网络设备。
2. IPv4 和 IPv6 协议栈。IP 层以及上面的 TCP 和 UPD 协议栈也是每个 Namespace 独立工作的。它们的相关参数也是每个 Namespace 独立的，这些参数大多数都在 `/proc/sys/net/` 目录下面，同时也包括了 TCP 和 UPD 的 port 资源。
3. IP 路由表
4. iptables 规则

发现容器网络不通怎么办？容器中继续 ping 外网的 IP ，然后在容器的 eth0 ，容器外的 veth，docker0，宿主机的 eth0 这一条数据包的路径上运行 tcpdump。这样就可以查到，到底在哪个设备接口上没有收到 ping 的 icmp 包。

![](/public/upload/container/container_practice.jpg)

在 v1.5.0 版本中，Koordinator 联动 Terway 社区提供了网络 QoS 能力。Terway QoS[3] 的诞生是为了解决混部场景下的网络带宽争抢问题，它支持按单个 Pod 或 QoS 类型进行带宽限制，与其他方案相比具备以下优势：
1. 支持按业务类型限制带宽，适用于多种业务混部的场景。
2. 支持动态调整 Pod 带宽限制。
3. 提供整机带宽限制，支持多网卡，支持容器网络和 HostNetwork Pod 的带宽限制。
在混部场景中，我们希望在线业务具有最大的带宽保障，以避免争抢；在空闲时，离线业务也可以充分利用所有带宽资源。因此，用户可以为业务流量定义为 3 个优先级，从高到低依次为：L0、L1、L2。我们定义争用场景为：当 L0+L1+L2 的总流量超过整机带宽时。L0 的最大带宽根据 L1 和 L2 的实时流量动态调整。它可以高至整机带宽，低至“整机带宽 - L1 最小带宽 - L2 最小带宽”。在任何情况下，L1 和 L2 的带宽都不会超过各自的上限。在争用场景中，L1 和 L2 的带宽不会低于各自的下限。在争用场景中，带宽将按 L2、L1 和 L0 的顺序进行限制。由于 Terway QoS 只有三个优先级，因此只能设置 LS 和 BE 的全机带宽限制，其余 L0 部分根据整机的带宽上限计算。