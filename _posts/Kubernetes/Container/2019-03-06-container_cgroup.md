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

Kubernetes 采用内存工作集(workingset)来监控和管理容器的内存使用，当容器内存使用量超过了设置的内存限制或者节点出现内存压力时，kubernetes 会根据 workingset 来决定是否驱逐或者杀死容器。内存工作集计算公式：Workingset = 匿名内存 + active_file。匿名内存一般是程序通过 new/malloc/mmap 方式分配，而 active_file 是进程读写文件引入，程序一般对这类内存使用存在不透明情况，经常容易出问题。客户通过容器监控发现其 K8s 集群中某个 pod 的 Workingset 内存持续走高，无法进一步定位究竟是什么原因导致的 Workingset 内存使用高。

隐式内存占用指业务运行中间接产生的系统内存消耗，未体现在应用进程的常规指标（如 RSS/PSS）中，因而难以被监控或业务感知。尽管不表现为“显式”使用，却真实占用物理内存。由于缺乏有效暴露与归因机制，这类内存往往在系统层面持续累积，最终导致可用内存下降、频繁回收甚至 OOM。
1. 文件缓存(filecache)高。filecache 用来提升文件访问性能，并且理论上可以在内存不足时被回收，但高 filecache 在生产环境中也引发了诸多问题：
    1. filecache 回收时，直接影响业务响应时间（RT），在高并发环境中，这种延时尤为显著，可能导致用户体验急剧下降。例如，在电商网站的高峰购物时段，filecache 的回收延时可能会引起用户购物和付款卡顿，直接影响用户体验。
    2. 在 Kubernetes（k8s）环境中，workingset 包含活跃的文件缓存，如果这些活跃缓存过高，会直接影响 K8s 的调度决策，导致容器无法被高效调度到合适的节点，从而影响应用的可用性和整体的资源利用率。
2. SReclaimable 高SReclaimable 是内核维护的可回收缓存，虽不计入用户进程内存统计，但受应用行为（如频繁文件操作、临时文件创建/删除）显著影响。尽管系统可在内存压力下回收它，但回收过程涉及复杂的锁竞争与同步，常引发较高的 CPU 开销和延迟抖动。SReclaimable 长期高位会占用大量物理内存，却因监控通常只关注进程 RSS 或容器内存而被忽视，造成内存压力误判。
3. memory group 残留cgroup 与 namespace 是容器运行时的核心机制。在高频调度场景（如大规模微服务或批处理系统）中，若清理不及时或内核释放延迟，易引发 cgroup 泄漏——即无关联进程的 cgroup 目录未被回收。这不仅占用内核内存，还会引起内存统计误差，导致监控异常、延时抖动等问题。
4. 内存不足，却找不到去哪儿了当系统内存紧张时，常规工具（如 top）难以揭示真实内存去向——它们无法观测内核驱动（如 GPU、网卡、RDMA）直接分配的内存。在 AI 训练等高性能场景中，GPU 驱动会大量申请  memory、DMA buffer 等系统内存用于显存映射与通信，但这些关键开销对用户“不可见”。运维人员只能看到 MemAvailable 骤降甚至耗尽，却无法定位具体任务、机制或判断是否存在泄漏。

以该场景为例，核心问题是：哪些进程在读写哪些文件，导致缓存堆积？在四种隐式内存占用场景中，文件缓存（page cache）过高最为常见。以该场景为例，核心问题是：哪些进程在读写哪些文件，导致缓存堆积？
1. 由 page 定位 inode：通过 page->mapping 和 index 找到其所属的 `address\_space` 和文件 inode；
2. 由 inode 还原文件路径：遍历 dentry 缓存，在挂载命名空间中重建完整路径（如 `/data/model/xxx.bin`）。

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