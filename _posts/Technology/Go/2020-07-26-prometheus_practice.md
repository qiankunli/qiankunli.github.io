---

layout: post
title: 《Prometheus监控实战》笔记
category: 技术
tags: Go
keywords: Prometheus 实战

---

## 前言

* TOC
{:toc}

## metric 种类

1. counter（计数器），始终增加，比如http请求数、下单数
2. gauge（测量仪），当期值的一次快照测量，可增可减。比如磁盘使用率、当前同时在线用户数
3. Histogram（直方图），通过分桶方式统计样本分布。**一个histogram会生成三个指标，分别是_count，_sum，_bucket**。举个铁球做例子，我们一共有1000个大小不同的铁球，质量从1kg-100kg不等，假设我分5个桶，每个桶存储不同质量的铁球，第一个桶0-20kg，第二个20-40kg，后面依此。然后1000个铁球，就是histogram的count，而1000个铁球的总质量就是histogram的sum，那么bucket就是那五个桶，当然几个桶，桶的规则怎样都是可以设计的，这五个桶每个的0-20,20-40…就是横坐标，而每个桶中的个数，就是纵坐标。根据这些数据就可以形成一个直方图。
4. Summary（汇总），根据样本统计出百分位，比如客户端计算。**summary也会产生三个指标，分别是_count，_sum，和{quantile}** ,count和sum与histogram的概念相同，quantile的含义是分位数，我们都知道中位数，那么中位数其实就是一个五分quantile，而summary可以在定义时指定很多分位数，如五分数，九分数，九九分数。九分数的概念就是比这个数小的数占百分之九十。

4个黄金指标可以在服务级别帮助衡量终端用户体验、服务中断、业务影响等层面的问题。
1. 延迟：服务请求所需时间。
2. 通讯量：监控当前系统的流量，用于衡量服务的容量需求。例如，在HTTP REST API中, 流量通常是每秒HTTP请求数；
3. 错误：监控当前系统所有发生的错误请求，衡量当前系统错误发生的速率。
4. 饱和度：衡量当前服务的饱和度。比如，“磁盘是否可能在4个小时候就满了”。

这四个指标并不是唯一的系统性能或状况的衡量标准，系统可以简单分为两类

1. 资源提供系统 - 对外提供简单的资源，比如CPU（计算资源），存储，网络带宽。 针对资源提供型系统，有一个更简单直观的USE标准
    1. Utilization - 往往体现为资源使用的百分比
    2. Saturation - 资源使用的饱和度或过载程度，**过载的系统往往意味着系统需要辅助的排队系统完成相关任务**。这个和上面的Utilization指标有一定的关系但衡量的是不同的状况，以CPU为例，Utilization往往是CPU的使用百分比而Saturation则是当前等待调度CPU的县城或进程队列长度
    3. Errors - 这个可能是使用资源的出错率或出错数量，比如网络的丢包率或误码率等等
2. 服务提供系统 - 对外提供更高层次与业务相关的任务处理能力，比如订票，购物等等。针对服务型系统，则往往用RED方式进行衡量
    1. Rate - 单位时间内完成服务请求的能力
    2. Errors - 错误率或错误数量：单位时间内服务出错的比列或数量
    3. Duration - 平均单次服务的持续时长（或用户得到服务响应的时延）


**Prometheus 提供的许多exporter 或者直接提供上述metric，或者通过计算可以得到上述metric**。或者反过来说，这些原则指导了exporter 去暴露哪些metric。

## node-exporter

node-exporter提供了近1000个指标，以`node_` 为前缀，包括node_cpu_*,node_memory_*, node_filesystem_*/node_disk_*, node_network_* 等。

node-exporter 控制启用哪些[收集器](https://github.com/prometheus/node_exporter#collectors)，许多收集器默认都是启用的。它们的状态要么是启用要 么是禁用，你可以通过使用no-前缀来修改状态。

1. textfile收集器，它允许我们暴露自定义指标。这些自定义指标可能是批处理或cron作业等无法抓取的，可能是没有exporter的源，甚至可能是为主机提供上下文的静态指标。收集器通过扫描指定目录中的文件，提取所有格式为Prometheus指标的字符串，然后暴露它们以 便抓取。比如

    ```sh
    # 创建一个目录来保存指标定义文件
    $ mkdir -p /var/lib/node_exporter/textfile_collector
    # 指标在以.prom 结尾的文件内定义，并且使用Prometheus 特定文本格式
    echo 'metadata{role="docker_server",datacenter="NJ"} 1' > /var/lib/node_exporter/textfile_collector/metadata.prom
    ```
2. systemd 收集器， 记录systemd中的服务和系统状态。支持将特定服务列入白名单（比如docker），只收集以下服务的指标

除了通过本地配置来控制 Node Exporter在本地运行哪些收集器之外，Prometheus还提供了一种方式来限制收集器从服务器端实 际抓取的数据

```yaml
scrap_configs:
  - job_name : "node"
    static_configs:
    - targets: ['192.168.1.1:9100']
    params: 
      collect[]:
        - cpu
        - meminfo
        - diskstats
        ...
```

cpu 相关metric  node_cpu_seconds_total ，数据从`/proc/stat`中抽取，以计数的形式告诉我们每个cpu在每种模式下使用了多少秒，包含标签

1. cpu
2. instance
3. job
4. mode  包括idle/iowait/irp/nice/system/user 等

内存相关 metric (对应标签instance/job)，以node_memory 为前缀，均以字节为单位
1. node_memory_MemTotal_bytes  主机上的总内存
2. node_memory_MemFree_bytes   主机上的可用内存
3. node_memory_Buffers_bytes   缓冲缓存中的内存
4. node_memory_Cached_bytes    页面缓存中的内存

磁盘相关以 node_filesystem 为前缀，包含mountpoint 等标签
1. node_filesystem_size_bytes，被监控的每个文件系统挂载的大小

systemd 收集器的数据，比如node_systemd_unit_state， 包括标签name 和state， 示例：`node_systemd_unit_state(name="docker.service",state="active") 0`

对于每个instance 的抓取，Prometheus 会填充一些监控指标
1. up，示例`up{job="<job-name>",instance="instance-id"} 1` 即数据抓取成功返回。
2. scrap_duration_seconds， 抓取的持续时间
3. scrap_samples_scraped，目标暴露的样本数

元数据风格的 指标，许多现有的exporter 使用“元数据”模式来提供额外的状态信息，比如cadvisor_version_info，包含标签 cadvisorRevision/dockerVersion/instance/job/kernelVersion/osVersion

不只是node-exporter，mysql-exporter 等也有很多的收集器可选

## 可靠性与可扩展性

Prometheus 本身自带了监控自己的许多指标，比如
1. 收集的指标数量（`sum(count by(__name__)({__name__=\~"\.\+"}))`），进而估算Prometheus 需要的内存和磁盘空间。可以用来辅助做Prometheus 的运维决策
2. top10的 metric 数量： 按 metric 名字分 `topk(10, count by (__name__)({__name__=~".+"}))`
3. top10的 metric 数量： 按 job 名字分 `topk(10, count by (__name__, job)({__name__=~".+"}))`

### 可靠性

1. **Prometheus推荐的容错解决方案**是并 行运行两个配置相同的Prometheus服务器，并且这两个服务器同时处于活动状态。该配置生成的重复警报可以交由上游 Alertmanager 使用其分组 (及抑制)功能进行处 理 。一个推荐的方法是尽可能使上游Alertmanager高度容错 ，而不是关注Prometheus服务器的容错能力。Alertmanager包含由HashiCorp Memberlist库提供的集群功能。Memberlist是一个Go语言库，使
用基于gossip的协议来管理集群成员和成员故障检测。**Alertmanager集群本身负责与集群的其他活动成员共享所有收到的警报**，并处理数据去重(如果需要的话)。
2. HA + 远程存储
3. 联邦集群
4. 使用thanos 或者victoriametrics，来解决全局查询、多副本数据 join 问题。 [高可用prometheus：thanos 实践](https://yasongxu.gitbook.io/container-monitor/yi-.-kai-yuan-fang-an/di-2-zhang-prometheus/thanos)

### 大内存问题

[高可用prometheus：常见问题](https://yasongxu.gitbook.io/container-monitor/yi-.-kai-yuan-fang-an/di-2-zhang-prometheus/prometheus-use)随着规模变大，prometheus需要的cpu和内存都会升高，内存一般先达到瓶颈，这个时候要么加内存，要么集群分片减少单机指标。这里我们先讨论单机版prometheus的内存问题

1. prometheus 的内存消耗主要是因为每隔2小时做一个 block 数据落盘，落盘之前所有数据都在内存里面，因此和采集量有关。
2. 加载历史数据时，是从磁盘到内存的，查询范围越大，内存越大。这里面有一定的优化空间
3. 一些不合理的查询条件也会加大内存，如 group、大范围rate

作者给了一个计算器，设置指标量、采集间隔之类的，计算 prometheus 需要的理论内存值：https://www.robustperception.io/how-much-ram-does-prometheus-2-x-need-for-cardinality-and-ingestion

优化方案：
1. sample 数量超过了 200 万，就不要单实例了，做下分片，然后通过victoriametrics，thanos，trickster等方案合并数据
2. 评估哪些metric 和 label占用较多，去掉没用的指标。
3. 查询时尽量避免大范围查询，注意时间范围和 step 的比例，慎用 group
4. 如果需要关联查询，先想想能不能通过 relabel 的方式给原始数据多加个 label，一条sql 能查出来的何必用join，时序数据库不是关系数据库。

### 可扩展性

单个Prometheus监控方案 毕竟性能有限，Prometheus支持集群联邦。这种分区的方式增加了Prometheus自身的可扩展性，常见分区两种方式：

1. 每一个Prometheus Server实例只负责采集当前数据中心中的一部分任务（Job）。例如可以将不同的监控任务（比如机器和应用程序的）分配到不同的Prometheus实例当中
2. 其二是水平扩展，将同一任务的不同实例的监控数据采集任务划分到不同的Prometheus实例。

如果需要对某些区域或功能进行整体视图查看，那么可以使用federat ion功能将时间序列提取到集中的Prometheus服务器。



### 存储扩展

Prometheus includes a local on-disk time series database, but also optionally integrates with remote storage systems.

```yaml
# prometheus.yml
remote_write:
- url: "http://localhost:1234/receive"
remote_read
- url: xx
```

可以这么理解， Prometheus 定义了一套协议规范，可以和Adapter (及其背后的remote storage) 进行数据交互

可以在Prometheus配置文件中指定Remote Write（远程写）的URL地址，一旦设置了该配置项，Prometheus将采集到的样本数据通过HTTP的形式发送给适配器（Adapter）。而用户则可以在适配器中对接外部任意的服务。外部服务可以是真正的存储系统，公有云的存储服务，也可以是消息队列等任意形式。

同样地，Promthues的Remote Read（远程读）也通过了一个适配器实现。在远程读的流程当中，当用户发起查询请求后，Promthues将向remote_read中配置的URL发起查询请求（matchers，time ranges），Adapter根据请求条件从第三方存储服务中获取响应的数据。同时将数据转换为Promthues的原始样本数据返回给Prometheus Server。当获取到样本数据后，Promthues在本地使用PromQL对样本数据进行二次处理。

## 其它

一些比较有意思的exporter
1. mtail，专门用于从应用程序日志中提取要导出到时间序列数据库中的metric。从无法导出自己内部状态的应用程序中解析日志数据。
2. Blackbox exporter，探针监控，exporter通过 HTTP、HTTPS、DNS、TCP和ICMP来探测端点，执行检查并将生成的指标返回给Prometheus

Pushgateway位于发送指标的应用程序和Prometheus服务器之间。Pushgateway接收指标，然后作为目标 被抓取，以将指标提供给Prometheus服务器。