---

layout: post
title: Prometheus 学习
category: 技术
tags: Go
keywords: Prometheus

---

## 简介

* TOC
{:toc}

devops基本理念：
1. if you can't measure it,you can't improve it
2. you build it,you run it, you monitor it.  谁开发，谁运维，谁监控，

四种主要的监控方式
1. Logging
2. Tracing
3. Metric
4. Healthchecks

监控是分层次的， 以metric 为例

1. 系统层，比如cpu、内存监控，面向运维人员
2. 应用层，应用出错、请求延迟等，业务开发、框架开发人员
3. 业务层，比如下了多少订单等，业务开发人员

## 整体结构

![](/public/upload/go/monitor_overview.png)

Prometheus is an open-source systems monitoring and alerting toolkit. 数据采集、简单计算、存储、展示、报警都支持，部分能力可以使用其他组件替代，比如存储，Prometheus本来支持存储在磁盘，也可以通过adapter将数据存在influxdb，进而就可以复用influxdb的一系列能力。

![](/public/upload/ops/prometheus.png)

Prometheus is a monitoring platform that collects metrics from monitored targets by scraping metrics HTTP endpoints on these targets. prometheus 通过被抓取对象 暴露出的http 端口抓取metrics，可以看作是一个按配置拉取特定url的“爬虫”

[Prometheus官网](https://prometheus.io/)[Prometheus官方文档](https://prometheus.io/docs)Prometheus 是由SoundCloud 开发的开源监控报警系统和时序数据库（TSDB），由Golang编写。

由于数据采集可能会有丢失，所以 Prometheus 不适用于对采集数据要 100% 准确的情形，例如实时监控

## 配置文件

Prometheus is configured via command-line flags and a configuration file. While the command-line flags configure immutable system parameters (such as storage locations, amount of data to keep on disk and in memory, etc.), the configuration file defines everything related to scraping jobs and their instances, as well as which rule files to load.

从配置文件来看 prometheus

```yaml
$ cat /usr/local/prometheus/prometheus.yml
# 全局配置
global:
    scrape_interval:     15s # 默认抓取间隔, 15秒向目标抓取一次数据。
    evaluation_interval: 15s # 执行rules的频率
alerting:
    alertmanagers:  ## 配置alertmanager的地址
rule_files:
# - "first.rules"
# - "second.rules"
# controls what resources Prometheus monitors.
scrape_configs:
# 这里是抓取promethues自身的配置
- job_name: 'prometheus'
    # metrics_path defaults to '/metrics'
    # scheme defaults to 'http'.
    # 重写了全局抓取间隔时间，由15秒重写成5秒。
    scrape_interval: 5s
    static_configs:
    - targets: ['localhost:9090']
- job_name: 'mysql'
    scrape_interval: 5s
    static_configs:
    - targets: ['localhost:9104','localhost:9105']
```

启动prometheus `prometheus --config.file=prometheus.yml`

[Reloading Prometheus’ Configuration](https://www.robustperception.io/reloading-prometheus-configuration)类似于nginx 运行时reload 配置文件一样`nignx -c nginx.conf -s reload`， prometheus 也支持运行时reload
 
1. 给Prometheus 进程发送SIGHUP信号 `kill -HUP $Prometheus_Pid`
2. 在Prometheus 启动时携带command line flag `--web.enable-lifecycle`的前提下，调用http reload 接口，`curl -X POST http://ip:9090/-/reload`

## prometheus 服务发现

在基于云(IaaS或者CaaS)的基础设施环境中用户可以像使用水、电一样按需使用各种资源（计算、网络、存储）。按需使用就意味着资源的动态性，这些资源可以随着需求规模的变化而变化。这种按需的资源使用方式对于监控系统而言就意味着没有了一个固定的监控目标，所有的监控对象(基础设施、应用、服务)都在动态的变化。对于Prometheus这一类基于Pull模式的监控系统，显然也无法继续使用的static_configs的方式静态的定义监控目标。解决方案就是引入一个中间的代理人（服务注册中心），这个代理人掌握着当前所有监控目标的访问信息，Prometheus只需要向这个代理人询问有哪些监控目标即可， 这种模式被称为服务发现。

服务发现方式

1. 基于文件，通过任意的方式将监控Target的信息写入文件，Prometheus会定时从文件中读取最新的Target信息

    ```yaml
    global:
      scrape_interval: 15s
      scrape_timeout: 10s
       evaluation_interval: 15s
    scrape_configs:
    - job_name: 'file_ds'
        file_sd_configs:
        - files:
          - targets.json
    ```

2. 基于DNS
3. 基于consul

    ```yaml
    - job_name: node_exporter
        metrics_path: /metrics
        scheme: http
        consul_sd_configs:
          - server: localhost:8500
            services:
              - node_exporter
    ```
    在consul_sd_configs定义当中通过server定义了Consul服务的访问地址，services则定义了当前需要发现哪些类型服务实例的信息，这里限定了只获取node_exporter的服务实例信息。

4. 平台提供的api，比如Kubernetes

    ```yaml
    - job_name: 'kubernetes-cadvisor'
      kubernetes_sd_configs:
      - api_server: 'http://localhost:8080';
        role: node
    ```

[服务发现与Relabel](https://yunlzheng.gitbook.io/prometheus-book/part-ii-prometheus-jin-jie/sd/service-discovery-with-relabel)

## 存储

### data model

Prometheus fundamentally stores all data as time series: streams of timestamped values belonging to the same metric and the same set of labeled dimensions. time series 最直观的数据格式是`(t0,v0),(t1,v1),(t2,v2)...` 一个time series 属于一个metric + labels。 我们日常看到的格式是`(metric name, label.., value) `，因为 数据产生的时候是没有时间的，等prometheus 抓取落盘后 会有一个抓取的时间。

我们访问 `http://ip:9090/metrics` 来看一次实际的数据scrape返回结果

```
# HELP go_gc_duration_seconds A summary of the GC invocation durations.
# TYPE go_gc_duration_seconds summary
go_gc_duration_seconds{quantile="0"} 1.3428e-05
go_gc_duration_seconds{quantile="0.25"} 3.5274e-05
go_gc_duration_seconds{quantile="0.5"} 5.292e-05
go_gc_duration_seconds{quantile="0.75"} 6.7349e-05
go_gc_duration_seconds{quantile="1"} 0.000192367
go_gc_duration_seconds_sum 0.00562896
go_gc_duration_seconds_count 101
# HELP go_goroutines Number of goroutines that currently exist.
# TYPE go_goroutines gauge
go_goroutines 175
# HELP prometheus_sd_discovered_targets Current number of discovered targets.
# TYPE prometheus_sd_discovered_targets gauge
prometheus_sd_discovered_targets{config="6577653216d30e75870c4b843dfbafd6",name="notify"} 1
prometheus_sd_discovered_targets{config="cadvisor",name="scrape"} 16
```

以`prometheus_sd_discovered_targets{config="6577653216d30e75870c4b843dfbafd6",name="notify"} 1`（称之为sample） 为例

1. prometheus_sd_discovered_targets 叫metric name
2. `config="6577653216d30e75870c4b843dfbafd6"` 属于`<label name>=<label value>` 从上述例子看，label 可以没有
3. 最后的数值是一个  float64 value

Prometheus Server 对`http://ip:9090/metrics` 返回的数据 不是直接原样存储

1. 对拉取的metric 数据进行补充，When Prometheus scrapes a target, it attaches some labels automatically to the scraped time series which serve to identify the scraped target:

    * job: The configured job name that the target belongs to.
    * instance: The `<host>:<port>` part of the target's URL that was scraped.

2. 自动生成几个 sample 来描述此次scrape，比如

        # if the instance is healthy, i.e. reachable, or 0 if the scrape failed. 可以用来监控instance 是否可用
        up{job="<job-name>", instance="<instance-id>"}: 1 
        # duration of the scrape.
        scrape_duration_seconds{job="<job-name>", instance="<instance-id>"}: xx

### 和存储结果 对接

Prometheus includes a local on-disk time series database, but also optionally integrates with remote storage systems.

    # prometheus.yml
    remote_write:
    - url: "http://localhost:1234/receive"
    remote_read
    - url: xx

可以这么理解， Prometheus 定义了一套协议规范，可以和Adapter (及其背后的remote storage) 进行数据交互

可以在Prometheus配置文件中指定Remote Write（远程写）的URL地址，一旦设置了该配置项，Prometheus将采集到的样本数据通过HTTP的形式发送给适配器（Adapter）。而用户则可以在适配器中对接外部任意的服务。外部服务可以是真正的存储系统，公有云的存储服务，也可以是消息队列等任意形式。

同样地，Promthues的Remote Read（远程读）也通过了一个适配器实现。在远程读的流程当中，当用户发起查询请求后，Promthues将向remote_read中配置的URL发起查询请求（matchers，time ranges），Adapter根据请求条件从第三方存储服务中获取响应的数据。同时将数据转换为Promthues的原始样本数据返回给Prometheus Server。当获取到样本数据后，Promthues在本地使用PromQL对样本数据进行二次处理。

## 其它

### 后台ui

假设prometheus 运行在`localhost:9090`，则访问`localhost:9090` 可以直接打开其后台ui 

![](/public/upload/go/prometheus_status.png)

后台ui包括几个菜单，`localhost:9090` 加上菜单名 即可直接打开 对应菜单，比如 `localhost:9090/alerts` 即可打开 Alerts菜单

1. Alerts 当前的报警配置 及报警状态
2. Graph 对prometheus 抓取的数据进行查询
3. Status 有几个子菜单
    1. Runtime & Build Information  当前prometheus 进程的构建和运行时信息
    2. Command-Line Flags 启动当前prometheus 进程时的命令行参数
    3. Configuration 启动当前prometheus 进程时的配置
    4. Rules, **Prometheus uses rules to create new time series and to generate alerts**.
    5. Targets 可以看到当前Prometheus 抓取的Target的状态，以UP 和 Down区分
    6. Service Discovery  基本对应 配置文件中的 job_name

### metric 种类

1. counter（计数器），始终增加，比如http请求数、下单数
2. gauge（测量仪），当期值的一次快照测量，可增可减。比如磁盘使用率、当前同时在线用户数
3. Histogram（直方图），通过分桶方式统计样本分布
4. Summary（汇总），根据样本统计出百分位，比如客户端计算

### Prometheus expression language

[QUERYING PROMETHEUS](https://prometheus.io/docs/prometheus/latest/querying/basics/)即便一个表达语言，那也是麻雀虽小五脏俱全，字面量、运算符、语法规则、函数等都有，虽然没有编程语言全面，但也像SQL一样很完备了



```
http_requests_total{code="200",handler="/api/v1/label/:name/values",instance="0.0.0.0:9099",job="prometheus"}	802
http_requests_total{code="200",handler="/api/v1/query",instance="0.0.0.0:9099",job="prometheus"}	188683
http_requests_total{code="200",handler="/api/v1/query_range",instance="0.0.0.0:9099",job="prometheus"}	121281
http_requests_total{code="200",handler="/api/v1/series",instance="0.0.0.0:9099",job="prometheus"}	12512
http_requests_total{code="200",handler="/config",instance="0.0.0.0:9099",job="prometheus"}	1
http_requests_total{code="200",handler="/flags",instance="0.0.0.0:9099",job="prometheus"}	2
http_requests_total{code="200",handler="/graph",instance="0.0.0.0:9099",job="prometheus"}	11
http_requests_total{code="200",handler="/metrics",instance="0.0.0.0:9099",job="prometheus"}	17605
http_requests_total{code="200",handler="/rules",instance="0.0.0.0:9099",job="prometheus"}
```
Time series Selectors 从time series 中选择需要的数据

1. Instant vector selectors 基于metric name 、label 做选择，以下3个实例
    ```
    ## 根据metric name 选择
    http_requests_total
    ## 根据metric name + label 选择
    http_requests_total{job="prometheus",group="canary"}
    ## label 支持多个运算符
    http_requests_total{environment=~"staging|testing|development",method!="GET"}
    ```
2. Range Vector Selectors  为查询数据指定一个时间范围
    ```
    # 使用[]指定一个range duration
    http_requests_total{job="prometheus"}[5m]
    ```
3. 对指标进行 函数计算，比如`sum(http_requests_total)` 支持的函数[expression language functions](https://prometheus.io/docs/prometheus/latest/querying/functions/)

## rules

Prometheus uses rules to create new time series and to generate alerts. rule 分为两种类型：RecordingRule 和 AlertingRule

**Recording rules** allow you to precompute frequently needed or computationally expensive expressions and save their result as a new set of time series. Querying the precomputed result will then often be much faster than executing the original expression every time it is needed. This is especially useful for dashboards, which need to query the same expression repeatedly every time they refresh. dashboard每次query，一下子计算上千条time series 肯定会很耗时，因此可以预置一些规则，比如每5分钟汇总一次，即可大大减少计算最终结果时的数据量。

### 4个黄金指标

4个黄金指标可以在服务级别帮助衡量终端用户体验、服务中断、业务影响等层面的问题。
1. 延迟：服务请求所需时间。
2. 通讯量：监控当前系统的流量，用于衡量服务的容量需求。例如，在HTTP REST API中, 流量通常是每秒HTTP请求数；
3. 错误：监控当前系统所有发生的错误请求，衡量当前系统错误发生的速率。
4. 饱和度：衡量当前服务的饱和度。比如，“磁盘是否可能在4个小时候就满了”。

### 告警

**告警能力在Prometheus的架构中被划分成两个独立的部分**。通过在Prometheus中定义AlertRule（告警规则），Prometheus会周期性的对告警规则进行计算，如果满足告警触发条件就会向Alertmanager发送告警信息。

在Prometheus中一条告警规则主要由以下几部分组成：
1. 告警名称：用户需要为告警规则命名，当然对于命名而言，需要能够直接表达出该告警的主要内容
2. 告警规则：**告警规则实际上主要由PromQL进行定义**，其实际意义是当表达式（PromQL）查询结果持续多长时间（During）后出发告警

在Prometheus中，还可以通过Group（告警组）对一组相关的告警进行统一定义。比如 “最大响应时间超过xx”、“4个9影响时间超过xx” 都属于“http 请求超时”的范畴。

Alertmanager作为一个独立的组件，负责接收并处理（去重、分组、路由（基于alert携带的标签将告警发给不同的receiver）、抑制、静默）来自Prometheus Server(也可以是其它的客户端程序)的告警信息。

![](/public/upload/go/prometheus_alertmanager_overview.png)

prometheus 本身的报警机制基本能够满足各种报警需求，唯一的缺憾就是 配置变更要通过 修改配置文件 以及reload prometheus server，国内开源了[Qihoo360/doraemon](https://github.com/Qihoo360/doraemon) 来解决该问题。

## 集群联邦

单个Prometheus监控方案 毕竟性能有限，Prometheus支持集群联邦。这种分区的方式增加了Prometheus自身的可扩展性，常见分区两种方式：

1. 每一个Prometheus Server实例只负责采集当前数据中心中的一部分任务（Job）。例如可以将不同的监控任务分配到不同的Prometheus实例当中，再由中心Prometheus实例进行聚合。PS：也就是由一个中心prometheus server 去拉取另一个 prometheus server的数据。
2. 其二是水平扩展，将同一任务的不同实例的监控数据采集任务划分到不同的Prometheus实例。
