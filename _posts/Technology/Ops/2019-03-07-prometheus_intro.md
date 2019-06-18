---

layout: post
title: Prometheus 学习
category: 技术
tags: Ops
keywords: Prometheus

---

## 简介

* TOC
{:toc}

[Prometheus官网](https://prometheus.io/)

[Prometheus官方文档](https://prometheus.io/docs)

Prometheus 是由SoundCloud 开发的开源监控报警系统和时序数据库（TSDB），由Golang编写

由于数据采集可能会有丢失，所以 Prometheus 不适用于对采集数据要 100% 准确的情形，例如实时监控

## 整体结构

![](/public/upload/ops/prometheus.png)

Prometheus is an open-source systems monitoring and alerting toolkit. 数据采集、简单计算、存储、展示、报警都支持，部分能力可以使用其他组件替代，比如存储，Prometheus本来支持存储在磁盘，也可以通过adapter将数据存在influxdb，进而就可以复用influxdb的一系列能力。

Prometheus is a monitoring platform that collects metrics from monitored targets by scraping metrics HTTP endpoints on these targets. prometheus 通过被抓取对象 暴露出的http 端口抓取metrics，可以看作是一个按配置拉取特定url的“爬虫”

## 配置文件

Prometheus is configured via command-line flags and a configuration file. While the command-line flags configure immutable system parameters (such as storage locations, amount of data to keep on disk and in memory, etc.), the configuration file defines everything related to scraping jobs and their instances, as well as which rule files to load.

从配置文件来看 prometheus

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

启动prometheus `prometheus --config.file=prometheus.yml`

[Reloading Prometheus’ Configuration](https://www.robustperception.io/reloading-prometheus-configuration)类似于nginx 运行时reload 配置文件一样`nignx -c nginx.conf -s reload`， prometheus 也支持运行时reload
 
1. 给Prometheus 进程发送SIGHUP信号 `kill -HUP $Prometheus_Pid`
2. 在Prometheus 启动时携带command line flag `--web.enable-lifecycle`的前提下，调用http reload 接口，`curl -X POST http://ip:9090/-/reload`

## 存储

### data model

Prometheus fundamentally stores all data as time series: streams of timestamped values belonging to the same metric and the same set of labeled dimensions. 


我们访问 `http://ip:9090/metrics` 来看一次实际的数据scrape返回结果

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

## Prometheus expression language

[QUERYING PROMETHEUS](https://prometheus.io/docs/prometheus/latest/querying/basics/)即便一个表达语言，那也是麻雀虽小五脏俱全，字面量、运算符、语法规则、函数等都有，虽然没有编程语言全面，但也像SQL一样很完备了

Time series Selectors 从time series 中选择需要的数据

1. Instant vector selectors 以下3个实例

        ## 根据metric name 选择
        http_requests_total
        ## 根据metric name + label 选择
        http_requests_total{job="prometheus",group="canary"}
        ## label 支持多个运算符
        http_requests_total{environment=~"staging|testing|development",method!="GET"}

2. Range Vector Selectors

        # 使用[]指定一个range duration
        http_requests_total{job="prometheus"}[5m]

[expression language functions](https://prometheus.io/docs/prometheus/latest/querying/functions/)

## rules

Prometheus uses rules to create new time series and to generate alerts.

### recording rules

Recording rules allow you to precompute frequently needed or computationally expensive expressions and save their result as a new set of time series. Querying the precomputed result will then often be much faster than executing the original expression every time it is needed. This is especially useful for dashboards, which need to query the same expression repeatedly every time they refresh.

每次query，一下子计算上千条time series 肯定会很耗时，因此可以预置一些规则，比如每5分钟汇总一次，即可大大减少计算最终结果时的数据量。

## 集群联邦

单个Prometheus监控方案 毕竟性能有限，Prometheus支持集群联邦。这种分区的方式增加了Prometheus自身的可扩展性，常见分区两种方式：

1. 每一个Prometheus Server实例只负责采集当前数据中心中的一部分任务（Job）。例如可以将不同的监控任务分配到不同的Prometheus实例当中，再由中心Prometheus实例进行聚合。
2. 其二是水平扩展，将同一任务的不同实例的监控数据采集任务划分到不同的Prometheus实例。
