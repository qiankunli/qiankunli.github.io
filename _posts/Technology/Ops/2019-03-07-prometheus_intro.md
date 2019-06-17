---

layout: post
title: Prometheus 学习
category: 技术
tags: Ops
keywords: Prometheus

---

## 简介（未完成）

[Prometheus官网](https://prometheus.io/)

[Prometheus官方文档](https://prometheus.io/docs)

Prometheus 是由SoundCloud 开发的开源监控报警系统和时序数据库（TSDB），由Golang编写

## 整体结构

![](/public/upload/ops/prometheus.png)

Prometheus is an open-source systems monitoring and alerting toolkit. 数据采集、简单计算、存储、展示、报警都支持，部分能力可以使用其他组件替代，比如存储，Prometheus本来支持存储在磁盘，也可以通过adapter将数据存在influxdb，进而就可以复用influxdb的一系列能力。

Prometheus is a monitoring platform that collects metrics from monitored targets by scraping metrics HTTP endpoints on these targets. prometheus 通过被抓取对象 暴露出的http 端口抓取metrics，可以看作是一个按配置拉取特定url的“爬虫”

## 配置文件

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


Prometheus uses rules to create new time series and to generate alerts.

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

本身支持 metric 数据存储在本地磁盘