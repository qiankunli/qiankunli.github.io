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

## 整体流程

[基于Prometheus的云原生监控系统架构演进](https://mp.weixin.qq.com/s/SBqYGeWDMQwmente8JBaHA)

1. 从配置文件加载采集配置
2. 通过服务发现探测有哪些需要抓取的对象
3. 周期性得往抓取对象发起抓取请求，得到数据
4. 将数据写入本地盘或者写往远端存储

具体的说：服务发现 ==> targets ==> relabel ==> 抓取 ==> metrics_relabel ==> 缓存 ==> 2小时落盘。

1. relabel：当服务发现得到所有target后，Prometheus会根据job中的relabel_configs配置对target进行relabel操作，得到target最终的label集合。每个Job都可以配置一个或多个relabel_config，relabel_config会对Target的label集合进行处理，可以根据label过滤一些Target或者修改，增加，删除一些label。relabel_config过程发生在Target开始进行采集之前，针对的是通过服务发现得到的label集合。
2. Prometheus为这些target创建采集循环，按配置文件里配置的采集间隔进行周期性拉取，采集到的数据根据Job中的metrics_relabel_configs进行relabel，然后再加入上边得到的target最终label集合，综合后得到最终的数据。每个Job还可以配置一个或者多个metrics_relabel_config，其配置方式和relabel_configs一模一样，但是其用于处理的是从Target采集到的数据中的label。

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

4. 平台提供的api，比如Kubernetes. 
    ```yaml
    - job_name: 'kubernetes-cadvisor'
      kubernetes_sd_configs:
      - api_server: 'http://localhost:8080';
        role: node
    ```

[服务发现与Relabel](https://yunlzheng.gitbook.io/prometheus-book/part-ii-prometheus-jin-jie/sd/service-discovery-with-relabel)

## data model

Prometheus 采用**多维**时间序列数据模型。这个时间序列数据模型结合了时间序列名称和称为标签(label)的键/值对，这些**标签提供了维度**（为特定时间序列添加上下文）。每个时间序列由名称和标签的组合唯一标识（从技术上讲，名称本身也是名为__name__的标签）。时间序列的真实值是采样(sample)的结果，它包括两部分：一个float 64类型的数值；一个毫秒精度的时间戳。

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

1. 标签共有两大类：插桩标签(instrumentation label)和目标标签(target label)，插桩标签来自被 监控的资源，目标标签由Prometheus在抓取期间和之后添加。When Prometheus scrapes a target, it attaches some labels automatically to the scraped time series which serve to identify the scraped target:

    * job: The configured job name that the target belongs to.
    * instance: The `<host>:<port>` part of the target's URL that was scraped.
2. 自动生成几个 sample 来描述此次scrape，比如

        # if the instance is healthy, i.e. reachable, or 0 if the scrape failed. 可以用来监控instance 是否可用
        up{job="<job-name>", instance="<instance-id>"}: 1 
        # duration of the scrape.
        scrape_duration_seconds{job="<job-name>", instance="<instance-id>"}: xx

带有__前缀的标签名称保留给Prometheus内部使用



## 其它

### 后台ui

Prometheus服务器还提供了一套内置查询语言PromQL、一个表达式浏览器以及 一个用于浏览服务器上数据的图形界面。

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



### PromQL/Prometheus expression language

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

在 Prometheus 的表达语言中，一个表达式或子表达式可以计算为以下四种类型之一：

1. 瞬时向量（Instant vector）：一组时间序列，每个时间序列包含一个样本，所有样本共享相同的时间戳。基于metric name 、label 做选择，以下3个实例
    ```
    ## 根据metric name 选择
    http_requests_total
    ## 根据metric name + label 选择
    http_requests_total{job="prometheus",group="canary"}
    ## label 支持多个运算符
    http_requests_total{environment=~"staging|testing|development",method!="GET"}
    ```
    
2. 范围向量（Range vector）：一组时间序列，其中包含每个时间序列随时间变化的一系列数据点。
    ```
    # 使用[]指定一个range duration
    http_requests_total{job="prometheus"}[5m]
    ```
3. 标量（Scalar）：一个简单的数字浮点值。
4. 字符串（String）：一个简单的字符串值，目前未使用。


对指标进行 函数计算，比如`sum(http_requests_total)` 支持的函数[expression language functions](https://prometheus.io/docs/prometheus/latest/querying/functions/)

[Prometheus 常用 PromQL 语句](https://mp.weixin.qq.com/s/vr1C6S_jAnMMu_5sUmYPMQ)



## rules

Prometheus uses rules to create new time series and to generate alerts. rule 分为两种类型：RecordingRule 和 AlertingRule

**Recording rules** allow you to precompute frequently needed or computationally expensive expressions and save their result as a new set of time series. Querying the precomputed result will then often be much faster than executing the original expression every time it is needed. This is especially useful for dashboards, which need to query the same expression repeatedly every time they refresh. dashboard每次query，一下子计算上千条time series 肯定会很耗时，因此可以预置一些规则，比如每5分钟汇总一次，即可大大减少计算最终结果时的数据量。

```yaml
groups:
- name: node_rules
  rules:
  - record: instance:node_cpu:avg_rate5m
    expr: 100 - avg(irate(node_cpu_seconds_total{job="node",mode="idle"}[5m])) by (instance) * 100
```

记录规则命名的推荐格式`level:metric:operations`
1. level 表示聚合级别，以及规则的输出标签
2. metric 是指标名称
3. operations 应用于指标的操作列表
