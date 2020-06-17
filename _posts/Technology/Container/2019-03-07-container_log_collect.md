---

layout: post
title: 容器日志采集
category: 技术
tags: Container
keywords: container log collect

---

## 简介

* TOC
{:toc}

[直击痛点，详解 K8s 日志采集最佳实践](https://mp.weixin.qq.com/s/PVYbdryPvSegWGdB7evBZg)

## 采集方式

![](/public/upload/container/collect_log.png)

### SideCar 模式 VS Node 模式

[容器日志采集利器Log-Pilot](https://developer.aliyun.com/article/674327)

![](/public/upload/container/log_sidecar_vs_agent.png)

1. sidecar 模式，每个 Pod 中都附带一个 logging 容器来进行本 Pod 内部容器的日志采集，一般采用共享卷的方式
    1. 在集群规模比较大的情况下，或者说单个节点上容器特别多的情况下，很明显的一个问题就是占用的资源比较多
    2. 对日志存储后端占用过多的连接数。
2. node 模式，在每个 Node 节点上仅需布署一个 logging 容器来进行本 Node 所有容器的日志采集。最明显的优势就是占用资源比较少，同样在集群规模比较大的情况下表现出的优势越明显。但对于这种模式来说我们就需要一个更加智能的日志采集工具来配合

### 采集组件

1. fluentd，CNCF社区
2. filebeat，来自Elastic
3. flume

## Stdout VS 文件

容器提供标准输出和文件两种方式，

1. 在容器中，标准输出将日志直接输出到 stdout 或 stderr，实际的业务场景中建议大家尽可能使用文件的方式
    1. Stdout 性能问题，从应用输出 stdout 到服务端，中间会经过好几个流程（例如普遍使用的 JSON LogDriver）：应用 stdout -> DockerEngine -> LogDriver -> 序列化成 JSON -> 保存到文件 -> Agent 采集文件 -> 解析 JSON -> 上传服务端。整个流程相比文件的额外开销要多很多，在压测时，每秒 10 万行日志输出就会额外占用 DockerEngine 1 个 CPU 核；
    2. Stdout 不支持分类，即所有的输出都混在一个流中，无法像文件一样分类输出，通常一个应用中有 AccessLog、ErrorLog、InterfaceLog（调用外部接口的日志）、TraceLog 等，而这些日志的格式、用途不一，如果混在同一个流中将很难采集和分析；
    3. Stdout 只支持容器的主程序输出，如果是 daemon/fork 方式运行的程序将无法使用 stdout；
    4. 文件的 Dump 方式支持各种策略，例如同步/异步写入、缓存大小、文件轮转策略、压缩策略、清除策略等，相对更加灵活。
2. 日志打印到文件的方式和虚拟机/物理机基本类似，只是日志可以使用不同的存储方式，例如默认存储、EmptyDir、HostVolume、NFS 等。

## 采集什么

![](/public/upload/container/collect_what.png)

1. 容器文件，比如容器运行了Tomcat，则Tomcat 的启动日志也在采集范围之内
2. 容器 Stdout
3. 宿主机文件
4. Journal
5. Event 

[使用日志服务进行Kubernetes日志采集](https://help.aliyun.com/document_detail/87540.html)其它

1. 支持多种采集部署方式，包括 DaemonSet、Sidecar、DockerEngine LogDriver 等；
2. 支持对日志数据进行富化，包括附加 Namespace、Pod、Container、Image、Node 等信息；
3. 稳定、高可靠，基于阿里自研的 Logtail 采集 Agent 实现，目前全网已有几百万的部署实例；
4. 基于 CRD 进行扩展，可使用 Kubernetes 部署发布的方式来部署日志采集规则，与 CICD 完美集成。

[9 个技巧，解决 K8s 中的日志输出问题](https://mp.weixin.qq.com/s/fLNzHS_6V78pSJ_zqTWhZg)

## log-pilot（待补充）

### 启动

启动命令`/pilot/pilot -template /pilot/filebeat.tpl -base /host -log-level debug`

filebeat.tpl 内容

```
{{range .configList}}
- type: log
  enabled: true
  paths:
      - {{ .HostDir }}/{{ .File }}
  scan_frequency: 10s
  fields_under_root: true
  {{if .Stdout}}
  docker-json: true
  {{end}}
  {{if eq .Format "json"}}
  json.keys_under_root: true
  {{end}}
  fields:
      {{range $key, $value := .Tags}}
      {{ $key }}: {{ $value }}
      {{end}}
      {{range $key, $value := $.container}}
      {{ $key }}: {{ $value }}
      {{end}}
  tail_files: false
  close_inactive: 2h
  close_eof: false
  close_removed: true
  clean_removed: true
  close_renamed: false
{{end}}
```

![](/public/upload/container/log_pilot_object.png)

![](/public/upload/container/log_pilot_sequence.png)

1. log-pilot 比较喜欢用环境变量，比如采集插件/组件 使用fluentd 还是filebeat 都是由环境变量指定 `PILOT_TYPE=filebeat`
2. 待确认：fluentd/filebeat 就像nginx 一样，根据配置文件运行，本身不具备动态发现容器日志文件的能力，log-pilot 对其封装了下。就像istio pilot-agent 对envoy 所做的那样。








