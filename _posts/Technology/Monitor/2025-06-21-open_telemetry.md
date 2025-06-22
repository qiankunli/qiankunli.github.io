---

layout: post
title: OpenTelemetry及生态
category: 架构
tags: Monitor
keywords: observability

---

## 简介

* TOC
{:toc}

## 简介（未完成）

[OPLG：新一代云原生可观测最佳实践](https://mp.weixin.qq.com/s/Bf6nmOymcG9bk91VxLL_Kw) 未细读

[使用 OpenTelemetry Tracing 最大化 Kubernetes 效率](https://mp.weixin.qq.com/s/ieBQx0z1ZofKkgHvYOq_aQ) 未读

[如何利用 OpenTelemetry 监控和优化 Kubernetes 的性能](https://mp.weixin.qq.com/s/zA5NZhDOPFDzzuAAnsI_pA) 未读

## OpenTelemetry

OpenTelemetry 不是凭空出现的。在 OpenTelemetry 出现之前，还出现过 OpenTracing 和 OpenCensus 两套标准。应用性能监控 APM领域有 Jaeger、Pinpoint、Zipkin 等多个开源产品，商业玩家也有很多，可谓竞争非常激烈。然而，这也带来了一个问题，那就是每一家都有一套自己的数据采集标准和 SDK，实现上各不相同，很难实现厂商或者技术中立。OpenTracing 制定了一套与平台和厂商无关的协议标准，让开发人员能够方便地添加或更换底层 APM 的实现。另一套标准 OpenCensus 是谷歌发起的，它和 OpenTracing 最大的不同之处在于，除了链路追踪，它还把指标也包括进来。OpenTracing 和 OpenCensus，这两套框架都有很多追随者，而且二者都想统一对方。但是从功能和特性上来看，它们各有优缺点，半斤八两。所谓天下大势，合久必分，分久必合，既然没办法分个高低，那统一就是大势所趋。于是， OpenTelemetry 横空出世。

OpenTelemetry 主要包括了下面三个部分：
1. 跨语言规范 （Specification）；
2. API / SDK；
3. 接收、转换和导出遥测数据的工具，又称为 OpenTelemetry Collector。

跨语言规范
1. API：定义用于生成和关联追踪、指标和日志的数据类型和操作。
2. SDK：定义 API 特定语言实现的要求，同时还定义配置、数据处理和导出等概念。
3. 数据：定义遥测后端可以提供支持的 OpenTelemetry 协议 （OTLP） 和与供应商无关的语义约定。OTLP 的数据模型定义是基于 ProtoBuf 完成的。

API / SDK
1. API 可以让开发者对应用程序代码进行插桩（Instrument），而 SDK 是 API 的具体实现，是和开发语言相关的。在软件业内，在应用中进行插桩是指将系统状态数据发送到后端，例如日志或者监控系统。发送的数据叫做 Telemetry，也就是遥测数据，包括日志、指标以及追踪等。这些数据记录了处理特定请求时的代码行为，可以对应用系统的状态进行分析。
2. **插桩有两种方式，一是通过手动增加代码生成遥测数据，二是以探针的方式自动收集数据**。OpenTelemetry 为每种语言提供了基础的监测客户端 API 和 SDK。这些包一般都是根据前面介绍的规范里的定义，又结合了语言自身的特点，实现了在客户端采集遥测数据的基本能力。

Collector 针对如何接收、处理和导出遥测数据提供了与供应商无关的实现，消除了运行、操作和维护多个代理 / 收集器的需要，它支持将开源可观测性数据格式（例如 Jaeger、Prometheus 等）发送到一个或多个开源或商业后端。在 Collector 的内部，有一套负责接收、处理和导出数据的流程被称为 Pipeline。 每个 Pipeline 由下面三部分组件组合而成。
1. Receiver：负责按照对应的协议格式监听和接收遥测数据，并把数据转给一个或者多个 Processor。
2. Processor：负责加工处理遥测数据，如丢弃数据、增加信息、转批处理等，并把数据传递给下一个 Processor 或者一个或多个 Exporter。
3. Exporter：负责把数据发送给下一个接收端（一般是指后端），比如将指标数据存储到 Prometheus 中。

从部署的角度来说，Collector 有下面两种模式。
1. 第一种模式可以统称为 Agent 模式。它是把 Collector 部署在应用程序所在的主机内（在 Kubernetes 环境中，可以使用 DaemonSet），或者是在 Kubernetes 环境中通过边车（Sidecar）的方式进行部署。这样，应用采集到的遥测数据可以直接传递给 Collector。
2. 另一种模式是 Gateway 模式。它把 Collector 当作一个独立的中间件，应用会把采集到的遥测数据往这个中间件里传递。

OpenTelemetry to 开源工具组合：作为经典的对各种遥测数据的处理架构，开源工具可将不同类型的数据存储在不同的平台，比如日志存放在 ELK，追踪存放在 Jaeger 这类的 APM 工具，而指标保存在 Prometheus 并通过 Grafana 进行视图展示。