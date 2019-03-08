---

layout: post
title: Prometheus 学习
category: 技术
tags: Ops
keywords: Prometheus

---

## 简介（未完成）

[Prometheus官网](https://prometheus.io/)

Prometheus 项目与 Kubernetes 项目一样，也来自于 Google 的 Borg 体系，它的原型系统，叫作 BorgMon，是一个几乎与 Borg 同时诞生的内部监控系统。

## 整体结构

![](/public/upload/ops/prometheus.png)

Prometheus 项目工作的核心，是使用 Pull （抓取）的方式去搜集被监控对象的 Metrics 数据（监控指标数据），然后，再把这些数据保存在一个 TSDB （时间序列数据库，比如 OpenTSDB、InfluxDB 等）当中，以便后续可以按照时间进行检索。

那么问题来了，TSDB、Grafana 都是现成组件，Prometheus的工作体现在哪里呢？


