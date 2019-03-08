---

layout: post
title: 容器监控
category: 技术
tags: Docker
keywords: jib

---

## 简介（未完成）

* TOC
{:toc}


## 为什么要监控

参见 [容器狂打日志怎么办？](http://qiankunli.github.io/2019/03/05/container_log.html)

[容器狂占cpu怎么办？](http://qiankunli.github.io/2019/03/06/container_cpu.html)

参见[Prometheus 学习](http://qiankunli.github.io/2019/03/07/prometheus_intro.html)

## 监控什么

![](/public/upload/kubernetes/kubernetes_monitor.png)

可以发现一个共同点，应用要自己暴露出一个/metrics，而不是单纯依靠监控组件从”外面“ 分析它

1. 依靠现成组件提供 通用metric
2. 自己实现

## 自定义监控指标

在过去的很多 PaaS 项目中，其实都有一种叫作 Auto Scaling，即自动水平扩展的功能。只不过，这个功能往往只能依据某种指定的资源类型执行水平扩展，比如 CPU 或者 Memory 的使用值。而在真实的场景中，用户需要进行 Auto Scaling 的依据往往是自定义的监控指标。比如，某个应用的等待队列的长度，或者某种应用相关资源的使用情况。


## 可以根据监控数据做什么

1. 根据容器内存的耗费统计，个性化容器的内存 limit
2. cpu 负载过高时，报警提醒
3. 混部
4. Auto Scaling



