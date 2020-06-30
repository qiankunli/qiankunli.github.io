---

layout: post
title: Prometheus 源码分析
category: 技术
tags: Go
keywords: Prometheus Source

---

## 前言(未完成)

* TOC
{:toc}

## 源码目录说明

[硬核源码解析Prometheus系列 ：一 、初入Prometheus](https://mp.weixin.qq.com/s/JUBe3D_gIIoC1Wi-jMYJTw)

1. cmd目录是prometheus的入口和promtool规则校验工具的源码
2. discovery是prometheus的服务发现模块，主要是scrape targets，其中包含consul, zk, azure, file,aws, dns, gce等目录实现了不同的服务发现逻辑，可以看到静态文件也作为了一种服务发现的方式，毕竟静态文件也是动态发现服务的一种特殊形式
3. config用来解析yaml配置文件，其下的testdata目录中有非常丰富的各个配置项的用法和测试
4. notifier负责通知管理，规则触发告警后，由这里通知服务发现的告警服务，之下只有一个文件，不需要特别关注
5. pkg是内部的依赖
    - relabel ：根据配置文件中的relabel对指标的label重置处理 
    - pool：字节池
    - timestamp：时间戳
    - rulefmt：rule格式的验证
    - runtime：获取运行时信息在程序启动时打印
6. prompb定义了三种协议，用来处理远程读写的远程存储协议，处理tsdb数据的rpc通信协议，被前两种协议使用的types协议，例如使用es做远程读写，需要远程端实现远程存储协议(grpc)，远程端获取到的数据格式来自于types中，就是这么个关系
7. promql处理查询用的promql语句的解析
8. rules负责告警规则的加载、计算和告警信息通知
9. scrape是核心的根据服务发现的targets获取指标存储的模块
10. storge处理存储，其中fanout是存储的门面，remote是远程存储，本地存储用的下面一个文件夹
11. tsdb时序数据库，用作本地存储

prometheus的启动也可以看作十个不同职能组件的启动。 启动用到了 `github.com/oklog` 的Group struct， Group collects actors (functions) and runs them concurrently. When one actor (function) returns, all actors are interrupted. 实现多个协程”共进退“的效果（实际上Group 自己也没干啥事儿， 就是封装了业务函数 和 interrupt 两个函数）。

## metric scrape 组件

源代码就3个文件

```
$GOPATH/src/github.com/prometheus/prometheus/scrape
    manager.go
    scrape.go
    target.go
```

![](/public/upload/go/prometheus_scraper_object.png)