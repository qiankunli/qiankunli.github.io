---

layout: post
title: MOSN简介
category: 技术
tags: Mesh
keywords: Go mosn service mesh

---

## 前言

* TOC
{:toc}

Mosn是基于Go开发的sidecar，用于service mesh中的数据面代理，建议先看下[一个sidecar的自我修养](http://qiankunli.github.io/2020/01/14/self_cultivation_of_sidecar.html) 对sidecar 的基本概念、原理有所了解。


云原生网络代理 MOSN 定位是一个全栈的网络代理，支持包括网络接入层(Ingress)、API Gateway、Service Mesh 等场景。PS：所以不只是作为sidecar

## 手感

[使用 SOFAMosn 作为 HTTP 代理](https://github.com/mosn/mosn/tree/master/examples/cn_readme/http-sample)

通过设置 log level 为debug， 代码中加更多日志 来辅助分析代码。**本文主要以http-example 为例来分析**。

### http-example

[SOFA-MOSN源码解析—配置详解](https://juejin.im/post/5c62344f6fb9a049c232e821)

    sofa-mosn
        examples
            codes
                http-example
                    // mosn start -c config.json 即可启动mosn
                    config.json
                    // golang 实现的一个简单的http server，直接go run server.go 即可启动
                    server.go     


![](/public/upload/go/mosn_http_example.png)  

使用`http://localhost:8080` 和 `http://localhost:2345` 都可以拿到数据

![](/public/upload/go/mosn_http_example_diff.png)

### 配置理解

对应config.json 的内容如下

    {
        "servers": [
            {
                "default_log_path": "stdout", 
                "listeners": [
                    {
                        "name": "serverListener", 
                        "address": "127.0.0.1:2046", 
                        "bind_port": true, 
                        "log_path": "stdout", 
                        "filter_chains": [
                            {}
                        ]
                    }, 
                    {
                        "name": "clientListener", 
                        "address": "127.0.0.1:2045", 
                        "bind_port": true, 
                        "log_path": "stdout", 
                        "filter_chains": [
                            {}
                        ]
                    }
                ]
            }
        ], 
        "cluster_manager": {}, 
        "admin": {}
    }

单拎出来 admin 部分， envoy 监听34901 端口

    "admin": {
      "address": {
        "socket_address": {
          "address": "0.0.0.0",
          "port_value": 34901
        }
      }
    }


访问`http://localhost:34901/`的返回结果

    support apis:
    /api/v1/update_loglevel
    /api/v1/enable_log
    /api/v1/disbale_log
    /api/v1/states
    /api/v1/config_dump
    /api/v1/stats

### dubbo-example 现成的java 代码及示例（未完成）

## 代码结构

![](/public/upload/mesh/mosn_package.png)

很多 go 的项目都将 程序入口写在 cmd 文件夹中，然后具体的实现写在 pkg 中，MOSN 项目也是如此。


几乎所有的interface 定义在 `pkg/types` 中，mosn 基于四层 架构实现（见下文），每一个layer 在types 中有一个go 文件，在`pkg` 下有一个专门的文件夹。

[MOSN 源码解析 - filter扩展机制](https://mosn.io/zh/blog/code/mosn-filters/)MOSN 使用了过滤器模式来实现扩展。MOSN 把过滤器相关的代码放在了 pkg/filter 目录下，包括 accept 过程的 filter，network 处理过程的 filter，以及 stream 处理的 filter。其中 accept filters 目前暂不提供扩展（加载、运行写死在代码里面，如要扩展需要修改源码）， steram、network filters 是可以通过定义新包在 pkg/filter 目录下实现扩展。





