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

![](/public/upload/go/mosn_http_example.png)  

注意

1. 官方示例 client、mosn、server 都运行在本地非容器环境下， 与真实环境有一定差异， **只做体验用**。
2. MOSN 配置支持静态配置、动态配置和混合模式，**示例中只有静态配置**。除协议不同外，其它配置比如server 监听端口等 都是一致的
2. 通过设置 log level 为debug， 代码中加更多日志 来辅助分析代码。

### http-example

[使用 MOSN 作为 HTTP 代理](https://github.com/mosn/mosn/tree/master/examples/cn_readme/http-sample)

对应config.json 的内容如下

```json
{
    "servers": [
        {
            "default_log_path": "stdout", 
            "listeners": [
                {
                    "name": "serverListener", 
                    "address": "127.0.0.1:2046", 
                    "log_path": "stdout", 
                    "filter_chains": [
                        {}
                    ]
                }, 
                {
                    "name": "clientListener", 
                    "address": "127.0.0.1:2045", 
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
```

单拎出来 admin 部分， envoy 监听34901 端口

```
"admin": {
    "address": {
    "socket_address": {
        "address": "0.0.0.0",
        "port_value": 34901
    }
    }
}
```

使用`http://localhost:8080` 和 `http://localhost:2345` 都可以拿到数据。mosn 本身也对外提供一些接口，访问`http://localhost:34901/`的返回结果

    support apis:
    /api/v1/update_loglevel
    /api/v1/enable_log
    /api/v1/disbale_log
    /api/v1/states
    /api/v1/config_dump
    /api/v1/stats

### dubbo-example

[使用 MOSN 作为 Dubbo 代理](https://github.com/mosn/mosn/tree/master/examples/cn_readme/dubbo-examples)

![](/public/upload/mesh/dubbo_config_json.png)

一个client 一个 server 的配置解释：

1. dubbo provider 启动时监听 `0.0.0.0:20880`，dubbo consumer 启动时连接 `localhost:2045`
2. 请求链路：clientListener 2045 ==> client_router ==> clientCluster 2046 ==> serverListener ==>  server_router ==> serverCluster 20880
3. `clientListener 2045 ==> client_router ==> clientCluster` 模拟client + mosn 场景。如果示例 是一个client 多个server 的话，会更清晰一些。
4. `serverListener ==>  server_router ==> serverCluster 20880` 模拟mosn + server 场景。

## 配置

[MOSN配置概览](https://mosn.io/zh/docs/configuration/)MOSN 的配置文件可以分为以下四大部分：

1. Servers 配置，目前仅支持最多 1 个 Server 的配置，Server 中包含一些基础配置以及对应的 Listener 配置
2. ClusterManager 配置，包含 MOSN 的 Upstream 详细信息
3. 对接控制平面（Pilot）的 xDS 相关配置
4. 其他配置
    * Trace、Metrics、Debug、Admin API 相关配置
    * 扩展配置，提供自定义配置扩展需求

```json
{
  "servers": [], ## 目前仅支持最多 1 个 Server 的配置，Server 中包含一些基础配置以及对应的 Listener 配置
  "cluster_manager": {},    ## 包含 MOSN 的 Upstream 详细信息
  "dynamic_resources": {}, ## 对接控制平面（Pilot）的 xDS 相关配置
  "static_resources": {},
  "admin":{},
  "pprof":{},
  "tracing":{},
  "metrics":{}
}
```

### 配置类型

1. 静态配置，指 MOSN 启动时，不对接控制平面 Pilot 的配置，用于一些相对固定的简单场景（如 MOSN 的示例）。使用静态配置启动的 MOSN，也可以通过扩展代码，调用动态更新配置的接口实现动态修改。
2. 动态配置，会向管控面请求获取运行时所需要的配置，管控面也可能在运行时推送更新 MOSN 运行配置。动态配置启动时必须包含 DynamicResources 和 StaticResources 配置。
3. 混合模式，以混合模式启动的 MOSN 会先以静态配置完成初始化，随后可能由控制平面获取配置更新。

### server 配置

```json
{
  "default_log_path":"",
  "default_log_level":"",
  "global_log_roller":"",
  "graceful_timeout":"",
  "processor":"",
  "listeners":[], ## 描述了 MOSN 启动时监听的端口，以及对应的端口对应不同逻辑的配置
  "routers":[] ## 描述 MOSN 的路由配置，通常与 proxy 配合使用
}
```

Listener 的配置可以通过Listener动态接口进行添加和修改。

```json
"listeners":[
    {
        "name":"",
        "type":"",
        "address":"", ## Listener 监听的地址
        "bind_port":"", 
        "use_original_dst":"",
        "access_logs":[],
        "filter_chains":[  ##  MOSN 仅支持一个 filter_chain
            {
                "filters": [ ## 一组 network filter 配置，描述了 MOSN 在连接建立以后如何在 4 层处理连接数据
                    {
                    "type":"",
                    "config": {}
                    }
                ]
            }
            ],
        "stream_filters":[], ## 一组 stream_filter 配置，目前只在 filter_chain 中配置了 filter 包含 proxy 时生效
        "inspector":"",
        "connection_idle_timeout":""
    }
]
```
network filter 可自定义扩展实现，默认支持的 type 包括 proxy、tcp proxy、connection_manager。
connection_manager 是一个特殊的 network filter，它需要和 proxy 一起使用，用于描述 proxy 中路由相关的配置，是一个兼容性质的配置，后续可能有修改。

**路由 ，一个请求所属的domains  绑定了许多路由规则，目的将一个请求 路由到一个cluster 上**。

```json
"routers":[
    {
        "router_config_name":"",
        "virtual_hosts": [ ## 描述具体的路由规则细节
            {
                "domains":[], ## 表示一组可以匹配到该 virtual host 的 domain，支持配置通配符
                "routers":[] ## 一组具体的路由匹配规则
            }
        ]
    }
]
```
Router prefix,path,regex优先级从高到低。
```json
"virtual_hosts": [
    {
        "routers":[
            {
                "match":{ ## 路由的匹配参数。
                    "prefix":"", ## 路由会匹配 path 的前缀
                    "path":"",   ## 路由会匹配精确的 path
                    "regex":"",  ## 路由会按照正则匹配的方式匹配 path
                    "headers": [] ## 组请求需要匹配的 header。请求需要满足配置中所有的 Header 配置条件才算匹配成功
                },   
                "route":{## 路由行为，描述请求将被路由的 upstream 信息
                    "cluster_name":"", ## 表示请求将路由到的 upstream cluster
                    "metadata_match":"",
                    "timeout":"",   ## 表示默认情况下请求转发的超时时间
                    "retry_policy":{} ## 表示如果请求在遇到了特定的错误时采取的重试策略，默认没有配置的情况下，表示没有重试
                },   
                "per_filter_config":{} ## 其中 key 需要匹配一个 stream filter 的 type，key 对应的 json 是该 stream filter 的 config。
            }
        ]
    }
]
```






## 代码结构

![](/public/upload/mesh/mosn_package.png)

很多 go 的项目都将 程序入口写在 cmd 文件夹中，然后具体的实现写在 pkg 中，MOSN 项目也是如此。


几乎所有的interface 定义在 `pkg/types` 中，mosn 基于四层 架构实现（见下文），每一个layer 在types 中有一个go 文件，在`pkg` 下有一个专门的文件夹。







