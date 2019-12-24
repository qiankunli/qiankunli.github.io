---

layout: post
title: MOSN源码分析
category: 技术
tags: Go
keywords: Go

---

## 前言（未完成）

* TOC
{:toc}


SOFAMosn是基于Go开发的sidecar，用于service mesh中的数据面代理[sofastack/sofa-mosn](https://github.com/sofastack/sofa-mosn)

可以学到一个代理程序可以玩多少花活儿

[Envoy 官方文档中文版](https://www.servicemesher.com/envoy/)

## 手感

[使用 SOFAMosn 作为 HTTP 代理](https://github.com/sofastack/sofa-mosn/blob/master/examples/cn_readme/http-sample/README.md)

### 基本使用

[SOFA-MOSN源码解析—配置详解](https://juejin.im/post/5c62344f6fb9a049c232e821)

    sofa-mosn
        examples
            codes
                http-example
                    // mosn start -c config.json 即可启动mosn
                    config.json
                    // golang 实现的一个简单的http server，直接go run server.go 即可启动
                    server.go     


单机视角

![](public/upload/go/mosn_http_example_single.png)

多机视角

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

## 代码结构

![](/public/upload/go/mosn_package.png)

![](/public/upload/go/mosn_object.png)

## 启动流程

![](/public/upload/go/mosn_start.png)

## 一次请求的处理过程

### 和kubernetes 结合使用




