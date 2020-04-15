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

## 手感

[使用 SOFAMosn 作为 HTTP 代理](https://github.com/mosn/mosn/tree/master/examples/cn_readme/http-sample)

通过设置 log level 为debug， 代码中加更多日志 来辅助分析代码。**本文主要以http-example 为例来分析**。

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

很多 go 的项目都将 程序入口写在 cmd 文件夹中，然后具体的实现写在 pkg 中，MOSN 项目也是如此。几乎所有的interface 定义在 `pkg/types` 中，mosn 基于四层 架构实现（见下文），每一个layer 在types 中有一个go 文件，在`pkg` 下有一个专门的文件夹。

## 分层架构

![](/public/upload/go/mosn_layer_process.png)

一般的服务端编程，二级制数据经过协议解析为 协议对应的model（比如HttpServletRequest） 进而交给上层业务方处理，对于mosn 

1. 协议上数据统一划分为header/data/Trailers 三个部分，转发也是以这三个子部分为基本单位
2. 借鉴了http2 的stream 的理念（所以Stream interface 上有一个方法是`ID()`），Stream 可以理解为一个子Connection，Stream 之间可以并行请求和响应，通过StreamId关联，用来实现在一个Connection 之上的“多路复用”。PS：为了连接数量与请求数量解耦。

代码的组织（pkg/stream,pkg/protocol,pkg/proxy）  跟上述架构是一致的

![](/public/upload/go/mosn_layer.png)

1. `pkg/types/connection.go` Connection
2. `pkg/types/stream.go` StreamConnection is a connection runs multiple streams
3. `pkg/types/stream.go` Stream is a generic protocol stream
4. 一堆listener 和filter 比较好理解：Method in listener will be called on event occur, but not effect the control flow.Filters are called on event occurs, it also returns a status to effect control flow. Currently 2 states are used: Continue to let it go, Stop to stop the control flow.
5. protocol 和 stream 两个layer 因和协议有关，不同协议之间实现差异很大，层次不是很清晰
6. 跨层次调用/数据传输通过跨层次struct 的“组合”来实现。也有一些特别的，比如http net/io 和 stream 分别启动goroutine read/write loop，通过共享数据来 变相的实现跨层调用
 
[SOFAMosn Introduction](https://github.com/sofastack/sofastack-doc/blob/master/sofa-mosn/zh_CN/docs/Introduction.md) 

![](/public/upload/go/mosn_io_process.png)

1. MOSN 在 IO 层读取数据，通过 read filter 将数据发送到 Protocol 层进行 Decode
2. Decode 出来的数据，根据不同的协议，**回调到 stream 层**，进行 stream 的创建和封装
3. stream 创建完毕后，会回调到 Proxy 层做路由和转发，Proxy 层会关联上下游（upstream,downstream）间的转发关系
4. Proxy 挑选到后端后，会根据后端使用的协议，将数据发送到对应协议的 Protocol 层，对数据重新做 Encode
5. Encode 后的数据会发经过 write filter 并最终使用 IO 的 write 发送出去

一个请求可能会触发多次 读取操作，因此单个请求可能会多次调用插件的onData 函数。






## 与control plan 的交互

`pkg/xds/v2/adssubscriber.go` 启动发送线程和接收线程

    func (adsClient *ADSClient) Start() {
        adsClient.StreamClient = adsClient.AdsConfig.GetStreamClient()
        utils.GoWithRecover(func() {
            adsClient.sendThread()
        }, nil)
        utils.GoWithRecover(func() {
            adsClient.receiveThread()
        }, nil)
    }

定时发送请求

    func (adsClient *ADSClient) sendThread() {
        refreshDelay := adsClient.AdsConfig.RefreshDelay
        t1 := time.NewTimer(*refreshDelay)
        for {
            select {
            ...
            case <-t1.C:
                err := adsClient.reqClusters(adsClient.StreamClient)
                if err != nil {
                    log.DefaultLogger.Infof("[xds] [ads client] send thread request cds fail!auto retry next period")
                    adsClient.reconnect()
                }
                t1.Reset(*refreshDelay)
            }
        }
    }

接收响应

    func (adsClient *ADSClient) receiveThread() {
        for {
            select {
        
            default:
                adsClient.StreamClientMutex.RLock()
                sc := adsClient.StreamClient
                adsClient.StreamClientMutex.RUnlock()
                ...
                resp, err := sc.Recv()
                ...
                typeURL := resp.TypeUrl
                HandleTypeURL(typeURL, adsClient, resp)
            }
        }
    }

处理逻辑是事先注册好的函数

    func HandleTypeURL(url string, client *ADSClient, resp *envoy_api_v2.DiscoveryResponse) {
        if f, ok := typeURLHandleFuncs[url]; ok {
            f(client, resp)
        }
    }
    func init() {
        RegisterTypeURLHandleFunc(EnvoyListener, HandleEnvoyListener)
        RegisterTypeURLHandleFunc(EnvoyCluster, HandleEnvoyCluster)
        RegisterTypeURLHandleFunc(EnvoyClusterLoadAssignment, HandleEnvoyClusterLoadAssignment)
        RegisterTypeURLHandleFunc(EnvoyRouteConfiguration, HandleEnvoyRouteConfiguration)
    }

以cluster 信息为例 HandleEnvoyCluster

    func HandleEnvoyCluster(client *ADSClient, resp *envoy_api_v2.DiscoveryResponse) {
        clusters := client.handleClustersResp(resp)
        ...
        conv.ConvertUpdateClusters(clusters)
        clusterNames := make([]string, 0)
        ...
        for _, cluster := range clusters {
            if cluster.GetType() == envoy_api_v2.Cluster_EDS {
                clusterNames = append(clusterNames, cluster.Name)
            }
        }
        ...
    }

会触发ClusterManager 更新cluster 

    func ConvertUpdateEndpoints(loadAssignments []*envoy_api_v2.ClusterLoadAssignment) error {
        for _, loadAssignment := range loadAssignments {
            clusterName := loadAssignment.ClusterName
            for _, endpoints := range loadAssignment.Endpoints {
                hosts := ConvertEndpointsConfig(&endpoints)
                clusterMngAdapter := clusterAdapter.GetClusterMngAdapterInstance()
                ...
                clusterAdapter.GetClusterMngAdapterInstance().TriggerClusterHostUpdate(clusterName, hosts); 
                ...
                
            }
        }
        return errGlobal
    }

## 学到的

不要硬看代码，尤其对于多协程程序

1. 打印日志
2. `debug.printStack` 来查看某一个方法之前的调用栈
3. `fmt.Printf("==> %T\n",xx)`  如果一个interface 有多个“实现类” 可以通过`%T` 查看struct 的类型



