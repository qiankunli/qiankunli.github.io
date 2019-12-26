---

layout: post
title: MOSN源码分析
category: 技术
tags: Go
keywords: Go

---

## 前言

* TOC
{:toc}

SOFAMosn是基于Go开发的sidecar，用于service mesh中的数据面代理[sofastack/sofa-mosn](https://github.com/sofastack/sofa-mosn)

阿里官方介绍文档[SOFAMosn Introduction](https://github.com/sofastack/sofastack-doc/blob/master/sofa-mosn/zh_CN/docs/Introduction.md)

[Envoy 官方文档中文版](https://www.servicemesher.com/envoy/)

[蚂蚁金服大规模微服务架构下的Service Mesh探索之路](https://www.servicemesher.com/blog/the-way-to-service-mesh-in-ant-financial/) 很不错的文章值得多看几遍。

理解mosn 主要有两个方向

1. 任何tcp server 都要处理的：网络io，拿到字节流后如何根据协议解析数据（协议层/encoder/decoder）。 mosn 的特别之处是 在Connection 和 协议层之间加了 Stream的概念（可能是为了兼容http2，在http1协议中这一层的实现就很单薄）
2. 代理业务特别需要的：loader balancer、router 等如何可插拔的 加入到代理 逻辑中。

## 手感

[使用 SOFAMosn 作为 HTTP 代理](https://github.com/sofastack/sofa-mosn/blob/master/examples/cn_readme/http-sample/README.md)

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


单机视角

![](/public/upload/go/mosn_http_example_single.png)

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

几乎所有的interface 定义在 `pkg/types` 中，mosn 基于四层 架构实现（见下文），每一个layer 在types 中有一个go 文件，在`pkg` 下有一个专门的文件夹。

## 分层架构

![](/public/upload/go/mosn_layer_process.png)

一般的服务端编程，二级制数据经过协议解析为 协议对应的model（比如HttpServletRequest） 进而交给上层业务方处理，对于mosn 

1. 协议上数据统一划分为header/data/Trailers 三个部分，转发也是以这三个子部分为基本单位
2. 借鉴了http2 的stream 的理念（所以Stream interface 上有一个方法是`ID()`），Stream 可以理解为一个子Connection，Stream 之间可以并行请求和响应，通过StreamId关联，用来实现在一个Connection 之上的“多路复用”。PS：为了连接数量与请求数量解耦。

代码的组织（pkg/stream,pkg/protocol,pkg/proxy）  跟上述架构是一致的。但是类的组织 就有点不明觉厉。

![](/public/upload/go/mosn_layer.png)

除了主线之外，listener 和filter 比较好理解：Method in listener will be called on event occur, but not effect the control flow.Filters are called on event occurs, it also returns a status to effect control flow. Currently 2 states are used: Continue to let it go, Stop to stop the control flow.

[SOFAMosn Introduction](https://github.com/sofastack/sofastack-doc/blob/master/sofa-mosn/zh_CN/docs/Introduction.md) **类与layer 之间不存在一一对应关系**，比如proxy package 中包含ReadFilter 和 StreamReceiveListener 实现，分别属于多个层次。 

![](/public/upload/go/mosn_io_process.png)

1. MOSN 在 IO 层读取数据，通过 read filter 将数据发送到 Protocol 层进行 Decode
2. Decode 出来的数据，根据不同的协议，**回调到 stream 层**，进行 stream 的创建和封装
3. stream 创建完毕后，会回调到 Proxy 层做路由和转发，Proxy 层会关联上下游（upstream,downstream）间的转发关系
4. Proxy 挑选到后端后，会根据后端使用的协议，将数据发送到对应协议的 Protocol 层，对数据重新做 Encode
5. Encode 后的数据会发经过 write filter 并最终使用 IO 的 write 发送出去


跨层次调用 通过注册 listener、filter  或者 直接跨层次类的“组合” 来实现，也有一些特别的，比如http net/io 和 stream 分别启动goroutine read/write loop，通过共享数据来 变相的实现跨层调用。此外，protocol 和 stream 两个layer 因和协议有关，不同协议之间实现差异很大，层次不是很清晰。

该表格自相矛盾，有待进一步完善。

|层次|interface|关键成员/方法|实现类|所在文件|
|---|---|---|---|---|
|Net/IO|Listener|OnAccept|activeListener|server/handler.go|
|Net/IO|Connection||connection|network/connection.go|								
|Net/IO|ReadFilter|OnData<br>OnNewConnection|proxy|proxy/proxy.go|
|衔接|StreamConnection|map[uint32]*serverStream<br>types.ProtocolEngine<br>方法：Dispatch|streamConnection|stream/http2/stream.go|
|Protocol|Decoder|Decode|serverCodec|protocol/http/codec.go|
|衔接|Stream|types.StreamReceiveListener<br>types.HeaderMap<br>*serverStreamConnection|serverStream|stream/http2/stream.go|   	
|proxy|StreamReceiveListener|OnReceive|downStream|proxy/downstream.go|


## 连接管理

该图主要说的连接管理部分

![](/public/upload/go/mosn_object.png)

1. 不同颜色 表示所处的 package 不同
2. 因为mosn主要是的用途是“代理”， 所以笔者一开始一直在找代理如何实现，但其实呢，mosn 首先是一个tcp server，像tomcat一样，mosn 主要分为连接管理和业务处理两个部分
3. 业务处理的入口 就是filterManager， 主要由`filterManager.onRead` 和 `filterManager.onWrite` 来实现。 filterManager 聚合ReadFilter 链和WriterFilter链，构成对数据的处理

![](/public/upload/go/mosn_start.png)


## 数据处理

![](/public/upload/go/mosn_http_object.png)

一次http1协议请求的处理过程

![](/public/upload/go/mosn_http_read.png)

绿色部分表示另起一个协程

Downstream stream, as a controller to handle downstream and upstream proxy flow `downStream.OnReceive` 逻辑

    func (s *downStream) OnReceive(ctx context.Context,..., data types.IoBuffer, ...) {
        ...
        pool.ScheduleAuto(func() {
            phase := types.InitPhase
            for i := 0; i < 10; i++ {
                s.cleanNotify()

                phase = s.receive(ctx, id, phase)
                switch phase {
                case types.End:
                    return
                case types.MatchRoute:
                    log.Proxy.Debugf(s.context, "[proxy] [downstream] redo match route %+v", s)
                case types.Retry:
                    log.Proxy.Debugf(s.context, "[proxy] [downstream] retry %+v", s)
                case types.UpFilter:
                    log.Proxy.Debugf(s.context, "[proxy] [downstream] directResponse %+v", s)
                }
            }
        }
    }

`pkg/types/proxy.go` 有phase 的定义

    type Phase int
    const (
        InitPhase Phase = iota
        DownFilter
        MatchRoute
        DownFilterAfterRoute
        DownRecvHeader
        DownRecvData
        DownRecvTrailer
        Oneway
        Retry
        WaitNofity
        UpFilter
        UpRecvHeader
        UpRecvData
        UpRecvTrailer
        End
    )

`downStream.receive` 会根据当前所处的phase 进行对应的处理


    func (s *downStream) receive(ctx context.Context, id uint32, phase types.Phase) types.Phase {
        for i := 0; i <= int(types.End-types.InitPhase); i++ {
            switch phase {
            // init phase
            case types.InitPhase:
                phase++
            // downstream filter before route
            case types.DownFilter:
                s.runReceiveFilters(phase, s.downstreamReqHeaders, s.downstreamReqDataBuf, s.downstreamReqTrailers)
                phase++
            // match route
            case types.MatchRoute:
                s.matchRoute()
                phase++
            // downstream filter after route
            case types.DownFilterAfterRoute:
                s.runReceiveFilters(phase, s.downstreamReqHeaders, s.downstreamReqDataBuf, s.downstreamReqTrailers)
                phase++
            // downstream receive header
            case types.DownRecvHeader:
                //check not null
                s.receiveHeaders(s.downstreamReqDataBuf == nil && s.downstreamReqTrailers == nil)
                phase++
            // downstream receive data
            case types.DownRecvData:
                //check not null
                s.receiveData(s.downstreamReqTrailers == nil)
                phase++
            // downstream receive trailer
            case types.DownRecvTrailer:
                // check not null
                s.receiveTrailers()
                phase++
            // downstream oneway
            case types.Oneway:
                ...
            case types.Retry:
                ...
            case types.WaitNofity:
                ...
            // upstream filter
            case types.UpFilter:
                s.runAppendFilters(phase, s.downstreamRespHeaders, s.downstreamRespDataBuf, s.downstreamRespTrailers)
                // maybe direct response
                phase++
            // upstream receive header
            case types.UpRecvHeader:
                // send downstream response
                // check not null
                s.upstreamRequest.receiveHeaders(s.downstreamRespDataBuf == nil && s.downstreamRespTrailers == nil)
                phase++
            // upstream receive data
            case types.UpRecvData:
                // check not null
                s.upstreamRequest.receiveData(s.downstreamRespTrailers == nil)
                phase++
            // upstream receive triler
            case types.UpRecvTrailer:
                //check not null
                s.upstreamRequest.receiveTrailers()
                phase++
            // process end
            case types.End:
                return types.End
            default:
                return types.End
            }
        }
        return types.End
    }

## 与control plan 的交互（未完成）

## 学到的

不要硬看代码，尤其对于多协程程序

1. 打印日志
2. `debug.printStack` 来查看某一个方法之前的调用栈




