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

[Envoy 官方文档中文版](https://www.servicemesher.com/envoy/)

理解mosn 主要有两个方向

1. 任何tcp server 都要处理的：网络io，拿到字节流后如何根据协议解析数据（协议层/encoder/decoder）。 mosn 的特别之处是 在Connection 和 协议层之间加了 Stream（可能是为了兼容http2）
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

几乎所有的interface 定义在 `pkg/types` 中，mosn 基于四层 架构实现（见下文），每一个layer 在types 中有一个go 文件，在`pkg` 下有一个专门的文件夹。因为是支持多协议的，所以每一个layer 协议的自定义部分 又会独立拆分文件夹。

## 连接管理

该图主要说的连接管理部分

![](/public/upload/go/mosn_object.png)

1. 不同颜色 表示所处的 package 不同
2. 因为mosn主要是的用途是“代理”， 所以笔者一开始一直在找代理如何实现，但其实呢，mosn 首先是一个tcp server，像tomcat一样，mosn 主要分为连接管理和业务处理两个部分
3. 业务处理的入口 就是filterManager， 主要由`filterManager.onRead` 和 `filterManager.onWrite` 来实现。 filterManager 聚合ReadFilter 链和WriterFilter链，构成对数据的处理

![](/public/upload/go/mosn_start.png)

## 数据处理

![](/public/upload/go/mosn_http_object.png)

1. 一定是先进行协议的解析，再进行router、cluster 等操作，因为可能根据协议的某个字段进行router
2. mosn 作为一个tcp server，从收到数据，转发数据到 Upstream 并拿到响应，再返回给请求方，整个流程都是高层 类定制好的，不同的协议只是实现对应的“子类”即可。

一次http请求的处理过程

![](/public/upload/go/mosn_http_read.png)

`downStream.OnReceive` 逻辑

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

浏览器发出一次 `http://localhost:2045` 对应的debug 日志

    2019-12-25 11:08:47,169 [DEBUG] [server] [listener] accept connection from 127.0.0.1:2045, condId= 3, remote addr:127.0.0.1:62683
    2019-12-25 11:08:47,171 [DEBUG] [network] [check use writeloop] Connection = 3, Local Address = 127.0.0.1:2045, Remote Address = 127.0.0.1:62683
    2019-12-25 11:08:47,167 [DEBUG] [server] [listener] accept connection from 127.0.0.1:2045, condId= 2, remote addr:127.0.0.1:62681
    2019-12-25 11:08:47,189 [DEBUG] [network] [check use writeloop] Connection = 2, Local Address = 127.0.0.1:2045, Remote Address = 127.0.0.1:62681
    2019-12-25 11:08:47,205 [DEBUG] [2,c0a87072157724332720510012708] [stream] [http] new stream detect, requestId = 1
    2019-12-25 11:08:47,208 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] new stream, proxyId = 1 , requestId =1, oneway=false
    2019-12-25 11:08:47,212 [DEBUG] [2,c0a87072157724332720510012708] [proxy][downstream] 0 stream filters in config
    2019-12-25 11:08:47,214 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] OnReceive headers:GET / HTTP/1.1
    User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36
    Host: localhost:2045
    ...
    , data:<nil>, trailers:<nil>
    2019-12-25 11:08:47,220 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] enter phase 1, proxyId = 1
    2019-12-25 11:08:47,221 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] enter phase 2, proxyId = 1
    2019-12-25 11:08:47,224 [DEBUG] [router] [routers] [MatchRoute] GET / HTTP/1.1
    User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36
    Host: localhost:2045
    ...

    2019-12-25 11:08:47,224 [DEBUG] [router] [routers] [findVirtualHost] found default virtual host only
    2019-12-25 11:08:47,224 [DEBUG] [router] [config utility] [try match header] GET / HTTP/1.1
    User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36
    Host: localhost:2045
    ...

    2019-12-25 11:08:47,226 [DEBUG] [2,c0a87072157724332720510012708] [router] [DefaultHandklerChain] [MatchRoute] matched a route: &{0xc0003c7000 /}
    2019-12-25 11:08:47,226 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] enter phase 3, proxyId = 1
    2019-12-25 11:08:47,227 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] enter phase 4, proxyId = 1
    2019-12-25 11:08:47,227 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] route match result:&{RouteRuleImplBase:0xc0003c7000 prefix:/}, clusterName=clientCluster
    2019-12-25 11:08:47,227 [DEBUG] [upstream] [cluster manager] clusterSnapshot.loadbalancer.ChooseHost result is 127.0.0.1:2046, cluster name = clientCluster
    2019-12-25 11:08:47,227 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] timeout info: {GlobalTimeout:1m0s TryTimeout:0s}
    2019-12-25 11:08:47,227 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [upstream] append headers: GET / HTTP/1.1
    User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36
    Host: localhost:2045
    ...

    2019-12-25 11:08:47,230 [DEBUG] [network] [check use writeloop] Connection = 4, Local Address = 127.0.0.1:62686, Remote Address = 127.0.0.1:2046
    2019-12-25 11:08:47,231 [DEBUG] [network] [client connection connect] connect raw tcp, remote address = 127.0.0.1:2046 ,event = ConnectedFlag, error = <nil>
    2019-12-25 11:08:47,232 [DEBUG] [5,-] new http2 server stream connection
    2019-12-25 11:08:47,232 [DEBUG] [server] [listener] accept connection from 127.0.0.1:2046, condId= 5, remote addr:127.0.0.1:62686
    2019-12-25 11:08:47,232 [DEBUG] [network] [check use writeloop] Connection = 5, Local Address = 127.0.0.1:2046, Remote Address = 127.0.0.1:62686
    2019-12-25 11:08:47,233 [ERROR] [normal] [4,-] new http2 client stream connection
    2019-12-25 11:08:47,233 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [upstream] connPool ready, proxyId = 1, host = 127.0.0.1:2046
    2019-12-25 11:08:47,234 [DEBUG] [2,c0a87072157724332720510012708] http2 client AppendHeaders: id = 0, headers = map[Accept:[text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,...]]
    2019-12-25 11:08:47,235 [DEBUG] [2,c0a87072157724332720510012708] http2 client SendRequest id = 1
    2019-12-25 11:08:47,235 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] start a request timeout timer
    2019-12-25 11:08:47,235 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] enter phase 9, proxyId = 1
    2019-12-25 11:08:47,235 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] waitNotify begin 0xc0004ce000, proxyId = 1
    2019-12-25 11:08:47,238 [DEBUG] [5,-] [proxy] [downstream] new stream, proxyId = 2 , requestId =1, oneway=false
    2019-12-25 11:08:47,238 [DEBUG] [5,-] [proxy][downstream] 0 stream filters in config
    2019-12-25 11:08:47,238 [DEBUG] [5,-] http2 server header: 1, map[Accept:[text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,...]]
    2019-12-25 11:08:47,238 [DEBUG] [5,-] [proxy] [downstream] OnReceive headers:&{HeaderMap:0xc000336040 Req:0xc000378100}, data:<nil>, trailers:<nil>
    2019-12-25 11:08:47,238 [DEBUG] [5,-] [proxy] [downstream] enter phase 1, proxyId = 2
    2019-12-25 11:08:47,238 [DEBUG] [5,-] [proxy] [downstream] enter phase 2, proxyId = 2
    2019-12-25 11:08:47,238 [DEBUG] [router] [routers] [MatchRoute] &{0xc000336040 0xc000378100}
    2019-12-25 11:08:47,238 [DEBUG] [router] [routers] [findVirtualHost] found default virtual host only
    2019-12-25 11:08:47,238 [DEBUG] [router] [config utility] [try match header] &{0xc000336040 0xc000378100}
    2019-12-25 11:08:47,238 [DEBUG] [5,-] [router] [DefaultHandklerChain] [MatchRoute] matched a route: &{0xc0003c6e00 /}
    2019-12-25 11:08:47,238 [DEBUG] [5,-] [proxy] [downstream] enter phase 3, proxyId = 2
    2019-12-25 11:08:47,238 [DEBUG] [5,-] [proxy] [downstream] enter phase 4, proxyId = 2
    2019-12-25 11:08:47,238 [DEBUG] [5,-] [proxy] [downstream] route match result:&{RouteRuleImplBase:0xc0003c6e00 prefix:/}, clusterName=serverCluster
    2019-12-25 11:08:47,238 [DEBUG] [upstream] [cluster manager] clusterSnapshot.loadbalancer.ChooseHost result is 127.0.0.1:8080, cluster name = serverCluster
    2019-12-25 11:08:47,238 [DEBUG] [5,-] [proxy] [downstream] timeout info: {GlobalTimeout:1m0s TryTimeout:0s}
    2019-12-25 11:08:47,238 [DEBUG] [5,-] [proxy] [upstream] append headers: &{HeaderMap:0xc000336040 Req:0xc000378100}
    2019-12-25 11:08:47,242 [DEBUG] [network] [check use writeloop] Connection = 6, Local Address = 127.0.0.1:62687, Remote Address = 127.0.0.1:8080
    2019-12-25 11:08:47,242 [DEBUG] [network] [client connection connect] connect raw tcp, remote address = 127.0.0.1:8080 ,event = ConnectedFlag, error = <nil>
    2019-12-25 11:08:47,242 [DEBUG] client OnEvent ConnectedFlag, connected false
    2019-12-25 11:08:47,243 [DEBUG] [5,-] [proxy] [upstream] connPool ready, proxyId = 2, host = 127.0.0.1:8080
    2019-12-25 11:08:47,243 [DEBUG] [5,-] [stream] [http] send client request, requestId = 2
    2019-12-25 11:08:47,243 [DEBUG] [5,-] [proxy] [downstream] start a request timeout timer
    2019-12-25 11:08:47,243 [DEBUG] [5,-] [proxy] [downstream] enter phase 9, proxyId = 2
    2019-12-25 11:08:47,243 [DEBUG] [5,-] [proxy] [downstream] waitNotify begin 0xc0004ce300, proxyId = 2
    2019-12-25 11:08:47,244 [DEBUG] [5,-] [stream] [http] receive response, requestId = 2
    2019-12-25 11:08:47,244 [DEBUG] [5,-] [proxy] [upstream] OnReceive headers: HTTP/1.1 200 OK
    Date: Wed, 25 Dec 2019 03:08:46 GMT
    Content-Type: text/plain
    ...

    2019-12-25 11:08:47,245 [DEBUG] [5,-] [proxy] [downstream] enter phase 10, proxyId = 2
    2019-12-25 11:08:47,245 [DEBUG] [5,-] [proxy] [downstream] enter phase 11, proxyId = 2
    2019-12-25 11:08:47,245 [DEBUG] [5,-] http2 server ApppendHeaders id = 1, headers = map[Content-Length:[814] Content-Type:[text/plain] Date:[Wed, 25 Dec 2019 03:08:47 GMT]]
    2019-12-25 11:08:47,245 [DEBUG] [5,-] [proxy] [downstream] enter phase 12, proxyId = 2
    2019-12-25 11:08:47,245 [DEBUG] [5,-] http2 server ApppendData id = 1
    2019-12-25 11:08:47,246 [DEBUG] [5,-] http2 server SendResponse id = 1
    2019-12-25 11:08:47,247 [DEBUG] [2,c0a87072157724332720510012708] http2 client header: id = 1, headers = map[Content-Length:[814] Content-Type:[text/plain] Date:[Wed, 25 Dec 2019 03:08:47 GMT] X-Mosn-Status:[200]]
    2019-12-25 11:08:47,247 [DEBUG] [2,c0a87072157724332720510012708] http2 client receive data: id = 1
    2019-12-25 11:08:47,248 [DEBUG] [2,c0a87072157724332720510012708] http2 client data: id = 1
    2019-12-25 11:08:47,246 [DEBUG] [5,-] [proxy] [downstream] giveStream 0xc0004ce300 &{ID:2 proxy:0xc0000b60d0 route:0xc00046eae0 cluster:0xc0003fb680 element:0xc00032cc90 bufferLimit:0 timeout:{GlobalTimeout:60000000000 TryTimeout:0} retryState:0xc000340460 requestInfo:0xc0004ce530 responseSender:0xc0000d57c0 upstreamRequest:0xc0004ce4a8 perRetryTimer:<nil> responseTimer:<nil> downstreamReqHeaders:0xc00016e480 downstreamReqDataBuf:<nil> downstreamReqTrailers:<nil> downstreamRespHeaders:{ResponseHeader:0xc0001d2258 EmptyValueHeaders:map[]} downstreamRespDataBuf:0xc00003c600 downstreamRespTrailers:<nil> downstreamResponseStarted:true downstreamRecvDone:true upstreamRequestSent:true upstreamProcessDone:true noConvert:false directResponse:false oneway:false notify:0xc00035c360 downstreamReset:0 downstreamCleaned:1 upstreamReset:0 reuseBuffer:1 resetReason: senderFilters:[] senderFiltersIndex:0 receiverFilters:[] receiverFiltersIndex:0 receiverFiltersAgain:false context:0xc0000f8500 streamAccessLogs:[] logDone:1 snapshot:0xc00043a180}
    2019-12-25 11:08:47,248 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [upstream] OnReceive headers: &{HeaderMap:0xc0000ca030 Rsp:0xc0003ae120}, data: , trailers: <nil>
    2019-12-25 11:08:47,248 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] OnReceive send downstream response &{HeaderMap:0xc0000ca030 Rsp:0xc0003ae120}
    2019-12-25 11:08:47,248 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] enter phase 10, proxyId = 1
    2019-12-25 11:08:47,248 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] enter phase 11, proxyId = 1
    2019-12-25 11:08:47,248 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] enter phase 12, proxyId = 1
    2019-12-25 11:08:47,248 [DEBUG] [2,c0a87072157724332720510012708] [stream] [http] send server response, requestId = 1
    2019-12-25 11:08:47,248 [DEBUG] [2,c0a87072157724332720510012708] [proxy] [downstream] giveStream 0xc0004ce000 &{ID:1 proxy:0xc00048c000 route:0xc00046edc0 cluster:0xc000438000 element:0xc000322bd0 bufferLimit:0 timeout:{GlobalTimeout:60000000000 TryTimeout:0} retryState:0xc000340000 requestInfo:0xc0004ce230 responseSender:0xc0004bc000 upstreamRequest:0xc0004ce1a8 perRetryTimer:<nil> responseTimer:<nil> downstreamReqHeaders:{RequestHeader:0xc0004bc088 EmptyValueHeaders:map[]} downstreamReqDataBuf:<nil> downstreamReqTrailers:<nil> downstreamRespHeaders:0xc00018e110 downstreamRespDataBuf:0xc00012c3c0 downstreamRespTrailers:<nil> downstreamResponseStarted:true downstreamRecvDone:true upstreamRequestSent:true upstreamProcessDone:true noConvert:false directResponse:false oneway:false notify:0xc0000a89c0 downstreamReset:0 downstreamCleaned:1 upstreamReset:0 reuseBuffer:1 resetReason: senderFilters:[] senderFiltersIndex:0 receiverFilters:[] receiverFiltersIndex:0 receiverFiltersAgain:false context:0xc000486140 streamAccessLogs:[] logDone:1 snapshot:0xc00043a300}

## 分层架构


    -----------------------
    |        PROXY          |
    -----------------------
    |       STREAMING       |
    -----------------------
    |        PROTOCOL       |
    -----------------------
    |         NET/IO        |
    -----------------------

In mosn, we have 4 layers to build a mesh,

1. net/io layer is the fundamental layer to support upper level's functionality.
2. protocol is the core layer to do protocol related encode/decode.
3. stream is the inheritance layer to bond protocol layer and proxy layer together.Stream layer leverages protocol's ability to do binary-model conversation. In detail, Stream uses Protocols's encode/decode facade method and DecodeFilter to receive decode event call.

代码的组织  跟上述架构是一致的

![](/public/upload/go/mosn_layer.png)

### net/io层

`pkg/types/network.go` 的描述如下：

Core model in network layer are listener and connection. Listener listens specified port, waiting for new connections.
Both listener and connection have a extension mechanism, implemented as listener and filter chain, which are used to fill in customized logic.

1. Event listeners are used to subscribe important event of Listener and Connection. Method in listener will be called on event occur, but **not effect the control flow**.
2. Filters are called on event occurs, it also returns a status to effect control flow. Currently 2 states are used: Continue to let it go, Stop to stop the control flow.Filter has a callback handler to interactive with core model. For example, ReadFilterCallbacks can be used to continue filter chain in connection, on which is in a stopped state.

    Listener:
        - Event listener
            - ListenerEventListener
    - Filter
            - ListenerFilter
    Connection:
        - Event listener
            - ConnectionEventListener
        - Filter
            - ReadFilter
            - WriteFilter

### Stream 层

`pkg/types/stream.go` 的描述如下：

The bunch of interfaces are structure skeleton to build a extensible stream multiplexing architecture. The core concept is mainly refer to golang HTTP2 and envoy.

Core model in stream layer is stream, which manages process of a round-trip, a request and a corresponding response.
Event listeners can be installed into a stream to monitor event.Stream has two related models, encoder and decoder:

- StreamSender: a sender encodes request/response to binary and sends it out, flag 'endStream' means data is ready to sendout, no need to wait for further input.
- StreamReceiveListener: It's more like a decode listener to get called on a receiver receives binary and decodes to a request/response.
- Stream does not have a predetermined direction, so StreamSender could be a request encoder as a client or a response encoder as a server. It's just about the scenario, so does StreamReceiveListener.

    Stream:
    - Encoder
            - StreamSender
        - Decoder
            - StreamReceiveListener

    Event listeners:
        - StreamEventListener: listen stream event: reset, destroy.
        - StreamConnectionEventListener: listen stream connection event: goaway.

In order to meet the expansion requirements in the stream processing, StreamSenderFilter and StreamReceiverFilter are introduced as a filter chain in encode/decode process.Filter's method will be called on corresponding stream process stage and returns a status(Continue/Stop) to effect the control flow.

From an abstract perspective, stream represents a virtual process on underlying connection. To make stream interactive with connection, some intermediate object can be used.**StreamConnection is the core model to connect connection system to stream system**. As a example, when proxy reads binary data from connection, it dispatches data to StreamConnection to do protocol decode.
Specifically, ClientStreamConnection uses a NewStream to exchange StreamReceiveListener with StreamSender.
Engine provides a callbacks(StreamSenderFilterHandler/StreamReceiverFilterHandler) to let filter interact with stream engine.
As a example, a encoder filter stopped the encode process, it can continue it by StreamSenderFilterHandler.ContinueSending later. Actually, a filter engine is a encoder/decoder itself.

## 与control plan 的交互（未完成）

## 学到的

不要硬看代码，通过打印日志、打断点的方式来 查看代码的执行链条




