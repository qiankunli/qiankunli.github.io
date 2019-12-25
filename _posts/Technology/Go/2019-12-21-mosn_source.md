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

可以学到一个代理程序可以玩多少花活儿

[Envoy 官方文档中文版](https://www.servicemesher.com/envoy/)

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

![](/public/upload/go/mosn_object.png)

1. 不同颜色 表示所处的 package 不同
2. 因为mosn主要是的用途是“代理”， 所以笔者一开始一直在找代理如何实现，但其实呢，mosn 首先是一个tcp server，像tomcat一样，mosn 主要分为连接管理和业务处理两个部分
3. 业务处理的入口 就是filterManager， 主要由`filterManager.onRead` 和 `filterManager.onWrite` 来时间。 filterManager 聚合ReadFilter 链和WriterFilter链，构成对数据的处理


**proxy 和 tcpproxy 是两种不同的proxy**

## 启动流程

该图主要说的连接管理部分

![](/public/upload/go/mosn_start.png)

## 数据处理

![](/public/upload/go/mosn_http_object.png)

1. 一定是先进行协议的解析，再进行router、cluster 等操作，因为可能根据协议的某个字段进行router
2. mosn 作为一个tcp server，从收到数据，转发数据到 Upstream 并拿到响应，再返回给请求方，整个流程都是高层 类定制好的，不同的协议只是实现对应的“子类”即可。

一次请求的处理过程

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

## 分层架构

### net/io层

`pkg/types/network.go` 的描述如下：

In mosn, we have 4 layers to build a mesh, net/io layer is the fundamental layer to support upper level's functionality.

    -----------------------
    |        PROXY          |
    -----------------------
    |       STREAMING       |
    -----------------------
    |        PROTOCOL       |
    -----------------------
    |         NET/IO        |
    -----------------------

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

Below is the basic relation on listener and connection:

    --------------------------------------------------
    |                                      			|
    | 	  EventListener       EventListener     		|
    |        *|                   |*          		    |
    |         |                   |       				|
    |        1|     1      *      |1          			|
    |	    Listener --------- Connection      			|
    |        1|      [accept]     |1          			|
    |         |                   |-----------         |
    |        *|                   |*          |*       |
    |	 ListenerFilter       ReadFilter  WriteFilter   |
    |                                                  |
    --------------------------------------------------

### Stream 层

`pkg/types/stream.go` 的描述如下：

The bunch of interfaces are structure skeleton to build a extensible stream multiplexing architecture. The core concept is mainly refer to golang HTTP2 and envoy.

In mosn, we have 4 layers to build a mesh, stream is the inheritance layer to bond protocol layer and proxy layer together.

    -----------------------
    |        PROXY          |
    -----------------------
    |       STREAMING       |
    -----------------------
    |        PROTOCOL       |
    -----------------------
    |         NET/IO        |
    -----------------------

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

Below is the basic relation on stream and connection:

    --------------------------------------------------------------------------------------------------------------
    |																												|
    |                              ConnPool                                                                         |  
    |                                 |1                                                                            |
    |                                 |                                                                             |
    |                                 |*                                                                            |
    |                               Client                                                                          |
    |                                 |1                                                                            |
    | 	  EventListener   			   |				StreamEventListener											|
    |        *|                       |                       |*													|
    |         |                       |                       |													    |
    |        1|        1    1  	   |1 		1        *     |1													    |
    |	    Connection -------- StreamConnection ---------- Stream													|
    |        1|                   	   |1				   	   |1                                                   |
    |		   |					   |				   	   |                                                    |
    |         |                   	   |					   |--------------------------------					|
    |        *|                   	   |					   |*           	 				|*					|
    |	 ConnectionFilter    		   |			      StreamSender      		        StreamReceiveListener	|
    |								   |*					   |1				 				|1					|
    |						StreamConnectionEventListener	   |				 				|					|
    |													       |*				 				|*					|
    |										 	 		StreamSenderFilter	   			StreamReceiverFilter	    |
    |													   	   |1								|1					|
    |													   	   |								|					|
    |													       |1								|1					|
    |										 		StreamSenderFilterHandler     StreamReceiverFilterHandler	    |
    |																												|
    --------------------------------------------------------------------------------------------------------------


## 学到的

不要硬看代码，通过打印日志、打断点的方式来 查看代码的执行链条




