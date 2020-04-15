---

layout: post
title: mosn源码浅析
category: 技术
tags: Mesh
keywords: mesh microservice

---

## 前言（未完成）

* TOC
{:toc}

## 初始化和启动

[MOSN 源码解析 - 启动流程](https://mosn.io/zh/blog/code/mosn-startup/)`Mosn := NewMosn(c)` 实例化了一个 Mosn 实例。`Mosn.Start()` 开始运行。我们先看 MOSN 的结构：

```go
type Mosn struct {
	servers        []server.Server
	clustermanager types.ClusterManager
	routerManager  types.RouterManager
	config         *v2.MOSNConfig
	adminServer    admin.Server
	xdsClient      *xds.Client
	wg             sync.WaitGroup
	// for smooth upgrade. reconfigure
	inheritListeners []net.Listener
	reconfigure      net.Conn
}
```

1. servers 是一个数组，server.Server 是接口类型。但是目前的代码逻辑中只会有一个 server。
2. clustermanager 顾名思义就是集群管理器。 `types.ClusterManager` 也是接口类型。这里的 cluster 指得是 MOSN 连接到的一组逻辑上相似的上游主机。MOSN 通过服务发现来发现集群中的成员，并通过主动运行状况检查来确定集群成员的健康状况。MOSN 如何将请求路由到集群成员由负载均衡策略确定。
3. routerManager 是路由管理器，MOSN 根据路由规则来对请求进行代理。
4. adminServer 是一个服务，可以通过 http 请求获取 MOSN 的配置、状态等等
5. xdsClient 是 xds 协议的客户端。关于 xds, Envoy 通过查询文件或管理服务器来动态发现资源。概括地讲，对应的发现服务及其相应的 API 被称作 xDS。mosn 也使用 xDS，这样就可以兼容 istio。
6. inheritListeners 和 reconfigure 都是为了实现 MOSN 的平滑升级和重启。具体参见[MOSN 源码解析 - 启动流程](https://mosn.io/zh/blog/code/mosn-startup/)

### 初始化

1. 初始化配置文件路径，日志，进程id路径，unix socket 路径，trace的开关（SOFATracer）以及日志插件。
2. 通过 server.GetInheritListeners() 来判断启动模式（普通启动或热升级/重启），并在热升级/重启的情况下继承旧 MOSN 的监听器文件描述符。
3. 如果是热升级/重启，则设置 Mosn 状态为 Active_Reconfiguring;如果是普通启动，则直接调用 StartService()，关于 StartService 会在之后分析。
4. 初始化指标服务。
5. 根据是否是 Xds 模式初始化配置。

    ||xds 模式|非 Xds 模式|
    |---|---|---|
    |clustermanager|使用 nil 来初始化|从配置文件中初始化|
    |routerManager|使用默认配置来实例化|初始化routerManager，<br>并从配置文件中读取路由配置更新|
    |server|使用默认配置来实例化|从配置文件中读取 listener 并添加|

### 启动

1. 启动 xdsClient, xdsClient 负责从 pilot 周期地拉取 listeners/clusters/clusterloadassignment 配置。这个特性使得用户可以通过 crd 来动态的改变 service mesh 中的策略。
2. 开始执行所有注册在 featuregate中 feature 的初始化函数。
3. 解析服务注册信息
4. MOSN 启动前的准备工作  beforeStart()
5. 正式启动 MOSN 的服务

    ```go
    for _, srv := range m.servers {
        utils.GoWithRecover(func() {
            srv.Start()
        }, nil)
    }
    ```
    对于当前来说，只有一个 server，这个 server 是在 NewMosn 中初始化的。

![](/public/upload/go/mosn_start.png)

## 整理逻辑



该图主要说的连接管理部分

![](/public/upload/go/mosn_object.png)

1. 不同颜色 表示所处的 package 不同
2. 因为mosn主要是的用途是“代理”， 所以笔者一开始一直在找代理如何实现，但其实呢，mosn 首先是一个tcp server，像tomcat一样，mosn 主要分为连接管理和业务处理两个部分
3. 业务处理的入口 就是filterManager， 主要由`filterManager.onRead` 和 `filterManager.onWrite` 来实现。 filterManager 聚合ReadFilter 链和WriterFilter链，构成对数据的处理


envoy 对应逻辑 [深入解读Service Mesh的数据面Envoy](https://sq.163yun.com/blog/article/213361303062011904)

![](/public/upload/mesh/envoy_new_connection.jpg)

## 数据处理

一些细节：

1. [SOFAMesh中的多协议通用解决方案x-protocol介绍系列(1)-DNS通用寻址方案](https://skyao.io/post/201809-xprotocol-common-address-solution/)iptables在劫持流量时，除了将请求转发到localhost的Sidecar处外，还额外的在请求报文的TCP options 中将 ClusterIP 保存为 original dest。在 Sidecar （Istio默认是Envoy）中，从请求报文 TCP options 的 original dest 处获取 ClusterIP
2. [SOFAMesh中的多协议通用解决方案x-protocol介绍系列(2)-快速解码转发](https://skyao.io/post/201809-xprotocol-rapid-decode-forward/)

    1. 转发请求时，由于涉及到负载均衡，我们需要将请求发送给多个服务器端实例。因此，有一个非常明确的要求：就是必须以单个请求为单位进行转发。即单个请求必须完整的转发给某台服务器端实例，负载均衡需要以请求为单位，不能将一个请求的多个报文包分别转发到不同的服务器端实例。所以，拆包是请求转发的必备基础。
    2. 多路复用的关键参数：RequestId。RequestId用来关联request和对应的response，请求报文中携带一个唯一的id值，应答报文中原值返回，以便在处理response时可以找到对应的request。当然在不同协议中，这个参数的名字可能不同（如streamid等）。严格说，RequestId对于请求转发是可选的，也有很多通讯协议不提供支持，比如经典的HTTP1.1就没有支持。但是如果有这个参数，则可以实现多路复用，从而可以大幅度提高TCP连接的使用效率，避免出现大量连接。稍微新一点的通讯协议，基本都会原生支持这个特性，比如SOFARPC，Dubbo，HSF，还有HTTP/2就直接內建了多路复用的支持。

我们可以总结到，对于Sidecar，要正确转发请求：

1. 必须获取到destination信息，得到转发的目的地，才能进行服务发现类的寻址
2. 必须要能够正确的拆包，然后以请求为单位进行转发，这是负载均衡的基础
3. 可选的RequestId，这是开启多路复用的基础

[深入解读Service Mesh的数据面Envoy](https://sq.163yun.com/blog/article/213361303062011904)下文以envoy 实现做一下类比 用来辅助理解mosn 相关代码的理念：

![](/public/upload/mesh/envoy_on_data.jpg)

对于每一个Filter，都调用onData函数，咱们上面解析过，其中HTTP对应的ReadFilter是ConnectionManagerImpl，因而调用ConnectionManagerImpl::onData函数。ConnectionManager 是协议插件的处理入口，**同时也负责对整个处理过程的流程编排**。

![](/public/upload/mesh/envoy_data_parse.jpg)

### 数据“上传”

一次http1协议请求的处理过程

![](/public/upload/go/mosn_http_read.png)

绿色部分表示另起一个协程

### 转发流程

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


`pkg/types/proxy.go` 有phase 的定义

|phase|对应方法|执行逻辑（部分）|
|---|---|---|
|InitPhase|||
|DownFilter|runReceiveFilters|
|MatchRoute|matchRoute|
|DownFilterAfterRoute|runReceiveFilters||
|DownRecvHeader|receiveHeaders|==> upstreamRequest.appendHeaders|
|DownRecvData|receiveData|==> upstreamRequest.appendData|
|DownRecvTrailer|receiveTrailers|==> upstreamRequest.appendTrailers()|
|Oneway/Retry/WaitNofity||
|UpFilter|runAppendFilters|
|UpRecvHeader|upstreamRequest.receiveHeaders|==> downStream.onUpstreamData|
|UpRecvData|upstreamRequest.receiveData|==> downStream.onUpstreamData|
|UpRecvTrailer|upstreamRequest.receiveTrailers|==> downStream.onUpstreamTrailers|
|End||

上述流程才像是一个 proxy 层的活儿，请求转发到 upstream，从upstream 拿到响应， 再转回给downStream