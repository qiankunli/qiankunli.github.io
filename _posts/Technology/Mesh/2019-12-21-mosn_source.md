---

layout: post
title: mosn源码浅析
category: 技术
tags: Mesh
keywords: mesh source

---

## 前言

* TOC
{:toc}

## 先聊聊七层负载均衡

[现代网络负载均衡与代理（上）](https://mp.weixin.qq.com/s/FwuEUAKU245tCa-UNtVYLw) 客户端建立一个到负载均衡 器的 TCP 连接。负载均衡器**终结该连接**(即直接响应 SYN)，然后选择一个后端，并与该后端建立一个新的 TCP 连接(即发送一个新的 SYN)。四层负载均衡器通常只在四层 TCP/UDP 连接/会话级别上运行。因此， 负载均衡器通过转发数据，并确保来自同一会话的字节在同一后端结束。四层负载均衡器 不知道它正在转发数据的任何应用程序细节。数据内容可以是 HTTP, Redis, MongoDB，或任 何应用协议。

四层负载均衡有哪些缺点是七层(应用)负载均衡来解决的呢? 假如两个 gRPC/HTTP2 客户端通过四层负载均衡器连接想要与一个后端通信。四层负载均衡器为每个入站 TCP 连接创建一个出站的 TCP 连接，从而产生两个入站和两个出站的连接（CA ==> loadbalancer ==> SA, CB ==> loadbalancer ==> SB）。假设，客户端 A 每分钟发送 1 个请求，而客户端 B 每秒发送 50 个请求，则SA 的负载是 SB的 50倍。所以四层负载均衡器问题随着时 间的推移变得越来越不均衡。

![](/public/upload/network/seven_layer_load_balance.jpeg)

上图 显示了一个七层 HTTP/2 负载均衡器。在本例中，客户端创建一个到负载均衡器的HTTP/2 TCP 连接。负载均衡器创建连接到两个后端。当客户端向负载均衡器发送两个HTTP/2 流时，流 1 被发送到后端 1，流 2 被发送到后端 2。因此，即使请求负载有很大差 异的客户端也会在后端之间实现高效地分发。这就是为什么七层负载均衡对现代协议如此 重要的原因。对于mosn来说，还支持协议转换，比如client mosn 之间是http，mosn 与server 之间是 grpc 协议。

## 初始化和启动

[MOSN 源码解析 - 启动流程](https://mosn.io/zh/blog/code/mosn-startup/)`Mosn := NewMosn(c)` 实例化了一个 Mosn 实例。`Mosn.Start()` 开始运行。我们先看 MOSN 的结构：

```go
// mosn.io/mosn/pkg/mosn/starter.go
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

1. servers 是一个数组，server.Server 是接口类型。**目前最多只支持配置一个server**。
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

## 协议无关L4 数据接收

activeListener.Start ==> listener.Start ==> listener.acceptEventLoop ==> listener.accept ==> activeListener.OnAccept ==> activeRawConn.ContinueFilterChain ==> activeListener.newConnection ==> activeListener.OnNewConnection ==> connection.Start ==> connection.startRWLoop ==> connection.startReadLoop and startWriteLoop 以read 为例 ==> connection.doRead ==> connection.onRead ==> filterManager.OnRead ==> filterManager.onContinueReading ==> proxy.OnData 

上述链路与源码位置结合且省略一下

```
mosn.io/mosn/pkg/
    server/handler.go           // activeListener.Start ==> activeListener.OnAccept ==> activeListener.OnNewConnection
    network/connection.go       // connection.Start ==> connection.onRead
           /filterManager.go    // filterManager.OnRead
    proxy/proxy.go              // proxy.OnData 
```

如果mosn 只支持四层负载均衡的话，到proxy.OnData  就可以可以通过 负载均衡 选择一个下游 节点发送数据了。mosn 支持七层负载均衡，L7就涉及到 协议处理了。

```go
func (c *connection) startReadLoop() {
	for {
		select {
		case <-c.readEnabledChan:
		default:
			//真正的读取数据逻辑在这里
			err := c.doRead()
		}
	}
}
func (c *connection) doRead() (err error) {
    //为该连接创建一个buffer来保存读入的数据
	//从连接中读取数据，返回实际读取到的字节数，rawConnection对应的就是原始连接
	bytesRead, err = c.readBuffer.ReadOnce(c.rawConnection)
	//通知上层读取到了新的数据
	c.onRead()
}
func (c *connection) onRead() {
	//filterManager过滤器管理者，把读取到的数据交给过滤器链路进行处理
	c.filterManager.OnRead()
}
func (fm *filterManager) onContinueReading(filter *activeReadFilter) {
	//这里可以清楚的看到网络层读取到数据以后，通过filterManager把数据交给整个过滤器链路处理
	for ; index < len(fm.upstreamFilters); index++ {
		uf = fm.upstreamFilters[index]
        //针对还没有初始化的过滤器回调其初始化方法OnNewConnection
		buf := fm.conn.GetReadBuffer()	
        //通知过滤器进行处理
        status := uf.filter.OnData(buf)
        if status == api.Stop {
            return
        }
	}
}
```

network 层的read filter 过滤器对应的实现就在proxy.go文件中
```go
func (p *proxy) OnData(buf buffer.IoBuffer) api.FilterStatus {
    //针对使用的协议类型初始化serverStreamConn
	if p.serverStreamConn == nil {
		protocol, err := stream.SelectStreamFactoryProtocol(p.context, prot, buf.Bytes())
		p.serverStreamConn = stream.CreateServerStreamConnection(p.context, protocol, p.readCallbacks.Connection(), p)
	}
	//把数据分发到对应协议的的解码器
	p.serverStreamConn.Dispatch(buf)
}
```

## L7 基于多路复用的转发 

```
mosn.io/mosn/pkg
    /types/stream.go        // Stream is a generic protocol stream  定义了stream 层的很多接口
    /stream
        /http/stream.go
        /http2/stream.go
        /xprotocol
            /stream.go
            /conn.go
    /stream.go      // 通用/父类实现
    /proxy
        /downstream.go
        /upstream.go
```


[MOSN 多协议机制解析](https://mosn.io/zh/blog/posts/multi-protocol-deep-dive/) MOSN 的底层机制与 Envoy、Nginx 并没有核心差异，同样支持基于 I/O 多路复用的 L4 读写过滤器扩展，并在此基础之上再封装 L7 的处理。但是与前两者不同的是，**MOSN 针对典型的微服务通信场景**，抽象出了一套适用于基于**多路复用** RPC 协议的扩展框架。

![](/public/upload/mesh/mosn_protocol.png)

多路复用的定义：允许在单条链接上，并发处理多个请求/响应。 对于http2 来说，在connection 之上专门提了一个stream 概念 以表达多路复用，一般rpc 框架则 只是通过`<requestId,Request> ` 将请求暂存起来，当收到 响应时，从Response 中提取requestId 就可以与 Request建立关联。这样rpc 框架不用 发一个Request 收到Response 之后再发下一个 Request。像http2 一样，**mosn 也专门显式化了 Stream 的概念**（以及一个stream 包含header/data/trailer 3个Frame），dubbo 实现中，mosn 的Steam.streamId 就是 dubbo Frame.Header.Id 

![](/public/upload/mesh/mosn_multiplexing.png)

1. MOSN 从 downstream(conn=2) 接收了一个请求 request，依据报文扩展多路复用接口 GetRequestId 获取到请求在这条连接上的身份标识(requestId=1)，并记录到关联映射中待用；
2. 请求经过 MOSN 的路由、负载均衡处理，选择了一个 upstream(conn=5)，同时在这条链接上新建了一个请求流(requestId=30)，并调用文扩展多路复用接口 SetRequestId 封装新的身份标识，并记录到关联映射中与 downstream 信息组合；
3. MOSN 从 upstream(conn=5) 接收了一个响应 response，依据报文扩展多路复用接口 GetRequestId 获取到请求在这条连接上的身份标识(requestId=30)。此时可以从上下游关联映射表中，根据 upstream 信息(connId=5, requestId=30) 找到对应的 downstream 信息(connId=2, requestId=1)；
4. 依据 downstream request 的信息，调用文扩展多路复用接口 SetRequestId 设置响应的 requestId，并回复给 downstream；

[MOSN 源码解析 - 协程模型](https://mosn.io/zh/blog/code/mosn-eventloop/)

mosn 作为一个七层代理，其核心工作就是转发，L7 层转发支持http、http2  和针对微服务场景xprotocol。 
1. mosn proxy **架设了基于多路复用/Stream机制的转发**：多路复用由Stream 概念表示，一个 请求/响应 对应多个frame（至少包含header 和 data 2个frame）。哪怕http 不是多路复用也 迁就了这一套约定。在proxy包中，转发逻辑由 downstream.go 和 upstream.go 完成，**各个协议不需要自己实现转发逻辑，只需要向 mosn 的Stream 机制靠拢即可**：实现ServerStreamConnection 和 ClientStreamConnection interface
2. 对于微服务框架，xprotocol 进一步的封装了功能代码，各rpc 协议只需实现xprotocol.XProtocol interface。

![](/public/upload/mesh/mosn_process.png)

从下游接收请求 handleRequest：proxy.onData ==> xprotocol.streamConn(serverStreamConnection).Dispatch ==> xprotocol.streamConn.handleFrame ==> xprotocol.streamConn.handleRequest ==> create serverStream(xStream) 并关联 downStream ==> downStream.OnReceive ==> downStream.receive ==> downStream.matchRoute ==> downStream.chooseHost 确定下游主机 ==> downStream.receiveHeaders/upstreamRequest.appendHeaders/xprotocol.xStream.AppendHeaders ==> downStream.receiveData/upstreamRequest.appendData/xprotocol.xStream.AppendData ==>  downStream.receiveTrailers/upstreamRequest.appendTrailers/xprotocol.xStream.AppendTrailers ==> xprotocol.xStream.endStream ==> buf = xprotocol.xStream.streamConn.protocol.Encode(frame) ==> xprotocol.xStream.streamConn.netConn.Write(buf)  转发暂停

从上游接收响应 handleResponse：client.onData ==> xprotocol.streamConn(clientStreamConnection).Dispatch ==> xprotocol.streamConn.handleFrame ==> xprotocol.streamConn.handleResponse ==> clientStream = `xprotocol.streamConn.clientStreams[requestId]` ==> clientStreamReceiverWrapper.onReceive ==>  upstreamRequest.OnReceive ==> downStream.sendNotify ==>  接收协程从先前中断的地方继续 downStream.receive  ==> upstreamRequest.receiveHeaders/downStream.appendHeaders/xprotocol.xStream.AppendHeaders ==> upstreamRequest.receiveData/downStream.appendData/xprotocol.xStream.AppendData ==>  upstreamRequest.receiveTrailers/downStream.appendTrailers/xprotocol.xStream.AppendTrailers ==> xprotocol.xStream.endStream ==> buf = xprotocol.xStream.streamConn.protocol.Encode(frame) ==> xprotocol.xStream.streamConn.netConn.Write(buf)  

两个体会：

1. 可以看到 基于io事件 如何实现了转发逻辑
2. **网络数据 从字节数组都frame 再到协议对象 都是实际存在的，Connection/StreamConnection/Stream 则都是 通信两端 为维护数据状态 而产生的**。

以官方 dubbo example 运行为例： dubbo consumer  ==> client mosn `localhost:2045` ==> server mosn `localhost:2046` ==> dubbo provider `0.0.0.0:20880`

```
[DEBUG] new idlechecker: maxIdleCount:6, conn:2
// 收到client connection=2；
[DEBUG] [server] [listener] accept connection from 0.0.0.0:2045, condId= 2, remote addr:192.168.104.18:60625
[DEBUG] [2,-] [stream] [xprotocol] new stream detect, requestId = 0
[DEBUG] [2,] [proxy] [downstream] new stream, proxyId = 1 , requestId =0, oneway=false
[DEBUG] [2,] [proxy] [downstream] 0 stream filters in config
// receive header frame；
[DEBUG] [2,] [proxy] [downstream] OnReceive headers   跟上header 的具体内容
[DEBUG] [2,] [proxy] [downstream] enter phase DownFilter[1], proxyId = 1
[DEBUG] [2,] [proxy] [downstream] enter phase MatchRoute[2], proxyId = 1
[DEBUG] [router] [routers] [MatchRoute]
[DEBUG] [router] [routers] [findVirtualHost] found default virtual host only
[DEBUG] [2,] [router] [DefaultHandklerChain] [MatchRoute] matched a route: &{0xc00023d900 .*}
[DEBUG] [2,] [proxy] [downstream] enter phase DownFilterAfterRoute[3], proxyId = 1
[DEBUG] [2,] [proxy] [downstream] enter phase ChooseHost[4], proxyId = 1
[DEBUG] [2,] [proxy] [downstream] route match result:&{RouteRuleImplBase:0xc00023d900 matchValue:.*}, clusterName=clientCluster
[DEBUG] [upstream] [cluster manager] clusterSnapshot.loadbalancer.ChooseHost result is 127.0.0.1:2046, cluster name = clientCluster
[DEBUG] [stream] [sofarpc] [connpool] init host 127.0.0.1:2046
[INFO] remote addr: 127.0.0.1:2046, network: tcp
// client mosn 建立与上游mosn 的connection =3；
[DEBUG] [network] [check use writeloop] Connection = 3, Local Address = 127.0.0.1:60626, Remote Address = 127.0.0.1:2046
[DEBUG] [network] [client connection connect] connect raw tcp, remote address = 127.0.0.1:2046 ,event = ConnectedFlag, error = <nil>
[DEBUG] client OnEvent ConnectedFlag, connected false
[DEBUG] [2,] [proxy] [downstream] timeout info: {GlobalTimeout:1m0s TryTimeout:0s}
[DEBUG] [2,] [proxy] [downstream] enter phase DownFilterAfterChooseHost[5], proxyId = 1
[DEBUG] [2,] [proxy] [downstream] enter phase DownRecvHeader[6], proxyId = 1
// 向上游发送 header frame； 
[DEBUG] [2,] [proxy] [upstream] append headers: xx  跟上header 的具体内容
[DEBUG] [2,] [proxy] [upstream] connPool ready, proxyId = 1, host = 127.0.0.1:2046
[DEBUG] [2,] [stream] [xprotocol] appendHeaders, direction = 0, requestId = 1
// receive data frame；
[DEBUG] [2,] [proxy] [downstream] enter phase DownRecvData[7], proxyId = 1
[DEBUG] [2,] [proxy] [downstream] receive data = 2.0.2mosn.io.dubbo.DemoService0.0.sayHelloLjava/lang/String;MOSNHpathmosn.io.dubbo.DemoService	interfacemosn.io.dubbo.DemoServiceversion0.0.0Z
[DEBUG] [2,] [proxy] [downstream] start a request timeout timer
// 向上游发送 data frame；
[DEBUG] [2,] [proxy] [upstream] append data:2.0.2mosn.io.dubbo.DemoService0.0.sayHelloLjava/lang/String;MOSNHpathmosn.io.dubbo.DemoService	interfacemosn.io.dubbo.DemoServiceversion0.0.0Z
[DEBUG] [2,] [stream] [xprotocol] appendData, direction = 0, requestId = 1
[DEBUG] [2,] [stream] [xprotocol] connection 3 endStream, direction = 0, requestId = 1
[DEBUG] [2,] [proxy] [downstream] enter phase WaitNotify[11], proxyId = 1
[DEBUG] [2,] [proxy] [downstream] waitNotify begin 0xc00046c000, proxyId = 1
[DEBUG] [2,] [stream] [xprotocol] connection 3 receive response, requestId = 1
// 从上游 接收 header frame 发往下游；
[DEBUG] [2,] [proxy] [upstream] OnReceive headers: xx跟上响应header 的具体内容
[DEBUG] [2,] [proxy] [downstream] OnReceive send downstream response xx
[DEBUG] [2,] [proxy] [downstream] enter phase UpFilter[12], proxyId = 1
[DEBUG] [2,] [proxy] [downstream] enter phase UpRecvHeader[13], proxyId = 1
[DEBUG] [2,] [stream] [xprotocol] appendHeaders, direction = 1, requestId = 0
从上游接收 data frame 发往下游；
[DEBUG] [2,] [proxy] [downstream] enter phase UpRecvData[14], proxyId = 1
[DEBUG] [2,] [stream] [xprotocol] appendData, direction = 1, requestId = 0
[DEBUG] [2,] [stream] [xprotocol] connection 2 endStream, direction = 1, requestId = 0
[DEBUG] update listener write bytes: 89
```

![](/public/upload/mesh/mosn_overview.png)

基于上述认识，[云原生网络代理 MOSN 的进化之路](https://mp.weixin.qq.com/s/5X8ZCO9a9nZE1oAMCNKVzw)我们再来看 mosn 的分层结构设计。其中，**每一层通过工厂设计模式向外暴露其接口**，方便用户灵活地注册自身的需求。

1. NET/IO 作为网络层，监测连接和数据包的到来，同时作为 listener filter 和 network filter 的挂载点;
2. Protocol 作为多协议引擎层，对数据包进行检测，并使用对应协议做 decode/encode 处理。**Protocol 层对应了代码中的 StreamConnection struct，将各个协议映射为 stream 处理机制：**`Dispatch(buf)` 将字节数组 decode 为frame，并 ，非常重要，这也与 代码中的package 包 单纯负责 编解码是 不一样的。
3. Stream **对 decode 的数据包做二次封装为 stream**，作为 stream filter 的挂载点; 
4. Proxy 作为 MOSN 的转发框架，对封装的 stream 做 proxy 处理;

## 转发代码分析（从下游接收请求部分）

mosn 数据接收时，从`proxy.onData` 收到传上来的数据，执行对应协议的`serverStreamConnection.Dispatch` 经过协议解析， **字节流转成了协议的数据包**，转给了`StreamReceiveListener.OnReceive`。proxy.downStream 实现了 StreamReceiveListener。

```go
func (sc *streamConn) Dispatch(buf types.IoBuffer) {
	for {
		// 针对读取到的数据，按照协议类型进行解码
		frame, err := sc.protocol.Decode(streamCtx, buf)
		// No enough data
		//如果没有报错且没有解析成功，那就说明当前收到的数据不够解码，退出循环，等待更多数据到来
		if frame == nil && err == nil {
			return
		}
		//解码成功以后，开始处理该请求
		sc.handleFrame(streamCtx, xframe)
	}
}
func (sc *streamConn) handleFrame(ctx context.Context, frame xprotocol.XFrame) {
	switch frame.GetStreamType() {
	case xprotocol.Request:
		sc.handleRequest(ctx, frame, false)
	case xprotocol.RequestOneWay:
		sc.handleRequest(ctx, frame, true)
	case xprotocol.Response:
		sc.handleResponse(ctx, frame)
	}
}
func (sc *streamConn) handleRequest(ctx context.Context, frame xprotocol.XFrame, oneway bool) {
	// 1. heartbeat process
	if frame.IsHeartbeatFrame() {...}
	// 2. goaway process
	if ...{...}
	// 3. create server stream
	serverStream := sc.newServerStream(ctx, frame)
	// 4. tracer support
	// 5. inject service info
    // 6. receiver callback
    serverStream.receiver = sc.serverCallbacks.NewStreamDetect(serverStream.ctx, sender, span)
	serverStream.receiver.OnReceive(serverStream.ctx, frame.GetHeader(), frame.GetData(), nil)
}
```

在xprotocol 对应的ServerStreamConnection 中，每次收到一个新的xprotocol.xStream，xStream.receiver 即downStream ，downStream代码注释中提到： Downstream stream, as a controller to handle downstream and upstream proxy flow。 downStream  同时持有responseSender 成员指向Stream，用于upstream收到响应数据时 回传给client。

`downStream.OnReceive` 逻辑

```go
func (s *downStream) OnReceive(ctx context.Context,..., data types.IoBuffer, ...) {
    ...
    //把给任务丢给协程池进行处理即可
    pool.ScheduleAuto(func() {
        phase := types.InitPhase
        for i := 0; i < 10; i++ {
            ...
            phase = s.receive(ctx, id, phase)
            ...
        }
    }
}
```

`downStream.receive` 总体来说在请求转发阶段，依次需要经过DownFilter -> MatchRoute -> DownFilterAfterRoute -> DownRecvHeader -> DownRecvData -> DownRecvTrailer -> WaitNofity这么几个阶段。

1. DownFilter, mosn 的配置文件config.json 中的Listener 配置包含 stream filter 配置，就是在此处被使用

    ```json
    "listeners":[
        {
            "name":"",
            "address":"", ## Listener 监听的地址
            "filter_chains":[],  ##  MOSN 仅支持一个 filter_chain
            "stream_filters":[], ## 一组 stream_filter 配置，目前只在 filter_chain 中配置了 filter 包含 proxy 时生效
        }
    ]
    ```
2. MatchRoute，一个请求所属的domains  绑定了许多路由规则，目的将一个请求 路由到一个cluster 上
3. ChooseHost，每一个cluster 对应一个连接池。  从池中 选出一个连接 赋给 downStream.upstreamRequest  
3. WaitNofity则是转发成功以后，等待被响应数据唤醒。

```go
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
            //生成服务提供者的地址列表以及路由规则
            s.matchRoute()
            phase++
        // downstream filter after route
        case types.DownFilterAfterRoute:
            s.runReceiveFilters(phase, s.downstreamReqHeaders, s.downstreamReqDataBuf, s.downstreamReqTrailers)
            phase++
        // downstream receive header
        case types.DownRecvHeader:
            //这里开始依次发送数据
            s.receiveHeaders(s.downstreamReqDataBuf == nil && s.downstreamReqTrailers == nil)
            phase++
        // downstream receive data
        case types.DownRecvData:
            //check not null
            s.receiveData(s.downstreamReqTrailers == nil)
            phase++
        // downstream receive trailer
        case types.DownRecvTrailer:
            s.receiveTrailers()
            phase++
        // downstream oneway
        case types.Oneway:
            ...
        case types.Retry:
            ...
            phase++
        case types.WaitNofity:
            //这里阻塞等待返回及结果
            if p, err := s.waitNotify(id); err != nil {
				return p
            }
            phase++
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
```
真正的发送数据逻辑是在receiveHeaders、receiveData、receiveTrailers这三个方法里，当然每次请求不一定都需要有这三部分的数据，这里我们以receiveHeaders方法为例来进行说明：

```go
func (s *downStream) receiveHeaders(endStream bool) {
	s.downstreamRecvDone = endStream
    ...
	clusterName := s.route.RouteRule().ClusterName()
	s.cluster = s.snapshot.ClusterInfo()
	s.requestInfo.SetRouteEntry(s.route.RouteRule())
	//初始化连接池
	pool, err := s.initializeUpstreamConnectionPool(s)
	parseProxyTimeout(&s.timeout, s.route, s.downstreamReqHeaders)
	prot := s.getUpstreamProtocol()
	s.retryState = newRetryState(s.route.RouteRule().Policy().RetryPolicy(), s.downstreamReqHeaders, s.cluster, prot)
	//构建对应的upstream请求
	proxyBuffers := proxyBuffersByContext(s.context)
	s.upstreamRequest = &proxyBuffers.request
	s.upstreamRequest.downStream = s
	s.upstreamRequest.proxy = s.proxy
	s.upstreamRequest.protocol = prot
	s.upstreamRequest.connPool = pool
	s.route.RouteRule().FinalizeRequestHeaders(s.downstreamReqHeaders, s.requestInfo)
	//这里发送数据
	s.upstreamRequest.appendHeaders(endStream)
	//这里开启超时计时器
	if endStream {
		s.onUpstreamRequestSent()
	}
}
func (r *upstreamRequest) appendHeaders(endStream bool) {
	... 
	r.connPool.NewStream(r.downStream.context, r, r)
}
```
**与一个 downStream  struct 对应的是upstreamRequest** ，倒不算一对一关系。downStream  聚合一个upstreamRequest 成员，从bufferPool（本质是go的对象池sync.Pool）取出一个成员赋给 downStream.upstreamRequest，结束后会调用 downStream.cleanStream 回收。

connPool也是每个协议 不同的，以xprotocol 为例

```go
func (p *connPool) NewStream(ctx context.Context, responseDecoder types.StreamReceiveListener, listener types.PoolEventListener) {
	subProtocol := getSubProtocol(ctx)
    //从连接池中获取连接
	client, _ := p.activeClients.Load(subProtocol)
	activeClient := client.(*activeClient)
    var streamEncoder types.StreamSender
    //这里会把streamId对应的stream保存起来
    streamEncoder = activeClient.client.NewStream(ctx, responseDecoder)
    streamEncoder.GetStream().AddEventListener(activeClient)
    //发送数据
    listener.OnReady(streamEncoder, p.host)
}
```
从 xprotocol.connPool 取出一个client ，创建了一个协议对应的 Stream(变量名为 streamEncoder)，对xprotocol 就是xStream，最终执行了AppendHeaders

```go
func (r *upstreamRequest) OnReady(sender types.StreamSender, host types.Host) {
	r.requestSender = sender
	r.host = host
	r.requestSender.GetStream().AddEventListener(r)
	r.startTime = time.Now()

	endStream := r.sendComplete && !r.dataSent && !r.trailerSent
    //发送数据
	r.requestSender.AppendHeaders(r.downStream.context, r.convertHeader(r.downStream.downstreamReqHeaders), endStream)

	r.downStream.requestInfo.OnUpstreamHostSelected(host)
	r.downStream.requestInfo.SetUpstreamLocalAddress(host.AddressString())
}
func (s *xStream) AppendHeaders(ctx context.Context, headers types.HeaderMap, endStream bool) (err error) {
	// type assertion
	// hijack process
    s.frame = frame
	// endStream
	if endStream {
		s.endStream()
	}
	return
}
func (s *xStream) endStream() {
    // replace requestID
    s.frame.SetRequestId(s.id)
    // remove injected headers
    buf, err := s.sc.protocol.Encode(s.ctx, s.frame)
    err = s.sc.netConn.Write(buf)
}
```

xStream.endStream 真正触发 网络数据的发送，网络层的write

```go
func (c *connection) Write(buffers ...buffer.IoBuffer) (err error) {
    //同样经过过滤器
	fs := c.filterManager.OnWrite(buffers)
	if fs == api.Stop {
		return nil
	}
	if !UseNetpollMode {
		if c.useWriteLoop {
			c.writeBufferChan <- &buffers
		} else {
			err = c.writeDirectly(&buffers)
		}
	} else {
		//netpoll模式写
	}
	return
}
//在对应的eventloop.go中的startWriteLoop方法：
func (c *connection) startWriteLoop() {
	var err error
	for {
		select {
		case <-c.internalStopChan:
			return
		case buf, ok := <-c.writeBufferChan:
			c.appendBuffer(buf)
            c.rawConnection.SetWriteDeadline(time.Now().Add(types.DefaultConnWriteTimeout))
			_, err = c.doWrite()
		}
	}
}
```

请求数据发出去以后当前协程就阻塞了，看下waitNotify方法的实现：

```go
func (s *downStream) waitNotify(id uint32) (phase types.Phase, err error) {
	if s.ID != id {
		return types.End, types.ErrExit
	}
	//阻塞等待
	select {
	case <-s.notify:
	}
	return s.processError(id)
}
```

## 学到的

不要硬看代码，尤其对于多协程程序

1. 打印日志
2. `debug.PrintStack()` 来查看某一个方法之前的调用栈。 再进一步，runtime.Caller 可以查看调用者所在源码文件与行号
    ```go
    // 一次获取一个调用者
    // 0 表示当前函数，1 表示上一层函数，依次往上
    if pc, file, line, ok := runtime.Caller(1); ok{
		fmt.Println(runtime.FuncForPC(pc).Name(), file, line)
    }
    // 一次获取多个调用者
    pc := make([]uintptr, 10)
    n := runtime.Callers(1, pc)
    for i := 0; i < n; i++ {
        f := runtime.FuncForPC(pc[i])
        file, line := f.FileLine(pc[i])
        fmt.Printf("%s %d %s\n", file, line, f.Name())
    }
    ```
3. `fmt.Printf("==> %T\n",xx)`  如果一个interface 有多个“实现类” 可以通过`%T` 查看struct 的类型

