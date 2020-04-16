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

## 整体逻辑

[SOFAMosn Introduction](https://github.com/sofastack/sofastack-doc/blob/master/sofa-mosn/zh_CN/docs/Introduction.md) 

![](/public/upload/go/mosn_io_process.png)

1. MOSN 在 IO 层读取数据，通过 read filter 将数据发送到 Protocol 层进行 Decode
2. Decode 出来的数据，根据不同的协议，**回调到 stream 层**，进行 stream 的创建和封装
3. stream 创建完毕后，会回调到 Proxy 层做路由和转发，Proxy 层会关联上下游（upstream,downstream）间的转发关系
4. Proxy 挑选到后端后，会根据后端使用的协议，将数据发送到对应协议的 Protocol 层，对数据重新做 Encode
5. Encode 后的数据会发经过 write filter 并最终使用 IO 的 write 发送出去

![](/public/upload/mesh/mosn_process.png)

## 数据接收

network 层 读取

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
**之后streamConnection/stream 等逻辑就是 根据协议的不同而不同了**（初次看代码时踩了很大的坑），下面代码以xprotocol 协议为例

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



mosn 数据接收时，从`proxy.onData` 收到传上来的数据，执行对应协议的`serverStreamConnection.Dispatch` ==> 根据协议解析数据 ，经过协议解析，收到一个完整的请求时`serverStreamConnection.handleFrame` 会创建一个 Stream，然后逻辑 转给了`StreamReceiveListener.OnReceive`。proxy.downStream 实现了 StreamReceiveListener。


## 转发流程

Downstream stream, as a controller to handle downstream and upstream proxy flow `downStream.OnReceive` 逻辑

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

`downStream.receive` 总体来说在请求转发阶段，依次需要经过DownFilter -> MatchRoute -> DownFilterAfterRoute -> DownRecvHeader -> DownRecvData -> DownRecvTrailer -> WaitNofity这么几个阶段，从字面意思可以知道MatchRoute就是构建路由信息，也就是转发给哪个服务，而WaitNofity则是转发成功以后，等待被响应数据唤醒。

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

```
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

网络层的write

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
## 与envoy 对比

envoy 对应逻辑 [深入解读Service Mesh的数据面Envoy](https://sq.163yun.com/blog/article/213361303062011904)

![](/public/upload/mesh/envoy_new_connection.jpg)

[深入解读Service Mesh的数据面Envoy](https://sq.163yun.com/blog/article/213361303062011904)下文以envoy 实现做一下类比 用来辅助理解mosn 相关代码的理念：

![](/public/upload/mesh/envoy_on_data.jpg)

对于每一个Filter，都调用onData函数，咱们上面解析过，其中HTTP对应的ReadFilter是ConnectionManagerImpl，因而调用ConnectionManagerImpl::onData函数。ConnectionManager 是协议插件的处理入口，**同时也负责对整个处理过程的流程编排**。

![](/public/upload/mesh/envoy_data_parse.jpg)

## 补充细节

不同颜色 表示所处的 package 不同

![](/public/upload/go/mosn_object.png)

一次http1协议请求的处理过程（绿色部分表示另起一个协程）

![](/public/upload/go/mosn_http_read.png)

代码的组织（pkg/stream,pkg/protocol,pkg/proxy）  跟架构是一致的

![](/public/upload/go/mosn_layer.png)

1. `pkg/types/connection.go` Connection
2. `pkg/types/stream.go` StreamConnection is a connection runs multiple streams
3. `pkg/types/stream.go` Stream is a generic protocol stream
4. 一堆listener 和filter 比较好理解：Method in listener will be called on event occur, but not effect the control flow.Filters are called on event occurs, it also returns a status to effect control flow. Currently 2 states are used: Continue to let it go, Stop to stop the control flow.

## 学到的

不要硬看代码，尤其对于多协程程序

1. 打印日志
2. `debug.printStack` 来查看某一个方法之前的调用栈
3. `fmt.Printf("==> %T\n",xx)`  如果一个interface 有多个“实现类” 可以通过`%T` 查看struct 的类型

