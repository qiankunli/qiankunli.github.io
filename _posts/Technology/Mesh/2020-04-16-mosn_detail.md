---

layout: post
title: mosn细节
category: 技术
tags: Mesh
keywords: mosn detail

---

## 前言

* TOC
{:toc}

在某种程度上，除了代理组件的业务逻辑=转发外，代理组件和web/rpc服务非常像，代理组件接收请求转发到另一个服务跟web/rpc接收请求调用依赖服务是一样一样的。与rpc 框架（比如netty）做类比的话，就会发现在多协议支持/内存池管理/连接池管理/扩展机制等方面非常相似。

## 多协议机制的实现——多路复用 

[MOSN 多协议机制解析](https://mosn.io/zh/blog/posts/multi-protocol-deep-dive/)

![](/public/upload/mesh/mosn_protocol.png)

借鉴了http2 的stream 的理念（所以Stream interface 上有一个方法是`ID()`），Stream 是虚拟的，**在“连接”的层面上看，消息却是乱序收发的“帧”（http2 frame）**，通过StreamId关联，用来实现在一个Connection 之上的“多路复用”。tcp 数据包在网络上流转，os 维护了socket 对象，随着连接创建、关闭而新建和销毁。 frame 数据包在 连接中传输，网络 应用层维护了 stream 对象，随着 request-response 产生、结束而新建和销毁。 

![](/public/upload/network/network_buffer.png)

Stream 的概念并不新鲜，在微服务通信中，一般要为client request 分配一个requestId，并将requestId 暂存在 client cache中，当client 收到response 时， 从response 数据包中提取requestId，进而确定 response 是client cache中哪个request 的响应。**之前这只是一种“技巧” 或“惯例”，Http2将其 正式 称之为多路复用/Stream**。

![](/public/upload/mesh/rpc_network.png)

mosn http2 和 xprotocol 的StreamConnection 中都保存有 requestId 与 Stream 映射。当发现 frame 携带的 requestId 不存在时，则NewStream，否则读取Stream。然后拿着 Stream 对象 执行`stream.receiver.OnReceive`

以http2 和 xprotocol 对比来说，在收到 network 出来的字节数据时，执行Dispatch 方法

```go
    frame, err := sc.protocol.Decode(streamCtx, buf)
    // 如果没有足够数据，则直接返回，若是凑够了一个frame ，则handleFrame
    handleFrame(streamCtx, xframe)
```

对于http2 来说，一个 请求/响应 对应多个frame（至少包含header 和 data 2个frame），一个stream 对应一个请求加 多个响应  即 多个frame。除了数据frame，http2 还支持很多的control frame。

对于xprotocol 来说，对于普通rpc 协议（支持steaming rpc的协议除外）

1. 一个 请求/响应 对应一个frame（或者说一个frame 只区分 request/response），一个stream 对应一个请求 加一个响应 2个frame。
2. 请求和 响应 一般共用一个统一的数据格式，因此可以用一个 frame struct 表示
3. 一般会支持 心跳机制，即心跳frame

## 多路复用的代码实现 StreamConnection

StreamConnection is a connection runs multiple streams

一个dubbo 协议的config.json 例子

```json
{
    "servers":[
        {
            "listeners":[
                {
                    "filter_chains":[
                        {
                            "filters":[
                                {
                                    "downstream_protocol": "X",
                                    "upstream_protocol": "X",
                                    "extend_config": {
                                        "sub_protocol": "dubbo"
                                    }
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}
```

初始化相关链路：activeListener.OnAccept ==> activeRawConn.ContinueFilterChain ==> activeListener.newConnection ==> activeListener.OnNewConnection ==> genericProxyFilterConfigFactory.CreateFilterChain ==> filterManager.AddReadFilter ==> proxy.InitializeReadFilterCallbacks ==> stream.CreateServerStreamConnection

```go
// mosn.io/mosn/pkg/proxy/proxy.go
func (p *proxy) InitializeReadFilterCallbacks(cb api.ReadFilterCallbacks) {
    p.readCallbacks = cb
    ...
	p.readCallbacks.Connection().AddConnectionEventListener(p.downstreamListener)
	if p.config.DownstreamProtocol != string(protocol.Auto) {
        // 创建好的ServerStreamConnection 赋给proxy  
        // proxy.config.DownstreamProtocol = X 即 Xprotocol
		p.serverStreamConn = stream.CreateServerStreamConnection(p.context, types.ProtocolName(p.config.DownstreamProtocol), p.readCallbacks.Connection(), p)
	}
}
// mosn.io/mosn/pkg/stream/factory.go
var streamFactories map[types.ProtocolName]ProtocolStreamFactory
func init() {
	streamFactories = make(map[types.ProtocolName]ProtocolStreamFactory)
}
func Register(prot types.ProtocolName, factory ProtocolStreamFactory) {
	streamFactories[prot] = factory
}
func CreateServerStreamConnection(context context.Context, prot api.Protocol, connection api.Connection,
	callbacks types.ServerStreamConnectionEventListener) types.ServerStreamConnection {
	if ssc, ok := streamFactories[prot]; ok {
		return ssc.CreateServerStream(context, connection, callbacks)
	}
	return nil
}
```
在新的Connection 被创建时，会创建一个ServerStreamConnection 赋给proxy。所有L7 层协议要实现 stream.ProtocolStreamFactory interface 用于构建 StreamConnection。示例中 `proxy.config.DownstreamProtocol` = X 即 Xprotocol ，即L7 层 数据读取 及转发采用 Xprotocol 多用复用机制（而不是http或http2）

```go
// mosn.io/mosn/pkg/stream/xprotocol/factory.go
func (f *streamConnFactory) CreateServerStream(context context.Context, connection api.Connection,
	serverCallbacks types.ServerStreamConnectionEventListener) types.ServerStreamConnection {
	return newStreamConnection(context, connection, nil, serverCallbacks)
}
// mosn.io/mosn/pkg/stream/xprotocol/conn.go
func newStreamConnection(ctx context.Context, conn api.Connection, clientCallbacks types.StreamConnectionEventListener,
	serverCallbacks types.ServerStreamConnectionEventListener) types.ClientStreamConnection {
	// 1. init first context
	// 2. prepare protocols
	subProtocol := mosnctx.Get(ctx, types.ContextSubProtocol).(string)
	subProtocols := strings.Split(subProtocol, ",")
	// 2.1 exact protocol, get directly
	// 2.2 multi protocol, setup engine for further match 
	if len(subProtocols) == 1 {
		proto := xprotocol.GetProtocol(types.ProtocolName(subProtocol))
		if proto == nil {
			log.Proxy.Errorf(ctx, "[stream] [xprotocol] no such protocol: %s", subProtocol)
			return nil
		}
		sc.protocol = proto
	} else {   // mosn 支持多个subProtocol ，估计是可以根据 连接中的信息自动匹配
		engine, err := xprotocol.NewXEngine(subProtocols)
		if err != nil {
			log.Proxy.Errorf(ctx, "[stream] [xprotocol] create XEngine failed: %s", err)
			return nil
		}
		sc.engine = engine
	}
	// client
	if sc.clientCallbacks != nil {
		// default client concurrency capacity: 8
		sc.clientStreams = make(map[uint64]*xStream, 8) // Stream 抽象了一般rpc 框架维护的 <requestId,Request>
	}
	// set support transfer connection
	return sc
}
// mosn.io/mosn/pkg/protocol/xprotocol/factory.go
var (
	protocolMap = make(map[types.ProtocolName]XProtocol)
	matcherMap  = make(map[types.ProtocolName]types.ProtocolMatch)
)
// RegisterProtocol register the protocol to factory
func RegisterProtocol(name types.ProtocolName, protocol XProtocol) error {
	// check name conflict
	_, ok := protocolMap[name]
	if ok {
		return errors.New("duplicate protocol register:" + string(name))
	}
	protocolMap[name] = protocol
	return nil
}
func GetProtocol(name types.ProtocolName) XProtocol {
	return protocolMap[name]
}
```

L7 及 Xprotocol subProtocol 对应的包在启动 时会将 自己的实现 register 到相关数据结构

```
mosn/pkg/stream
    http
        stream.go
            func init() {
                str.Register(protocol.HTTP1, &streamConnFactory{})
            }
            type streamConnFactory struct{}
    http2
        stream.go
            func init() {
                str.Register(protocol.HTTP2, &streamConnFactory{})
            }
            type streamConnFactory struct{}
    xprotocol
        factory.go
            func init() {
                stream.Register(protocol.Xprotocol, &streamConnFactory{})
            }
            type streamConnFactory struct{}
    factory.go
        var streamFactories map[types.ProtocolName]ProtocolStreamFactory
        func init() {
            streamFactories = make(map[types.ProtocolName]ProtocolStreamFactory)
        }
        func Register(prot types.ProtocolName, factory ProtocolStreamFactory) {
            streamFactories[prot] = factory
        }
        type ProtocolStreamFactory interface {
            CreateClientStream(...) types.ClientStreamConnection
            CreateServerStream(...) types.ServerStreamConnection
            CreateBiDirectStream(...) types.ClientStreamConnection
            ProtocolMatch(context context.Context, prot string, magic []byte) error
        }
    stream.go
    types.go
```

Xprotocol 支持多个 subProtocol，按照示例 配置 创建 dubbo 对应的 Xprotocol ServerStreamConnection。所有的 subProtocol 要实现 xprotocol.Xprotocol interface。 

```
mosn/pkg/protocol/xprotocol
    bolt
        protocol.go
            func init() {
	            xprotocol.RegisterProtocol(ProtocolName, &boltProtocol{})
            }
    dubbo
        protocol.go
            func init() {
                xprotocol.RegisterProtocol(ProtocolName, &dubboProtocol{})
            }
    factory.go
        var ( // 工厂模式，各个协议的包在启动时，将自己注册到protocolMap 和 matcherMap 中。
            protocolMap = make(map[types.ProtocolName]XProtocol)
            matcherMap  = make(map[types.ProtocolName]types.ProtocolMatch)
        )
        func RegisterProtocol(name types.ProtocolName, protocol XProtocol) error {
            ...
            protocolMap[name] = protocol
            ...
        }
```

streamConn 承上启下，分为client 和 server两侧，持有协议、连接对象，可以创建clientSteam（且暂存） 和 serverStream
1. server 侧，从Dispatch 接收数据，解码，创建 serverStream 关联downStream ，downStream 处理，创建针对上游的clientStream， 操作Connection 写数据到上游
1. client 侧，从Dispatch 接收数据，解码，寻找 clientStream ，downStream 恢复处理，找到关联的serverStream， 操作Connection 写数据到下游

```go
// mosn.io/mosn/pkg/stream/xprotocol/conn.go
type streamConn struct {
    ctx        context.Context
    ctxManager *stream.ContextManager

	netConn    api.Connection               // 底层连接
	engine   *xprotocol.XEngine             // xprotocol fields
	protocol xprotocol.XProtocol
    // server side fields
	serverCallbacks types.ServerStreamConnectionEventListener 
    // client side fields
	clientMutex        sync.RWMutex                           
	clientStreamIDBase uint64
	clientStreams      map[uint64]*xStream
	clientCallbacks    types.StreamConnectionEventListener
}
```

## 路由

1. mosn 启动时会初始化 cluster_manager 管理所有的cluster_manager 并通过xds 同步cluster 数据。
2. proxy.routersWrapper 持有了 所有的routers 数据
3. 转发流程中 downStream.matchRoute 逻辑 负责从 virtual_hosts 中获取匹配的 router(对http 是域名:path；对rpc 是serviceName:methodName等)，进而拿到对应的 cluster_name ==> cluster。之后是下文针对具体host建立连接池的事儿。

```json
// servers.listeners
"filter_chains": [ {
    "type": "connection_manager",
    "config": {
        "router_config_name": "client_router",
        "virtual_hosts": [{
            "name": "clientHost",
            "domains": ["*"],
            "routers": [{
                "match": {...},
                "route": {
                    "cluster_name": "clientCluster"
                }
            }]
        }]
    }
}]
...
"cluster_manager": {
    "clusters": [{
        "Name": "clientCluster",
        "type": "SIMPLE",
        "lb_type": "LB_RANDOM",
        "hosts": [{
            "address": "127.0.0.1:2046"
        }]
    }]
},
```

## 连接池管理

[云原生网络代理 MOSN 的进化之路](https://mp.weixin.qq.com/s/5X8ZCO9a9nZE1oAMCNKVzw)为了提升服务网格之间的建连性能还设计了多种协议的连接池从而方便地实现连接复用及管理。在连接管理方面，MOSN 设计了多协议连接池， 当 Proxy 模块在 Downstream 收到 Request 的时候，在经过路由、负载均衡等模块处理获取到 Upstream Host 以及对应的转发协议时，通过 Cluster Manager 获取对应协议的连接池 ，如果连接池不存在则创建并加入缓存中，之后在长连接上创建 Stream，并发送数据

![](/public/upload/mesh/mosn_conn_pool.png)

同样应用了工厂模式

```
mosn/pkg/types
    upstream.go
        func init() {
	        ConnPoolFactories = make(map[api.Protocol]bool)
        }
        var ConnPoolFactories map[api.Protocol]bool
        func RegisterConnPoolFactory(protocol api.Protocol, registered bool) {
            ConnPoolFactories[protocol] = registered
        }
mosn/pkg/network
    connpool.go
        func init() {
            ConnNewPoolFactories = make(map[types.ProtocolName]connNewPool)
        }
        var ConnNewPoolFactories map[types.ProtocolName]connNewPool
        func RegisterNewPoolFactory(protocol types.ProtocolName, factory connNewPool) {
            ConnNewPoolFactories[protocol] = factory
        }
mosn/pkg/stream
    http
        connpool.go
            func init() {
                network.RegisterNewPoolFactory(protocol.HTTP1, NewConnPool)
	            types.RegisterConnPoolFactory(protocol.HTTP1, true)
            }
    http2
        connpool.go
            func init() {
                network.RegisterNewPoolFactory(protocol.HTTP2, NewConnPool)
	            types.RegisterConnPoolFactory(protocol.HTTP2, true)
            }
    xprotocol
        factory.go
            func init() {
                network.RegisterNewPoolFactory(protocol.Xprotocol, NewConnPool)
	            types.RegisterConnPoolFactory(protocol.Xprotocol, true)
            }
    factory.go
    stream.go
    types.go
        type Client interface
    client.go
        type client struct
```

![](/public/upload/mesh/mosn_ConnectionPool.png)

## 内存管理

在池管理的术语上，mosn的习惯是 take 是申请元素，give 是回收元素。此外，池管理对象/变量 不直接对外使用，仅通过`包名.方法` 进行操作。

### 对象池管理

[云原生网络代理 MOSN 的进化之路](https://mp.weixin.qq.com/s/5X8ZCO9a9nZE1oAMCNKVzw)MOSN 为了降低 Runtime GC 带来的卡顿，MOSN 在 `sync.Pool` 之上封装了一层资源对的注册管理模块，可以方便的扩展各种类型的对象进行复用和管理。

java 中使用commons-pool 初始化一个 ObjectPool `ObjectPool<Demo> objectPool = new GenericObjectPool(PooledObjectFactory,GenericObjectPoolConfig)`，其中PooledObjectFactory 告知如何创建 Demo 对象，config 配置了池大小等，之后就可以 `objectPool.borrowObject/returnObject` 来管理和回收对象了。

在mosn 对象池管理机制中，与ObjectPool 比较对应的 是bufferPool，BufferPoolCtx 对应PooledObjectFactory， `bufferPool.take/give` 分别对应`objectPool.borrowObject/returnObject`。
```go
// mosn.io/mosn/pkg/types/buffer.go
type BufferPoolCtx interface {
	// Index returns the bufferpool's Index
	Index() int
	New() interface{}           // new 一个元素
	Reset(interface{})          // reset 一个元素  放回到buffer 中的元素要 reset 到一种clean 的状态
}
// mosn.io/mosn/pkg/buffer/buffer.go
// bufferPool is buffer pool
type bufferPool struct {
	ctx types.BufferPoolCtx
	sync.Pool
}
// 取出一个元素，有则取出，无则new 一个
func (p *bufferPool) take() (value interface{}) {
	value = p.Get()
	if value == nil {
		value = p.ctx.New()
	}
	return
}
// 放回元素 到buffer 中
func (p *bufferPool) give(value interface{}) {
	p.ctx.Reset(value)
	p.Put(value)
}
```

到这按说就够用了，mosn 又进一步提出一个 bufferValue struct，`bufferValue.Take(BufferPoolCtx)` 根据 BufferPoolCtx 参数便可以返回池化对象。此时bufferValue 很像一个通用对象池，根据传入的BufferPoolCtx 缓存任何对象（最大16个对象类型），底层实际上是一个 bufferPool 的数组bPool在支持。

![](/public/upload/mesh/mosn_buffer_pool.png)

```go
type bufferValue struct {
	value    [maxBufferPool]interface{}
	transmit [maxBufferPool]interface{}
}
func (bv *bufferValue) Take(poolCtx types.BufferPoolCtx) (value interface{}) {
	i := poolCtx.Index()
	value = bPool[i].take()
	bv.value[i] = value
	return
}
// 返回poolCtx 对应的上一次take 的对象，如果没有，则新生成一个
func (bv *bufferValue) Find(poolCtx types.BufferPoolCtx, ...) interface{} {
	i := poolCtx.Index()
	if i <= 0 || i > int(index) {
		panic("buffer should call buffer.RegisterBuffer()")
	}
	if bv.value[i] != nil {
		return bv.value[i]
	}
	return bv.Take(poolCtx)
}
```

bufferValue 本身也受 对象次vPool/valuePool 管理，bufferValue 的申请（newBufferValue） 和 回收(bufferValue.Give） 都经过了 vPool

```go
const maxBufferPool = 16
var (
	index int32
	bPool = bufferPoolArray[:]
    bufferPoolArray [maxBufferPool]bufferPool
    
    vPool = new(valuePool)
)
type valuePool struct {
	sync.Pool
}
```

### buffer 管理

通信中的编解码涉及到 大量小字节数组的 分配和释放，类似netty 中的ByteBuf，在mosn 中是IoBuffer。

```go
// mosn.io/pkg/buffer/types.go
type IoBuffer interface {
	Read(p []byte) (n int, err error)
	ReadOnce(r io.Reader) (n int64, err error)
	Write(p []byte) (n int, err error)
	WriteString(s string) (n int, err error)
	WriteByte(p byte) error
	WriteUint32(p uint32) error
	WriteUint64(p uint64) error
	WriteTo(w io.Writer) (n int64, err error)
	Len() int
	Cap() int
	Reset()
	// String returns the contents of the unread portion of the buffer as a string. If the Buffer is a nil pointer, it returns "<nil>".
	String() string
	// Alloc alloc bytes from BytePoolBuffer
	Alloc(int)
	// Free free bytes to BytePoolBuffer
	Free()
	Append(data []byte) error
	CloseWithError(err error)
}
// mosn.io/pkg/buffer/iobuffer.go
type ioBuffer struct {
	buf     []byte // contents: buf[off : len(buf)]
	off     int    // read from &buf[off], write to &buf[len(buf)]
	offMark int
	count   int32
	eof     bool
	b *[]byte
}
func newIoBuffer(capacity int) IoBuffer {
	buffer := &ioBuffer{
		offMark: ResetOffMark,
		count:   1,
	}
	if capacity <= 0 {
		capacity = DefaultSize
	}
	buffer.b = GetBytes(capacity)
	buffer.buf = (*buffer.b)[:0]
	return buffer
}
```
在netty 中ByteBuf 是“门脸”，封装了 基本数据类型（Int/Long/String等）的 读写操作，真正管 内存/字节数组分配的是Arena，mosn 中 IoBuffer 也是如此， 只是IoBuffer 本身有一个池管理变量 ibPool，对外提供 `buffer.GetIoBuffer/PutIoBuffer` 进行IoBuffer的分配和回收。

```go
// mosn.io/pkg/buffer/iobuffer_pool.go
var ibPool IoBufferPool
// IoBufferPool is Iobuffer Pool
type IoBufferPool struct {
	pool sync.Pool
}
var ibPool IoBufferPool
func GetIoBuffer(size int) IoBuffer {
	return ibPool.take(size)
}
func PutIoBuffer(buf IoBuffer) error {
	count := buf.Count(-1)
	if count > 0 {
		return nil
	} else if count < 0 {
		return errors.New("PutIoBuffer duplicate")
	}
	if p, _ := buf.(*pipe); p != nil {
		buf = p.IoBuffer
	}
	ibPool.give(buf)
	return nil
}
```
真正管内存/字节数组分配的是 byteBufferPool，byteBufferPool 包含一个 bufferSlot 数组，每个 bufferSlot 是一个字节数组的对象池（字节数组大小为 defaultSize）

![](/public/upload/mesh/mosn_byte_buffer_pool.png)

```go
var bbPool *byteBufferPool
func init() {
	bbPool = newByteBufferPool()
}
func GetBytes(size int) *[]byte {
	return bbPool.take(size)
}
// mosn.io/pkg/buffer/bytebuffer_pool.go
// byteBufferPool is []byte pools
type byteBufferPool struct {
	minShift int
	minSize  int
	maxSize  int
	pool []*bufferSlot
}
type bufferSlot struct {
	defaultSize int
	pool        sync.Pool
}
func (p *byteBufferPool) take(size int) *[]byte {
	slot := p.slot(size)
	if slot == errSlot {
		b := newBytes(size)
		return &b
	}
	v := p.pool[slot].pool.Get()
	if v == nil {
		b := newBytes(p.pool[slot].defaultSize)
		b = b[0:size]
		return &b
	}
	b := v.(*[]byte)
	*b = (*b)[0:size]
	return b
}
func (p *byteBufferPool) give(buf *[]byte) {
	if buf == nil {
		return
	}
	size := cap(*buf)
	slot := p.slot(size)
	if slot == errSlot {
		return
	}
	if size != int(p.pool[slot].defaultSize) {
		return
	}
	p.pool[slot].pool.Put(buf)
}
```

在Protocol 层的编解码中，便可以直接使用 IoBuffer 作为Request/Response.Header/Data 使用。

```go
// mosn.io/mosn/pkg/protocol/buffer.go
type ProtocolBuffers struct {
	reqData     buffer.IoBuffer
	reqHeader   buffer.IoBuffer
	reqHeaders  map[string]string
	reqTrailers map[string]string

	rspData     buffer.IoBuffer
	rspHeader   buffer.IoBuffer
	rspHeaders  map[string]string
	rspTrailers map[string]string
}
// GetReqData returns IoBuffer for request data
func (p *ProtocolBuffers) GetReqData(size int) buffer.IoBuffer {
	if size <= 0 {
		size = defaultDataSize
	}
	p.reqData = buffer.GetIoBuffer(size)
	return p.reqData
}
```

## filter扩展机制

[MOSN 源码解析 - filter扩展机制](https://mosn.io/zh/blog/code/mosn-filters/)MOSN 使用了过滤器模式来实现扩展。MOSN 把过滤器相关的代码放在了 pkg/filter 目录下，包括 accept 过程的 filter，network 处理过程的 filter，以及 stream 处理的 filter。其中 accept filters 目前暂不提供扩展（加载、运行写死在代码里面，如要扩展需要修改源码）， steram、network filters 是可以通过定义新包在 pkg/filter 目录下实现扩展。

mosn 的配置文件config.json 中的Listener 配置包含 stream filter 配置

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
代码中的示例
```
mosn/pkg/filter/stream
    faultinject
        factory.go
            func init() {
                api.RegisterStream(v2.FaultStream, CreateFaultInjectFilterFactory)
            }
            type FilterConfigFactory struct {
                Config *v2.StreamFaultInject
            }
    mixer
        func init() {
            api.RegisterStream(v2.MIXER, CreateMixerFilterFactory)
        }
        type FilterConfigFactory struct {
	        MixerConfig *v2.Mixer
        }
mosn.io/api
    filter_factory.go
        func init() {
            creatorListenerFactory = make(map[string]ListenerFilterFactoryCreator)
            creatorStreamFactory = make(map[string]StreamFilterFactoryCreator)
            creatorNetworkFactory = make(map[string]NetworkFilterFactoryCreator)
        }
        func RegisterStream(filterType string, creator StreamFilterFactoryCreator) {
            creatorStreamFactory[filterType] = creator
        }
```

## 与control plan 的交互

`pkg/xds/v2/adssubscriber.go` 启动发送线程和接收线程

```go
func (adsClient *ADSClient) Start() {
    adsClient.StreamClient = adsClient.AdsConfig.GetStreamClient()
    utils.GoWithRecover(func() {
        adsClient.sendThread()
    }, nil)
    utils.GoWithRecover(func() {
        adsClient.receiveThread()
    }, nil)
}
```go

定时发送请求
```go
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
```

接收响应

```go
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
```

处理逻辑是事先注册好的函数

```go
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
```

以cluster 信息为例 HandleEnvoyCluster

```go
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
```

会触发ClusterManager 更新cluster 

```go
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
```

## 服务发现注册

从一个公司的实际来说，不可能一下子所有的服务都在容器环境内运行。容器环境内的rpc 服务启动时需要将自己的服务信息注册到 registry 上，进而可以被容器环境外的服务访问到。有几种方式

1. 从k8s向registry 同步数据
2. 业务容器的sdk 直接向registry 写入数据
3. 业务容器的sdk 通过 sidecar 向registry 写入数据

```
mosn/pkg/upstream/servicediscovery/dubbod
    init.go
        func init() {
	        Init()
        }
    bootstrap.go
        func Init( /*port string, dubboLogPath string*/ ) {
	        r := chi.NewRouter()
            r.Post("/sub", subscribe)
	        r.Post("/unsub", unsubscribe)
	        r.Post("/pub", publish)
	        r.Post("/unpub", unpublish)
            ...
        }
    pub.go
        func publish(w http.ResponseWriter, r *http.Request) {
	        err = doPubUnPub(req, true)
	        return
        }
```

sidecar 启动时，通过func init 启动一个webserver组件，和业务容器约定一个pub请求，sidecar web server 收到请求之后，将信息写到 registry（比如zk） 上。 

![](/public/upload/mesh/mosn_pub.png)