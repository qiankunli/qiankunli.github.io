---

layout: post
title: mosn细节
category: 技术
tags: Mesh
keywords: mesh microservice

---

## 前言（未完成）

* TOC
{:toc}

## Stream 

借鉴了http2 的stream 的理念（所以Stream interface 上有一个方法是`ID()`），Stream 是虚拟的，**在“连接”的层面上看，消息却是乱序收发的“帧”（http2 frame）**，通过StreamId关联，用来实现在一个Connection 之上的“多路复用”。

![](/public/upload/network/network_buffer.png)

Stream 的概念并不新鲜，在微服务通信中，一般要为client request 分配一个requestId，并将requestId 暂存在 client cache中，当client 收到response 时， 从response 数据包中提取requestId，进而确定 response 是client cache中哪个request 的响应。

![](/public/upload/mesh/rpc_network.png)

除了 request-response 一次应答之外，基于Stream 机制还可以实现Streaming rpc的效果，具体可以参见grpc 的示例代码。

### 工厂模式

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

stream 包最外层 定义了ProtocolStreamFactory interface
，针对每个协议 都有一个对应的 streamConnFactory 实现（维护在`var streamFactories map[types.ProtocolName]ProtocolStreamFactory`），协议对应的pkg 内启动时自动执行 init 方法，注册到map。最终 实现根据 Protocol 得到 streamConnFactory 进而得到 ServerStreamConnection 实例

```go
// mosn/pkg/stream/factory.go
func CreateServerStreamConnection(context context.Context, prot api.Protocol, connection api.Connection,
	callbacks types.ServerStreamConnectionEventListener) types.ServerStreamConnection {

	if ssc, ok := streamFactories[prot]; ok {
		return ssc.CreateServerStream(context, connection, callbacks)
	}

	return nil
}
```

### StreamConnection

StreamConnection is a connection runs multiple streams

![](/public/upload/mesh/mosn_StreamConnection.png)

mosn 数据接收时，从`proxy.onData` 收到传上来的数据，执行对应协议的`serverStreamConnection.Dispatch` ==> 根据协议解析数据 ，经过协议解析，收到一个完整的请求时`serverStreamConnection.handleFrame` 会创建一个 Stream，然后逻辑 转给了`StreamReceiveListener.OnReceive`。proxy.downStream 实现了 StreamReceiveListener

![](/public/upload/mesh/mosn_Stream.png)

## ConnectionPool

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




