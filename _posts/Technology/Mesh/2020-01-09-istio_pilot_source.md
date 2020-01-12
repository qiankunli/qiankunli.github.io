---

layout: post
title: Pilot源码分析
category: 技术
tags: Mesh
keywords: Go

---

## 前言

* TOC
{:toc}

![](/public/upload/mesh/pilot_package.png)

## pilot-discovery

[Istio Pilot代码深度解析](https://www.servicemesher.com/blog/201910-pilot-code-deep-dive/)

**如果把Pilot看成一个处理数据的黑盒，则其有两个输入，一个输出**。

1. 目前Pilot的输入包括两部分数据来源：

    1. 服务数据（随着服务的启停、灰度等自动的）： 来源于各个服务注册表(Service Registry)，例如Kubernetes中注册的Service，Consul Catalog中的服务等。
    2. 配置规则（人为的）： 各种配置规则，包括路由规则及流量管理规则等，通过Kubernetes CRD(Custom Resources Definition)形式定义并存储在Kubernetes中。
2. Pilot的输出为符合xDS接口的数据面配置数据，并通过gRPC Streaming接口将配置数据推送到数据面的Envoy中。

![](/public/upload/mesh/pilot_input_output.svg)

## 处理输入

底层平台 多种多样，istio 抽象一套自己的数据模型，以进行平台间的交互。

### 服务数据部分

[Istio 服务注册插件机制代码解析](https://zhaohuabing.com/post/2019-02-18-pilot-service-registry-code-analysis/)

![](/public/upload/mesh/pilot_discovery.png)

中间Abstract Model 层 实现如下

![](/public/upload/mesh/pilot_service_object.png)

Service describes an Istio service (e.g., catalog.mystore.com:8080)Each service has a fully qualified domain name (FQDN) and one or more ports where the service is listening for connections. Service用于表示Istio服务网格中的一个服务（例如 catalog.mystore.com:8080)。每一个服务有一个全限定域名(FQDN)和一个或者多个接收客户端请求的监听端口。

SercieInstance中存放了服务实例相关的信息，一个Service可以对应到一到多个Service Instance，Istio在收到客户端请求时，会根据该Service配置的LB策略和路由规则从可用的Service Instance中选择一个来提供服务。

ServiceDiscovery抽象了一个服务发现的接口，所有接入istio 的平台应提供该接口实现。

Controller抽象了一个Service Registry变化通知的接口，该接口会将Service及Service Instance的增加，删除，变化等消息通知给ServiceHandler(也就是一个func)。**调用Controller的Run方法后，Controller会一直执行，将监控Service Registry的变化，并将通知到注册到Controller中的ServiceHandler中**。

由上图可知，底层平台 接入时必须实现 ServiceDiscovery 和 Controller，提供Service 数据，并在Service 变动时 执行handler。 整个流程 由Controller.Run 触发，将平台数据 同步and 转换到 istio 内部数据模型（ServiceDiscovery实现），若数据有变化，则触发handler。 

### 配置数据部分

![](/public/upload/mesh/pilot_config_object.png)

ConfigStore describes a set of platform agnostic APIs that must be supported by the underlying platform to store and retrieve Istio configuration. ConfigStore定义一组平台无关的，但是底层平台（例如K8S）必须支持的API，通过这些API可以存取Istio配置信息每个配置信息的键，由type + name + namespace的组合构成，确保每个配置具有唯一的键。写操作是异步执行的，也就是说Update后立即Get可能无法获得最新结果。

ConfigStoreCache表示ConfigStore的本地完整复制的缓存，此缓存主动和远程存储保持同步，并且在获取更新时提供提供通知机制。为了获得通知，事件处理器必须在Run之前注册，缓存需要在Run之后有一个初始的同步延迟。

IstioConfigStore扩展ConfigStore，增加一些针对Istio资源的操控接口

由上图可知，底层平台 接入时必须实现 ConfigStoreCache，提供Config 数据，并在Config 变动时 执行handler。 整个流程 由ConfigStoreCache.Run 触发，将平台数据 同步and 转换到 istio 内部数据模型（ConfigStore实现），若数据有变化，则触发handler。 

### Environment 聚合

![](/public/upload/mesh/pilot_environment_object.png)

Environment provides an aggregate environmental API for Pilot. Environment为Pilot提供聚合的环境性的API

由上文可知，启动时，向  Controller 和ConfigStoreCache 注册handler，执行 ConfigStoreCache.Run 和  Controller.Run，便可以同步 service 和config 数据，并在数据变动时 触发handler 执行。**pilot数据输入的部分就解决了**

![](/public/upload/mesh/pilot_component.svg)

### proxy

Proxy contains information about an specific instance of a proxy (envoy sidecar, gateway,etc). The Proxy is initialized when a sidecar connects to Pilot, and populated from 'node' info in the protocol as well as data extracted from registries.

    type Proxy struct {
        ClusterID string
        // Type specifies the node type. First part of the ID.
        Type NodeType
        IPAddresses []string
        ID string
        Locality *core.Locality
        // DNSDomain defines the DNS domain suffix for short hostnames (e.g.
        // "default.svc.cluster.local")
        DNSDomain string
        ConfigNamespace string
        // Metadata key-value pairs extending the Node identifier
        Metadata *NodeMetadata
        // the sidecarScope associated with the proxy
        SidecarScope *SidecarScope
        // The merged gateways associated with the proxy if this is a Router
        MergedGateway *MergedGateway
        // service instances associated with the proxy
        ServiceInstances []*ServiceInstance
        // labels associated with the workload
        WorkloadLabels labels.Collection
        // Istio version associated with the Proxy
        IstioVersion *IstioVersion
    }

## 启动

启动命令示例：`/usr/local/bin/pilot-discovery discovery --monitoringAddr=:15014 --log_output_level=default:info --domain cluster.local --secureGrpcAddr  --keepaliveMaxServerConnectionAge 30m`

    package bootstrap
    func NewServer(args *PilotArgs) (*Server, error) {
        s.initKubeClient(args)
        s.initMeshConfiguration(args, fileWatcher)
        s.initMeshNetworks(args, fileWatcher)
        s.initCertController(args)
        s.initConfigController(args)
        s.initServiceControllers(args)
        s.initDiscoveryService(args)
        s.initMonitor(args.DiscoveryOptions.MonitoringAddr)
        s.initClusterRegistries(args)
        s.initDNSListener(args)
        // Will run the sidecar injector in pilot.Only operates if /var/lib/istio/inject exists
        s.initSidecarInjector(args)
        s.initSDSCA(args)
    }

## 处理请求

### envoy 向pilot 发送请求

envoy 通过grpc 协议与 pilot-discovery 交互，因此首先找 ads.proto 文件

[ads.proto](https://github.com/envoyproxy/data-plane-api/blob/master/envoy/service/discovery/v2/ads.proto)

基于ads.proto 生成 ads.pb.go 文件`github.com/envoyproxy/go-control-plane/envoy/service/discovery/v2/ads.pb.go` 其中定义了 服务接口 AggregatedDiscoveryServiceServer，其实现类 DiscoveryServer，DiscoveryServer 方法分散于多个go 文件中

![](/public/upload/mesh/pilot_discovery_server.png)

grpc 请求通过 StreamAggregatedResources 来处理

    func (s *DiscoveryServer) StreamAggregatedResources(stream ads.AggregatedDiscoveryService_StreamAggregatedResourcesServer) error {
        peerInfo, ok := peer.FromContext(stream.Context())
        ...
        con := newXdsConnection(peerAddr, stream)
        ...
        reqChannel := make(chan *xdsapi.DiscoveryRequest, 1)
        go receiveThread(con, reqChannel, &receiveError)
        for {
            select {
            case discReq, ok := <-reqChannel:
                switch discReq.TypeUrl {
                case ClusterType:
                    ...
                    err := s.pushCds(con, s.globalPushContext(), versionInfo())
                case ListenerType:
                    ...
                case RouteType:
                    ...
                case EndpointType:
                    ...
                }
            case pushEv := <-con.pushChannel:
                ...
            }
        }
    }

DiscoveryServer 收到 ClusterType 的请求要生成 cluster 数据响应

    func (s *DiscoveryServer) pushCds(con *XdsConnection, push *model.PushContext, version string) error {
        rawClusters := s.generateRawClusters(con.node, push)
        ...
        response := con.clusters(rawClusters, push.Version)
        err := con.send(response)
        ...
        return nil
    }

cluster 数据实际由ConfigGenerator 生成

    func (s *DiscoveryServer) generateRawClusters(node *model.Proxy, push *model.PushContext) []*xdsapi.Cluster {
        rawClusters := s.ConfigGenerator.BuildClusters(node, push)
        ...
        return rawClusters
    }

数据来自PushContext.Services 方法

    func (configgen *ConfigGeneratorImpl) buildOutboundClusters(proxy *model.Proxy, push *model.PushContext) []*apiv2.Cluster {
        clusters := make([]*apiv2.Cluster, 0)
        networkView := model.GetNetworkView(proxy)
        for _, service := range push.Services(proxy) {
            ...
        }
        return clusters
    }

cluster 数据来自 PushContext的privateServicesByNamespace 和 publicServices， 通过代码可以发现，它们都是初始化时从model.Environment 取Service 数据的。并通过更新机制（待研究） 保持数据最新。

    func (ps *PushContext) Services(proxy *Proxy) []*Service {
        ...
        out := make([]*Service, 0)
        if proxy == nil {
            for _, privateServices := range ps.privateServicesByNamespace {
                out = append(out, privateServices...)
            }
        } else {
            out = append(out, ps.privateServicesByNamespace[proxy.ConfigNamespace]...)
        }
        out = append(out, ps.publicServices...)
        return out
    }


### pilot 监控到配合变化 将数据推给envoy（未完成）

### Mesh Configuration Protocol（未完成）

## pilot-agent

![](/public/upload/mesh/pilot_agent.png)

1. 所谓sidecar 容器， 不是直接基于envoy 制作镜像，容器启动后，entrypoint 也是envoy 命令
2. sidecar 容器的entrypoint 是 `/usr/local/bin/pilot-agent proxy`，首先生成 一个envoyxx.json 文件，然后 使用 exec.Command启动envoy
3. 进入sidecar 容器，`ps -ef` 一下， 是两个进程

        ## 具体明令参数 未展示
        UID        PID  PPID  C STIME TTY          TIME CMD
        1337         1     0  0 May09 ?        00:00:49 /usr/local/bin/pilot-agent proxy
        1337       567     1  1 09:18 ?        00:04:42 /usr/local/bin/envoy -c envoyxx.json


为什么要用pilot-agent？负责Envoy的生命周期管理（生老病死）

1. 启动envoy
2. 热更新envoy，poilt-agent只负责启动另一个envoy进程，其他由新旧两个envoy自行处理
3. 抢救envoy
4. 优雅关闭envoy



