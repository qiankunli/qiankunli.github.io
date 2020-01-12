---

layout: post
title: Pilot源码分析
category: 技术
tags: Mesh
keywords: Go

---

## 前言（未完成）

* TOC
{:toc}

[Istio Pilot代码深度解析](https://www.servicemesher.com/blog/201910-pilot-code-deep-dive/)

![](/public/upload/mesh/pilot_package.svg)

## pilot-discovery

**如果把Pilot看成一个处理数据的黑盒，则其有两个输入，一个输出**。

1. 目前Pilot的输入包括两部分数据来源：

    1. 服务数据（随着服务的启停、灰度等自动的）： 来源于各个服务注册表(Service Registry)，例如Kubernetes中注册的Service，Consul Catalog中的服务等。
    2. 配置规则（人为的）： 各种配置规则，包括路由规则及流量管理规则等，通过Kubernetes CRD(Custom Resources Definition)形式定义并存储在Kubernetes中。
2. Pilot的输出为符合xDS接口的数据面配置数据，并通过gRPC Streaming接口将配置数据推送到数据面的Envoy中。

![](/public/upload/mesh/pilot_input_output.svg)


对应的struct 的设计上，服务数据部分

![](/public/upload/mesh/pilot_service_object.png)

配置数据部分

![](/public/upload/mesh/pilot_config_object.png)




整体的感觉就是 configcontroller servicecontroller 想办法拿到数据，discovery server 根据 数据答复 envoy grpc 请求 或者监听到 变化后主动push （待验证）

![](/public/upload/mesh/pilot_object.png)

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

### envoy 向pilot 发送请求（未完成）

ads.proto 文件

[ads.proto](https://github.com/envoyproxy/data-plane-api/blob/master/envoy/service/discovery/v2/ads.proto)

基于ads.proto 生成 ads.pb.go 文件`github.com/envoyproxy/go-control-plane/envoy/service/discovery/v2/ads.pb.go` 其中定义了 服务接口 AggregatedDiscoveryServiceServer，其实现类 DiscoveryServer，DiscoveryServer 方法分散于多个go 文件中

下面就是深挖 DiscoveryServer 代码实现，看下DiscoveryServer 从哪拿的数据

上层是model 是逻辑，下层是adapter [Istio 技术与实践01： 源码解析之 Pilot 多云平台服务发现机制](https://juejin.im/post/5b965d646fb9a05d37617754)

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



