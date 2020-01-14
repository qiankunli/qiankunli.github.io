---

layout: post
title: 一个sidecar的自我修养
category: 技术
tags: Mesh
keywords: pilot service mesh

---

## 前言

* TOC
{:toc}

## 分类与配置分类

Envoy按照使用 场景可以分三种：

1. sidecar，和应用一起部署在容器中，对进出应用服务的容量进行拦截
2. router，作为独立的代理服务，对应用的L4/L7层流量进行代理
3. ingress，作为集群入口的Ingress代理，对集群的入口流量进行拦截和代理

router 和ingress 均属于和应用服务不在一起的纯代理场景，可以归为一类，成为Gateway模式。对于sidecar 模式来说， envoy 负责服务出入方向流量的透明拦截，并且出入方向的流量在监听管理、路由管理等方面有很大的区别，因此**sidecar 的xds配置是按照出入方向分别进行组织和管理**。因此从xds 配置的视角上 配置可以划分为

1. sidecar inbound，inbound 将发往本节点的流量转发到 对应的服务节点，因此inbound 方向的集群和路由信息都比较确定：单一的集群，单一的VirtualHost，并且集群固定只有一个节点信息。对于Http来说，会拼装HTTP 对应的路由信息，对于TCP来说，直接通过Tcp Proxy方式进行路由，只做全局统计和管控，无法进行协议相关的链路治理。
2. sidecar outbound，从当前节点发往节点外的流量。**根据协议的不同有所不同，待进一步认识**。
3. gateway


envoy 是一个proyx 组件，一个proxy 具体的说是listener、filter、route、cluster、endpoint 的协同工作

![](/public/upload/practice/istio_envoy_flow.png)

istio 对流量采取了透明拦截的方式

    ## 所有入口流量 redirect 到 8090 端口
    Iptables -t nat -A PREROUTING -p tcp -j REDIRECT -to-port 8090
    ## 所有出口流量 redirect 到 8090 端口
    Iptables -t nat -A OUTPUT -p tcp -j REDIRECT -to-port 8090

目标端口被改写后， 可以通过SO_ORIGINAL_DST TCP 套件获取原始的ipport

为了实现正确的流量路由与转发，envoy 的监听器分为两类

1. 虚拟监听器，需要绑定相应的端口号，iptables 拦截的流量会转发到这个端口上
2. 真实监听器，用于处理iptables 拦截前的”真实目的地址“，虚拟机监听器接收到监听请求时，按照一定的匹配规则找到对应的真实监听器进行处理。真实监听器因为不需要和网络交互，因此不需要配置和绑定端口号。


## 端到端流转案例

一个istio 自带的Bookinfo 为例，对应[istio-1.4.2-linux.tar.gz](https://github.com/istio/istio/releases/download/1.4.2/istio-1.4.2-linux.tar.gz) 解压后`istio-1.4.2/samples/bookinfo`

    kubectl label namespace default istio-injection=enabled
    kubectl apply -f samples/bookinfo/platform/kube/bookinfo.yaml
    # 安装 bookinfo 的 ingress gateway：
    kubectl apply -f samples/bookinfo/networking/bookinfo-gateway.yaml

全流程都是 http 协议

![](/public/upload/practice/istio_bookinfo.jpg)

`istioctl proxy-config listener $podname` 可以查看Pod 中的具有哪些 Listener，也可以使用`istioctl proxy-config listener $podname -o json` 查看更详细的配置

### Productpage服务调用Reviews服务的请求流程

[Istio流量管理实现机制深度解析](https://zhaohuabing.com/post/2018-09-25-istio-traffic-management-impl-intro/)Productpage服务调用Reviews服务的请求流程

![](/public/upload/practice/bookinfo_envoy_flow.png)

将details 服务扩容到2个实例，可以通过Pilot的调试接口获取该Cluster的endpoint`http://pilot_service_ip:15014/debug/edsz` ，可以看到 details 对应的cluster的endpoints 变成了两个。查看 productpage pod中 envoy 的endpoint 配置发现也对应有了2个endpoint

    $ istioctl pc endpoint productpage-v1-596598f447-nn64q
    ENDPOINT                STATUS      OUTLIER CHECK     CLUSTER
    10.20.0.10:9080         HEALTHY     OK                outbound|9080||details.default.svc.cluster.local
    10.20.0.2:9080          HEALTHY     OK                outbound|9080||details.default.svc.cluster.local

### 请求从ingress/gateway 流向productpage

[istio网络转发分析](https://yq.aliyun.com/articles/564983)

涉及到的 istio kubernetes resource

1.  Gateway描述了在网络边缘运行的负载均衡器，用于接收传入或传出的HTTP / TCP连接。

        apiVersion: networking.istio.io/v1alpha3
        kind: Gateway

2. VirtualService实际上将Kubernetes服务连接到Istio网关。

        apiVersion: networking.istio.io/v1alpha3
        kind: VirtualService