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

[Envoy 官方文档中文版](https://www.servicemesher.com/envoy/)

![](/public/upload/mesh/envoy_work.jpg)

Envoy的工作模式如图所示，横向是管理平面/管理流，纵向是数据流。Envoy会暴露admin的API，可以通过API查看Envoy中的路由或者集群的配置。

## 分类与配置分类

Envoy按照使用 场景可以分三种：

1. sidecar，和应用一起部署在容器中，对进出应用服务的容量进行拦截
2. router，作为独立的代理服务，对应用的L4/L7层流量进行代理
3. ingress，作为集群入口的Ingress代理，对集群的入口流量进行拦截和代理

router 和ingress 均属于和应用服务不在一起的纯代理场景，可以归为一类，成为Gateway模式。对于sidecar 模式来说， envoy 负责服务出入方向流量的透明拦截，并且出入方向的流量在监听管理、路由管理等方面有很大的区别，因此**sidecar 的xds配置是按照出入方向分别进行组织和管理**。因此从xds 配置的视角上 配置可以划分为

1. sidecar inbound，inbound 将发往本节点的流量转发到 对应的服务节点，因此inbound 方向的集群和路由信息都比较确定：单一的集群，单一的VirtualHost，并且集群固定只有一个节点信息。对于Http来说，会拼装HTTP 对应的路由信息，对于TCP来说，直接通过Tcp Proxy方式进行路由，只做全局统计和管控，无法进行协议相关的链路治理。
2. sidecar outbound，从当前节点发往节点外的流量。**根据协议的不同有所不同，待进一步认识**。
3. gateway


envoy 是一个proxy 组件，一个proxy 具体的说是listener、filter、route、cluster、endpoint 的协同工作

![](/public/upload/practice/istio_envoy_flow.png)

[深入解读Service Mesh背后的技术细节](https://mp.weixin.qq.com/s/hq9KTc9fm8Nou8hXmqdKuw)istio 对流量采取了透明拦截的方式

![](/public/upload/mesh/envoy_iptables.jpeg)

1. 在PREROUTING规则中，使用这个转发链，从而进入容器的所有流量，都被先转发到envoy的15000端口。
2. envoy作为一个代理，已经被配置好了，将请求转发给productpage程序。
3. productpage程序接受到请求，会转向调用外部的reviews或者ratings，当productpage往后端进行调用的时候，就碰到了output链，这个链会使用转发链，将所有出容器的请求都转发到envoy的15000端口。**这样无论是入口的流量，还是出口的流量，全部用envoy做成了汉堡包**。
4. envoy根据服务发现的配置，知道reviews或者ratings如何访问，于是做最终的对外调用。iptables规则会对从envoy出去的流量做一个特殊处理，允许他发出去，不再使用上面的output规则。

目标端口被改写后， 可以通过SO_ORIGINAL_DST TCP 套件获取原始的ipport

为了实现正确的流量路由与转发，envoy 的监听器分为两类

1. 虚拟监听器，需要绑定相应的端口号，iptables 拦截的流量会转发到这个端口上
2. 真实监听器，用于处理iptables 拦截前的”真实目的地址“，虚拟机监听器接收到监听请求时，按照一定的匹配规则找到对应的真实监听器进行处理。真实监听器因为不需要和网络交互，因此不需要配置和绑定端口号。

## 配置与xds协议

Envoy是一个高性能的C++写的proxy转发器，那Envoy如何转发请求呢？需要定一些规则，然后按照这些规则进行转发。规则可以是静态的，放在配置文件中的，启动的时候加载，要想重新加载，一般需要重新启动。当然最好的方式是规则设置为动态的，放在统一的地方维护，这个统一的地方在Envoy眼中看来称为Discovery Service，Envoy过一段时间去这里拿一下配置，就修改了转发策略。无论是静态的，还是动态的，在配置里面往往会配置四个东西。

||xds|备注|
|---|---|---|
|Listener|LDS|既然是proxy，就得监听一个端口|
|Endpoints|EDS|目标的ip地址和端口，这个是proxy最终将请求转发到的地方|
|Routes|RDS|一个cluster是具有完全相同行为的多个endpoint<br>它们组成一个Cluster，从cluster到endpoint的过程称为负载均衡|
|Cluters|CDS|有时候多个cluster具有类似的功能，但是是不同的版本号，<br>可以通过route规则，选择将请求路由到某一个版本号|

![](/public/upload/mesh/envoy_config.png)

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