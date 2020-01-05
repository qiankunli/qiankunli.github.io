---

layout: post
title: istio学习
category: 架构
tags: Practice
keywords: window

---

## 简介

* TOC
{:toc}

类似产品 [SOFAMesh 介绍](https://www.sofastack.tech/projects/sofa-mesh/overview/)

[Istio 庖丁解牛一：组件概览](https://www.servicemesher.com/blog/istio-analysis-1/)未读

[使用 Istio 实现基于 Kubernetes 的微服务应用](https://www.ibm.com/developerworks/cn/cloud/library/cl-lo-implementing-kubernetes-microservice-using-istio/index.html)


## 安装手感——使用istioctl安装

[istio-1.4.2-linux.tar.gz](https://github.com/istio/istio/releases/download/1.4.2/istio-1.4.2-linux.tar.gz)

[Istio 1.4 部署指南](https://juejin.im/post/5e0062ae6fb9a0163a483ea5)istioctl 提供了多种安装配置文件，可以通过下面的命令查看：

    $ istioctl profile list
    Istio configuration profiles:
        default
        demo
        minimal
        remote
        sds

istio 包含istio-citadel/istio-egressgateway/istio-galley/istio-ingressgateway/istio-nodeagent/istio-pilot/istio-policy/istio-sidecar-injector/istio-telemetry/Grafana/istio-tracing/kiali/prometheus等组件。不同模式对各个组件进行了取舍，其中 minimal 模式下，只启动了istio-pilot 一个组件。

安装profile=demo的 istio

    $ istioctl manifest apply --set profile=demo \
    --set cni.enabled=true --set cni.components.cni.namespace=kube-system \
    --set values.gateways.istio-ingressgateway.type=ClusterIP

容器列表如下：

    ➜  ~ kubectl get pods -n istio-system
    NAME                                      READY   STATUS             RESTARTS   AGE
    grafana-6b65874977-bc8tm                  1/1     Running            0          141m
    istio-citadel-86dcf4c6b-kp52x             1/1     Running            0          8d
    istio-egressgateway-68f754ccdd-sndrl      1/1     Running            0          141m
    istio-galley-5fc6d6c45b-sg6dw             1/1     Running            0          141m
    istio-ingressgateway-6d759478d8-b476j     1/1     Running            0          141m
    istio-pilot-5c4995d687-jfp7l              1/1     Running            0          141m
    istio-policy-57b99968f-xckd9              1/1     Running            36         141m
    istio-sidecar-injector-746f7c7bbb-xzzpw   1/1     Running            0          8d
    istio-telemetry-854d8556d5-9754v          1/1     Running            1          141m
    istio-tracing-c66d67cd9-pszx9             0/1     CrashLoopBackOff   47         141m
    kiali-8559969566-5svn6                    1/1     Running            0          141m
    prometheus-66c5887c86-62c9l               1/1     Running            0          8d

istioctl 提供了一个子命令来从本地打开各种 Dashboard。例如，要想在本地打开 Grafana 页面，只需执行下面的命令：

    ## 自动打开浏览器
    $ istioctl dashboard grafana
    http://localhost:36813


## 整体架构

![](/public/upload/practice/istio.jpg)

控制平面的三大模块，其中的Pilot和Citadel/Auth都不直接参与到traffic的转发流程，因此他们不会对运行时性能产生直接影响。

### Envoy

Envoy 是 Istio 中最基础的组件，所有其他组件的功能都是通过调用 Envoy 提供的 API，在请求经过 Envoy 转发时，由 Envoy 执行相关的控制逻辑来实现的。

[浅谈Service Mesh体系中的Envoy](https://yq.aliyun.com/articles/606655)

类似产品 [MOSN](https://github.com/sofastack/sofa-mosn) [MOSN 文档](https://github.com/sofastack/sofa-mosn)

### Mixer

![](/public/upload/practice/istio_mixer.svg)

mixer 的变更是比较多的，有v1 architecture 和 v2 architecture，社区还尝试将其与proxy/envoy 合并。

[WHY DOES ISTIO NEED MIXER?](https://istio.io/faq/mixer/#why-mixer)Mixer provides a rich intermediation layer between the Istio components as well as Istio-based services, and the infrastructure backends used to perform access control checks and telemetry capture. Mixer enables extensible policy enforcement and control within the Istio service mesh. It is responsible for insulating（隔离） the proxy (Envoy) from details of the current execution environment and the intricacies of infrastructure backends. 

理解“为什么需要一个Mixer” 的关键就是 理解infrastructure backend， 它们可以是Logging/metric 等，mixer 将proxy 与这些系统隔离（proxy通常是按照无状态目标设计的），代价就是每一次proxy间请求需要两次与mixer的通信 影响了性能，这也是社区想将proxy与mixer合并的动机（所以现在proxy是不是无状态就有争议了）。

[Service Mesh 发展趋势(续)：棋到中盘路往何方](https://www.sofastack.tech/blog/service-mesh-development-trend-2/)

![](/public/upload/practice/istio_mixer_evolution.png)

### pilot

[服务网格 Istio 初探 -Pilot 组件](https://www.infoq.cn/article/T9wjTI2rPegB0uafUKeR)

![](/public/upload/practice/istio_pilot_detail.png)

1. Pilot 的架构，最下面一层是 Envoy 的 API，提供 Discovery Service 的 API，这个 API 的规则由 Envoy 约定，Pilot 实现 Envoy API Server，**Envoy 和 Pilot 之间通过 gRPC 实现双向数据同步**。
2. Pilot 最上面一层称为 Platform Adapter，这一层不是 Kubernetes 调用 Pilot，而是 **Pilot 通过调用 Kubernetes 来发现服务之间的关**系，Pilot 通过在 Kubernetes 里面注册一个 Controller 来监听事件，从而获取 Service 和 Kubernetes 的 Endpoint 以及 Pod 的关系。

Istio 通过 Kubernets CRD 来定义自己的领域模型，使大家可以无缝的从 Kubernets 的资源定义过度到 Pilot 的资源定义。

## 端到端流转

一个istio 自带的Bookinfo 为例，对应[istio-1.4.2-linux.tar.gz](https://github.com/istio/istio/releases/download/1.4.2/istio-1.4.2-linux.tar.gz) 解压后`istio-1.4.2/samples/bookinfo`

    kubectl label namespace default istio-injection=enabled
    kubectl apply -f samples/bookinfo/platform/kube/bookinfo.yaml
    # 安装 bookinfo 的 ingress gateway：
    kubectl apply -f samples/bookinfo/networking/bookinfo-gateway.yaml

全流程都是 http 协议

![](/public/upload/practice/istio_bookinfo.jpg)

![](/public/upload/practice/istio_envoy_flow.png)

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

## 其它

任何软件架构设计，其核心都是围绕数据展开的，基本上如何定义数据结构就决定了其流程的走向，剩下的不外乎加上一些设计手法，抽离出变与不变的部分，不变的部分最终会转化为程序的主流程，基本固化，变的部分尽量保证拥有良好的扩展性、易维护性，最终会转化为主流程中各个抽象的流程节点。

