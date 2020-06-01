---

layout: post
title: istio学习
category: 架构
tags: Mesh
keywords: istio

---

## 简介

* TOC
{:toc}

![](/public/upload/mesh/istio_functions.png)

## 安装手感——使用istioctl安装

github 下载istio，解压完毕后，istio-${version}/bin 下包含istioctl，用来进行istio 的安装及日常运维

```
$ istioctl profile list
Istio configuration profiles:
    default
    demo
    minimal
    remote
    sds
```

istio 包含多个组件，不同模式对各个组件进行了取舍

![](/public/upload/mesh/istio_configuration_profile.png)


安装profile=demo的 istio
```
$ istioctl manifest apply --set profile=demo \
--set cni.enabled=true --set cni.components.cni.namespace=kube-system \
--set values.gateways.istio-ingressgateway.type=ClusterIP
```

容器列表如下：

```
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
```

istioctl 提供了一个子命令来从本地打开各种 Dashboard。例如，要想在本地打开 Grafana 页面，只需执行下面的命令：

```
## 自动打开浏览器
$ istioctl dashboard grafana
http://localhost:36813
```

## 整体架构

![](/public/upload/mesh/istio.png)

### Envoy

Envoy 是 Istio 中最基础的组件，所有其他组件的功能都是通过调用 Envoy 提供的 API，在请求经过 Envoy 转发时，由 Envoy 执行相关的控制逻辑来实现的。

[浅谈Service Mesh体系中的Envoy](https://yq.aliyun.com/articles/606655)

类似产品 [MOSN](https://github.com/sofastack/sofa-mosn) [MOSN 文档](https://github.com/sofastack/sofa-mosn)

### pilot

[服务网格 Istio 初探 -Pilot 组件](https://www.infoq.cn/article/T9wjTI2rPegB0uafUKeR)

![](/public/upload/practice/istio_pilot_detail.png)

1. Pilot 的架构，最下面一层是 Envoy 的 API，提供 Discovery Service 的 API，这个 API 的规则由 Envoy 约定，Pilot 实现 Envoy API Server，**Envoy 和 Pilot 之间通过 gRPC 实现双向数据同步**。
2. Pilot 最上面一层称为 Platform Adapter，这一层不是 Kubernetes 调用 Pilot，而是 **Pilot 通过调用 Kubernetes 来发现服务之间的关**系，Pilot 通过在 Kubernetes 里面注册一个 Controller 来监听事件，从而获取 Service 和 Kubernetes 的 Endpoint 以及 Pod 的关系。

Istio 通过 Kubernets CRD 来定义自己的领域模型，使大家可以无缝的从 Kubernets 的资源定义过度到 Pilot 的资源定义。

## 其它

任何软件架构设计，其核心都是围绕数据展开的，基本上如何定义数据结构就决定了其流程的走向，剩下的不外乎加上一些设计手法，抽离出变与不变的部分，不变的部分最终会转化为程序的主流程，基本固化，变的部分尽量保证拥有良好的扩展性、易维护性，最终会转化为主流程中各个抽象的流程节点。

控制平面和数据平面解耦，主要基于变和不变的考虑， 数据平面可以说是Service Mesh的内核，负责提供Service Mesh的最核心价值， 因此从架构设计上考虑， 应该尽量减少核心内核的变化，而将变化频繁的控制逻辑移到控制平面，可以很好的保证数据平面的稳定性和可维护性。

任何一个系统，随着使用场景和使用方式的不断变化，随时会面对很多新的挑战。为了应对这些挑战，需要保证在内核基本稳定的前提下建立一套完善的插件机制。插件从实现层面看其实很简单，本质上是一个钩子回调函数，插件注册就是将钩子回调函数挂在插件机制上，在事件到来时，触发回调函数的调用。因此研究插件机制有两点：

1. 插件的抽象
2. 回调和通知的机制

### 相关文章

[使用 Istio 实现基于 Kubernetes 的微服务应用](https://www.ibm.com/developerworks/cn/cloud/library/cl-lo-implementing-kubernetes-microservice-using-istio/index.html)

[蚂蚁金服大规模微服务架构下的Service Mesh探索之路](https://www.servicemesher.com/blog/the-way-to-service-mesh-in-ant-financial/) 很不错的文章 

### Mixer（已被进行重大调整）

![](/public/upload/practice/istio_mixer.svg)

mixer 的变更是比较多的，有v1 architecture 和 v2 architecture，社区还尝试将其与proxy/envoy 合并。

[WHY DOES ISTIO NEED MIXER?](https://istio.io/faq/mixer/#why-mixer)Mixer provides a rich intermediation layer between the Istio components as well as Istio-based services, and the infrastructure backends used to perform access control checks and telemetry capture. Mixer enables extensible policy enforcement and control within the Istio service mesh. It is responsible for insulating（隔离） the proxy (Envoy) from details of the current execution environment and the intricacies of infrastructure backends. 

理解“为什么需要一个Mixer” 的关键就是 理解infrastructure backend， 它们可以是Logging/metric 等，mixer 将proxy 与这些系统隔离（proxy通常是按照无状态目标设计的），代价就是每一次proxy间请求需要两次与mixer的通信 影响了性能，这也是社区想将proxy与mixer合并的动机（所以现在proxy是不是无状态就有争议了）。

[Service Mesh 发展趋势(续)：棋到中盘路往何方](https://www.sofastack.tech/blog/service-mesh-development-trend-2/)

![](/public/upload/practice/istio_mixer_evolution.png)

istio中的mixer 模板、adapter 适配器均可通过代码自动生成功能 生成，增加新的模板时，只需增加模板的proto 描述，调用mixer_codegen.sh即可。