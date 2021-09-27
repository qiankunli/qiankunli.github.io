---

layout: post
title: Kubernetes监控
category: 技术
tags: Kubernetes
keywords: Kubernetes monitor

---

## 简介

* TOC
{:toc}

![](/public/upload/go/prometheus_k8s.png)

[容器监控实践—K8S常用指标分析](http://www.xuyasong.com/?p=1717)

[Kubernetes 稳定性保障手册 -- 极简版](https://mp.weixin.qq.com/s/kZmi2gK16qe2yMYMRS3Etg)未细读
[KubeNode：阿里巴巴云原生容器基础设施运维实践](https://mp.weixin.qq.com/s/Fm_Pbz0ltu2mpFlDP74_fw) 仅仅是一个kubelet 是不够用的。

[KubeEye：Kubernetes 集群自动巡检工具](https://segmentfault.com/a/1190000039173086)

[报警的哲学](https://mp.weixin.qq.com/s/lJRPt7I0SeUwZ4HhVZn8AQ)追踪所有收到的报警。如果收到了报警，而人们只是说 "我看了，没有什么问题"，这是一个相当强烈的信号，你需要删除报警规则，或者降级，或者以其他方式收集数据。准确率低于 50% 报警是不能使用的；即使是那些 10% 的假阳性警报，也值得多加考虑是否对齐进行修改。

[通过Kubernetes监控探索应用架构，发现预期外的流量](https://mp.weixin.qq.com/s/RasRiNYo8OyTTselaSwHKA) 是一个系列文章，阿里已经将应用 ==> k8s ==> 内核监控打通。构建拓扑图 ==>  发现异常流量、阈值报警（拓扑图的边黄色或红色）==> 异常流量上下游各种信息。

## 容器监控与常规监控的差异

[基于Prometheus的云原生监控系统架构演进](https://mp.weixin.qq.com/s/SBqYGeWDMQwmente8JBaHA)

[Kubernetes监控在小米的落地](https://mp.weixin.qq.com/s/ewwD6A3-ClbotdfFmYY3KA) 为了更方便的管理容器，Kubernetes对Container进行了封装，拥有了Pod、Deployment、Namespace、Service等众多概念。与传统集群相比，Kubernetes集群监控更加复杂：

1. 监控维度更多，除了传统物理集群的监控，还包括核心服务监控（apiserver，etcd等）、容器监控、Pod监控、Namespace监控等。
2. 监控对象动态可变，在集群中容器的销毁创建十分频繁，无法提前预置。
3. 监控指标随着容器规模爆炸式增长，如何处理及展示大量监控数据。
4. 随着集群动态增长，监控系统必须具备动态扩缩的能力。

## 能搞到哪些metric

在Kubernetes从1.10版本后采用Metrics Server作为默认的性能数据采集和监控，主要用于提供核心指标（Core Metrics），包括Node、Pod的CPU和内存使用指标。对其他自定义指标（Custom Metrics）的监控则由Prometheus等组件来完成。

k8s 社区对k8s 监控的表述 [Kubernetes monitoring architecture](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/instrumentation/monitoring_architecture.md) 将metric 分为

1. core metrics, which are metrics that Kubernetes understands and uses for operation of its internal components and core utilities. metric 应用于k8s 内部组件，比如调度、**扩缩容**、dashboard、kubectl top. 由k8s 提供统一规范和支持（即metrics-server）。
2. non-core metrics

![](/public/upload/go/kubernetes_metric.png)

[prometheus使用missing-container-metrics监控pod oomkill](https://mp.weixin.qq.com/s/IDmuoPOcYsGISrYb1n9aKw)

### Metrics Server/cadvisor

Metrics server复用了api-server的库来实现自己的功能，比如鉴权、版本等，为了实现将数据存放在内存中，去掉了默认的etcd存储，引入了内存存储。因为存放在内存中，因此监控数据是没有持久化的，可以通过第三方存储来拓展

![](/public/upload/kubernetes/kubernetes_metric_server.png)

1. Metrics API URI 为 `/apis/metrics.k8s.io/`，在 `k8s.io/metrics` 维护
2. 必须部署 metrics-server 才能使用该 API，metrics-server 通过调用 Kubelet Summary API 获取数据，Summary API 返回的信息，既包括了 cAdVisor 的监控数据，也包括了 kubelet 本身汇总的信息。Pod 的监控数据是从kubelet 的 Summary API （即 `<kubelet_ip>:<kubelet_port>/stats/summary`）采集而来的。
3. **Metrics server以Deployment 形式存在**，复用了api-server的库来实现自己的功能，比如鉴权、版本等，为了实现将数据存放在内存中吗，去掉了默认的etcd存储，引入了内存存储。因此监控数据是没有持久化的，可以通过第三方存储来拓展

从cadvisor 视角看

![](/public/upload/kubernetes/kubernetes_cadvisor.png)

cadvisor 指标分析

cadvisor 是监控容器的，容器像物理机一样为业务提供运算资源，因此按照 USE 对容器的指标进行分析。[监控的黄金指标](https://zhuanlan.zhihu.com/p/75875469)

cadvisor 指标以`container_` 为前缀，包括`container_cpu_*`,`container_memory_*`, `container_fs_*`, `container_network_*` 等，还有 `container_spec_*` 获取了 container 配置相关的内容。部分指标如下

cpu
1. container_cpu_user_seconds_total —“用户”时间的总数（即不在内核中花费的时间）
2. container_cpu_system_seconds_total —“系统”时间的总数（即在内核中花费的时间）
3. container_cpu_usage_seconds_total—以上总和
4. container_cpu_cfs_throttled_seconds_total  当容器超出其CPU限制时，Linux运行时将“限制”该容器并在container_cpu_cfs_throttled_seconds_total指标中记录其被限制的时间

cAdvisor中提供的内存指标是从node_exporter公开的43个内存指标的子集。以下是容器内存指标：

1. container_memory_cache-页面缓存的字节数。
2. container_memory_rss -RSS的大小（以字节为单位）。
3. container_memory_swap-容器交换使用量（以字节为单位）。
4. container_memory_usage_bytes-当前内存使用情况（以字节为单位,包括所有内存，无论何时访问。) 包括了文件系统缓存
5. container_memory_max_usage_bytes- 以字节为单位记录的最大内存使用量。
6. container_memory_working_set_bytes-当前工作集（以字节为单位）。
7. container_memory_failcnt-内存使用次数达到限制。
8. container_memory_failures_total-内存 分配失败的累积计数。

### node-exporter

node-exporter提供了近1000个指标，以`node_` 为前缀，包括`node_cpu_*`,`node_memory_*`, `node_filesystem_*/node_disk_*`, `node_network_*` 等。

```yaml
apiVersion: extensions/v1beta1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: monitoring
  ...
  spec:
    tolerations:
    - key: node-role.kubernetes.io/master
      effect: NoSchedule
    hostNetwork: true
    hostPID: true
    hostIPC: true
    securityContext:
      runAsUser: 0
```

利用Kubernetes DaemonSet控制器在集群中的每个节点上自动node-exporter pod。
1. 启用systemd收 集器，并指定要监控的特定服务的正则表达式，而不是主机上的所有服务。
2. 使用toleration来确保 node-exporter pod也会被调度到Kubernetes主节点，而不仅是工作节点。
3. 以用户0或root运行pod(这允许访问systemd)，并 且还启用了hostNetwork、hostPID和hostIPC，以指定实例的网络、进程和IPC命名空间在容器中可用。

```yaml
containers:
- images: prom/node-exporter:latest
  name: node-exporter
  volumeMounts:
    - mountPath: /run/systemd/private
      name: systemd-socket
      readOnly: true
  args:
    - "--collector.systemd"
    - "--collector.systemd.unit-whitelist=(docker|ssh|rsyslog|kubelet).service"
  ports:
    - containerPort: 9100
      hostPort: 9100
      name: scrape
```

配置一个Prometheus scrape job，结合Kubernetes daemonset, 只需要定义一次，未来所有Kubernetes服务端点都将被自动发现 和监控。

### k8s 组件 指标分析


Kubernetes 各组件的 Healthz 和 Metrics API 

|Components|	Healthz API|	Metrics API|
|---|---|---|
|Apiserver|	`:6443/healthz`|	`:6443/metrics`|
|Controller Manager|	`:10252/healthz`|	`:10252/metrics`|
|Scheduler|	`:10251/healthz`|	`:10251/metrics`|
|Kube-proxy|	`:10249/healthz`|	`:10249/metrics`|
|Kubelet|	`:10248/healthz`|	`:10250/metrics`|
|ETCD|	`:2379/healthz`|	`:2379/metrics`|

kube-apiserver 是集群所有请求的入口，指标的分析可以反应集群的健康状态。Apiserver 的指标可以分为以下几大类：

1. 请求速率和延迟,  `apiserver_request_*/apiserver_response_*`
2. 控制器队列的性能, `apiserver_admission_*`
3. etcd 的性能, `etcd_*`
4. 进程状态：文件系统、内存、CPU
5. golang 程序的状态：GC、进程、线程, `go_gc_*/go_info`

[Pinterest如何平稳扩展K8s？](https://mp.weixin.qq.com/s/YwZsSfWO-xIvbJlLk9146w)监控apiserver，我们通过查看 QPS 和并发请求、错误率，以及请求延迟来监控 kube-apiserver 的负载。我们也可以将流量按照资源类型、请求动词以及相关的服务账号进行细分。而对于 listing 这类的昂贵流量，我们通过对象计数和字节大小来计算请求负载，即使只有很小的 QPS，这类流量也很容易导致 kube-apiserver 过载。最后，我们还监测了 etcd 的 watch 事件处理 QPS 和延迟处理的计数，以作为重要的服务器性能指标。

### ETCD 指标分析

Kubernetes使用etcd来存储集群中组件的所有状态，是 Kubernetes数据库，监视etcd的性能和行为应该是整个Kubernetes监控计划的一部分。

etcd服务器指标以 `etcd_*` 为前缀，分为几个主要类别：

1. Leader的存在和Leader变动率, `etcd_server_leader_*`
2. 请求已提交/已应用/正在等待/失败, `etcd_server_proposals_*`
3. 磁盘写入性能 , `etcd_disk_*`
4. 入站gRPC统计信息，集群内gRPC统计信息, `etcd_grpc_*`

### 业务Pod监控

业务监控 一般由业务直接暴露metric或通过边车模式暴露metric

**在Pod 或Service 中定义注解，可以让Prometheus 自动发现当前metric endpoint 并抓取数据**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: tornado-db
  annotations:
    prometheus.io/scrape: 'true'  # 告诉Prometheus抓取这个服务
    prometheus.io/port: '9104'    # 告诉 Prometheus要抓取的端口，将被放入__address__标签中
```

### kube-state-metrics

[kubernetes/kube-state-metrics](https://github.com/kubernetes/kube-state-metrics)

1. kube-state-metrics is a simple service that listens to the Kubernetes API server and generates metrics about the state of the objects. It is not focused on the health of the individual Kubernetes components, but rather on the health of the various objects inside, such as deployments, nodes and pods.  上文关注的是 k8s组件是否健康， kube-state-metrics 关注的Kubernetes 的object 是否健康。
2. kube-state-metrics uses client-go to talk with Kubernetes clusters，**将Kubernetes的结构化信息转换为metrics**，很多metric 都来自 Kubernetes object 的Status 或 Conditions 等字段，`kubectl describe xx` 一样可以看到。
3. k8s custom resource 比如 verticalpodautoscalers 默认不采集， 需要额外配置
4. 以Deployment 方式运行，以Service 对外服务

所有metric 以 `kube_*` 为前缀，每一个k8s resource 对应一批metric [Exposed Metrics](https://github.com/kubernetes/kube-state-metrics/tree/master/docs) ，以`kube_资源名_*` 为前缀，以`kube_deployment_*`为例

```
kube_deployment_status_replicas
kube_deployment_status_replicas_available
kube_deployment_spec_replicas
...
```

以pod为例：

```
kube_pod_info
kube_pod_owner
kube_pod_status_phase
kube_pod_status_ready
kube_pod_status_scheduled
kube_pod_container_status_waiting
kube_pod_container_status_terminated_reason  # 比如OOMKilled
```

Kube-state-metrics self metrics，描述自身的工作状态，比如 

```
kube_state_metrics_list_total{resource="*v1.Node",result="success"} 1
kube_state_metrics_list_total{resource="*v1.Node",result="error"} 52
kube_state_metrics_watch_total{resource="*v1beta1.Ingress",result="success"} 1
```

**kube-state-metrics 可以采集到的 自定义的k8s label（比如deployment等），基于此经常将kube-state-metrics metric 与其它metric 聚合后 制作报警规则**。

[有道 Kubernetes 容器API监控系统设计和实践](https://mp.weixin.qq.com/s/K6UJnnpbhciHyvrACo1xAw)

### 基于k8s 抓取metric

[Kubernetes 集群监控 kube-prometheus 自动发现](https://cloud.tencent.com/developer/article/1802679)
Prometheus 通过与 Kubernetes API 集成主要支持5种服务发现模式：
1. Node, 适用于与主机相关的监控资源
2. Service、Ingress, 适用于通过黑盒监控的场景，如对服务的可用性以及服务质量的监控
3. Pod、Endpoints, 获取 Pod 实例的监控数据

通过添加额外的配置来进行服务发现进行自动监控（自声明），比如 在 kube-prometheus 当中去自动发现并监控具有 `prometheus.io/scrape=true` 这个 annotations 的 Service。

```
# kubectl describe service xxx
Annotations:        example.com/port: 2121
                    example.com/scrape: true
IP:                10.103.173.42
Port:              <unset>  8080/TCP
Endpoints:         172.31.10.228:8080
```


```yaml
- job_name: 'kubernetes-service-endpoints'
    kubernetes_sd_configs:
      role: endpoints # 从列出的服务端点发现目标，这个endpoints来自于Kubernetes中的service， 每一个service都有对应的endpoints，所以endpoints角色就是用来发现server对应的pod的IP的
    relabel_configs:  # 对采集过来的指标做二次处理，比如要什么不要什么以及替换什么等等
    - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape] # 以__meta_开头的这些元数据标签都是实例中包含的
      action: keep
      regex: true   # 仅抓取到的具有 "prometheus.io/scrape: true" 的annotation的端点
    - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scheme]
      action: replace # 根据regex来去匹配source_labels标签上的值，并将并将匹配到的值写入target_label中
      target_label: __scheme__
      regex: (https?)
    - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
    - source_labels: [__address__, __meta_kubernetes_service_annotation_prometheus_io_port] # 以__开头的标签通常是系统内部使用的
      action: replace
      target_label: __address__   
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
    # 下面主要是为了给样本添加额外信息
    - action: labelmap  
      regex: __meta_kubernetes_service_label_(.+)
    - source_labels: [__meta_kubernetes_namespace]    # namespace 名称
      action: replace
      target_label: kubernetes_namespace
    - source_labels: [__meta_kubernetes_service_name] # service 对象的名称
      action: replace
      target_label: kubernetes_name
```

## 需要哪些 alert rule

[monitoring.mixin](https://monitoring.mixins.dev) 列出了各个组件建议配置的alert 规则。

指导原则：宁缺毋滥。其实有十几个报警就不少了，再多也处理不过来。 可以参照《Prometheus 监控实战》中提到的对Prometheus 的报警规则。

## 制作哪些dashboard

Grafana 官方有一个 [dashboard 市场](https://grafana.com/grafana/dashboards)，可以针对各个组件找到 全面丰富的dashboard 

[KubeEye：Kubernetes 集群自动巡检工具](https://segmentfault.com/a/1190000039173086)

## 其它

![](/public/upload/kubernetes/kubernetes_monitor.png)

[当容器应用越发广泛，我们又该如何监测容器？](https://mp.weixin.qq.com/s/i_OpJyCJQR5ZbyorDppisg)阿里云推出 Kubernetes 监测服务
1. 代码无侵入：通过旁路技术，无需代码埋点，即可获取到网络性能数据。
2. 多语言支持：通过内核层进行网络协议解析，支持任意语言及框架。
3. 低耗高性能：基于 eBPF 技术，以极低消耗获取网络性能数据。
4. 资源自动拓扑：通过网络拓扑，资源拓扑展示相关资源的关联情况。
5. 数据多维展现：支持可观测的各种类型数据（监测指标、链路、日志和事件）。
6. 打造关联闭环：完整关联架构层、应用层、容器运行层、容器管控层、基础资源层相关可观测数据。
应用在以下场景：
1. 通过 Kubernetes 监测的系统默认或者自定义巡检规则，发现节点，服务与workload的异常。
2. 使用 Kubernetes 监测定位服务与工作负载响应失败根因，Kubernetes 监测通过分析网络协议对失败请求进行明细存储，利用失败请求指标关联的失败请求明细定位失败原因。
3. 使用 Kubernetes 监测定位服务与工作负载响应慢根因，Kubernetes 监测通过抓取网络链路关键路径的指标，查看 DNS 解析性能，TCP 重传率，网络包 rtt 等指标。利用网络链路关键路径的指标定位响应慢的原因，进而优化相关服务。
4. 使用 Kubernetes 监测探索应用架构，发现预期外的网络流量。Kubernetes 监测支持查看全局流量构建起来的拓扑大图，支持配置静态端口标识特定服务。

