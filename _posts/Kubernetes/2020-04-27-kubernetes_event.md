---

layout: post
title: Kubernetes events学习及应用
category: 架构
tags: Kubernetes
keywords:  Kubernetes event

---

## 简介

* TOC
{:toc}

## Kubernetes events 

### 是什么

[Understanding Kubernetes cluster events](https://banzaicloud.com/blog/k8s-cluster-logging/)Kubernetes events are objects that show you what is happening inside a cluster, such as what decisions were made by the scheduler or why some pods were evicted from the node. All core components and extensions (operators) may create events through the API Server.

Developers, application and infrastructure operators can use the `kubectl describe` command against specific resources, or use the more generic `kubectl get event` command to list events for a specific resource, or for the entire cluster.

[How to Export Kubernetes Events for Observability and Alerting](https://engineering.opsgenie.com/how-to-export-kubernetes-events-for-observability-and-alerting-a9b4a953363d)**Events events 在 `k8s.io/api/core/v1/types.go` 中进行定义**，比较重要的几个字段
1. Message: A human-readable description of the status of this operation
2. Involved Object: The object that this event is about, like Pod, Deployment, Node, etc.
3. Reason: Short, machine-understandable string, in other words: Enum
4. Source: The component reporting this event, short machine-understandable string. i.e kube-scheduler
5. Type: K8s 中 events 目前只有两种类型：“Normal” 和 “Warning”
6. Count: The number of times the event has occurred

### 从哪里来

k8s 多个组件均会产生 event， 下文以Kubelet 为例

![](/public/upload/kubernetes/kubernetes_event_object.png)

1. kubelet 日常向打日志一样 记录event，比如 `kl.recorder.Eventf(pod, v1.EventTypeWarning, events.FailedToKillPod, "error killing pod: %v", err)`，干活的主角是 EventRecorder（实现类recorderImpl）
2. Kubelet 在启动的时候会初始化一个 EventBroadcaster（连带初始化Broadcaster），新建EventRecorder，并将已经初始化的 Broadcaster 对象作为参数传给了 EventRecorder。
3. 直接或间接使用`EventBroadcaster.StartEventWatcher` 为Broadcaster 注册一个 watcher/eventhandler（也就是一个函数），eventhandler 定义了处理event 的方法，比如写到apiserver 或写到日志上
4. EventBroadcaster 和 EventRecorder 的中介是Broadcaster

    1. EventRecorder 提供event 入口，传递event到 Broadcaster。就像log4j日志中的logger 对象一样只负责将log 写入到缓冲区。
    2. Broadcaster 的作用就是接收所有的 events 并广播到 注册的watcher

5. golang 里喜欢 拿channel 当队列使用

![](/public/upload/kubernetes/kubernetes_event_sequence.png)

### 到哪里去/events 妙用

Events 量非常大，只能存在一个很短的时间，很有必要将它们export 出来持久化（尤其是到时序数据库内） 以便异常状态下的分析，以及日常监控。

![](/public/upload/kubernetes/kubernetes_event_monitor.png)

工作日开工后飙升的 Schedule 和 pod create event。

![](/public/upload/kubernetes/kubernetes_event_monitor_2.png)

[Kubernetes可观察性：全方位事件监控](https://yq.aliyun.com/articles/745567)

![](/public/upload/kubernetes/aliyun_k8s_event_center.png)

## 获取event 数据

因为Event 一般会暂存在 apiserver，因此想要获取到 k8s 的event 数据，有多种方式

1. 通过client-go 的informer 机制监听 event 数据，[opsgenie/kubernetes-event-exporter](https://github.com/opsgenie/kubernetes-event-exporter) 即使用该方式
2. 直接使用kubeClient api `kubeClient.CoreV1().Events`，[AliyunContainerService/kube-eventer](https://github.com/AliyunContainerService/kube-eventer) 即使用该方式
 

## kube-eventer

### 源码包

```
kube-eventer
    core
        types.go
    sources
        kubernetes
        factory.go
    sinks
        dingtalk
        influxdb
        factory.go
        manager.go
    eventer.go  // main入口
```

### 核心概念

![](/public/upload/kubernetes/kube_eventer.png)

概念上主要包括 事件源、事件和时间存储

```go
// kube-eventer/core/types.go
type EventBatch struct {
	// When this batch was created.
	Timestamp time.Time
	// List of events included in the batch.
	Events []*kube_api.Event
}
// A place from where the events should be scraped.
type EventSource interface {
	// This is a mutable method. Each call to this method clears the internal buffer so that
	// each event can be obtained only once.
	GetNewEvents() *EventBatch
}
type EventSink interface {
	Name() string
	// Exports data to the external storage. The function should be synchronous/blocking and finish only
	// after the given EventBatch was written. This will allow sink manager to push data only to these
	// sinks that finished writing the previous data.
	ExportEvents(*EventBatch)
	// Stops the sink at earliest convenience.
	Stop()
}
```

### event 读写

启动
```go
// kube-eventer/eventer.go
func main() {
	quitChannel := make(chan struct{}, 0)
	// sources
	sourceFactory := sources.NewSourceFactory()
	sources, err := sourceFactory.BuildAll(argSources)
	// sinks
	sinksFactory := sinks.NewSinkFactory()
    sinkList := sinksFactory.BuildAll(argSinks)
    
	sinkManager, err := sinks.NewEventSinkManager(sinkList, sinks.DefaultSinkExportEventsTimeout, sinks.DefaultSinkStopTimeout)
	// main manager
	manager, err := manager.NewManager(sources[0], sinkManager, *argFrequency)
	manager.Start()

	go startHTTPServer()
	<-quitChannel
}
```

SourceFactory.BuildAll ==> SourceFactory.Build ==> NewKubernetesSource  ==> KubernetesEventSource.watch作为生产者 通过apiServer 的event api 往KubernetesEventSource.localEventsBuffer  塞event

```go
func NewKubernetesSource(uri *url.URL) (*KubernetesEventSource, error) {
	kubeConfig, err := kubeconfig.GetKubeClientConfig(uri)
	kubeClient, err := kubeclient.NewForConfig(kubeConfig)
	eventClient := kubeClient.CoreV1().Events(kubeapi.NamespaceAll)
	result := KubernetesEventSource{
		localEventsBuffer: make(chan *kubeapi.Event, LocalEventsBufferSize),
		stopChannel:       make(chan struct{}),
		eventClient:       eventClient,
	}
	go result.watch()
	return &result, nil
}
func (this *KubernetesEventSource) watch() {
	// Outer loop, for reconnections.
	for {
		events, err := this.eventClient.List(metav1.ListOptions{})
		// Do not write old events.
		resourceVersion := events.ResourceVersion
		watcher, err := this.eventClient.Watch(
			metav1.ListOptions{
				Watch:           true,
				ResourceVersion: resourceVersion})
		watchChannel := watcher.ResultChan()
		// Inner loop, for update processing.
	inner_loop:
		for {
			select {
			case watchUpdate, ok := <-watchChannel:
				if event, ok := watchUpdate.Object.(*kubeapi.Event); ok {
                    ...
                    this.localEventsBuffer <- event:
                    ...
				}
			case <-this.stopChannel:
				klog.Infof("Event watching stopped")
				return
			}
		}
	}
}
```
realManager.Start ==> realManager.Housekeep ==> realManager.housekeep ==> KubernetesEventSource.GetNewEvents 作为消费者 从KubernetesEventSource.localEventsBuffer 取出event，触发EventSink.ExportEvents 来执行。 

```go
func (this *KubernetesEventSource) GetNewEvents() *core.EventBatch {
	startTime := time.Now()
	defer func() {
		lastEventTimestamp.Set(float64(time.Now().Unix()))
		scrapEventsDuration.Observe(float64(time.Since(startTime)) / float64(time.Millisecond))
	}()
	result := core.EventBatch{
		Timestamp: time.Now(),
		Events:    []*kubeapi.Event{},
	}
	// Get all data from the buffer.
event_loop:
	for {
		select {
		case event := <-this.localEventsBuffer:
			result.Events = append(result.Events, event)
		default:
			break event_loop
		}
	}

	totalEventsNum.Add(float64(len(result.Events)))

	return &result
}
```

## kubernetes-event-exporter

主要是 EventWatcher 和 Engine  两个角色的交互，watcher 拿到事件 通过Engine 传递给Sink 消费。

![](/public/upload/kubernetes/kubernetes_event_exporter.png)

构造watcher时，初始化了 k8s apiclient ，进而构造了event Informer. watcher 本身实现了 ResourceEventHandler（onAdd/onUpdate/onDelete）

```go
func NewEventWatcher(config *rest.Config, fn EventHandler) *EventWatcher {
	clientset := kubernetes.NewForConfigOrDie(config)
	factory := informers.NewSharedInformerFactory(clientset, 0)
	informer := factory.Core().V1().Events().Informer()

	watcher := &EventWatcher{
		informer:        informer,
		stopper:         make(chan struct{}),
		labelCache:      NewLabelCache(config),
		annotationCache: NewAnnotationCache(config),
		fn:              fn,
	}
	informer.AddEventHandler(watcher)
	return watcher
}
```