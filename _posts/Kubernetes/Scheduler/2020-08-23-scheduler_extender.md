---

layout: post
title: Scheduler扩展
category: 架构
tags: Kubernetes
keywords: Scheduler Extender
---

## 简介

* TOC
{:toc}

[Scheduler extender](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/scheduling/scheduler_extender.md) 扩展Scheduler 的三种方式

1. by adding these rules to the scheduler and recompiling, [described here](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-scheduling/scheduler.md) 改源码
2. implementing your own scheduler process that runs instead of, or alongside of, the standard Kubernetes scheduler,  另写一个scheduler。 多个Scheduler 可以共存，pod `spec.schedulerName` 来指定 pod 由哪个Scheduler 调度
3. implementing a "scheduler extender" process that the standard Kubernetes scheduler calls out to as a final pass when making scheduling decisions. 给默认Scheduler 做参谋长

This approach is needed for use cases where scheduling decisions need to be made on resources not directly managed by the standard Kubernetes scheduler. The extender helps make scheduling decisions based on such resources. (Note that the three approaches are not mutually exclusive.) 第三种方案一般用在 调度决策依赖于 非默认支持的资源的场景

[一篇读懂Kubernetes Scheduler扩展功能](https://mp.weixin.qq.com/s/e4VfnUpEOmVxx_zwXOMCPg)

## scheduler 调用 SchedulerExtender 

### 调用过程

![](/public/upload/kubernetes/scheduler_extender.png)

[Create a custom Kubernetes scheduler](https://developer.ibm.com/technologies/containers/articles/creating-a-custom-kube-scheduler/#) 以二进制运行为例，kube-scheduler 启动命令一般为  `kube-scheduler --address=0.0.0.0 --kubeconfig=/etc/kubernetes/kube-scheduler.kubeconfig --leader-elect=true`。kube-scheduler 启动时可以指定 --config 参数，对应一个yaml 配置文件，带有 scheduler-extender 示例如下

```yaml
apiVersion: kubescheduler.config.k8s.io/v1alpha1
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "/etc/kubernetes/kube-scheduler.kubeconfig"       # kubeconfig 文件
algorithmSource:
  policy:
    file:
      path: "/etc/kubernetes/scheduler-extender-policy.json"    # 指定自定义调度策略文件
```
scheduler-extender 策略文件示例如下 

```json
{
    "kind" : "Policy",
    "apiVersion" : "v1",
    "extenders" : [{
        "urlPrefix": "http://localhost:8888/",
        "filterVerb": "filter",
        "prioritizeVerb": "prioritize",
        "weight": 1,
        "enableHttps": false
    }]
}
```

policy文件定义了一个 HTTP 的扩展程序服务，该服务运行在 `127.0.0.1:8888` 下面，并且已经将该策略注册到了默认的调度器中，这样在过滤和打分阶段结束后，可以将结果分别传递给该扩展程序的端点 `<urlPrefix>/<filterVerb>=http://localhost:8888/filter` 和 `<urlPrefix>/<prioritizeVerb>=http://localhost:8888/prioritize` 做进一步过滤和打分。

### 源码分析

```go
// Scheduler 的核心组件genericScheduler 聚合了 SchedulerExtender
type genericScheduler struct {
	cache                    internalcache.Cache
	schedulingQueue          internalqueue.SchedulingQueue
	extenders                []SchedulerExtender
	...
}
// k8s.io/kubernetes/pkg/scheduler/core/extender.go
type SchedulerExtender interface {
    Name() string
	Filter(pod *v1.Pod, nodes []*v1.Node) (filteredNodes []*v1.Node, failedNodesMap extenderv1.FailedNodesMap, err error)
	Prioritize(pod *v1.Pod, nodes []*v1.Node) (hostPriorities *extenderv1.HostPriorityList, weight int64, err error)
	Bind(binding *v1.Binding) error
    ...
}
// SchedulerExtender 默认实现 HTTPExtender
type HTTPExtender struct {
	extenderURL      string
	preemptVerb      string
	filterVerb       string
	prioritizeVerb   string
	bindVerb         string
	weight           int64
	client           *http.Client
	ignorable        bool
}
```
**HTTPExtender本质上是一个 webhook**， SchedulerExtender.Filter 会在 genericScheduler.Schedule 时被执行 

```go
func (g *genericScheduler) Schedule(ctx context.Context, prof *profile.Profile, state *framework.CycleState, pod *v1.Pod) (result ScheduleResult, err error) {
    ...
    filteredNodes, filteredNodesStatuses, err := g.findNodesThatFitPod(ctx, prof, state, pod)
    ...
}
func (g *genericScheduler) findNodesThatFitPod(ctx context.Context, prof *profile.Profile, state *framework.CycleState, pod *v1.Pod) ([]*v1.Node, framework.NodeToStatusMap, error) {
	filteredNodesStatuses := make(framework.NodeToStatusMap)
	filtered, err := g.findNodesThatPassFilters(ctx, prof, state, pod, filteredNodesStatuses)
	filtered, err = g.findNodesThatPassExtenders(pod, filtered, filteredNodesStatuses)
	return filtered, filteredNodesStatuses, nil
}
func (g *genericScheduler) findNodesThatPassExtenders(pod *v1.Pod, filtered []*v1.Node, statuses framework.NodeToStatusMap) ([]*v1.Node, error) {
	for _, extender := range g.extenders {
		...
		filteredList, failedMap, err := extender.Filter(pod, filtered)
		...
	}
	return filtered, nil
}
```

## 示例实现

SchedulerExtender 首先是一个 http server，为了不影响scheduler 的调度， 应确保 http 接口响应时间不要过长。

```go
func main() {
    router := httprouter.New()
    router.GET("/", Index)
    router.POST("/filter", Filter)
    router.POST("/prioritize", Prioritize)
    log.Fatal(http.ListenAndServe(":8888", router))
}
```
以Filter 逻辑为例，Filter 方法入参和出餐 被限定为 schedulerapi.ExtenderArgs 和 schedulerapi.ExtenderFilterResult
```go
func filter(args schedulerapi.ExtenderArgs) *schedulerapi.ExtenderFilterResult {
    var filteredNodes []v1.Node
    failedNodes := make(schedulerapi.FailedNodesMap)
    pod := args.Pod
    for _, node := range args.Nodes.Items {
        fits, failReasons, _ := podFitsOnNode(pod, node)
        if fits {
            filteredNodes = append(filteredNodes, node)
        } else {
            failedNodes[node.Name] = strings.Join(failReasons, ",")
        }
    }
    result := schedulerapi.ExtenderFilterResult{
        Nodes: &v1.NodeList{Items: filteredNodes,},
        FailedNodes: failedNodes,
        Error:       "",
    }
    return &result
}
func podFitsOnNode(pod *v1.Pod, node v1.Node) (bool, []string, error) {
    fits := true
    failReasons := []string{}
    // 做一下逻辑判断
    return fits, failReasons, nil
}
```


