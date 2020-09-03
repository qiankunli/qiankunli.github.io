---

layout: post
title: Scheduler如何给Node打分
category: 架构
tags: Kubernetes
keywords: Scheduler Score prioritizeNodes
---

## 简介

* TOC
{:toc}

scheduler 负责给一个pod 找一个node，经过预选和优选两个阶段，优选阶段会为 通过预选的node 打一个分，然后择优录取。k8s 很多调度策略 比如nodeAffinity 本质上就是 打分实现的。

打分组件主要包括scheduler 内嵌的 ScorePlugin 以及 扩展的Scheduler extender 组件。难点

1. 给定一个pod 和node，如何计算分数
2. ScorePlugin 与 Scheduler extender 均有多个，多个score 如何算出一个总分？ 权重。

## 整体设计

对每一个node 可以返回score 1到10 。每个score plugin 为node 算一个分（有权重），scheduler extender为node 算一个分（有权重），按权重为一个node累计一个总分。

```go
// k8s.io/kubernetes/pkg/scheduler/core/generic_scheduler.go
func (g *genericScheduler) prioritizeNodes(
	ctx context.Context,
	prof *profile.Profile,
	state *framework.CycleState,
	pod *v1.Pod,
	nodes []*v1.Node,
) (framework.NodeScoreList, error) {
	// 如果既没有扩展插件， 也没有内嵌score插件，每个node score=1
	if len(g.extenders) == 0 && !prof.HasScorePlugins() {
		result := make(framework.NodeScoreList, 0, len(nodes))
		for i := range nodes {
			result = append(result, framework.NodeScore{
				Name:  nodes[i].Name,
				Score: 1,
			})
		}
		return result, nil
	}
	// Run the Score plugins. 每个插件 针对每个node 算一个评分
	scoresMap, scoreStatus := prof.RunScorePlugins(ctx, state, pod, nodes)
	// Summarize all scores.  对评分求和
	result := make(framework.NodeScoreList, 0, len(nodes))
	for i := range nodes {
		result = append(result, framework.NodeScore{Name: nodes[i].Name, Score: 0})
		for j := range scoresMap {
			result[i].Score += scoresMap[j][i].Score
		}
	}
	if len(g.extenders) != 0 && nodes != nil {
		combinedScores := make(map[string]int64, len(nodes))
		for i := range g.extenders {
			go func(extIndex int) {
				prioritizedList, weight, err := g.extenders[extIndex].Prioritize(pod, nodes)
				for i := range *prioritizedList {
					host, score := (*prioritizedList)[i].Host, (*prioritizedList)[i].Score
				    // 每个extender 插件再算一个分，并基于权重累计
					combinedScores[host] += score * weight
				}
			}(i)
		}
	    // result 内嵌评分（最大100），combinedScores extenders 评分（最大10），最大值不一致， 所以把combinedScores 扩大10倍 加入到 result 上
		for i := range result {
			// MaxExtenderPriority may diverge from the max priority used in the scheduler and defined by MaxNodeScore,
			// therefore we need to scale the score returned by extenders to the score range used by the scheduler.
			result[i].Score += combinedScores[result[i].Name] * (framework.MaxNodeScore / extenderv1.MaxExtenderPriority)
		}
	}
	return result, nil
}
```

每个node 计算出一个score之后，哪个node 的score 最高，pod 就调度到哪个node 上。

```go
// k8s.io/kubernetes/pkg/scheduler/core/generic_scheduler.go
func (g *genericScheduler) Schedule(ctx context.Context, prof *profile.Profile, state *framework.CycleState, pod *v1.Pod) (result ScheduleResult, err error) {
	...
	priorityList, err := g.prioritizeNodes(ctx, prof, state, pod, filteredNodes)
	host, err := g.selectHost(priorityList)
	return ScheduleResult{
		SuggestedHost:  host,
		EvaluatedNodes: len(filteredNodes) + len(filteredNodesStatuses),
		FeasibleNodes:  len(filteredNodes),
	}, err
}
func (g *genericScheduler) selectHost(nodeScoreList framework.NodeScoreList) (string, error) {
	if len(nodeScoreList) == 0 {
		return "", fmt.Errorf("empty priorityList")
	}
	maxScore := nodeScoreList[0].Score
	selected := nodeScoreList[0].Name
	cntOfMaxScore := 1
	for _, ns := range nodeScoreList[1:] {
		if ns.Score > maxScore {
			maxScore = ns.Score
			selected = ns.Name
			cntOfMaxScore = 1
		} else if ns.Score == maxScore {
			cntOfMaxScore++
			if rand.Intn(cntOfMaxScore) == 0 {
				// Replace the candidate with probability of 1/cntOfMaxScore
				selected = ns.Name
			}
		}
	}
	return selected, nil
}
```

各个score 插件的默认权重

```go
// k8s.io/kubernetes/pkg/scheduler/algorithmprovider/registry.go
func getDefaultConfig() *schedulerapi.Plugins {
	return &schedulerapi.Plugins{
	    ...
		Score: &schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: noderesources.BalancedAllocationName, Weight: 1},
				{Name: imagelocality.Name, Weight: 1},
				{Name: interpodaffinity.Name, Weight: 1},
				{Name: noderesources.LeastAllocatedName, Weight: 1},
				{Name: nodeaffinity.Name, Weight: 1},
				{Name: nodepreferavoidpods.Name, Weight: 10000},
				{Name: defaultpodtopologyspread.Name, Weight: 1},
				{Name: tainttoleration.Name, Weight: 1},
			},
		},
	}
}
```

每个ScorePlugin 计算的score 限定在`[1,100]`，可以看到大部分插件默认为1，个别插件nodepreferavoidpods 权重为10000（基本就是一票否决了）。

调用各个ScorePlugin，过滤非法score，加入权重计算。

```go
// k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1/framework.go
func (f *framework) RunScorePlugins(ctx context.Context, state *CycleState, pod *v1.Pod, nodes []*v1.Node) (ps PluginToNodeScores, status *Status) {
	pluginToNodeScores := make(PluginToNodeScores, len(f.scorePlugins))
	for _, pl := range f.scorePlugins {
		pluginToNodeScores[pl.Name()] = make(NodeScoreList, len(nodes))
	}
	// Run Score method for each node in parallel.
	workqueue.ParallelizeUntil(ctx, 16, len(nodes), func(index int) {
		for _, pl := range f.scorePlugins {
			nodeName := nodes[index].Name
			s, status := f.runScorePlugin(ctx, pl, state, pod, nodeName)
			pluginToNodeScores[pl.Name()][index] = NodeScore{
				Name:  nodeName,
				Score: int64(s),
			}
		}
	})
	// Run NormalizeScore method for each ScorePlugin in parallel.
	workqueue.ParallelizeUntil(ctx, 16, len(f.scorePlugins), func(index int) {
		pl := f.scorePlugins[index]
		nodeScoreList := pluginToNodeScores[pl.Name()]
		status := f.runScoreExtension(ctx, pl, state, pod, nodeScoreList)
	})

	// Apply score defaultWeights for each ScorePlugin in parallel.
	workqueue.ParallelizeUntil(ctx, 16, len(f.scorePlugins), func(index int) {
		pl := f.scorePlugins[index]
		weight := f.pluginNameToWeightMap[pl.Name()]
		nodeScoreList := pluginToNodeScores[pl.Name()]
		for i, nodeScore := range nodeScoreList {
			// return error if score plugin returns invalid score.
			if nodeScore.Score > int64(MaxNodeScore) || nodeScore.Score < int64(MinNodeScore) {
				return
			}
			nodeScoreList[i].Score = nodeScore.Score * int64(weight)
		}
	})
	return pluginToNodeScores, nil
}
```

## ScorePlugin 实现

主要是 `k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources`   noderesources

noderesources.BalancedAllocationName 和 noderesources.LeastAllocatedName 都作为score plugin 被使用了。具体实现上，BalancedAllocation 和 LeastAllocated 负责最外层接口实现和 具体score 逻辑，resourceAllocationScorer 负责通用pod node 数据获取 及通用score 逻辑。执行链路为：`子类.Score ==> resourceAllocationScorer.score ==> 子类.核心score 逻辑`

```go
type resourceAllocationScorer struct {
	Name                string
	scorer              func(requested, allocable resourceToValueMap, includeVolumes bool, requestedVolumes int, allocatableVolumes int) int64
	resourceToWeightMap resourceToWeightMap
}
type BalancedAllocation struct {
	handle framework.FrameworkHandle
	resourceAllocationScorer
}
type LeastAllocated struct {
	handle framework.FrameworkHandle
	resourceAllocationScorer
}
```

### BalancedAllocationName

```go
// k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources/resource_allocation.go
type resourceAllocationScorer struct {
	Name                string
	scorer              func(requested, allocable resourceToValueMap, includeVolumes bool, requestedVolumes int, allocatableVolumes int) int64
	resourceToWeightMap resourceToWeightMap
}
func (r *resourceAllocationScorer) score(
	pod *v1.Pod,
	nodeInfo *schedulernodeinfo.NodeInfo) (int64, *framework.Status) {
	node := nodeInfo.Node()
	requested := make(resourceToValueMap, len(r.resourceToWeightMap))
	allocatable := make(resourceToValueMap, len(r.resourceToWeightMap))
	for resource := range r.resourceToWeightMap {
		allocatable[resource], requested[resource] = calculateResourceAllocatableRequest(nodeInfo, pod, resource)
	}
	var score int64
	// Check if the pod has volumes and this could be added to scorer function for balanced resource allocation.  
	if len(pod.Spec.Volumes) >= 0 && utilfeature.DefaultFeatureGate.Enabled(features.BalanceAttachedNodeVolumes) && nodeInfo.TransientInfo != nil {
		score = r.scorer(requested, allocatable, true, nodeInfo.TransientInfo.TransNodeInfo.RequestedVolumes, nodeInfo.TransientInfo.TransNodeInfo.AllocatableVolumesCount)
	} else {
		score = r.scorer(requested, allocatable, false, 0, 0)
	}
	return score, nil
}
// k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources/balanced_allocation.go
func balancedResourceScorer(requested, allocable resourceToValueMap, includeVolumes bool, requestedVolumes int, allocatableVolumes int) int64 {
	cpuFraction := fractionOfCapacity(requested[v1.ResourceCPU], allocable[v1.ResourceCPU])
	memoryFraction := fractionOfCapacity(requested[v1.ResourceMemory], allocable[v1.ResourceMemory])
	// This to find a node which has most balanced CPU, memory and volume usage.
	if cpuFraction >= 1 || memoryFraction >= 1 {
		// if requested >= capacity, the corresponding host should never be preferred.
		return 0
	}
	if includeVolumes && utilfeature.DefaultFeatureGate.Enabled(features.BalanceAttachedNodeVolumes) && allocatableVolumes > 0 {
		...
		return int64((1 - variance) * float64(framework.MaxNodeScore))
	}
	// Upper and lower boundary of difference between cpuFraction and memoryFraction are -1 and 1
	// respectively. Multiplying the absolute value of the difference by 10 scales the value to
	// 0-10 with 0 representing well balanced allocation and 10 poorly balanced. Subtracting it from
	// 10 leads to the score which also scales from 0 to 10 while 10 representing well balanced.
	diff := math.Abs(cpuFraction - memoryFraction)
	return int64((1 - diff) * float64(framework.MaxNodeScore))
}
```


BalancedAllocationName 首先会计算 cpu 、men 以及可能volume reqeust 占有node  allocable 的比例Fraction

```go
diff := math.Abs(cpuFraction - memoryFraction)
return int64((1 - diff) * float64(framework.MaxNodeScore))
```
按照上市逻辑来说

||cpuFraction|memoryFraction|score|
|---|---|---|---|
|node1|10%|20%|(1-0.1)*100=90|
|node2|20%|10%|(1-0.1)*100=90|
|node2|20%|40%|(1-0.2)*100=80|
|node2|20%|50%|(1-0.3)*100=70|

从上面可以看到 BalancedAllocationName 是从单个node 的角度出发的，尽量让一个node 上已经分配的 cpu 和mem 百分比是平衡的。减少出现比如 cpu 很富余但内存已经不够的情况。BalancedAllocation is a score plugin that calculates the difference between the cpu and memory fraction of capacity, and prioritizes the host based on how close the two metrics are to each other. 

### LeastAllocatedName

cpu 和内存分别算一个分数，加权求和。node剩余资源越富余，score 越高。也就是负载越低的node 分数高。

```go
// k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources/least_allocated.go
var defaultRequestedRatioResources = resourceToWeightMap{v1.ResourceMemory: 1, v1.ResourceCPU: 1}
func leastResourceScorer(requested, allocable resourceToValueMap, includeVolumes bool, requestedVolumes int, allocatableVolumes int) int64 {
	var nodeScore, weightSum int64
	for resource, weight := range defaultRequestedRatioResources {
		resourceScore := leastRequestedScore(requested[resource], allocable[resource])
		nodeScore += resourceScore * weight
		weightSum += weight
	}
	return nodeScore / weightSum
}
func leastRequestedScore(requested, capacity int64) int64 {
	if capacity == 0 {
		return 0
	}
	if requested > capacity {
		return 0
	}
	return ((capacity - requested) * int64(framework.MaxNodeScore)) / capacity
}
```


## 如何观察节点计算的score 

scheduler 在score结束会记录日志

```go
// k8s.io/kubernetes/pkg/scheduler/core/generic_scheduler.go
func (g *genericScheduler) prioritizeNodes(
	ctx context.Context,
	prof *profile.Profile,
	state *framework.CycleState,
	pod *v1.Pod,
	nodes []*v1.Node,
) (framework.NodeScoreList, error) {
    ...
    if klog.V(10) {
		for i := range result {
			klog.Infof("Host %s => Score %d", result[i].Name, result[i].Score)
		}
	}
}
```

假设k8s 集群存在 192.168.60.237 和 192.168.60.238 两个slave 节点，在192.168.60.89 运行一个scheduler extender，当把kube-scheduler 启动参数的日志level 调整为10 时（值越小估计级别越高，输出日志越少），apply 一个 nginx example， `journalctl -u kube-scheduler | grep -i score`可以看到代score 关键字的日志（日志部分去掉了前缀）

```
nginx-deployment-85ff79dd56-972cg -> 192.168.60.238: BalancedResourceAllocation, map of allocatable resources map[cpu:47800 memory:66054406144], map of requested resources map[cpu:7850 memory:6786383872] ,score 93,
nginx-deployment-85ff79dd56-972cg -> 192.168.60.238: LeastResourceAllocation, map of allocatable resources map[cpu:47800 memory:66054406144], map of requested resources map[cpu:7850 memory:6786383872] ,score 86,
nginx-deployment-85ff79dd56-972cg -> 192.168.60.237: BalancedResourceAllocation, map of allocatable resources map[cpu:47800 memory:66054406144], map of requested resources map[cpu:6950 memory:7627341824] ,score 97,
nginx-deployment-85ff79dd56-972cg -> 192.168.60.237: LeastResourceAllocation, map of allocatable resources map[cpu:47800 memory:66054406144], map of requested resources map[cpu:6950 memory:7627341824] ,score 86,
nginx-deployment-85ff79dd56-972cg -> 192.168.60.238: SelectorSpreadPriority, Score: (100)
nginx-deployment-85ff79dd56-972cg -> 192.168.60.237: SelectorSpreadPriority, Score: (100)
nginx-deployment-85ff79dd56-972cg_default -> 192.168.60.238: http://192.168.60.89:8889/, Score: (2)
nginx-deployment-85ff79dd56-972cg_default -> 192.168.60.237: http://192.168.60.89:8889/, Score: (5)
Host 192.168.60.238 => Score 1000404
Host 192.168.60.237 => Score 1000438
```
metric ：可以看 `http://$scheduler_ip:10251/metrics`  的metric。 但没有直接找到相关的 metric ，或者信息不充分。