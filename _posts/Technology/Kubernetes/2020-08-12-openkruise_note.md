---

layout: post
title: openkruise 学习
category: 架构
tags: Kubernetes
keywords: openkruise
---

## 简介

* TOC
{:toc}

[openkruise](http://openkruise.io/) 面向自动化场景的 Kubernetes workload扩展controller，它是一组controller，可在应用程序工作负载管理上扩展和补充Kubernetes核心控制器。

[Kruise 控制器分类指引](http://openkruise.io/zh-cn/blog/blog1.html)Controller 命名惯例
1. Set 后缀：这类 controller 会直接操作和管理 Pod，比如 CloneSet, ReplicaSet, SidecarSet 等。它们提供了 Pod 维度的多种部署、发布策略。
2. Deployment 后缀：这类 controller 不会直接地操作 Pod，它们通过操作一个或多个 Set 类型的 workload 来间接管理 Pod，比如 Deployment 管理 ReplicaSet 来提供一些额外的滚动策略，以及 UnitedDeployment 支持管理多个 StatefulSet/AdvancedStatefulSet 来将应用部署到不同的可用区。
3. Job 后缀：这类 controller 主要管理短期执行的任务，比如 BroadcastJob 支持将任务类型的 Pod 分发到集群中所有 Node 上。

## 源码包

```
github.com/openkruise/kruise
    /apis       // kubebuilder自动生成
    /charts     // helm 安装相关
    /config     // kubebuilder自动生成， 部署crd 、controller 等的yaml 文件、 controller 运行所需的 rbac 权限等
    /docs
    /pkg
        /controller
    /main.go
```

## CloneSet

[CloneSet](http://openkruise.io/zh-cn/docs/cloneset.html) 控制器提供了高效管理无状态应用的能力，一个简单的 CloneSet yaml 文件如下：

```yaml
apiVersion: apps.kruise.io/v1alpha1
kind: CloneSet
metadata:
  labels:
    app: sample
  name: sample
spec:
  replicas: 5
  selector:
    matchLabels:
      app: sample
  template:
    metadata:
      labels:
        app: sample
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
```
`spec.template` 中定义了当前 CloneSet 中最新的 Pod 模板。 控制器会为每次更新过的 `spec.template` 计算一个 revision hash 值。在运行过程中，还会额外 为cloneset 管理的pod 加上 label=controller-revision-hash  标记pod 所属的revision。

CloneSet可以对标原生的 Deployment，但 CloneSet 提供了很多增强功能，比如扩缩容时删除特定pod，**尤其提供了丰富的升级策略**。

```
spec:
    // 升级功能
    updateStrategy:
        type:               // 升级方式，默认为重建升级ReCreate，支持尽可能原地升级InPlaceIfPossible 和 只能原地升级InPlaceOnly
        inPlaceUpdateStrategy:  // 原地升级策略，比如graceful period  等
        partition:          // Partition 的语义是 保留旧版本 Pod 的数量，默认为 0。 如果在发布过程中设置了 partition，则控制器只会将 (replicas - partition) 数量的 Pod 更新到最新版本。
        MaxUnavailable:     // 最大不可用的Pod数量，是一个绝对值或百分比
        MaxSurge:           // 最大弹性数量，即最多能扩出来超过 replicas 的 Pod 数量，是一个绝对值或百分比。
        priorityStrategy:   // 优先级策略，升级顺序相关
            ...
        scatterStrategy:    // 打散策略，升级顺序相关
        paused:             // 为true时发布暂停
        PreUpdate:          // 升级前钩子函数
        PostUpdate:         // 升级后钩子函数
    // 扩缩容功能
    scaleStrategy:
        podsToDelete:       // 允许用户在缩小 replicas 数量的同时，指定想要删除的 Pod 名字
```

### Reconcile 逻辑

在kubebuilder 把Controller  控制器模型 的代码 都自动生成之后，不同Controller 之间的逻辑差异便只剩下 Reconcile 了

```go
// kruise/pkg/controller/cloneset/cloneset_controller.go
func (r *ReconcileCloneSet) Reconcile(request reconcile.Request) (reconcile.Result, error) {
    // ReconcileCloneSet.reconcileFunc = ReconcileCloneSet.doReconcile
	return r.reconcileFunc(request)
}
// 获取CloneSet desire state=instance， CloneSet 对应的实际的Pod 数据=filteredPods，syncCloneSet，然后更新CloneSet status
func (r *ReconcileCloneSet) doReconcile(request reconcile.Request) (res reconcile.Result, retErr error) {
	// Fetch the CloneSet instance
	instance := &appsv1alpha1.CloneSet{}
	err := r.Get(context.TODO(), request.NamespacedName, instance)
	selector, err := metav1.LabelSelectorAsSelector(instance.Spec.Selector)
	// list all active Pods and PVCs belongs to cs
	filteredPods, filteredPVCs, err := r.getOwnedResource(instance)
	// list all revisions and sort them
	revisions, err := r.controllerHistory.ListControllerRevisions(instance, selector)
	history.SortControllerRevisions(revisions)
	// get the current, and update revisions
    currentRevision, updateRevision, collisionCount, err := r.getActiveRevisions(instance, revisions, clonesetutils.GetPodsRevisions(filteredPods))
	newStatus := appsv1alpha1.CloneSetStatus{
        ObservedGeneration: instance.Generation,
        //  控制器会为每次更新过的 spec.template 计算一个 revision hash 值并上报到 CloneSet status 中
		UpdateRevision:     updateRevision.Name,    
		CollisionCount:     new(int32),
		LabelSelector:      selector.String(),
	}
	*newStatus.CollisionCount = collisionCount
	// scale and update pods
	delayDuration, syncErr := r.syncCloneSet(instance, &newStatus, currentRevision, updateRevision, revisions, filteredPods, filteredPVCs)
	// update new status
	if err = r.statusUpdater.UpdateCloneSetStatus(instance, &newStatus, filteredPods); err != nil {
		return reconcile.Result{}, err
    }
    // 对于已经被删除的 Pod，控制器会自动从 podsToDelete 列表中清理掉。
	if err = r.truncatePodsToDelete(instance, filteredPods); err != nil {...}
	if err = r.truncateHistory(instance, filteredPods, revisions, currentRevision, updateRevision); err != nil {...}
	return reconcile.Result{RequeueAfter: delayDuration}, syncErr
}
```
CloneSet status 中的字段说明：

1. status.replicas: Pod 总数
2. status.readyReplicas: ready Pod 数量
3. status.availableReplicas: ready and available Pod 数量 (满足 minReadySeconds)
4. status.updateRevision: 最新版本的 revision hash 值
5. status.updatedReplicas: 最新版本的 Pod 数量
6. status.updatedReadyReplicas: 最新版本的 ready Pod 数量

```go
// kruise/pkg/controller/cloneset/cloneset_controller.go
func (r *ReconcileCloneSet) syncCloneSet(
	instance *appsv1alpha1.CloneSet, newStatus *appsv1alpha1.CloneSetStatus,
	currentRevision, updateRevision *apps.ControllerRevision, revisions []*apps.ControllerRevision,
	filteredPods []*v1.Pod, filteredPVCs []*v1.PersistentVolumeClaim,
) (time.Duration, error) {
	// get the current and update revisions of the set. 获取当下 CloneSet(currentSet) 和 目标CloneSet(updateSet) 的对象表示
	currentSet, err := r.revisionControl.ApplyRevision(instance, currentRevision)
    updateSet, err := r.revisionControl.ApplyRevision(instance, updateRevision)
	scaling, podsScaleErr = r.scaleControl.Manage(currentSet, updateSet, currentRevision.Name, updateRevision.Name, filteredPods, filteredPVCs)
	if scaling {
		return delayDuration, podsScaleErr
	}
	delayDuration, podsUpdateErr = r.updateControl.Manage(updateSet, updateRevision, revisions, filteredPods, filteredPVCs)
	return delayDuration, err
}
```
syncCloneSet  根据 cloneSet 期望状态（ 由replicas 以及updateStrategy描述 ）以及pod的真实状态，**scale  负责 新旧Revision pod 的数量符合replicas/partition/MaxSurge/maxUnavailable 要求（删除或创建特定Revision的pod），update方法 负责需要原地升级的pod（如果有的话）**。在运行过程中，经常updatePod 或者 update crd 自身的数据，将一些状态数据持久化 以供下次 Reconcile 使用。graceful period  等策略的存在也决定了 Reconcile 工作不会一次就完成。

![](/public/upload/kubernetes/cloneset_reconcile.png)


### scale：确保新旧Revision pod 的数量符合要求

如果发布的时候设置了 maxSurge，控制器会先多扩出来 maxSurge 数量的 Pod（此时 Pod 总数为 (replicas+maxSurge))，然后再开始发布存量的 Pod。 然后，当新版本 Pod 数量已经满足 partition 要求之后，控制器会再把多余的 maxSurge 数量的 Pod 删除掉，保证最终的 Pod 数量符合 replicas。此外，maxSurge 还受 升级方式（type）的影响：maxSurge 不允许配合 InPlaceOnly 更新模式使用（可以认为此时maxSurge=0？）。 另外，如果是与 InPlaceIfPossible 策略配合使用，控制器会先扩出来 maxSurge 数量的 Pod，再对存量 Pod 做原地升级。

```go
// kruise/pkg/controller/cloneset/scale/cloneset_scale.go
func (r *realControl) Manage(
	currentCS, updateCS *appsv1alpha1.CloneSet,
	currentRevision, updateRevision string,
	pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim,
) (bool, error) {
	// 删除 podsToDelete 中指定的pod
	if podsToDelete := getPodsToDelete(updateCS, pods); len(podsToDelete) > 0 {
		return r.deletePods(updateCS, podsToDelete, pvcs)
    }
    // 符合 updateRevision 的pod 为 updatedPods ，不符合的为notUpdatedPods
    updatedPods, notUpdatedPods := clonesetutils.SplitPodsByRevision(pods, updateRevision)
    // 一个CloneSet 最多允许(replicas + MaxSurge)个pod 存在，如果实际pod 小于这个数量(diff<0)则需要创建pod，否则(diff>0) 删除pod
	diff, currentRevDiff := calculateDiffs(updateCS, updateRevision == currentRevision, len(pods), len(notUpdatedPods))
	if diff < 0 {
		// total number of this creation
		expectedCreations := diff * -1
		// lack number of current version
		expectedCurrentCreations := 0
		if currentRevDiff < 0 {
			expectedCurrentCreations = currentRevDiff * -1
		}
		// available instance-id come from free pvc
        availableIDs := getOrGenAvailableIDs(expectedCreations, pods, pvcs)
        // pod 数量不足，就要创建expectedCreations 个pod，创建时需指明 currentRevision和updateRevision 的pod 分别创建几个
		return r.createPods(expectedCreations, expectedCurrentCreations,
			currentCS, updateCS, currentRevision, updateRevision, availableIDs.List(), existingPVCNames)
	} else if diff > 0 {
        // pod 数量较多 则选择 多余的 新老Revision pod 删除
		podsToDelete := choosePodsToDelete(diff, currentRevDiff, notUpdatedPods, updatedPods)
		return r.deletePods(updateCS, podsToDelete, pvcs)
    }
    // 如果diff = 0 则 重建升级逻辑什么都不做
	return false, nil
}
```

### 原地升级pod

当一个 Pod 被原地升级时，控制器会先利用 readinessGates 把 Pod status 中修改为 not-ready 状态，然后再更新 Pod spec 中的 image 字段来触发 Kubelet 重建对应的容器。 不过这样可能存在的一个风险：有时候 Kubelet 重建容器太快，还没等到其他控制器如 endpoints-controller 感知到 Pod not-ready，可能会导致流量受损。因此又在原地升级中提供了 graceful period 选项，作为优雅原地升级的策略。用户如果配置了 gracePeriodSeconds 这个字段，控制器在原地升级的过程中会先把 Pod status 改为 not-ready，然后等一段时间（gracePeriodSeconds），最后再去修改 Pod spec 中的镜像版本。 这样，就为 endpoints-controller 这些控制器留出了充足的时间来将 Pod 从 endpoints 端点列表中去除。

```go
// 处理 升级间隔，计算真正需要 更新的pod
// kruise/pkg/controller/cloneset/update/cloneset_update.go
func (c *realControl) Manage(cs *appsv1alpha1.CloneSet,
	updateRevision *apps.ControllerRevision, revisions []*apps.ControllerRevision,
	pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim,
) (time.Duration, error) {
	// 1. find currently updated and not-ready count and all pods waiting to update
	var waitUpdateIndexes []int
	for i := range pods {
        // 支持 在pod 上设置一些注解，控制升级间隔，比如inplace-update-grace，如果距离上一次pod 的grace period 还未到，则先放弃本次升级
        // 如果pod 的revision 不是最新的revision ，则加入到waitUpdateIndexes
		if clonesetutils.GetPodRevision(pods[i]) != updateRevision.Name {
			waitUpdateIndexes = append(waitUpdateIndexes, i)
		}
	}
	// 2. sort all pods waiting to update
	// 3. 根据 replicas/partition/MaxSurge/maxUnavailable 以及pod Status（比如not ready）等 calculate max count of pods can update
	needToUpdateCount := calculateUpdateCount(coreControl, cs.Spec.UpdateStrategy, cs.Spec.MinReadySeconds, int(*cs.Spec.Replicas), waitUpdateIndexes, pods)
	if needToUpdateCount < len(waitUpdateIndexes) {
		waitUpdateIndexes = waitUpdateIndexes[:needToUpdateCount]
	}
	// 4. update pods
	for _, idx := range waitUpdateIndexes {
		pod := pods[idx]
		if duration, err := c.updatePod(cs, coreControl, updateRevision, revisions, pod, pvcs); err != nil {...}
	}
	return requeueDuration.Get(), nil
}
// 根据CloneSet 升级策略，执行原地升级逻辑
func (c *realControl) updatePod(cs *appsv1alpha1.CloneSet, coreControl clonesetcore.Control,
	updateRevision *apps.ControllerRevision, revisions []*apps.ControllerRevision,
	pod *v1.Pod, pvcs []*v1.PersistentVolumeClaim,
) (time.Duration, error) {
    // 如果clone set 升级策略为 （尽量）原地升级，则进入原地升级流程
	if cs.Spec.UpdateStrategy.Type == appsv1alpha1.InPlaceIfPossibleCloneSetUpdateStrategyType ||
		cs.Spec.UpdateStrategy.Type == appsv1alpha1.InPlaceOnlyCloneSetUpdateStrategyType {
		var oldRevision *apps.ControllerRevision
		for _, r := range revisions {
			if r.Name == clonesetutils.GetPodRevision(pod) {
				oldRevision = r
				break
			}
		}
		res := c.inplaceControl.Update(pod, oldRevision, updateRevision, coreControl.GetUpdateOptions())
        ...
        return ...
    }
	// handle pvc
	return 0, nil
}
```
计算待更新pod 的spec ,condition,container status 等数据， 加上revision label, inplace-update-grace annotation ，最终使用k8s api 更新pod 到k8s cluster

```go
// kruise/pkg/util/inplaceupdate/inplace_utils.go
func (c *realControl) Update(pod *v1.Pod, oldRevision, newRevision *apps.ControllerRevision, opts *UpdateOptions) UpdateResult {
	// 1. calculate inplace update spec 
	var spec *UpdateSpec
	if opts == nil || opts.CustomizeSpecCalculate == nil {
        //  只对更新了 spec.containers[x].image 的pod 进行原地升级
		spec = calculateInPlaceUpdateSpec(oldRevision, newRevision)
	}
	if opts != nil && opts.GracePeriodSeconds > 0 {
		spec.GraceSeconds = opts.GracePeriodSeconds
	}
	// 2. update condition for pod with readiness-gate
	if containsReadinessGate(pod) {
		newCondition := v1.PodCondition{
			Type:               appsv1alpha1.InPlaceUpdateReady,
			LastTransitionTime: c.now(),
			Status:             v1.ConditionFalse,
			Reason:             "StartInPlaceUpdate",
		}
		if err := c.updateCondition(pod, newCondition); err != nil {
			return UpdateResult{InPlaceUpdate: true, UpdateErr: err}
		}
	}
	// 3. update container images
	if err := c.updatePodInPlace(pod, spec, opts); err != nil {
		return UpdateResult{InPlaceUpdate: true, UpdateErr: err}
	}
    ...
}
func (c *realControl) updatePodInPlace(pod *v1.Pod, spec *UpdateSpec, opts *UpdateOptions) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		clone, err := c.adp.getPod(pod.Namespace, pod.Name)
		// update new revision   给pod 加上revision label
		if c.revisionKey != "" {
			clone.Labels[c.revisionKey] = spec.Revision
		}
		// record old containerStatuses
		inPlaceUpdateStateJSON, _ := json.Marshal(inPlaceUpdateState)
		clone.Annotations[appsv1alpha1.InPlaceUpdateStateKey] = string(inPlaceUpdateStateJSON)
		if spec.GraceSeconds <= 0 {
			if clone, err = patchUpdateSpecToPod(clone, spec, opts); err != nil {
				return err
			}
			delete(clone.Annotations, appsv1alpha1.InPlaceUpdateGraceKey)
		} else {
			inPlaceUpdateSpecJSON, _ := json.Marshal(spec)
			clone.Annotations[appsv1alpha1.InPlaceUpdateGraceKey] = string(inPlaceUpdateSpecJSON)
        }
        // 使用k8s api 更新pod 
		return c.adp.updatePod(clone)
	})
}
```
