---

layout: post
title: openkruise 学习
category: 架构
tags: Kubernetes
keywords: openkruise
---

## 简介（未完成）

* TOC
{:toc}

[openkruise](http://openkruise.io/) 面向自动化场景的 Kubernetes workload扩展controller，它是一组controller，可在应用程序工作负载管理上扩展和补充Kubernetes核心控制器。

## 源码包

```
github.com/openkruise/kruise
    /apis       // kubebuilder自动生成
    /charts
    /config     // kubebuilder自动生成， 部署crd 、controller 等的yaml 文件、 controller 运行所需的 rbac 权限等
    /docs
    /pkg
        /controller
    /main.go
```

## CloneSet

CloneSet 控制器提供了高效管理无状态应用的能力，它可以对标原生的 Deployment，但 CloneSet 提供了很多增强功能。 一个简单的 CloneSet yaml 文件如下：

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

`spec.template` 中定义了当前 CloneSet 中最新的 Pod 模板。 控制器会为每次更新过的 `spec.template` 计算一个 revision hash 值

### Reconcile 逻辑

在控制器模型、 informer 机制固化下来后，不同Controller 之间的逻辑差异便只剩下 Reconcile 了

```go
func (r *ReconcileCloneSet) Reconcile(request reconcile.Request) (reconcile.Result, error) {
	return r.reconcileFunc(request)
}
// 获取CloneSet desire state， CloneSet 对应的实际的Pod 数据，syncCloneSet，然后更新CloneSet 状态
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
	// Refresh update expectations
	for _, pod := range filteredPods {
		updateExpectations.ObserveUpdated(request.String(), updateRevision.Name, pod)
	}
	newStatus := appsv1alpha1.CloneSetStatus{
		ObservedGeneration: instance.Generation,
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
	if err = r.truncatePodsToDelete(instance, filteredPods); err != nil {...}
	if err = r.truncateHistory(instance, filteredPods, revisions, currentRevision, updateRevision); err != nil {...}
	return reconcile.Result{RequeueAfter: delayDuration}, syncErr
}
// syncCloneSet 决定scale pod 或更新 pod
func (r *ReconcileCloneSet) syncCloneSet(
	instance *appsv1alpha1.CloneSet, newStatus *appsv1alpha1.CloneSetStatus,
	currentRevision, updateRevision *apps.ControllerRevision, revisions []*apps.ControllerRevision,
	filteredPods []*v1.Pod, filteredPVCs []*v1.PersistentVolumeClaim,
) (time.Duration, error) {
	// get the current and update revisions of the set.
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

scale 逻辑：对比 期待和 实际的pod 数量，决定create 或 delete pod

```go
func (r *realControl) Manage(
	currentCS, updateCS *appsv1alpha1.CloneSet,
	currentRevision, updateRevision string,
	pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim,
) (bool, error) {
	controllerKey := clonesetutils.GetControllerKey(updateCS)
	coreControl := clonesetcore.New(updateCS)
	if podsToDelete := getPodsToDelete(updateCS, pods); len(podsToDelete) > 0 {
		return r.deletePods(updateCS, podsToDelete, pvcs)
	}
	updatedPods, notUpdatedPods := clonesetutils.SplitPodsByRevision(pods, updateRevision)
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
		return r.createPods(expectedCreations, expectedCurrentCreations,
			currentCS, updateCS, currentRevision, updateRevision, availableIDs.List(), existingPVCNames)
	} else if diff > 0 {
		podsToDelete := choosePodsToDelete(diff, currentRevDiff, notUpdatedPods, updatedPods)
		return r.deletePods(updateCS, podsToDelete, pvcs)
	}
	return false, nil
}
```

update逻辑：涉及到原地升级逻辑，待学习

```go
func (c *realControl) Manage(cs *appsv1alpha1.CloneSet,
	updateRevision *apps.ControllerRevision, revisions []*apps.ControllerRevision,
	pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim,
) (time.Duration, error) {

	requeueDuration := requeueduration.Duration{}
	coreControl := clonesetcore.New(cs)

	if cs.Spec.UpdateStrategy.Paused {
		return requeueDuration.Get(), nil
	}

	// 1. find currently updated and not-ready count and all pods waiting to update
	var waitUpdateIndexes []int
	for i := range pods {
		if coreControl.IsPodUpdatePaused(pods[i]) {
			continue
		}

		if res := c.inplaceControl.Refresh(pods[i], coreControl.GetUpdateOptions()); res.RefreshErr != nil {
			klog.Errorf("CloneSet %s/%s failed to update pod %s condition for inplace: %v",
				cs.Namespace, cs.Name, pods[i].Name, res.RefreshErr)
			return requeueDuration.Get(), res.RefreshErr
		} else if res.DelayDuration > 0 {
			requeueDuration.Update(res.DelayDuration)
		}

		if clonesetutils.GetPodRevision(pods[i]) != updateRevision.Name {
			waitUpdateIndexes = append(waitUpdateIndexes, i)
		}
	}

	// 2. sort all pods waiting to update
	waitUpdateIndexes = sortUpdateIndexes(coreControl, cs.Spec.UpdateStrategy, pods, waitUpdateIndexes)
	// 3. calculate max count of pods can update
	needToUpdateCount := calculateUpdateCount(coreControl, cs.Spec.UpdateStrategy, cs.Spec.MinReadySeconds, int(*cs.Spec.Replicas), waitUpdateIndexes, pods)
	if needToUpdateCount < len(waitUpdateIndexes) {
		waitUpdateIndexes = waitUpdateIndexes[:needToUpdateCount]
	}
	// 4. update pods
	for _, idx := range waitUpdateIndexes {
		pod := pods[idx]
		if duration, err := c.updatePod(cs, coreControl, updateRevision, revisions, pod, pvcs); err != nil {
			return requeueDuration.Get(), err
		} else if duration > 0 {
			requeueDuration.Update(duration)
		}
	}
	return requeueDuration.Get(), nil
}
```