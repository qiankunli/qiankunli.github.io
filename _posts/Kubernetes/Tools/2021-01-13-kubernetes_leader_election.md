---

layout: post
title: 基于Kubernetes选主及应用
category: 架构
tags: Kubernetes
keywords:  Kubernetes event

---

## 简介

* TOC
{:toc}

## 选主原理

代码结构
```
k8s.io/client-go/tools/leaderelection
    /resourcelock
        /configmaplock.go
        /endpointslock.go
        /interface.go  // 定义了锁的操作接口
        /leaselock.go
    /leaderelection.go
    /metrics.go
```

![](/public/upload/kubernetes/leader_election.png)

### 乐观锁

[K8S 中 scheduler 组件的选主逻辑](http://www.xuyasong.com/?p=2037)存在形式：configmap/endpoint 的annotation 上，key = `control-plane.alpha.kubernetes.io/leader`， 值对应了 LeaderElectionRecord struct，记录了当前leader 的Identity 以及renewTime

```yml
apiVersion: v1
kind: Endpoints
metadata:
  annotations:
    control-plane.alpha.kubernetes.io/leader: '{"holderIdentity":"instance-o24xykos-3_1ad55d32-2abe-49f7-9d68-33ec5eadb906","leaseDurationSeconds":15,"acquireTime":"2020-04-23T06:45:07Z","renewTime":"2020-04-25T07:55:58Z","leaderTransitions":1}'
  creationTimestamp: "2020-04-22T12:05:29Z"
  name: kube-scheduler
  namespace: kube-system
  resourceVersion: "467853"
  selfLink: /api/v1/namespaces/kube-system/endpoints/kube-scheduler
  uid: f3535807-0575-483f-8471-f8d4fd9eeac6
```
代码体现
```go
// k8s.io/client-go/tools/leaderelection/resourcelock/interface.go
type Interface interface {
	// Get returns the LeaderElectionRecord
	Get() (*LeaderElectionRecord, error)
	// Create attempts to create a LeaderElectionRecord
	Create(ler LeaderElectionRecord) error
	// Update will update and existing LeaderElectionRecord
    Update(ler LeaderElectionRecord) error
    ...
}
```
kubernetes 的 update 是原子的、安全的：Kubernetes 通过定义资源版本字段实现了乐观并发控制，资源版本 (ResourceVersion)字段包含在 Kubernetes 对象的元数据 (Metadata)中。这个字符串格式的字段标识了对象的内部版本号，其取值来自 etcd 的 modifiedindex，且当对象被修改时，该字段将随之被修改。值得注意的是该字段由服务端维护

```go
type ObjectMeta struct {
    // An opaque value that represents the internal version of this object that can be used by clients to determine when objects have changed. May be used for optimistic concurrency, change detection, and the watch operation on a   resource or set of resources.Clients must treat these values as opaque and passed unmodified   back to the server.They may only be valid for a particular resource or set of resources.
    // Populated by the system.Read-only.
    ResourceVersion string
    ...
}
```

所谓的选主，就是看哪个follower能将自己的信息更新到 object 的annotation 上。 

### 选主逻辑

```go
// 等待，直到ctx 取消/成为leader再失去leader 后返回
func (le *LeaderElector) Run(ctx context.Context) {
	defer func() {
		runtime.HandleCrash()
		le.config.Callbacks.OnStoppedLeading()
    }()
    // 等待，除非成为leader（返回true） 或者ctx 取消（返回false）
	if !le.acquire(ctx) {
		return 
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
    go le.config.Callbacks.OnStartedLeading(ctx)
    // 成为leader后周期性续期，如果ctx 取消或失去leader 则立即返回
	le.renew(ctx)
}
```

选主核心逻辑
1. 没有lock，抢占/Create
2. 已有lock，但是别人的，租约没过期则退出 再试，过期则抢占/Update
3. 已有lock，自己的，续期/Update

函数返回 True 说明本 goroutine 已成功抢占到锁，获得租约合同，成为 leader

```go
func (le *LeaderElector) tryAcquireOrRenew() bool {
	now := metav1.Now()
	leaderElectionRecord := rl.LeaderElectionRecord{...}
	// 1. obtain or create the ElectionRecord
	oldLeaderElectionRecord, err := le.config.Lock.Get()
	if err != nil {
        // 获取锁信息失败则直接返回
		if !errors.IsNotFound(err) {return false}
        // 锁不存在则创建，创建失败则返回
		if err = le.config.Lock.Create(leaderElectionRecord); err != nil {return false}
        // 创建lock成功 即第一次选主抢占leader 成功，则返回
		return true
	}
	// 2. Record obtained, check the Identity & Time
	if len(oldLeaderElectionRecord.HolderIdentity) > 0 && le.observedTime.Add(le.config.LeaseDuration).After(now.Time) &&!le.IsLeader() { // 其他leader 未过期
		return false
	}
	// 3. We're going to try to update. The leaderElectionRecord is set to it's default
	// here. Let's correct it before updating.
	if le.IsLeader() {
		leaderElectionRecord.AcquireTime = oldLeaderElectionRecord.AcquireTime
		leaderElectionRecord.LeaderTransitions = oldLeaderElectionRecord.LeaderTransitions
	} else {
		leaderElectionRecord.LeaderTransitions = oldLeaderElectionRecord.LeaderTransitions + 1
	}
	// update the lock itself
	if err = le.config.Lock.Update(leaderElectionRecord); err != nil {
        return false
    }
	return true
}
```

通过LeaderCallbacks 感知leader 状态变化。回调OnStartedLeading 和 OnNewLeader 都会另起协程执行。

```go
type LeaderCallbacks struct {
    // OnStartedLeading is called when a LeaderElector client starts leading
    // 当选主逻辑退出时，会通过 context 传给OnStartedLeading
	OnStartedLeading func(context.Context)
	// OnStoppedLeading is called when a LeaderElector client stops leading
	OnStoppedLeading func()
	// OnNewLeader is called when the client observes a leader that is not the previously observed leader. This includes the first observed leader when the client starts.
	OnNewLeader func(identity string)
}
```

## 应用示例

k8s scheduler 调度器的执行入口是  `sched.Run`

```go
// k8s.io/kubernetes/cmd/kube-scheduler/app/server.go
func Run(ctx context.Context, cc schedulerserverconfig.CompletedConfig, outOfTreeRegistryOptions ...Option) error {
    ...
    // If leader election is enabled, runCommand via LeaderElector until done and exit.
    if cc.LeaderElection != nil {
        cc.LeaderElection.Callbacks = leaderelection.LeaderCallbacks{
            OnStartedLeading: sched.Run,    // 本节点成为leader时运行
            OnStoppedLeading: func() {
                klog.Fatalf("leaderelection lost")
            },
        }
        leaderElector, err := leaderelection.NewLeaderElector(*cc.LeaderElection)
        if err != nil {
            return fmt.Errorf("couldn't create leader elector: %v", err)
        }
        leaderElector.Run(ctx)
        return fmt.Errorf("lost lease")
    }
    // 如果未开启选主
    sched.Run(ctx)
	return fmt.Errorf("finished without leader elect")
}
```
k8s controller-manager 的选主逻辑
```go
// Run runs the KubeControllerManagerOptions.  This should never exit.
func Run(c *config.CompletedConfig, stopCh <-chan struct{}) error {
    ...
    run := func(ctx context.Context) {
        ...
    }
    ...
    rl, err := resourcelock.New(c.ComponentConfig.Generic.LeaderElection.ResourceLock,...)
	if err != nil {
		klog.Fatalf("error creating lock: %v", err)
    }
    leaderelection.RunOrDie(context.TODO(), leaderelection.LeaderElectionConfig{
		Lock:          rl,
		LeaseDuration: c.ComponentConfig.Generic.LeaderElection.LeaseDuration.Duration,
		RenewDeadline: c.ComponentConfig.Generic.LeaderElection.RenewDeadline.Duration,
		RetryPeriod:   c.ComponentConfig.Generic.LeaderElection.RetryPeriod.Duration,
		Callbacks: leaderelection.LeaderCallbacks{
			OnStartedLeading: run,
			OnStoppedLeading: func() {
				klog.Fatalf("leaderelection lost")
			},
		},
		WatchDog: electionChecker,
		Name:     "kube-controller-manager",
    })
    panic("unreachable")
}
```
从示例中可以看到，选主一般是一次性的，成为leader 后即执行核心业务逻辑。如果成为leader 后失去leader，则主协程执行结束。scheduler 和 controller-manager 部署在容器中，所以主协程执行结束后，一般会自动重启。