---

layout: post
title: karmada支持crd
category: 架构
tags: Kubernetes
keywords:  集群

---

## 简介

* TOC
{:toc}

## 背景

In the progress of propagating a resource from karmada-apiserver to member clusters, Karmada needs to know the resource definition. For Kubernetes native resources, Karmada knows how to parse them, but for custom resources defined by CRD, as lack of the knowledge of the resource structure, they can only be treated as normal resources. Therefore, the advanced scheduling algorithms cannot be used for them.The [Resource Interpreter Framework](https://karmada.io/docs/next/userguide/globalview/customizing-resource-interpreter/)is designed for interpreting resource structure. 

karmada 从 karmada-apiserver 向多个cluster 分发crd的过程中，一边是用户创建的、存在于karmada-apiserver 的crd，一边是运行在各个cluster 的crd，对于两边的crd 需要进行各种操作和状态同步。
1. 比如使用GetReplicas 获取crd的副本数，计算完cluster1 应负责的crd 的replica1 之后，向cluster1 创建 replica=replica1 的crd。
2. 假如deployment有10个pod，两个cluster 分别运行2/8 个pod，用户创建的deployment 的status 等字段 也应该能感知到真实情况。

ResourceDetector 组件监听 用户创建的object，查找 object 匹配的 PropagationPolicy 并创建ResourceBinding。创建 ResourceBinding 时会执行 `ResourceInterpreter.GetReplicas(object)` 拿到crd 的副本数 为集群层的调度提供依据。
```
// ResourceDetector is a resource watcher which watches all resources and reconcile the events.
controllermanager.Run ==> ResourceDetector.Reconcile
	object, err := d.GetUnstructuredObject(clusterWideKey)
	propagationPolicy, err := d.LookForMatchedPolicy(object, clusterWideKey)
	d.ApplyPolicy(object, clusterWideKey, propagationPolicy)
		binding, err := d.BuildResourceBinding(object, objectKey, policyLabels, policy.Spec.PropagateDeps)
			d.ResourceInterpreter.GetReplicas(object)
		controllerutil.CreateOrUpdate(context.TODO(), d.Client, bindingCopy...)
```



## 代码分析

resourceinterpreter 相关的目录结构

```
karmada
  /examples
    /customresourceinterpreter  # 示例一个名为workload的crd如何实现
      /apis
      /webhook
        /app
          /webhook.go
          /workloadwebhook.go
        /main.go
  /pkg
    /resourceinterpreter
	    /customizedInterpreter
        /configmanager          # 管理所有ResourceInterpreterWebhookConfiguration
        /webhook                # 对应 /pkg/webhook，提供了request 和 response 
        /customized.go          # ResourceInterpreter 的  CustomizedInterpreter 实现
	    /defaultInterpreter
	    /interpreter.go
    /webhook                    # 对自定义实现webhook 进行了简单的封装（比如请求的编解码等）， 自定义webhook时只需要实现一个子类即可。
      /interpreter
```
### karmada 侧


|ResourceInterpreter函数|功能|触发时机|
|---|---|---|
|HookEnabled|检查是否有对应的 的ResourceInterpreterWebhookConfiguration|进行下列操作之前|
|GetReplicas|创建ResourceBinding 时获取karmada-apiserver  crd的replica|ResourceDetector.Reconcile ==> ApplyPolicy ==> BuildResourceBinding|
|ReviseReplica|分发到每个 cluster 时调整 cluster内crd的replica |ResourceBindingController.syncBinding ==> ensureWork|
|GetDependencies|比如对于  deployment 来说 configmap/secret/serviceaccount/pvc 即为 dependency|DependenciesDistributor.ReconcileResourceBinding|
|Retain|将用户对karmada-apiserver  object 的变更（比如label、annotation等）同步到各个cluster的object|objectWatcherImpl.retainClusterFields|
|ReflectStatus|解析 各个cluster 下crd 的status 并汇总到所属work 的status下|WorkStatusController.RunWorkQueue ==> syncWorkStatus|
|AggregateStatus| 将分发到各个cluster 的object 的状态（从worker.status）汇总并更新到 karmada-apiserver  crd status上|ResourceBindingController.syncBinding ==> AggregateResourceBindingWorkStatus + updateResourceStatus|

[Design Details](https://github.com/karmada-io/karmada/tree/master/docs/proposals/resource-interpreter-webhook)

![](/public/upload/kubernetes/karmada_interpreter_operation.png)

### interpreter webhook 侧

webhook 本质就是一个简单的 webserver，The webhook handles the ResourceInterpreterRequest request sent by the Karmada components (such as karmada-controller-manager), and sends back its decision as an ResourceInterpreterResponse.

```go
// karmada/examples/customresourceinterpreter/webhook/app/webhook.go
func Run(ctx context.Context, opts *options.Options) error {
	config, err := controllerruntime.GetConfig()
	hookManager, err := controllerruntime.NewManager(config, controllerruntime.Options{
		Logger:         klog.Background(),
		Host:           opts.BindAddress,
		Port:           opts.SecurePort,
		CertDir:        opts.CertDir,
		LeaderElection: false,
	})
	klog.Info("registering webhooks to the webhook server")
	hookServer := hookManager.GetWebhookServer()
	hookServer.Register("/interpreter-workload", interpreter.NewWebhook(&workloadInterpreter{}, interpreter.NewDecoder(gclient.NewSchema())))
	hookServer.WebhookMux.Handle("/readyz/", http.StripPrefix("/readyz/", &healthz.Handler{}))

	// blocks until the context is done.
	hookManager.Start(ctx)
	// never reach here
	return nil
}
```

当收到 `/interpreter-workload` 请求，webhook.ServeHTTP ==> webhook.Handle ==> workloadInterpreter.Handle，之后根据请求的具体类型进行相应的处理。

```go
func (e *workloadInterpreter) Handle(ctx context.Context, req interpreter.Request) interpreter.Response {
	workload := &workloadv1alpha1.Workload{}
	err := e.decoder.Decode(req, workload)
	klog.Infof("Explore workload(%s/%s) for request: %s", workload.GetNamespace(), workload.GetName(), req.Operation)
	switch req.Operation {
	case configv1alpha1.InterpreterOperationInterpretReplica:
		return e.responseWithExploreReplica(workload)
	case configv1alpha1.InterpreterOperationReviseReplica:
		return e.responseWithExploreReviseReplica(workload, req)
	case configv1alpha1.InterpreterOperationRetain:
		return e.responseWithExploreRetaining(workload, req)
	case configv1alpha1.InterpreterOperationAggregateStatus:
		return e.responseWithExploreAggregateStatus(workload, req)
	case configv1alpha1.InterpreterOperationInterpretHealth:
		return e.responseWithExploreInterpretHealth(workload)
	case configv1alpha1.InterpreterOperationInterpretStatus:
		return e.responseWithExploreInterpretStatus(workload)
	default:
		return interpreter.Errored(http.StatusBadRequest, fmt.Errorf("wrong request operation type: %s", req.Operation))
	}
}
```

任何 访问这个 webhook 的 组件（在karmada 中主要是 controller ）都需要 知道 webserver 提供服务的url 地址等信息，这些信息 可以作为 请求组件的启动参数传入，也可以写入到crd（即 ResourceInterpreterWebhookConfiguration），请求组件直接解析 ResourceInterpreterWebhookConfiguration 即可，karmada 选择后者。以下列配置为例，当karmada resourceInterpreter 需要 获取 Workload 的副本数时，便去 解析 名为 `workloads.example.com` 的ResourceInterpreterWebhookConfiguration 拿到 url，发出请求，访问用户自定义的crd 解析 replicas逻辑。

```yaml
apiVersion: config.karmada.io/v1alpha1
kind: ResourceInterpreterWebhookConfiguration
metadata:
  name: examples
webhooks:
  - name: workloads.example.com
    rules:
      - operations: [ "InterpretReplica","ReviseReplica","Retain","AggregateStatus", "InterpretHealth", "InterpretStatus" ]
        apiGroups: [ "workload.example.io" ]
        apiVersions: [ "v1alpha1" ]
        kinds: [ "Workload" ]
    clientConfig:
      url: https://{{karmada-interpreter-webhook-example-svc-address}}:443/interpreter-workload
      caBundle: {{caBundle}}
    interpreterContextVersions: [ "v1alpha1" ]
    timeoutSeconds: 3
```