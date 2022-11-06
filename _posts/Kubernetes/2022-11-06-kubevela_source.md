---

layout: post
title: kubevela源码分析
category: 架构
tags: Kubernetes
keywords:  Kubernetes kubevela

---

## 简介（未完成）

* TOC
{:toc}

## 使用

[Deploy First Application](https://kubevela.io/docs/quick-start)

```yaml
apiVersion: core.oam.dev/v1beta1
kind: Application
metadata:
  name: first-vela-app
spec:
  components:
    - name: express-server
      type: webservice
      properties:
        image: oamdev/hello-world
        ports:
         - port: 8000
           expose: true
      traits:
        - type: scaler
          properties:
            replicas: 1
  policies:
    - name: target-default
      type: topology
      properties:
        # The cluster with name local is installed the KubeVela.
        clusters: ["local"]
        namespace: "default"
    - name: target-prod
      type: topology
      properties:
        clusters: ["local"]
        # This namespace must be created before deploying.
        namespace: "prod"
    - name: deploy-ha
      type: override
      properties:
        components:
          - type: webservice
            traits:
              - type: scaler
                properties:
                  replicas: 2
  workflow:
    steps:
      - name: deploy2default
        type: deploy
        properties:
          policies: ["target-default"]
      - name: manual-approval
        type: suspend
      - name: deploy2prod
        type: deploy
        properties:
          policies: ["target-prod", "deploy-ha"]
```

`vela def list` 查看涉及到的 Definition，对于webservice，`kubectl get ComponentDefinition -n vela-system webservice -o yaml` 或者 `vela def get $defname` 看下其内容

<table>
  <tr>
    <th></th>
    <th>Definition 种类</th>
    <th>cue template内容</th>
    <th>处理方式</th>
  </tr>
  <tr>
    <td>webservice</td>
    <td>ComponentDefinition</td>
    <td>Deployment</td>
    <td>转为 Deployment并分发</td>
  </tr>
  <tr>
    <td>scaler</td>
    <td>TraitDefinition</td>
    <td><pre>template: {
	parameter: {
		// +usage=Specify the number of workload
		replicas: *1 | int
	}
	// +patchStrategy=retainKeys
	patch: spec: replicas: parameter.replicas
}</pre></td>
  <td></td>
  </tr>
  <tr>
    <td>topology</td>
    <td>PolicyDefinition</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>override</td>
    <td>PolicyDefinition</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>deploy</td>
    <td>WorkflowStepDefinition</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>suspend</td>
    <td>WorkflowStepDefinition</td>
    <td></td>
    <td></td>
  </tr>
</table>



## 源码结构

```
kubevela
  /cmd
    /core
  /pkg
    /controller
      /standard.oam.dev
        /v1alpha1
          /rollout
      /core.oam.dev
        /v1alpha2
          /application
            /application_controller.go
          /applicationconfiguration
          /core
            /components
            /policies
            /scopes
            /traits
            /workflow

```



## 启动


```go
// kubevela/cmd/core/main.go
func main() {
  mgr, err := ctrl.NewManager(restConfig, ctrl.Options{
    ...
  }
  oamv1alpha2.Setup(mgr, controllerArgs)
  ...
  mgr.Start(ctrl.SetupSignalHandler())
}
// kubevela/pkg/controller/core.oam.dev/v1alpha2/setup.go
func Setup(mgr ctrl.Manager, args controller.Args) error {
  switch args.OAMSpecVer {
	case "all":
		for _, setup := range []func(ctrl.Manager, controller.Args) error{
			application.Setup, 
      traitdefinition.Setup, 
      componentdefinition.Setup, 
      policydefinition.Setup, 
      workflowstepdefinition.Setup,
			applicationconfiguration.Setup,
		} {
			setup(mgr, args)
		}
  case "minimal":
  case "v0.3":
  case "v0.2":
}
```


## 引入workflow之前的处理流程

## 引入workflow之后的处理流程

```go
// kubevela/pkg/controller/core.oam.dev/v1alpha2/application/application_controller.go
func (r *Reconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
  app := new(v1beta1.Application)
  r.Get(ctx, client.ObjectKey{Name:      req.Name,Namespace: req.Namespace,}, app)

  // 解析为appFile
  appParser := appfile.NewApplicationParser(r.Client, r.dm, r.pd)
  handler, err := NewAppHandler(logCtx, r, app, appParser)
  appFile, err := appParser.GenerateAppFile(logCtx, app)

  handler.PrepareCurrentAppRevision(logCtx, appFile)
  handler.FinalizeAndApplyAppRevision(logCtx)

  handler.ApplyPolicies(logCtx, appFile);

  workflowInstance, runners, err :=handler.GenerateApplicationSteps(logCtx, app, appParser, appFile, handler.currentAppRev)

  executor := executor.New(workflowInstance, r.Client)

  workflowState, err := executor.ExecuteRunners(authCtx, runners)
}
```
从 Application 解析为appFile，将部署计划拆分为 runners（一个runner即TaskRunner，包含一个Run方法） 由executor 驱动执行。

```go
// kubevela/workflow/pkg/executor/workflow.go
func (w *workflowExecutor) ExecuteRunners(ctx monitorContext.Context, taskRunners []types.TaskRunner) (v1alpha1.WorkflowRunPhase, error) {
  e := newEngine(ctx, wfCtx, w, status)

  err = e.Run(ctx, taskRunners, dagMode)
}
// kubevela/workflow/pkg/executor/workflow.go
func (e *engine) Run(ctx monitorContext.Context, taskRunners []types.TaskRunner, dag bool) error {
	if dag {
		err = e.runAsDAG(ctx, taskRunners, false)
	} else {
		err = e.steps(ctx, taskRunners, dag)
	}
	return err
}

func (e *engine) steps(ctx monitorContext.Context, taskRunners []types.TaskRunner, dag bool) error {
  ...
	for index, runner := range taskRunners {
    ...
		status, operation, err := runner.Run(wfCtx, options)
    ...
	}
	return nil
}
```