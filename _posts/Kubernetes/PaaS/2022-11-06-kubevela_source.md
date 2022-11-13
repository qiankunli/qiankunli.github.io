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
  <td>将parameter.replicas patch到deployment的spec.replicas上</td>
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
    /core                               ## controller-runtime manager 启动
  /pkg
    /controller
      /standard.oam.dev
        /v1alpha1
          /rollout
      /core.oam.dev
        /v1alpha2       
          /application                  ## Application controller
            /application_controller.go  
          /applicationconfiguration
          /core                         ## XXDefinition controller
            /components
            /policies
            /scopes
            /traits
            /workflow

```

## 启动

controller-runtime manager 启动的的常见风格

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

xxDefinition 主要是管理 用户 定义的cue 模版，核心逻辑是 application 和  applicationconfiguration（在较新的版本去除了）。

## 引入workflow之前的处理流程

[源码解读：KubeVela 是如何将 appfile 转换为 K8s 特定资源对象的](https://developer.aliyun.com/article/783169)
1. 起点：Application
2. 中点：ApplicationConfiguration, Component
3. 终点：Deployment, Service

![](/public/upload/kubernetes/kubevela_model.png)

### 根据 Application 创建ApplicationConfiguration和Component

```go
// kubevela/pkg/controller/core.oam.dev/v1alpha2/application/application_controller.go
// Reconcile process app event
func (r *Reconciler) Reconcile(req ctrl.Request) (ctrl.Result, error) {
	app := new(v1beta1.Application)
	r.Get(ctx, client.ObjectKey{Name:      req.Name,Namespace: req.Namespace,}, app)

	handler := &appHandler{
		r:      r,
		app:    app,
		logger: applog,
	}
	// parse template
	appParser := appfile.NewApplicationParser(r.Client, r.dm, r.pd)
	handler.appfile = generatedAppfile

	// build template to applicationconfig & component
	ac, comps, err := appParser.GenerateApplicationConfiguration(generatedAppfile, app.Namespace)
  // apply application revision & component to the cluster
  handler.apply(ctx, appRev, ac, comps)
  ...
	return ctrl.Result{}, r.UpdateStatus(ctx, app)
}
```
Application 提供 component/trait 的名字 找到cue 模版，加上 Application 配置的properties 拼凑ApplicationConfiguration（没找到create 逻辑）和Component 并创建。 

```go
// kubevela/pkg/appfile/parser.go
// GenerateApplicationConfiguration converts an appFile to applicationConfig & Components
func (p *Parser) GenerateApplicationConfiguration(app *Appfile, ns string) (*v1alpha2.ApplicationConfiguration,
	[]*v1alpha2.Component, error) {
	appconfig := &v1alpha2.ApplicationConfiguration{}
  var components []*v1alpha2.Component
	
	for _, wl := range app.Workloads {
    // 将 Application 转换为 ApplicationConfiguration 和 Component
		comp, acComp, err = generateComponentFromCUEModule(p.client, wl, app.Name, app.RevisionName, ns)
		components = append(components, comp)
		appconfig.Spec.Components = append(appconfig.Spec.Components, *acComp)
	}
	return appconfig, components, nil
}

// kubevela/pkg/controller/core.oam.dev/v1alpha2/application/apply.go
// 在集群中创建 ApplicationConfiguration 和 Component 
func (h *appHandler) apply(..., ac *v1alpha2.ApplicationConfiguration) error {
	for _, comp := range comps {
		newComp := comp.DeepCopy()
		// newComp will be updated and return the revision name instead of the component name
		revisionName, err := h.createOrUpdateComponent(ctx, newComp)
	}
	return nil
}
```
### 根据 ApplicationConfiguration和Component 创建workload

```go
// kubevela/pkg/controller/core.oam.dev/v1alpha2/applicationconfiguration/applicationconfiguration.go
// Reconcile an OAM ApplicationConfigurations by rendering and instantiating its Components and Traits.
func (r *OAMApplicationReconciler) Reconcile(req reconcile.Request) (reconcile.Result, error) {
  ac := &v1alpha2.ApplicationConfiguration{}
    // 获取 ApplicationConfiguration
  r.client.Get(ctx, req.NamespacedName, ac);
  reconResult := r.ACReconcile(ctx, ac, log)
  return reconResult, err
}

// ACReconcile contains all the reconcile logic of an AC, it can be used by other controller
func (r *OAMApplicationReconciler) ACReconcile(ctx context.Context, ac *v1alpha2.ApplicationConfiguration,log logging.Logger) (result reconcile.Result) {
  // execute the prehooks
  // 渲染
  workloads, depStatus, err := r.components.Render(ctx, ac)
  applyOpts := []apply.ApplyOption{apply.MustBeControllableBy(ac.GetUID()), applyOnceOnly(ac, r.applyOnceOnlyMode, log)}
  // 创建 workload 和 traits 对应的 k8s 资源对象
  r.workloads.Apply(ctx, ac.Status.Workloads, workloads, applyOpts...)
  return reconcile.Result{RequeueAfter: waitTime}
}
```
渲染主要由 components 来完成，将 workload 形态从json 转换为 *unstructured.Unstructured，之后被部署到 k8s 。

```go
// kubevela/pkg/controller/core.oam.dev/v1alpha2/applicationconfiguration/render.go
type components struct {
	// indicate that if this is generated by application
	client   client.Reader
	dm       discoverymapper.DiscoveryMapper
	params   ParameterResolver
	workload ResourceRenderer
	trait    ResourceRenderer
}
func (r *components) Render(..., ac *v1alpha2.ApplicationConfiguration) ([]Workload, *v1alpha2.DependencyStatus, error) {...}
func (r *components) renderComponent(..., ac *v1alpha2.ApplicationConfiguration, ) (*Workload, error) {...}
func (r *components) renderTrait(..., ac *v1alpha2.ApplicationConfiguration) (*unstructured.Unstructured, *v1alpha2.TraitDefinition, error) {
```
有的trait是对workload的 patch，有的trait对应的独立的crd。通过trait.spec.workloadRefPath 建立trait与workload 之间的关系。
当trait 是patch时，kubevela 负责将patch 和 workload 做strategy merge 然后apply 到k8s。
```
workloads,... := components.Render(ctx context.Context, ac *v1alpha2.ApplicationConfiguration) 
  for _, acc := range ac.Spec.Components {
    components.renderComponent
    renderWorkload  # 返回 unstructured.Unstructured
    for _, ct := range acc.Traits {
      components.renderTrait
        renderTrait
    }
    components.renderScope
  }
  for i, acc := range ac.Spec.Components {
    components.handleDependency(ctx, workloads[i], acc, dag, ac)
  }
workloads.Apply(...,workloads,...)
  for _, wl := range w {
    ApplyInputRef
    APIApplicator.Apply
    ApplyOutputRef
    for _, trait := range wl.Traits {
      ApplyInputRef
      APIApplicator.Apply   # 当trait 本身就对应一个crd时， 会直接apply
      ApplyOutputRef        # 当trait 是patch 类型时，会在这里通过 workloadRefPath 拿到ref object，对其进行json merge，之后apply
    }
    for _, s := range wl.Scopes {
			applyScope(ctx, wl, s, workloadRef)
		}
  }
```

APIApplicator.Apply 更多是在 服务端模拟了 `kubeclt apply -f`的能力。

```go
// kubevela/pkg/utils/apply/apply.go
// Apply applies new state to an object or create it if not exist
func (a *APIApplicator) Apply(ctx context.Context, desired runtime.Object, ao ...ApplyOption) error {
	existing, err := a.createOrGetExisting(ctx, a.c, desired, ao...)
	if existing == nil {  // 说明是新建
		return nil
	}
  // 如果是object已经存在，则发送patch
  // threeWayMergePatch
	patch, err := a.patcher.patch(existing, desired)
	return errors.Wrapf(a.c.Patch(ctx, desired, patch), "cannot patch object")
}
```


## 引入workflow之后的处理流程

kubevela/workflow 是一个独立的工作流实现，有workflowInstance（对应一个工作流）/workflowStep/taskRunner（对应一个step）/executor等概念，将appFile 转为workflowInstance 即进入了workflow的工作流程。 

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
  // appFile 转为 workflowInstance
  workflowInstance, runners, err :=handler.GenerateApplicationSteps(logCtx, app, appParser, appFile, handler.currentAppRev)
  // executor 执行 workflowInstance
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
引入workflow之前，主要由applicationconfiguration controller 负责创建workload。if you only use components in the Application and do not declare a workflow, KubeVela will automatically create a default workflow for deploying the components when running the Application.
```
Parser.GenerateAppFile(ctx context.Context, app *v1beta1.Application)
  GenerateAppFileFromApp(ctx context.Context, app *v1beta1.Application)
    appfile := Parser.newAppfile(appName, ns, app)
    for _, comp := range app.Spec.Components {
      wd, err := Parser.parseWorkload(ctx, comp)
      ...
    }
    Parser.parseWorkflowSteps(ctx, appfile)
      Parser.loadWorkflowToAppfile(ctx, af)
        af.WorkflowSteps, err = step.NewChainWorkflowStepGenerator(
		    &step.RefWorkflowStepGenerator{Client: af.WorkflowClient(p.client), Context: ctx},
		    &step.DeployWorkflowStepGenerator{},
		    &step.Deploy2EnvWorkflowStepGenerator{},
		    &step.ApplyComponentWorkflowStepGenerator{},
		    &step.DeployPreApproveWorkflowStepGenerator{},
	).Generate(af.app, af.WorkflowSteps)
      for _, workflowStep := range af.WorkflowSteps {
        parseWorkflowStep(ctx, af, workflowStep.Type)
      }
    ...
```

核心是理解 type=deploy的step，其对应的 WorkflowStepDefinition 定义如下

```
import (
	"vela/op"
)
deploy: {
	alias: ""
	annotations: {}
	attributes: {}
	description: "A powerful and unified deploy step for components multi-cluster delivery with policies."
	labels: {}
	type: "workflow-step"
}
template: {
	deploy: op.#Deploy & {
		policies:                 parameter.policies
		parallelism:              parameter.parallelism
		ignoreTerraformComponent: parameter.ignoreTerraformComponent
	}
	parameter: {
		//+usage=If set to false, the workflow will suspend automatically before this step, default to be true.
		auto: *true | bool
		//+usage=Declare the policies that used for this deployment. If not specified, the components will be deployed to the hub cluster.
		policies?: [...string]
		//+usage=Maximum number of concurrent delivered components.
		parallelism: *5 | int
		//+usage=If set false, this step will apply the components with the terraform workload.
		ignoreTerraformComponent: *true | bool
	}
}
```

