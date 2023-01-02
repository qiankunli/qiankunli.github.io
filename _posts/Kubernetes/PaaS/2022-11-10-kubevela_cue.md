---

layout: post
title: kubevela中cue的应用
category: 架构
tags: Kubernetes
keywords:  kubevela cue

---

## 简介

* TOC
{:toc}


[使用 CUE 语言管理 K8s 资源清单模板](https://mp.weixin.qq.com/s/8xnlkhi3tq6Cay_hbyRcqQ)

[数据约束语言 CUE 简易教程](https://mp.weixin.qq.com/s/yWKdXsfAZvwc5Y1aao6XAw)

BCL 全名 Borg Configuration Language，是 Google 内部基于 GCL (Generic Configuration Language) 在 Borg 场景的实践。**用户通过 BCL 描述对 Borg 的使用需求**，通过基于 BCL 的抽象省去对 Borg 复杂配置细节的感知提高单位效率，通过工程化手段满足可抽象、可复用、可测试的协作方式提高团队效率和稳定性，并在其上建立了相应的生态平台，作为 Borg 生态的重要抽象层在 Google 内部服务了超过 10 年，帮助 Google 内部数万开发者更好的使用 Infra。CUE 是一种服务于云化配置的**强类型配置语言**，由 Go team 成员 Marcel van Lohiuzen 结合 BCL 及多种其他语言研发并开源，可以说是 BCL 思路的开源版实现。PS：大家都对直接使用yaml 的方式不满意

## 基本使用

### 数据类型

cue 的基本数据类型

```
// first.cue
a: 1.5
a: float
b: 1
b: int
d: [1, 2, 3]
g: {
 h: "abc"
}
e: "abc"
```

如何自定义 CUE 类型？使用 # 符号来指定一些表示 CUE 类型的变量

```
#abc: {
  x: int
  y: string
  z: {
    a: float
    b: bool
  }
}
```

### 渲染

```
$ cue eval first.cue
a: 1.5
b: 1
d: [1, 2, 3]
g: {
h: "abc"
}
e: "abc"    
```
默认情况下, 渲染结果会被格式化为 JSON 格式。
```
$ cue export first.cue
{
    "a": 1.5,
    "b": 1,
    "d": [
        1,
        2,
        3
    ],
    "g": {
        "h": "abc"
    },
    "e": "abc"
}
```

渲染为yaml 格式

```
$ cue export first.cue --out yaml
a: 1.5
b: 1
d:
- 1
- 2
- 3
g:
  h: abc
e: abc
```

## 与k8s结合

```
// deployment.cue
parameter:{
   name: "mytest"
   image: "nginx:v1"
}
template: {
 apiVersion: "apps/v1"
 kind:       "Deployment"
 spec: {
  selector: matchLabels: {
   "app.oam.dev/component": parameter.name
  }
  template: {
   metadata: labels: {
    "app.oam.dev/component": parameter.name
   }
   spec: {
    containers: [{
     name:  parameter.name
     image: parameter.image
    }]
   }}}
}
```

`cue export deployment.cue -e template --out yaml` 导出指定template变量的结果。

### 导入包

可以在 CUE 模版中通过 `kube/<apiVersion>` 导入 kubernetes 的包，就像使用 CUE 内部包一样。

```
import (
   apps "kube/apps/v1"
)
parameter: {
    name:  string
}
output: apps.#Deployment                ## output 的类型是 deployment
output: {
    metadata: name: parameter.name      ## 给output 的metadata.name 字段赋值
}
```

## kubevela 中使用

```
kubevela
  /pkg
    /appfile
      /parser.go
    /cue
      /definition
      /model
      /process
      /task
      utils.go
    /workflow             # 工作流相关
      /operation
      /providers
      /step
      /template
      /workflow.go
    /stdlib
      /op.cue
```

**kubevela 可扩展的的核心是复用，主要复用两方面的能力**
1. component/trait 主要复用 crd 的能力，由ComponentDefinition/TraitDefinition 使用cue 来定义crd 模版，Application controller 负责渲染出 真实crd
2. workflow 复用http/email/k8s(apply等) 等能力，由 WorkflowStepDefinition 使用cue 来定义对底层provider 能力/函数的调用（provider 函数被封装为taskRunner），executor 从 WorkflowStepDefinition 拿到 provider+op（底层能力标识） “渲染”为taskRunner 并调用执行。

你可以通过 在StepDefinition 中引用cue 官方以及KubeVela内置的包，来使用HTTP 请求、crud 资源、配置日志来源、条件等待、使当前步骤失败等函数，从而通过配置的方式来覆盖你的业务场景，极大提升 Step的可扩展性。 

### 描述 Component/Trait/Policy等Definition

webservice ComponetDefinition 内容
```
import (
	"strconv"
)
webservice: {...}
template: {
  mountsArray: {...}  # 根据 parameter.volumeMounts 计算mount 相关的内容，最终会被output 使用
  volumesList: {...}
  ...
  output: {            # deployment yaml的核心部分
    apiVersion: "apps/v1"
		kind:       "Deployment"
		spec: {...}
  }         
  exposePorts: {}
  outputs: {
    if len(exposePorts) != 0 {
      webserviceExpose: {
				apiVersion: "v1"
				kind:       "Service"
				metadata: name: context.name
				spec: {
					selector: "app.oam.dev/component": context.name
					ports: exposePorts
					type:  parameter.exposeType
				}
			}
    }
  }
  parameter: {...}    # template的参数部分
}
```

```go
// kubevela/pkg/appfile/parser.go
func generateComponentFromCUEModule(c client.Client, wl *Workload, appName, revision, ns string) (*v1alpha2.Component, ..., error) {
	pCtx, err := PrepareProcessContext(c, wl, appName, revision, ns)
	for _, tr := range wl.Traits {
		tr.EvalContext(pCtx)
	}
	comp, acComp, err = evalWorkloadWithContext(pCtx, wl, appName, wl.Name)
	for _, sc := range wl.Scopes {
		acComp.Scopes = append(acComp.Scopes, v1alpha2.ComponentScope{...})
	}
	return comp, acComp, nil
}
```
从Application 中拿到 component/trait name，找到对应的Definition 拿到cue 模版，加上Application 提供的参数，转成Component ，并将trait 挂在 ApplicationConfigurationComponent 对象里。
```
PrepareProcessContext
  Workload.EvalContext
    workloadDef.Complete(ctx process.Context, abstractTemplate string, params interface{})
      bi := build.NewContext().NewInstance("", nil)
      bi.AddFile("-", abstractTemplate)
      var paramFile = "parameter: {}"
      bt, err := json.Marshal(params)
      paramFile = fmt.Sprintf("%s: %s", mycue.ParameterTag, string(bt))
      bi.AddFile("parameter", paramFile)
      bi.AddFile("-", ctx.ExtendedContextFile())
      var r cue.Runtime
      inst, err := r.Build(bi)
      output := inst.Lookup("output") # 返回cue文件中 output 字段的值 
      base, err := model.NewBase(output)
      ctx.SetBase(base)
Trait.EvalContext
  traitDef.Complete(ctx, trait.Template, trait.Params)
    bi := build.NewContext().NewInstance("", nil)
    bi.AddFile("-", abstractTemplate)
    var paramFile = "parameter: {}"
    bt, err := json.Marshal(params)
    paramFile = fmt.Sprintf("%s: %s", mycue.ParameterTag, string(bt))
    bi.AddFile("parameter", paramFile)
    bi.AddFile("context", ctx.ExtendedContextFile())
    instances := cue.Build([]*build.Instance{bi})
    for _, inst := range instances {
      patcher := inst.Lookup("patch") # 返回cue文件中 patch 字段的值
      base, _ := ctx.Output()
      p, err := model.NewOther(patcher)
      base.Unify(p)
    }
evalWorkloadWithContext
  base, assists := pCtx.Output()
	componentWorkload, err := base.Unstructured()
  component := &v1alpha2.Component{}
  // we need to marshal the workload to byte array before sending them to the k8s
	component.Spec.Workload = util.Object2RawExtension(componentWorkload)

  acComponent := &v1alpha2.ApplicationConfigurationComponent{}
  for _, assist := range assists {
		tr, err := assist.Ins.Unstructured()
    acComponent.Traits = append(acComponent.Traits, v1alpha2.ComponentTrait{
			Trait: util.Object2RawExtension(tr),
		})
  }
```

k8s patch语法参考 [Update API Objects in Place Using kubectl patch](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/update-api-object-kubectl-patch/)

### 描述WorkflowStepDefinition

deploy step 示例

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
`kubevela/pkg/providers` 设计了一套机制，当workflow executor 执行deploy step时，executor.doSteps 首先从 template.deploy 的  `#provider` 和 `#do` 中拿到 provider和 do，接着执行exec.Handle(...,provider,do) ==> handler := providers.GetHandler(provider,do);handler()。

```
// kubevela/pkg/stdlib/op.cue
#Deploy: multicluster.#Deploy
// kubevela/pkg/stdlib/pkgs/multicluster.cue
#Deploy: {
	#provider: "multicluster"
	#do:       "deploy"
	policies: [...string]
	parallelism:              int
	ignoreTerraformComponent: bool
}
// kubevela/pkg/workflow/providers/multicluster/multicluster.go
// Install register handlers to provider discover.
func Install(p wfTypes.Providers, c client.Client, app *v1beta1.Application, af *appfile.Appfile,...) {
	prd := &provider{Client: c, app: app, af: af, apply: apply, healthCheck: healthCheck, renderer: renderer}
	p.Register(ProviderName, map[string]wfTypes.Handler{
		"read-placement-decisions":              prd.ReadPlacementDecisions,
		"make-placement-decisions":              prd.MakePlacementDecisions,
		"patch-application":                     prd.PatchApplication,
		"list-clusters":                         prd.ListClusters,
		"get-placements-from-topology-policies": prd.GetPlacementsFromTopologyPolicies,
		"deploy":                                prd.Deploy,
	})
}
```

从效果看，根据 op.Deploy ==>  multicluster.#Deploy 可以找到 `kubevela/pkg/workflow/providers/multicluster/multicluster.go` 的Deploy 方法。换句话说，workflow/provider 实现了一些通用的能力函数，加上stdlib 等机制，我们在workflowStep cue 模版中可以直接使用这些函数。

suspend step 示例

```
suspend: {
	alias: ""
	annotations: {}
	attributes: {}
	description: "Suspend the current workflow, it can be resumed by 'vela workflow resume' command."
	labels: {}
	type: "workflow-step"
}

template: parameter: {
	// +usage=Specify the wait duration time to resume workflow such as "30s", "1min" or "2m15s"
	duration?: string
}
```

### provider 机制 （未完成）

