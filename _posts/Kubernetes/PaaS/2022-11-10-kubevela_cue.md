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


## 基本使用

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

## kubevela 中使用

### 描述 Component/Trait/Policy等Definition

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
    /workflow
      /operation
      /providers
      /step
      /template
      /workflow.go
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

### 描述WorkflowStepDefinition（未完成）

### provider 机制 （未完成）
