---

layout: post
title:  Admission Controller 与  Admission Webhook 
category: 架构
tags: Kubernetes
keywords:  Kubernetes webhook

---

## 简介

* TOC
{:toc}

## Admission Controller 以及为什么需要 webhook

[为什么需要 Kubernetes 准入控制器](https://mp.weixin.qq.com/s/TjvIdKY6EJMVx6TiagM7Jg)

准入控制器是kubernetes 的API Server上的一个链式Filter，它根据一定的规则决定是否允许当前的请求生效，并且有可能会改写资源声明。比如

1. enforcing all container images to come from a particular registry, and prevent other images from being deployed in pods. 
2. applying pre-create checks
3. setting up default values for missing fields.

Kubernetes 的 apiserver 一开始就有 AdmissionController 的设计，这个设计和各类 Web 框架中的 Filter  很像，就是一个插件化的责任链，责任链中的每个插件针对 apiserver 收到的请求做一些操作或校验。The problem with admission controllers are:

1. **They’re compiled into Kubernetes**: If what you’re looking for is missing, you need to fork Kubernetes, write the admission plugin and keep maintaining a fork yourself.
2. You need to enable each admission plugin by passing its name to --admission-control flag of kube-apiserver. In many cases, this means redeploying a cluster.
3. Some managed cluster providers may not let you customize API server flags, therefore you may not be able to enable all the admission controllers available in the source code.

K8s支持30多种admission control 插件，其中有两个具有强大的灵活性，即ValidatingAdmissionWebhooks和MutatingAdmissionWebhooks，这两种控制变换和准入以Webhook的方式提供给用户使用，大大提高了灵活性，用户可以在集群创建自定义的AdmissionWebhookServer进行调整准入策略。

## 配置apiserver 发起webhook

![](/public/upload/kubernetes/admission_controller.png)

2. MutatingWebhookConfiguration，在对象持久化之前，修改对象的内容或者拒绝请求。 会对request的resource，进行转换，比如填充默认的request/limit。因为对象的字段可能被不同的准入控制器修改多次，所以准入控制器链的顺序就尤其重要。
1. ValidatingWebhookConfiguration，在对象持久化之前，校验对象的内容或者拒绝请求。 比如校验Pod副本数必须大于2。

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
...
webhooks:
- name: my-webhook.example.com
  rules:
  - operations: ["CREATE", "UPDATE"]
    apiGroups: ["apps"]
    apiVersions: ["v1", "v1beta1"]
    resources: ["deployments", "replicasets"]
    scope: "Namespaced"
```

rules 控制apiserver 何时 因为何种资源 发出请求：上例中 匹配针对 `apps/v1` 和 `apps/v1beta1` 组中 deployments 和 replicasets 资源的 CREATE 或 UPDATE 请求

clientConfig 描述如何调用webhook

1. url 方式

    ```yaml
    apiVersion: admissionregistration.k8s.io/v1
    kind: MutatingWebhookConfiguration
    ...
    webhooks:
    - name: my-webhook.example.com
    clientConfig:
        url: "https://my-webhook.example.com:9443/my-webhook-path"
    ```
2. service 方式

    ```yaml
    apiVersion: admissionregistration.k8s.io/v1
    kind: MutatingWebhookConfiguration
    ...
    webhooks:
    - name: my-webhook.example.com
    clientConfig:
        caBundle: "Ci0tLS0tQk...<base64-encoded PEM bundle containing the CA that signed the webhook's serving certificate>...tLS0K"
        service:
            namespace: my-service-namespace
            name: my-service-name
            path: /my-path
            port: 1234
    ```



## 请求响应参数

webhook本身是一个约定接口的**http callback/web server**。

```go
type AdmissionReview struct {
	metav1.TypeMeta `json:",inline"`
	// Request describes the attributes for the admission request.
	// +optional
	Request *AdmissionRequest `json:"request,omitempty" protobuf:"bytes,1,opt,name=request"`
	// Response describes the attributes for the admission response.
	// +optional
	Response *AdmissionResponse `json:"response,omitempty" protobuf:"bytes,2,opt,name=response"`
}
```

1. 向 Webhook 发送 POST 请求时，请设置 Content-Type: application/json 并对 admission.k8s.io API 组中的 AdmissionReview 对象进行序列化，将所得到的 JSON 作为请求的主体。
2. Webhook 使用 HTTP 200 状态码、Content-Type: application/json 和一个包含 AdmissionReview 对象的 JSON 序列化格式来发送响应。该 AdmissionReview 对象与发送的版本相同，且其中包含的 response 字段已被有效填充。response 至少必须包含以下字段：

    1. uid，从发送到 webhook 的 request.uid 中复制而来
    2. allowed，设置为 true 或 false

Webhook 禁止请求的最简单响应示例：

```json
{
    "apiVersion": "admission.k8s.io/v1",
    "kind": "AdmissionReview",
    "response": {
        "uid": "<value from request.uid>",
        "allowed": false
    }
}
```

当允许请求时，mutating admission webhook 也可以选择修改传入的对象。 这是通过在响应中使用 patch 和 patchType 字段来完成的。 当前唯一支持的 patchType 是 JSONPatch。 对于 patchType: JSONPatch，patch 字段包含一个以 base64 编码的 JSON patch 操作数组。例如，设置 `spec.replicas` 的单个补丁操作将是 `[{"op": "add", "path": "/spec/replicas", "value": 3}]`。

```json
{
  "apiVersion": "admission.k8s.io/v1",
  "kind": "AdmissionReview",
  "response": {
    "uid": "<value from request.uid>",
    "allowed": true,
    "patchType": "JSONPatch",
    "patch": "W3sib3AiOiAiYWRkIiwgInBhdGgiOiAiL3NwZWMvcmVwbGljYXMiLCAidmFsdWUiOiAzfV0="
  }
}
```

## 实现

### apiserver调用

```
apiserver
  /pkg
     /addmission
       /plugin
          /namespace
          /resourcequota
          /webhook
    /authentication
    /authorization
    /registry
    /server
    /storage
```

![](/public/upload/kubernetes/webhook_admission_controller.jpg)

```go
// apiserver/pkg/admission/plugin/webhook/mutating/plugin.go
func (a *Plugin) Admit(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	return a.Webhook.Dispatch(ctx, attr, o)
}
Webhook.Dispatch
    hooks := a.hookSource.Webhooks()
    mutatingDispatcher/validatingDispatcher.Dispatch(ctx, attr, o, hooks)
        for i, hook := range hooks {
            invocation, statusErr := a.plugin.ShouldCallHook(hook, attrForCheck, o)
            hook, ok := invocation.Webhook.GetMutatingWebhook()
            changed, err := a.callAttrMutatingHook(ctx, hook, invocation, versionedAttr, o, round, i)
        }
```

### webhook 自身实现

webhook 是一个http server，是一个限制了请求与响应格式（AdmissionReview/AdmissionResponse）的http server

一个最简单的web server
```go
http.HandleFunc("/", func (w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "hello world")
})
http.ListenAndServe(":8000", nil)
```
volcano  自己从0 开始写webhook，设计了一个AdmissionService 抽象，一个crd 对应一个AdmissionService，之后将 注册的所有 AdmissionService 封装为一个http server。

```go
// volcano/pkg/webhooks/router/interface.go
type AdmitFunc func(admissionv1.AdmissionReview) *admissionv1.AdmissionResponse
type AdmissionService struct {
	Path    string                  // 请求路径
	Func    AdmitFunc               // 处理函数
	Handler AdmissionHandler        // 会有工具函数将 Func 转为 http.handler
    // Webhook 对应的配置，volcano webhook pod 启动时自动注册到apiserver中
	ValidatingConfig *whv1.ValidatingWebhookConfiguration
	MutatingConfig   *whv1.MutatingWebhookConfiguration
	Config *AdmissionServiceConfig
}
```

在volcano 的webhook中，ValidatingWebhookConfiguration 的配置是通过代码写入到 apiserver 的。对于volcano 这种大型框架，可能包含多个crd，每个crd 都会注册一个VatingWebhookConfiguration，如果是自己写的话，yaml 文件要写五六个。

openkruise/karmara 则使用了controller-runtime 库，一个 crd 对应一个webhook，仅需实现 `Handle(context.Context, admission.Request) admission.Response` 方法，框架负责将其转为 http.handler

```go
// karmada-io/karmada/cmd/webhook/app/webhook.go
func Run(ctx context.Context, opts *options.Options) error {
    hookManager, err := controllerruntime.NewManager(config, controllerruntime.Options{...})
    hookServer := hookManager.GetWebhookServer()
    hookServer.Register("/mutate-propagationpolicy", &webhook.Admission{Handler: propagationpolicy.NewMutatingHandler(
		opts.DefaultNotReadyTolerationSeconds, opts.DefaultUnreachableTolerationSeconds)})
    hookServer.Register("/validate-propagationpolicy", &webhook.Admission{Handler: &propagationpolicy.ValidatingAdmission{}})
    ...
    hookManager.Start(ctx)
}
```

## 应用场景

使用场景：[使用 Admission Webhook 机制实现多集群资源配额控制](https://mp.weixin.qq.com/s/i3KtTSfab2JrjeFR4tdy_A)集群分配给多个用户使用时，需要使用配额以限制用户的资源使用，包括 CPU 核数、内存大小、GPU 卡数等，以防止资源被某些用户耗尽，造成不公平的资源分配。大多数情况下，集群原生的 ResourceQuota 机制可以很好地解决问题。但随着集群规模扩大，以及任务类型的增多，我们对配额管理的规则需要进行调整：
1. ResourceQuota 针对单集群设计，但实际上，开发/生产中经常使用 多集群 环境。
2. 集群大多数任务通过比如deployment、mpijob 等 高级资源对象 进行提交，我们希望在高级资源对象的 提交阶段 就能对配额进行判断。但 ResourceQuota 计算资源请求时以 pod 为粒度，从而无法满足此需求。

[Kubernetes 中如何保证优雅地停止 Pod](https://mp.weixin.qq.com/s/NwJbBLhomaHBhCkIDR1KWA)利用 ValidatingAdmissionWebhook，在重要的 Pod 收到删除请求时，先在 webhook server 上请求集群进行下线前的清理和准备工作，并直接返回拒绝。这时候重点来了，Control Loop 为了达到目标状态（比如说升级到新版本），会不断地进行 reconcile，尝试删除 Pod，而我们的 webhook 则会不断拒绝，除非集群已经完成了所有的清理和准备工作。

