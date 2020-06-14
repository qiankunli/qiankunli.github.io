---

layout: post
title: Kubernetes源码分析——apiserver
category: 技术
tags: Kubernetes
keywords: kubernetes 源码分析

---

## 简介

* TOC
{:toc}


背景知识

1. GO语言的http包使用
2. 基于RESTful风格的[go-restful](https://github.com/emicklei/go-restful) 
3. 声明式API
4. Go 语言的代码生成

## 声明式API

1. 命令式命令行操作，比如直接 `kubectl run`
2. 命令式配置文件操作，比如先`kubectl create -f xx.yaml` 再 `kubectl replace -f xx.yaml` 
3. 声明式API 操作，比如`kubectl apply -f xx.yaml`。**声明式api 有两个要素 且 都与命令式 api 不同。** PS： 有点团队使命式管理和命令式管理的意思

	||声明式|命令式|
	|---|---|---|
	|描述任务|期望目标 desired state|如何做： command<br>不管最后结果是否符合你的预期|
	|执行器|根据  desired state 决定自己的行为<br>通常用到控制器模式|直接执行 command|
	
**命令式api 描述和执行 是一体的，声明式api 则需要额外的 执行器**（下文叫Controller） sync desired state 和 real state。declarative API 的感觉也可以参见 [ansible 学习](http://qiankunli.github.io/2018/12/29/ansible.html) ，Controller 有点像 ansible 中的module

声明式API 有以下优势

1. 实现层的逻辑不同。kube-apiserver 在响应命令式请求（比如，kubectl replace）的时候，一次只能处理一个写请求，否则会有产生冲突的可能。而对于声明式请求（比如，kubectl apply），一次能处理多个写操作，并且具备 Merge 能力。
2. 如果xx.yaml 不变，可以任意多次、同一时间并发 执行apply 操作。
3. “声明式 API”允许有多个 API 写端，以 PATCH 的方式对 API 对象进行修改，而无需关心本地原始 YAML文件的内容。例如lstio 会自动向每一个pod 写入 envoy 容器配置（用户无感知），如果xx.yaml 是一个 xx.sh 则该效果很难实现。

[火得一塌糊涂的kubernetes有哪些值得初学者学习的？](https://mp.weixin.qq.com/s/iI5vpK5bVkKmdbf9sbAGWw)在分布式系统中，任何组件都可能随时出现故障。当组件恢复时，需要弄清楚要做什么，使用命令式 API 时，处理起来就很棘手。但是使用声明式 API ，组件只需查看 API 服务器的当前状态，即可确定它需要执行的操作。《阿里巴巴云原生实践15讲》 称之为：**面向终态**自动化。


## 整体架构

![](/public/upload/kubernetes/apiserver_overview.png)

在[Kubernetes源码分析——从kubectl开始](http://qiankunli.github.io/2018/12/23/kubernetes_source_kubectl.html) [Kubernetes源码分析——kubelet](http://qiankunli.github.io/2018/12/31/kubernetes_source_kubelet.html)系列博客中，笔者都是以创建pod 为主线来学习k8s 源码。 在学习api server 之初，笔者想当然的认为 `kubectl create -f xxpod.yaml` 发出http 请求，apiserver 收到请求，然后有一个PodHandler的东西处理相关逻辑， 比如将信息保存在etcd 上。结果http.server 启动部分都正常，但PodHandler 愣是没找到。k8s apiserver 刷新了笔者对http.server 开发的认知。

1. apiserver 将resource/kubernetes object 数据保存在etcd 上，因为resource 通过etcd 持久化的操作模式比较固定，一个通用http.Handler 根据resource 元数据 即可完成 crud，无需专门的PodHandler/ServiceHandler 等
2. apiserver 也不单是built-in resource的api server，**是一个通用api server**，这也是为何相关的struct 叫 GenericAPIServer。 就是你定义一个user.yaml，dynamic registry user.yaml 到 apiserver上，然后就可以直接 `http://apiserver/api/users` 返回所有User 数据了。

## 启动

[kube-apiserver 启动流程 - 1](https://segmentfault.com/a/1190000013954577) kube-apiserver 主要功能是提供 api 接口给客户端访问 后端 etcd 存储，当然这中间不光是简单的 key/value 存储，为了方便扩展，kube-apiserver 设计了一套代码框架将 "资源对象" 映射到 RESTful API

![](/public/upload/kubernetes/apiserver_init_sequence.png)

启动流程比较好理解，最后的落脚点 肯定是 http.Server.Serve

![](/public/upload/kubernetes/apiserver_object.png)

GenericAPIServer 初始化时构建 http.Handler，描述了怎么处理用户请求，然后以此触发http.Server.Serve 开始对外提供 http 服务。所以难点不再这，难点在

1. `/api/v1/namespaces/{namespace}/pods/{name}` 请求必然 有一个 XXHandler 与之相对应，这个XXHandler的代码在哪？逻辑是什么？
2. Pod/Service 等kubernetes object http 请求与之对应的 XXHandler 如何绑定？

## etcd crud

apiserver 启动时 自动加载 built-in resource/object 的scheme 信息， 对crud  url 根据 crud 的不同分别绑定在 通用的 Creater/Deleter 等http.Handler 进行处理

![](/public/upload/kubernetes/kubernetes_object_save.png)

### 处理请求的url

使用[go-restful](https://github.com/emicklei/go-restful)时，一个很重要的过程就是 url 和 handler 绑定，绑定逻辑在 CreateServerChain 中。  

![](/public/upload/kubernetes/http_handler_init_sequence.png)

观察 endpoints.APIInstaller 可以看到 其install 方法返回一个 `*restful.WebService`，这些信息最终给了 APIServerHandler

	func (g *APIGroupVersion) InstallREST(container *restful.Container) error {
		installer := &APIInstaller{
			group:                        g,
			prefix:                       prefix,
			minRequestTimeout:            g.MinRequestTimeout,
			enableAPIResponseCompression: g.EnableAPIResponseCompression,
		}
		apiResources, ws, registrationErrors := installer.Install()
		container.Add(ws)
	}

![](/public/upload/kubernetes/http_handler_object.png)

上图要从右向左看（UML 软件的缘故，不过有一阵儿分析进入瓶颈，的确是从storage 自下往下来串调用链）。 右边是绑定部分，绑定完成后，当接到restful 请求时 会路由到 `apiserver/pkg/registry/rest/rest.go` 定义的XXCreater/XXDeleter 等接口的实现类上。

### Schema 信息加载

观察`storage.etcd3.store` 的Create 方法，可以看到其根据runtime.Object 保存数据，那么具体的类信息如何加载呢？

`kubernetes/pkg/apis/core/register.go`

	// Adds the list of known types to the given scheme.
	func addKnownTypes(scheme *runtime.Scheme) error {
		scheme.AddKnownTypes(SchemeGroupVersion,
			&Pod{},
			&PodList{},
			&PodStatusResult{},
			&PodTemplate{},
			&PodTemplateList{},
			&ReplicationController{},
			&ReplicationControllerList{},
			&Service{},
			&ServiceProxyOptions{},
			&ServiceList{},
			&Endpoints{},
			&EndpointsList{},
			&Node{},
			&NodeList{},
			&NodeProxyOptions{},
			...
		)
		

addKnownTypes  的执行链条为 `kubernetes/pkg/apis/core/install/install.go` init ==> Install ==> core.AddToScheme(scheme) ==> `kubernetes/pkg/apis/core/register.go` ==> addKnownTypes 

init 函数 初始化了一个全局变量 `legacyscheme.Scheme`

CreateServerChain 方法内 调用的CreateKubeAPIServerConfig 方法用到了 `legacyscheme.Scheme`

## 拦截api请求

1. Admission Controller
2. Initializers
3. webhooks, If you’re not planning to modify the object and intercepting just to read the object, [webhooks](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/#external-admission-webhooks) might be a faster and leaner alternative to get notified about the objects. Make sure to check out [this example](https://github.com/caesarxuchao/example-webhook-admission-controller) of a webhook-based admission controller.

### Admission Controller

准入控制器是kubernetes 的API Server上的一个链式Filter，它根据一定的规则决定是否允许当前的请求生效，并且有可能会改写资源声明。比如

1. enforcing all container images to come from a particular registry, and prevent other images from being deployed in pods. 
2. applying pre-create checks
3. setting up default values for missing fields.

The problem with admission controllers are:

1. **They’re compiled into Kubernetes**: If what you’re looking for is missing, you need to fork Kubernetes, write the admission plugin and keep maintaining a fork yourself.
2. You need to enable each admission plugin by passing its name to --admission-control flag of kube-apiserver. In many cases, this means redeploying a cluster.
3. Some managed cluster providers may not let you customize API server flags, therefore you may not be able to enable all the admission controllers available in the source code.

### Initializers

[How Kubernetes Initializers work](https://ahmet.im/blog/initializers/)

Initializers are not part of Kubernetes source tree, or compiled into it; you need to write a controller yourself.

**When you intercept Kubernetes objects before they are created, the possibilities are endless**: You can mutate the objects in any way you like, or prevent the objects from being created.Here are some ideas for initializers, each enforce a particular policy in your cluster:

1. Inject a proxy sidecar container to the pod if it has port 80, or has a particular annotation.
2. Inject a volume with test certificates to all pods in the test namespace automatically.
3. If a Secret is shorter than 20 characters (probably a password), prevent its creation.

Anatomy of Initialization

1. Configure which resource types need initialization
2. API server will assign initializers to the new resources
3. You will write a controller to watch for the resources
4. Wait for your turn to modify the resource
5. Finish modifying, yield to the next initializer
6. No more initializers, resource ready to be realized. When Kubernetes API server sees that the object has no more pending initializers, it considers the object “initialized”. **Now the Kubernetes scheduler and other controllers can see the fully initialized object and make use of them**.

从中可以看到，为啥Admission Controller 干活要改源码，Initializers 不用呢？ 因为干活要改源码，Initializer 只是给待处理资源加上了标记`metadata.initalizers.pending=InitializerName`，需要相应的Controller 打辅助。

[示例](https://github.com/kelseyhightower/kubernetes-initializer-tutorial)

## udpate 机制

[理解 K8s 资源更新机制，从一个 OpenKruise 用户疑问开始](https://mp.weixin.qq.com/s/jWH7jVxj20bmc60_C-w9wQ)

![](/public/upload/kubernetes/update_resource.png)

## etcd: Kubernetes’ brain

**Every component in Kubernetes (the API server, the scheduler, the kubelet, the controller manager, whatever) is stateless**. All of the state is stored in a key-value store called etcd, and communication between components often happens via etcd.

For example! Let’s say you want to run a container on Machine X. You do not ask the kubelet on that Machine X to run a container. That is not the Kubernetes way! Instead, this happens:

1. you write into etcd, “This pod should run on Machine X”. (technically you never write to etcd directly, you do that through the API server, but we’ll get there later)
2. the kublet on Machine X looks at etcd and thinks, “omg!! it says that pod should be running and I’m not running it! I will start right now!!”


Similarly, if you want to put a container somewhere but you don’t care where:

1. you write into etcd “this pod should run somewhere”
2. the scheduler looks at that and thinks “omg! there is an unscheduled pod! This must be fixed!“. It assigns the pod a machine (Machine Y) to run on
3. like before, the kubelet on Machine Y sees that and thinks “omg! that is scheduled to run on my machine! Better do it now!!”
When I understood that basically everything in Kubernetes works by watching etcd for stuff it has to do, doing it, and then writing the new state back into etcd, Kubernetes made a lot more sense to me.

[Reasons Kubernetes is cool](https://jvns.ca/blog/2017/10/05/reasons-kubernetes-is-cool/)Because all the components don’t keep any state in memory(stateless), you can just restart them at any time and that can help mitigate a variety of bugs.The only stateful thing you have to operate is etcd
