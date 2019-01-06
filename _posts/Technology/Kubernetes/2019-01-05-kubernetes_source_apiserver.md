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


建议先看下前文 [Kubernetes源码分析——从kubectl开始](http://qiankunli.github.io/2018/12/23/kubernetes_source_kubectl.html)

背景知识

1. GO语言的http包使用
2. 基于RESTful风格的[go-restful](https://github.com/emicklei/go-restful) 
3. 声明式API
4. Go 语言的代码生成

## 铺垫一下

### 声明式API

1. 命令式命令行操作，比如直接 `kubectl run`
2. 命令式配置文件操作，比如先`kubectl create -f xx.yaml` 再 `kubectl replace -f xx.yaml` 
3. 声明式API 操作，比如`kubectl apply -f xx.yaml`。**声明式api 有两个要素 且 都与命令式 api 不同。** PS： 有点团队使命式管理和命令式管理的意思

	||声明式|命令式|
	|---|---|---|
	|描述任务|期望目标 desired state|如何做： command|
	|执行器|根据  desired state 决定自己的行为<br>通常用到控制器模式|直接执行 command|
	
**命令式api 描述和执行 是一体的，声明式api 则需要额外的 执行器**（下文叫Controller） sync desired state 和 real state。declarative API 的感觉也可以参见 [ansible 学习](http://qiankunli.github.io/2018/12/29/ansible.html) ，Controller 有点像 ansible 中的module

声明式API 有以下优势

1. 如果xx.yaml 不变，可以任意多次、同一时间并发 执行apply 操作。
2. “声明式 API”允许有多个 API 写端，以 PATCH 的方式对 API 对象进行修改，而无需关心本地原始 YAML文件的内容。例如lstio 会自动向每一个pod 写入 envoy 容器配置，如果xx.yaml 是一个 xx.sh 则该效果很难实现。
 
### 重新理解API Server

在[Kubernetes源码分析——从kubectl开始](http://qiankunli.github.io/2018/12/23/kubernetes_source_kubectl.html) [Kubernetes源码分析——kubelet](http://qiankunli.github.io/2018/12/31/kubernetes_source_kubelet.html)系列博客中，笔者都是以创建pod 为主线来学习k8s 源码。 在学习api server 之初，笔者想当然的认为 `kubectl create -f xxpod.yaml` 发出http 请求，apiserver 收到请求，然后有一个PodHandler的东西处理相关逻辑， 比如将信息保存在etcd 上。结果http.server 启动部分都正常，但PodHandler 愣是没找到。k8s apiserver 刷新了笔者对http.server 开发的认知。

1. apiserver 将resource/kubernetes object 数据保存在etcd 上，因为resource 通过etcd 持久化的操作模式比较固定，一个通用http.Handler 根据resource 元数据 即可完成 crud，无需专门的PodHandler/ServiceHandler 等
2. **apiserver 也不单是built-in resource的api server，是一个通用api server**，这也是为何相关的struct 叫 GenericAPIServer。类比到 spring mvc 领域是什么感觉呢？ 就是你定义一个user.yaml，dynamic registry user.yaml 到 apiserver上，然后就可以直接 `http://apiserver/api/users` 返回所有User 数据了。
2. 声明式api 若想生效，每个resource 有一个对应的Controller， 但不是处理 Pod的http api的，而是跟api server 交互，比如ReplicaController 读取 Replica 和关联的pod数据， 发现pod 数不够，会通过api server 向etcd 再写一个pod
3. Controller 和 api server 交互逻辑 的大部分 也是自动生成的。就好比一个thrift rpc server 只需要实现 thrfit interface 方法即可，其它tcp 通信、序列化的代码都是自动生成的

这里的apiserver 给人的感觉更像是 api gateway，两者的目的不同。相同之处是，只要将相关信息注册到 api gateway后，api gateway 便可以接收相关的http请求，然后将请求转发到实际的后端（apiserver 叫[API Aggregation](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/apiserver-aggregation/)）。对于固定模式的业务 比如db/etcd crud，api server 干脆采用了反射方式直接crud，动态扩充自己 支持的api。

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

上图要从右向左看（UML 软件的缘故）。 右边是绑定部分，绑定完成后，当接到restful 请求时 会路由到 `apiserver/pkg/registry/rest/rest.go` 定义的XXCreater/XXDeleter 等接口的实现类上。

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


## 如何扩展api server——CRD


### Custom Resource

与custome resource 对应的词儿 是   built-in Kubernetes resources  (like pods).

**In the Kubernetes API, a resource is an endpoint that stores a collection of API objects of a certain kind**

[Custom Resources](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)Custom resources can appear and disappear in a running cluster through dynamic registration

**下文都假设自定义 一个 pod2 resource**

### CustomResourceDefinition

Spring 提供了扩展 xml 的机制，用来编写自定义的 xml bean ，例如 dubbo 框架，就利用这个机制实现了好多的 dubbo bean，比如 `<dubbo:application>` 、`<dubbo:registry>`。 spring ioc 启动时

1. 会扫描带有类似@Component 注解的类，将它们纳入到ioc中
2. [【spring系列】- Spring自定义标签工作原理](https://blog.csdn.net/yang1464657625/article/details/79034641)自定义标签 会被转为一个BeanDefinition ，registerBeanDefinition 到ioc ，进而初始化为bean，并纳入到ioc 的管理，你可以在自己的类 @Autowire 使用这个registry 对象。

**Pod 是kubernetes 的一个Resource ，会不会像spring 的BeanDefinition 一样有一个ResourceDefinition ，然后允许我们 自定义ResourceDefinition ？ 答案是肯定的——[CustomResourceDefinition](https://kubernetes.io/docs/tasks/access-kubernetes-api/custom-resources/custom-resource-definitions/)。** custom resource 是自定义的，但描述custom resource的 CustomResourceDefinition 是built-in的，官网例子

	apiVersion: apiextensions.k8s.io/v1beta1
	kind: CustomResourceDefinition
	metadata:
	  # name must match the spec fields below, and be in the form: <plural>.<group>
	  name: crontabs.stable.example.com
	spec:
		xxx
		
[Extending Kubernetes APIs with Custom Resource Definitions (CRDs)](https://medium.com/velotio-perspectives/extending-kubernetes-apis-with-custom-resource-definitions-crds-139c99ed3477)Custom resources definition (CRD) is a powerful feature introduced in Kubernetes 1.7 which enables users to add their own/custom objects to the Kubernetes cluster and use it like any other native Kubernetes objects. 

|k8s|spring|
|---|---|
|k8s Resource|spring bean|
| CustomResourceDefinition |BeanDefinition|
|ResourceDefinition/code-generator ==> Generic APIServer |ioc ==> bean factory|

两相对比，还是很有味道的。


在nginx 中，你可以自定义 指令。比如笔者实现过一个 upsync 指令[qiankunli/nginx-upsync-module-zk](https://github.com/qiankunli/nginx-upsync-module-zk) ，在nginx conf 中出现 upsync 指令时可以执行笔者的自定义逻辑。但自定义的指令 要和nginx 重新编译后 才可以生效，api server 可以不重启  支持 `/api/v1/namespaces/{namespace}/pod2s/{name}` 么？

### Custom controllers

对于k8s 来说，api server 只是将 pod2 这个resource crud 到 etcd 上，若要使其真正“干活儿” 还要实现 对应的Pod2 Controller 

[Custom Resources](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)On their own, custom resources simply let you store and retrieve structured data. **It is only when combined with a controller that they become a true declarative API**. A declarative API allows you to declare or specify the desired state of your resource and tries to match the actual state to this desired state. Here, the controller interprets the structured data as a record of the user’s desired state, and continually takes action to achieve and maintain this state. 

基于声明式 API 的业务功能实现，往往需要通过控制器模式来“监视”API 对象的变化（比如，创建或者删除Pod2），然后以此来决定实际要执行的具体工作。

自定义custom controller 就有点 自定义 ansible module的意思。

### 实操

来自极客时间 《深入剖析Kubernetes》 

	$ tree $GOPATH/src/github.com/<your-name>/pod2
	.
	├── controller.go
	├── crd
	│   └── pod2.yaml	// custom resource definition 文件
	├── example	
	│   └── example-pod2.yaml	// 一个pod2.yaml 的例子
	├── main.go
	└── pkg
	    └── apis
	        └── pod2
	            ├── register.go	// 放置后面要用到的全局变量
	            └── v1
	                ├── doc.go
	                ├── register.go	
	                └── types.go		// 定一个pod2 到底有哪些字段




pod2 资源类型在服务器端的注册的工作，APIServer 会自动帮我们完成。但与之对应的，我们还需要让客户端也能“知道”pod2资源类型的定义。这就需要 `pkg/apis/pod2/v1/register.go`。

1. 自定义资源类型的 API 描述，包括：组（Group）、版本（Version）等
2. 自定义资源类型的对象描述，包括：Spec、Status 等  
5. `kubectl apply -f crd/pod2.yaml`
6. `kubectl apply -f example/example-pod2.yaml`

然后可以发现，单纯pod2 数据的crud 是没问题了，但crud 不是目的，我们希望能够根据 pod2 crud 做出反应，这就需要Controller 的协作了


1. 使用 Kubernetes 提供的代码生成工具，为上面定义的pod2资源类型自动生成 clientset、informer和 lister。clientset 就是操作pod2 对象所需要使用的客户端。Informer，其实就是一个带有本地缓存和索引机制的、可以注册 EventHandler 的 client（三个 Handler（AddFunc、UpdateFunc 和 DeleteFunc）。通过监听到的事件变化，Informer 就可以实时地更新本地本地缓存，并且调用这些事件对应的 EventHandler

		$ tree
		.
		├── controller.go
		├── crd
		│   └── pod2.yaml
		├── example
		│   └── example-pod2.yaml
		├── main.go
		└── pkg
		    ├── apis
		    │   └── pod2
		    │       ├── constants.go
		    │       └── v1
		    │           ├── doc.go
		    │           ├── register.go
		    │           ├── types.go
		    │           └── zz_generated.deepcopy.go
		    └── client
		        ├── clientset
		        ├── informers
		        └── listers

1. 编写 main 函数
2. 编写自定义控制器的定义
3. 编写控制器里的业务逻辑，APIServer 里保存的“期望状态”，“实际状态”来自 实际的集群，通过对比“期望状态”和“实际状态”的差异，完成了一次调协（Reconcile）的过程。

![](/public/upload/kubernetes/k8s_custom_controller.png)

这套流程不仅可以用在自定义 API 资源上，也完全可以用在Kubernetes 原生的默认 API 对象上。

不管 built-in resource 还是 custom resource ，controller 不是 api server的一部分，自然也不用和 apiserver 一起启动，apiserver 只管crud etcd。

其它例子：一个自定义的crd（CustomResourceDefinition） 实现 [resouer/k8s-controller-custom-resource](https://github.com/resouer/k8s-controller-custom-resource)

### [扩展API](https://jimmysong.io/kubernetes-handbook/concepts/custom-resource.html)

自定义资源实际上是为了扩展kubernetes的API，向kubenetes API中增加新类型，可以使用以下三种方式：

1. 修改kubenetes的源码，显然难度比较高，也不太合适
2. 创建自定义API server并聚合到API中 Aggregated APIs are subordinate APIServers that sit behind the primary API server, which acts as a proxy. 
3. 1.7以下版本编写TPR，kubernetes1.7及以上版本用CRD

使用CRD 有如下优势

1. 你的API是否属于声明式的
2. 是否想使用kubectl命令来管理
3. 是否要作为kubenretes中的对象类型来管理，同时显示在kuberetes dashboard上
4. 是否可以遵守kubernetes的API规则限制，例如URL和API group、namespace限制
4. 是否可以接受该API只能作用于集群或者namespace范围
5. 想要复用kubernetes API的公共功能，比如CRUD、watch、内置的认证和授权等





[Kubernetes Deep Dive: Code Generation for CustomResources](https://blog.openshift.com/kubernetes-deep-dive-code-generation-customresources/)	

