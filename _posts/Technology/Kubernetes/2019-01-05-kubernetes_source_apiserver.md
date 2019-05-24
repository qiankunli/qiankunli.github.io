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

1. 实现层的逻辑不同。kube-apiserver 在响应命令式请求（比如，kubectl replace）的时候，一次只能处理一个写请求，否则会有产生冲突的可能。而对于声明式请求（比如，kubectl apply），一次能处理多个写操作，并且具备 Merge 能力。
2. 如果xx.yaml 不变，可以任意多次、同一时间并发 执行apply 操作。
3. “声明式 API”允许有多个 API 写端，以 PATCH 的方式对 API 对象进行修改，而无需关心本地原始 YAML文件的内容。例如lstio 会自动向每一个pod 写入 envoy 容器配置（用户无感知），如果xx.yaml 是一个 xx.sh 则该效果很难实现。

[火得一塌糊涂的kubernetes有哪些值得初学者学习的？](https://mp.weixin.qq.com/s/iI5vpK5bVkKmdbf9sbAGWw)在分布式系统中，任何组件都可能随时出现故障。当组件恢复时，需要弄清楚要做什么，使用命令式 API 时，处理起来就很棘手。但是使用声明式 API ，组件只需查看 API 服务器的当前状态，即可确定它需要执行的操作。


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

## Admission Controller

准入控制器是kubernetes 的API Server上的一个链式Filter，它根据一定的规则决定是否允许当前的请求生效，并且有可能会改写资源声明。

	

