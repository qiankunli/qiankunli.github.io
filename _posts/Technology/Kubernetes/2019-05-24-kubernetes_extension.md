---

layout: post
title: 扩展Kubernetes
category: 技术
tags: Kubernetes
keywords: kubernetes crd

---

## 简介

* TOC
{:toc}

## CRD概念

建议先查看[Kubernetes 控制器模型](http://qiankunli.github.io/2019/03/07/kubernetes_controller.html)

### 先从扩展k8s object开始——Custom Resource

与custome resource 对应的词儿 是   built-in Kubernetes resources  (like pods).

**In the Kubernetes API, a resource is an endpoint that stores a collection of API objects of a certain kind**

[Custom Resources](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)Custom resources can appear and disappear in a running cluster through dynamic registration

**下文都假设自定义 一个 pod2 resource**

### 描述CustomResource——CustomResourceDefinition

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

### 使得custom object生效——Custom controllers

对于k8s 来说，api server 只是将 pod2 这个resource crud 到 etcd 上，若要使其真正“干活儿” 还要实现 对应的Pod2 Controller 

[Custom Resources](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)On their own, custom resources simply let you store and retrieve structured data. **It is only when combined with a controller that they become a true declarative API**. A declarative API allows you to declare or specify the desired state of your resource and tries to match the actual state to this desired state. Here, the controller interprets the structured data as a record of the user’s desired state, and continually takes action to achieve and maintain this state. 

基于声明式 API 的业务功能实现，往往需要通过控制器模式来“监视”API 对象的变化（比如，创建或者删除Pod2），然后以此来决定实际要执行的具体工作。

自定义custom controller 就有点 自定义 ansible module的意思。

## 实操——极客时间

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

### 原理预览

![](/public/upload/kubernetes/k8s_custom_controller.png)

### 定义custom object使得apiserver 支持crud

一个自定义的crd（Network） 实现 [resouer/k8s-controller-custom-resource](https://github.com/resouer/k8s-controller-custom-resource)

Network 资源类型在服务器端的注册的工作，APIServer 会自动帮我们完成。但与之对应的，我们还需要让客户端也能“知道”Network资源类型的定义。这就需要 `pkg/apis/pod2/v1/register.go`。

1. 自定义资源类型的 API 描述，包括：组（Group）、版本（Version）等
2. 自定义资源类型的对象描述，包括：Spec、Status 等  
5. `kubectl apply -f crd/network.yaml`
6. `kubectl apply -f example/example-network.yaml`

然后可以发现，单纯Network数据的crud 是没问题了，但crud 不是目的，我们希望能够根据 Network crud 做出反应，这就需要Controller 的协作了

### 定义custom controller使得custom object起作用

**使用 Kubernetes 提供的代码生成工具，为上面定义的pod2资源类型自动生成 clientset、informer和 lister**。

1. clientset 就是操作pod2 对象所需要使用的客户端。
2. Informer，其实就是一个带有本地缓存和索引机制的、可以注册 EventHandler 的 client（三个 Handler（AddFunc、UpdateFunc 和 DeleteFunc）。通过监听到的事件变化，Informer 就可以实时地更新本地本地缓存，并且调用这些事件对应的 EventHandler

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

### main 函数逻辑

![](/public/upload/kubernetes/crd_sequence.png)

因为Controller 包含kubeClient,networkClient,networkInformer成员

    type Controller struct {
        // kubeclientset is a standard kubernetes clientset 可以理解为api server的客户端
        kubeclientset kubernetes.Interface
        // networkclientset is a clientset for our own API group 可以理解api server的客户端，但专门访问自定义的resource
        networkclientset clientset.Interface
        networksLister listers.NetworkLister
        ...
    }

所以main 函数的主要逻辑

1. 构造kubeClient,networkClient,networkInformer 对象，进而构造Controller，其中networkInformer比较复杂，各种封装和工厂模式
2. 异步启动networkInformer
3. 启动Controller 主流程，即run方法

### controller逻辑

![](/public/upload/kubernetes/crd_object.png)

1. 在main 函数构造Controller 时，NewController方法中执行了`networkInformer.AddEventHandler`逻辑，当informer 收到network对象时 ==> AddEventHandler ==> network 对象加入到workqueue中
2. Controller processNextWorkItem 中仅负责 从workqueue中取出network对象 执行自己的逻辑即可

### 其它

这套流程不仅可以用在自定义 API 资源上，也完全可以用在Kubernetes 原生的默认 API 对象上。

不管 built-in resource 还是 custom resource ，controller 不是 api server的一部分，自然也不用和 apiserver 一起启动，apiserver 只管crud etcd。

[Kubernetes Deep Dive: Code Generation for CustomResources](https://blog.openshift.com/kubernetes-deep-dive-code-generation-customresources/)

## 实例——autoscaler

[kubernetes自动扩容缩容](http://qiankunli.github.io/2019/09/19/kubernetes_auto_scaler.html)

![](/public/upload/kubernetes/auto_scaler.png)

### CustomResourceDefinition定义

    apiVersion: apiextensions.k8s.io/v1beta1
    kind: CustomResourceDefinition
    metadata:
    # 名称必须符合下面的格式：<plural>.<group>
    name: verticalpodautoscalers.autoscaling.k8s.io
    spec:
    group: autoscaling.k8s.io
    scope: Namespaced
    version: v1beta1
    names:
        kind: VerticalPodAutoscaler
        ## 复数名称
        plural: verticalpodautoscalers
        ## 单数名称
        singular: verticalpodautoscaler
        shortNames:
        - vpa

### VerticalPodAutoscaler 示例

	apiVersion: autoscaling.k8s.io/v1beta2
	kind: VerticalPodAutoscaler
	metadata:
	  name: my-rec-vpa
	spec:
	  targetRef:
	    apiVersion: "extensions/v1beta1"
	    kind:       Deployment
	    name:       my-rec-deployment
	  updatePolicy:
	    updateMode: "Off"

## [扩展API](https://jimmysong.io/kubernetes-handbook/concepts/custom-resource.html)

自定义资源实际上是为了扩展kubernetes的API，向kubenetes API中增加新类型，可以使用以下三种方式：

1. 修改kubenetes的源码，显然难度比较高，也不太合适
2. 创建自定义API server并聚合到API中 Aggregated APIs are subordinate APIServers that sit behind the primary API server, which acts as a proxy. 

	![](/public/upload/kubernetes/kubernetes_aggregator.png)

3. 1.7以下版本编写TPR，kubernetes1.7及以上版本用CRD

使用CRD 有如下优势

1. 你的API是否属于声明式的
2. 是否想使用kubectl命令来管理
3. 是否要作为kubenretes中的对象类型来管理，同时显示在kuberetes dashboard上
4. 是否可以遵守kubernetes的API规则限制，例如URL和API group、namespace限制
4. 是否可以接受该API只能作用于集群或者namespace范围
5. 想要复用kubernetes API的公共功能，比如CRUD、watch、内置的认证和授权等


k8s核心对象的api，带上各种自定义api/非核心api，apiserver 支持的url就很多了

![](/public/upload/kubernetes/k8s_api.png)

## 另一种扩展——operator

[Kubernetes Controller vs Kubernetes Operator?](https://stackoverflow.com/questions/47848258/kubernetes-controller-vs-kubernetes-operator)

The list of controller in the Control-plane，比如

1. Deployment
2. ReplicaSet
3. StatefulSet
4. DaemonSet

From the Google Search, I found out that there are K8s Operators such as

1. etcd Operator
2. Prometheus Operator
3. kong Operators


All Operators use the controller pattern, but not all controllers are Operators. It's only an Operator if it's got: controller pattern + API extension + single-app focus.

Operator is a customized controller implement with CRD. It follow the same pattern with build-in controllers (i.e. watch, diff, action).


作者 believe the term "kubernetes operator" was introduced by [the CoreOS people](https://coreos.com/operators/) here

	An Operator is a method of packaging, deploying and managing a Kubernetes application. A Kubernetes application is an application that is both deployed on Kubernetes and managed using the Kubernetes APIs and kubectl tooling.

	An Operator is an application-specific controller that extends the Kubernetes API to create, configure and manage instances of complex stateful applications on behalf of a Kubernetes user. It builds upon the basic Kubernetes resource and controller concepts, but also includes domain or application-specific knowledge to automate common tasks better managed by computers.

So basically, a kubernetes operator is the name of a pattern that consists of a kubernetes controller that adds new objects to the Kubernetes API, in order to configure and manage an application, such as Prometheus or etcd. 为应用kubernetes 化而生的。

In one sentence: An operator is a domain specific controller.