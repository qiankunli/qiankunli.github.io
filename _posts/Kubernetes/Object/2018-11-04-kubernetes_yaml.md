---

layout: post
title: kubernetes yaml配置
category: 技术
tags: Kubernetes
keywords: kubernetes yaml

---

## 简介

* TOC
{:toc}


Kubernetes 跟 Docker 等很多项目最大的不同，就在于它不推荐你使用命令行的方式直接运行容器（虽然 kubectl run 支持)，而是采用yaml/json 文件的方式。最直接的好处是，你会有一个文件能记录下 Kubernetes到底“run”了什么。使用文件的优点归纳起来

1. Convenience，比如`kubectl create -f https://k8s.io/examples/application/deployment.yaml --record` 命令行可这样玩不了
2. Maintenance， 比如使用git 管理
3. Flexibility，也就是说表达能力更强


[简化 Kubernetes Yaml 文件创建](https://yq.aliyun.com/articles/341213)由于Yaml文件格式比较复杂，即使是老司机有时也不免会犯错或需要查询文档，因此可以dry-run 一下，`kubectl run myapp --image=nginx --dry-run -o yaml` 会输出模拟运行 nginx 镜像的yaml 文件内容，copy-paste 即可。或者你可以` kubectl get deployment my-nginx -o yaml ` 查看一个已有 kubernetes object 的配置，依葫芦画瓢。

了解kubernetes yaml 主要从两个维度：

1. yaml 文件的普遍特征
2. Kubernetes Object 的共同特征


## yaml 的一些知识

[Introduction to YAML: Creating a Kubernetes deployment](https://www.mirantis.com/blog/introduction-to-yaml-creating-a-kubernetes-deployment/)

1. YAML, which stands for Yet Another Markup Language，yaml 是一个标记语言
1. YAML is a superset of JSON， yaml 是json 的超集
2. there are only two types of structures you need to know about in YAML:

	* Lists
	* Maps

### yaml Maps

	apiVersion: v1
	kind: Pod
	metadata:
	  name: rss-site
	  labels:
	    app: web

 1. Maps let you associate name-value pairs
 2. 只要“平行/级”，就是同一个层级的key-value。有了缩进，就表示一个map value。层级之间缩进空格数任意，哪怕一个空格也可以，但不要使用tab。 For example, name and labels are at the same indentation level, so the processor knows they’re both part of the same map; it knows that app is a value for labels because it’s indented further.

 
### yaml list

	args
	  - sleep
	  - "1000"
	  - message
	  - "Bring back Firefly!"


you can have virtually any number of items in a list, which is defined as items that start with a dash (-) indented from the parent. 

## Describing a Kubernetes Object

[Understanding Kubernetes Objects](https://kubernetes.io/docs/concepts/overview/working-with-objects/kubernetes-objects/)

### Kubernetes Object

Kubernetes Objects are persistent entities in the Kubernetes system. A Kubernetes object is a “record of intent”–once you create the object, the Kubernetes system will constantly work to ensure that object exists. 

1. What containerized applications are running (and on which nodes)
2. The resources available to those applications
3. The policies around how those applications behave, such as restart policies, upgrades, and fault-tolerance


Every Kubernetes object includes two nested object fields that govern the object’s configuration: the object spec and the object status.

1. The spec, which you must provide, describes your desired state 
2. The status describes the actual state of the object, and is supplied and updated by the Kubernetes system. pod 状态可以使用 `kubectl get pod pod_name  -o yaml` 来查看，或者 `kubectl describe pod pod_name` 。


At any given time, the Kubernetes Control Plane actively manages an object’s actual state to match the desired state you supplied. 基于这种机制 不管是`kubectl create -f ` 还是 `kubectl replace -f` 都可以是 `kubectl apply -f`，这或许也是kubernetes 声明式api 的一个体现吧。

### yaml 配置共同点


1. apiVersion - Which version of the Kubernetes API you’re using to create this object
2. kind - What kind of object you want to create
3. metadata - Data that helps uniquely identify the object, including a name string, UID, and optional namespace
4. spec - The precise format of the object spec is different for every Kubernetes object, and contains nested fields specific to that object. 每一个 Kubernetes object 就得参见  [Kubernetes API Reference](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.12/)了

metadata 与 spec 分别代表了 共性与个性，数据表设计也可以参照这个思路

metadata 中包含Label 和 Annotation，作用差不多，但有两个区别
1. k8s 支持根据 label 对object 进行检索， Annotation 不行
2. 因为label 需要支持检索，所以label 只能是kv 结构，Annotation value 可以是复杂一点，比如json 字符串

## PodPreset

开发人员习惯的写的，是最简单的pod

	apiVersion: v1
	kind: Pod
	metadata: 
		name: website 
		labels:
			app: website 
			role: frontend
	spec: 
		containers: 
			- name:website 
			  image: nginx
			  ports: 
				- containerPort:80

但对运维来说，在实际环境中还需添加大量的配置，此时，运维可以事先定义一个PodPreset.yaml，并创建一个PodPreset`kubectl create -f preset.yaml`。 之后开发创建的pod（有一个规则匹配） 都会自动加上 preset.yaml 指定的配置。



## 访问多个kubernetes 集群

1. 一般情况，kubernetes 单独搭建在一个集群上，开发者通过开发机 或某一个跳板机上 通过kubectl 操作kubernetes，kubectl 会读取`~/.kube/config` 文件读取集群信息
2. kubernetes 一般会有多个集群：测试环境（运行公司测试环境的服务），开发环境（用来验证新功能）==> developer 需要在本机 上使用kubectl 访问多个k8s集群

[配置对多集群的访问](https://kubernetes.io/zh/docs/tasks/access-application-cluster/configure-access-multiple-clusters/)

`~/.kube/config` 是一个yaml 文件，可以配置多个集群的信息

    apiVersion: v1
    kind: Config
    clusters:
    users:
    contexts:

可以看到 几个核心配置都是数组

    apiVersion: v1
    kind: Config
    clusters:
    - cluster:
    name: development
    - cluster:
    name: scratch
    users:
    - name: developer
    - name: experimenter
    contexts:
    - context:
        cluster: development
        user: developer
      name: dev-frontend
    name: dev-frontend
    - context:
        cluster: scratch
        user: experimenter
      name: exp-scratch