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
3. Flexibility，也就是说表达能力更强，也是helm 这些工具工作的基础


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

## 更新原理

[为什么用户修改 YAML 文件后无法直接调用 update 接口更新，却可以通过 kubectl apply 命令更新呢？](https://mp.weixin.qq.com/s/jWH7jVxj20bmc60_C-w9wQ)

### 调用 K8s api 接口做更新

对于一个 K8s 资源对象比如 Deployment，我们尝试在修改其中 image 镜像时，如果有其他人同时也在对这个 Deployment 做修改，会发生什么？当然，这里还可以引申出两个问题：
1. 如果双方修改的是同一个字段，比如 image 字段，结果会怎样？
2. 如果双方修改的是不同字段，比如一个修改 image，另一个修改 replicas，又会怎么样？

其实，对一个 Kubernetes 资源对象做“更新”操作，简单来说就是通知 kube-apiserver 组件我们希望如何修改这个对象。而 K8s 为这类需求定义了两种“通知”方式，分别是 update 和 patch。
1. 在 update 请求中，我们需要将整个修改后的对象提交给 K8s；
2. 对于 patch 请求，我们只需要将对象中某些字段的修改提交给 K8s。

**K8s 要求用户 update 请求中提交的对象必须带有 resourceVersion**，也就是说我们提交 update 的数据必须先来源于 K8s 中已经存在的对象。因此，一次完整的 update 操作流程是：
1. 首先，从 K8s 中拿到一个已经存在的对象（可以选择直接从 K8s 中查询；如果在客户端做了 list watch，推荐从本地 informer 中获取）；
2. 然后，基于这个取出来的对象做一些修改，比如将 Deployment 中的 replicas 做增减，或是将 image 字段修改为一个新版本的镜像；
2. 最后，将修改后的对象通过 update 请求提交给 K8s；
此时，kube-apiserver 会校验用户 update 请求提交对象中的 resourceVersion 一定要和当前 K8s 中这个对象最新的 resourceVersion 一致，才能接受本次 update。否则，K8s 会拒绝请求，并告诉用户发生了版本冲突（Conflict）。

**当用户对某个资源对象提交一个 patch 请求时，kube-apiserver 不会考虑版本问题**，而是“无脑”地接受用户的请求（只要请求发送的 patch 内容合法），也就是将 patch 打到对象上、同时更新版本号。不过，patch 的复杂点在于，目前 K8s 提供了 4 种 patch 策略：json patch、merge patch、strategic merge patch、apply patch（从 K8s 1.14 支持 server-side apply 开始）。通过 `kubectl patch -h` 命令可以看到这个策略选项（默认采用 strategic）。

1. json patch。要指定操作类型，比如 add 新增还是 replace 替换，另外在修改 containers 列表时要通过元素序号来指定容器。这样一来，如果我们 patch 之前这个对象已经被其他人修改了，那么我们的 patch 有可能产生非预期的后果。比如在执行 app 容器镜像更新时，我们指定的序号是 0，但此时 containers 列表中第一个位置被插入了另一个容器，则更新的镜像就被错误地插入到这个非预期的容器中。
	```
	kubectl patch deployment/foo --type='json' -p \
  	'[
		{
			"op":"replace",
			"path":"/spec/template/spec/containers/0/image",
			"value":"app-image:v2"
		}
	]'
  	```
2. merge patch。无法单独更新一个列表中的某个元素，因此不管我们是要在 containers 里新增容器、还是修改已有容器的 image、env 等字段，都要用整个 containers 列表来提交 patch。显然，这个策略并不适合我们对一些列表深层的字段做更新，更适用于大片段的覆盖更新。不过对于 labels/annotations 这些 map 类型的元素更新，merge patch 是可以单独指定 key-value 操作的，相比于 json patch 方便一些，写起来也更加直观。
	```
	kubectl patch deployment/foo --type='merge' -p \
  	'{
		"spec":{
			"template":{
				"spec":{
					"containers":[
						{
							"name":"app",
							"image":"app-image:v2"
						},
						{
							"name":"nginx",
							"image":"nginx:alpline"}
					]
				}
			}
		}
	}'
	kubectl patch deployment/foo --type='merge' -p 
	'{
		"metadata":{
			"labels":{
				"test-key":"foo"
			}
		}
	}'
	```
3. strategic merge patch
	```
	在 K8s 原生资源的数据结构定义中额外定义了一些的策略注解，比如下面patchMergeKey 就代表了 containers 列表使用 strategic merge patch 策略更新时，会把下面每个元素中的 name 字段看作 key。
	// ...
	// +patchMergeKey=name
	// +patchStrategy=merge
	Containers []Container `json:"containers" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,2,rep,name=containers"`
	在我们 patch 更新 containers 不再需要指定下标序号了，而是指定 name 来修改，K8s 会把 name 作为 key 来计算 merge。
	kubectl patch deployment/foo -p \
  	'{
		"spec":{
			"template":{
				"spec":{
					"containers":[
						{
							"name":"nginx",
							"image":"nginx:mainline"
						}
					]
				}
			}
		}
	}'
	如果 K8s 发现当前 containers 中已经有名字为 nginx 的容器，则只会把 image 更新上去；而如果当前 containers 中没有 nginx 容器，K8s 会把这个容器插入 containers 列表。
	```
4. apply patch


### apply 更新

kubectl 为了给命令行用户提供良好的交互体感，设计了较为复杂的内部执行逻辑，诸如 apply、edit 这些常用操作其实背后并非对应一次简单的 update 请求。毕竟 update 是有版本控制的，如果发生了更新冲突对于普通用户并不友好。在使用默认参数执行 apply 时，触发的是 client-side apply。kubectl 逻辑如下：

1. 首先解析用户提交的数据（YAML/JSON）为一个对象 A；然后调用 Get 接口从 K8s 中查询这个资源对象：
2. 如果查询结果不存在，kubectl 将本次用户提交的数据记录到对象 A 的 annotation 中（key 为 `kubectl.kubernetes.io/last-applied-configuration`），最后将对象 A提交给 K8s 创建；
3. 如果查询到 K8s 中已有这个资源，假设为对象 B：
	1. kubectl 尝试从对象 B 的 annotation 中取出 `kubectl.kubernetes.io/last-applied-configuration` 的值（对应了上一次 apply 提交的内容）；
	2. kubectl 根据前一次 apply 的内容和本次 apply 的内容计算出 diff（默认为 strategic merge patch 格式，如果非原生资源则采用 merge patch）；
	3. 将 diff 中添加本次的 `kubectl.kubernetes.io/last-applied-configuration` annotation，最后用 patch 请求提交给 K8s 做更新。

这里只是一个大致的流程梳理，真实的逻辑会更复杂一些，而从 K8s 1.14 之后也支持了 [server-side apply](https://kubernetes.io/docs/reference/using-api/server-side-apply/)，字面地理解为将 kubectl apply的工作（read + update）迁移到 Server 端来进行，此外Server-Side Apply 利用 managedFields 字段，追踪了各个字段的归属，并且能在更新的时候提供冲突检测(manager不匹配)，并提供更为准确的提示。

### edit

kubectl edit 逻辑上更简单一些。在用户执行命令之后，kubectl 从 K8s 中查到当前的资源对象，并打开一个命令行编辑器（默认用 vi）为用户提供编辑界面。当用户修改完成、保存退出时，kubectl 并非直接把修改后的对象提交 update（避免 Conflict，如果用户修改的过程中资源对象又被更新），而是会把修改后的对象和初始拿到的对象计算 diff，最后将 diff 内容用 patch 请求提交给 K8s。