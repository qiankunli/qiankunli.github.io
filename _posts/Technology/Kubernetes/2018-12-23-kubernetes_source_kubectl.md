---

layout: post
title: Kubernetes源码分析——从kubectl开始
category: 技术
tags: Kubernetes
keywords: kubernetes 源码分析 kubectl

---

## 简介

* TOC
{:toc}

从API Server看，kubectl 其实是高级定制版的 curl 工具

![](/public/upload/kubernetes/kubectl_overview.png)

## 从kubectl 开始

[A Tour of the Kubernetes Source Code Part One: From kubectl to API Server](https://developer.ibm.com/opentech/2017/06/21/tour-kubernetes-source-code-part-one-kubectl-api-server/) 

以`kubectl create -f xx.yaml` 为例，你会在`pkg/kubectl/cmd/create` 下找到一个create.go。类似的，所有kubectl 的命令 都可以在 `pkg/kubectl/cmd` 下找到。kubectl 命令行库 用的是 [spf13/cobra](https://github.com/spf13/cobra)


**可以看到，阅读Kubernetes 源码需要一些go 语言基础、常用库、go上的设计模式等沉淀，否则会将技术细节与kubernetes 思想混在在一起**

### package结构

[Go 常用的一些库](http://qiankunli.github.io/2015/05/31/go_library.html) 在介绍 [spf13/cobra](https://github.com/spf13/cobra)会提到 一个command line application 的推荐结构

  	appName/
    	cmd/
        	add.go
        	your.go
        	commands.go
        	here.go
      	main.go
      	
但我们要注意几个问题

1. 其实大部分golang 应用 都可以以command line application 的结构来组织
2. k8s 是由 多个command line application 组成的集合，所以其package 结构又有一点不一样。`pkg/kubectl` 是根据 cobra 推荐样式来的，main.go 的角色由`pkg/kubectl/cmd.go` 来承担（cmd.go 的方法被`cmd/kubectl/kubectl.go` 引用），kubectl 具体命令实现在 `pkg/kubectl/cmd` 下

## Builders and Visitors 

### 背景

1. 而kubectl本身并不包含对其核心资源的访问与控制，而是通过http通信与api-server进行交互实现资源的管理，**所以kubectl 操作的最后落脚点 是发送 http 请求**
2. kubectl 读取用户输入（包括参数、yaml 文件、yaml http地址）后，肯定要在内部用一个数据结构来表示，然后针对这个数据结构 发送http 请求。
3. 实际实现中，数据结构有倒是有，但是

	1. k8s 笼统的称之为 resource，但没有一个实际的resource 对象存在。
	2. resource 可能有多个，因为 `kubectl create -f` 可以创建多个文件，便有了多个resource
	3. resource 有多个处理步骤，比如对于 `kubectl create -f http://xxx` 要先下载yaml 文件、校验、再发送http 请求

4. 针对这几个问题

	* k8s 没有使用一个 类似Resources 的对象来聚合所有 resource
	* 针对一个resource 的多个处理步骤，k8s 也没有为resource 提供download、sendHttp、downloadAndSendHttp 之类的方法，**而是通过对象的 聚合来 代替方法的顺序调用**。

所以，k8s 采用了访问者 模式，利用其 **动态双分派** 特性，参见[函数式编程的设计模式](http://qiankunli.github.io/2018/12/15/functional_programming_patterns.html) 

Visitor 接口

	type VisitorFunc func(*Info, error) error
	// Visitor lets clients walk a list of resources.
	type Visitor interface {
		Visit(VisitorFunc) error
	}


### visitor 模式

这部分代码主要在 `k8s.io/cli-runtime/pkg/genericclioptions/resource` 包下，可以使用goland 单独打开看

![](/public/upload/kubernetes/k8s_builder.png)


以`pkg/kubectl/cmd/create/create.go` 为demo

	// 根据 请求参数 构造 resource
	r := f.NewBuilder().
			Unstructured().
			Schema(schema).			// 简单赋值
			ContinueOnError().		// 简单赋值
			NamespaceParam(cmdNamespace).DefaultNamespace().	// 简单赋值
			FilenameParam(enforceNamespace, &o.FilenameOptions). // 文件可以是http 也可以 本地文件，最终都是为了给 builder.path 赋值，path 是一个Visitor 集合
			LabelSelectorParam(o.Selector).	// 简单赋值
			Flatten(). 	// 简单赋值
			Do()
	...
	// 发送http 请求
	err = r.Visit(func(info *resource.Info, err error) error {
		...
		if !o.DryRun {
			if err := createAndRefresh(info); err != nil {
				return cmdutil.AddSourceToErr("creating", info.Source, err)
			}
		}
		...
		return o.PrintObj(info.Object)
	})
		
		
1. Builder provides convenience functions for taking arguments and parameters from the command line and converting them to a list of resources to iterate over using the Visitor interface.
2. Result contains helper methods for dealing with the outcome of a Builder. Info contains temporary info to execute a REST call, or show the results of an already completed REST call.

![](/public/upload/kubernetes/k8s_kubectl_create.png)	

Builder  build 了什么东西？将输入转换为 resource，还不只一个。每个resource 都实现了 Visitor 接口，可以接受 VisitorFunc。Result 持有  Builder 的结果——多个实现visitor 接口的resource ，并提供对它们的快捷 访问操作。

在Result 层面，一个resource/visitor 一般代表一个yaml 文件（无论local 还是http），一个resource 内部 是多个visitor 的纵向聚合，比如 URLVisitor.visit


	type URLVisitor struct {
		URL *url.URL
		*StreamVisitor
		HttpAttemptCount int
	}
	
	func (v *URLVisitor) Visit(fn VisitorFunc) error {
		body, err := readHttpWithRetries(httpgetImpl, time.Second, v.URL.String(), v.HttpAttemptCount)
		if err != nil {
			return err
		}
		defer body.Close()
		v.StreamVisitor.Reader = body
		return v.StreamVisitor.Visit(fn)
	}
	
URLVisitor.visit 自己实现了 读取http yaml 文件内容，然后通过StreamVisitor.Visit 负责后续的发送http 逻辑。
	
### visitor 函数聚合

`k8s.io/cli-runtime/pkg/genericclioptions/resource/interfaces.go`

这里面比较有意思的 是  DecoratedVisitor (纵向聚合，扩充一个visitor的 visit 逻辑)和 VisitorList（横向聚合，将多个平级的visitor 聚合为1个）

	type DecoratedVisitor struct {
		visitor    Visitor
		decorators []VisitorFunc
	}
	func (v DecoratedVisitor) Visit(fn VisitorFunc) error {
		return v.visitor.Visit(func(info *Info, err error) error {
			if err != nil {
				return err
			}
			for i := range v.decorators {
				if err := v.decorators[i](info, nil); err != nil {
					return err
				}
			}
			return fn(info, nil)
		})
	}

DecoratedVisitor 封装了一个visitor， DecoratedVisitor.Visit  执行时 会先让	visitor 执行DecoratedVisitor 自己的私货VisitorFunc ，然后再执行 传入的VisitorFunc，相当于对函数逻辑做了一个聚合。 

	type VisitorList []Visitor

	// Visit implements Visitor
	func (l VisitorList) Visit(fn VisitorFunc) error {
		for i := range l {
			if err := l[i].Visit(fn); err != nil {
				return err
			}
		}
		return nil
	}

VisitorList 则是将多个 Visitor 聚合成一个visitor，VisitorList.visit 执行时会依次执行 其包含的visitor的Visit 逻辑

### 回头看

整这么复杂 `k8s.io/cli-runtime/pkg/genericclioptions/resource` 

1. 对上层提供两个 抽象Builder 和 Result，上层的调用模式很固定

	1. 构造Builder、result
	2. result.visit
2. 尽可能在通用层面 实现了kubectl 的所有逻辑，上层通过配置、传入function 即可个性化整体流程

其实最初看完这个实现，笔者在质疑这样做是否有必要，因为**kubectl 就是一个发送http 请求的工具（http client）**。从常规的实现角度看，create/deploy 等操作各干个的，然后共用一些抽象（比如pod）、工具类（比如HttpUtils） 就可以了。

![](/public/upload/kubernetes/k8s_kubectl_impl_1.png)

可以看到这个复用层次是比较浅的

![](/public/upload/kubernetes/k8s_kubectl_impl_2.png)

1. k8s 将一些公共组件单独提取出来，作为一个库，有点layer的感觉，但还不完全是。
2. 借助visitor 模式，完全依靠 函数聚合来聚合逻辑，或许要从函数式编程模式的一些角度来找找感觉。




