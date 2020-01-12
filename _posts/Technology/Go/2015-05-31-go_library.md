---

layout: post
title: Go 常用的一些库
category: 技术
tags: Go
keywords: Go

---

## 一 前言

* TOC
{:toc}

本文主要阐述一下golang中常用的库。

## 如何组织一个大项目的go 代码

![](/public/upload/go/go_module.png)

### 项目代码划分

[使用 Go 语言开发的一些经验（含代码示例）](https://mp.weixin.qq.com/s?__biz=MjM5MDE0Mjc4MA==&mid=2651008064&idx=2&sn=cdc19d0db8decad85b671ba79fd2d1f5&chksm=bdbed4138ac95d05dbfd6672babba8e4d4a547d7845cd46b23fe3802dd5a1c49777b476fadd5&mpshare=1&scene=23&srcid=0708wchJyw4BGm9vtQxV8qaT%23rd) 要点如下

1. 可见性和代码划分

	* c++ 在类上，即哪怕在同一个代码文件中，仍然无法访问一个类的私有方法
	* java 是 类 + 包名
	* go 在包上，从其他包中引入的常量、变量、函数、结构体以及接口，都需要加上包的前缀来进行引用。Golang 也可以 dot import 来去掉这个前缀。不幸的是，这个做法并不常规，并且不被建议。

1. 假设有一个用户信息管理系统，直观感觉上的分包方式

	* 单一package
	* 按mvc划分，比如controller包、model包，缺点就是你使用 controller类时 就只得`controller.UserController`,controller 重复了
	* 按模块划分。比如`user/UserControler.go,user/User.go`，缺点就是使用User类时只得 `user.User`

2. 按依赖划分，即根包下 定义接口文件`servier.go`，包含User和UserController 接口定义，然后定义`postgresql/UserService.go` 或者`mysql/UserService.go`

github 也有一些demo 项目layout [golang-standards/project-layout](https://github.com/golang-standards/project-layout)

[作为一名Java程序员，我为什么不在生产项目中转向Go](http://www.infoq.com/cn/articles/why-not-go)

并发中处理的内容才是关键，新启一个线程或者协程才是万里长城的第一步，如果其中的业务逻辑有10个分支，还要多次访问数据库并调用远程服务，那无论用什么语言都白搭。所以在业务逻辑复杂的情况下，语言的差异并不会太明显，至少在Java和Go的对比下不明显	
[Organizing Go source code part 2](http://neurocline.github.io/dev/2016/02/01/organizing-go-source-code.html) 未读


## Go 依赖包管理

[Go的包管理工具（一）](https://juejin.im/post/5c6ac37cf265da2de7134242)

[Go的包管理工具（二）：glide](https://juejin.im/post/5c769eae6fb9a049d05e682b)

[Go的包管理工具（三）：Go Modules](https://juejin.im/post/5c7fc2b1f265da2dac4575fc)

[官方对比](https://github.com/golang/go/wiki/PackageManagementTools)

||版本||
|---|---|---|
|vendor机制|1.5发布，1.6默认启用，1.7去掉环境变量设置默认开启||
|govendor|1.5以后可用|基于 vendor 目录机制的包管理工具|
|godep|1.5之前可以用，1.6依赖vendor||
|Go Modules|1.11 发布，1.12 增强，1.13正式默认开启||

### 最早的GOPATH

对于go来说，其实并不在意你的代码是内部还是外部的，总之都在GOPATH里，任何import包的路径都是从GOPATH开始的；唯一的区别，就是内部依赖的包是开发者自己写的，外部依赖的包是go get下来的。Go 语言原生包管理的缺陷：

1. 能拉取源码的平台很有限，绝大多数依赖的是 github.com
2. 不能区分版本，以至于令开发者以最后一项包名作为版本划分
3. 依赖 列表/关系 无法持久化到本地，需要找出所有依赖包然后一个个 go get
4. 只能依赖本地全局仓库（GOPATH/GOROOT），无法将库放置于局部仓库（$PROJECT_HOME/vendor）

### vendor

vendor属性就是让go编译时，优先从项目源码树根目录下的vendor目录查找代码(可以理解为切了一次GOPATH)，如果vendor中有，则不再去GOPATH中去查找。

通过如上vendor解决了部分问题，然而又引起了新的问题：

1. vendor目录中依赖包没有版本信息。这样依赖包脱离了版本管理，对于升级、问题追溯，会有点困难。
2. 如何方便的得到本项目依赖了哪些包，并方便的将其拷贝到vendor目录下？依靠人工实在不现实。

### godep/govendor

在支持vendor机制之后， gopher 们把注意力都集中在如何利用 vendor 解决包依赖问题，从手工添加依赖到 vendor、手工更新依赖，到一众包依赖管理工具的诞生，比如：govendor、glide 以及号称准官方工具的 dep，努力地尝试着按照当今主流思路解决着诸如 “钻石型依赖” 等难题。

`godep go build main.go` godep中的go命令，就是将原先的go命令加了一层壳，执行godep go的时候，会将当前项目的workspace目录加入GOPATH变量中。

`godep save`命令将会自动扫描当前目录所属包中import的所有外部依赖库（非系统库），并将所有的依赖库下来下来到当前工程中，产生文件 `Godeps/Godeps.json` 文件。把所有依赖包代码从GOPATH路径拷贝到Godeps目录下(vendor推出后也改用vendor了)

`govendor init`生成vendor/vendor.json

`govendor add +external`更新vendor/vendor.json，并拷贝GOPATH下的代码到vendor目录中。

**vendor机制有一个问题**：同样的库，同样的版本，就因为在不同的工程里用了，就要在vendor里单独搞一份，不浪费吗？所有这些基于vendor的包管理工具，都会有这个问题。

### Go Modules 一统天下

Go Modules 提供了统一的依赖包管理工具 go mod，其思想类似maven：摒弃vendor和GOPATH，拥抱本地库。

1. 依赖包统一收集在 `$GOPATH/pkg/mod` 中进行集中管理。有点mvn `.m2` 文件夹的意思
2. `$GOPATH/pkg/mod` 中的按版本管理

        $GOPATH/pkg/mod/k8s.io
            api@v0.17.0
            client-go@v0.17.0
            kube-openapi@v0.0.0-20191107075043-30be4d16710a

3. go build 代码时，自动解析代码中的import 生成 go.mod 文件。PS：相当于`maven package` 解析代码自动生成pom.xml。
4. 将 import 路径与项目代码的实际存放路径解耦。PS：import 依赖包在$GOPATH 中的路径，编译时使用依赖包在`$GOPATH/pkg/mod`中的文件。

## Go代码中的依赖注入

[Go中的依赖注入](https://www.jianshu.com/p/cb3682ad34a7) 推荐使用 [uber-go/dig](https://github.com/uber-go/dig) 
A reflection based dependency injection toolkit for Go.

依赖注入是你的组件（比如go语言中的structs）在创建时应该接收它的依赖关系。PS：这个理念在java、spring 已经普及多年。这与在初始化期间构建其自己的依赖关系的组件的相关反模式相反。

**设计模式分为创建、结构和行为三大类，如果自己构造依赖关系， 则创建 与 行为 两个目的的代码容易耦合在一起， 代码较长，给理解造成困难。**

## 三 json字符串与结构体的转换

Go中的json处理，跟结构体是密切相关的，一般要为json字符串建好相对应的struct。

如果只是想获取json串中某个key的值，可以使用`github.com/bitly/go-simplejson`

    js, err := simplejson.NewJson([]byte(inputStr))
	path := js.Get("path").MustString()
	
如果知道json字符串对应结构体（该结构体可能会嵌套），则可以使用golang自带的encoding/json包。

    type name struct {
    	FN string `json:"fn"`
    	LN string `json:"ln"`
    }
    type user struct {
    	Id       int64  `json:"id"`            // 指定转换为字符串后，该字段显示的key值
    	Username string `json:"username"`
    	Password string `json:"-"`            
    	N        name   `json:"name"`
    }
    func main() {
    	n := name{
    		FN: "li",
    		LN: "qk",
    	}
    	u := user{
    		Id:       1,
    		Username: "lqk",
    		Password: "123456",
    		N:        n,
    	}
    	// 根据结构体生成字符串
    	b, _ := json.Marshal(u)
    	
        jsonStr := string(b)
    	fmt.Println(jsonStr)
    	
        var u2 user
        // 根据字符串生成结构体
        err := json.Unmarshal([]byte(jsonStr), &u2)
    }

## 读写锁

    package main
    import (
    	"errors"
    	"fmt"
    	"sync"
    )
    var (
    	pcodes         = make(map[string]string)
    	mutex          sync.RWMutex
    	ErrKeyNotFound = errors.New("Key not found in cache")
    )
    func Add(address, postcode string) {
        // 写入的时候要完全上锁
    	mutex.Lock()
    	pcodes[address] = postcode
    	mutex.Unlock()
    }
    func Value(address string) (string, error) {
        // 读取的时候，只用读锁就可以
    	mutex.RLock()
    	pcode, ok := pcodes[address]
    	mutex.RUnlock()
    	if !ok {
    		return "", ErrKeyNotFound
    	}
    	return pcode, nil
    }
    func main() {
    	Add("henan", "453600")
    	v, err := Value("henan")
    	if err == nil {
    		fmt.Println(v)
    	}
    }
    
## command line application

go 可执行文件没有复杂的依赖（java依赖jvm、python 依赖python库），特别适合做一些命令行工具

大概的套路都是

1. 定义一个Command对象
2. Command 对象一般有一个 name，多个flag（全写和简写） 以及一个处理函数

### [urfave/cli](https://github.com/urfave/cli)

cli is a simple, fast, and fun package for building command line apps in Go. The goal is to enable developers to write fast and distributable command line applications in an expressive way.

Things like generating help text and parsing command flags/options should not hinder productivity when writing a command line app.This is where cli comes into play. cli makes command line programming fun, organized, and expressive!

### [spf13/cobra](https://github.com/spf13/cobra) 

这个库牛就牛在k8s 用的也是它

The best applications will read like sentences when used(命令执行起来应该像句子一样). Users will know how to use the application because they will natively understand how to use it.

The pattern to follow is `APPNAME VERB NOUN --ADJECTIVE`. or `APPNAME COMMAND ARG --FLAG`

A flag is a way to modify the behavior of a command 这句说的很有感觉


## http

go语言中的`github.com/gorilla`可以方便的进行http url 到处理方法的dispatch，`github.com/urfave/cli` 则实现了用户输入命令到处理方法的dispatch。