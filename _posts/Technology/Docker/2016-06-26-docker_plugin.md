---

layout: post
title: 通过Docker Plugin来扩展Docker Engine
category: 技术
tags: Docker
keywords: Docker,plugin

---

## 前言（未完待续）

笔者决定了解docker plugin，是因为笔者设计了一套基于Centos的跨主机容器通信方案。方案中，除了调用docker原生的api外，还有许多脚本。

此时，要对外简化这套网络模型的使用，有以下几种方法：

1. 类似于docker swarm，做一个类似“中间件”的东西。docker swarm对外提供docker api，自己封装了调度逻辑，调用其它docker engine的api。我也可以做一个类似docker centosnetwork之类的东西，封装网络配置逻辑。
2. 使用docker plugin。将新特性“添加到”docker engine中。`docker run`和`docker rm`能比以往做更多的工作（比如删除容器后，还能执行特定脚本）。


方案1中，虽然一个容器的创建和回收，都可以通过脚本来完成，但将其融入公司现有的场景还需要做大量的工作和适配，并且非常不优雅（比如要ssh远程命令）。从这个角度而言，通过docker plugin可以将很多操作内置到docker engine中，增加功能的“切入点”在底层，提高了使用场景的适用性。

## docker plugin的类型

1. 授权，自定义插件接管Docker守护进程和其远程调用接口的认证和授权。
2. 卷驱动（待深入理解）
3. 网络驱动（待深入理解）
4. Ipam驱动，ip地址管理。 IPAM是Libnetwork的一个负责管理网络和终端IP地址分配的接口。 Ipam驱动在你需要引入自定义容器IP地址分配规则的时候非常有用。



## docker plugin的实现思路

在linux下，插件不是一个稀罕物，build一个应用时，可以指定某个参数，从而决定是否将某个特性加入到最终的可执行文件中。

**Docker插件是增强Docker引擎功能的进程外扩展**，新的插件运行在守护进程之外， 这意味着守护进程本身需要寻找一种合适的方式去和他们进行交互。 每个插件都内建了一个HTTP服务器，这个服务器会被守护进程所探测到，并且提供一系列的远程调用接口，通过HTTP POST方法来交换JSON化的信息。具体的说

1. 要有一个plugin discovery机制帮助docker engine发现docker plugin。假设自定义插件叫myplugin

    a. docker plugin和docker engine在一个主机
    
        docker engine会检查`/run/docker/plugins/myplugin.sock`
       
    b. docker plugin和docker engine不在一个主机
    
        
        docker engine会检查`/etc/docker/plugins`或者`/usr/lib/docker/plugins`目录下的`.spec`（一种纯文本文件，可以参见相关文档）和`.json`文件。比如
        
        {
            "Name": "myplugin",
            "Addr": "https://fntlnz.wtf/myplugin"
        } 

2. docker plugin和docker engine之间约定交互协议：包括激活接口和特定plugin类型的接口。

插件只需要提供指定接口的http服务就可以，那么插件就不用非得是go语言写的喽。

## go-plugins-helpers(待整理)

上面提到，插件要支持http服务，go语言有一个docker插件助手[docker/go-plugins-helpers][]，已经完成了http服务等大部分工作，只需要在指定接口上实现自己的逻辑即可。

`docker/go-plugins-helpers`包括authorization、ipam、network、volume、sdk包。sdk负责公共部分并提供基本结构，http请求数据与业务model的序列化和反序列化、服务端口的监听等（docker plugin要提供Http服务嘛），其它包负责某个业务块api的抽象，最终我们自己的包中实现一个docker plugin。

handler.go中用到了"net"和"net/http"包，分别负责提供tcp和http服务。在sdk包中提供的抽象是`HandleFunc(path string, fn func(w http.ResponseWriter, r *http.Request))`，具体到各个业务包，比如volume，则将抽象细化到只实现Driver接口就可以，请求与方法的对应关系已经被封装好了。

    type Driver interface {
    	Create(Request) Response
    	List(Request) Response
    	Get(Request) Response
    	Remove(Request) Response
    	Path(Request) Response
    	Mount(Request) Response
    	Unmount(Request) Response
    	Capabilities(Request) Response
    }
    
也就是说`docker/go-plugins-helpers`最终处理好了tcp/http服务的启动以及请求与请求处理函数的映射，我们只需实现其抽象好的接口即可。

## 引用

[使用插件扩展Docker][]

[使用插件扩展Docker]: http://dockone.io/article/1295
[docker/go-plugins-helpers]: https://github.com/docker/go-plugins-helpers