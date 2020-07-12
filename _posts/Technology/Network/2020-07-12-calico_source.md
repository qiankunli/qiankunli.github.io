---

layout: post
title: calico源码分析
category: 技术
tags: Network
keywords: Docker, calico

---

## 简介

* TOC
{:toc}

![](/public/upload/network/calico_route.png)

##  calico node

每个节点启动了一个名为calico-node 的daemonset（namespace=kube-system），calico-node 容器以runit 作为进程管理工具，运行多个进程

```
/etc/service/enabled
    /bird
    /bird6
        /run
        /supervise
    /confd
        /run
        /supervise
    /felix
        /run
        /supervise
```
1. bird, calico 中的 Bird是一个BGP client，它会读取host上的路由信息（由felix写入），然后通过BGP协议广播出去
2. confd, confd根据etcd上状态信息，与本地模板，生成并更新BIRD配置
3. felix,  [Felix](https://docs.projectcalico.org/reference/architecture/overview)
    1. Interface management, Felix programs some information about interfaces into the kernel in order to get the kernel to correctly handle the traffic emitted by that endpoint. In particular, it will ensure that the host responds to ARP requests from each workload with the MAC of the host, and will enable IP forwarding for interfaces that it manages.It also monitors for interfaces to appear and disappear so that it can ensure that the programming for those interfaces is applied at the appropriate time.
    2. Route programming, Felix is responsible for programming routes to the endpoints on its host into the Linux kernel FIB (Forwarding Information Base) . This ensures that packets destined for those endpoints that arrive on at the host are forwarded accordingly.
    3. ACL programming, Felix is also responsible for programming ACLs into the Linux kernel. These ACLs are used to ensure that only valid traffic can be sent between endpoints, and ensure that endpoints are not capable of circumventing Calico’s security measures.
    4. State reporting, Felix is responsible for providing data about the health of the network. In particular, it reports errors and problems with configuring its host. This data is written into etcd, to make it visible to other components and operators of the network.

`/etc/service/enabled/felix/run` 
```sh
#!/bin/sh
exec 2>&1
# Felix doesn't understand NODENAME, but the container exports it as a common
# interface. This ensures Felix gets the right name for the node.
if [ ! -z $NODENAME ]; then
    export FELIX_FELIXHOSTNAME=$NODENAME
fi
export FELIX_ETCDADDR=$ETCD_AUTHORITY
export FELIX_ETCDENDPOINTS=$ETCD_ENDPOINTS
export FELIX_ETCDSCHEME=$ETCD_SCHEME
export FELIX_ETCDCAFILE=$ETCD_CA_CERT_FILE
export FELIX_ETCDKEYFILE=$ETCD_KEY_FILE
export FELIX_ETCDCERTFILE=$ETCD_CERT_FILE
# Felix hangs if DATASTORETYPE is empty: see projectcalico/felix issue #1156.
if [ ! -z $DATASTORE_TYPE ]; then
    export FELIX_DATASTORETYPE=$DATASTORE_TYPE
fi
exec calico-node -felix
```

`/etc/service/enabled/confd/run` 
```sh
#!/bin/sh
exec 2>&1
exec calico-node -confd
```

`/etc/service/enabled/bird6/run` 
```sh
#!/bin/sh
exec 2>&1
exec bird6 -R -s /var/run/calico/bird6.ctl -d -c /etc/calico/confd/config/bird6.cfg
```


```go
// github.com/projectcalico/node/cmd/calico-node/main.go
func main() {
    ...
    if *version {
		fmt.Println(startup.VERSION)
		os.Exit(0)
	} else if *runFelix {
		logrus.SetFormatter(&logutils.Formatter{Component: "felix"})
		felix.Run("/etc/calico/felix.cfg", buildinfo.GitVersion, buildinfo.GitRevision, buildinfo.BuildDate)
	} else if *runConfd {
		logrus.SetFormatter(&logutils.Formatter{Component: "confd"})
		cfg, err := confdConfig.InitConfig(true)
		if err != nil {
			panic(err)
		}
		cfg.ConfDir = "/etc/calico/confd"
		cfg.KeepStageFile = *confdKeep
		cfg.Onetime = *confdRunOnce
		confd.Run(cfg)
    }
    ...
}
```

## felix 

![](/public/upload/network/calico_felix.png)

1. felix 需要与k8s apiserver 通信获取pod 数据，若是集群节点太多，会给apiserver 带来较大负担，所以calico 提供typha 机制，缓存apiserver 数据，减少felix 对apiserver的压力。 
2. [calico node felix源码分析之二](https://blog.csdn.net/zhonglinzhang/article/details/97660972)Syncer 协程负责监听 datastore 中的更新，并将更新的内容通过 channel 发送给 Validator 协程。Validator 完成校验后，将其发送给 Calc graph 协程。Calc graph 完成计算后，发送给dataplane协程。最后dataplane完成数据平面处理。
3. felix 主要的工作组件就是syncer, CalcGraph, dataplane
    1. syncer, calicoctl 可以直接向datastore crud 一系列Resource，参见[Resource definitions](https://docs.projectcalico.org/reference/resources/overview)，代码定义在`github.com/projectcalico/libcalico-go/lib/backend/model`。syncer 同步且监听这些Resource，当资源变动时，通过回调onUpdate 通知下游组件（比如CalcGraph）。 
    2. CalcGraph, syncer 弄过来的 datastore的数据 通常不能直接使用，需要CalcGraph 做一些计算和归并 再交给dataplane 
    3. dataplane, 负责真正对 node 做出处理， 分为 本地和远程（使用grpc 通信）两种形态，为此定义了一个 proto，`github.com/projectcalico/felix/proto/felixbackend.pb.go` ，如果是本地运行，则通过channel 直接传输 proto model，如果是dataplane 远程独立运行，则执行grpc 调用。InternalDataplane implements an in-process Felix dataplane driver based on **iptables and ipsets**.  It communicates with the datastore-facing part of Felix via the Send/RecvMessage methods, which operate on the protobuf-defined API objects.

dataplane 一方面通过ifaceMonitor 监听网卡，发出 ifaceUpdates, ifaceAddrUpdates 事件；另一方面接收 来自syncer/CalcGraph 的proto model。 dataplane 聚合一系列 Manager(felix 剩下的大部分代码 都是按manager 分包的)， 监听上述两类事件变化并执行。 

![](/public/upload/network/calico_felix_object.png)

`github.com/projectcalico/felix/daemon/daemon.go`

```go
// Run is the entry point to run a Felix instance.
//
// Its main role is to sequence Felix's startup by:
//
// Initialising early logging config (log format and early debug settings).
//
// Parsing command line parameters.
//
// Loading datastore configuration from the environment or config file.
//
// Loading more configuration from the datastore (this is retried until success).
//
// Starting the configured internal (golang) or external dataplane driver.
//
// Starting the background processing goroutines, which load and keep in sync with the
// state from the datastore, the "calculation graph".
//
// Starting the usage reporting and prometheus metrics endpoint threads (if configured).
//
// Then, it defers to monitorAndManageShutdown(), which blocks until one of the components
// fails, then attempts a graceful shutdown.  At that point, all the processing is in
// background goroutines.
//
// To avoid having to maintain rarely-used code paths, Felix handles updates to its
// main config parameters by exiting and allowing itself to be restarted by the init
// daemon.
func Run(configFile string, gitVersion string, buildDate string, gitRevision string) {
    ...   
}
```




