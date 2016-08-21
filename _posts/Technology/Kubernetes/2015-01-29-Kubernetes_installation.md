---
layout: post
title: 在CoreOS集群上搭建Kubernetes
category: 技术
tags: Kubernetes
keywords: CoreOS Docker Kubernetes
---

## 简介

原文地址：[How To Install and Configure Kubernetes on top of a CoreOS Cluster][]，有删减。有的地方没有翻译，或者因为不太确定，或者因为翻译成中文后就没有感觉了。

Kubernetes 可以在集群环境下，管理以Docker containers为基础的应用。它可以管理一个容器化应用的整个生命周期，包括部署和扩展。

在这篇文章中，我们将讨论如何在一个CoreOS集群上搭建Kubernetes。它将一组彼此关联的服务作为一个单元（Kubernetes称为“pods”）部署在一台主机上，并提供health checking（如果翻译出来很不直观）,高可用性和提高资源利用率。

Kubernetes版本更新很快，所以不能保证该文档长期有效。

## 前期准备

你需要准备一个CoreOS集群，可以参见[CoreOS clustering guide][]

现在我们拥有了一个包含3个节点的CoreOS集群，从Kubernetes的角度，这三个节点需要担当不同的角色，我们需要一个节点作为master，master需要运行一些额外的服务，例如API server and a controller manager，详情如下：

    Hostname	Public IPv4	Private IPv4	Role
    coreos-1	192.168.2.1	10.120.0.1	Master
    coreos-2	192.168.2.2	10.120.0.2	Minion1
    coreos-3	192.168.2.3	10.120.0.3	Minion2

master同时也是一个minion。每一个节点的etcd和fleet按Private IPv4配置。

为了搭建Kubernetes，首先我们需要配置flannel，a network fabric layer that provides each machine with an individual subnet for container communication.是 CoreOS 团队针对 Kubernetes 设计的一个覆盖网络 (overlay network) 工具。

我们将配置docker使用它的网络模型，在此基础上来搭建Kubernetes。它涉及到多个组件，包括proxy 服务,API层，和一个node-level的“pod”管理系统kubelect。

## 编译flannel

首先，我们需要配置flannel服务，这个组件为集群中的每个节点提供独立的子网（替换docker默认的网络模型），因为这是一个基础前提，所以先做这个。

在写这篇文章时，flannel还没有编译好的二进制版本。所以我们需要自己编译并安装它，为了节省编译时间，我们在一台节点上编译好，然后将其拷贝到其它节点上。

像Coreos的其它组件一样，Flannel由Go语言编写。搭建一个完整的Go环境太麻烦，我们使用容器来编译它（flannel源代码），google为类似的场景维护了一个GO container。

我们即将安装的所有应用都放在`/opt/bin`目录下，现在创建这个目录：

    sudo mkdir -p /opt/bin
    
现在使用Go container来编译flannel，拉取镜像并运行，调用Go命令下载代码并编译。

    docker run -i -t google/golang /bin/bash -c "go get github.com/coreos/flannel"
    
当这个操作完成，我们将生成的可执行文件从容器中拷贝出来。首先我们需要知道containerId：
   
    docker ps -l -q
    
结果如下：

    004e7a7e4b70
    
生成文件在容器的`/gopath/bin/flannel`目录下，根据containerId，我们使用如下命令将其拷贝到coreos的`/opt/bin`目录下：

    sudo docker cp 004e7a7e4b70:/gopath/bin/flannel /opt/bin/
    
现在，我们可以在节点上使用flannle了，稍后，我们将其拷贝到其它节点上。

## 编译Kubernetes 
Kubernetes 由一些不同的应用组件组成。目前还没有现成的编译好的可执行文件，我们将自己编译。

首先从github上将项目文件拷贝下来

    cd ~
    git clone https://github.com/GoogleCloudPlatform/kubernetes.git
    
接下来，运行其自带的编译脚本：
    
    cd kubernetes/build
    ./release.sh
    
这个脚本会启动一个docker container来编译项目文件（喔喔，container的用处之一），这个过程将持续很长时间。

编译结束后，在`~/kubernetes/_output/dockerized/bin/linux/amd64`查看生成文件，并将其复制到`/opt/bin`目录下。

    cd ~/kubernetes/_output/dockerized/bin/linux/amd64
    ls
    e2e          kube-apiserver           kube-proxy      kubecfg  kubelet
    integration  kube-controller-manager  kube-scheduler  kubectl  kubernetes
    sudo cp * /opt/bin
    
现在我们需要的所有文件都已生成，将其复制到其它节点上。

友情提示：这一步编译很不易，不如自己直接到人家的编译好的地方下载：

    wget https://storage.googleapis.com/kubernetes/binaries.tar.gz
    
## 构建 Master-Specific 服务

接下来的任务是通过创建systemd unit文件来配置和运行Kubernetes ，所有的unit文件存放在`/etc/systemd/system`

master端需要配置：

- apiserver.service
- controller-manager.service
- scheduler.service

Minion端需要配置：

- flannel.service
- docker.service
- proxy.service
- kubelet.service

### 创建APIServer unit 文件

API Server用来管理集群信息，处理请求，同步共享信息和调度任务。我们将其unit 文件命名为`apiserver.service`。

    [Unit]
    Description=Kubernetes API Server
    After=etcd.service
    After=docker.service
    Wants=etcd.service
    Wants=docker.service
    
    [Service]
    ExecStart=/opt/bin/kube-apiserver \
    -address=127.0.0.1 \
    -port=8080 \
    -etcd_servers=http://127.0.0.1:4001 \
    -machines=10.120.0.1,10.120.0.2,10.120.0.3
    -logtostderr=true
    ExecStartPost=-/bin/bash -c "until /usr/bin/curl http://127.0.0.1:8080; do echo \"waiting for API server to come online...\"; sleep 3; done"
    Restart=on-failure
    RestartSec=5
    
    [Install]
    WantedBy=multi-user.target

首先，对于unit小节，填写描述信息，并确保该service运行时etcd和docker已启动完毕。

在service小节，设置了api servier运行的address和port，监听的etcd的地址。启动service后，通过一个循环来确认其是否启动成功。否则依赖该service的的service可能无法正确启动。

在Install小节，配置其开机启动。

### 创建Controller Manager unit 文件

这个组件用来在集群中执行data replication，我们命名其配置文件为`controller-manager.service`

    [Unit]
    Description=Kubernetes Controller Manager
    After=etcd.service
    After=docker.service
    After=apiserver.service
    Wants=etcd.service
    Wants=docker.service
    Wants=apiserver.service
    
    [Service]
    ExecStart=/opt/bin/kube-controller-manager \
    -master=http://127.0.0.1:8080 \
    Restart=on-failure
    RestartSec=5
    
    [Install]
    WantedBy=multi-user.target
    
在unit小节，确保其启动前，etcd，docker和api server已正确启动。

对于service小节，我们配置api server的地址，并确保服务down掉时自动重启。最后，让服务开机启动。

### 创建scheduler unit 文件

scheduler 为pod选择在哪一台机器上运行，并确认其运行成功。其配置文件命名为“scheduler.service”。

  
    [Unit]
    Description=Kubernetes Scheduler
    After=etcd.service
    After=docker.service
    After=apiserver.service
    Wants=etcd.service
    Wants=docker.service
    Wants=apiserver.service
    
    [Service]
    ExecStart=/opt/bin/kube-scheduler -master=127.0.0.1:8080
    Restart=on-failure
    RestartSec=5
    
    [Install]
    WantedBy=multi-user.target

service小节的配置非常直接，只需配置api service的运行ip和port就可以了。

## 创建Cluster Services

现在master独有的服务已经配置好了，接下来将的讲的配置文件在所有节点上都会被用到，包括：

- flannel.service
- docker.service
- proxy.service
- kubelet.service

### Create the Flannel Unit File

fannel工作在网络结构层，This will be used to give each node its own subnet for Docker containers。文件名为“flannel.service”


    [Unit]
    Description=Flannel network fabric for CoreOS
    Requires=etcd.service
    After=etcd.service
    
    [Service]
    EnvironmentFile=/etc/environment
    ExecStartPre=-/bin/bash -c "until /usr/bin/etcdctl set /coreos.com/network/config '{\"Network\": \"10.100.0.0/16\"}'; do echo \"waiting for etcd to become available...\"; sleep 5; done"
    ExecStart=/opt/bin/flannel -iface=${COREOS_PRIVATE_IPV4}
    ExecStartPost=-/bin/bash -c "until [ -e /run/flannel/subnet.env ]; do echo \"waiting for write.\"; sleep 3; done"
    Restart=on-failure
    RestartSec=5
    
    [Install]
    WantedBy=multi-user.target

在unit小节，flannel需要向etcd注册子网信息。

在service小节，我们首先`source /etc/environment`，这样我们就可以访问节点的IP地址。（`/etc/environment`文件含有本节点的PUBLIC_IPV4和PRIVATE_IPV4信息）

接下来的一个`ExecStartPre=`，试图向etcd注册子网信息，知道注册成功。这个guide中，我们使用的子网是`10.100.0.0/16`。

接下来根据从`/etc/environment`拿到的PRIVATE_IPV4信息启动flannel。

最后，我们判断flannel是否将其信息写入到了文件中`/run/flannel/subnet.env`（以便一会儿启动docker时读取它），如果没有，就循环检测。这样，确保在文件可用之前，docker不会去读取它。

在install小节，配置其开机启动。

### Create the Docker Unit File

接下里我们要自定义docker的service文件，以使它在启动时能够采用flannel的网络配置。文件名为“docker.service”。

    [Unit]
    Description=Docker container engine configured to run with flannel
    Requires=flannel.service
    After=flannel.service
    
    [Service]
    EnvironmentFile=/run/flannel/subnet.env
    ExecStartPre=-/usr/bin/ip link set dev docker0 down
    ExecStartPre=-/usr/sbin/brctl delbr docker0
    ExecStart=/usr/lib/coreos/dockerd --daemon --host=fd:// $DOCKER_OPTS --bip=${FLANNEL_SUBNET} --mtu=${FLANNEL_MTU} $DOCKER_OPTS
    Restart=on-failure
    RestartSec=5
    
    [Install]
    WantedBy=multi-user.target

在unit小节，配置docker服务启动顺序在flannel之后。

在service小节，我们需要引入`/run/flannel/subnet.env`,flannel会将其创建的网络信息写入到这个文件中。

我们停止docker0网桥并删除它，当重启docker后，docker将采用新的网络模型。

在install小节，配置docker服务开机启动。

### Create the Proxy Unit File

The next logical unit to discuss is the proxy server that each of the cluster members runs. The Kubernetes proxy server is used to route and forward traffic to and from containers.

    vim proxy.service
    [Unit]
    Description=Kubernetes proxy server
    After=etcd.service
    After=docker.service
    Wants=etcd.service
    Wants=docker.service
    
    [Service]
    ExecStart=/opt/bin/kube-proxy -etcd_servers=http://127.0.0.1:4001 -logtostderr=true
    Restart=on-failure
    RestartSec=5
    
    [Install]
    WantedBy=multi-user.target
    
文件内容比较简单，不再赘述。

### Create the Kubelet Unit File

This component is used to manage container deployments. It ensures that the containers are in the state they are supposed to be in and monitors the system for changes in the desired state of the deployments.

    vim kubelet.service
    [Unit]
    Description=Kubernetes Kubelet
    After=etcd.service
    After=docker.service
    Wants=etcd.service
    Wants=docker.service
    
    [Service]
    EnvironmentFile=/etc/environment
    ExecStart=/opt/bin/kubelet \
    -address=${COREOS_PRIVATE_IPV4} \
    -port=10250 \
    -hostname_override=${COREOS_PRIVATE_IPV4} \
    -etcd_servers=http://127.0.0.1:4001 \
    -logtostderr=true
    Restart=on-failure
    RestartSec=5
    
    [Install]
    WantedBy=multi-user.target

在service小节，我们同样引入`/etc/environment`文件来获取主机的IP信息。我们启动kubelet，为其配置IP和PORT（以便让别的程序访问它），我们同样告诉它etcd的ip和端口。其它细节不再赘述。

### Enabling the Services

为了启动这些service，要先使其生效。

我们所有的配置文件都配置了"WantedBy" multi-user target，这意味着这些服务将会被开机启动。

    cd /etc/systemd/system
    sudo systemctl enable *

在所有的节点上，重复这些操作。然后重启所有节点。先重启master节点，当master节点重启成功后，再重启其它minion节点。

我们可以在master节点上查看所有节点：

    kubecfg list minions
    Minion identifier
    ----------
    10.120.0.1
    10.120.0.2
    10.120.0.3
    
在以后的guide中，我们将讨论在CoreOS集群中，如何控制和调度服务。

## 小结

现在，你应该可以在你的CoreOS集群中搭建Kubernetes了。上述搭建过程包含了太多的手工操作，其实也可以将这些过程写入到cloud-config中，但国内不能直接连外网，只能自己一步一步来了。

本文已授权在[dockerpool][]上发表，欢迎大家参与讨论！


[CoreOS clustering ]: https://www.digitalocean.com/community/tutorials/how-to-set-up-a-coreos-cluster-on-digitalocean
[CoreOS clustering guide]: https://www.digitalocean.com/community/tutorials/how-to-set-up-a-coreos-cluster-on-digitalocean
[How To Install and Configure Kubernetes on top of a CoreOS Cluster]: https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-kubernetes-on-top-of-a-coreos-cluster
[http://qiankunli.github.io/ ]: http://qiankunli.github.io/ 
[qiankun.li@qq.com]: qiankun.li@qq.com
[dockerpool]: http://www.dockerpool.com/