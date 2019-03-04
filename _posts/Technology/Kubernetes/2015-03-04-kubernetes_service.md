---
layout: post
title: Kubernetes service
category: 技术
tags: Kubernetes
keywords: Docker Kubernetes
---


## 简介

* TOC
{:toc}

本文主要来自对[https://cloud.google.com/container-engine/docs](https://cloud.google.com/container-engine/docs "")的摘抄，有删改。



2019.1.29补充：之前理解的服务对外部的访问，是服务级别的。而Kubernetes 是打算将pod 级别的对外访问也交给service（哪怕是mysql 主从之间的访问），因为ip 是变化的 以及 多个pod 负载均衡的需求，这也是kubernetes 不提倡ip 访问而提供的替换方案。但在复杂系统中，网络处理越简单越好。iptables是万恶之源，给实际工作中的运维排错带来极大的麻烦。在一些大厂的实践中，一般很少使用service 方案。

![](/public/upload/kubernetes/kubernetes_service_access.png)

## What is a service?

Container Engine pods are ephemeral（短暂的）. They can come and go over time, especially when driven by things like replication controllers. While each pod gets its own IP address, **those IP addresses cannot be relied upon to be stable over time（直接通过一个pod的ip来访问它是不可靠的）**. This leads to a problem: if some set of pods (let's call them backends) provides functionality to other pods (let's call them frontends) inside a cluster, how do those frontends find the backends?
(Pod组件的状态经常变化，也可能存在多个副本，那么其他组件如何来访问它呢)

Enter services.

A Container Engine service is an abstraction which defines a logical set of pods and a policy by which to access them. **The goal of services is to provide a bridge for non-Kubernetes-native applications to access backends without the need to write code that is specific to Kubernetes. A service offers clients an IP and port pair which, when accessed, redirects to the appropriate backends.(service会提供一个稳定的ip，作为桥梁，让其它pod访问。而service负责将请求转发到其对应的pod上)** The set of pods targeted is determined by a label selector.

![kubernete_service_model.png](/public/upload/kubernetes/kubernete_service_model.png)

As an example, consider an image-process backend which is running with 3 live replicas. Those replicas are fungible—frontends do not care which backend they use. While the actual pods that comprise the set may change, the frontend client(s) do not need to know that. The service abstraction enables this decoupling.

## How do they work?——基于iptables实现

Service 是由 kube-proxy 组件，加上 iptables 来共同实现的。

Every node in a Kubernetes cluster runs a kube-proxy. **This application watches the Kubernetes master for the addition and removal of Service and Endpoints objects. For each Service it opens a port (random) on the local node. Any connections made to that port will be proxied to one of the corresponding backend Pods.（这句非常关键）** Which backend to use is decided based on the AffinityPolicy of the Service. Lastly, it installs iptables rules which capture traffic to the Service's Port on the Service's portal IP and redirects that traffic to the previously described port.

The net result is that any traffic bound for the Service is proxied to an appropriate backend without the clients knowing anything about Kubernetes or Services or Pods.

When a pod is scheduled, the master adds a set of environment variables for each active service.

两台主机`192.168.56.101`和`192.168.56.102`，假设我运行一个apache2 pod和apache2 service，查看：

    $ kubectl get service
    NAME                LABELS                                    SELECTOR            IP                  PORT
    apache2-service     <none>                                    name=apache2        10.100.62.248       9090
    kubernetes          component=apiserver,provider=kubernetes   <none>              10.100.0.2          443
    kubernetes-ro       component=apiserver,provider=kubernetes   <none>              10.100.0.1          80
    
    $ kubectl get pods
    POD                 IP                  CONTAINER(S)        IMAGE(S)                     HOST                            LABELS              STATUS
    apache2-pod         10.100.83.5         apache2             docker-registry.sh/apache2   192.168.56.102/192.168.56.102   name=apache2        Running
    
Kubernetes内部如何访问这个Pod对应的Service？

1. 通过DNS
2. 环境变量

    apache pod的ip是`10.100.83.5`，pod的ip是不可靠的，所以其它pod要通过pod对应的service `10.100.62.248:9090`来访问这个pod。Kubernetes为minion节点添加iptables规则（将`10.100.62.248:9090`转发到一个任意端口），当minion收到来自`10.100.62.248:9090`请求时，将请求转发到任意端口，Kube-proxy进程再将其转发（根据一定的分发策略）到Pod上。
    
那么外界如何访问这个pod呢？我们可以看到这个pod被分配到了`192.168.56.102`这台主机上，查看这台主机的iptables。（`192.168.56.101`也为该service做了映射，此处不再赘述）
   
    $ sudo iptables -nvL -t nat 
    Chain KUBE-PORTALS-CONTAINER (1 references)
    pkts bytes target     prot opt in     out     source               destination
    0     0 REDIRECT   tcp  --  *      *       0.0.0.0/0            10.100.0.1           /* kubernetes-ro */ tcp dpt:80 redir ports 37483
    0     0 REDIRECT   tcp  --  *      *       0.0.0.0/0            10.100.0.2           /* kubernetes */ tcp dpt:443 redir ports 46593
    0     0 REDIRECT   tcp  --  *      *       0.0.0.0/0            10.100.62.248        /* apache2-service */ tcp dpt:9090 redir ports 36036
    
可以发现kube_proxy为其分配了36036端口，以后其它应用就可以通过`192.168.56.102  :36036`来访问`10.100.62.248:9090`，进而访问`10.100.83.5:80`

查看etcd：

    $ etcdctl get /registry/services/endpoints/default/apache2-service
    {"kind":"Endpoints","id":"apache2-service","uid":"09a711e5-c3d0-11e4-b06e-0800272b69e4","creationTimestamp":"2015-03-06T07:11:41Z","selfLink":"/api/v1beta1/endpoints/apache2-service?namespace=default","resourceVersion":14119,"apiVersion":"v1beta1","namespace":"default","endpoints":["10.100.83.5:80"]}
    
    $ etcdctl get /registry/services/specs/default/apache2-service
    {"kind":"Service","id":"apache2-service","uid":"07cc9702-c3d0-11e4-b06e-0800272b69e4","creationTimestamp":"2015-03-06T07:11:37Z","apiVersion":"v1beta1","namespace":"default","port":9090,"protocol":"TCP","selector":{"name":"apache2"},"containerPort":80,"portalIP":"10.100.62.248","sessionAffinity":"None"}
    

可以看到，一些相关信息都会被记录到etcd中。

Service 的 VIP 只是一条 iptables 规则上的配置，并没有真正的网络设备，所以你 ping 这个地址，是不会有任何响应的。

A service, through its label selector(a key:value pair), can resolve to 0 or more pods. Over the life of a service, the set of pods which comprise that service can grow, shrink, or turn over completely. Clients will only see issues if they are actively using a backend when that backend is removed from the service (and even then, open connections will persist for some protocols).

### Service Operations

Services map a port on each cluster node to ports on one or more pods.The mapping uses a selector key:value pair in the service, and the labels property of pods. Any pods whose labels match the service selector are made accessible through the service's port.

#### Create a service

    $ kubectl create -f FILE
    
Where:

- -f FILE or --filename FILE is a relative path to a service configuration file in either JSON or YAML format.

A successful service create request returns the service name.

#### Service configuration file

When creating a service, you must point to a service configuration file as the value of the -f flag. The configuration file can be formatted as YAML or as JSON, and supports the following fields:

    {
      "id": string,
      "kind": "Service",
      "apiVersion": "v1beta1",
      "selector": {
        string: string
      },
      "containerPort": int,
      "protocol": string,
      "port": int,
      "createExternalLoadBalancer": bool
    }
    
Required fields are:

- id: The name of this service.
- kind: Always Service.
- apiVersion: Currently v1beta1.
- selector: The label key:value pair that defines the pods to target.
- containerPort The port to target on the pod.
- port: The port on the node instances to map to the containerPort.

Optional fields are:

- protocol: The Internet protocol to use when connecting to the container port. Must be TCP.
- createExternalLoadBalancer: If true, sets up Google Compute Engine network load balancing for your service. This provides an externally-accessible IP address that sends traffic to the correct port on your cluster nodes. To do this, a target pool is created that contains all nodes in the cluster. A forwarding rule defines a static IP address and maps it to the service's port on the target pool. Traffic is sent to clusters in the pool in round-robin order.

#### Sample files

The following service configuration files assume that you have a set of pods that expose port 9376 and carry the label app=example.

Both files create a new service named myapp which resolves to TCP port 9376 on any pod with the app=example label.

The difference in the files is in how the service is accessed. The first file does not create an external load balancer; the service can be accessed through port 8765 on any of the nodes' IP addresses.

    {
      "id": "myapp",
      "kind": "Service",
      "apiVersion": "v1beta1",
      "selector": {
        "app": "example"
      },
      "containerPort": 9376,
      "protocol": "TCP",
      "port": 8765
    }
（一个服务有多个pod，如果我们想对其进行调度的话，可以使用gce）
The second file uses Google Compute Engine network load balancing to create a single IP address that spreads traffic to all of the nodes in your cluster. This option is specified with the "createExternalLoadBalancer": true property.

    {
      "id": "myapp",
      "kind": "Service",
      "apiVersion": "v1beta1",
      "selector": {
        "app": "example"
      },
      "containerPort": 9376,
      "protocol": "TCP",
      "port": 8765,
      "createExternalLoadBalancer": true
    }

To access the service, a client connects to the external IP address, which forwards to port 8765 on a node in the cluster, which in turn accesses port 9376 on the pod. 

#### View a service

    $ kubectl get services
    
A successful get request returns all services that exist on the specified cluster:

    NAME                LABELS                                    SELECTOR            IP                  PORT
    apache2-service     <none>                                    name=apache2        10.100.123.196      9090
    
To return information about a specific service,

    $ kubectl describe service NAME
    
Details about the specific service are returned:

    Name:     myapp
    Labels:   <none>
    Selector: app=MyApp
    Port:     8765
    No events.
    
#### Delete a service

    $ kubectl delete service NAME
    
A successful delete request returns the deleted service's name.

## Other

### kubernetes和kubernetes-ro

kubernetes启动时，默认有两个服务kubernetes和kubernetes-ro

    $ kubectl get services
    NAME                LABELS                                    SELECTOR            IP                  PORT
    kubernetes          component=apiserver,provider=kubernetes   <none>              10.100.0.2          443
    kubernetes-ro       component=apiserver,provider=kubernetes   <none>              10.100.0.1          80
    
Kubernetes uses service definitions for the API service as well（kubernete中的pod可以通过`10.100.0.1:80/api/xxx`来向api server发送控制命令，kubernetes-ro可操作的api应该都是只读的）.

### PublicIPs

service configure文件中有一个`PublicIPs`属性

    {
      "id": "myapp",
      "kind": "Service",
      "apiVersion": "v1beta1",
      "selector": {
        "app": "example"
      },
      "containerPort": 9376,
      "port": 8765
      "PublicIPs": [192.168.56.102,192.168.56.103]
    }
    
在这里`192.168.56.102`和`192.168.56.103`是k8s集群从节点的ip（**主节点ip不行**）。这样，我们就可以通过`192.168.56.102:8765`和`192.168.56.102:8765`来访问这个service了。其好处是，kube-proxy为我们映射的端口是确定的。

## NodePort

所谓 Service 的访问入口，其实就是每台宿主机上由 kube-proxy 生成的 iptables 规则，以及 kube-dns 生成的 DNS 记录。而一旦离开了这个集群，这些信息对用户来说，也就自然没有作用了。比如，一个集群外的host 对service vip 一点都不感冒。
