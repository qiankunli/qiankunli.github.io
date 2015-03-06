---
layout: post
title: Kubernetes —— The Google Container Engine(3)
category: 技术
tags: CoreOS
keywords: CoreOS Docker Kubernetes
---


## 简介

本文主要来自对[https://cloud.google.com/container-engine/docs](https://cloud.google.com/container-engine/docs "")的摘抄，有删减。

本文主要讲了service组件

## What is a service?

Container Engine pods are ephemeral. They can come and go over time, especially when driven by things like replication controllers. While each pod gets its own IP address, those IP addresses cannot be relied upon to be stable over time. This leads to a problem: if some set of pods (let's call them backends) provides functionality to other pods (let's call them frontends) inside a cluster, how do those frontends find the backends?
(Pod组件的状态经常变化，也可能存在多个副本，那么其他组件如何来访问它呢)

Enter services.

A Container Engine service is an abstraction which defines a logical set of pods and a policy by which to access them. **The goal of services is to provide a bridge for non-Kubernetes-native applications to access backends without the need to write code that is specific to Kubernetes. A service offers clients an IP and port pair which, when accessed, redirects to the appropriate backends.** The set of pods targeted is determined by a label selector.

As an example, consider an image-process backend which is running with 3 live replicas. Those replicas are fungible—frontends do not care which backend they use. While the actual pods that comprise the set may change, the frontend client(s) do not need to know that. The service abstraction enables this decoupling.

### How do they work?

Each node in a cluster runs a service proxy. This application watches the cluster master for the addition and removal of service objects. If a service's label selector matches the labels on a pod, the proxy opens a port on the local node for that service and forwards traffic to the pod.
（每一个从节点监控master的service状态（新增或删除），如果一个新的service的label本节点的某个pod，那么本节点为那个service开一个端口，并将那个service传来的请求转发给该pod）

When a pod is scheduled, the master adds a set of environment variables for each active service.

A service, through its label selector, can resolve to 0 or more pods. Over the life of a service, the set of pods which comprise that service can grow, shrink, or turn over completely. Clients will only see issues if they are actively using a backend when that backend is removed from the service (and even then, open connections will persist for some protocols).

## Service Operations

Services map a port on each cluster node to ports on one or more pods.The mapping uses a selector key:value pair in the service, and the labels property of pods. Any pods whose labels match the service selector are made accessible through the service's port.

### Create a service

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

### View a service

    $ kubectl get services
    
A successful get request returns all services that exist on the specified cluster:

    NAME                LABELS                                    SELECTOR            IP                  PORT
    apache2-service     <none>                                    name=apache2        10.100.123.196      9090

    
也就是`10.100.123.196:9090`将会代表这个服务，相对于经常变化状态的pod，这个是不变的。然后，对于cluster的每个节点，如果用`sudo iptables -nvL -t nat`查看一下:

    Chain KUBE-PORTALS-CONTAINER (1 references)
     pkts bytes target     prot opt in     out     source               destination
        0     0 REDIRECT   tcp  --  *      *       0.0.0.0/0            10.100.0.1           /* kubernetes-ro */ tcp dpt:80 redir ports 39058
        0     0 REDIRECT   tcp  --  *      *       0.0.0.0/0            10.100.0.2           /* kubernetes */ tcp dpt:443 redir ports 48474
        0     0 REDIRECT   tcp  --  *      *       0.0.0.0/0            10.100.123.196       /* apache2-service */ tcp dpt:9090 redir ports 35788
        
可以看到示例节点中，该结点通过35788端口来访问`10.100.123.196:9090`，其它节点不再赘述。
（换句话说，对于一个service（本例的10.100.123.196:9090），每个节点的kube_proxy会使用iptables为其映射一个端口（端口号随机生成））
    
To return information about a specific service,

    $ kubectl describe service NAME
    
Details about the specific service are returned:

    Name:     myapp
    Labels:   <none>
    Selector: app=MyApp
    Port:     8765
    No events.
    
### Delete a service

    $ kubectl delete service NAME
    
A successful delete request returns the deleted service's name.