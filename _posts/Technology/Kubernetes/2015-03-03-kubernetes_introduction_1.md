---
layout: post
title: Kubernetes 基本概念和 pod 组件
category: 技术
tags: Kubernetes
keywords: CoreOS Docker Kubernetes
---


## 简介

本文主要来自对[https://cloud.google.com/container-engine/docs](https://cloud.google.com/container-engine/docs "")的摘抄，有删减。

本文主要讲了Container Engine cluster和Pod的概念

##  Container Engine cluster

A Container Engine cluster is a group of Compute Engine instances running Kubernetes. It consists of one or more node instances, and a Kubernetes master instance. A cluster is the foundation of a Container Engine application—pods,(如果说在coreos中，一个应用用container（及相关的service文件）表示。那么kubernete中，一个应用由pod表示) services, and replication controllers all run on top of a cluster.

（一个Container Engine cluster主要包含一个master和多个slave节点）

### The Kubernetes master

Every cluster has a single master instance. The master provides a unified view into the cluster and, through its publicly-accessible endpoint, is the doorway for interacting with the cluster.

**The master runs the Kubernetes API server, which services REST requests, schedules pod creation and deletion on worker nodes, and synchronizes pod information (such as open ports and location) with service information.**

### Nodes

A cluster can have one or more node instances. These are managed from the master, and run the services necessary to support Docker containers. Each node runs the Docker runtime and hosts a Kubelet agent（管理docker runtime）, which manages the Docker containers scheduled on the host. Each node also runs a simple network proxy（网络代理程序）.

## What is a pod?

A pod models an application-specific "logical host(逻辑节点)" in a containerized environment. It may contain one or more containers which are relatively tightly coupled—in a pre-container world（包含多个紧密联系的容器）, they would have executed on the same physical or virtual host.a pod has a single IP address.  Multiple containers that run in a pod all share that common network name space。（一个pod中包含ca cb两个container，ca暴露80端口，cb即可以`localhost:80`访问ca中的服务）

Like running containers, pods are considered to be relatively ephemeral rather than durable entities. Pods are scheduled to nodes and remain there until termination (according to restart policy) or deletion. When a node dies, the pods scheduled to that node are deleted. Specific pods are never rescheduled to new nodes; instead, they must be replaced.

### Motivation for pods

#### Resource sharing and communication

Pods facilitate data sharing and communication among their constituents.（pods（概念）促进了其组件的数据共享和交流）

The containers in the pod all use the same network namespace/IP and port space, and can find and communicate with each other using localhost. Each pod has an IP address in a flat shared networking namespace that has full communication with other physical computers and containers across the network. The hostname for each container within the pod is set to the pod's name.

（Pod内的每个容器有一个套IP及PORT命名空间，每个Pod有一个IP（在另一个命名空间内）来同外界交流，（以IP来标记这个Pod及其提供的服务））

In addition to defining the containers that run in the pod, the pod specifies a set of shared storage volumes. Volumes enable data to survive container restarts and to be shared among the containers within the pod.

（Pod还定义了一系列volumes用于数据共享）

#### pod实现原理

容器是基于linux namespace实现资源隔离的，但是一个pod中的所有容器则共享部分namespace，没有完全隔离。

1. 网络名字空间，在同一Pod中的多个容器访问同一个IP和端口空间，即可能访问同一个network namespace。
2. IPC名字空间，同一个Pod中的应用能够使用SystemV IPC和POSIX消息队列进行通信。
3. UTS名字空间，同一个Pod中的应用共享一个主机名。
4. mount命名空间，Pod中的各个容器应用还可以访问Pod级别定义的共享卷。

#### Management

Pods also simplify application deployment and management by providing a higher-level abstraction than the raw, low-level container interface（通过提出Pod这个更高抽象来简化应用的部署和管理）. Pods serve as units of deployment and horizontal scaling/replication. Co-location, fate sharing, coordinated replication, resource sharing, and dependency management are handled automatically.

### Uses of pods（应用场景）

Pods can be used to host vertically integrated application stacks, but their primary motivation is to support co-located, co-managed helper programs, such as:

Content management systems, file and data loaders, local cache managers, etc.
Log and checkpoint backup, compression, rotation, snapshotting, etc.
Data-change watchers, log tailers, logging and monitoring adapters, event publishers, etc.
Proxies, bridges, and adapters.
Controllers, managers, configurators, and updaters.
Individual pods are not intended to run multiple instances of the same application, in general.

## Pod Operations

**A pod is a relatively tightly coupled group of containers that are scheduled onto the same host. It models an application-specific "virtual host" in a containerized environment. Pods serve as units of scheduling, deployment, and horizontal scaling/replication. Pods share fate, and share some resources, such as storage volumes and IP addresses.**

### Creating a pod

#### Pod configuration file

A pod configuration file specifies required information about the pod/ It can be formatted as YAML or as JSON, and supports the following fields:

    {
      "id": string,
      "kind": "Pod",
      "apiVersion": "v1beta1",
      "desiredState": {
        "manifest": {
          manifest object
        }
      },
      "labels": { string: string }
    }
    
Required fields are:

- id: The name of this pod. It must be an RFC1035 compatible value and be unique on this container cluster.
- kind: Always Pod.
- apiVersion: Currently v1beta1.
- desiredState: The configuration for this pod. It must contain a child manifest object.

Optional fields are:

- labels are arbitrary key:value pairs that **can be used by replication controllers and services for grouping and targeting pods**.

#### Manifest

Manifest部分的内容不再赘述（所包含字段，是否必须，以及其意义），可以参见文档

#### Sample file

    {
      "id": "redis-controller",
      "kind": "Pod",
      "apiVersion": "v1beta1",
      "desiredState": {
        "manifest": {
          "version": "v1beta1",
          "containers": [{
            "name": "redis",
            "image": "dockerfile/redis",
            "ports": [{
              "containerPort": 6379,
              "hostPort": 6379
            }]
          }]
        }
      },
      "labels": {
        "name": "redis-controller"
      }
    }

### Viewing a pod

    kubectl get pod xxx
    ## list pod
    kubectl get pods

### Deleting a pod

    kubectl delete pod xxx