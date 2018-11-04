---
layout: post
title: Kubernetes 基本概念和 pod 组件
category: 技术
tags: Kubernetes
keywords: CoreOS Docker Kubernetes

---


## 简介

* TOC
{:toc}

本文主要来自对[https://cloud.google.com/container-engine/docs](https://cloud.google.com/container-engine/docs)的摘抄，有删减。

本文主要讲了Container Engine cluster和Pod的概念

##  Container Engine cluster

A Container Engine cluster is a group of Compute Engine instances running Kubernetes. It consists of one or more node instances, and a Kubernetes master instance. A cluster is the foundation of a Container Engine application—pods,services, and replication controllers all run on top of a cluster.

一个Container Engine cluster主要包含一个master和多个slave节点，它是上层的pod、service、replication controllers的基础。

### The Kubernetes master

Every cluster has a single master instance. The master provides a unified view into the cluster and, through its publicly-accessible endpoint, is the doorway(途径) for interacting with the cluster.

**The master runs the Kubernetes API server, which services REST requests, schedules pod creation and deletion on worker nodes, and synchronizes pod information (such as open ports and location) with service information.**

1. 提供统一视图
2. service REST requests
3. 调度
4. 控制，使得actual state满足desired state 

### Nodes

A cluster can have one or more node instances. These are managed from the master, and run the services necessary to support Docker containers. Each node runs the Docker runtime and hosts a Kubelet agent（管理docker runtime）, which manages the Docker containers scheduled on the host. Each node also runs a simple network proxy（网络代理程序）.

## What is a pod?

A pod models an application-specific "logical host(逻辑节点)" in a containerized environment. It may contain one or more containers which are relatively tightly coupled—in a pre-container world（在 pre-container 时代紧密联系的进程 ，在container 时代放在一个pod里）, they would have executed on the same physical or virtual host.a pod has a single IP address.  Multiple containers that run in a pod all share that common network name space。

Like running containers, pods are considered to be relatively ephemeral rather than durable entities. Pods are scheduled to nodes and remain there until termination (according to restart policy) or deletion. When a node dies, the pods scheduled to that node are deleted. Specific pods are never rescheduled to new nodes; instead, they must be replaced.

重点不是pod 是什么，而是什么情况下， 我们要将多个容器放在pod 里。 "为什么需要一个pod?" 详细论述 [kubernetes objects再认识](http://qiankunli.github.io/2018/11/04/kubernetes_objects.html)


### Uses of pods（应用场景）

Pods can be used to host vertically integrated application stacks, but their primary motivation is to support co-located, co-managed （这两个形容词绝了）helper programs, such as:

1. Content management systems, file and data loaders, local cache managers, etc.
2. Log and checkpoint backup, compression, rotation, snapshotting, etc.
3. Data-change watchers, log tailers, logging and monitoring adapters, event publishers, etc.
4. Proxies, bridges, and adapters.
5. Controllers, managers, configurators, and updaters.

**Individual pods are not intended to run multiple instances of the same application**, in general.

## 小结

A pod is a relatively tightly coupled group of containers that are scheduled onto the same host. 

1. It models an application-specific(面向应用) "virtual host" in a containerized environment. 
2. Pods serve as units of scheduling, deployment, and horizontal scaling/replication. 
3. Pods share fate（命运）, and share some resources, such as storage volumes and IP addresses.(网络通信和数据交互就非常方便且高效)

## Pod Operations

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