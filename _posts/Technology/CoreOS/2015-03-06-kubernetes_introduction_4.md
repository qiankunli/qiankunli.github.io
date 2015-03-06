---
layout: post
title: Kubernetes —— The Google Container Engine(4)
category: 技术
tags: CoreOS
keywords: CoreOS Docker Kubernetes
---


## 简介


本文主要讲了service组件一些其它特性，以及kubernete的pod之间协作的时候的一些问题

## Services without selectors

**Services, in addition to providing abstractions to access Pods, can also abstract any kind of backend(service不仅可以做访问pod的桥梁，还可以做访问任何后端的桥梁)**. For example:

- you want to have an external database cluster in production, but in test you use your own databases.
- you want to point your service to a service in another Namespace or on another cluster.
- you are migrating your workload to Kubernetes and some of your backends run outside of Kubernetes.

In any of these scenarios you can define a service without a selector:

      "kind": "Service",
      "apiVersion": "v1beta1",
      "id": "myapp",
      "port": 80
  
Then you can explicitly map the service to a specific endpoint(s):

      "kind": "Endpoints",
      "apiVersion": "v1beta1",
      "id": "myapp",
      "endpoints": ["173.194.112.206:80"]
  
Accessing a Service without a selector works the same as if it had selector. The traffic will be routed to endpoints defined by the user (173.194.112.206:80 in this example).（以后，pod访问这个serivce的ip：80 就会被转到`173.194.112.206:80`了）