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

## Discovering services

Kubernetes supports 2 primary modes of finding a Service - environment variables and DNS.

### Environment variables

(类似于docker container中的`--link`)

When a Pod is run on a Node, the kubelet adds a set of environment variables for each active Service. It supports both Docker links compatible variables (see makeLinkVariables) and simpler {SVCNAME}_SERVICE_HOST and {SVCNAME}_SERVICE_PORT variables, where the Service name is upper-cased and dashes are converted to underscores.

For example, the Service "redis-master" which exposes TCP port 6379 and has been allocated portal IP address 10.0.0.11 produces the following environment variables:（我们可以`docker exec -it containerid bash`Pod中的一个container，使用`printenv`来查看其可以使用的环境变量）

    REDIS_MASTER_SERVICE_HOST=10.0.0.11
    REDIS_MASTER_SERVICE_PORT=6379
    REDIS_MASTER_PORT=tcp://10.0.0.11:6379
    REDIS_MASTER_PORT_6379_TCP=tcp://10.0.0.11:6379
    REDIS_MASTER_PORT_6379_TCP_PROTO=tcp
    REDIS_MASTER_PORT_6379_TCP_PORT=6379
    REDIS_MASTER_PORT_6379_TCP_ADDR=10.0.0.11
    
This does imply an ordering requirement - **any Service that a Pod wants to access must be created before the Pod itself**, or else the environment variables will not be populated. DNS does not have this restriction.



### DNS

问题，为什么kubernete 需要dns服务？

我们大可不管kubernete的底层网络细节，就认为现在有一个10.100.0.0/16的网络，每个pod对应一个ip，可以相互通信。只是每个pod的状态经常变化。

每个pod提供的服务，但其状态在不断变化，所以出现了service，将ip（估计也是10.100开头的）：port 与 服务绑定起来。所以我们访问一个pod，实际是访问其对应的serviceip。现在我们将所有的serviceip以及其名字servicename存在dns pod中，这就是kubernete中dns服务。

那么在其它pod中，就可以直接使用hostname:port来使用其它pod。本pod无法解析hostname时，便会去查dns pod获取ip。当然，道理是这样，但实际实现还有好多细节。

An optional (though strongly recommended) cluster add-on is a DNS server. The DNS server watches the Kubernetes API for new Services and creates a set of DNS records for each. If DNS has been enabled throughout the cluster then all Pods should be able to do name resolution of Services automatically.

For example, if you have a Service called "my-service" in Kubernetes Namespace "my-ns" a DNS record for "my-service.my-ns" is created. Pods which exist in the "my-ns" Namespace should be able to find it by simply doing a name lookup for "my-service". Pods which exist in other Namespaces must qualify the name as "my-service.my-ns". The result of these name lookups is the virtual portal IP.

## External Services

（为service弄一个外部的ip（而不是现在的端口映射的方式））

For some parts of your application (e.g. frontends) you may want to expose a Service onto an external (outside of your cluster, maybe public internet) IP address.

On cloud providers which support external load balancers, this should be as simple as setting the createExternalLoadBalancer flag of the Service to true. This sets up a cloud-specific load balancer and populates the publicIPs field (see below). Traffic from the external load balancer will be directed at the backend Pods, though exactly how that works depends on the cloud provider.

For cloud providers which do not support external load balancers, there is another approach that is a bit more "do-it-yourself" - the publicIPs field. Any address you put into the publicIPs array will be handled the same as the portal IP - the kube-proxy will install iptables rules which proxy traffic through to the backends. You are then responsible for ensuring that traffic to those IPs gets sent to one or more Kubernetes Nodes. As long as the traffic arrives at a Node, it will be be subject to the iptables rules.

An example situation might be when a Node has both internal and an external network interfaces. If you assign that Node's external IP as a publicIP, you can then aim traffic at the Service port on that Node and it will be proxied to the backends.

