---

layout: post
title: Kubernetes 其它特性
category: 技术
tags: Kubernetes
keywords: Docker Kubernetes

---

## 简介


## 访问多个kubernetes 集群

1. 一般情况，kubernetes 单独搭建在一个集群上，开发者通过开发机 或某一个跳板机上 通过kubectl 操作kubernetes，kubectl 会读取`~/.kube/config` 文件读取集群信息
2. kubernetes 一般会有多个集群：测试环境（运行公司测试环境的服务），开发环境（用来验证新功能）==> developer 需要在本机 上使用kubectl 访问多个k8s集群

[配置对多集群的访问](https://kubernetes.io/zh/docs/tasks/access-application-cluster/configure-access-multiple-clusters/)

`~/.kube/config` 是一个yaml 文件，可以配置多个集群的信息

    apiVersion: v1
    kind: Config
    clusters:
    users:
    contexts:

可以看到 几个核心配置都是数组

    apiVersion: v1
    kind: Config
    clusters:
    - cluster:
    name: development
    - cluster:
    name: scratch
    users:
    - name: developer
    - name: experimenter
    contexts:
    - context:
        cluster: development
        user: developer
      name: dev-frontend
    name: dev-frontend
    - context:
        cluster: scratch
        user: experimenter
      name: exp-scratch



## Environment variables

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



## DNS

问题，为什么kubernete 需要dns服务？

我们大可不管kubernete的底层网络细节，就认为现在有一个10.100.0.0/16的网络，每个pod对应一个ip，可以相互通信。只是每个pod的状态经常变化。

每个pod提供的服务，但其状态在不断变化，所以出现了service，将ip（估计也是10.100开头的）：port 与 服务绑定起来。所以我们访问一个pod，实际是访问其对应的serviceip。现在我们将所有的serviceip以及其名字servicename存在dns pod中，这就是kubernete中dns服务。

那么在其它pod中，就可以直接使用hostname:port来使用其它pod。本pod无法解析hostname时，便会去查dns pod获取ip。当然，道理是这样，但实际实现还有好多细节。

An optional (though strongly recommended) cluster add-on is a DNS server. The DNS server watches the Kubernetes API for new Services and creates a set of DNS records for each. If DNS has been enabled throughout the cluster then all Pods should be able to do name resolution of Services automatically.

For example, if you have a Service called "my-service" in Kubernetes Namespace "my-ns" a DNS record for "my-service.my-ns" is created. Pods which exist in the "my-ns" Namespace should be able to find it by simply doing a name lookup for "my-service". Pods which exist in other Namespaces must qualify the name as "my-service.my-ns". The result of these name lookups is the virtual portal IP.



## k8s 在 etcd中的存在

    /registry/minions
    /registry/minions/192.168.56.102    # 列出该节点的信息，包括其cpu和memory能力
    /registry/minions/192.168.56.103
    /registry/controllers
    /registry/controllers/default
    /registry/controllers/default/apache2-controller	# 跟创建该controller时信息大致相同，分为desireState和currentState
    /registry/controllers/default/heapster-controller
    /registry/pods
    /registry/pods/default
    /registry/pods/default/128e1719-c726-11e4-91cd-08002782f91d   	# 跟创建该pod时信息大致相同，分为desireState和currentState
    /registry/pods/default/128e7391-c726-11e4-91cd-08002782f91d
    /registry/pods/default/f111c8f7-c726-11e4-91cd-08002782f91d
    /registry/nodes
    /registry/nodes/192.168.56.102
    /registry/nodes/192.168.56.102/boundpods	# 列出在该主机上运行pod的信息，镜像名，可以使用的环境变量之类，这个可能随着pod的迁移而改变
    /registry/nodes/192.168.56.103
    /registry/nodes/192.168.56.103/boundpods
    /registry/events
    /registry/events/default
    /registry/events/default/704d54bf-c707-11e4-91cd-08002782f91d.13ca18d9af8857a8		# 记录操作，比如将某个pod部署到了某个node上
    /registry/events/default/f1ff6226-c6db-11e4-91cd-08002782f91d.13ca07dc57711845
    /registry/services
    /registry/services/specs
    /registry/services/specs/default
    /registry/services/specs/default/monitoring-grafana		#  基本跟创建信息大致一致，但包含serviceip
    /registry/services/specs/default/kubernetes
    /registry/services/specs/default/kubernetes-ro
    /registry/services/specs/default/monitoring-influxdb
    /registry/services/endpoints
    /registry/services/endpoints/default
    /registry/services/endpoints/default/monitoring-grafana	  	# 终端（traffic在这里被处理），和某一个serviceId相同，包含了service对应的几个pod的ip，这个可能经常变。
    /registry/services/endpoints/default/kubernetes
    /registry/services/endpoints/default/kubernetes-ro
    /registry/services/endpoints/default/monitoring-influxdb
    
endpoint 换个说法，
    
## Services without selectors

**Services, in addition to providing abstractions to access Pods, can also abstract any kind of backend(service不仅可以做访问pod的桥梁，还可以做访问任何后端的桥梁)**. For example:

- you want to have an external database cluster in production, but in test you use your own databases.
- you want to point your service to a service in another Namespace or on another cluster.
- you are migrating your workload to Kubernetes and some of your backends run outside of Kubernetes.

In any of these scenarios you can define a service without a selector:

    {
      "kind": "Service",
      "apiVersion": "v1beta1",
      "id": "myapp",
      "port": 80
    }
    
Then you can explicitly map the service to a specific endpoint(s):

    {    
        "kind": "Endpoints",
        "apiVersion": "v1beta1",
        "id": "myapp",
        "endpoints": ["173.194.112.206:80"]
    }
  
Accessing a Service without a selector works the same as if it had selector. The traffic will be routed to endpoints defined by the user (173.194.112.206:80 in this example).（以后，pod访问这个serivce的ip：80 就会被转到`173.194.112.206:80`了）（此时，一旦这个endpoint终止或者转移，k8s就不负责跟踪，并将specs与新的endpoint绑定了）

(serivce将请求转发到各个pod也是 "specs + endpoint"方式实现的 )

有了个这个endpoint + service，基本就解决了pod如何访问外网数据的问题。比如

etcd-service.json 文件

    {
      "kind": "Service",
      "apiVersion": "v1beta1",
      "id": "myetcd",
      "port": 80
    }

etcd-endpoints.json
    
    {    
        "kind": "Endpoints",
        "apiVersion": "v1beta1",
        "id": "myetcd",
        "endpoints": ["masterip:4001"]
    }

那么在pod中，就可以通过myetcd的`serviceip:80`来访问master主机上的etcd服务了。



