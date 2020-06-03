---
layout: post
title: 访问Kubernetes上的Service
category: 技术
tags: Kubernetes
keywords: Docker Kubernetes
---


## 简介

* TOC
{:toc}

本文均在“访问Pod 必须通过 Service的范畴”


![](/public/upload/kubernetes/kubernetes_service_access.png)

## What is a service?

Kubernetes 之所以需要 Service，一方面是因为 Pod 的 IP 不是固定的，另一方面则是因为一组 Pod 实例之间总会有负载均衡的需求。

### How do they work?——基于iptables实现

K8S 集群的服务，本质上是负载均衡，即反向代理;在实际实现中，这个反向代理，并不是部署在集群某一个节点上（有单点问题），而 是**作为集群节点的边车**，部署在每个节点上的。把服务照进反向代理这个现实的，是 K8S 集群的一个控制器，即 kube-proxy。简单来 说，kube-proxy 通过集群 API Server 监听 着集群状态变化。当有新的服务被创建的时候，kube-proxy 则会把集群服务的状 态、属性，翻译成反向代理的配置。**K8S 集群节点实现服务反向代理的方法，目前主要有三种，即 userspace、 iptables 以及 ipvs**，k8s service 选了iptables。实现反向代理，归根结底，就是做 DNAT，即把发送给集群服务 IP 和端口的数 据包，修改成发给具体容器组的 IP 和端口。

A service provides a stable virtual IP (VIP) address for a set of pods. It’s essential to realize that VIPs do not exist as such in the networking stack. For example, **you can’t ping them.** They are only Kubernetes- internal administrative entities. Also note that the format is IP:PORT, so the IP address along with the port make up the VIP. **Just think of a VIP as a kind of index into a data structure mapping to actual IP addresses.**

![](/public/upload/kubernetes/service_communicate.png)

When a pod is scheduled, the master adds a set of environment variables for each active service.

### 示例

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hostnames
spec:
  selector:
    matchLabels:
      app: hostnames
  replicas: 3
  template:
    metadata:
      labels:
        app: hostnames
    spec:
      containers:
      - name: hostnames
        image: k8s.gcr.io/serve_hostname
        ports:
        - containerPort: 9376
          protocol: TCP
```

用 selector 字段来声明这个 Service 只代理携带了 app=hostnames 标签的 Pod

```yaml
apiVersion: v1
kind: Service
metadata:
  name: hostnames
spec:
  selector:
    app: hostnames
  ports:
  - name: default
    protocol: TCP
    port: 80
    targetPort: 9376
```

一旦它被提交给 Kubernetes，那么 kube-proxy 就可以通过 Service 的 Informer 感知到这样一个 Service 对象的添加。而作为对这个事件的响应，它就会在宿主机上创建这样一条 iptables 规则

```
-A KUBE-SERVICES -d 10.0.1.175/32 -p tcp -m comment --comment "default/hostnames: cluster IP" -m tcp --dport 80 -j KUBE-SVC-NWV5X2332I4OT4T3
```
这条 iptables 规则的含义是：凡是目的地址是 10.0.1.175、目的端口是 80 的 IP 包，都应该跳转到另外一条名叫 KUBE-SVC-NWV5X2332I4OT4T3 的 iptables 链进行处理。这一条规则就为这个 Service 设置了一个固定的入口地址。并且，由于 10.0.1.175 只是一条 iptables 规则上的配置，并没有真正的网络设备，所以你 ping 这个地址，是不会有任何响应的。

```
-A KUBE-SVC-NWV5X2332I4OT4T3 -m comment --comment "default/hostnames:" -m statistic --mode random --probability 0.33332999982 -j KUBE-SEP-WNBA2IHDGP2BOBGZ
-A KUBE-SVC-NWV5X2332I4OT4T3 -m comment --comment "default/hostnames:" -m statistic --mode random --probability 0.50000000000 -j KUBE-SEP-X3P2623AGDH6CDF3
-A KUBE-SVC-NWV5X2332I4OT4T3 -m comment --comment "default/hostnames:" -j KUBE-SEP-57KPRZ3JQVENLNBR
```

KUBE-SVC-NWV5X2332I4OT4T3 规则实际上是一组随机模式（–mode random）的 iptables 链，而随机转发的目的地，分别是 KUBE-SEP-WNBA2IHDGP2BOBGZ、KUBE-SEP-X3P2623AGDH6CDF3 和 KUBE-SEP-57KPRZ3JQVENLNBR，其实就是这个 Service 代理的三个 Pod。

```

-A KUBE-SEP-57KPRZ3JQVENLNBR -s 10.244.3.6/32 -m comment --comment "default/hostnames:" -j MARK --set-xmark 0x00004000/0x00004000
-A KUBE-SEP-57KPRZ3JQVENLNBR -p tcp -m comment --comment "default/hostnames:" -m tcp -j DNAT --to-destination 10.244.3.6:9376

-A KUBE-SEP-WNBA2IHDGP2BOBGZ -s 10.244.1.7/32 -m comment --comment "default/hostnames:" -j MARK --set-xmark 0x00004000/0x00004000
-A KUBE-SEP-WNBA2IHDGP2BOBGZ -p tcp -m comment --comment "default/hostnames:" -m tcp -j DNAT --to-destination 10.244.1.7:9376

-A KUBE-SEP-X3P2623AGDH6CDF3 -s 10.244.2.3/32 -m comment --comment "default/hostnames:" -j MARK --set-xmark 0x00004000/0x00004000
-A KUBE-SEP-X3P2623AGDH6CDF3 -p tcp -m comment --comment "default/hostnames:" -m tcp -j DNAT --to-destination 10.244.2.3:9376
```

当你的宿主机上有大量 Pod 的时候，成百上千条 iptables 规则不断地被刷新，会大量占用该宿主机的 CPU 资源，甚至会让宿主机“卡”在这个过程中。

## Service 的特别形式

在yaml 配置层面 LoadBalancer/NodePort/ExternalName 的kind 都是 Service



### LoadBalancer

```yaml
kind: Service
apiVersion: v1
metadata:
  name: example-service
spec:
  type: LoadBalancer
  ports:
  - port: 8765
    targetPort: 9376
  selector:
    app: example
```

### NodePort

所谓 Service 的访问入口，其实就是每台宿主机上由 kube-proxy 生成的 iptables 规则，以及 kube-dns 生成的 DNS 记录。而一旦离开了这个集群，这些信息对用户来说，也就自然没有作用了。比如，一个集群外的host 对service vip 一点都不感冒。


```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-nginx
  labels:
    run: my-nginx
spec:
  type: NodePort
  ports:
  - nodePort: 8080
    targetPort: 80
    protocol: TCP
    name: http
  - nodePort: 443
    protocol: TCP
    name: https
  selector:
    run: my-nginx
```

### ExternalName

```yaml
kind: Service
apiVersion: v1
metadata:
  name: my-service
spec:
  type: ExternalName
  externalName: my.database.example.com
```

## Ingress

[Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/) exposes HTTP and HTTPS routes from outside the cluster to services within the cluster. Traffic routing is controlled by rules defined on the Ingress resource. An Ingress may be configured to **give Services externally-reachable URLs**, load balance traffic, terminate SSL / TLS, and offer name based virtual hosting. 

![](/public/upload/kubernetes/ingress_communicate.png)

|对外形态|Kubernetes Object|工作组件|
|---|---|---|
|ip|Pod|docker|
|stable virtual ip|Service|apiserver + kube-proxy|
|externally-reachable URLs|Ingress|apiserver + nginx-ingress|

Ingress和Pod、Servce等等类似，被定义为kubernetes的一种资源。本质上说Ingress只是存储在etcd上面一些数据，我们可以通过kubernetes的apiserver添加删除和修改ingress资源。

有了 Ingress 这样一个统一的抽象，Kubernetes 的用户就无需关心 Ingress 的具体细节了。在实际的使用中，你只需要从社区里选择一个具体的 Ingress Controller，把它部署在 Kubernetes 集群里即可。这个 Ingress Controller 会根据你定义的 Ingress 对象，提供对应的代理能力。目前，业界常用的各种反向代理项目，比如 Nginx、HAProxy、Envoy、Traefik 等，都已经为 Kubernetes 专门维护了对应的 Ingress Controller。

### Ingress Controller 部署和实现

部署 Nginx Ingress Controller [Installation with Manifests](https://docs.nginx.com/nginx-ingress-controller/installation/installation-with-manifests/#deploy-the-ingress-controller)  简单安装可以使用 helm [Installation with Helm](https://docs.nginx.com/nginx-ingress-controller/installation/installation-with-helm/) 注意处理下google 的镜像源问题。nginx ingress 会安装一个 Deployment：ingress-nginx-controller  在 ingress-nginx namespace 下（helm 安装的好像不是 这个namespace），这个 Pod 本身就是一个监听 Ingress 对象以及它所代理的后端 Service 变化的控制器。

当一个新的 Ingress 对象由用户创建后，nginx-ingress-controller 就会根据Ingress 对象里定义的内容，生成一份对应的 Nginx 配置文件（`/etc/nginx/nginx.conf`），并使用这个配置文件启动一个 Nginx 服务。而一旦 Ingress 对象被更新，nginx-ingress-controller 就会更新这个配置文件。如果这里只是被代理的 Service 对象被更新，nginx-ingress-controller 所管理的 Nginx 服务是不需要重新加载（reload）的，因为其通过Nginx Lua方案实现了 Nginx Upstream 的动态配置。

### Ingress 示例

```yaml
apiVersion: v1
kind: Pod
metadata:
 name: nginx-pod-demo-for-test-ingress
 labels:
   app: nginx-pod-demo-for-test-ingress
spec:
 containers:
   - name: nginx-container
     image: nginx
     ports:
       - containerPort: 80

---
apiVersion: v1
kind: Service
metadata:
  name: nginx-service-demo-for-test-ingress
spec:
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP
    name: http
  selector:
    app: nginx-pod-demo-for-test-ingress
---
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: nginx-ingress 
spec:
  rules:
  - host: localhost
    http:
      paths:
      - path: /
        backend:
          serviceName: nginx-service-demo-for-test-ingress
          servicePort: 80
```

一个 Ingress 对象的主要内容，实际上就是一个“反向代理”服务（比如：Nginx）的配置文件的描述。而这个代理服务对应的转发规则，就是 IngressRule。


```
root@ubuntu-01:~# kubectl get pod
NAME                                        READY   STATUS    RESTARTS   AGE
my-release-nginx-ingress-5c57477464-gssbk   1/1     Running   0          71m
nginx-pod-demo-for-test-ingress             1/1     Running   0          57m
root@ubuntu-01:~# kubectl get svc
NAME                                  TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)                      AGE
my-release-nginx-ingress              LoadBalancer   10.101.164.118   <pending>     80:32000/TCP,443:31723/TCP   64m
nginx-service-demo-for-test-ingress   ClusterIP      10.103.226.14    <none>        80/TCP                       49m
```

每个ingress 对象会有一个对应的pod 实际运行nginx ，以及一个LoadBalancer Service 作为这个pod的访问入口。

kubectl 进入my-release-nginx-ingress-5c57477464-gssbk，在`/etc/nginx/conf.d` 下可以看到 ingress 对应一个conf 文件。upstream 指定了 nginx pod对应的ip。 可见 虽然创建了 nginx service，但网络包并没有真正走 service iptables。

```
# configuration for default/nginx-ingress
upstream default-nginx-ingress-localhost-nginx-service-demo-for-test-ingress-80 {
	zone default-nginx-ingress-localhost-nginx-service-demo-for-test-ingress-80 256k;
	random two least_conn;
	server 10.244.2.52:80 max_fails=1 fail_timeout=10s max_conns=0;
}
server {
	listen 80;
	server_tokens on;
	server_name localhost;
	location / {
		proxy_http_version 1.1;
		proxy_connect_timeout 60s;
		proxy_read_timeout 60s;
		proxy_send_timeout 60s;
		client_max_body_size 1m;
		proxy_set_header Host $host;
		proxy_set_header X-Real-IP $remote_addr;
		proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
		proxy_set_header X-Forwarded-Host $host;
		proxy_set_header X-Forwarded-Port $server_port;
		proxy_set_header X-Forwarded-Proto $scheme;
		proxy_buffering on;
		proxy_pass http://default-nginx-ingress-localhost-nginx-service-demo-for-test-ingress-80;
	}
}
```


## 其它

[Kubernetes networking 101 – Services](http://www.dasblinkenlichten.com/kubernetes-networking-101-services/)

[Kubernetes networking 101 – (Basic) External access into the cluster](http://www.dasblinkenlichten.com/kubernetes-networking-101-basic-external-access-into-the-cluster/)

[Kubernetes Networking 101 – Ingress resources](http://www.dasblinkenlichten.com/kubernetes-networking-101-ingress-resources/)

[Getting started with Calico on Kubernetes](http://www.dasblinkenlichten.com/getting-started-with-calico-on-kubernetes/)（未读）

### kubernetes和kubernetes-ro

kubernetes启动时，默认有两个服务kubernetes和kubernetes-ro

    $ kubectl get services
    NAME                LABELS                                    SELECTOR            IP                  PORT
    kubernetes          component=apiserver,provider=kubernetes   <none>              10.100.0.2          443
    kubernetes-ro       component=apiserver,provider=kubernetes   <none>              10.100.0.1          80
    
Kubernetes uses service definitions for the API service as well（kubernete中的pod可以通过`10.100.0.1:80/api/xxx`来向api server发送控制命令，kubernetes-ro可操作的api应该都是只读的）.
