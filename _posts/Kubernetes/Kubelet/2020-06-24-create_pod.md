---

layout: post
title: Pod是如何被创建出来的？
category: 架构
tags: Kubernetes
keywords:  create pod

---

## 简介

* TOC
{:toc}

[一篇文章搞定大规模容器平台生产落地十大实践](https://mp.weixin.qq.com/s/Cv4i5bxseMEwx1C_Annqig)

![](/public/upload/kubernetes/create_pod.png)

所有的Kubernetes组件Controller, Scheduler, Kubelet都使用Watch机制来监听API Server，来获取对象变化的事件

1. 用户通过Kubectl提交Pod Spec给API Server；
2. API Server将Pod对象的信息存入Etcd中
3. Pod的创建会生成一个事件，返回给API Server
4. Controller监听到这个事件
5. Controller知道这个Pod要mount一个盘，于是查看是否有能够满足条件的PV
6. 假设有满足条件的PV，就将Pod和PV绑定在一起，将绑定关系告知API Server
7. API Server将绑定信息写入Etcd中
8. 生成一个Pod Update事件
9. Scheduler监听到了这个事件
10. Scheduler需要为Pod选择一个Node
11. 假设有满足条件的Node，就讲Pod和Node绑定在一起，将绑定关系告知API Server
12. API Server将绑定信息写入Etcd中
13. 生成一个Pod Update事件
14. Kubelet监听到了这个事件，开始创建Pod
15. Kubelet告知CRI去下载镜像
16. Kubelet告知CRI去运行容器
17. CRI调用Docker运行容器
18. Kubelet告知Volume Manager，将盘挂在到Node上，然后mount到Pod中
19. CRI调用CNI给容器配置网络


![](/public/upload/kubernetes/kubelet_create_pod.png)

1. Kubelet是一切容器 feature的落地点
2. 所有的组件只与apiserver交互，不直接互通，更不直接控制kubelet 做某事

![](/public/upload/kubernetes/create_deployment.png)


![](/public/upload/kubernetes/apiserver_create_pod.png)

![](/public/upload/kubernetes/kubelet_create_pod_detail.png)

[源码解析：Kubernetes 创建 Pod 时，背后发生了什么](https://mp.weixin.qq.com/s/c78hVA0xRJLlTsVa40-FEA)