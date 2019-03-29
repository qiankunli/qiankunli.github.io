---

layout: post
title: Kubernetes安全机制
category: 技术
tags: Kubernetes
keywords: kubernetes security

---

## 简介（未完成）

* TOC
{:toc}

[火得一塌糊涂的kubernetes有哪些值得初学者学习的？](https://mp.weixin.qq.com/s/iI5vpK5bVkKmdbf9sbAGWw) Kubernetes 是透明的，它没有隐藏的内部 API。换句话说 **Kubernetes 系统内部用来交互的 API 和我们用来与 Kubernetes 交互的 API 相同**。这样做的好处是，当 Kubernetes 默认的组件无法满足我们的需求时，我们可以利用已有的 API 实现我们自定义的特性。

![](/public/upload/kubernetes/k8s_security.png)


## 用户

[Managing Service Accounts](https://kubernetes.io/docs/reference/access-authn-authz/service-accounts-admin/)

1. User accounts are for humans. Service accounts are for processes, which run in pods.
2. User accounts are intended to be global. Names must be unique across all namespaces of a cluster, future user resource will not be namespaced. Service accounts are namespaced.




## access the API 

[Controlling Access to the Kubernetes API](https://kubernetes.io/docs/reference/access-authn-authz/controlling-access/)

![](/public/upload/kubernetes/k8s_api_access_control.svg)



[Self-signed certificate](https://en.wikipedia.org/wiki/Self-signed_certificate)In cryptography and computer security, a self-signed certificate is an identity certificate that is signed by the same entity whose identity it certifies. 

## 访问一个普通的https api 要什么

对比一下
