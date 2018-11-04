---

layout: post
title: kubernetes yaml配置
category: 技术
tags: Kubernetes
keywords: kubernetes yaml

---

## 简介(会不断增补)


Kubernetes 跟 Docker 等很多项目最大的不同，就在于它不推荐你使用命令行的方式直接运行容器（虽然 kubectl run 支持)，而是采用yaml/json 文件的方式。最直接的好处是，你会有一个文件能记录下 Kubernetes到底“run”了什么。


[简化 Kubernetes Yaml 文件创建](https://yq.aliyun.com/articles/341213)由于Yaml文件格式比较复杂，即使是老司机有时也不免会犯错或需要查询文档，因此可以dry-run 一下，`kubectl run myapp --image=nginx --dry-run -o yaml` 会输出模拟运行 nginx 镜像的yaml 文件内容，copy-paste 即可。

## yaml 的一些知识

[Introduction to YAML: Creating a Kubernetes deployment](https://www.mirantis.com/blog/introduction-to-yaml-creating-a-kubernetes-deployment/)


笔者个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)
