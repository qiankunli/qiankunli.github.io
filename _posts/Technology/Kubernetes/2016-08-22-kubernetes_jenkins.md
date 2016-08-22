---
layout: post
title: Integrating Kubernetes and Jenkins
category: 技术
tags: Kubernetes
keywords: Jenkins Docker Kubernetes
---

## 简介

## 项目结构

![kubernetes_demo_project_structure.png](/public/upload/kubernetes/kubernetes_demo_project_structure.png "")


## jenkins

jenkins涉及到的配置如下

### git
	
    git@xxx:bert/k8s-web-demo.git
### maven

	clean install -Ptest-out -Dmaven.test.skip=true 
### shell

	#!/bin/bash
	set +e
    REGISTRY_ADDRESS=192.168.3.56:5000
    IMAGE_NAME=$JOB_NAME
    FULL_IMAGE_NAME=$REGISTRY_ADDRESS/$IMAGE_NAME
    /usr/bin/docker build -t $FULL_IMAGE_NAME $WORKSPACE | tee Docker_build_result.log
    echo ">>>docker push image"
    /usr/bin/docker push $FULL_IMAGE_NAME 
    ## delete old image and container in cluster
    ## delete old if exist or just update image,rolling update
    echo ">>>k8s create rc"
    /usr/local/bin/kubectl create -f $WORKSPACE/k8s/rc.yaml
    ## if not exist create
    echo ">>>k8s create service"
    /usr/local/bin/kubectl create -f $WORKSPACE/k8s/svc.yaml

