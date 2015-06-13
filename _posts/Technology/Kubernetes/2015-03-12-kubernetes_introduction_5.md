---
layout: post
title: Kubernetes volume
category: 技术
tags: Kubernetes
keywords: CoreOS Docker Kubernetes Volume
---

## 简介


A Volume is a directory, possibly with some data in it, which is accessible to a Container. Kubernetes Volumes are similar to but not the same as Docker Volumes.

A Pod specifies which Volumes its containers need in its ContainerManifest property.

A process in a Container sees a filesystem view composed from two sources: a single Docker image and zero or more Volumes. A Docker image is at the root of the file hierarchy. Any Volumes are mounted at points on the Docker image; Volumes do not mount on other Volumes and do not have hard links to other Volumes. Each container in the Pod independently specifies where on its image to mount each Volume（一个pod中的container各自挂自己的volume）. This is specified a VolumeMounts property.

The storage media (Disk, SSD, or memory) of a volume is determined by the media of the filesystem holding the kubelet root dir (typically `/var/lib/kubelet`)(volumn的存储类型（硬盘，固态硬盘等）是由kubelet所在的目录决定的). There is no limit on how much space an EmptyDir or PersistentDir volume can consume（大小也是没有限制的）, and no isolation between containers or between pods.（在一个pod的container之间以及pod之前也没有隔离（是不指，大家都挂了/tmp,就都可以用这块））

（以后版本，会尽量让大家可以控制存储类型和存储大小）

## Types of Volumes

目前支持三种类型

### EmptyDir（仅container或container之间使用）

An EmptyDir volume is created when a Pod is bound to a Node. It is initially empty, when the first Container command starts. Containers in the same pod can all read and write the same files in the EmptyDir（这是pod之间信息共享的另一种方式）. When a Pod is unbound, the data in the EmptyDir is deleted forever.

Some uses for an EmptyDir are:

- scratch space, such as for a disk-based mergesort or checkpointing a long computation.
- a directory that a content-manager container fills with data while a webserver container serves the data.
Currently, the user cannot control what kind of media is used for an EmptyDir. If the Kubelet is configured to use a disk drive, then all EmptyDirectories will be created on that disk drive. In the future, it is expected that Pods can control whether the EmptyDir is on a disk drive, SSD, or tmpfs.

### HostDir（和主机共同使用某个目录）

A Volume with a HostDir property allows access to files on the current node.

Some uses for a HostDir are:

- running a container that needs access to Docker internals; use a HostDir of /var/lib/docker.
- running cAdvisor in a container; use a HostDir of /dev/cgroups.

Watch out when using this type of volume, because:

- pods with identical configuration (such as created from a podTemplate) may behave differently on different nodes due to different files on different nodes.
- When Kubernetes adds resource-aware scheduling, as is planned, it will not be able to account for resources used by a HostDir.

### GCEPersistentDisk

Important: You must create a PD using gcloud or the GCE API before you can use it
这个有限制， 就不多谈了

## Sample

### EmptyDir

    apiVersion: "v1beta1"
    id: "share-apache2-controller"
    kind: "ReplicationController"
    desiredState:
      replicas: 1
      replicaSelector:
        name: "share-apache2"
      podTemplate:
        desiredState:
          manifest:
            version: "v1beta1"
            id: "share-apache2"
            containers:
              - name: "share-apache2-1"
                image: "docker-registry.sh/myapp"
                ports:
                  - containerPort: 8080
                volumeMounts:
                  - name: data
                    mountPath: /data
              - name: "share-apache2-2"
                image: "docker-registry.sh/apache2"
                ports:
                  - containerPort: 80
                volumeMounts:
                  - name: data
                    mountPath: /data
            volumes:
              - name: data
                source:
                  emptyDir: {}
        labels:
          name: "share-apache2"
    labels:
      name: "share-apache2"
      
此时，share-apache2-1 container对`/data`目录所做操作都将反映到 share-apache2-2的`/data`目录中。

