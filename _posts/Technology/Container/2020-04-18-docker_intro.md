---

layout: post
title: docker 架构
category: 技术
tags: Container
keywords: docker

---

## 简介

* TOC
{:toc}

容器将操作系统虚拟化，虚拟机则是将硬件虚拟化。容器多用于表示软件的一个标准化单元。
## 容器运行时

Linux 提供了cgroup 和 namespace 两大系统功能，是容器的基础，但是要把进程运行在容器中，还需要有便捷的SDK 或命令来调用Linux 的系统功能，从而创建出容器，容器运行时 就是容器进程运行和管理的工具。 

![](/public/upload/container/container_runtime.png)

1. runC是最常用的 容器低层运行时，不包含镜像管理，它假定容器的文件包已经从镜像里解压出来并存放于文件系统中。runC 创建的容器需要手动配置网络才能与其他容器或者网络节点连通。 ——**镜像并不是运行容器所必须的**。
2. containerd是最常用的 容器高层运行时，提供镜像下载、解压等功能， 不包含镜像构建、上传等功能
3. 再往上，Docker 提供了许多 UX 增强功能，比如ps/system prune 这些。 UX 增强的功能并不是 Kubernetes 所必须的

## Docker Daemon 调用栈分析

docker在 1.11 之 后，被拆分成了多个组件以适应 OCI 标准。拆分之后，其包括 docker daemon， containerd，containerd-shim 以及 runC。组件 containerd 负责集群节点上容器 的生命周期管理，并向上为 docker daemon 提供 gRPC 接口。

![](/public/upload/docker/docker_call_stack.png)

在一个docker 节点上执行 `ps -ef | grep docker`

```
/usr/bin/dockerd
docker-containerd  
docker-containerd-shim $containerId /var/run/docker/libcontainerd/  $containerId docker-runc
docker-containerd-shim $containerId /var/run/docker/libcontainerd/  $containerId docker-runc
...
```
1. dockerd 是docker-containerd 的父进程， docker-containerd 是n个docker-containerd-shim 的父进程。
2. Containerd 是一个 gRPC 的服务器。它会在接到 docker daemon 的远程请 求之后，新建一个线程去处理这次请求。依靠 runC 去创建容器进程。而在容器启动之后， runC 进程会退出。
3.  runC 命令，是 libcontainer 的一个简单的封装。这个工具可以 用来管理单个容器，比如容器创建，或者容器删除。

## 排查工具

1. 使用` kill -USR1 <pid>`命令发送 USR1 信号给 docker daemon， docker daemon 收到信号之后，会把其所有线程调用栈输出 到文件 /var/run/docker 文件夹里。
2. 我们可以通过` kill -SIGUSR1 <pid>` 命令来输出 containerd 的调用栈。不同的是，这次调用栈会直接输出到 messages 日志。
3. 向 kubelet 进程发送 SIGABRT 信号，golang 运行时就会帮我们输出 kubelet 进程的所有调 用栈。需要注意的是，这个操作会杀死 kubelet 进程。


