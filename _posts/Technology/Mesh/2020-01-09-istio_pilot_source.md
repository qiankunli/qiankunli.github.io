---

layout: post
title: Pilot源码分析
category: 技术
tags: Mesh
keywords: Go

---

## 前言（未完成）

* TOC
{:toc}

[Istio Pilot代码深度解析](https://www.servicemesher.com/blog/201910-pilot-code-deep-dive/)

## pilot-agent

![](/public/upload/mesh/pilot_agent.png)

1. 所谓sidecar 容器， 不是直接基于envoy 制作镜像，容器启动后，entrypoint 也是envoy 命令
2. sidecar 容器的entrypoint 是 `/usr/local/bin/pilot-agent proxy`，首先生成 一个envoyxx.json 文件，然后 使用 exec.Command启动envoy
3. 进入sidecar 容器，`ps -ef` 一下， 是两个进程

        ## 具体明令参数 未展示
        UID        PID  PPID  C STIME TTY          TIME CMD
        1337         1     0  0 May09 ?        00:00:49 /usr/local/bin/pilot-agent proxy
        1337       567     1  1 09:18 ?        00:04:42 /usr/local/bin/envoy -c envoyxx.json


