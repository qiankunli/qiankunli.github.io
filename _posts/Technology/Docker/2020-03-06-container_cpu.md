---

layout: post
title: 容器狂占cpu怎么办？
category: 技术
tags: Docker
keywords: jib

---

## 简介（未完成）

* TOC
{:toc}

笔者曾经碰到一个现象， 物理机load average 达到120，第一号进程的%CPU 达到了 1091，比第二名大了50倍，导致整个服务器非常卡，本文尝试分析和解决这个问题


docker ps -q | xargs docker inspect -f '{{.State.Pid}} {{.Id}}'

## 如何感知某个项目占用了过多的cpu

### linux

top 命令找到`%CPU` 排位最高的进程id，根据`docker ps -q | xargs docker inspect -f '{{.State.Pid}} {{.Id}}'`找到对应的容器

### docker


## cpu 隔离的正确姿势







