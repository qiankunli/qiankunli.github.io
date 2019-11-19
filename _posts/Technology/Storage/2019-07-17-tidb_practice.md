---

layout: post
title: TIDB实践
category: 技术
tags: Storage
keywords: TIDB

---

## 前言

* TOC
{:toc}

[新一代数据库TiDB在美团的实践](https://tech.meituan.com/2018/11/22/mysql-pingcap-practice.html)

[TiDB 用户文档](https://pingcap.com/docs-cn/)未读

## TIDB接入

tidb 通过tidb server 对外提供服务，tidb server 是无状态的，可以部署多台，每一个tidb server 都可以像 msyql server 一样对外提供服务。在业务接入上，应避免业务方只配置某几个tidb server，导致其负载过高。办法

1. 使用dns 做 tidb server 间的负载均衡


## 监控报警

## 服务部署

ansible 脚本

### 物理机方式/ansible

### Kubernetes方式

