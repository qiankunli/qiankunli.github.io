---

layout: post
title: 服务治理框架的几个点
category: 技术
tags: Architecture
keywords: dubbo micro service

---

## 简介（未完成）

## 三大基本套路

## 从0 到 1 实现一个服务治理框架

## 路由

Router
请求路由器，由服务端配置，在客户端工作，帮助调用方找到合适的被调用实例。

## 集群


假设一个服务有3个实例ABC，称为集群cluster，客户端去调用A

1. failfast，A 实例故障，则调用失败
2. failover，A 实例故障，则重试BC
3. failsafe，A 实例故障，返回实现配置的默认值

## Filter

请求过滤器，扩展接口，允许在调用链上加入自定义的逻辑。

![](/public/upload/architecture/service_manage_1.png)

