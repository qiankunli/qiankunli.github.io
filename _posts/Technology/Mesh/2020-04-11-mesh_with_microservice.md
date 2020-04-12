---

layout: post
title: 与微服务框架整合
category: 技术
tags: Mesh
keywords: mesh microservice

---

## 前言（未完成）

* TOC
{:toc}

[落地三年，两次架构升级，网易的Service Mesh实践之路](https://mp.weixin.qq.com/s/2UIp6l1haH6z6ISxHM4UjA)

服务框架在微服务架构中占据核心位置，因此，使用 Service Mesh 来替换正在使用的微服务框架，除了需要在 Service Mesh 数据面和控制面组件中对服务注册发现、RPC 协议、配置下发进行扩展之外，还要对现有的上层研发工作台、运维效能平台等支撑平台进行兼容设计

## 与微服务框架整合

||数据面|控制面|
|---|---|---|
|服务注册发现|||
|RPC 协议|||
|配置下发|||

### 数据平面

[陌陌 Service Mesh 架构的探索与实践](https://mp.weixin.qq.com/s/EeJTpAMlx_mFZp6mh2i2xw) 

1. 平滑升级
2. Agent 容灾
3. 代理性能

## 周边支撑

## 部署架构

## 其它

笔者在最开始推进 service mesh 落地时，潜意识中过于以“追求技术”来影响方案的取舍（比如偏好使用istio），陌陌这把篇文章很好的指导了技术与实践的取舍[陌陌 Service Mesh 架构的探索与实践](https://mp.weixin.qq.com/s/EeJTpAMlx_mFZp6mh2i2xw)

1. 当前架构的痛点是 SDK 升级与跨语言，而不是缺少控制平面功能；
2. 现阶段我们关注的核心收益均由数据平面产生，因此整体方案中将着重进行数据平面 Agent 的建设

[陌陌 Service Mesh 架构的探索与实践](https://mp.weixin.qq.com/s/EeJTpAMlx_mFZp6mh2i2xw) 这篇文章的思路非常好，从业界方案 ==> 自己痛点 ==> 数据面和控制面的侧重 ==> 提炼数据面的几个关注点 ，条清缕析的给出了一条实践路线。