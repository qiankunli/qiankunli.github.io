---

layout: post
title: 平台支持类系统的几个点
category: 技术
tags: Architecture
keywords: abtest

---

## 简介（未完成）

此处主要指 客户端版本管理系统、配置中心系统、abtest 系统、上传系统、推送系统等 跟app 业务不直接关联，但仍是每个app 的必要的 “业务” 系统。


## 几个基本的功能点

### 条件匹配

各种数据 总会越来越多，因避免后端数据量 影响 响应时间。

手段：diff  等

### 个性化下发

### 灰度下发

### 支持多应用

## 基本准备

### 客户端信息 标准化

### 安全

### 权限管理

### 下发进度统计

### 精确到个人的日志记录

## 性能

### 缓存/打掉缓存

### 推拉

## 维护

### 警惕app 发版、新数据 对系统的影响

## 生命周期

平台支持类的项目，基本在跟着 客户端发版走，所以很多开发节点 要跟得上 客户端发版。

## 通用架构

两个方案 一个侧重客户端，一个侧重服务端。理论上，第一个更轻省些，但对客户端的能力要求较高。业务架构 影响/被影响 团队架构，笔者第一次有如此切身的体会。  

### 一切皆配置

![](/public/upload/architecture/client_support_client_1.png)

### 各自为政

![](/public/upload/architecture/client_support_client_2.png)