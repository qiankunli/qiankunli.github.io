---

layout: post
title: docker环境下的持续发布
category: 技术
tags: Docker
keywords: 持续交付

---

## 简介（待填充）

* TOC
{:toc}


阅读前建议先看下 前文[docker环境下的持续构建](http://qiankunli.github.io/2018/11/18/ci_in_docker.html)

[流水线即代码](http://insights.thoughtworkers.org/pipeline-as-code/)具体实施过程中，考虑到项目，尤其是遗留项目当前的特点和团队成员的“产能”，我们会先将构建和部署自动化；部署节奏稳定后，开始将单元测试和代码分析自动化；接着可以指导测试人员将验收测试自动化；然后尝试将发布自动化。


一个发布系统的职责

1. 谁什么时候发布了什么东西，发布结果是成功还是失败
2. 必要的审批流程和权限管理
3. 回滚
4. 对接底层运行环境，分为以下几种

	1. 物理机
	2. 基于docker 的 PaaS 集群
	3. 很多时候，要应对底层运行环境的变迁，比如笔者最开始选用了marathon，后续又逐步的迁到了k8s。

5. 为持续构建系统 提供调用接口


## 对接marathon/k8s

1. 必须将调度接口 对开发人员隐藏掉，因为开发人员对这些概念部署
2. 有能力对开发人员的危险操作进行拦截

### 调度

### 状态回显

## 对环境隔离的支持

环境变量即回调

个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)