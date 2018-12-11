---

layout: post
title: jib源码分析之细节
category: 技术
tags: Docker
keywords: jib

---

## 简介

* TOC
{:toc}

阅读本文前，建议事先了解下 [jib源码分析及应用](http://qiankunli.github.io/2018/11/19/jib_source.html)

### 几个问题

通过docker registry v2 api，是可以上传镜像的

1. jib 的最后，是不是也是调用 docker registry v2 api？ 比如对于golang语言 就有针对 registry api 的库 [github.com/heroku/docker-registry-client](https://github.com/heroku/docker-registry-client)
2. 重新梳理 jib runner 的结构
3. jib maven plugin 与 jib-core 分工的边界在哪里？ 直接的代码调用，使用jib-core 即可
4. 源代码的调用 最终 是调用了 BuildSteps.run （它的前面实质都是在搞信息采集，准备上下文），也就是 jib的核心原理是在 jib-core 中体现的
5. BuildSteps.run 之前和之后主要两个事情，信息采集，开始干活儿。是否可以做一个假设，底层使用registry api 发送数据。 那么jib的难点主要有几个部分

	1. 如何将不同的数据分layer
	2. 复杂的流程 如何 以一个 简单的链式调用 呈现。这个复杂流程的基本抽象是什么？基本的单位是什么？**一定有一个基本单元类 ，然后有一个机制，将这些基本单元类串在一起，最终呈现给调用方。**

	
 针对 这几个问题 还没有比较好的解答


个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)