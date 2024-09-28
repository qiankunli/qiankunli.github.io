---

layout: post
title: 大模型推理tips
category: 架构
tags: MachineLearning
keywords: llm vLLM

---

* TOC
{:toc}

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## 简介（未完成）

## 以kvcache为核心的分布式架构

Mooncake 采用了以 KVCache 为中心的分离式推理架构，主要由三个核心部分组成：

1. Prefill 池：这个部分负责集中管理所有的预填充阶段的计算任务。
2. Decoding 池：这个部分集中处理所有解码阶段的任务。
3. KVCache 池：这个部分负责存储所有中间过程中应用到的 KVCache，并决定何时使用这些缓存，何时释放它们。

Context Caching 

![](/public/upload/machine/context_caching.jpg)