---

layout: post
title: rerank微调
category: 架构
tags: MachineLearning
keywords: llm emebedding

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

* TOC
{:toc}

## 简介（未完成）


## 为什么用rerank

是使用elasticsearch的retrieval召回的内容相关度有问题，多数情况下score最高的chunk相关度没问题，但是top2-5的相关度就很随机了，这是最影响最终结果的。我们看了elasticsearch的相似度算法，es用的是KNN算法（开始我们以为是暴力搜索），但仔细看了一下，在es8的相似度检索中，用的其实是基于HNSW（分层的最小世界导航算法），HNSW是有能力在几毫秒内从数百万个数据点中找到最近邻的。为了检索的快速，HNSW算法会存在一些随机性，反映在实际召回结果中，最大的影响就是返回结果中top_K并不是我们最想要的，至少这K个文件的排名并不是我们认为的从高分到低分排序的。

因为在搜索的时候存在随机性，这应该就是我们在RAG中第一次召回的结果往往不太满意的原因。但是这也没办法，如果你的索引有数百万甚至千万的级别，那你只能牺牲一些精确度，换回时间。这时候我们可以做的就是增加top_k的大小，比如从原来的10个，增加到30个。然后再使用更精确的算法来做rerank，使用一一计算打分的方式，做好排序。

## 微调

微调数据集格式为[query，正样本集合，负样本集合]。微调在Embeding模型与Reranker模型采用同类型数据集，并将语义相关性任务视为二分类任务，采用BCE作为损失函数。

https://zhuanlan.zhihu.com/p/704562748 未细读