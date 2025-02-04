---

layout: post
title: after transformer
category: 架构
tags: MachineLearning
keywords:  gcn

---

## 简介

* TOC
{:toc}

## moe

在构建MoE语言模型时，通常会将Transformer中的某些FFN替换为MoE层（MOE FFN，陆续还出现了MOE Attention）。具体来说，MoE层由多个专家组成，每个专家的结构与标准的FFN相同。每个token会被分配给一个或两个专家。MoE模型的推理过程主要包含三个阶段:
1. 路由计算:通过路由器计算专家选择概率
2. 专家选择:基于概率选择Top-K个专家
3. 并行计算:选中的专家并行处理输入并聚合结果

![](/public/upload/machine/moe.png)

MoE架构的主要优势在于其能够通过激活部分专家来降低计算成本，从而在扩展模型参数的同时保持计算效率。然而，现有的MoE架构在专家专业化方面面临挑战，具体表现为知识混杂和知识冗余。这些问题限制了MoE模型的性能，使其无法达到理论上的性能上限。
1. 知识混杂（Knowledge Hybridity）：现有的MoE实践通常使用较少的专家（例如8或16个），因此分配给特定专家的token可能涵盖多种知识。这导致每个专家需要在其参数中学习多种不同类型的知识，而这些知识难以同时被有效利用。
2. 知识冗余（Knowledge Redundancy）：分配给不同专家的token可能需要一些共同的知识，导致多个专家在其参数中收敛于共享知识，从而造成专家参数的冗余。

## deepseek

deepseek有很多自己特色性的技术。从这些特色性的技术中可以看到，他们的出发点都是尽最大努力去减少人工智能中的各项成本。例如：
1. 不依赖于对用于训练的数据进行人工打标签。
2. 混合专家架构（Mixture of Experts：MoE）。
3. 多头潜在注意力（Multi-Head Latent Attention，MLA）

### MTP

MTP的研究并不是大模型时代的新物种，而是在第一代Transformer base的模型上，就有相应的研究了。

为什么要做MTP(Multi-Token Prediction)? 当前主流的大模型(LLMs)都是decoder-base的模型结构，也就是无论在模型训练还是在推理阶段，对于一个序列的生成过程，都是token-by-token的。每次在生成一个token的时候，都要频繁跟访存交互，加载KV-Cache，再通过多层网络做完整的前向计算。对于这样的访存密集型的任务，通常会因为访存效率形成训练或推理的瓶颈。

MTP核心思想：通过解码阶段的优化，将1-token的生成，转变成multi-token的生成，从而提升训练和推理的性能。具体来说，在训练阶段，一次生成多个后续token，可以一次学习多个位置的label，进而有效提升样本的利用效率，提升训练速度；在推理阶段通过一次生成多个token，实现成倍的推理加速来提升推理性能。