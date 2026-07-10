---

layout: post
title: 群聊
category: 技术
tags: MachineLearning
keywords: agent team

---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$']], // 支持 $和$$ 作为行内公式分隔符
      displayMath: [['$$', '$$']], // 块级公式分隔符
    },
    svg: {
      fontCache: 'global'
    }
  };
</script>
<script async src="/public/js/mathjax/es5/tex-mml-chtml.js"></script>

* TOC
{:toc}

## 简介


## 场景

不是所有 Agent 场景都需要群聊。引入群聊的代价是显著的，无论是用户的使用习惯，还是背后的 Agent Infra，包括技术架构、上下文管理、权限治理、并发调度、成本归因，每一项的复杂度都会迅速拉高。
1. 跨领域协作和长链路工作流。
  1. 跨领域协作需要多智能体，本质原因是 Agent 的上下文和记忆等是有限的，和人一样，一个 Agent 塞太多职责，注意力一分散，每件事都做不好。跨领域是注意力在空间上的分散，比如需求分析、编码、测试本身就是不同领域的工作，拆成多个 Agent 按阶段接力，每段上下文干净、注意力集中，中间状态可持久化，断了能从断点续跑，更健壮，并且要发生在所有人都能看到、都能介入、都能纠偏的频道里。
  2. 长链路则是注意力在时间上的衰减。比如从需求到发布的软件研发流程，跑几小时甚至几天，上下文越积越长，Agent 对早期信息的注意力不断被稀释，推理质量持续下降。
2. 多智能体治理
3. 沉淀组织级知识


## 对 Agent Infra提出了哪些新的挑战？


