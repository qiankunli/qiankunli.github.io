---

layout: post
title: rl微调
category: 技术
tags: MachineLearning
keywords: deepresearch deepsearch

---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['$$', '$$']], // 支持 $和$$ 作为行内公式分隔符
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

## 简介(未完成)


## 实践

[Multi-Agent 的灵活编排之路](https://mp.weixin.qq.com/s/0c8hTMdIALjYdGZkmwLFDg) 案例，multiagent 背景下，训练plannning 模块生成plan（每一个step 是一个选中agent及其要解决的问题）

[强化学习用于端到端文档解析Infinity Parser](https://github.com/infly-ai/INF-MLLM/tree/main/Infinity-Parser)核心是训练数据的处理以及强化奖励函数的设计，通过优化多个方面的奖励函数来训练模型，使其对文档布局更加敏感。
1. 奖励函数的设计，在强化阶段，使用Qwen2.5-VL-7B模型进行微调，采用GRPO进行强化学习优化，通过生成一组候选Markdown输出，并使用多方面奖励函数进行评估，从而优化模型。分别是编辑距离奖励（Edit Distance Reward）、计数奖励（Count Reward）以及顺序奖励（Order Reward）。最后总合成一个奖励函数：$R_{multi-aspect} = R_{dist} + R_{count} + R_{order}$
    1. 编辑距离奖励（Edit Distance Reward）：基于预测输出和参考输出之间的归一化Levenshtein距离，衡量语义和格式的差异；
    2. 计数奖励（Count Reward）：鼓励准确的段落分割，通过惩罚缺失或多余的段落来实现。
    3. 顺序奖励（Order Reward）：通过计算参考和预测段落之间的成对逆序数来衡量序列级别的保真度。
2. 训练数据集，包括Infinity-Doc-55K数据集，包含55,066个样本。合成方式是通过HTML模板和浏览器渲染生成，具体使用方式是：从维基百科、网络爬虫和在线语料库等来源收集文本和图像，并使用Jinja模板将采样的内容注入预定义的单列、双列或三列 HTML布局中。这些页面通过浏览器引擎渲染成扫描文档，随后进行自动过滤以去除低质量或重叠的图像。通过解析原始HTML 提取真实标注，生成对齐的Markdown表示。最终形成的数据集涵盖七个不同的文档领域，包括财务报告、医疗报告、学术论文、书籍、杂志、网页等。