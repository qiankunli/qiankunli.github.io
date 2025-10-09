---

layout: post
title: Agent工具
category: 技术
tags: MachineLearning
keywords: agent software

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

## 简介(未完成)

## Browser Use

https://mp.weixin.qq.com/s/rfdYPW2GRFybVd9DEbYOZg

https://mp.weixin.qq.com/s/HNPuJVr0j44Y2PhcgaE-ew

[如何让AI“看懂”网页？拆解 Browser-Use 的三大核心技术模块](https://mp.weixin.qq.com/s/KLk-m_E2zx_q-v4ZLobetw)
1. Browser Use 是一种基于 AI 模型的**浏览器自动化工具**，基于 Playwright（浏览器自动化框架）
1. Browser Use 是是一个开源的Python库，基于 LangChain 生态构建


Playwright 是一个底层浏览器自动化驱动层，负责：
1. 启动、控制浏览器进程；
2. 执行 DOM 操作、点击、输入、截图；
3. 模拟网络请求、地理位置、时间；
4. 拦截 / 注入脚本；
5. 提供 API 接口给上层应用调用。
它相当于“浏览器的远程控制器”，用来跑测试、爬取数据、或者做自动化交互。