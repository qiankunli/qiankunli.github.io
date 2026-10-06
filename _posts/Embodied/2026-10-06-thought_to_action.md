---

layout: post
title: 用户语言如何转换为机器人动作
category: 技术
tags: MachineLearning
keywords: deepresearch deepsearch

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


有几条路线
1. 分层规划与控制，根据动作表达粒度的不同又分为
    1. 语言 → LLM → Command／Skill（结构化指令） → 运动 Policy → 动作。 
    2. 分层规划与控制，语言 → LLM → 自然语言子任务 → 运动 Policy → 动作。  
3. Flat VLA,语言 + 图像／身体状态 → VLA → 动作

## 如何屏蔽不同microduck 舵机和身体的差异

GPT-3.5推出之前NLP领域，那时，Pretrained Model/Foundation Model已经出现，但情感分析、信息抽取、问答等任务，仍经常需要各自的数据、微调模型和专门接口。当前机器人策略仍具有较强的 embodiment dependence（身体依赖性）。基础模型正在提高跨任务与跨身体的知识复用，但对新身体的可靠控制，往往仍需要接口适配、目标身体数据微调，或经过训练的在线适应机制。且机器人的统一更困难：不同 NLP 任务大多共享文本输入输出，而不同机器人不仅动作接口不同，同一个动作产生的物理结果也不同。

| 差异 | 主要处理方式 |
|---|---|
| 舵机品牌、通信协议、设备 ID | SDK、硬件 adapter 封装 |
| 零点、转向、单位、关节顺序 | 标定与动作转换 |
| 力矩、响应延迟、摩擦、重量略有变化 | 仿真参数辨识、Domain Randomization、必要时微调 |
| 腿明显更长、重心变化、关节布局变化 | 更新身体模型，并重新训练或适配策略 |

几个思路
1. 把“如何根据身体反馈修正动作”也学进模型，使换组件更多变成运行时适应，而不是重新训练
2. 


## 工程

## 长期：人形机器人和 Microduck 能共用吗？
PS： 对同一种构型的机器人，能搞定一个机制 部件、尺寸等差异已经有很大价值了。 沉淀下来的思路/方法论、框架等迁移成本并不是很高。

把动作向量从 14 维补齐到更多维，解决不了动作语义和动力学差异。
1. 框架可以共用：数据格式、训练流程、推理服务。
2. 部分模型参数可以共用：例如视觉、语言 backbone


