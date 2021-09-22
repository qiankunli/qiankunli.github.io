---

layout: post
title: AI云平台
category: 架构
tags: MachineLearning
keywords:  ai platform

---

## 简介

* TOC
{:toc}

海量数据标注 + 大规模计算 + 工程化（python或c++）=AI系统

![](/public/upload/machine/ai_develop_progress.jpeg)

## 虎牙

[互动直播场景下的AI基础设施建设](https://time.geekbang.org/qconplus/detail/100059720)

手动时代

![](/public/upload/machine/huya_ai_manual.png)

AI平台的定位：面向算法工程师，围绕AI 模型的全生命周期去提供一个一站式的机器学习服务平台。

![](/public/upload/machine/huya_platform_1.png)
![](/public/upload/machine/huya_platform_2.png)


## 腾讯

[高性能深度学习平台建设与解决业务问题实践](https://time.geekbang.org/qconplus/detail/100059719)构建公司统一的大规模算力，方便好用的提供GPU

![](/public/upload/machine/tecent_platform_2.png)

## 其它

深度学习平台的搭建，将遇到诸多挑战，主要体现在以下方面：

1. 数据管理自动化。在深度学习的业务场景中，从业人员会花费大量的时间获取和转换建立模型需要的数据。数据处理过程中还将产生新的数据，这些数据不单单应用于本次训练，很可能用于后续推理过程。并且新 生成的数据不需要传给数据源，而是希望放置在新的存储空间。这需要基础平台提供可扩展的存储系统。
2. 资源的有效利用。深度学习相关的应用是资源密集型，资源使用波动大，存在峰值和谷值。在应用开始运行的时候，快速获取计算资源；在应用结束后，回收不适用的计算资源对于资源利用率的提升相当重要。数据处理、模型训练和推理所使用的计算资源类型和资源占用时间有所不同，这就需要计算平台提供弹性的资源供应机制。
3. 屏蔽底层技术的复杂性