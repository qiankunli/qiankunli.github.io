---

layout: post
title: 如何学习tensorflow
category: 架构
tags: MachineLearning
keywords: tensorflow 

---

## 简介

* TOC
{:toc}


tensorflow 如此复杂，以至于如何学习tensorflow 也是门学问，根据踩过的坑，汇总一下学习路径

1. 机器学习基本原理
    1. 基本原理：前向后向，梯度下降（自动微分），特征工程
    2. 可以进行矩阵推导
    3. dnn,cnn,gnn
    4. 具体领域的模型，比如推荐模型等，发展脉络
1. 应用层面，对应工程代码 tf/pytorch
    1. 高层api 使用：抽象层度比较高， 优先 estimator（单机和分布式代码一致） ，kearas 也可以学一下
    2. 中层api ：tf 原生api，自己定义几个 variable 乘一乘，计算y_label，loss，执行Optimizer
2. 原理层面
    1. 机器学习框架的基本思路，Model/Layer抽象、前向后向，可以先参考纯python 代码实现 降低难度，推荐《用python实现深度学习框架》
    2. k8s 单机原理：又分为python 层（feature_column/layer/optimizer 等概念如何转换为算子、计算图）和 core 层（计算图的切分和执行）
    3. 分布式原理：Client-Master-Worker
3. 工程层面
    1. 分布式训练：ps/allreduce（基本原理、api、适用场景比如大模型等）; collective communication比如NCCL；
    2. embedding：基本原理；tf原生支持代码；各公司二次开发框架优化
    3. 弹性训练：框架、operator、调度器volcano的相关支持
    4. k8s：对应的operator；共享存储pv/pvc；日志采集
    5. 推理：镜像管理、灰度、ab等
    6. 资源利用率：gpu的基本原理；分时复用；更优化的框架
3. 扩展tf
    1. 扩展的基本方向：embedding ；通信
    2. 扩展的基本手段：自定义算子，底层能力形成c api ==> python 层函数
    1. 知道推荐系统模型（大模型）的特性及优化方向、常见优化手段，开源扩展框架tensornet、deeprec等
    2. python调用c
    3. 自定义算子


