---

layout: post
title: 如何学习机器学习
category: 技术
tags: MachineLearning
keywords: 深度学习

---


## 前言

机器通过分析大量数据来进行学习。比如说，不需要通过编程来识别猫或人脸，它们可以通过使用图片来进行训练，从而归纳和识别特定的目标。

ML 是一个通过算法和统计模型从数据中学习知识的学科，当我们遇到的问题可以用一套可管理的确定性规则（且随着数据变化并不需要变更规则）来解决时，这类问题便不需要ML。

![](/public/upload/machine/machine_learning_xmind.png)

人工神经网络具有自学习和自适应能力，可以通过预先提供的一批相对应的输入输出数据，分析两者的内在关系和规律， 最终通过这些规律形成一个**复杂的非线性系统函数**，这种学习分析过程被称作训练。PS：模型可以看作一个函数，它模拟了人类智能的运行方式。在模型训练中，你所做的就是解出这个函数中未知变量的值。

机器学习（Machine Learning），它指的是这样一种计算过程：对于给定的训练数据（Training samples），选择一种先验的数据分布模型（Models），然后借助优化算法（Learning Algorithms）自动地持续调整模型参数（Model Weights / Parameters），从而让模型不断逼近训练数据的原始分布。这个持续调整模型参数的过程称为“模型训练”（Model Training）。模型的训练依赖于优化算法，基于过往的计算误差（Loss），优化算法以不断迭代的方式，自动地对模型参数进行调整。由于模型训练是一个持续不断的过程，那么自然就需要一个收敛条件（Convergence Conditions），来终结模型的训练过程。一旦收敛条件触发，即宣告模型训练完毕。模型训练完成之后，我们往往会用一份新的数据集（Testing samples），去测试模型的预测能力，从而验证模型的训练效果，这个过程，我们把它叫作“模型测试”（Model Testing）。测试数据用于考察模型的泛化能力（Generalization），也就是说，对于一份模型从来没有“看见过”的数据，我们需要知道，模型的预测能力与它在训练数据上的表现是否一致。PS：机器学习就是找函数

## AI 的范畴

人工智能三大主义
1. 符号主义，旨在用数学和物理学中的逻辑符号来表达思维的形成，通过大量的”if-then“ 规则定义，产生像人一样的智能，这是一个自上而下的过程，包括专家系统、知识工程等。
2. 连接主义，主张智能来自神经元之间的连接，它让计算机模拟人类大脑中的神经网络及其连接机制，这是一个自下而上的过程，包括人工神经网络等。 
3. 行为主义，指基于感知行为的控制系统，使每个基本单元实现自我优化和适应，这也是一个自下而上的过程，典型的代表有进步算法、多智能体等。
然而，基于人工规则的「符号主义」的智能水平受限于规则的规模，而以概率统计为基础的「连接主义」在推理时的幻觉总是不可避免，同时「行为主义」从环境中学习的能力很难有效地迁移泛化。单项能力的不足反而为优势互补提供了空间，比如大模型与强化学习结合的推理模型（LRM）、环境反馈赋能的大模型智能体（Agent）等。

[浅谈以数据为中心的人工智能](https://mp.weixin.qq.com/s/siBNAdxEO7c0DM_fh3UAzA)过去传统的人工智能是以模型为中心的，在这样的过程中大家更关注如何设计并训练更好的模型。但随着开源框架不断落地之后，大家开始关注数据能够带来的提升。以数据为中心的 AI 也成为了 AI 的新趋势：
1. 经典路线（以模型为中心的人工智能）：关心如何迭代模型来提高效能
2. 新趋势（以数据为中心的人工智能）：关心如何系统性地迭代数据输入和数据标签来提高效能

机器学习是由人工智能的连接主义发展形成的一个重要领域分支。从广义上来说，这是一类方法学，当我们从问题世界观测到一些数据，如果没有能力或者没有必要建立严格的物理模型时，可以使用数学方法从这些数据中推理出数学模型。这里的数学模型一般是没有详细的物理解释的，不过会在输入和输出的关系中反映实际问题。

![](/public/upload/machine/what_is_ai.png)

机器学习下面应该是表示学习（Representation Learning），即概括了所有使用机器学习挖掘表示本身的方法。相比传统 ML 需要手动设计数据特征，这类方法能自己学习好用的数据特征。整个深度学习也是一种表示学习，通过一层层模型从简单表示构建复杂表示。

机器学习的主要目的是把人类归纳经验的过程，转化为由计算机自己来归纳总结的过程。经典的机器学习模型包含 线性回归、Logistic回归、决策树、支持向量机、贝叶斯模型、随机森林、集成模型、神经网络。深度学习起源于机器学习中的人工神经网络，所以从工作机制上讲机器学习与深度学习是完全一致的，机器学习是一种通过利用数据，训练出模型，然后使用模型预测的一种方法。

**传统的机器学习模型的一个主要瓶颈在于特征工程环节**（特征工程是把任务的原始数据转化为模型输入数据的过程），特征工程环节主要依靠手工设计，需要大量的领域专门知识，尤其针对非结构化数据（语言、文本、图像等），设计有效的特征成为大多数机器学习任务的主要瓶颈。深度学习是机器学习中一种对数据进行**表征学习**的方法，可以直接学习低级感知层数据（如图像像素和语音信号），且特征学习与分类模型一起训练（端到端的学习），节约精力也更通用化。 

![](/public/upload/machine/svm_vs_cnn.png)

[ 图解机器学习：人人都能懂的算法原理](https://mp.weixin.qq.com/s?__biz=MzI5ODQxMTk5MQ==&mid=2247487773&idx=2&sn=ae1eadb1bbe0b5f83bd97c1874dbf3d9&chksm=eca763a5dbd0eab38c343c5f85d38e5cce087bcce39dd60ec7e94caadd69e8d0b2f2468cc6ae&cur_album_id=1815350871241637891&scene=190#rd)深度学习都是神经网络吗？明显并不一定是，例如周志华老师的深度森林，它就是第一个基于不可微构件的深度学习模型。

三大应用场景
1. 分类，即机器被训练来完成对一组数据进行特定的分类。在机器学习的场景中，分类算法解决分类问题也是利用相似的原理，可用的算法非常多，常见的有逻辑回归、朴素贝叶斯、决策树、随机森林、K 近邻、支持向量机，以及神经网络等等。
    1. 二分类：预测结果只有两个离散的值，如是否、1/0
    2. 多分类：预测结果是多个离散的值，如A/B/C
2. 回归，即机器根据先前标记的数据来预测未来。预测结果是连续的值，如房价的预测、库存的预测。
3. 聚类，无监督学习，将相似的样本归类在一起，如细分用户、新闻聚类。假设，你现在是一个客服系统负责人，为了减轻人工客服的压力，想把一部分常见的问题交给机器人来回复。解决这件事情的前提，就是我们要对用户咨询的商品问题先进行分组，找到用户最关心的那些问题。这种需要根据用户的特点或行为数据，对用户进行分组，让组内数据尽可能相似的的问题，就属于聚类问题，用一个词概括它的特点就是 “物以类聚”。常见的聚类算法有层次聚类、原型聚类（K-means）、密度聚类（DBSCAN）。其实，聚类算法的原理很简单，就是根据样本之间的距离把距离相近的聚在一起，在实际应用场景里，衡量样本之间距离关系的方法会更复杂，可能会用语义相似度、情感相似度等等。聚类分析较为重要的一个应用就是用户画像。

[人工智能是不是走错了方向？](https://mp.weixin.qq.com/s/tUva9bOOCV8NNurEnd9LWg)

李宏毅： 机器学习就是让机器找一个函数f。
![](/public/upload/machine/deep_learning_step.jpg)

## 机器学习的数学基础

[分层的概念——认知的基石](https://mp.weixin.qq.com/s?__biz=MzA4NTg1MjM0Mg==&mid=2657261549&idx=1&sn=350d445acf339ce19e7aab1ff19d92d0&chksm=84479e34b3301722aea0aaaa6f74656dd3e9509d70bf5719fb3992d744312bdd1484fc0c1852&mpshare=1&scene=23&srcid=1105hMUVZrVwuoX8KbtS0Vl0%23rd)

利用机器学习，我们至少能解决两个大类的问题：

1. 分类(Classification)
2. 回归(Regression)。

为了解决这些问题，机器学习就像一个工具箱，为我们提供了很多现成的的算法框架，比如：LR, 决策树，随机森林，Gradient boosting等等，还有近两年大热的深度学习的各种算法，但要想做到深入的话呢，只是会使用这些现成的算法库还不够，还需要在底层的数学原理上有所把握。比如

1. 研究优化理论，才能够有更好的思路去设计和优化目标函数；
2. 研究统计学，才能够理解机器学习本质的由来，理解为什么机器学习的方法能够使得模型一步步地逼近真实的数据分布；
3. 研究线性代数，才能够更灵活地使用矩阵这一数学工具，提高了性能且表达简洁，才能够更好地理解机器学习中涉及到的维数灾难及降维问题；
4. 研究信息论，才能够准确地度量不同概率分布之间的差异。

王天一：人工智能虽然复杂，但并不神秘。它建立在数学基础上，通过简单模型的组合实现复杂功能。在工程上，深度神经网络通常以其恒河沙数般的参数让人望而却步；可在理论上，各种机器学习方法的数学原理却具有更优的可解释性。从事高端研究工作固然需要非凡的头脑，但理解人工智能的基本原理绝非普通人的遥不可及的梦想。


## 学习路线与材料

现在学习 AI，特别是上手深度学习，已经清楚的出现了两条路子。

1. 以理论为中心，扎扎实实从数学基础开始，把数据科学、机器学习大基础夯实，然后顺势向上学习Deep Learning，再往前既可以做研究，也可以做应用创新。
2. 以工具为中心，直接从Tensorflow、Caffe、MXNET、PyTorch 这些主流的工具着手，以用促练，以练促学。

一般来说，第一条路子适合于还在学校里、离毕业还有两年以上光景的青年学生，而第二条路子适合于已经工作，具有一定开发经验的人，也适合时间有限的转型开发者，这条路见效快，能很快出成果，受到更多人的青睐。但是它也同样需要一个健康的框架，如果自己瞎撞，表面上看很快也能重复别人已经做出来的成果，但是外强中干，并不具备解决新问题的能力，而且一般来说在知识和技能体系里会存在重大的缺陷。

斯坦福大学今年上半年开了一门课程，叫做 CS20SI: Tensorflow for Deep Learning Research。可以说这门课程为上面所说的第二条路径规划了一个非常漂亮的框架。学习斯家的课程，你很容易找到一种文武双修、理论与实践生命大和谐的感觉。特别是斯家课程的课件之细致完备、练习之精到舒适，处处体现一种“生怕你学不会、学不懂”的关怀。[stanford-tensorflow-tutorials](https://github.com/chiphuyen/stanford-tensorflow-tutorials)

王天一：人工智能的价值在于落地，它的优势则是几乎所有领域都有用武之地。与其星辰大海，不如近水楼台。将自身专业的领域知识和人工智能的方法结合，以解决实际问题，才是搭上人工智能这趟快列的正确方法。 

三百多页ppt，就说比较好的学习材料[李宏毅一天搞懂深度學習](https://www.slideshare.net/tw_dsconf/ss-62245351?qid=108adce3-2c3d-4758-a830-95d0a57e46bc)

[Deep Learning](http://www.deeplearningbook.org/)

[吴恩达给你的人工智能第一课](https://mooc.study.163.com/smartSpec/detail/1001319001.htm) 这是笔者实际的入门课程

<font color="red">才云内部的课程资料</font>[适合传统软件工程师的 Machine Learning 学习路径](https://github.com/caicloud/mlsys-ladder?from=timeline)

[ 2017 斯坦福李飞飞视觉识别课程](https://github.com/caicloud/mlsys-ladder?from=timeline) 虽然说得是视觉识别，但一些机器学习的基本原理也值得一看。[CS231n课程笔记翻译：神经网络笔记1（上）](https://zhuanlan.zhihu.com/p/21462488?refer=intelligentunit) 未读

[李宏毅2020机器学习深度学习(完整版)国语](https://www.bilibili.com/video/BV1JE411g7XF?p=16) 也非常的不错。

《李沐的深度学习课》  有一点非常好，就是针对线性回归/softmax回归/感知机 都提供了一个 基于numpy 的实现以及pytorch 的简单实现。

## 实践

实践说的是：选择入门的编程语言（基本是python）以及编程语言在机器学习方面的库

[机器学习实践指南](https://zhuanlan.zhihu.com/p/29743418)

[CS231n课程笔记翻译：Python Numpy教程](https://zhuanlan.zhihu.com/p/20878530?refer=intelligentunit) 未读

基本原理搞懂之后，可以先找个实际问题+dataset 实操一下 [Kaggle入门，看这一篇就够了](https://zhuanlan.zhihu.com/p/25686876)Kaggle成立于2010年，是一个进行数据发掘和预测竞赛的在线平台。入门级的三个经典练习题目：

1. [逻辑回归应用之Kaggle泰坦尼克之灾](https://blog.csdn.net/han_xiaoyang/article/details/49797143)未做
2. [Kaggle竞赛 — 2017年房价预测](https://www.kaggle.com/neviadomski/how-to-get-to-top-25-with-simple-model-sklearn)未做
3. [大数据竞赛平台——Kaggle 入门](https://blog.csdn.net/u012162613/article/details/41929171)未做

[初学者如何从零开始征战Kaggle竞赛](https://mp.weixin.qq.com/s?__biz=MzI5ODQxMTk5MQ==&mid=2247489435&idx=1&sn=b9ebf571e145b25a33021f31a5f6270f&chksm=eca76523dbd0ec3543903843ee28af0f2dca11264b2b29d87b20c8575ea292d3ffb75f64d8a5&cur_album_id=1815350871241637891&scene=190#rd)
[从技术到科学，中国AI向何处去？](https://mp.weixin.qq.com/s/GV0UFUvDtIiBong0ZAe0LA)

《Approaching (Almost) Any Machine Learning Problem》未读

## 深度学习框架

《深入浅出Pytorch》一个深度学习框架首先需要支持的功能是张量的定义。张量的运算是深度学习的核心，比如，一个迷你批次大小的图片可以看成四维的张量，一个迷你批次的文本可以看成二维张量等。基本上所有深度学习模型的神经网络层（包括线性变换层和激活函数层等）都可以表示为张量的操作，梯度和反向传播算法也可以表示成张量和张量的运算。

有了张量的定义和运算还不够，我们需要让一个深度学习框架支持反向传播计算和梯度计算。为了能够计算权重梯度和数据梯度，一般来说，神经网络需要记录运算的过程，并构建出计算图。计算图的最后输出是一个损失函数的标量值，从标量值反推计算图权重张量的梯度（导数），这个过程被称为自动求导。动态计算图和静态计算图的求导方式不同。

综上，深度学习对于张量计算性能、算子灵活性、 自动微分能力、分布式训练、可视化和端侧部署都有很强的诉求。

学习路径
1. 知道原理
2. 可以进行矩阵推导
3. 对的上工程代码 tf/pytorch
4. 知道tf原理
5. 知道分布式架构，ps/allreduce
6. 知道推荐系统模型的特性，可以进行针对性的优化

## 工程/平台化

海量数据标注 + 大规模计算（训练和推理） + 工程化（python或c++）=AI系统

1. 数据侧：数据整理、数据标注系统
2. 训练侧：集群管理、分布式训练，更快、利用率高更高
2. 推理侧：对训练得到的模型剪枝、蒸馏、量化、压缩，更换算子等 [携程AI推理性能的自动化优化实践](https://mp.weixin.qq.com/s/jVnNMQNo_MsX3uSFRDmevA) [百度大模型与小模型联动及落地](https://mp.weixin.qq.com/s/iHO8zy6oKikgGHIamkam5w)

最终目标就是，缩短算法工程师训练一个模型的时间。

[淘系端智能技术体系概述](https://mp.weixin.qq.com/s/GhS88dAjlIJFmrfrRv7vzw)
