---

layout: post
title: 神经网络模型优化
category: 技术
tags: MachineLearning
keywords: 深度学习

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## 前言（未完成）

## 梯度下降法的实现方式

1. batch gradient descent
2. 随机梯度下降

## bias 偏差 和 variance 方差

[一文读懂深度学习：从神经元到BERT](https://www.jiqizhixin.com/articles/2019-05-28-5) 形象的说，拟和就是把平面上一系列的点，用一条光滑的曲线连接起来。因为这条曲线有无数种可能，从而有各种拟和方法。拟和的曲线一般可以用函数表示，根据这个函数的不同有不同的拟和的名字：欠拟合（underfitting） 和过拟合（overfitting）

![](/public/upload/machine/bias_variance.jpg)

假设存在多个数据集\\(D\_1,D\_2,...\\)，

1. `f(x;D)`由训练集 D 学得的模型 f 对 x 的预测输出。
2. y 表示x 的真实值
3. 针对所有训练集，学习算法 f 对测试样本 x 的 期望预测 为:

	$$
	\overline{f}(x)=E_D[f(x;D)]
	$$

4. 偏差，偏差度量了学习算法的期望预测与真实结果的偏离程度，即 刻画了学习算法本身的拟合能力。

	$$
	偏差=(\overline{f}(x)-y)^2
	$$
5. 方差，方差度量了同样大小的训练集的变动所导致的学习性能的变化，即 刻画了数据扰动所造成的影响。

	$$
	方差=E_D[(f(x;D)-\overline{f}(x))^2]
	$$

![](/public/upload/machine/bias_variance_model_complexity.png)


我们希望偏差与方差越小越好，但一般来说偏差与方差是有冲突的, 称为偏差-方差窘境 (bias-variance dilemma).

1. 给定一个学习任务, 在训练初期, 由于训练不足, 学习器的拟合能力不够强, 偏差比较大, 也是由于拟合能力不强, 数据集的扰动也无法使学习器产生显著变化, 也就是欠拟合的情况;
2. 随着训练程度的加深, 学习器的拟合能力逐渐增强, 训练数据的扰动也能够渐渐被学习器学到;
3. 充分训练后, 学习器的拟合能力已非常强, 训练数据的轻微扰动都会导致学习器发生显著变化, 当训练数据自身的、非全局的特性被学习器学到了, 则将发生过拟合.

![](/public/upload/machine/bias_variance_optimization.jpg)


1. 初始训练模型完成后，我们首先要知道算法的偏差高不高
2. 如果偏差很高，甚至无法拟合训练集，可以尝试

    1. **更大的网络** 比如更多的隐层或隐藏单元
    2. 花费更多的时间训练算法
    3. 更先进的优化算法
    4. 新的神经网络结构
    5. 准备更多的训练数据对高偏差没啥大用
3. 反复尝试，直到可以拟合训练集，使得偏差降低到可以接受的值
4. 如果方差比较高

    1. **更多的训练数据**
    2. regularization/正则化
    3. 新的神经网络结构
5. 不断尝试，直到找到一个低偏差、低方差的框架

## 防止过拟合

数学原理上理解起来还比较困难

### 正则化/规则化

### dropout 正则化