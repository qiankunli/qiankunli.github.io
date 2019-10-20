---

layout: post
title: 机器学习用到的一些python库
category: 技术
tags: MachineLearning
keywords: 深度学习

---

## 前言（未完成）

[逻辑回归应用之Kaggle泰坦尼克之灾](https://blog.csdn.net/han_xiaoyang/article/details/49797143)

## numpy

![](/public/upload/machine/numpy.png)

## Matplotlib

[十分钟入门Matplotlib](https://codingpy.com/article/a-quick-intro-to-matplotlib/) Matplotlib 是 Python 的一个绘图库。它包含了大量的工具，你可以使用这些工具创建各种图形，包括简单的散点图，正弦曲线，甚至是三维图形。Python 科学计算社区经常使用它完成数据可视化的工作。

### 绘制数据集——包含xy数据

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.linspace(0, 2 * np.pi, 50)
    plt.plot(x, np.sin(x)) # 如果没有第一个参数 x，图形的 x 坐标默认为数组的索引
    plt.show() # 显示图形

上面的代码将画出一个简单的正弦曲线。`np.linspace(0, 2 * np.pi, 50)` 这段代码将会生成一个包含 50 个元素的数组，这 50 个元素均匀的分布在 `[0, 2pi]` 的区间上

![](/public/upload/machine/sin.png)

### 绘制直方图

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.random.randn(1000)
    plt.hist(x,50)
    plt.show()

## pandas

假设有一个学生表，想知道是女生多还是男生多，用sql 来表示就是`select sex,count(*) from student group by sex`

那么给定一个数据集/csv文件等，如何用python 做类似的分析呢？

[pandas与sql 对比,持续更新...](https://blog.csdn.net/weixin_39791387/article/details/81391621)

## sklearn（待理解）

所谓逻辑回归，正向传播、反向传播、梯度下降等 体现在python上，就是两行代码：

    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(xx)
    predictions = clf.predict(xx)