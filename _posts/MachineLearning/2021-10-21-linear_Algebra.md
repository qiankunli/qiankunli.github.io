---

layout: post
title: python与线性代数
category: 架构
tags: MachineLearning
keywords:  mpi

---

## 简介（未完成）

* TOC
{:toc}


## numpy

NumPy是Python中用于数据分析、机器学习、科学计算的重要软件包。它极大地简化了向量和矩阵的操作及处理。《用python实现深度学习框架》用python 和numpy 实现了一个深度学习框架MatrixSlow
1. 支持计算图的搭建、前向、反向传播
2. 支持多种不同类型的节点，包括矩阵乘法、加法、数乘、卷积、池化等，还有若干激活函数和损失函数。比如向量操作pytorch 跟numpy 函数命名都几乎一样
3. 提供了一些辅助类和工具函数，例如构造全连接层、卷积层和池化层的函数、各种优化器类、单机和分布式训练器类等，为搭建和训练模型提供便利


![](/public/upload/machine/numpy.png)

多维数组的属性
1. ndim, 数组维度（或轴）的个数
2. shape, 表示数组的维度或形状
3. size, 也就是数组元素的总数
4. type , 数组所属的数据类型
5. axis, 数组的轴, 即数组的维度，它是从 0 开始的。对于我们这个二维数组来说，有两个轴，分别是代表行的 0 轴与代表列的 1 轴。比如沿着 0轴求和 `np.sum(interest_score, axis=0)`


### 数组访问

```python
import numpy as np
b = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype=int)
c = b[0,1]  #1行 第二个单元元素
# 输出： 2
d = b[:,1]  #所有行 第二个单元元素
# 输出： [ 2  5  8 11]
e = b[1,:]  #2行 所有单元元素
# 输出： [4 5 6]
f = b[1,1:]  #2行 第2个单元开始以后所有元素
# 输出： [5 6]
g = b[1,:2]  #2行 第1个单元开始到索引为2以前的所有元素
```

## pytorch 张量/Tensor

在深度学习过程中最多使用`$5$`个维度的张量，张量的维度（dimension）通常叫作轴（axis）
1. 标量（0维张量）
2. 向量（1维度张量）
3. 矩阵（2维张量）
4. 3维张量，最常见的三维张量就是图片，例如`$[224, 224, 3]$`
4. 4维张量，4维张量最常见的例子就是批图像，加载一批 `$[64, 224, 224, 3]$` 的图片，其中 `$64$` 表示批尺寸，`$[224, 224, 3]$` 表示图片的尺寸。
5. 5维张量，使用5维度张量的例子是视频数据。视频数据可以划分为片段，一个片段又包含很多张图片。例如，`$[32, 30, 224, 224, 3]$` 表示有 `$32$` 个视频片段，每个视频片段包含 `$30$` 张图片，每张图片的尺寸为 `$[224, 224, 3]$`。

张量运算

1. 标量运算：（标量之间）加减乘除 长度 求导（导数是切线的斜率）
2. 向量运算：（标量向量之间，向量之间）加减乘除 长度 求导（也就是梯度，跟等高线正交）
3. 矩阵运算：（标量矩阵之间，向量矩阵之间，矩阵矩阵之间）加减乘 转置 求导

本质上来说，**PyTorch 是一个处理张量的库**。能计算梯度、指定设备等是 Tensor 相对numpy ndarray 特有的
1. Tensor的结构操作包括：创建张量，查看属性，修改形状，**指定设备**，数据转换， 索引切片，**广播机制（不同形状的张量相加）**，元素操作，归并操作；
2. Tensor的数学运算包括：标量运算，向量运算，矩阵操作，比较操作。

张量有很多属性，下面我们看看常用的属性有哪些？

1. tensor.shape，tensor.size(): 返回张量的形状；
2. tensor.ndim：查看张量的维度；
3. tensor.dtype，tensor.type()：查看张量的数据类型；
4. tensor.is_cuda：查看张量是否在GPU上；
5. tensor.grad：查看张量的梯度；
6. tensor.requires_grad：查看张量是否可微。


使用gpu
```python
cpu = torch.device("cpu")
gpu = torch.device("cuda:0")  # 使用第一个cpu
x = torch.rand(10)
x = x.to(gpu)
```

梯度
```python
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True) # 是否要求计算 y 相对 w 的梯度
b = torch.tensor(5., requires_grad=True) # 是否要求计算 y 相对 b 的梯度
y = w * x + b
print y
# tensor(17. grad_fn=<AddBackward0>)
y.backward()
print('dy/dx:', x.grad)
# dy/dx: Node
print('dy/dw:', w.grad)
# dy/dx: tensor(3.)
print('dy/db:', b.grad)
# dy/dx: tensor(1.)
```
我们已经创建了 3 个张量：x、w 和 b, y 是值为 3 * 4 + 5 = 17 的张量