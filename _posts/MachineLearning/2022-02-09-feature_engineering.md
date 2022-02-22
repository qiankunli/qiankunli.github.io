---

layout: post
title: 特征工程
category: 架构
tags: MachineLearning
keywords: feature engineering

---

## 简介

* TOC
{:toc}

[机器学习之 特征工程](https://juejin.im/post/5b569edff265da0f7b2f6c65) 是一个系列

特征： 是指数据中抽取出来的对结果预测有用的信息，也就是数据的相关属性。

特征工程：使用专业背景知识和技巧处理数据，使得 特征能在机器学习算法上发挥更好的作用的过程。把原始数据转变为 模型的训练数据的过程。

![](/public/upload/machine/feature_enginering.png)

数据经过整理变成信息，信息能解决某个问题就是知识，知识通过反复实践形成才能，才能融会贯通就是智慧。 


[机器学习-特征工程.pptx](https://mp.weixin.qq.com/s/k9DCuocCL44Dzv5Tn9i7Hw)

[Scaling Distributed Machine Learning with the Parameter Server](https://web.eecs.umich.edu/~mosharaf/Readings/Parameter-Server.pdf)Machine learning systems are widely used in Web search,spam detection, recommendation systems, computational advertising, and document analysis. These systems automatically learn models from examples, termed training data, and typically consist of three components: feature extraction, the objective function, and learning.Feature extraction processes the raw training data, such as documents, images and user query logs, to obtain feature vectors, where each feature captures an attribute of the training data. Preprocessing can be executed efficiently by existing frameworks such as MapReduce.

## 特征构建

在原始数据集中的特征的形式不适合直接进行建模时，使用一个或多个原特征构造新的特征 可能会比直接使用原有特征更有效。

1. 数据规范化，使不同规格的数据转换到 同一规格
    1. 归一化
    2. Z-Score 标准化
2. 定量特征二值化，设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
3. 定性特征哑编码
4. 分箱，一般在建立分类模型时，需要对连续变量离散化
5. 聚合特征构造，对多个特征分组聚合
6. 转换特征构造，比如幂变换、log变换、绝对值等

## 特征提取
将原始数据转换为 一组具有明显 物理意义（比如几何特征、纹理特征）或统计意义的特征
1. 降维方面的 PCA、ICA、LDA 等
2. 图像方面的SIFT、Gabor、HOG 等
3. 文本方面的词袋模型、词嵌入模型等

## 特征选择

1. 过滤式 Filter
2. 包裹式 Wrapper
3. 嵌入式 embedding 

## embedding

### 基本概念

[从论文源码学习 之 embedding_lookup](https://mp.weixin.qq.com/s/FsqCNPtDPMdH0WGI0niELw) Embedding最重要的属性是：**越“相似”的实体，Embedding之间的距离越小。比如用one-hot编码来表示4个梁山好汉。
```
李逵   [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
刘唐   [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
武松   [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
鲁智深 [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] 
==>
        二  出  官   武
        货  家  阶   力
李逵    [1   0   0   0.5]
刘唐    [1   0   0   0.4]
武松    [0   1   0.5 0.8]
鲁智深  [0   1   0.75 0.8] 
```

Embedding层把我们的稀疏矩阵，通过一些线性变换（比如用全连接层进行转换，也称为查表操作），变成了一个密集矩阵，这个密集矩阵用了N（例子中N=4）个特征来表征所有的好汉。在这个密集矩阵中，表象上代表着密集矩阵跟单个好汉的一一对应关系，实际上还蕴含了大量的好汉与好汉之间的内在关系（如：我们得出的李逵跟刘唐的关系）。它们之间的关系，用嵌入层学习来的参数进行表征。这个从稀疏矩阵到密集矩阵的过程，叫做embedding，很多人也把它叫做查表，因为它们之间也是一个一一映射的关系。这种映射关系在反向传播的过程中一直在更新。因此能在多次epoch后，使得这个关系变成相对成熟，即：正确的表达整个语义以及各个语句之间的关系。这个成熟的关系，就是embedding层的所有权重参数。Embedding最大的劣势是无法解释每个维度的含义，这也是复杂机器学习模型的通病。

Embedding除了把独立向量联系起来之外，还有两个作用：降维，升维。
1. embedding层 降维的原理就是矩阵乘法。比如一个 1 x 4 的矩阵，乘以一个 4 x 3 的矩阵，得倒一个 1 x 3 的矩阵。4 x 3 的矩阵缩小了 1 / 4。假如我们有一个100W X 10W的矩阵，用它乘上一个10W X 20的矩阵，我们可以把它降到100W X 20，瞬间量级降了。
2. 升维可以理解为：前面有一幅图画，你离远了看不清楚，离近了看就可以看清楚细节。当对低维的数据进行升维时，可能把一些其他特征给放大了，或者把笼统的特征给分开了。同时这个embedding是一直在学习在优化的，就使得整个拉近拉远的过程慢慢形成一个良好的观察点。

如何生成?
1. 矩阵分解
2. 无监督建模
3. 有监督建模

### Embedding与深度学习推荐系统的结合

[NVIDIA HugeCTR，GPU版本参数服务器--- (5) 嵌入式hash表](https://mp.weixin.qq.com/s/W2bhGVdTB7z_TBec7KyYSA) 具有两个嵌入表和多个全连接层的神经网络

![](/public/upload/machine/rec_embedding.png)

Embedding 权重矩阵可以是一个 [item_size, embedding_size] 的稠密矩阵，item_size是需要embedding的物品个数，embedding_size是映射的向量长度，或者说矩阵的大小是：特征数量 * 嵌入维度。Embedding 权重矩阵的每一行对应输入的一个维度特征（one-hot之后的维度）。用户可以用一个index表示选择了哪个特征。

![](/public/upload/machine/onehot_embedding.png)

这样就把两个 1 x 9 的高维度，离散，稀疏向量，压缩到 两个 1 x 3 的低维稠密向量。这里把  One-Hot 向量中 “1”的位置叫做sparseID，就是一个编号。这个独热向量和嵌入表的矩阵乘法就等于利用sparseID进行的一次查表过程。

TensorFlow 的 embedding_lookup(params, ids) 函数的目的是按照ids从params这个矩阵中拿向量（行），所以ids就是这个矩阵索引（行号），需要int类型。即按照ids顺序返回params中的第ids行。比如说，ids=[1,3,2],就是返回params中第1,3,2行。返回结果为由params的1,3,2行组成的tensor。

embedding_lookup是一种特殊的全连接层的实现方法，其针对 输入是超高维 one hot向量的情况。
1. 神经网络处理不了onehot编码，Z = WX + b。由于X是One-Hot Encoding 的原因，WX 的矩阵乘法看起来就像是取了Weights矩阵中对应的一列，看起来就像是在查表，所以叫做 lookup。`embedding_lookup(W,X)`等于说进行了一次矩阵相乘运算，其实就是一次线性变换。
2. 假设embedding权重矩阵是一个`[vocab_size, embed_size]`的稠密矩阵W，vocab_size是需要embed的所有item的个数（比如：所有词的个数，所有商品的个数），embed_size是映射后的向量长度。所谓embedding_lookup(W, id1)，可以想像成一个只在id1位为1的[1, vocab_size]的one_hot向量，与[vocab_size, embed_size]的W矩阵相乘，结果是一个[1, embed_size]的向量，它就是id1对应的embedding向量，实际上就是W矩阵的第id1行。但是，以上过程只是前代，因为W一般是随机初始化的，是待优化的变量。因此，embedding_lookup除了要完成以上矩阵相乘的过程（实现成“抽取id对应的行”），还要完成自动求导，以实现对W的更新。PS: 所以embedding_lookup 是一个op


```
import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()
embedding = tf.Variable(np.identity(6, dtype=np.int32))    # 创建一个embedding词典
input_ids = tf.placeholder(dtype=tf.int32, shape=[None])
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)  # 把input_ids中给出的tensor表现成embedding中的形式

sess.run(tf.global_variables_initializer())
print("====== the embedding ====== ")
print(sess.run(embedding) )
print("====== the input_embedding ====== ")
print(sess.run(input_embedding, feed_dict={input_ids: [4, 0, 2]}))
====== the embedding ====== 
[[1 0 0 0 0 0]
 [0 1 0 0 0 0]
 [0 0 1 0 0 0]
 [0 0 0 1 0 0]
 [0 0 0 0 1 0]
 [0 0 0 0 0 1]]
====== the input_embedding ====== 
[[0 0 0 0 1 0]
 [1 0 0 0 0 0]
 [0 0 1 0 0 0]]
```

简单来说是通过输入的input_ids查询上部的字典得到embedding后的值。而字典是可以由用户随意创建的，例中给出的是一个one-hot字典，还可以自由创建其他字典，例如使用正态分布或均匀分布产生（0，1）的随机数创建任意维度的embedding字典。

[从论文源码学习 之 embedding层如何自动更新](https://mp.weixin.qq.com/s/v0K_9Y6aWAyHj7N1bIGvBw)`input_embedding = embedding * input_ids` 从效果上 可以把 input_ids 视为索引的作用，返回第4、0、2 行数据，**但 embedding_lookup 函数 也可以看做是一个 矩阵乘法**，也因此 embedding层可以通过 optimizer 进行更新。

DL 推荐模型的嵌入层是比较特殊的：它们为模型贡献了大量参数，但几乎不需要计算，而计算密集型denser layers的参数数量则要少得多。所以对于推荐系统，嵌入层的优化十分重要。

Embedding 优化
1. 把嵌入层分布在多个 GPU 和多个节点上
3. Embedding 层模型并行，dense 层数据并行。

## 训练框架实现

[推理性能提升一倍，TensorFlow Feature Column性能优化实践](https://mp.weixin.qq.com/s/2pV38VbvwCJkNA44HfcPuA) 未读
[TensorFlow 指标列，嵌入列](https://mp.weixin.qq.com/s/rR0wfJyWzX36tQ9tGSao6A)