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

数据经过整理变成信息，信息能解决某个问题就是知识，知识通过反复实践形成才能，才能融会贯通就是智慧。 

[机器学习-特征工程.pptx](https://mp.weixin.qq.com/s/k9DCuocCL44Dzv5Tn9i7Hw)

[Scaling Distributed Machine Learning with the Parameter Server](https://web.eecs.umich.edu/~mosharaf/Readings/Parameter-Server.pdf)Machine learning systems are widely used in Web search,spam detection, recommendation systems, computational advertising, and document analysis. These systems automatically learn models from examples, termed training data, and typically consist of three components: feature extraction, the objective function, and learning.Feature extraction processes the raw training data, such as documents, images and user query logs, to obtain feature vectors, where each feature captures an attribute of the training data. Preprocessing can be executed efficiently by existing frameworks such as MapReduce.

## 特征构建

在原始数据集中的特征的形式不适合直接进行建模时，使用一个或多个原特征构造新的特征 可能会比直接使用原有特征更有效。

1. 数据规范化，使不同规格的数据转换到 同一规格。否则，大数值特征会主宰模型训练，这会导致更有意义的小数值特征被忽略
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

## tf 特征处理 feature column

[TensorFlow Feature Column性能可以这样玩](https://mp.weixin.qq.com/s/9lu0WQZHLC0XvyfSjaEK4w)Feature Column是TensorFlow提供的用于处理结构化数据的工具，是将**样本特征**映射到用于**训练模型特征**的桥梁。[看Google如何实现Wide & Deep模型（2.1）](https://zhuanlan.zhihu.com/p/47965313)**Feature Column本身并不存储数据，而只是封装了一些预处理的逻辑**。

![](/public/upload/machine/tf_feature_column.png)

其中只有一个numeric_column是纯粹处理数值特征的，其余的都与处理categorical特征有关，从中可以印证：categorical特征才是推荐、搜索领域的一等公民。

### 心脏病数据集

[对结构化数据进行分类](https://www.tensorflow.org/tutorials/structured_data/feature_columns)以心脏病数据集为例

![](/public/upload/machine/heart_disease_dataset.png)
![](/public/upload/machine/heart_disease_data.png)

```python
feature_columns = []

# 数值列
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))
# 分桶列，如果不希望将数字直接输入模型，而是根据数值范围将其值分成不同的类别。考虑代表一个人年龄的原始数据。我们可以用 分桶列（bucketized column）将年龄分成几个分桶（buckets），而不是将年龄表示成数值列。
age = feature_column.numeric_column("age")
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# 分类列，thal 用字符串表示（如 ‘fixed’，‘normal’，或 ‘reversible’）。我们无法直接将字符串提供给模型。相反，我们必须首先将它们映射到数值。分类词汇列（categorical vocabulary columns）提供了一种用 one-hot 向量表示字符串的方法
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# 嵌入列，假设我们不是只有几个可能的字符串，而是每个类别有数千（或更多）值。 由于多种原因，随着类别数量的增加，使用 one-hot 编码训练神经网络变得不可行。我们可以使用嵌入列来克服此限制。嵌入列（embedding column）将数据表示为一个低维度密集向量，而非多维的 one-hot 向量，该低维度密集向量可以包含任何数，而不仅仅是 0 或 1。
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# 组合列，将多种特征组合到一个特征中，称为特征组合（feature crosses），它让模型能够为每种特征组合学习单独的权重。此处，我们将创建一个 age 和 thal 组合的新特征。
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)
```

现在我们已经定义了我们的特征列，我们将使用密集特征（DenseFeatures）层将特征列输入到我们的 Keras 模型中。

```
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
```

```python
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'],run_eagerly=True)
model.fit(train_ds,validation_data=val_ds,epochs=5)
```

### 实现

```python
class DenseColumn(FeatureColumn):
    def get_dense_tensor(self, transformation_cache, state_manager):
class NumericColumn(DenseColumn,...):
    def _transform_feature(self, inputs):
        input_tensor = inputs.get(self.key)
        return self._transform_input_tensor(input_tensor)
    def get_dense_tensor(self, transformation_cache, state_manager):
        return transformation_cache.get(self, state_manager)
```

inputs dataset可以看成{feature_name: feature tensor}的dict，而每个feature column定义时需要指定一个名字，feature column与input就是通过这个名字联系在一起。

**基于 feature_columns 构造 DenseFeatures Layer，反过来说，DenseFeatures Layer 在前向传播时被调用，Layer._call/DenseFeatures.call ==> feature_column.get_dense_tensor ==> feature_column._transform_feature(inputs)**。

```python
class DenseFeatures(kfc._BaseFeaturesLayer): 
    def call(self, features, cols_to_output_tensors=None, training=None):
        transformation_cache = fc.FeatureTransformationCache(features)
        output_tensors = []
        for column in self._feature_columns:
            tensor = column.get_dense_tensor(transformation_cache, self._state_manager, training=training)
            processed_tensors = self._process_dense_tensor(column, tensor)
            output_tensors.append(processed_tensors)
    return self._verify_and_concat_tensors(output_tensors)
```

1. 继承关系，DenseFeatures ==> _BaseFeaturesLayer ==> Layer， DenseFeatures 是一个layer，描述了对input dataset 的处理，位于模型的第一层（也叫特征层）
2. Generally a single example in training data is described with FeatureColumns. At the first layer of the model, this column-oriented data should be converted to a single `Tensor`.

## 其它

![](/public/upload/machine/feature_enginering.png)


[推理性能提升一倍，TensorFlow Feature Column性能优化实践](https://mp.weixin.qq.com/s/2pV38VbvwCJkNA44HfcPuA) 未读
[TensorFlow 指标列，嵌入列](https://mp.weixin.qq.com/s/rR0wfJyWzX36tQ9tGSao6A)
[看Google如何实现Wide & Deep模型（2.1）](https://zhuanlan.zhihu.com/p/47965313)