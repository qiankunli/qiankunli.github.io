---

layout: post
title: X的生成——特征工程
category: 架构
tags: MachineLearning
keywords: feature engineering

---

## 简介

* TOC
{:toc}

[机器学习之 特征工程](https://juejin.im/post/5b569edff265da0f7b2f6c65) 是一个系列

特征： 是指数据中抽取出来的对结果预测有用的信息，也就是数据的相关属性。

特征工程：使用专业背景知识和技巧处理数据，使得 特征能在机器学习算法上发挥更好的作用的过程。从原始数据生成可用于模型训练的训练样本（原始特征 ==> 模型特征）。这个领域有专门的书《Feature Engineering for Machine Learning》。

![](/public/upload/machine/feature_engineer_process.png)

数据经过整理变成信息，信息能解决某个问题就是知识，知识通过反复实践形成才能，才能融会贯通就是智慧。 

[机器学习-特征工程.pptx](https://mp.weixin.qq.com/s/k9DCuocCL44Dzv5Tn9i7Hw)

[Scaling Distributed Machine Learning with the Parameter Server](https://web.eecs.umich.edu/~mosharaf/Readings/Parameter-Server.pdf)Machine learning systems are widely used in Web search,spam detection, recommendation systems, computational advertising, and document analysis. These systems automatically learn models from examples, termed training data, and typically consist of three components: feature extraction, the objective function, and learning.Feature extraction processes the raw training data, such as documents, images and user query logs, to obtain feature vectors, where each feature captures an attribute of the training data. Preprocessing can be executed efficiently by existing frameworks such as MapReduce.


[机器学习训练中常见的问题和挑战](https://mp.weixin.qq.com/s/Ez_ZpfCfXjeN9a4e72kIAA)
1. 在2001年发表的一篇著名论文中，微软研究员Michele Banko和Eric Brill表明，给定足够的数据，截然不同的机器学习算法（包括相当简单的算法）在自然语言歧义消除这个复杂问题上，表现几乎完全一致。对复杂问题而言，数据比算法更重要，这一想法被Peter Norvig等人进一步推广，于2009年发表论文“The Unreasonable Effectiveness of Data”。不过需要指出的是，中小型数据集依然非常普遍，获得额外的训练数据并不总是一件轻而易举或物美价廉的事情，所以暂时先不要抛弃算法。
2. 如果训练集满是错误、异常值和噪声（例如，低质量的测量产生的数据），系统将更难检测到底层模式，更不太可能表现良好。如果某些实例明显是异常情况，那么直接将其丢弃，或者尝试手动修复错误，都会大有帮助。如果某些实例缺少部分特征（例如，5%的顾客没有指定年龄），你必须决定是整体忽略这些特征、忽略这部分有缺失的实例、将缺失的值补充完整（例如，填写年龄值的中位数），还是训练一个带这个特征的模型，再训练一个不带这个特征的模型。[机器学习中数据缺失值这样处理才香！](https://mp.weixin.qq.com/s/aUmu_qU5G2ifUR4wX9l75Q)
3. 只有训练数据里包含足够多的相关特征以及较少的无关特征，系统才能够完成学习。一个成功的机器学习项目，其关键部分是提取出一组好的用来训练的特征集。

[特征工程如何找有效的特征？ - 杨旭东的回答 - 知乎](https://www.zhihu.com/question/349860940/answer/2499614543)深度学习时期，与CV、语音、NLP领域不同（深度学习技术在计算机视觉、语音、NLP领域的成功，使得在这些领域手工做特征工程的重要性大大降低），搜推广场景下特征工程仍然对业务效果具有很大的影响，并且占据了算法工程师的很多精力。**数据决定了效果的上限，算法只能决定逼近上限的程度**，而特征工程则是数据与算法之间的桥梁。关于特征工程的三个误区：

1. 在搜索、推荐、广告等领域，特征数据主要以关系型结构组织和存储，在关系型数据上的特征生成和变换操作主要有两大类型
    1. 一种是基于行（row-based）的特征变换，也就是同一个样本的不同特征之间的变换操作，比如特征组合；
    2. 另一种是基于列（column-based）的特征变换，比如类别型特征的分组统计值，如最大值、最小值、平均值、中位数等。
    模型可以一定程度上学习到row-based的特征变换，比如PNN、DCN、DeepFM、xDeepFM、AutoInt等模型都可以建模特征的交叉组合操作。尽管如此，模型却很难学习到基于列的特征变换，这是因为深度模型一次只能接受一个小批次的样本，无法建模到全局的统计聚合信息，而这些信息往往是很重要的。综上，即使是深度学习模型也是需要精准特征工程的。
2. 有了AutoFE工具就不再需要手工做特征工程。AutoFE 还处于初级阶段，特征工程非常依赖数据科学家的业务知识、直觉和经验。
3. 特征工程是没有技术含量的脏活累活。很多学生和刚参加工作不久的同事会有一种偏见，那就是算法模型才是高大上的技术，特征工程是脏活累活，没有技术含量。算法模型的更新迭代速度太快了，总会有效率更高、效果更好的模型被提出，从而让之前的积累变得无用。另一方面，特征工程的经验沉淀就好比是一个滚雪球的过程，雪球会越滚越大，最终我们会成为一个业务的领域专家，对业务贡献无可代替的价值。

[什么是好的特征工程](https://yangxudong.github.io/good-feature/)，高质量特征需要满足以下标准：
1. 有区分性（Informative）
2. 特征之间相互独立（Independent）
3. 简单易于理解（Simple）
4. 伸缩性（ Scalable ）：支持大数据量、高基数特征
5. 高效率（ Efficient ）：支持高并发预测
6. 灵活性（ Flexible ）：对下游任务有一定的普适性
7. 自适应（ Adaptive ）：对数据分布的变化有一定的鲁棒性

## 特征构建

在原始数据集中的特征的形式不适合直接进行建模时，使用一个或多个原特征构造新的特征 可能会比直接使用原有特征更有效。

1. 数据规范化，使不同规格的数据转换到 同一规格（特征缩放/normalize）。否则，大数值特征会主宰模型训练，这会导致更有意义的小数值特征被忽略，导致梯度更新在误差超平面上不断震荡，模型的学习效率较低。
    1. 归一化
    2. Z-Score 标准化
2. 定量特征二值化，设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
3. 定性特征哑编码
4. 特征分箱/binning，即特征离散化/Bucketizer，按照某种方法把数值型特征值映射到有限的几个“桶（bin）”内。比如，可以把1天24个小时按照如下规则划分为5个桶，使得每个桶内的不同时间都有类似的目标预测能力，比如有类似的购买概率。高基数（high-cardinality）类别型特征也有必要做特征分箱。
5. 聚合特征构造，单特征区分性不强时，可尝试组合不同特征。
6. 转换特征构造，比如幂变换、log变换、绝对值等

常用的特征变换操作
1. 数值型特征的常用变换：特征缩放；特征分箱。
2. 类别型特征的常用变换：交叉组合；特征分箱（binning）；统计编码
3. 时序特征：历史事件分时段统计；环比、同比

在spark 和tf 中都有对应的处理函数。

## 特征提取

将原始数据转换为 一组具有明显 物理意义（比如几何特征、纹理特征）或统计意义的特征
1. 降维方面的 PCA、ICA、LDA 等
2. 图像方面的SIFT、Gabor、HOG 等
3. 文本方面的词袋模型、词嵌入模型等

## 特征选择

从现有特征中选择最有用的特征进行训练

1. 过滤式 Filter
2. 包裹式 Wrapper
3. 嵌入式 embedding 

## 特征准入和淘汰

CTR任务中，user_id和item_id的数量那么大，也常规embedding吗？如果是大厂的话，就是直接加。大厂所使用的parameter server一般都有特征的准入与逐出机制。尽管模型代码上都一视同仁地将所有user_id, item_id加入模型，但是只有那些出现超过一定次数的id，才最终被parameter server分配空间。而且对于非活跃用户、过时下架的item，parameter server也会逐出，释放空间。

商业场景中，时时刻刻都会有新的样本产生，新的样本带来新的特征。有一些特征出现频次较低，如果全部加入到模型中，一方面对内存来说是个挑战，另外一方面，低频特征会带来过拟合。因此XDL针对这样的数据特点，提供了一些特征准入机制，包括基于概率进行过滤，布隆过滤器等。比如DeepRec支持了两种特征准入的方式：基于Counter的特征准入和基于Bloom Filter的特征准入：

1. 基于Counter的特征准入：基于Counter的准入会记录每个特征在前向中被访问的次数，只有当统计的次出超过准入值的特征才会给分配embedding vector并且在后向中被更新。这种方法的好处子在于会精确的统计每个特征的次数，同时获取频次的查询可以跟查询embedding vector同时完成，因此相比不使用特征准入的时候几乎不会带来额外的时间开销。缺点则是为了减少查询的次数，即使对于不准入的特征，也需要记录对应特征所有的metadata，在准入比例较低的时候相比使用Bloom Filter的方法会有较多额外内存开销。
2. 基于Bloom Filter的准入：基于Bloom Filter的准入是基于Counter Bloom Filter实现的，这种方法的优点是在准入比例较低的情况下，可以比较大地减少内存的使用量。缺点是由于需要多次hash与查询，会带来比较明显的时间开销，同时在准入比例较高的情况下，Blomm filter数据结构带来的内存开销也比较大。

有一些特征长时间不更新会失效。为了缓解内存压力，提高模型的时效性，需要淘汰过时的特征，XDL支持算法开发者，通过写用户自定义函数(UDF)的方式，制定淘汰规则。[EmbeddingVariable进阶功能：特征淘汰](https://deeprec.readthedocs.io/zh/latest/Feature-Eviction.html)在DeepRec中我们支持了特征淘汰功能，每次存ckpt的时候会触发特征淘汰，目前我们提供了两种特征淘汰的策略：
1. 基于global step的特征淘汰功能：第一种方式是根据global step来判断一个特征是否要被淘汰。我们会给每一个特征分配一个时间戳，每次前向该特征被访问时就会用当前的global step更新其时间戳。在保存ckpt的时候判断当前的global step和时间戳之间的差距是否超过一个阈值，如果超过了则将这个特征淘汰（即删除）。这种方法的好处在于查询和更新的开销是比较小的，缺点是需要一个int64的数据来记录metadata，有额外的内存开销。 用户通过配置steps_to_live参数来配置淘汰的阈值大小。
2. 基于l2 weight的特征淘汰： 在训练中如果一个特征的embedding值的L2范数越小，则代表这个特征在模型中的贡献越小，因此在存ckpt的时候淘汰淘汰L2范数小于某一阈值的特征。这种方法的好处在于不需要额外的metadata，缺点则是引入了额外的计算开销。用户通过配置l2_weight_threshold来配置淘汰的阈值大小。

```python
#使用global step特征淘汰
evict_opt = tf.GlobalStepEvict(steps_to_live=4000)
#通过get_embedding_variable接口使用
emb_var = tf.get_embedding_variable("var", embedding_dim = 16, ev_option=ev_opt)
```

## 特征交叉/Feature crosses

Combining features, better known as feature crosses, enables the model to learn separate weights specifically for whatever that feature combination means.

[Introducing TensorFlow Feature Columns——Feature crosses](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html)

## tf 特征处理 feature column

[Introducing TensorFlow Feature Columns](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html) 必读文章

### 起源
    
[TensorFlow Feature Column性能可以这样玩](https://mp.weixin.qq.com/s/9lu0WQZHLC0XvyfSjaEK4w)Feature Column是TensorFlow提供的用于处理结构化数据的工具，是将**样本特征**映射到用于**训练模型特征**的桥梁。原始的输入数据中既有连续特征也有离散特征，这就需要我给每个特征的定义处理逻辑，来完成原始数据向模型真正用于计算的输入数据的**数值化转换**。[看Google如何实现Wide & Deep模型（2.1）](https://zhuanlan.zhihu.com/p/47965313)**Feature Column本身并不存储数据，而只是封装了一些预处理的逻辑**。

![](/public/upload/machine/tf_feature_column.png)

其中只有一个numeric_column是纯粹处理数值特征的，其余的都与处理categorical特征有关，从中可以印证：categorical特征才是推荐、搜索领域的一等公民。

**what kind of data can we actually feed into a deep neural network? The answer is, of course, numbers (for example, tf.float32)**. After all, every neuron in a neural network performs multiplication and addition operations on weights and input data. Real-life input data, however, often contains non-numerical (categorical) data. PS: feature_column 最开始是与Estimator 配合使用的，**任何input 都要转换为feature_column传给Estimator**， feature columns——a data structure describing the features that an Estimator requires for training and inference.
1. Numeric Column
2. Bucketized Column, Often, you don't want to feed a number directly into the model, but instead split its value into different categories based on numerical ranges. splits a single input number into a four-element vector. ==> 模型变大了，让model 能够学习不同年代的权重（相对只输入一个input year 来说）
    
    |Date Range|	 Represented as...|
    |---|---|
    |< 1960|	 [1, 0, 0, 0]|
    |>= 1960 but < 1980| 	 [0, 1, 0, 0]|
    |>= 1980 but < 2000| 	 [0, 0, 1, 0]|
    |> 2000|	 [0, 0, 0, 1]|
3. Categorical identity column, 用一个向量表示一个数字，意图与Bucketized Column 是在一致的，让模型可以学习每一个类别的权重

    |类别|数字|	 Represented as...|
    |---|---|---|
    |kitchenware|0|	 [1, 0, 0, 0]|
    |electronics|1| 	 [0, 1, 0, 0]|
    |sport|2| 	 [0, 0, 1, 0]|
    |history|3|	 [0, 0, 0, 1]|
    
4. Categorical vocabulary column, We cannot input strings directly to a model. Instead, **we must first map strings to numeric or categorical value**s. Categorical vocabulary columns provide a good way to represent strings as a one-hot vector.
5. indicator column, treats each category as an element in a one-hot vector, where the matching category has value 1 and the rest have 0
5. embedding column, Instead of representing the data as a one-hot vector of many dimensions, an embedding column represents that data as a lower-dimensional, ordinary vector in which each cell can contain any number, not just 0 or 1. By permitting a richer palette of numbers for every cell

一个脉络就是：除了数值数据，对于分类、 字符串数据，我们不是简单的将其转换为数值型，而是将其转换为了一个向量，目的是尽量学习每一个分类的weight，但是如果某个分类太多，用one-hot 表示太占内存，就需要考虑embedding

![](/public/upload/machine/embedding_column.png)

以上图为例,one of the categorical_column_with... functions maps the example string to a numerical categorical value. 
1. As an indicator column. A function converts each numeric categorical value into an **81-element vector** (because our palette consists of 81 words), placing a 1 in the index of the categorical value (0, 32, 79, 80) and a 0 in all the other positions.
2. As an embedding column. A function uses the numerical categorical values (0, 32, 79, 80) as indices to a lookup table. Each slot in that lookup table contains a **3-element vector**. **How do the values in the embeddings vectors magically get assigned?** Actually, the assignments happen during training. That is, the model learns the best way to map your input numeric categorical values to the embeddings vector value in order to solve your problem. Embedding columns increase your model's capabilities, since an embeddings vector learns new relationships between categories from the training data. Why is the embedding vector size 3 in our example? Well, the following "formula" provides a general rule of thumb about the number of embedding dimensions:`embedding_dimensions =  number_of_categories**0.25`

### Estimator 方式

花的识别，示例代码
```python
feature_names = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
def my_input_fn(...):
    ...<code>...
    return ({ 'SepalLength':[values], ..<etc>.., 'PetalWidth':[values] },
            [IrisFlowerType])
# Create the feature_columns, which specifies the input to our model.All our input features are numeric, so use numeric_column for each one.
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]
# Create a deep neural network regression classifier.Use the DNNClassifier pre-made estimator
classifier = tf.estimator.DNNClassifier(
   feature_columns=feature_columns, # The input features to our model
   hidden_units=[10, 10], # Two layers, each with 10 neurons
   n_classes=3,
   model_dir=PATH) # Path to where checkpoints etc are stored
# Train our model, use the previously function my_input_fn Input to training is a file with training example Stop training after 8 iterations of train data (epochs)
classifier.train(input_fn=lambda: my_input_fn(FILE_TRAIN, True, 8))
# Evaluate our model using the examples contained in FILE_TEST 
# Return value will contain evaluation_metrics such as: loss & average_loss
evaluate_result = estimator.evaluate(input_fn=lambda: my_input_fn(FILE_TEST, False, 4)
```

### 心脏病数据集/非 Estimator 方式

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

也可以 `tensor = tf.feature_column.input_layer(features, feature_columns)` 将输入 的features 数据转换为 (input)tensor。

### 非 Estimator 方式 实现

```python
class _FeatureColumn(object):
    # 将inputs 转为 tensor
    @abc.abstractmethod
    def _transform_feature(self, inputs):
class DenseColumn(FeatureColumn):
    def get_dense_tensor(self, transformation_cache, state_manager):
class NumericColumn(DenseColumn,...):
    def _transform_feature(self, inputs):
        # input_tensor ==> output_tensor
        input_tensor = inputs.get(self.key)
        return self._transform_input_tensor(input_tensor)
    def get_dense_tensor(self, transformation_cache, state_manager):
        return transformation_cache.get(self, state_manager)
```

inputs dataset 带有schema，而每个feature column定义时需要指定一个名字，feature column与input就是通过这个名字联系在一起。

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

## spark 特征处理

以kaggle 房价预测项目为例，将数据集读取为 spark DataFrame

```scala
import org.apache.spark.sql.DataFrame
 
// 这里的下划线"_"是占位符，代表数据文件的根目录
val rootPath: String = _
val filePath: String = s"${rootPath}/train.csv"
val sourceDataDF: DataFrame = spark.read.format("csv").option("header", true).load(filePath)

// 将engineeringDF定义为var变量，后续所有的特征工程都作用在这个DataFrame之上
var engineeringDF: DataFrame = sourceDataDF
```
“BedroomAbvGr”字段的含义是居室数量，在 train.csv 这份数据样本中，“BedroomAbvGr”包含从 1 到 8 的连续整数。现在，我们根据居室数量，把房屋粗略地划分为小户型、中户型和大户型。
```scala
// 原始字段
val fieldBedroom: String = "BedroomAbvGrInt"
// 包含离散化数据的目标字段
val fieldBedroomDiscrete: String = "BedroomDiscrete"
// 指定离散区间，分别是[负无穷, 2]、[3, 4]和[5, 正无穷]
val splits: Array[Double] = Array(Double.NegativeInfinity, 3, 5, Double.PositiveInfinity)
// 定义并初始化Bucketizer
val bucketizer = new Bucketizer()
    // 指定原始列
    .setInputCol(fieldBedroom)
    // 指定目标列
    .setOutputCol(fieldBedroomDiscrete)
    // 指定离散区间
    .setSplits(splits)
// 调用transform完成离散化转换
engineeringData = bucketizer.transform(engineeringData)
```

计算完之后，engineeringData DataFrame 多了一列 BedroomDiscrete，用012 来代表小中大户型。

作为特征工程的最后一个环节，向量计算主要用于构建训练样本中的特征向量（Feature Vectors）。在 Spark MLlib 框架下，训练样本由两部分构成，第一部分是预测标的（Label），在“房价预测”的项目中，Label 是房价。而第二部分，就是特征向量，在形式上，特征向量可以看作是元素类型为 Double 的数组。

```scala
import org.apache.spark.ml.feature.VectorAssembler
/**
入选的数值特征：selectedFeatures
归一化的数值特征：scaledFields
离散化的数值特征：fieldBedroomDiscrete
热独编码的非数值特征：oheFields
*/
val assembler = new VectorAssembler()
    .setInputCols(selectedFeatures ++ scaledFields ++ fieldBedroomDiscrete ++ oheFields)
    .setOutputCol("features")
engineeringData = assembler.transform(engineeringData)
```

转换完成之后，engineeringData 这个 DataFrame 就包含了一列名为“features”的新字段，这个字段的内容，就是每条训练样本的特征向量。接下来，通过 setFeaturesCol 和 setLabelCol 来指定特征向量与预测标的，定义出线性回归模型。

```scala
// 定义线性回归模型
val lr = new LinearRegression()
    .setFeaturesCol("features")
    .setLabelCol("SalePriceInt")
    .setMaxIter(100)
// 训练模型
val lrModel = lr.fit(engineeringData)
// 获取训练状态
val trainingSummary = lrModel.summary
// 获取训练集之上的预测误差
println(s"Root Mean Squared Error (RMSE) on train data: ${trainingSummary.rootMeanSquaredError}")
```

如果算法人员懂spark，可以把数据读取为dataframe，然后一路转换dataframe，最终转成训练集。但实际上机器学习平台为了支持特征处理 + 训练的可编排，一般会为一次数据转换启动一个spark任务，任务之间通过hdfs流转数据，hdfs ==> dataframe ==> hdfs落盘 ==> dataframe ==> hdfs落盘 ==> tfdataset ==> tf训练任务。

## 其它

[推理性能提升一倍，TensorFlow Feature Column性能优化实践](https://mp.weixin.qq.com/s/2pV38VbvwCJkNA44HfcPuA) 未读
[TensorFlow 指标列，嵌入列](https://mp.weixin.qq.com/s/rR0wfJyWzX36tQ9tGSao6A)
[看Google如何实现Wide & Deep模型（2.1）](https://zhuanlan.zhihu.com/p/47965313)