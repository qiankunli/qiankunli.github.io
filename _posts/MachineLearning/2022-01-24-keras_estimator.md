---

layout: post
title: keras 和 Estimator
category: 架构
tags: MachineLearning
keywords:  tensorflow keras Estimator

---

## 简介

* TOC
{:toc}


![](/public/upload/machine/tensorflow_overview.png)

1. 高层API (High level): 包括Estimators、Keras以及预构建好的Premade estimator(如线性回归、逻辑回归这些、推荐排序模型wide&deep)；
2. 中层API (Mid level): 包括layers, datasets, loss和metrics等具有功能性的函数，例如网络层的定义，Loss Function，对结果的测量函数等；
3. 底层API (Low level): 包括具体的加减乘除、具有解析式的数学函数、卷积、对Tensor属性的测量等。

## keras

Keras 是开源 Python 包，由于 Keras 的独立性，Keras 具有自己的图形数据结构，用于定义计算图形：它不依靠底层后端框架的图形数据结构。PS： 所有有一个model.compile 动作？
[聊聊Keras的特点及其与其他框架的关系](https://mp.weixin.qq.com/s/fgG95qqbrV07EgAqLXuFAg)

不足
1. keras 为了解耦后端引擎， 自己定义了一套优化器，如SGD/RMSprop 等，因为它没有继承和封装 Tensorflow 的优化器，所以无法使用Tensorflow 为分布式模型设计的同步优化器，这导致短期内无法使用keras 编写分布式模型。PS：分布式用h
2. 只能设计神经网络模型， 无法编写传统机器学习模型。

keras 提供两种模型范式
1. 顺序模型，keras.models.Sequential类，基于函数式模型实现，适用于单输出的线性神经网络模型
2. 函数式模型，keras.models.Model类，灵活性更好，适用于多输入多输出神经网络模型。

### 顺序模型

```python
# 声明个模型
model = models.Sequential()
# 把部件像乐高那样拼起来
model.add(keras.layers.Dense(512, activation=keras.activations.relu, input_dim=28*28))
model.add(keras.layers.Dense(10, activation=keras.activations.softmax))
# 配置学习过程
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
# 在训练数据上进行迭代，填充的数据和标签可以为 numpy 数组类型
model.fit(X_train,Y_train,nb_epoch=5,batch_size=32)
# 评估模型性能
model.evaluate(X_test,Y_test,batch_size=32)
# 对新的数据生成预测
classes = model.predict_classes(X_test,batch_size=32)
```

### 函数式模型

[Keras 高级用法：函数式 API 7.1（一）](https://mp.weixin.qq.com/s/XBkU_QnQ5OZzRLpz5yywmg)有些神经网络模型有多个独立的输入，而另一些又要求有多个输出。一些神经网络模型有 layer 之间的交叉分支，模型结构看起来像图（ graph ），而不是线性堆叠的 layer 。它们不能用Sequential 模型类实现，而要使用Keras更灵活的函数式 API ( functional API)。

函数式 API 可以直接操作张量，你可以将layers 模块**作为函数**使用，传入张量返回张量。下面来一个简单的例子，让你清晰的区别 Sequential 模型和其等效的函数式 API。

```python
from keras.models import Sequential, Model
from keras import layers
from keras import Input
# Sequential model, which you already know about
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,))) 
seq_model.add(layers.Dense(32, activation='relu')) seq_model.add(layers.Dense(10, activation='softmax'))
```
对等的函数式模型代码。PS： 使用起来有点pytorch 的意思了。
```python
# Its functional equivalent
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
## The Model class turns an input tensor and output tensor into a model.
model = Model(input_tensor, output_tensor) 
```
用一个输入张量和输出张量实例化了一个 Model 对象，在幕后，Keras从 input_tensor 到 output_tensor 检索每个 layer，将它们合并成一个有向无环图数据结构：一个 Model 对象。当然，这些可以运行的原因是输出张量output_tensor是从输入张量input_tensor不断转化得到的。

创建Model 对象后，可以 model.complile + model.fit 进行模型训练，也可以与tensorflow 原有api 结合，将Model 作为函数使用。

```python
y_hat = model(input)
loss = loss(y_hat,y)
train_op = optrimizer(loss)
train_op.minimize()
```

### 实现

[Keras源码分析之基础框架](https://mp.weixin.qq.com/s/SA6VEyllWF645u1g42ATHg)

```
|-- docs                      #说明文档
|-- examples                  #应用示例
|-- test                      #测试文件
|-- keras                     #核心源码
      |-- application         #应用实例，如VGG16,RESNET50
      |-- backend             #底层接口，如:tensorflow_backend,theano_backend
      |-- datasets            #数据获取，如boston_housing,mnist
      |-- engine              #网络架构
            |-- base_layer    #网络架构层的抽象类
            |-- network       #Network类所在，构建layer拓扑DAG
            |-- input_layer   #输入层
            |-- training      #Model类所在
            |-- training_generator  #训练函数
      |-- layers              #各类层相关实现
      |-- legacy              #遗留源码
      |-- preprocessing       #预处理函数
```

Layer是计算层的抽象，其主要的处理逻辑是，给定输入，产生输出。因为各种layer作用的不同，所以在些基类中无法做具体实现，它们的具体功能留待各种子类去实现。
```python
class Layer(module.Module, version_utils.LayerVersionSelector):
    def __init__(self,trainable=True,name=None,dtype=None,dynamic=False,**kwargs):
        self._inbound_nodes_value = []
        self._outbound_nodes_value = []
    # build方法一般定义Layer需要被训练的参数
    def build(self, input_shape):
    # call方法一般定义正向传播运算逻辑，__call__方法调用了它
    def call(self, inputs, *args, **kwargs): 
```
拓扑网络Network类
```python
class Model(base_layer.Layer, version_utils.ModelVersionSelector):
    def compile(...)    # 配置学习过程
    def fit(...)        # 在训练数据上进行迭代
    def evaluate(...)   # 评估模型性能
    def predict(...)    # 对新的数据生成预测
```

model.fit
```python
callbacks.set_model(callback_model)
callbacks.on_train_begin() 
for epoch in range(initial_epoch, nb_epoch):
    callbacks.on_epoch_begin(epoch)
    训练
    callbacks.on_epoch_end(epoch, epoch_logs)
    if callback_model.stop_training:
        break
callbacks.on_train_end()
``` 
训练代码
```python
batches = make_batches(nb_train_sample, batch_size)
for batch_index, (batch_start, batch_end) in enumerate(batches):
    callbacks.on_batch_begin(batch_index, batch_logs)
    for i in indices_for_conversion_to_dense:
        ins_batch[i] = ins_batch[i].toarray()
    outs = f(ins_batch) # 对于较新的版本，进入 Model.train_step
    outs = to_list(outs)
    callbacks.on_batch_end(batch_index, batch_logs)
    if callback_model.stop_training:
        break
```
Keras调用tf进行计算，是分batch进行操作，每个batch结束keras可以对返回进行相应的存储等操作。
``` python
def _call(self, inputs):
    if not isinstance(inputs, (list, tuple)):
        raise TypeError('`inputs` should be a list or tuple.')
    session = get_session()
    feed_arrays = []
    array_vals = []
    feed_symbols = []
    symbol_vals = []
    #数据处理转换
    for tensor, value in zip(self.inputs, inputs):
        if is_tensor(value):
            # Case: feeding symbolic tensor.
            feed_symbols.append(tensor)
            symbol_vals.append(value)
        else:
            feed_arrays.append(tensor)
            # We need to do array conversion and type casting at this level, since `callable_fn` only supports exact matches.
            array_vals.append(np.asarray(value,dtype=tf.as_dtype(tensor.dtype).as_numpy_dtype))
    if self.feed_dict:
        for key in sorted(self.feed_dict.keys()):
            array_vals.append(np.asarray(self.feed_dict[key],dtype=tf.as_dtype(key.dtype).as_numpy_dtype))

    # Refresh callable if anything has changed.
    if (self._callable_fn is None or feed_arrays != self._feed_arrays or symbol_vals != self._symbol_vals or feed_symbols != self._feed_symbols or session != self._session):
        #生成一个可以调用的graph
        self._make_callable(feed_arrays,feed_symbols,symbol_vals,session)
    #运行graph
    if self.run_metadata:
        fetched = self._callable_fn(*array_vals, run_metadata=self.run_metadata)
    else:
        fetched = self._callable_fn(*array_vals)
    #返回结果
    return fetched[:len(self.outputs)]
```
PS：**在高层是 layer/model 等概念，在底层，都是op 构成的数据流图，layer/model 隐藏了op**。

### 其它

#### 和session 的关系
和 TensorFlow session 的关系：Keras doesn't directly have a session because it supports multiple backends. Assuming you use TF as backend, you can get the global session as:

```
from keras import backend as K
sess = K.get_session()
```

If, on the other hand, yo already have an open Session and want to set it as the session Keras should use, you can do so via: `K.set_session(sess)`

#### callback

callbacks能在fit、evaluate和predict过程中加入伴随着模型的生命周期运行，目前tensorflow.keras已经构建了许多种callbacks供用户使用，用于防止过拟合、可视化训练过程、纠错、保存模型checkpoints和生成TensorBoard等。

callback 的关键是实现一系列 on_xx方法，callback与 训练流程的协作伪代码

```python
callbacks.on_train_begin(...)
for epoch in range(EPOCHS):
    callbacks.on_epoch_begin(epoch)
    for i, data in dataset.enumerate():
        callbacks.on_train_batch_begin(i)
        batch_logs = model.train_step(data)
        callbacks.on_train_batch_end(i, batch_logs)
    epoch_logs = ...
    callbacks.on_epoch_end(epoch, epoch_logs)
final_logs=...
callbacks.on_train_end(final_logs)
```

## Estimator

[Introduction to TensorFlow Datasets and Estimators](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html)

![](/public/upload/machine/estimator.png)

1. 创建一个或多个输入函数，即input_fn
    ```python
    # 第一个返回值 must be a dict in which each input feature is a key, and then a list of values for the training batch.
    # 第二个返回值 is a list of labels for the training batch.
    def input_fn(file_path,perform_shuffle,repeat_count):
    ...
    将输入 转为 Dataset 再转为 input_fn 要求的格式
    ...
    return ({ 'feature_name1':[values], ..<etc>.., 'feature_namen':[values] },
            [label_value])
    ```
2. 定义模型的特征列,即feature_columns
3. 实例化 Estimator，指定特征列和各种超参数。
4. 在 Estimator 对象上调用一个或多个方法，传递适当的输入函数作为数据的来源。（train, evaluate, predict)

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
