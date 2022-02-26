---

layout: post
title: tensornet源码分析
category: 架构
tags: MachineLearning
keywords: tensornet

---

## 简介(未完成)

* TOC
{:toc}

[tensorflow ML框架外接ps方案](https://mp.weixin.qq.com/s/lLI8JrWNQMPW3uSYf2bX7w) 可以看到一些扩展tf的思路

[tensornet框架初探-（一）从Wide&Deep模型Demo出发看框架实现](https://zhuanlan.zhihu.com/p/349313868)

Wide&Deep模型Demo
```python
import tensornet as tn
 
def columns_builder():
    columns = {}
    for slot in set(C.WIDE_SLOTS + C.DEEP_SLOTS):
        columns[slot] = tn.feature_column.category_column(key=slot)
 
    wide_columns = []
    for slot in C.WIDE_SLOTS:
        feature_column = tf.feature_column.embedding_column(columns[slot], dimension=1)
        wide_columns.append(feature_column)
 
    deep_columns = []
    for slot in C.DEEP_SLOTS:
        feature_column = tf.feature_column.embedding_column(columns[slot], dimension=8)
        deep_columns.append(feature_column)
 
    return wide_columns, deep_columns
 
def create_model(wide_columns, deep_columns):
    wide, deep = None, None
 
    inputs = {}
    for slot in set(C.WIDE_SLOTS + C.DEEP_SLOTS):
        inputs[slot] = tf.keras.layers.Input(name=slot, shape=(None,), dtype="int64", sparse=True)
 
    sparse_opt = tn.core.AdaGrad(learning_rate=0.01, initial_g2sum=0.1, initial_scale=0.1)
 
    if wide_columns:
        wide = tn.layers.EmbeddingFeatures(wide_columns, sparse_opt, name='wide_inputs', is_concat=True)(inputs)
 
    if deep_columns:
        deep = tn.layers.EmbeddingFeatures(deep_columns, sparse_opt, name='deep_inputs', is_concat=True)(inputs)
 
        for i, unit in enumerate(C.DEEP_HIDDEN_UNITS):
            deep = tf.keras.layers.Dense(unit, activation='relu', name='dnn_{}'.format(i))(deep)
 
    if wide_columns and not deep_columns:
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(wide)
    elif deep_columns and not wide_columns:
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(deep)
    else:
        both = tf.keras.layers.concatenate([deep, wide], name='both')
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(both)
 
    model = tn.model.Model(inputs, output)
 
    dense_opt = tn.core.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    model.compile(optimizer=tn.optimizer.Optimizer(dense_opt),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(),])
 
    return model
```
TensorNet版的wide-deep相较于TensorFlow原生版的主要由如下5个改动：
1. 分布式训练strategy改为了tn.distribute.PsStrategy()。
2. 将sparse特征的feature column统一使用tn.feature_column.category_column适配。
3. 将模型的第一层统一使用tn.layers.EmbeddingFeatures替换。
4. 将tf.keras.Model切换为tn.model.Model。
5. 将optimizer切换为tn.optimizer.Optimizer。


[tensornet源码调试解析](https://blog.csdn.net/horizonheart/article/details/112507439)

```
tensornet
    core        // c++ 部分
    tensornet   // python 部分
    examples
```


```python
import tensorflow as tf
import tensornet as tn
def create_model():
    if C.MODEL_TYPE == "DeepFM":
        return DeepFM(C.LINEAR_SLOTS, C.DEEP_SLOTS, C.DEEP_HIDDEN_UNITS)
    elif C.MODEL_TYPE == "WideDeep":
        return WideDeep(C.LINEAR_SLOTS, C.DEEP_SLOTS, C.DEEP_HIDDEN_UNITS)
    elif C.MODEL_TYPE == "DCN":
        return DCN(C.DEEP_SLOTS, C.DEEP_HIDDEN_UNITS)
    else:
        sys.exit("unsupported model type: " + C.MODEL_TYPE)
def main():
    strategy = tn.distribute.PsStrategy()
    model, sub_model = create_model()
    dense_opt = tn.core.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    model.compile(optimizer=tn.optimizer.Optimizer(dense_opt),loss='binary_crossentropy',metrics=['acc', "mse", "mae", 'mape', tf.keras.metrics.AUC(),tn.metric.CTR(), tn.metric.PCTR(), tn.metric.COPC()])

    train_dataset = read_dataset(...）
    model.fit(train_dataset, epochs=1, verbose=1, callbacks=[cp_cb, tb_cb])
if __name__ == "__main__":
    main()

```