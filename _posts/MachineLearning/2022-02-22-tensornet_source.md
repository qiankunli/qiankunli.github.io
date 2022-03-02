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

[TensorNet——基于TensorFlow的大规模稀疏特征模型分布式训练框架](https://mp.weixin.qq.com/s/v8HqeR7UYFs4Ex5p5tFyVQ)

## Wide&Deep模型Demo
```python
import tensornet as tn
import tensorflow as tf
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
TensorNet版的wide-deep相较于TensorFlow原生版的主要由如下5个改动：
1. 分布式训练strategy改为了tn.distribute.PsStrategy()。
2. 将sparse特征的feature column统一使用tn.feature_column.category_column适配。
3. 将模型的第一层统一使用tn.layers.EmbeddingFeatures替换。
4. 将tf.keras.Model切换为tn.model.Model。
5. 将optimizer切换为tn.optimizer.Optimizer。
PS：这五点也说明了我们可以如何扩展tf


## 源码结构

[tensornet源码调试解析](https://blog.csdn.net/horizonheart/article/details/112507439)

```
tensornet
    /core           // c++ 部分
        /main
            /py_wrapper.cc  // 使用 PYBIND11_MODULE 建立 python 与 c 函数的映射关系
        /kernels    // op kernel 实现
            /xx.cc
        /ops        // 为每个op kernel REGISTER_OP
        /BUILD      // 为每个op 生成so 及python wrapper
    tensornet       // python 部分
        /core       // op 对应的python wrapper，由 Bazel 生成
    examples
```


## PsStrategy 相对 MultiWorkerMirroredStrategy 做了什么

```
tn.distribute.PsStrategy()  # 执行 __init__
    super(OneDeviceStrategy, self).__init__(PsExtend(self))
        tn.core.init() # PsExtend() 执行
        # py_wrapper.cc 中由 PYBIND11_MODULE 注册，对应一个匿名函数
        []() {
            PsCluster* cluster = PsCluster::Instance(); 	// 获取ps集群的实例
            if (cluster->IsInitialized()) {                 // 判断集群是否初始化
                return true;
            }
            if (cluster->Init() < 0) {                      // 对ps集群进行初始化操作
                throw py::value_error("Init tensornet fail");
            }
            tensorflow::BalanceInputDataInfo* data_info = tensorflow::BalanceInputDataInfo::Instance();
            if (data_info->Init(cluster->RankNum(), cluster->Rank()) < 0) {
                throw py::value_error("Init BalanceInputDataInfo fail");
            }
            return true;
        }
```

## tn.feature_column.category_column相比tensorflow自带的 有何不同

## tn.layers.EmbeddingFeatures的核心实现逻辑

```
class EmbeddingFeatures(Layer):
```

## tn.model.Model都做了哪些工作

## tn.optimizer.Optimizer是如何实现梯度参数更新和自身参数存储的