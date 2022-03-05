---

layout: post
title: tensornet源码分析
category: 架构
tags: MachineLearning
keywords: tensornet

---

## 简介

* TOC
{:toc}

[tensorflow ML框架外接ps方案](https://mp.weixin.qq.com/s/lLI8JrWNQMPW3uSYf2bX7w) 可以看到一些扩展tf的思路

[tensornet框架初探-（一）从Wide&Deep模型Demo出发看框架实现](https://zhuanlan.zhihu.com/p/349313868)

[TensorNet——基于TensorFlow的大规模稀疏特征模型分布式训练框架](https://mp.weixin.qq.com/s/v8HqeR7UYFs4Ex5p5tFyVQ)

通过对 tensornet 的分析可以看到，机器学习框架中，python 调用c 有两种方式
1. 一种方式是直接通过 pybind11 调用c++ 对应函数，调用时即立即执行
2. 一种调用 自定义算子，由session.run 真正触发执行。 
所以机器学习框架的执行，从编程语言角度看
![](/public/upload/machine/tensornet_python_c.png)
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
# tensornet/examples/main.py
def main():
    strategy = tn.distribute.PsStrategy()
    model, sub_model = create_model()
    dense_opt = tn.core.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    model.compile(optimizer=tn.optimizer.Optimizer(dense_opt),loss='binary_crossentropy',metrics=['acc', "mse", "mae", 'mape',  tf.keras.metrics.AUC(),tn.metric.CTR(), tn.metric.PCTR(), tn.metric.COPC()])

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


```python
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
PsCluster 定义如下
```c++
// tensornet/core/ps/ps_cluster.h
class PsCluster {
public:
    static PsCluster* Instance();
    int Init();
    int Rank() const;           // 当期节点编号
    size_t RankNum() const;     // 集群节点数量
    const PsServerInterface* GetServer(int shard_id) const; // 获取PsServer
public:
    PsLocalServer local_server;     // 当前节点对应的PsServer
private:
    bool is_initialized_ = false;
    std::unique_ptr<brpc::Server> server_;
    PsServiceImpl ps_service_impl_;
    std::vector<std::unique_ptr<PsRemoteServer>> remote_servers_;       // 远端PsServer 引用
    std::vector<std::string> workers_;                                      
```
PsServer 接口定义如下，就是sparse 和 Dense 参数的pull 和push
```c++
// tensornet/core/ps/ps_server_interface.h
class PsServerInterface {
public:
    virtual void SparsePullAsync(brpc::Controller *cntl,const SparsePullRequest *request,SparsePullResponse *response,Callback done) const = 0;
    virtual void SparsePushAsync(brpc::Controller *cntl,const SparsePushRequest *request,SparsePushResponse *response,Callback done) const = 0;
    virtual void DensePushPullAsync(brpc::Controller *cntl,const DensePushPullRequest *request,DensePushPullResponse *response,Callback done) const = 0;
    virtual void DatasetPullAsync(brpc::Controller *cntl,const DatasetPullRequest *request,DatasetPullResponse *response,Callback done) const = 0;
```

## tn.feature_column.category_column相比tensorflow自带的 有何不同（未完成）

## tn.layers.EmbeddingFeatures的核心实现逻辑

### python 调用c++函数 和 opKernel

![](/public/upload/machine/tensornet_sparse_table.png)

sparse 参数的活儿都让 SparseTable 干了。
```python
# when this layer is been called, all the embedding data of `feature_columns` will be pulled from ps server and return as a tensor list.
# tensornet/tensornet/layers/embedding_features.py
class EmbeddingFeatures(Layer):
    def __init__(self,feature_columns,...):
        self._feature_columns = feature_columns
        self._state_manager = StateManagerImpl(self, name, sparse_opt, dim, self.trainable
    def call(self, features, cols_to_output_tensors=None):
        using_features = self.filter_not_used_features(features)
        transformation_cache = fc.FeatureTransformationCache(using_features)
        self.sparse_pulling_features = self.get_sparse_pulling_feature(using_features)
        pulled_mapping_values = self._state_manager.pull(self.sparse_pulling_features)
        output_tensors = []
        for column in self._feature_columns:
            mapping_value = pulled_mapping_values[column.categorical_column.name]
            with ops.control_dependencies([mapping_value]):
                tensor = column.get_dense_tensor(transformation_cache,self._state_manager)
            processed_tensors = self._process_dense_tensor(column, tensor)
            output_tensors.append(processed_tensors)
        return output_tensors
    def backwards(self, grads_and_vars):
        return self._state_manager.push(grads_and_vars, self.sparse_pulling_features)
class StateManagerImpl(fc.StateManager):
    def __init__(self, layer, name, ...):
        self.sparse_table_handle = tn.core.create_sparse_table(sparse_opt, name if name else "", dimension)
        #  py_wrapper.cc 中由 PYBIND11_MODULE 注册，对应一个匿名函数
        [](py::object obj, std::string name, int dimension) {
            OptimizerBase* opt = static_cast<OptimizerBase*>(PyCapsule_GetPointer(obj.ptr(), nullptr));
            PsCluster* cluster = PsCluster::Instance();
            SparseTable* table = CreateSparseTable(opt, name, dimension, cluster->RankNum(), cluster->Rank());
            return table->GetHandle();
        }
        self.pulled_mapping_values = {}
        self._cols_to_var_map = collections.defaultdict(lambda: None)
        self._var_to_cols_map = collections.defaultdict(lambda: None)
    def create_variable(self,feature_column,name,shape,...)
    def get_variable(self, feature_column, name)
    def pull(self, features):
        ...
        pulled_mapping_values = gen_sparse_table_ops.sparse_table_pull(
            [var.handle for var in vars],
            [f.values for f in feature_values],
            table_handle=self.sparse_table_handle)
        ...
    def push(self, grads_and_vars, features):
        ...
        return gen_sparse_table_ops.sparse_table_push(feature_values, grads,table_handle=self.sparse_table_handle)
    def get_feature_mapping_values(self, column_name)
    def save_sparse_table(self, filepath, mode):
        return tn.core.save_sparse_table(self.sparse_table_handle, filepath, mode)
    def load_sparse_table(self, filepath, mode):
        return tn.core.load_sparse_table(self.sparse_table_handle, filepath, mode)
    def show_decay(self):
        return tn.core.show_decay(self.sparse_table_handle)
```
StateManagerImpl 初始化时 返回了一个table Handle，传给op 时会携带table Handle(uint32_t)
```c++
// tensornet/core/ps/table/sparse_table.cc
SparseTable* CreateSparseTable(const OptimizerBase* opt, const std::string& name,int dimension, int shard_num, int self_shard_id) {
    SparseTable* table = new SparseTable(opt, name, dimension, shard_num, self_shard_id);
    table->SetHandle(SparseTableRegistry::Instance()->Register(table));
    return table;
}
SparseTable::SparseTable(const OptimizerBase* opt, const std::string& name,int dimension, int shard_num, int self_shard_id)
    : shard_num_(shard_num)
    , self_shard_id_(self_shard_id)
    , opt_(opt)
    , dim_(dimension)
    , name_(name) {
    CHECK(opt_ != nullptr);
    op_kernel_ = opt_->CreateSparseOptKernel(dim_);
}
// tensornet/core/ps/table/sparse_table.h
class SparseTable {
public:
    SparseTable(const OptimizerBase* opt, const std::string& name,int dimension, int shard_num, int self_shard_id);
    ~SparseTable() = default;
    void Pull(const SparsePullRequest* req, butil::IOBuf& out_emb_buf, SparsePullResponse* resp);
    void Push(const SparsePushRequest* req, butil::IOBuf& grad_buf, SparsePushResponse* resp);
    void SetHandle(uint32_t handle);
    uint32_t GetHandle() const {return handle_;}
    void Save(const std::string& filepath, const std::string& mode);
    void Load(const std::string& filepath, const std::string& mode);
    void ShowDecay() const;
```

### spare_table_pull 逻辑

gen_sparse_table_ops.spare_table_pull ==> SparseTablePullKernel.ComputeAsync ==> SparsePullCall.Start ==> PsLocalServer/PsRemoteServer.SparsePullAsync ==> SparseTable.Pull

```c++
// tensornet/core/kernels/sparse_table_ops.cc
class SparseTablePullKernel : public AsyncOpKernel {
    void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
        ...
        PsCluster* cluster = PsCluster::Instance();
        for (size_t shard_id = 0; shard_id < cluster->RankNum(); shard_id++) {
            calls.emplace_back(new SparsePullCall(table_handle_, shard_id, dim));
        }
        ...
        Semaphore semaphore(calls.size());
        for (auto& call : calls) {
            call->Start([this, call, &var_infos, &semaphore]() {
                PopulatePulledVariable_(var_infos, call->call_sign_infos,call->resp, call->cntl.response_attachment());
                semaphore.Notify();
                delete call;
            });
        }
        semaphore.WaitForSemaphore();
    }
}
class SparsePullCall {
public:
    SparsePullCall(int table_handle, int shard_id, int dim)
        : shard_id_(shard_id) {
        req.set_table_handle(table_handle);
        req.set_dim(dim);
    }
    void Start(const tensornet::Callback& done) {
        if (call_sign_infos.empty()) {
            done();
        } else {
            const PsServerInterface* si = PsCluster::Instance()->GetServer(shard_id_);
            si->SparsePullAsync(&cntl, &req, &resp, done);
        }
    }
}
public:
    brpc::Controller cntl;
    SparsePullRequest req;
    SparsePullResponse resp;
private:
    int shard_id_ = -1;
```
如果是本机，则PsServer = local_server，否则PsServer = remote_server
```c
// tensornet/core/ps/ps_cluster.cc
const PsServerInterface* PsCluster::GetServer(int shard_id) const {
    if (Rank() == shard_id) {
        return &local_server;
    } else {
        CHECK_LT(shard_id, (int)remote_servers_.size());
        return remote_servers_[shard_id].get();
    }
}
```

### 底层存储

SparseTable 底层 使用8个SparseKernelBlock，每个SparseKernelBlock 使用unordered_map 存储kv（ps就是一个kv嘛），key是id，value是一个封装的SparseXXValue结构体。
1. SparseTable.Pull ==> SparseOptimizerKernel.GetWeight ==> SparseKernelBlock.GetWeight ==> SparseAdaGradValue.Weight
2. SparseTable.Push ==> SparseOptimizerKernel.Apply ==> SparseKernelBlock.Apply ==> SparseAdaGradValue.Apply

```c++
// tensornet/core/ps/optimizer/optimizer_kernel.h
static constexpr size_t SPARSE_KERNEL_BLOCK_NUM = 8;
template <typename KernelBlockType>
class SparseOptimizerKernel : public SparseOptimizerKernelBase {
    public:
        SparseOptimizerKernel(const OptimizerBase* opt, int dimension) {
            // 创建八个块进行存储，每个块是一个SparseKernelBlock对象
            for (size_t i = 0; i < SPARSE_KERNEL_BLOCK_NUM; ++i) {
                blocks_.emplace_back(opt, dimension);
            }
        }
        float* GetWeight(uint64_t sign) {
            int block_num = GetBlockId_(sign);
            return blocks_[block_num].GetWeight(sign);
        }
    private:
        std::vector<KernelBlockType> blocks_;
}
template <typename OptType, typename ValueType>
class SparseKernelBlock {
public:
    float* GetWeight(uint64_t sign) {
        const std::lock_guard<std::mutex> lock(*mutex_);
        // 返回一个  pair<iterator, bool> ，当 insert 将 val 成功添加到容器中时，返回的迭代器指向新添加的键值对，bool 值为 True，头插法
        auto inserted = values_.insert({sign, nullptr});
        if (inserted.second) {
            // inserted.first = iterator 也可以视为 iterator 第一个pair，inserted.first->first 第一个pair的key，inserted.first->second 第一个pair的value
            inserted.first->second = alloc_.allocate(dim_, opt_);
        }
        return inserted.first->second->Weight();
    }
}
private:
    const OptType* opt_ = nullptr;
    // unordered_map 是标准库中一个无排序的map，key,value,哈希函数
    std::unordered_map<uint64_t, ValueType*, decltype(sparse_key_hasher)> values_;
    Allocator<ValueType> alloc_;
```
SparseXXValue存储了当前embedding的维度，以及优化器的参数。
```c++
// tensornet/core/ps/optimizer/ada_grad_kernel.h
class alignas(4) SparseAdaGradValue: public SparseOptValue {
public:
    float* Weight() {
        return data_;
    }
private:
    float g2sum_;
    float data_[0];
// tensornet/core/ps/optimizer/ada_grad_kernel.cc
void SparseAdaGradValue::Apply(const AdaGrad* opt, SparseGradInfo& grad_info, int dim) {
    delta_show_ += grad_info.batch_show;
    float* w = Weight();
    double add_g2sum = 0;
    for (int i = 0; i < dim; ++i) {
        add_g2sum += grad_info.grad[i] * grad_info.grad[i];
    }
    g2sum_ += add_g2sum / dim;
    for (int i = 0; i < dim; ++i) {
        w[i] -= opt->learning_rate * grad_info.grad[i] / (opt->epsilon + sqrt(g2sum_));
    }
}
```
## tn.optimizer.Optimizer是如何实现梯度参数更新和自身参数存储的

对Optimizer 的使用，可以看到 optimizer 主要是为了更新dense参数(全连接网络的参数)
```python 
# tensornet/examples/main.py
dense_opt = tn.core.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
model.compile(optimizer=tn.optimizer.Optimizer(dense_opt),loss='binary_crossentropy',metrics=['acc', "mse", "mae", 'mape',  tf.keras.metrics.AUC(),tn.metric.CTR(), tn.metric.PCTR(), tn.metric.COPC()])
```
原版 Optimizer 使用 `opt.minimize(loss, var_list=[var1, var2])` 更新`var1, var2`。 计算梯度_compute_gradients 更新梯度 apply_gradients，分布式场景下 apply_gradients 会用到_distributed_apply，_distributed_apply 会用到 _resource_apply_sparse 和_resource_apply_dense
```python
# tensorflow/python/keras/optimizer_v2/optimizer_v2.py
# 实际使用的是其子类 tf.keras.optimizers.SGD`, `tf.keras.optimizers.Adam`
class OptimizerV2(trackable.Trackable):
    def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
        grads_and_vars = self._compute_gradients(loss, var_list=var_list, grad_loss=grad_loss, tape=tape)
        return self.apply_gradients(grads_and_vars, name=name) 
    # Add ops to apply sparse gradients to the variable `handle`
    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        raise NotImplementedError("Must be implemented in subclasses.")
    # Add ops to apply dense gradients to the variable `handle`
    def _resource_apply_dense(self, grad, handle, apply_state):
        raise NotImplementedError("Must be implemented in subclasses.")
```
Optimizer 最重要的改动是_distributed_apply，使用了自定义OP DenseTablePushPullKernel， 一个op 把 apply gradients 和 pull 都干了，_resource_apply_sparse 和 _resource_apply_dense 也就啥都不用干了。
```python
# tensornet/optimizer/optimizer.py
class Optimizer(optimizer_v2.OptimizerV2):
    def __init__(self,dense_opt,name='TensornetOptimizer',**kwargs):
        self.dense_table_handle = tn.core.create_dense_table(dense_opt)
        ...
    def save_dense_table(self, filepath):
        return tn.core.save_dense_table(self.dense_table_handle, filepath)
    def load_dense_table(self, filepath):
        return tn.core.load_dense_table(self.dense_table_handle, filepath)
    def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
        ...
        # 对应 DenseTablePushPullKernel
        gen_dense_table_ops.dense_table_push_pull(vars, grads, table_handle=self.dense_table_handle)
        super(Optimizer, self)._distributed_apply(distribution, grads_and_vars, name, apply_state)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        return control_flow_ops.no_op()
    def _resource_apply_dense(self, grad, var, apply_state=None):
        return control_flow_ops.no_op()
```
gen_dense_table_ops.dense_table_push_pull ==> DenseTablePushPullKernel.ComputeAsync ==> DensePushPullCall.Start ==> PsServer.DensePushPullAsync ==>  SparseOptimizerKernel->Apply + SparseOptimizerKernel->GetWeight

## tn.model.Model都做了哪些工作
原版Model.train_step 实现
```python
# tensorflow/python/keras/engine/training.py
class Model(base_layer.Layer, version_utils.ModelVersionSelector):
    def train_step(self, data):
        ...
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        ...
```
train_step 只会调用优化器更新dense参数，然后调用backwards方法更新id特征的参数。
```python
# tensornet/tensornet/model/Model.py
class Model(tf.keras.Model):
    # This method should contain the mathematical logic for one step of training. This typically includes the forward pass,loss calculation, backpropagation,and metric updates.
    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True) # 计算预测值
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses) # 计算损失
        gradients = tape.gradient(loss, self.trainable_variables)                # 计算梯度
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) # 更新dense参数
        self.backwards(list(zip(gradients, self.trainable_variables)))           # 更新id特征的参数，更新方法在EmbeddingFeatures中
        ...
    def backwards(self, grads_and_vars):
        backward_ops = []
        for layer in self.layers:
            if not hasattr(layer, 'backwards'):
                continue
            op = layer.backwards(grads_and_vars)
            if op:
                backward_ops.append(op)
            ...
```

