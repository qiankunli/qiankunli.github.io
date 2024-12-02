---

layout: post
title: 大模型推理tips
category: 架构
tags: MachineLearning
keywords: llm vLLM

---

* TOC
{:toc}

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## 简介（未完成）

## 以kvcache为核心的分布式架构

Mooncake 采用了以 KVCache 为中心的分离式推理架构，主要由三个核心部分组成：

1. Prefill 池：这个部分负责集中管理所有的预填充阶段的计算任务。
2. Decoding 池：这个部分集中处理所有解码阶段的任务。
3. KVCache 池：这个部分负责存储所有中间过程中应用到的 KVCache，并决定何时使用这些缓存，何时释放它们。

prefill-decode 分离架构核心是解决Continous Batching在decode中会被插入prefill从而导致decode卡顿以及decode阶段MFU低下这两个问题。

Context Caching 

![](/public/upload/machine/context_caching.jpg)

[Mooncake](https://github.com/kvcache-ai/Mooncake)

## 框架抽象

### 通用流程的抽象 

`前处理 → DNN推理 → 后处理`，无论是分类（classification）、检测（detection）、分割（segmentation）还是姿态估计（pose estimation）等任务，这一流程都是适用的。差异主要体现在前处理和后处理的具体实现上。
引擎创建： `builder → network → config → parser → serialize → save file`。 network 估计指的model 计算图解析和加载
引擎推理：`load file → deserialize → engine → context → enqueue`。file 估计指的是图片文件

**为实现代码的可复用性，我们可以采用面向对象的编程思想**，将通用的流程和操作封装在基类中，不同的任务通过继承和重写基类的方法，实现各自的特定逻辑。

```c++
class InferenceEngine {
public:
    virtual void buildEngine() = 0;
    virtual void loadEngine(const std::string& engineFile) = 0;
    virtual void preprocess(const cv::Mat& image) = 0;
    virtual void infer() = 0;
    virtual void postprocess() = 0;
    virtual ~InferenceEngine() {}
};
```

### 模型文件加载

如果使用c++ 来写推理或训练引擎的话，就没有python调用c这个复杂的事儿了。对于一个推理框架，大概可以理解为，
1. 专用的推理框架入口是onnx/pnnx等模型文件，只需要graph、节点/等概念，不需要pytorch 中类似layer概念（那是为了编程上抽象复用的）。 
2. 先基于onnx/pnnx等模型文件，自己提一套抽象/对象比如RuntimeGraph+RuntimeGraph+Operator等（为此有一个全局的算子注册机制），将模型权重、参数加载进来 构成计算图对象/内存表示，Operator 分为有参数算子和无参数算子，weight也就是tensor会赋值给有参数 Operator.weight。
3. RuntimeGraph.run 按拓扑排序执行，执行到某个节点RuntimeNode时，RuntimeNode为算子准备入参、拿到出参（也就是tensor），可能跨节点通信，Operator为 cuda 函数准备入参（cuda 函数的入参、出参也就是tensor，必须事先准备好 指针形式传给cuda函数）。概念上从大到小是Graph ==> node ==> Operator ==> cuda 函数。
4. tensor/显存的申请、释放都是上层组件负责（cuda 函数内不管，cuda 函数是无状态的），会有一个DeviceAllocator（分别对应cpu和gpu）组件负责内存和显存的分配和释放、内存和显存之间的copy等接口（比如tensor.to_cuda。再复杂一点先提前申请一个大的，内部再复用一下），对DeviceAllocator封装后提供tensor对象（tensor持有DeviceAllocator 引用，初始化时调用DeviceAllocator.allocate，析构时调用DeviceAllocator.release）。只是给算子函数传入input/weight/output 指针，算子也分为cpu和gpu实现。

### 资源管理的抽象

对于资源的申请和释放，例如内存的分配和释放，我们也可以进行封装，使得这些操作对使用者透明。这不仅提高了代码的可复用性，也减少了内存泄漏的风险。

```c++
class MemoryManager {
public:
    MemoryManager(size_t size) {
        cudaMalloc(&devicePtr_, size);
    }
    ~MemoryManager() {
        cudaFree(devicePtr_);
    }
    void* getDevicePtr() const { return devicePtr_; }
private:
    void* devicePtr_;
};
```

我们希望我们的代码比较好的可读性，就意味着我们在设计的时候尽量通过接口来暴露或者隐蔽一些功能。比如说，我们可以使用worker作为接口进行推理。在main中，我们只需要做到`创建一个worker -> woker读取图片 -> worker做推理`就好了。同时，worker也只暴露这些接口。在worker内部，我们可以让worker根据main函数传入的参数，启动多种不同的task（分类、检测、分割）。

```c++
class Worker {
public:
    Worker(const std::string& taskType, const std::string& modelPath);
    void loadImage(const std::string& imagePath);
    void infer();
    void displayResult();
private:
    std::shared_ptr<InferenceEngine> engine_;
    cv::Mat image_;
};
```
在主程序中，我们只需要与 Worker 类交互：
```c++
int main() {
    Worker worker("classification", "model.engine");
    worker.loadImage("image.jpg");
    worker.infer();
    worker.displayResult();
    return 0;
}
```

为框架设计插件机制，允许用户自定义前处理、后处理等步骤。插件可以在运行时加载，方便功能的扩展。

```c++
class Plugin {
public:
    virtual void execute() = 0;
    virtual ~Plugin() {}
};

class CustomPreprocessor : public Plugin {
    void execute() override {
        // 自定义前处理逻辑
    }
};
```