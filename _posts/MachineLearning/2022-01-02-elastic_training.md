---

layout: post
title: 基于Volcano的弹性训练
category: 架构
tags: MachineLearning
keywords:  elastic training

---

## 简介

* TOC
{:toc}


## 背景介绍

训练任务基于k8s执行，由volcano负责调度，算法人员归属于不同的部门，一个部门对应一个volcano queue。

因为GPU 比较贵，AI部门常背的一个考核目标是提高GPU 资源利用率。其次是减少训练任务的耗时，这样一天可以多跑几次模型，提高算法工程师的工作效率。所以，尽量别让GPU 闲着，都跑起来。但算法工程师提交任务时，并不知道集群的资源使用情况，只能凭经验和感觉配一个资源request，本质上是容量规划与实际负载的矛盾，所以尝试用弹性来解决。

本文方案仅针对pytorch，但就实现原理来讲也可以很方便扩展到其它训练框架。

弹性训练要解决几个问题
1. 能发现集群资源的富余与紧张，触发扩缩容。
2. 支持容错，因为扩散容期间worker会重启，worker 个数会变化，不能挂了一个worker就导致 之前的训练成果都丢了。这个一般通过checkpoint 来解决。

## 如何触发扩缩容

### HPA方案

很自然的会想到通过HPA来实现弹性扩缩容，HPA 的关键是选择一个metric作为扩缩容决策依据，自然的，我们选择“queue的可用GPU个数” 作为扩缩容的依据metric。

hpa扩缩容的计算规则

```
desiredReplicas = ceil[
                        currentMetricValue
    currentReplicas * -----------------------
                        desiredMetricValue 
]
```

可以通过设置desiredMetricValue来干预desiredReplicas值。 问题就是这个desiredMetricValue 值很不好设置
1. 假设我们将desiredMetricValue设置小一点，比如3，扩容时问题不大，但如果想要缩容，则只能等currentMetricValue=2 或 currentMetricValue=1 的时候，job replicas 一下子就砍去三分之一。 
2. 假设我们将desiredMetricValue设置的大一些，比如10，缩容的时候问题不大，但如果想要扩容，则只能currentMetricValue>10 了，此时集群空闲的资源就有点多了。

抽象一下，对一般业务/微服务进行HPA时，决定一个deployment是否扩缩容，应该拿一个跟pod 相关的metric，比如cpu 利用率， 这个时候公式是对的。而对于训练任务，扩缩容的依据 是一个跟pod 不相关的queue 的 metric，这个时候跟公式的内涵就对不上了。

解决这个问题有两个办法
1. 改公式
2. 换metric，或使用多个metric

无论如何，HPA 只能从单个job 的视角出发来决策，做扩缩容决策要考虑 queue的状况 以及同一queue 下其它runing 和 pending job的情况，要干这个活儿就得掌握所有这些信息，但HPA 要能访问这么多信息，就不太像一个HPA 做的事儿，更像是资源调度做的事儿。

### 让volcano支持弹性调度

以volcano 自带的vcjob 为例

```
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: default-job
spec:
  minAvailable: 2
  queue: default
  tasks:
    - replicas: 4
      template:
        spec:
          containers:
            - image: nginx
              name: nginx
```

从crd定义上，volcano已经支持了弹性调度需要的字段，  minAvailable=2 表示最小需要的worker数，replicas=4表示最大可能运行的workers数。 

operator（或controller）与volcano 的协作逻辑为
1. operator 为job创建volcano podgroup的时候，传给podgroup.minMember 不是当前副本数，而是min 的值。
2. operator reconcile时负责pod 数量=max。
3. 对于volcano 来说，一个job的pending和 running task数量之和表示max，一个job的minAvailable 表示min。

volcano掌握了要调度的所有pod、queue的资源使用情况， 便有足够的信息做出扩缩容决策。


## 弹性扩缩容的实现

训练代码层面，支持弹性之后，worker数量经常变化，每次变化都会重启worker
1. 需使用checkpoint 来暂存训练成果。rank=0 启动时load checkpoint， 每次epoch 训练结束后 save checkpoint，checkpoint内容包括但不限于：model 数据、optimizer 数据、本次训练进行到了哪个epoch等。
2. batch size、学习率等也应根据 worker数量（可以从env中获取WORLD_SIZE）进行一定调整。 

pytorch 层面，必须使用`python -m torch.distributed.run train_script.py`来启动训练脚本，原理参见 [云原生的弹性 AI 训练系列之二：PyTorch 1.9.0 弹性分布式训练的设计与实现](https://mp.weixin.qq.com/s/hlOYLKSHFDZWN21AsUn6bg)，测试过程中发现pytorch v1.10 缩容的bug，由腾讯的同学提交issue推进修复。


operator 层面（指pytroch-operator）

```yaml
pytorchjob 
   spec:
      elasticPolicy:
         min: 5
         max: 10
      pytorchReplicaSpecs:
        Worker:
            replicas: 10
            spec: 
                ...
```

1. 弹性场景下不再明确区分master 和worker 角色，否则replicas=1时只有master，replicas > 1 时包含master 和worker，crd 内容不一致。没有master 之后，在backend=c10d场景下，由worker0担任类似master的角色，为保险起见，应确保worker0先于其它worker启动。
2. crd支持elasticPolicy 配置，以设置最大和最小worker数，这个信息最终要传递给volcano。
3. 改变operator 判断job 成功和失败的策略。在之前的分布式训练场景下，假设一个job 10个实例，10个实例都成功则job成功，有一个实例失败则job置为失败，并开始清理其它worker。在弹性场景下，判定成功与失败的逻辑应给予放宽。

  
资源调度层面，即volcano，假设一个弹性job min=5,max=10，则对于worker6到worker10(也可以认为是worker5到worker9)视为job的弹性worker，对于弹性worker，应保持一个原则：不能因为弹性worker占满了资源，让本来可以运行的任务无法开始。具体的说应对弹性worker“靠后创建，优先销毁”
1. 先创建出所有job的min 个worker 后，再为job 创建弹性worker。
2. 如果有新job提交，但没有富余资源时，应优先杀死弹性worker以腾出资源。

此外，训练任务多个worker 有角色之分：一个master（或rank=0的worker） 和多个worker，master应确保“优先创建，靠后删除”，让master 优先级大于worker 可以实现这个效果，volcano之前从Pod.Priority来获取任务优先级，这需要operator层面维护PriorityClass，不太方便，因此提供了一个pr 让volcano 可以从Pod Annotation中获取pod 优先级。 

后续volcano 还可以支持配置“冷却时间”，因为训练job重启一次代价不小，所以一个job 的worker数量应尽量避免经常变动，在一次扩缩动作后，冷却一段时间再进行下一次扩缩。

## 小结

在实现弹性训练的过程中，发现了pytorch、pytorch-operator的一些bug，同时pytorch-operator、volcano需进行不小的功能增强，早知道这么复杂就不一定敢接这个活儿了。 

在推进过程中，感受到了社区的巨大力量，除了整体方案外，我只进行了vocalno弹性调度的开发，对pytorch-operator 提了一些想法和bug，感谢腾讯的pytorch-operator 开发者高策、张望的对pytorch controller elastic 模式的开发支持，感谢华为的volcano社区的吴雷、王雷博的支持。




