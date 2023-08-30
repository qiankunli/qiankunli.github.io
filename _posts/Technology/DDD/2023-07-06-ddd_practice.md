---

layout: post
title: ddd从理念到代码
category: 架构
tags: DDD
keywords: ddd cqrs

---

## 简介

* TOC
{:toc}

## 理念

[How To Implement Domain-Driven Design (DDD) in Golang](https://programmingpercy.tech/blog/how-to-domain-driven-design-ddd-golang/)
Domain-Driven Design is a way of structuring and modeling the software after the Domain it belongs to. What this means is that a domain first has to be considered for the software that is written. The domain is the topic or problem that the software intends to work on. The software should be written to reflect the domain. The architecture in the code should also reflect on the domain. 

[迄今为止最完整的DDD实践](https://mp.weixin.qq.com/s/y6H8UG-g829o0V0EBeEwrw )一些套话
1. 边界清晰的设计方法：通过领域划分，识别哪些需求应该在哪些领域，不断拉齐团队对需求的认知，分而治之，控制规模。
2. 统一语言：团队在有边界的上下文中有意识地形成对事物进行统一的描述，形成统一的概念(模型)。
3. 业务领域的知识沉淀：通过反复论证和提炼模型，使得模型必须与业务的真实世界保持一致。促使知识(模型)可以很好地传递和维护。
4. 面向业务建模：领域模型与数据模型分离，业务复杂度和技术复杂度分离。

## 设计

## 从概念开始理解DDD
聚合根、领域对象、领域服务、领域事件、仓储、贫血充血模型、界限上下文、通用语言

## 从分层开始理解DDD
https://github.com/KendoCross/kendoDDD 概念太多了，换一个角度，从MVC/MVP/MVVM的分层架构入手，类比理解DDD经典的四层。然后融合自己已有的编码习惯和认知，按照各层的主要功能定位，可以写的代码范围约束，慢慢再结合理解DDD的概念完善代码编写。

分层，分层架构有一个重要的原则：每层只能与位于其下方的层发生耦合。
1. 严格分层架构，某层只能与直接位于其下方的层发生耦合；自然是最理想化的，但这样肯定会导致大量繁琐的适配代码出现，故在严格与松散之间，追寻和把握恰达好处的均衡。
2. 松散分层架构，则允许任意上方层与任意下方层发生耦合。

各层理解
1. 用户接口层/Presentation。负责向用户显示信息和解释用户指令
   1. 出入口主要的功用逻辑也尽量的简单，主要承接不同“表现”形式采集到的指令/出入参等，并进行转发给应用层。和不同的Web/RPC框架有一定的耦合，不同的框架代码不全一样。
   2. 该层的核心价值在于多样化，而不在于功能有多强大，**不涉及到具体的业务逻辑**。
2.  应用层，定义软件要完成的任务，并且**指挥**表达领域概念的对象来解决问题。
   1. 应用层要尽量简单，不包含业务规则，只为下一层中的领域对象协调任务，分配工作。应用层是很薄的一层，只作为计算机领域到业务领域的过渡层。
   2. 这一层直接消费领域层，并且开始记录一些系统型功能，比如运行日志、事件溯源。 这一层的也应该尽可能的业务无关，以公用代码逻辑为主。
   3. 通过直接持有领域层的聚合根，infra层等直接进行业务表达。并将不常变化的domain model，转换为可能经常变化的view model。
3. 领域层。负责表达业务概念，业务状态信息以及业务规则。尽管保存业务状态的技术细节由基础设施层提供，但反应业务情况的状态是由本层控制并使用的。 
   1. 可以细拆分为聚合根、实体，领域服务等一大堆其他概念。
      1. 聚合根，负责整个聚合业务的所有功能就行了。比如项目中的fileAggregate，该类直接负责与平台系统管理员相关的所有操作业务，对内依赖调用领域服务、其他实体，或封装一些不对外的方法、函数等，完成所有所需的功能，由聚合根对外统一提供方法。**聚合根和其附属模型间有个共生死的约定（附属不可独自苟存）**
      2. 实体有唯一的标识，有生命周期且具有延续性。例如一个交易订单，从创建订单我们会给他一个订单编号并且是唯一的这就是实体唯一标识。同时订单实体会从创建，支付，发货等过程最终走到终态这就是实体的生命周期。订单实体在这个过程中属性发生了变化，但订单还是那个订单，不会因为属性的变化而变化，这就是实体的延续性。
      3. 实体的代码形态：我们要保证实体代码形态与业务形态的一致性。那么实体的代码应该也有属性和行为，也就是我们说的充血模型，但实际情况下我们使用的是贫血模型。贫血模型缺点是业务逻辑分散，更像数据库模型，充血模型能够反映业务，但过重依赖数据库操作，而且复杂场景下需要编排领域服务，会导致事务过长，影响性能。所以我们使用充血模型，但行为里面只涉及业务逻辑的内存操作。
      4. 实体的运行形态：实体有唯一ID，当我们在流程中对实体属性进行修改，但ID不会变，实体还是那个实体。
      5. 实体的数据库形态：实体在映射数据库模型时，一般是一对一，也有一对多的情况。
      6. 值对象：在 DDD 中用来描述领域的特定方面，并且是一个没有标识符的对象。值对象没有唯一标识，没有生命周期，不可修改，当值对象发生改变时只能替换（例如String的实现）。值对象是描述实体的特征，作为实体的属性，数据库里一般是一个字段。
      7. 聚合。The reason for an aggregate is that the business logic will be applied on the aggregate, instead of each Entity holding the logic. An aggregate does not allow direct access to underlying entities( all fields in the aggregate struct begins with lower case letters). aggregates  should only have one entity act as a root entity, this means that the root entity ID is the unique identifier of aggregate.
   2. 领域层不依赖基础层的实现，Repository接口在领域层定义好，由infra层依赖领域层实现这个接口。数据库操作都是基于聚合根操作，保证聚合根里面的实体强一致性。PS: 一个domain一个repository接口
4. 基础设施层。为上面各层提供通用的技术能力：为应用层传递消息，为领域层提供持久化机制等
   1. **这一层也是讲究和业务逻辑无关**，只重点提供通用的功能。主要需要编码的部分是仓储功能，单纯的（增删改查）数据库持久化等功能，变更的概率不大。 

## CQRS

命令查询职责分离
1. 命令(Command):不返回任何结果(void)，但会改变对象的状态。实践时还是让命令返回了一些主键之类的。
   1. 命令抽象出来之后，可以串一下表现层/应用层对该命令的消费。
      1. 表现层：将表现层接受来的请求主体 转换为Command ，并且进行参数校验，触发Command执行。
      2. 应用层：命令的Handler，一般都有相关领域上下文的聚合根来承担。基本逻辑
     ```
     commandHanlder.Func{
        1. 实例化领域实体 // 查询我们需要处理的实体数据，然后创建对应的领域对象，构造我们所定义的聚合。
        2. 调用实体行为 // 调用行为会修改实体的属性是内存上的
        3. 保存实体变更 // 把变更保存到基础设施层，例如 MySQL，PG等
     ```

   2. 每一个确定的命令操作，不论成功还是失败，只要执行之后就产生相应的事件（Event）。
      1. 事件的Handler，某个命令处理完毕之后，也即某个事件发生了，有可能需要短信通知、邮件通知等等。事件订阅者继续后续的逻辑。
2. 查询(Query):返回结果，但是不会改变对象的状态，对系统没有副作用。查询可以从应用层进行分离，直接操作infra层获取业务不是特别复杂的查询，这与没有引入CQRS的代码可以保持一致。PS： domain只是服务command，且domain 的持久化相对简单，一般都是根据id 进行crud。

![](/public/upload/ddd/ddd_layer_call.png)

## ddd框架

ddd这么多年一直曲高和寡的一部分原因是，在代码层面缺少框架支持，用户从0到1使用ddd从概念理解上和代码实现上都成本非常大，给人带来的困惑、给团队带来的争论相比便利来说一点都不少，这点相对“声明式API + 控制器模型”之于kubebuilder/controller-runtime 都差距很大。既提供了大量辅助代码（比如client、workqueue等）、自动生成代码（比如clientset）以减少代码量，又显式定义了实现规范（比如crd包含spec和status）和约束（实现reconcile等）。

ddd 可以封装的部分。比如抽象一个入口对象engine/bootstrap
1. 用户接口层
  1. 触发comamnd执行。engine.disptch("xxcommand",param) / engine.runCommand(xxCommand{param})
2. 应用层
  1. command接口规范
    ```go
   // 通用command 接口
    // 将command的基本动作: 构建domain；domain.bizFunc; 保存domain 通过接口的形式固化下来
    type ICommand interface{
    }
    type xxCommand struct {

    }
    // xxHandler 为用户 在操作domain 时注入一些能力，或者说与engine 交互
    func (xx xxCommand) handle(ctx context.Context, xx xxHandler){
        1. 构建domain
        2. domain.bizFunc
        3. xxHandler.cud(domain)
        4. eventbus.publish(event)
    }
    ```
3. domain层
  1. 基础父类，比如IEntity，每个entity均需实现IEntity，以约束其提供GetID等实现。
4. infra层
  1. domain model基于id的crud
    1. domain model/entity与po的转换
5. 其它
  1. 事件机制，entity变更时对外发出通知，可以减少domain.bizFunc 中关于非核心域的代码
  2. 事务，一个聚合包含多个entity，持久化时保持一致性
  3. 锁机制，在操作一个entity的时候，不允许其他线程操作entity，以免破坏一致性

用户接口层、应用层模式化（不同项目基本一样，只是改改名），domain层+repo层规范化，非业务/技术特性隐藏化。

![](/public/upload/ddd/ddd_engine.png)

![](/public/upload/ddd/ddd_engine_run_command.png)