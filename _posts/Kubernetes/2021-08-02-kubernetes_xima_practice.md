---

layout: post
title: 喜马拉雅容器化实践
category: 架构
tags: Kubernetes
keywords:  Kubernetes event

---

## 简介

* TOC
{:toc}

喜马的容器化历程伴随着 公司的发展，并带有很深刻的喜马的烙印

1. 以java 项目为主
2. java 项目主要分为 web 项目和rpc 项目（基于公司自研的、类似dubbo的框架）
3. 发布平台、rpc微服务框架、网关自研

在这个过程中，我们坚持了一些原则
1. 开发不用写Dockerfile，即开发不需要懂容器相关知识
2. 测试环境下，开发的电脑可以直接访问容器，即ip 可以互通。在任何环境，容器集群的物理机上的容器可以和其它物理机ip互通。
3. k8s 分为test、uat、product 三个环境，项目配置分为 跟环境有关的部分和无关的部分，发布系统及项目信息db部署在测试环境，发布系统可以 跨环境访问 test、uat、product 的三个k8s 集群。
4. 项目启动失败后，保留现场，不要一直重启

## 如何让开发发布代码到容器环境

2016年底，我在jenkins 机器上安装了docker，制作一个基于docker的项目模板。开发克隆docker 项目模板，更改源码地址、cpu、内存等配置，由jenkins 触发shell 脚本：maven编译项目得到war/jar 包；根据war/jar 拼装Dockerfile；`docker build` 制作docker 镜像并push 到镜像仓库；调用marathon 发布指令。

当时大概用了5天时间搭建了组件，跑通了所有步骤，我们的容器化事业就是从这个最low 的版本开始发展起来的，后来逐步做了以下演进
1. 从marathon 逐步迁移到kubernetes，在过渡时期，我们自己实现了一个docker 发布系统，对外提供发布接口，屏蔽marathon 和 Kubernetes 的差异。 
2. 实现了一个命令行工具barge。开发在项目文件中附加一个barge.yaml 文件，里面设定了项目名等基本配置
    1. 开发只需`barge deploy` 便可以发布项目，`barge exec -m $projectName` 便可以进入容器内调试代码
    2. 使用google jib 框架基于代码直接制作docker 镜像，免去了开发电脑上装docker的要求，此外也将代码编译、镜像制作过程分散到了每个开发的电脑上，绕过了jenkins 单机负载的瓶颈。
    3. 镜像仓库 harbor的单词 含义的是海港（对应容器的集装箱），大船是没办法直接开到海港的，到港口后要靠接驳船弄进去，接驳船英文是barge，所以给项目起名叫barge， 形容给harbor 送镜像。至今仍很得意这个命名。
3. 与公司的发布系统对接（类似阿里云效），屏蔽物理机与容器环境的使用差异

## 一个容器要不要多进程 

容器世界盛行的理念是：one process per container。 但容器在最开始落地时，为了降低推广成本，减少使用差异过大（相比物理机）给开发带来的不适应，需要在容器内运行ssh，实质上要求在容器内运行多进程，这需要一个多进程管理工具（entrypoint不能是业务进程），最终在 runit/systemd/supervisor 中选择了runit。 

此外，web 服务每次发布时ip 会变化，需要让nginx 感知到web 服务ip的变化。我们在每个容器内启动了一个nile 进程，负责将项目信息注册到zookeeper 上。使用微博一位小伙伴 开源的[upsync](https://github.com/weibocom/nginx-upsync-module) 插件，并改写了部分源码使其支持了zookeeper（upsync 只支持etcd和consul），进而使得nginx 的upstream 可以感知到项目实例ip的变化。后来在另一个场景中，nginx 改用了其它插件，我们将实例信息写入到 consul 上即可。

随着Kubernetes 的铺开，我们使用gotty + `kubectl exec` 逐步推广了在 浏览器上通过web console 访问容器。专门的网关系统也投入使用，http访问由`nginx ==> web服务`变成了`nginx ==> 网关 ==> web服务`，网关提供了web 接口同步web实例数据。 ssh 及nile 进程 逐步退出历史舞台。目前，我们的entrypoint 仍是runit， 并对其做了微调，当业务进程 进程启动失败时，不会重启（如果entrypoint是业务进程时则会频繁重启），以保留现场方便开发排查问题。


## 健康检查的三心二意

Kubernetes 有一个readiness probe，用来检测项目的健康状态，上下游组件可以根据项目的 健康状态采取 相关措施。在readiness probe 的设置上，经过多次改变

1. 每一个web项目会提供一个`/healthcheck` 接口，`/healthcheck` 通过即表示项目启动成功
2. 后来发现，对于rpc 服务，`/healthcheck` 有时不能表示项目启动成功，需要检测rpc 服务启动时监听的端口 是否开启
3. readiness probe配置（http 和 tcp方式任选）加大了业务开发的配置负担，经常出现配置不准确或中途改变导致readiness probe 探测失败，进而发出报警，业务开发不胜其烦，我们也非常困扰
4. 于是将 readiness probe 配置为 exec 方式，由nile 根据项目情况 自动执行http 和tcp 探测，但仍然依赖 项目配置信息（比如rpc 服务端口）的准确性
5. 基于“rpc 服务场景下，`/healthcheck` 接口成功 但rpc 服务启动失败  的场景非常的少”的判断，我们将readiness probe 改回了http `/healthcheck` 方式。


liveness 起初是`/healthcheck`
1. 有一次运维改动机房网络，导致liveness probe 探测失败（kubelet 无法访问本机所在 容器ip），kubelet 认为liveness probe  不通过 ==> 大量项目重启 ==> 因为项目之间存在依赖关系，依赖服务失败，项目本身启动失败 ==> 频繁重启
2. 后来我们不再配置 liveness probe ，仅配置readiness probe
    1. 对于物理机坏等场景，k8s 可以正常恢复
    2. 如果容器内存不足 导致实例挂掉，k8s 无法自动重启，这个可以通过内存报警来预防

## 与发布平台对接 

![image](http://note.youdao.com/yws/res/12421/WEBRESOURCE02a66c2753792c71eaf78e7d21092fa0)

喜马拉雅在发布平台的实践中，为了保障线上服务的稳定，沉淀了一套自己的经验和规则。其中一条**硬性规定**就是：服务端发布任何项目，最开始只能先发布一个实例，观察效果，确认无误后，再发布剩余实例。 在实际使用时，验证时间可能会非常长，有时持续一周。这时，Kubernetes deployment 自带的滚动发布机制便有些弱。

因此，在容器发布系统的设计上，我们让一个项目对应两个deployment。新代码的发布 就是旧deployment replicas 逐步缩小，新deployment replicas 逐步扩大的过程。 业界有两种方案
1. 每次发布新实例，都创建一个deployment，扩大新deployment 数量，缩小旧deployment 数量。
2. 每次发布新实例，都创建一个deployment 作为灰度deployment。灰度验证成功，则更新旧的deployment。
我们最开始使用的是第一种方案，缺点就是发布工作分散在两个deployment 上，所有实例的销毁与创建速度无法控制。此外因为deployment 名字不确定，导致hpa 配置比较麻烦。

阿里的openkruise 实现了一个crd 名为 cloneset，可以实现上述类似的功能，后续计划逐步替换掉两个deployment方案 以精简发布系统代码。 全新设计一个crd 也是一个很不错的方案 [阿里云 ACK + OpenKruise 助力掌门教育实现下一代容器发布系统 Triton](https://mp.weixin.qq.com/s/EOrTBNDYp5ss0knOq_hGIQ)

此外，容器发布系统对下屏蔽Kubernetes ，对上提供项目发布、回滚、扩缩容等接口，在接口定义上也多次反复。kubernetes client-java库 也较为臃肿，为此对发布代码也进行了多次优化。

针对开发日常的各种问题，比如项目ip是多少，项目为什么启动失败等，我们专门开发了一个后台，起了一个很唬的名字：容器云平台。试图将我们日常碰到的客服问题通过技术问题来解决，**通过一个平台来归拢**，哪怕不能实现开发完全自助式的操作，也可以尽量减少容器开发的排查时间。比如我们开发了一个wrench 检查组件，用来扫描java 项目的classpath、日志等，分析项目是否有jar 冲突、类冲突、tomcat 日志报错、业务日志报错等，开发在web页面 上触发wrench 执行即可。

[云原生下的灰度体系建设](https://mp.weixin.qq.com/s/xKfBizlmImyGB0QRGKL_pQ)阿里对灰度发布的系统性描述。
[云原生 CI/CD 框架 Tekton 初体验](https://mp.weixin.qq.com/s/ZI9vWJ4giVsMhxZYHjjd5A)

## 与已有中间件对接

喜马有自己自研的网关及rpc 框架，最初很少考虑容器的使用场景（比如每次发布ip 会变化）。此外，服务实例需要先调用 网关和rpc 框架提供的上线接口，才可以对外提供服务。服务销毁时，也必须先调用下线接口，待服务实例将已有流量处理完毕后再销毁。

为此我们提供了一个k8s-sync 组件，监听pod 状态变化，在适当实际调用 实例的上下线接口。后续计划自定义crd 比如WebService/RpcService，由自定义controller 完成实例的上下线 及其它工作。

**如何做到web/rpc 服务的无损上下线？**
1. 上线，pod 配置readiness probe 对项目的healthcheck 进行健康检查。健康检查通过后，pod 进入ready 状态， k8s-sync 监听到pod ready事件，调用web/rpc 上线接口。readiness 由k8s 确保执行，可保证健康检查不成功不会上线。风险是如果k8s-sync 挂了 则健康检查通过的 pod 无法上线。解决办法
    1. k8s-sync 容器部署，或者systemctl 管理，挂了之后可以立即重启
    2. k8s-sync 通过Prometheus 抓取数据，配置了报警，挂了之后可以迅速通知到容器开发。
2. 下线，k8s 先执行pod preStop逻辑，preStop 先调用web/rpc 下线接口，执行xdcs 提供的零流量检查接口， 检查通过，则preStop 执行完毕，销毁。 若项目未接入xdcs，则等待preStop 等待10s后，执行完毕。 销毁 ==> preStop 由k8s 负责确保执行， 正常发布/删除/节点移除/节点驱逐 都可以确保被执行，以确保服务无损。**如果物理机节点直接挂了**，则无法保证无损，因为一系列机制都来不及执行。 


k8s-sync 同时也将 pod 信息同步到mysql 中，方便开发根据项目名、ip 等在后台查询并访问项目容器。随着k8s-sync 功能的丰富，我们对k8s-sync 也进行了多次重构。

[微服务与K8s生命周期对齐](https://mp.weixin.qq.com/s/ahqtXp56o4943wmJp6x4zg)

## 体会

从16年到现在，一路走来，容器的各种上下游组件、公司的中间件并不是一开始就有或成熟的，落地过程真的可以用“逢山开路遇水搭桥” 来形容。

1. 我们花了很大的精力 在容器 与公司已有环境的融合上，比学习容器技术本身花的精力都多。为此我经常跟小伙伴讲，我们的工作是容器化，其实质上容器技术“喜马拉雅化”
    1. 技术上与各种中间件做整合。为此开发了k8s-sync，改造了upsync
    2. 不要求开发docker，写dockerfile。为此开发barge，容器云平台等
2. **实践的过程仿佛在走夜路**，很多方案心里都吃不准，为此看了很多文章，加了很多微信，很感谢一些大牛无私的交流。对于一些方案，走了弯路，兜兜转转又回到了原点。但这个过程逼得我们去想本质，去反思哪些因素是真正重要的，哪些因素其实没那么重要，哪些问题必须严防死守，哪些问题事后处理即可。
3. **成长经常不是以你计划的方式得到的**。
    1. 比如最开始搞Kubernetes 落地的时候，想着成为一个Kubernetes 高手，但最后发现大量的时间在写发布系统、k8s-sync，但再回过头，发现增进了对k8s的理解。
    2. 笔者java 开发出身，容器相关的组件都是go语言实现的，在最开始给了笔者很大的学习负担。但两相交融，有幸对很多事情任何的更深刻，比如亲身接触java的共享内存模型与go的CSP模型，最终促进了对并发模型的理解。
    3. 为了大家多用容器，我们做了很多本不归属于容器开发的工作。尤其是在容器化初期，很多开发项目启动一有问题便认为是docker的问题， 我们经常要帮小伙伴排查到问题原因，以证明不是docker的问题。**这是一个很烦人的事情，但又逼着我们把事情做的更好，更有利于追求卓越**。 比如wrench 的开发，真的用技术解决了一个看似不是技术的问题。
4. 一定要共同成长，多沟通，容器改变了基础设施、开发模式、习惯，尽量推动跟上级、同事同步，单靠一个人寸步难行。 

最后想想，感慨很多，还是那句：凡是过往，皆为序章。