---

layout: post
title: 微服务编排
category: 技术
tags: Architecture
keywords: micro service

---

## 简介

## 微服务架构的大环境

[Spring Cloud技术分析（序）](http://tech.lede.com/2017/03/15/rd/server/SpringCloud0/)微服务化的核心就是将传统的一站式应用根据业务拆分成一个一个的服务，并且强调DevOps和快速演化。**原本一个服务拆成多个服务，相互依赖的代码不再直接相互调用，便衍生出一系列工具比如zk、queue等来解决彼此依赖的问题，以减少代码分拆对业务逻辑的影响。**


[来自京东、唯品会对微服务编排、API网关、持续集成的实践分享（上）](https://my.oschina.net/u/1777263/blog/827661)基本要点：

1. 我们都以为用zookeeper做一致性工具天经地义，文中却用db做一致性，一切看场景。就注册中心的功能来说，[Netflix/eureka](https://github.com/Netflix/eureka)也比zookeeper更好些。换个方式考虑，注册中心本质为了服务调用方和提供方的解耦，存储服务注册信息。也就是能存数据的都可以用来做注册中心，但从可用性上考虑，zookeeper因为副本因素可靠性高些。一致性 ==> 副本 ==> 高可用性存储，这或许才是zookeeper等一致性工具的本质，其它的才是kv存储、通知机制等枝节。 
2. “七牛采用的是Mesos+Docker+自研调度系统的架构。Docker做环境封将，Mesos做资源调度，自研的调度系统负责对Docker进行弹性的调度。”这段话和hadoop yarn二元调度的理念是相通的。

[来自京东、宅急送对微服务编排、API网关、持续集成的实践分享（下）](http://itindex.net/detail/56642-%E4%BA%AC%E4%B8%9C-%E5%AE%85%E6%80%A5%E9%80%81-%E5%BE%AE%E6%9C%8D%E5%8A%A1)要点：

1. 京东的 Docker 主要解决资源调度的问题。在分配时不会分到同一个机架上，不会分到同一个主机上，还有不会分到很繁忙的机器上。
2. 七牛：现在在我们的平台上运行了数千种数据处理应用，每种处理的的请求量不一样，比如有些图片处理每秒可以达到数十万的请求量，而有一些则可能是每秒几万的请求量，几千种数据处理应用的高峰期也不一样，有些可能在早上，有些可能在晚上，并且每种处理都会存在突发流量的情况，比如一些电商型的客户，在做大促销时，就会导致图片处理的请求量突增，而一些音视频的客户在会在一些活动时会突然增长对音视频的处理。这段话充分体现出了云计算的价值，所以说，做docker还是在云计算公司机会更多一些。

[拆分后台服务的各种姿势](https://juejin.im/entry/5a3621c76fb9a045016809c6?utm_source=gold_browser_extension)有几句话描述的比较精辟：

1. 老生长谈：一个应用 ==> nginx => api 层 ==> 微服务 ==> cache/db.**微服务进一步的拆分导致了原先的服务进程退化成API层，仅仅负责整合新形成的逻辑层的返回结果。** 这也是我们进行微服务编排的逻辑起点。
2. 微服务划分的几个原则：

	* 非关键服务，尤其是常常发布的服务和关键服务隔离
	* 快服务和慢服务分离。把这两类接口放到一个服务中，慢接口占满了资源（线程），并阻塞了快接口。就像如果给自行车和汽车公用车道，自行车会挡住汽车一样。


[Netflix Conductor : 一个微服务的编排器](http://www.infoq.com/cn/articles/netflix-conductor-a-micro-service-orchestration)

[借助Apache Camel实现企业应用集成（EAI）在日益增长](http://www.infoq.com/cn/articles/eai-with-apache-camel)

[RockScript：用于编配微服务的脚本语言和引擎](http://www.infoq.com/cn/news/2017/11/rockscript-preview)

“编配（orchestration）还是编排（choreography）

## 面临的业务需求（未完成）

请求进来，调用各个微服务，取到的结果根据业务揉和成一个新的model，返回给用户。具体的讲：

1. 根据业务场景，需要串行、并行、并行和串行交叉调用微服务
2. 应对业务异常情况，对于某项微服务调用，失败忽略或失败后立即返回

## 未来前景

请求 ==> nginx ==> api gateway ==> 编排系统(编排系统可以有一个界面，动态生成各种dsl，提交给编排系统) ==> 各种微服务 ==> db