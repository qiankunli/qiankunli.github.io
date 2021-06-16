---

layout: post
title: 《Apache Kafka源码分析》——server
category: 技术
tags: MQ
keywords: Scala  akka

---

## 前言

* TOC
{:toc}

建议先阅读下[《Apache Kafka源码分析》——Producer与Consumer](http://qiankunli.github.io/2017/12/08/kafka_clients.html)

服务端网络开发的基本套路

![](/public/upload/architecture/network_communication.png)

对应到kafka server

![](/public/upload/scala/kafka_server_framework.jpg)

对kafka好奇的几个问题：

1. 数据在磁盘上是如何存储的？

## 网络层

kafka client (producer/consumer) 与kafka server通信时使用自定义的协议，一个线程 一个selector 裸调java NIO 进行网络通信。 

面对高并发、低延迟的需求，kafka 服务端使用了多线程+多selector 

![](/public/upload/netty/kafka_server_nio.jpg)

这是一款常见的服务端 实现

1. 分层实现，网络io 部分负责读写数据，并将数据序列化/反序列化为协议请求。
2. 协议请求交给 上层处理， API层就好比 tomcat 中的servlet


## 单机——基本实现

### 初始化过程

![](/public/upload/scala/kafka_server_init.png)

### 整体思路

kafka 服务端核心是 KafkaServer，KafkaServer 没什么特别的，聚合和启动各个能力对象即可（kafka 称之为subsystem）。各能力对象都用到了统一的线程池，各自干各自的活儿。

1. LogManager。 The entry point to the kafka log management subsystem. The log manager is responsible for log creation, retrieval, and cleaning. All read and write operations are delegated to the individual log instances. LogManager 负责日志文件的维护，单纯的日志写入交给了Log 对象
2. ReplicaManager的主要功能是管理一个Broker 范围内的Partition 信息。代码上，Partition 对象为ReplicaManager 分担一部分职能
3. KafkaController，在Kafka集群的多个Broker中， 有一个Broker会被推举为Controller Leader，负责管理整个集群中分区和副本的状态。

![](/public/upload/scala/kafka_server_object.png)

从下文可以看到，broker 的主要逻辑就是接收各种请求并处理。除了使用了自定义网络协议导致网络层不一样，在api层/业务层，broker 与webserver 的开发逻辑是类似的。作为一个“webserver”，KafkaServer 的一个很重要组件是请求分发——KafkaApis。KafkaServer 将各个组件startup后，KafkaApis 聚合各个能力组件，将请求分发到 各个能力组件具体的方法上。

要注意两个层次的概念

1. broker 层次的leader 和 follower
2. replica 层次的leader 和 follower

![](/public/upload/scala/kafka_framework_3.png)

从单机角度看，自定义协议 + 主流程 + 旁路subsystem，与mysql 有点神似。

### 请求处理

![](/public/upload/scala/kafka_server_request_process.png)

kafka 的请求多达45种（2.3版本），分为两类：数据类请求；控制类请求。

当网络线程拿到请求后，它不是自己处理，而是将请求放入到一个共享请求队列中。Broker 端还有个 IO 线程池，负责从该队列中取出请求，执行真正的处理。如果是 PRODUCE 生产请求，则将消息写入到底层的磁盘日志中；如果是 FETCH 请求，则从磁盘或页缓存中读取消息。

上图的请求处理流程对于所有请求都是适用的，但因为控制类请求的重要性，社区于 2.3 版本正式实现了数据类请求和控制类请求的分离。

### 控制器

Broker 在启动时，会尝试去 ZooKeeper 中创建/controller 节点。Kafka 当前选举控制器的规则是：第一个成功创建 /controller 节点的 Broker 会被指定为控制器。

![](/public/upload/scala/kafka_controller.png)

假定最开始Broker 0 是控制器。当 Broker 0 宕机后，ZooKeeper 通过 Watch 机制感知到并删除了 `/controller` 临时节点。之后，所有存活的 Broker 开始竞选新的控制器身份。Broker 3 最终赢得了选举，成功地在 ZooKeeper 上重建了 `/controller` 节点。之后，Broker 3 会从ZooKeeper 中读取集群元数据信息，并初始化到自己的缓存中。至此，控制器的 Failover 完成，可以行使正常的工作职责了。

## 日志存储


### 整体思想

接收到客户端的请求之后，不同的系统（都要读写请求数据）有不同的反应

1. redis 读取/写入内存
2. mysql 读取/写入本地磁盘
3. web 系统，转手将请求数据处理后写入数据；或者查询数据库并返回响应信息，其本身就是一个中转站。

读写本地磁盘的系统 一般有考虑几个问题

1. 磁盘只能存储二进制数据或文本数据，文本数据你可以一行一行读取/写入。二进制数据则要求制定一个文件格式，一次性读写特定长度的数据。
2. 如果文件较大，为加快读写速度，还要考虑读写索引文件
3. 内存是否需要缓存热点磁盘数据

建议和mysql 对比学习下  [《mysql技术内幕》笔记1](http://qiankunli.github.io/2017/10/31/inside_mysql1.html)

|逻辑概念|对应的物理概念|备注|
|---|---|---|
|Log|目录|目录的命名规则`<topic_name>_<partition_id>`|
|LogSegment|一个日志文件、一个索引文件|命名规则`[baseOffset].log`和`[baseOffset].index` <br> baseOffset 是日志文件中第一条消息的offset|
|offset|消息在日志文件中的偏移|类似于数据库表中的主键<br> 由索引文件记录映射关系|

![](/public/upload/scala/kafka_index_file.jpg)

以索引文件中的3，205为例，在数据文件中表示第3个message（在全局partition表示第314个message），以及该消息的物理偏移地址为205。

当写满了一个日志段后，Kafka 会自动切分出一个新的日志段，并将老的日志段封存起来。Kafka 在后台还有定时任务会定期地检查老的日志段是否能够被删除，从而实现回收磁盘空间的目的。

### 写日志过程

![](/public/upload/scala/kafka_server_write_log.png)

## 分区 and 副本

副本机制可以保证数据的持久化或消息不丢失，但倘若Leader副本积累了太多的数据以至于单台 Broker 机器都无法容纳了，此时应该怎么办呢？如果你了解其他分布式系统，你可能听说过分片、分区域等提法，比如 MongoDB 和 Elasticsearch 中的 Sharding、HBase 中的 Region，其实它们都是相同的原理，只是 Partitioning 是最标准的名称。


follower replica是不对外提供服务的，只是定期地异步拉取领导者副本中的数据而已。既然是异步的，就存在着不可能与 Leader 实时同步的风险。

![](/public/upload/scala/kafka_replica_follower.png)

Kafka 引入了 In-sync Replicas，也就是所谓的 ISR 副本集合。ISR 中的副本都是与 Leader 同步的副本，相反，不在 ISR 中的追随者副本就被认为是与 Leader 不同步的。那么，到底什么副本能够进入到 ISR 中呢？

这个标准就是 Broker 端参数 `replica.lag.time.max.ms` 参数值。这个参数的含义是 Follower 副本能够落后 Leader 副本的最长时间间隔，当前默认值是 10 秒。这就是说，只要一个 Follower 副本落后 Leader 副本的时间不连连续超过 10 秒，那么 Kafka 就认为该 Follower 副本与 Leader 是同步的

![](/public/upload/scala/kafka_high_watermark.png)



我们假设这是某个分区 Leader 副本的高水位图。在分区高水位以下的消息被认为是已提交消息，反之就是未提交消息。消费者只能消费已提交消息，即图中位移小于 8 的所有消息。图中还有一个日志末端位移的概念，即 Log End Offset，简写是 LEO。它表示副本写入下一条消息的位移值。Kafka 所有副本都有对应的高水位和 LEO 值，只不过 Leader 副本比较特殊，Kafka 使用 Leader 副本的高水位来定义所在分区的高水位。

通过高水位，Kafka 既界定了消息的对外可见性（高水位之前的），又实现了异步的副本同步机制（高水位与LEO 之间的）。

||leader replica|follower replica|
|---|---|---|
|leo|接收到producer 发送的消息，写入到本地磁盘，更新其leo|follower replica从leader replica拉取消息，会告诉leader replica 从哪个位移处开始拉取<br>写入到本地磁盘后，会更新其leo|
|高水位|`currentHW = min(currentHW,LEO-1，LEO-2，……，LEO-n)`<br>所有replica leo的变化都会引起HW的重新计算|follower replica 更新完leo 之后，会比较其leo 与leader replica 发来的高水位值，用两者较小值作为自己的高水位|

## zookeeper

### 为什么要zookeeper，因为关联业务要交换元数据

kafka的主体是`producer ==> topic ==> consumer`，topic只是一个逻辑概念，topic包含多个分区，每个分区数据包含多个副本（leader副本，slave副本）。producer在发送数据之前，首先要确定目的分区（可能变化），其次确定目的分区的leader副本所在host，知道了目的地才能发送record，这些信息是集群的meta信息。producer每次向topic发送record，都要`waitOnMetadata(record.topic(), this.maxBlockTimeMs)`以拿到最新的metadata。

producer面对的是一个broker集群，这个meta信息找哪个broker要都不方便，也不可靠，本质上，还是从zookeeper获取比较方便。zookeeper成为producer与broker集群解耦的工具。

关联业务之间需要交换元数据，当然，数据库也可以承担这个角色，但数据库没有副本等机制保证可靠性

### 多机——基于zk协作的两种方式

在kafka中，broker、分区、副本状态等 作为集群状态信息，一旦发生改变，都会需要集群的broker作出反应，那么broker 之间如何协同呢？

在Kafka 早期版本中，每个broker 都会向zookeeper 上注册watcher，当分区或副本状态变化时会唤醒很多不必要的watcher， 导致羊群效应及zookeeper 集群过载。

在新版的设计中，只有Controller Leader 在zookeeper上注册wather，其它的broker 几乎不用再监听zookeeper 中的数据变化。 每个Broker 启动时都会创建一个KafkaController 对象，但是集群中只能存在一个Controller Leader来对外提供服务。在集群启动时，多个Broker上的KafkaController 会在指定路径下竞争创建节点，只有第一个成功创建节点的KafkaController 才能成为Leader（其余的成为Follower）。当Leader出现故障后，所有的Follower会收到通知，再次竞争新的Leader。KafkaController 与Broker 交互，Broker 处理来自KafkaController 的LeaderAndIsrRequest、StopReplicaRequest、UpdateMetadataRequest 等请求

简单说，老版本Broker 之间的数据传递依赖于Zookeeper，每个Broker 对zookeeper 的所有数据数据变化 相机做出反应 并更新zookeeper，比较低效。新版本Broker 选举出Controller Leader 后， 由Controller Leader 相机向各个Broker 发出指令。有人告诉你做什么，总比你拿到数据后自己分析判断再行动要容易些。

作为对比，hadoop 之类框架有明确的master/slave 之分，但为了高可用，master 往往要多个副本。除此之外，分布式框之间的协同 应该是相通的

1. 每个组件启动后，向zk 注册自己的信息
2. 接收master/leader 的各种请求（http协议或自定义协议） 并处理即可，处理完相机更新zk的数据

从这个角度看，每个slave 组件的逻辑与业务程序猿常写的web server 也别无二致

在安装kafka的时候，经常需要改三个配置文件。

1. server.properties, 配置zk地址
2. producer.properties, 配置broker列表，只有实际的生产端需要（估计是给命令行工具用的）
3. consumer.properties, 配置broker列表，只有实际的消费端需要（估计是给命令行工具用的）

早期consumer.properties 也是要配置 zk地址的，在靠后的版本就不需要了，这个变迁也体现了zk 作用的变化。producer.properties 未发现要配置zk 地址。

![](/public/upload/mq/kafka_zk.png)

图中圆角的矩形是临时节点，直角矩形是持久化的节点。
1. 左侧这棵树保存的是 Kafka 的 Broker 信息，/brokers/ids/[0…N]，每个临时节点对应着一个在线的 Broker，Broker 启动后会创建一个临时节点，代表 Broker 已经加入集群可以提供服务了，节点名称就是 BrokerID，节点内保存了包括 Broker 的地址、版本号、启动时间等等一些 Broker 的基本信息。如果 Broker 宕机或者与 ZooKeeper 集群失联了，这个临时节点也会随之消失。
2. 右侧部分的这棵树保存的就是主题和分区的信息。/brokers/topics/ 节点下面的每个子节点都是一个主题，节点的名称就是主题名称。每个主题节点下面都包含一个固定的 partitions 节点，pattitions 节点的子节点就是主题下的所有分区，节点名称就是分区编号。每个分区节点下面是一个名为 state 的临时节点，节点中保存着分区当前的 leader 和所有的 ISR 的 BrokerID。这个 state 临时节点是由这个分区当前的 Leader Broker 创建的。如果这个分区的 Leader Broker 宕机了，对应的这个 state 临时节点也会消失，直到新的 Leader 被选举出来，再次创建 state 临时节点。

## Kafka Pipeline

《软件架构设计》

![](/public/upload/scala/kafka_pipeline.png)

1. 对于ACK=ALL场景下，客户端每发送一条消息，要写入到Leader、Follower1和Follower2 之后，Leader 才会对客户端返回成功
2. Leader 不会主动给两个Follower 同步数据，而是等两个Follower 主动拉取，并且是批量拉取
3. 为什么叫pipeline呢？Leader处理完msg1就去处理msg2了，等Follower同步完成再告诉客户端msg1接收成功。**将一次消息处理分为接收消息和同步消息两个步骤，并且并行化了**。

主动拉取 是kafka 的一个重要特征，不仅是consumer 主动拉取broker， broker partition follower 也是主动拉取leader。

## 小结

面向对象的源码分析，一般先宏观（比如如何启动，比如业务逻辑的实现路径等）后细节，就是类图和主流程序列图

1. 类图表达依赖关系，反映了代码的组织和业务抽象
2. 主流程展示主要执行路径，反应了业务的逻辑

[声明式编程范式初探](http://www.nowamagic.net/academy/detail/1220525)命令式编程中的变量本质上是抽象化的内存，变量值是该内存的储存内容。JVM 线程通信 靠共享内存，反映在代码上 就是共享对象。

[源码分析体会](http://qiankunli.github.io/2019/01/24/source_parse.html)任何一个系统的设计都有功能和性能（泛化一下就是功能性和非功能性） 两个部分，识别系统模块属于哪个部分，有助于简化对系统的认识。通常，一个系统的最早版本只专注于功能，后续除非大的变动，后来的演化大部分都是为了性能上的追求。在kafka 这块，zk的协作方式等方面的变化 有很充分的体现。






