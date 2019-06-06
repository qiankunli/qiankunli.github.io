---

layout: post
title: 《Apache Kafka源码分析》——Producer与Consumer
category: 技术
tags: Scala
keywords: Scala  akka

---

## 前言

* TOC
{:toc}

建议先阅读下[消息/任务队列](http://qiankunli.github.io/2015/08/08/message_task_queue.html)，了解下消息队列中间件的宏观理论、概念及取舍

整体来说，本书是对源码的“照本宣科”，提炼的东西不多，试试另外一本书：《learning apache kafka》

[Apache Kafka](https://kafka.apache.org/intro) is a distributed streaming platform. What exactly does that mean?
A streaming platform has three key capabilities:

1. Publish and subscribe to streams of records, similar to a message queue or enterprise messaging system.
2. Store streams of records in a fault-tolerant durable way.
3. Process streams of records as they occur.

![](/public/upload/scala/kafka.png)

给自己提几个问题

1. kafka 将消息保存在磁盘中，在其设计理念中并不惧怕磁盘操作，它以顺序方式读写磁盘。具体如何体现？
3. 多面的offset。一个msg写入所有副本后才会consumer 可见（消息commit 成功）。leader / follower 拿到的最新的offset=LEO, 所有副本都拿到的offset = HW
4. 一个consumer 消费partition 到哪个offset 是由consumer 自己维护的

书中源码基于0.10.0.1

### 宏观概念

仅从逻辑概念上看

![](/public/upload/architecture/kafka_subscribe_publish_3.png)

每个topic包含多个分区，每个分区包含多个副本。作为producer，一个topic消息放入哪个分区，hash一下即可。 《learning apache kafka》every partition is mapped to a logical log file that is represented as a set of segment files of equal sizes. Every partition is an ordered, immutable sequence of messages; 

![](/public/upload/architecture/kafka_subscribe_publish.png)

整体架构图

![](/public/upload/scala/kafka_framework.jpg)

细化一下是这样的

![](/public/upload/scala/kafka_framework_2.jpg)

## 代码使用

    <dependency>
        <groupId>org.apache.kafka</groupId>
        <artifactId>kafka_2.8.2</artifactId>
        <version>0.8.0</version>
    </dependency>

### 生产者

    // 配置属性
    Properties props = new Properties();
    props.put("metadata.broker.list", "localhost:9092");
    props.put("serializer.class", "kafka.serializer.StringEncoder");
    props.put("request.required.acks", "1");
    ProducerConfig config = new ProducerConfig(props);
    // 构建Producer
    Producer<String, String> producer = new Producer<String, String>(config);
    // 构建msg
    KeyedMessage<String, String> data = new KeyedMessage<String, String>(topic, nEvents + "", msg);
    // 发送msg
    producer.send(data);
    // 关闭
    producer.close();

### 消费者

[Kafka系列（四）Kafka消费者：从Kafka中读取数据](http://www.dengshenyu.com/%E5%88%86%E5%B8%83%E5%BC%8F%E7%B3%BB%E7%BB%9F/2017/11/14/kafka-consumer.html)

    // 配置属性
    Properties props = new Properties();
    props.put("bootstrap.servers", "broker1:9092,broker2:9092");
    props.put("group.id", "CountryCounter");
    props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
    props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
    KafkaConsumer<String, String> consumer = new KafkaConsumer<String,String>(props);
    // 订阅主题
    consumer.subscribe(Collections.singletonList("customerCountries"));
    // 拉取循环
    try {
        while (true) {  //1)
            ConsumerRecords<String, String> records = consumer.poll(100);  //2)
            for (ConsumerRecord<String, String> record : records)  //3){
                log.debug("topic = %s, partition = %s, offset = %d,
                    customer = %s, country = %s\n",
                    record.topic(), record.partition(), record.offset(),
                    record.key(), record.value());
                int updatedCount = 1;
                if (custCountryMap.countainsValue(record.value())) {
                    updatedCount = custCountryMap.get(record.value()) + 1;
                }
                custCountryMap.put(record.value(), updatedCount)
                JSONObject json = new JSONObject(custCountryMap);
                System.out.println(json.toString(4))
            }
        }
    } finally {
        consumer.close(); //4
    }
	
## 背景知识

### 网络通信

kafka-producer/consumer 与zk 通信的部分相对有限，主要是与kafka server交互，通信时使用自定义的协议，一个线程（kafka 服务端一个线程就不够用了）裸调java NIO 进行网络通信。 

1. producer 使用 NetworkClient 与kafka server 交互
2. consumer 使用 ConsumerNetworkClient（聚合了NetworkClient）与kafka server 交互
3. 协议对象如下图所示，`org.apache.kafka.common.protocol.ApiKeys` 定义了所有 Request/Response类型，FetchXX 是一个具体的例子

    ![](/public/upload/scala/kafka_io_object.png)

4. NetworkClient 发请求比较“委婉” 先send（缓存），最后poll真正开始发请求

    1. send，Send a new request. Note that the request is not actually transmitted on the network until one of the `poll(long)` variants is invoked. At this point the request will either be transmitted successfully or will fail.Use the returned future to obtain the result of the send.
    2. poll，Poll for any network IO.   


### 传递保证语义（Delivery（guarantee） sematic）

Delivery guarantee 有以下三个级别

1. At most once，可以丢，但不能重复
2. At least once，不能丢，可能重复
3. exactly once，只会传递一次

这三个级别不是一个配置保证的，而是producer 与consumer 配合实现的。比如想实现“exactly once”，可以为每个消息标识唯一id，producer 可能重复发送，而consumer 忽略已经消费过的消息即可。

### consumer group rebalance

当consumer group 新加入一个consumer 时，首要解决的就是consumer 消费哪个分区的问题。这个方案kafka 演化了多次，在最新的方案中，分区分配的工作放到了消费端处理。

所谓的consumer group，指的是多个consumer实例共同组成一个组来消费topic。topic中的每个分区都只会被组内的一个consumer实例消费，其他consumer实例不能消费它。（从实现看，consumer 主动拉取的逻辑也不适合 多个consumer 同时拉取一个partition，因为宕机后无法重新消费。一个consumer 一个partition，server 端也无需考虑多线程竞争问题了）。

为什么要引入consumer group呢？主要是为了提升消费者端的吞吐量。多个consumer实例同时消费，加速整个消费端的吞吐量（TPS）。

consumer group里面的所有consumer实例不仅“瓜分”订阅topic的数据，而且更酷的是它们还能彼此协助。假设组内某个实例挂掉了，Kafka能够自动检测到，然后把这个 Failed 实例之前负责的分区转移给其他活着的consumer。这个过程就是 Kafka 中大名鼎鼎的Rebalance。其实既是大名鼎鼎，也是臭名昭著，因为由重平衡引发的消费者问题比比皆是。事实上，目前很多重平衡的Bug 社区都无力解决。

## 生产者

The producer connects to any of the alive nodes and requests metadata about the leaders for the partitions of a topic. This allows the producer to put the message directly to the lead broker for the partition.

大纲是什么？

1. 线程模型
2. 发送流程

![](/public/upload/scala/kafka_producer.jpg)

![](/public/upload/scala/kafka_producer_object.png)

1. producer 实现就是 业务线程（可能多个线程操作一个producer对象） 和 io线程（sender，看样子应该是一个producer对象一个）生产消费的过程
2. 从producer 角度看，topic 分区数量以及 leader 副本的分布是动态变化的，Metadata 负责屏蔽相关细节，为producer 提供最新数据
3. 发送消息时有同步异步的区别，其实底层实现相同，都是异步。业务线程通过`KafkaProducer.send`不断向RecordAccumulator 追加消息，当达到一定条件，会唤醒Sender 线程发送RecordAccumulator 中的消息
4. ByteBuffer的创建和释放是比较消耗资源的，为了实现内存的高效利用，基本上每个成熟的框架或工具都有一套内存管理机制，对应到kafka 就是 BufferPool
5. 业务线程和io线程协作靠队列，为什么不直接用队列？

    1. RecordAccumulator acts as a queue that accumulates records into MemoryRecords instances to be sent to the server.The accumulator uses a bounded amount of memory and append calls will block when that memory is exhausted, unless this behavior is explicitly disabled.
    3. 用了队列，才可以batch和压缩。

6. If the request fails, the producer can automatically retry, though since we have specified etries as 0 it won't. Enabling retries also opens up the possibility of duplicates 

### 业务线程

producer 在KafkaProducer 与 NetworkClient 之间玩了多好花活儿？

![](/public/upload/scala/kafka_producer_send.png)

### sender 线程

![](/public/upload/scala/kafka_sender.png)


### 值得学习的地方——interceptor

在发送端，record发送和执行record发送结果的callback之前，由interceptor拦截

1. 发送比较简单，record发送前由interceptors操作一把。`ProducerRecord<K, V> interceptedRecord = this.interceptors == null ? record : this.interceptors.onSend(record)`
2. `Callback interceptCallback = this.interceptors == null ? callback : new InterceptorCallback<>(callback, this.interceptors, tp);`

对于底层发送来说，`doSend(ProducerRecord<K, V> record, Callback callback)`interceptors的加入并不影响（实际代码有出入，但大意是这样）。

### 值得学习的地方——反射的另个一好处

假设你的项目，用到了一个依赖jar中的类，但因为策略问题，这个类对有些用户不需要，自然也不需要这个依赖jar。此时，在代码中，你可以通过反射获取依赖jar中的类，避免了直接写在代码中时，对这个jar的强依赖。

## 消费者

While subscribing, the consumer connects to any of the live nodes and requests metadata about the leaders for the partitions of a topic. The consumer then issues a fetch request to the lead broker to consume the message partition by specifying the message offset (the beginning position of the message offset). Therefore, the Kafka consumer works in the pull model and always pulls all available messages after its current position in the Kafka log (the Kafka internal data representation).

[读Kafka Consumer源码](https://www.cnblogs.com/hzmark/p/kafka_consumer.html) 对consumer 源码的实现评价不高

开发人员不必关心与kafka 服务端之间的网络连接的管理、心跳检测、请求超时重试等底层操作，也不必关心订阅Topic的分区数量、分区leader 副本的网络拓扑以及consumer group的Rebalance 等kafka的具体细节。

KafkaConsumer 依赖SubscriptionState 管理订阅的Topic集合和Partition的消费状态，通过ConsumerCoordinator与服务端的GroupCoordinator交互，完成Rebalance操作并请求最近提交的offset。Fetcher负责从kafka 中拉取消息并进行解析，同时参与position 的重置操作，提供获取指定topic 的集群元数据的操作。上述所有请求都是通过ConsumerNetworkClient 缓存并发送的，在ConsumerNetworkClient  中还维护了定时任务队列，用来完成HeartbeatTask 任务和AutoCommitTask 任务。NetworkClient 在接收到上述请求的响应时会调用相应回调，最终交给其对应的XXHandler 以及RequestFuture 的监听器进行处理。

![](/public/upload/scala/consumer_object.png)

![](/public/upload/scala/consumer_poll.png)

Kafka对外暴露了一个非常简洁的poll方法，其内部实现了协作、分区重平衡、心跳、数据拉取等功能，但使用时这些细节都被隐藏了

Kafka provides two types of API for Java consumers:

1. High-level API,  does not allow consumers to control interactions with brokers.
2. Low-level API, is stateless
and provides fine grained control over the communication between Kafka broker and the consumer.

那么consumer 与broker 交互有哪些细节呢？The high-level consumer API is used when only data is needed and the handling of message offsets is not required. This API hides broker details from the consumer and allows effortless communication with the Kafka cluster by providing an abstraction over the low-level implementation. The high-level consumer stores the last offset
(the position within the message partition where the consumer left off consuming the message), read from a specific partition in Zookeeper. This offset is stored based on the consumer group name provided to Kafka at the beginning of the process.

**主动拉取 是kafka 的一个重要特征，不仅是consumer 主动拉取broker， broker partition follower 也是主动拉取leader**。

##  《learning apache kafka》

1. producers and consumers work on the traditional push-and-pull model, where producers push the message to a Kafka broker and consumers pull the message from the broker.
2. Log compaction,相同key的value 只会保留最新的
3. Message compression in Kafka, For the cases where network bandwidth is a bottleneck, Kafka provides a message group compression feature for efficient message delivery.
4. replication modes。Asynchronous replication： as soon as a lead replica writes the message to its local log, it sends the acknowledgement to the message client and does not wait for acknowledgements from follower replicas。Synchronous replication 则反之


## 小结

1. 内存队列，push和poll 本质是对底层数组操作的封装
2. 消息中间件，push 和 poll 本质是数据的序列化、压缩、io发送 与 io 拉取、发序列化、解压缩，**这就有点rpc 的味道了**，只是rpc 是双方直接通信，而消息中间件是 producer/consumer 都与 kafka server 通信。

