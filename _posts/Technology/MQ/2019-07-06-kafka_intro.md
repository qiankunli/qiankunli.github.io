---

layout: post
title: 《Apache Kafka源码分析》——简介
category: 技术
tags: MQ
keywords: Scala  akka

---

## 前言

* TOC
{:toc}



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

**Kafka 的消息组织方式实际上是三级结构：主题 - 分区 - 消息**  Partitions are nothing but separate queues in Kafka to make it more scalable. When we increase partitions or we have 1+ number of Partitions it is expected that you run multiple consumers. Ideally number of Consumer should be equal to number of Partitions. 分区相当于把“车道”拓宽了。

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

##  《learning apache kafka》

1. producers and consumers work on the traditional push-and-pull model, where producers push the message to a Kafka broker and consumers pull the message from the broker.
2. Log compaction,相同key的value 只会保留最新的
3. Message compression in Kafka, For the cases where network bandwidth is a bottleneck, Kafka provides a message group compression feature for efficient message delivery.
4. replication modes。Asynchronous replication： as soon as a lead replica writes the message to its local log, it sends the acknowledgement to the message client and does not wait for acknowledgements from follower replicas。Synchronous replication 则反之








