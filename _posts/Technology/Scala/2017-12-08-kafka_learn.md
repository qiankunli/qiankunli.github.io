---

layout: post
title: 《Apache Kafka源码分析》小结
category: 技术
tags: Scala
keywords: Scala  akka

---

## 前言（未完成）

建议先阅读下[消息/任务队列](http://qiankunli.github.io/2015/08/08/message_task_queue.html)，了解下消息队列中间件的宏观理论、概念及取舍


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

每个topic包含多个分区，每个分区包含多个副本。作为producer，一个topic消息放入哪个分区，hash一下即可

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

    // 配置属性
    Properties props = new Properties();
    props.put("zookeeper.connect", a_zookeeper);
    props.put("group.id", a_groupId);
    props.put("zookeeper.session.timeout.ms", "400");
    props.put("zookeeper.sync.time.ms", "200");
    props.put("auto.commit.interval.ms", "1000");
    ConsumerConfig consumerConfig = new ConsumerConfig(props);
    // 构建consumer
    ConsumerConnector consumer = kafka.consumer.Consumer.createJavaConsumerConnector(consumerConfig);
    // consumer 根据目标 topic 返回 对应的stream
    Map<String, Integer> topicCountMap = new HashMap<String, Integer>();
    topicCountMap.put(topic, new Integer(a_numThreads));
    Map<String, List<KafkaStream<byte[], byte[]>>> consumerMap = consumer.createMessageStreams(topicCountMap);
    // 消费特定 topic 的stream
    List<KafkaStream<byte[], byte[]>> streams = consumerMap.get(topic);
    for (final KafkaStream stream : streams) {
        ConsumerIterator<byte[], byte[]> it = stream.iterator();
        while (it.hasNext()){
            System.out.println("Thread " + m_threadNumber + ": " + new String(it.next().message()));
        }
    }
	

## 生产者(未完成)

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

## 为什么要zookeeper，因为关联业务要交换元数据

今天笔者在学习kafka源码，kafka的主体是`producer ==> topic ==> consumer`，topic只是一个逻辑概念，topic包含多个分区，每个分区数据包含多个副本（leader副本，slave副本）。producer在发送数据之前，首先要确定目的分区（可能变化），其次确定目的分区的leader副本所在host，知道了目的地才能发送record，这些信息是集群的meta信息。producer每次向topic发送record，都要`waitOnMetadata(record.topic(), this.maxBlockTimeMs)`以拿到最新的metadata。

producer面对的是一个broker集群，这个meta信息找哪个broker要都不方便，也不可靠，本质上，还是从zookeeper获取比较方便。zookeeper成为producer与broker集群解耦的工具。

关联业务之间需要交换元数据，当然，数据库也可以承担这个角色，但数据库没有副本等机制保证可靠性

