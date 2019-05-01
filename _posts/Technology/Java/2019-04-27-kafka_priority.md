---

layout: post
title: 让kafka支持优先级队列
category: 技术
tags: Java
keywords: kafka

---

## 简介（持续更新）


kafka 官方需求  Kafka Improvement Proposals [KIP-349: Priorities for Source Topics](https://cwiki.apache.org/confluence/display/KAFKA/KIP-349%3A+Priorities+for+Source+Topics)


背景，我们希望kafka 可以支持“优先级”特性：即便队列里已经有了很多消息，但是高优先级消息可以“插队”进而立即被消费。自然地，在kafka 的概念里，我们建立多个topic，一个topic代表一个优先级，那么难点就转换为了 如何对 不同优先级的consumer 进行封装处理了。

![](/public/upload/java/priority_kafka_subscribe.png)

1. 优先级以数字表示，值越高优先级越高
2. 一个topic 对应有一个优先级，对应一个consumer

## 内部使用优先级队列重新缓冲

![](/public/upload/java/priority_kafka_internal_queue.png)

存在的风险

1. 内存的 priority queue 一般都有容量限制
1. 如果consumer 消费速度不够快，则priority queue 大部分时间处于满的状态，进而堵塞
3. priority queue 可以保证 已经插入的消息 按照priority 排队，但不能保证阻塞的几个插入方按优先级插入。（待确认）

## 优先级从高到低依次拉取，优先级越高拉取“配额”越大

下文主要来自对[flipkart-incubator/priority-kafka-client](https://github.com/flipkart-incubator/priority-kafka-client)的源码分析

![](/public/upload/java/priority_kafka_producer_class_diagram.png)

![](/public/upload/java/priority_kafka_consumer_class_diagram.png)

This client provides abstraction to implement Kafka's Producer<K, V> and Consumer<K, V> with priority support.  PriorityKafkaProducer 和 CapacityBurstPriorityKafkaConsumer 只是聚合了KafkaProducer 和 KafkaConsumer成员 ，其本身仍符合KafkaProducer 和 KafkaConsumer 接口

CapacityBurstPriorityKafkaConsumer 的类描述

1. regulates（管理、控制） capacity (record processing rate) across priority levels. 跨优先级管控 记录消费速度
2. When we discuss about priority topic XYZ and consumer group ABC, here XYZ and ABC are the logical names. For every logical priority topic XYZ one must define max supported priority level via the config ClientConfigs#MAX_PRIORITY_CONFIG property.**This property is used as the capacity lever（杠杆） for tuning（调音、调谐） processing rate across priorities**. Every object of the class maintains KafkaConsumer instance for every priority level topic [0, ${max.priority - 1}].

### 循环拉取

CapacityBurstPriorityKafkaConsumer聚合多个优先级的consumer，先不考虑优先级，循环拉取逻辑如下：

    // 同时维护多个consumer
    private Map<Integer, KafkaConsumer<K, V>> consumers;
    public ConsumerRecords<K, V> poll(long pollTimeoutMs) {
        Map<TopicPartition, List<ConsumerRecord<K, V>>> consumerRecords = 
            new HashMap<TopicPartition, List<ConsumerRecord<K, V>>>();
        long start = System.currentTimeMillis();
        do {
            for (int i = maxPriority - 1; i >= 0; --i) {
                ConsumerRecords<K, V> records = consumers.get(i).poll(0);
                for (TopicPartition partition : records.partitions()) {
                    consumerRecords.put(partition, records.records(partition));
                }
            }
        } while (consumerRecords.isEmpty() && System.currentTimeMillis() < (start + pollTimeoutMs));
        ...
    }

从上述代码可以看到， 单纯的循环拉取，只是做到了一次循环范围内先消费优先级高的，再消费优先级低的。消费次序有高低之分，但消费机会是均等的，无法做到当高中低都有消息待消费时，集中全力先消费高优先级的消息。

### 设定一次拉取多少个

    // CapacityBurstPriorityKafkaConsumer.java
    void updateMaxPollRecords(KafkaConsumer<K, V> consumer, int maxPollRecords) {
        try {
            Field fetcherField = org.apache.kafka.clients.consumer.KafkaConsumer.class.getDeclaredField(FETCHER_FIELD);
            fetcherField.setAccessible(true);
            Fetcher fetcher = (Fetcher) fetcherField.get(consumer);
            Field maxPollRecordsField = Fetcher.class.getDeclaredField(MAX_POLL_RECORDS_FIELD);
            maxPollRecordsField.setAccessible(true);
            maxPollRecordsField.set(fetcher, maxPollRecords);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

KafkaConsumer 包括fetcher 成员，通过设置`fetcher.maxPollRecords` 可以让`consumers.poll(0)` 受控的拉取fetcher.maxPollRecords 个消息

1. `ConsumerRecords<K, V> Consumer.poll(long pollTimeoutMs)` 指明线程如果没有数据时等待多长时间，0表示不等待立即返回。
2. `fetcher.maxPollRecords` 控制一个poll()调用返回的记录数，这个可以用来控制应用在拉取循环中的处理数据量。

### 一次拉取多少, 如何在各个优先级之间分配

CapacityBurstPriorityKafkaConsumer 配置

| Config | Mandatory | Description |
|:------------|:----------------|:------------|
| max.priority             | Yes      |    Defines max priority
| group.id             | Yes      |    Defines logical group ID for the consumer
| max.poll.records             | Yes      |    Defines max records to be polled across priority levels
| max.poll.history.window.size             | No      |    This is window length to track historical counts of records obtained from ```poll()```. Defaulted to 6
| min.poll.window.maxout.threshold             | No      |    This is threshold on how many historical counts os records obtained from ```poll()``` maxed out w.r.t. ```max.poll.records``` config. Defaulted to 4
Rest of the configs are similar to that of ```KafkaConsumer```.

CapacityBurstPriorityKafkaConsumer.poll 一次可以拉取的记录数由`max.poll.records` 配置，`max.poll.records`property is split across priority topic consumers based on maxPollRecordsDistributor - defaulted to ExpMaxPollRecordsDistributor. `max.poll.records` 按一定算法 split 分摊给 各个consumer

`Map<Integer, Integer> ExpMaxPollRecordsDistributor.distribution(maxPriority,maxPollRecords)` 假设输入是 `3,50` 则输出为`{2=29, 1=14, 0=7}`

1. 一共50个“名额”，3个优先级，优先级越高分的越多
2. 一种分配方式是指数分配（Exponential distribution of maxPollRecords across all priorities (upto maxPriority)），即高一个优先级的“配额”是低一个优先级“配额”的2倍。当然，你也可以选择 高一个优先级的“配额”比低一个优先级的多1个。 

以`{2=29, 1=14, 0=7}` 配额为例，假设实际拉取数据时

1. 第一次实际拉取情况 `{2=10, 1=10, 0=7}` 即高优先级 消息不多，低优先级 消息比较多。
2. 第二次拉取时，各个优先级的配额还按`{2=29, 1=14, 0=7}` 来么？
3. 如果每次拉取数据 都按`{2=29, 1=14, 0=7}` 的配额进行，考虑一种情况：在一段时间内（比如10分钟），高中低消息数量相同，但高优先级消息先少后多，低优先级消息先多后少。 则10分钟结束后，可能出现低优先级消费数量比高优先级 还多的情况。
4. 为此，priority_kafka 使用了Window机制，加入第一次实际拉取情况 `{2=10, 1=10, 0=7}` ，则第二次制定拉取配额时，为高优先级burst，比如制定配额为 `{2=29+19,1=14+4,0=7+3}` 来补偿高优先级的消费数量。window 部分的逻辑待进一步梳理。
